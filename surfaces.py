# surfaces.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch
from torch import Tensor

from rays import RayBundle, _normalize, _build_orthonormal_frame 


# -----------------------------
# Aperture primitives
# -----------------------------

@dataclass
class CircularAperture:
    radius: float

    def contains(self, x: Tensor, y: Tensor) -> Tensor:
        """
        x, y: [..., N_rays]
        Returns boolean mask [..., N_rays] of points inside the circle.
        """
        r2 = x * x + y * y
        return r2 <= (self.radius ** 2 + 1e-9)


@dataclass
class SquareAperture:
    half_width_x: float
    half_width_y: float

    def contains(self, x: Tensor, y: Tensor) -> Tensor:
        """
        x, y: [..., N_rays]
        Returns boolean mask [..., N_rays] of points inside the square.
        """
        return (
            (x.abs() <= self.half_width_x + 1e-9)
            & (y.abs() <= self.half_width_y + 1e-9)
        )


# -----------------------------
# Base surface utilities
# -----------------------------
class _SurfaceBase:
    """
    Common functionality: coordinate transforms, frame setup.
    """

    def __init__(
        self,
        origin: Tensor | list[float] | tuple[float, float, float],
        normal: Tensor | list[float] | tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        origin = torch.as_tensor(origin, dtype=torch.float32)
        normal = torch.as_tensor(normal, dtype=torch.float32)

        if origin.shape != (3,):
            raise ValueError(f"origin must be shape (3,), got {origin.shape}")
        if normal.shape != (3,):
            raise ValueError(f"normal must be shape (3,), got {normal.shape}")

        self.origin = origin          # stored on CPU by default
        self.normal = _normalize(normal)

        # Build orthonormal frame: u, v, w (w aligned with normal)
        u, v, w = _build_orthonormal_frame(self.normal)
        # Basis matrix with rows = [u; v; w]
        self.B = torch.stack([u, v, w], dim=0)  # (3, 3)

    @property
    def device(self):
        return self.origin.device

    @property
    def dtype(self):
        return self.origin.dtype

    # --- NEW helper: move surface tensors to match a reference tensor ---

    def _match_ref(self, ref: Tensor) -> tuple[Tensor, Tensor]:
        """
        Return (origin, B) moved to ref's device/dtype.
        """
        origin = self.origin.to(device=ref.device, dtype=ref.dtype)
        B = self.B.to(device=ref.device, dtype=ref.dtype)
        return origin, B

    # --- UPDATED transforms use _match_ref() ---

    def world_to_local(self, points: Tensor) -> Tensor:
        """
        points: [..., 3]
        Returns local coordinates: [..., 3]
        """
        origin, B = self._match_ref(points)
        p = points - origin  # broadcast-safe now (same device/dtype)
        # local = (world - origin) projected onto [u; v; w] rows
        return torch.matmul(p, B.T)  # [..., 3]

    def local_to_world(self, points_local: Tensor) -> Tensor:
        """
        points_local: [..., 3]
        Returns world coordinates: [..., 3]
        """
        origin, B = self._match_ref(points_local)
        return origin + torch.matmul(points_local, B)

    def local_normal_to_world(self, n_local: Tensor) -> Tensor:
        """
        n_local: [..., 3]
        Returns world normals: [..., 3]
        """
        _, B = self._match_ref(n_local)
        return torch.matmul(n_local, B)


# -----------------------------
# Plane surface with aperture
# -----------------------------

class PlaneSurface(_SurfaceBase):
    """
    Plane z_local = 0 with a given aperture in the (x_local, y_local) plane.

    Default orientation: normal pointing along +z in world coordinates.

    Local coordinates:
        - z_local = 0 is the plane.
        - aperture is defined in (x_local, y_local).
    """

    def __init__(
        self,
        origin: Tensor | list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
        normal: Tensor | list[float] | tuple[float, float, float] = (0.0, 0.0, 1.0),
        aperture: CircularAperture | SquareAperture | None = None,
    ) -> None:
        super().__init__(origin=origin, normal=normal)
        if aperture is None:
            raise ValueError("PlaneSurface requires an aperture (circular or square).")
        self.aperture = aperture

    def intersect_rays(self, rays: RayBundle) -> Dict[str, Tensor]:
        """
        Compute intersection of rays with the plane.

        Returns dict with:
            - "points":  [..., N_rays, 3] intersection points in world coords,
                         NaN where no valid intersection/aperture miss.
            - "normals": [..., N_rays, 3] surface normals in world coords,
                         NaN where invalid.
            - "mask":    [..., N_rays] boolean mask of valid intersections.
        """
        o_world = rays.origins   # [..., N, 3]
        d_world = rays.directions


        o_local = self.world_to_local(o_world)  # [..., N, 3]
        d_local = self.world_to_local(o_world + d_world) - o_local

        o_z = o_local[..., 2]
        d_z = d_local[..., 2]

        eps = 1e-9
        # Avoid division by zero (rays parallel to plane)
        parallel = d_z.abs() < eps
        t = -o_z / (d_z + eps)  # eps avoids NaN; we'll gate by mask

        # We want intersections in front of ray origin (t > 0) and non-parallel
        mask_t = (t > 0.0) & (~parallel)

        # Compute intersection in local coords
        t_expanded = t[..., None]  # [..., N, 1]
        p_local = o_local + t_expanded * d_local  # [..., N, 3]

        x = p_local[..., 0]
        y = p_local[..., 1]

        # Aperture mask
        mask_ap = self.aperture.contains(x, y)

        mask = mask_t & mask_ap  # [..., N]

        # Local normal is (0, 0, 1) everywhere on the plane
        n_local = torch.zeros_like(p_local)
        n_local[..., 2] = 1.0

        # World coords & normals
        p_world = self.local_to_world(p_local)
        n_world = self.local_normal_to_world(n_local)

        # Set invalid intersections / normals to NaN
        nan_vec = torch.full_like(p_world, float("nan"))
        nan_norm = torch.full_like(n_world, float("nan"))
        p_world = torch.where(mask[..., None], p_world, nan_vec)
        n_world = torch.where(mask[..., None], n_world, nan_norm)
        return {
            "points": p_world,
            "normals": n_world,
            "mask": mask,
        }


# -----------------------------
# Spherical surface with circular aperture
# -----------------------------

class SphericalSurface(_SurfaceBase):
    """
    Spherical surface with a circular aperture.

    The surface is the visible cap of a sphere of radius R whose apex is at
    `origin` and whose outward normal at the apex is `normal`.

    Local coordinates:
        - origin is at local (0, 0, 0) (the apex)
        - sphere center is at local (0, 0, -radius)
        - surface equation in local coords: (x^2 + y^2 + (z + R)^2) = R^2
        - we only keep intersection points with z_local >= 0 (the cap)

    Aperture:
        - circular in the (x_local, y_local) plane with radius `aperture.radius`.
    """

    def __init__(
        self,
        radius: float,
        origin: Tensor | list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
        normal: Tensor | list[float] | tuple[float, float, float] = (0.0, 0.0, 1.0),
        aperture: CircularAperture | None = None,
    ) -> None:
        super().__init__(origin=origin, normal=normal)
        if radius <= 0:
            raise ValueError("SphericalSurface radius must be positive.")
        self.radius = float(radius)
        if aperture is None:
            raise ValueError("SphericalSurface requires a CircularAperture.")
        self.aperture = aperture

        # Sphere center in local coordinates (0, 0, -R)
        self.center_local = torch.tensor(
            [0.0, 0.0, -self.radius],
            dtype=self.dtype,
            device=self.device,
        )

    def intersect_rays(self, rays: RayBundle) -> Dict[str, Tensor]:
        """
        Compute intersection of rays with the spherical cap.

        Returns dict with:
            - "points":  [..., N_rays, 3] intersection points in world coords,
                         NaN where no valid intersection/aperture miss.
            - "normals": [..., N_rays, 3] surface normals in world coords,
                         NaN where invalid.
            - "mask":    [..., N_rays] boolean mask of valid intersections.
        """
        o_world = rays.origins   # [..., N, 3]
        d_world = rays.directions

        # Transform to local coordinates
        o_local = self.world_to_local(o_world)  # [..., N, 3]
        d_local = self.world_to_local(o_world + d_world) - o_local

        # Sphere intersection in local frame
        # Sphere: |p - c|^2 = R^2, where c = (0, 0, -R)
        c = self.center_local  # (3,)
        oc = o_local - c       # [..., N, 3]

        a = (d_local * d_local).sum(dim=-1)  # [..., N]
        b = 2.0 * (d_local * oc).sum(dim=-1)
        c_quad = (oc * oc).sum(dim=-1) - (self.radius ** 2)

        disc = b * b - 4.0 * a * c_quad  # [..., N]
        no_real = disc <= 0.0

        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))

        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        # Candidates [t1, t2]
        t_candidates = torch.stack([t1, t2], dim=-1)  # [..., N, 2]

        eps = 1e-9
        # Compute candidate intersection points
        o_local_exp = o_local.unsqueeze(-2)      # [..., N, 1, 3]
        d_local_exp = d_local.unsqueeze(-2)      # [..., N, 1, 3]
        t_exp = t_candidates[..., None]          # [..., N, 2, 1]
        p_candidates = o_local_exp + t_exp * d_local_exp  # [..., N, 2, 3]
        z_candidates = p_candidates[..., 2]               # [..., N, 2]

        # Valid candidate: t > 0, z >= 0 (front cap), discriminant positive
        valid_candidates = (t_candidates > eps) & (z_candidates >= 0.0)
        valid_candidates = valid_candidates & (~no_real)[..., None]

        # Choose the smallest positive t with z>=0
        inf = torch.tensor(float("inf"), dtype=o_local.dtype, device=o_local.device)
        t_masked = torch.where(valid_candidates, t_candidates, inf)
        t_selected, _ = t_masked.min(dim=-1)  # [..., N]

        has_intersection = t_selected.isfinite()

        t_sel_exp = t_selected[..., None]
        p_local = o_local + t_sel_exp * d_local  # [..., N, 3]

        # Apply aperture in (x_local, y_local)
        x = p_local[..., 0]
        y = p_local[..., 1]
        mask_ap = self.aperture.contains(x, y)

        # We also require z_local >= 0 explicitly
        z = p_local[..., 2]
        mask_cap = z >= 0.0

        mask = has_intersection & mask_ap & mask_cap  # [..., N]

        # Local normal is (p_local - center) normalized
        n_local = _normalize(p_local - c)  # [..., N, 3]

        # Convert to world
        p_world = self.local_to_world(p_local)
        n_world = self.local_normal_to_world(n_local)

        # Mask invalid as NaN
        nan_vec = torch.full_like(p_world, float("nan"))
        nan_norm = torch.full_like(n_world, float("nan"))
        p_world = torch.where(mask[..., None], p_world, nan_vec)
        n_world = torch.where(mask[..., None], n_world, nan_norm)

        return {
            "points": p_world,
            "normals": n_world,
            "mask": mask,
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # 1) Build an input RayBundle (rays_in)
    # --------------------------------------------------
    # We create a uniform grid of rays starting at z = -0.1 m,
    # pointing towards +z, aimed at a circular plane at z = 0.

    Nx, Ny = 64, 64
    x = torch.linspace(-0.5e-3, 0.5e-3, Nx, device=device)  # +/- 0.5 mm
    y = torch.linspace(-0.5e-3, 0.5e-3, Ny, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")             # [Nx, Ny]

    zz = torch.full_like(xx, -0.1)                           # z = -0.1 m

    # Stack to get origins: [Nx, Ny, 3] -> [1, N, 3]
    origins = torch.stack([xx, yy, zz], dim=-1).reshape(1, Nx * Ny, 3)

    # Directions: all pointing straight along +z (collimated beam)
    directions = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 3)
    directions = directions.expand_as(origins)               # [1, N, 3]

    # Wavelength: monochromatic 550 nm, stored in meters
    wavelengths = torch.full(
        (1, Nx * Ny),
        550e-9,
        device=device,
        dtype=torch.float32,
    )

    # E-field in ray basis:
    # For this simple example, assume each ray has unit amplitude,
    # linearly polarized along the x-direction in the ray basis.
    # fields: [B=1, N, 3], interpreted as [E1, E2, E3] in ray-aligned basis.
    fields = torch.zeros(1, Nx * Ny, 3, device=device)
    fields[..., 0] = 1.0  # E along “first” transverse direction

    rays_in = RayBundle(
        origins=origins,
        directions=directions,
        fields=fields,
        wavelengths=wavelengths,
    )

    print("rays_in.origins shape:", rays_in.origins.shape)
    print("rays_in.directions shape:", rays_in.directions.shape)
    print("rays_in.wavelengths shape:", rays_in.wavelengths.shape)

    # --------------------------------------------------
    # 2) Define a plane surface with circular aperture
    # --------------------------------------------------
    aperture = CircularAperture(radius=0.5e-3)               # 0.5 mm radius
    plane = PlaneSurface(
        origin=(0.0, 0.0, 0.0),                             # plane at z = 0
        normal=(0.0, 0.0, 1.0),                             # facing +z
        aperture=aperture,
    )

    # --------------------------------------------------
    # 3) Intersect rays with the plane
    # --------------------------------------------------
    inter = rays_in.find_intersection(plane, drop_invalid=False)
    pts = inter["points"]     # [1, N, 3] with NaNs where no intersection
    norms = inter["normals"]  # [1, N, 3] with NaNs
    mask = inter["mask"]      # [1, N]

    print("Intersection points shape:", pts.shape)
    print("Intersection mask valid fraction:", mask.float().mean().item())

    # --------------------------------------------------
    # 4) Refract using Snell’s law (no phase gradient)
    # --------------------------------------------------
    rays_out_snell = rays_in.diffract(
        surface=plane,
        n1=1.0,      # medium 1 (air)
        n2=1.5,      # medium 2 (glass)
        phase_gradient=None,
        drop_invalid=True,
    )

    print("rays_out_snell.origins shape:", rays_out_snell.origins.shape)
    print("rays_out_snell.directions shape:", rays_out_snell.directions.shape)

    # --------------------------------------------------
    # 5) Refract with a constant grating vector (phase gradient)
    # --------------------------------------------------
    # Example: simple blazed grating along +x with period Λ
    period = 10e-6  # 10 microns
    phase_grad = torch.tensor(
        [2.0 * torch.pi / period, 0.0, 0.0],
        device=device,
        dtype=torch.float32,
    )

    rays_out_grating = rays_in.diffract(
        surface=plane,
        n1=1.0,
        n2=1.5,
        phase_gradient=phase_grad,
        drop_invalid=True,
    )

    print("rays_out_grating.origins shape:", rays_out_grating.origins.shape)
    print("rays_out_grating.directions shape:", rays_out_grating.directions.shape)