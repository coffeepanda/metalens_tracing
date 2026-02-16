from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from networks import MetaLensModel
import torch
from torch import Tensor


def _ensure_tensor(x: Any, *, device=None, dtype=None) -> Tensor | None:
    if x is None:
        return None
    if isinstance(x, Tensor):
        return x.to(device=device or x.device, dtype=dtype or x.dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)

def _normalize(v: Tensor, eps: float = 1e-9) -> Tensor:
    """
    Normalize a vector along the last dimension.
    """
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def _build_orthonormal_frame(axis: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Given a 3D axis vector field `axis` with shape [..., 3],
    build an orthonormal frame (u, v, w) where:

        w = normalized axis
        u, v are orthonormal and span the plane perpendicular to w

    The construction is robust to axis being (nearly) parallel to the initial
    "up" vector by using a fallback.
    """
    if axis.shape[-1] != 3:
        raise ValueError(f"_build_orthonormal_frame expects last dim=3, got {axis.shape[-1]}")

    w = _normalize(axis)  # [..., 3]

    # Prepare an "up" vector broadcastable to w
    up = torch.tensor([0.0, 0.0, 1.0], device=axis.device, dtype=axis.dtype)
    up = up.view(*([1] * (w.dim() - 1)), 3).expand_as(w)

    u = torch.cross(up, w, dim=-1)  # [..., 3]
    u_norm = u.norm(dim=-1, keepdim=True)

    # If axis is parallel to up, pick a different up
    near_parallel = u_norm < 1e-6
    if near_parallel.any():
        up2 = torch.tensor([0.0, 1.0, 0.0], device=axis.device, dtype=axis.dtype)
        up2 = up2.view(*([1] * (w.dim() - 1)), 3).expand_as(w)
        u2 = torch.cross(up2, w, dim=-1)
        u = torch.where(near_parallel, u2, u)

    u = _normalize(u)
    v = torch.cross(w, u, dim=-1)
    v = _normalize(v)

    return u, v, w  # [..., 3] each

@dataclass
class RayBundle:
    """
    Container for a bundle of rays.

    Shape conventions (similar to your meta-lens script):

        origins:    [..., N_rays, 3]
        directions: [..., N_rays, 3]
        fields:     [..., N_rays, C]   (complex fields can be packed as [Re, Im] or [Ex, Ey, Ez, ...])
        wavelengths:
            - None:          no wavelength attached
            - scalar:        single wavelength for all rays (monochromatic bundle)
            - [..., N_rays]: per-ray wavelengths

    Leading dimensions `...` are arbitrary (batch, time, lens index, etc.).
    We intentionally do **not** introduce a separate wavelength axis;
    instead we treat wavelength as a per-ray attribute. This keeps memory
    & shape handling manageable when you have many rays or many wavelengths.
    """

    origins: Tensor      # o
    directions: Tensor   # d
    fields: Tensor | None = None   # E
    wavelengths: Tensor | None = None

    def __post_init__(self) -> None:
        # Normalize to tensors and ensure consistent device/dtype
        device = self.origins.device
        dtype = self.origins.dtype

        self.origins = _ensure_tensor(self.origins, device=device, dtype=dtype)
        self.directions = _ensure_tensor(self.directions, device=device, dtype=dtype)
        if self.fields is not None:
            self.fields = _ensure_tensor(self.fields, device=device)
        if self.wavelengths is not None:
            self.wavelengths = _ensure_tensor(self.wavelengths, device=device)

        self._validate_shapes()

    # -------------------------
    # Shape utilities
    # -------------------------

    @property
    def base_shape(self) -> torch.Size:
        """
        Returns the common shape excluding the last dimension (vector dim).
        This is basically [..., N_rays] for origins/directions.
        """
        return self.origins.shape[:-1]

    @property
    def num_rays(self) -> int:
        return self.origins.shape[-2]

    @property
    def dim(self) -> int:
        """
        Spatial dimension: usually 3.
        """
        return self.origins.shape[-1]

    def _validate_shapes(self) -> None:
        if self.origins.shape != self.directions.shape:
            raise ValueError(
                f"origins.shape {self.origins.shape} and directions.shape {self.directions.shape} must match."
            )
        if self.dim not in (2, 3):
            # 2D or 3D rays both allowed; you can relax this if you want.
            raise ValueError(f"Expected origins/directions last dim to be 2 or 3, got {self.dim}.")

        if self.fields is not None:
            if self.fields.shape[:-1] != self.origins.shape[:-1]:
                raise ValueError(
                    f"fields.shape {self.fields.shape} must share leading dims with origins {self.origins.shape} "
                    "(excluding last channel dim)."
                )

        if self.wavelengths is not None:
            # Allowed patterns:
            #   1) scalar (monochromatic bundle)
            #   2) [..., N_rays] (per-ray wavelength)
            #   3) [..., 1] (broadcastable over N_rays)
            wl = self.wavelengths
            if wl.ndim == 0:
                # scalar λ – always ok
                return
            # treat last dimension as ray dimension or broadcast dim
            target_shape = self.origins.shape[:-1]  # [..., N_rays]
            if wl.shape == target_shape:
                # per-ray λ
                return
            if wl.shape == target_shape[:-1] + (1,):
                # broadcastable λ
                return
            raise ValueError(
                f"Unsupported wavelengths.shape {wl.shape} for origins.shape {self.origins.shape}. "
                "Use scalar, [..., N_rays], or [..., 1]."
            )

    # -------------------------
    # Wavelength helpers
    # -------------------------

    @property
    def is_monochromatic(self) -> bool:
        """
        Returns True if this bundle has a single wavelength for all rays
        (either scalar or broadcastable), False otherwise.
        """
        if self.wavelengths is None:
            return False
        if self.wavelengths.ndim == 0:
            return True
        # [..., 1] is also monochromatic
        target_shape = self.origins.shape[:-1]
        if self.wavelengths.shape == target_shape[:-1] + (1,):
            return True
        return False

    def wavelengths_per_ray(self) -> Tensor | None:
        """
        Returns wavelengths in shape [..., N_rays], broadcasting as needed.
        If wavelengths is None, returns None.
        """
        if self.wavelengths is None:
            return None

        wl = self.wavelengths
        target_shape = self.origins.shape[:-1]  # [..., N_rays]

        if wl.ndim == 0:
            # scalar -> broadcast
            return wl.expand(target_shape)

        if wl.shape == target_shape:
            return wl

        if wl.shape == target_shape[:-1] + (1,):
            return wl.expand(target_shape)

        # Should be unreachable if validation passed
        raise RuntimeError(f"Incompatible wavelengths shape {wl.shape} for target {target_shape}.")

    def wavevector(
        self,
        n: float | Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the wavevector k and its norm |k| for each ray.

        k direction is along the ray direction; magnitude is:

            |k| = 2π * n / λ

        where λ is the (vacuum) wavelength stored in this RayBundle.

        Args:
            n: refractive index (scalar or Tensor broadcastable to wavelengths_per_ray()).

        Returns:
            k_vec: [..., N_rays, dim]    – wavevector for each ray
            k_mag: [..., N_rays]         – magnitude |k| for each ray
        """
        if self.wavelengths is None:
            raise ValueError(
                "wavevector() requires wavelengths to be set on the RayBundle."
            )

        lam = self.wavelengths_per_ray()  # [..., N_rays]
        if lam is None:
            raise RuntimeError("wavelengths_per_ray returned None unexpectedly.")

        # Make n broadcastable to lam
        if not isinstance(n, Tensor):
            n = torch.tensor(n, dtype=lam.dtype, device=lam.device)
        else:
            n = n.to(device=lam.device, dtype=lam.dtype)

        # Broadcast n and lam together
        n_b, lam_b = torch.broadcast_tensors(n, lam)

        # |k| = 2π n / λ
        k_mag = (2.0 * math.pi) * n_b / lam_b  # [..., N_rays]

        # Unit direction from ray directions
        d_hat = _normalize(self.directions)  # [..., N_rays, dim]

        # k_vec = |k| * d_hat
        k_vec = k_mag[..., None] * d_hat  # [..., N_rays, dim]

        return k_vec, k_mag



    # -------------------------
    # Convenience constructors
    # -------------------------

    @classmethod
    def monochromatic(
        cls,
        origins: Tensor,
        directions: Tensor,
        wavelength: float | Tensor,
        fields: Tensor | None = None,
    ) -> "RayBundle":
        """
        Construct a monochromatic bundle where all rays share a single wavelength.
        """
        wl_tensor = _ensure_tensor(wavelength, device=origins.device, dtype=origins.dtype)
        return cls(origins=origins, directions=directions, fields=fields, wavelengths=wl_tensor)

    @classmethod
    def with_per_ray_wavelengths(
        cls,
        origins: Tensor,
        directions: Tensor,
        wavelengths: Tensor,
        fields: Tensor | None = None,
    ) -> "RayBundle":
        """
        Construct a bundle where each ray has its own wavelength.

        wavelengths: shape [..., N_rays] or [..., 1] (broadcastable).
        """
        return cls(origins=origins, directions=directions, fields=fields, wavelengths=wavelengths)


    def find_intersection(
        self,
        surface: Any,
        *,
        drop_invalid: bool = False,
    ) -> dict[str, Any]:
        """
        Intersect this RayBundle with a surface.

        Args:
            surface:
                A surface object with method:
                    intersect_rays(rays: RayBundle) -> {
                        "points":  [..., N_rays, 3],
                        "normals": [..., N_rays, 3],
                        "mask":    [..., N_rays] (bool)
                    }
            drop_invalid:
                If False (default), all rays are kept and intersection points /
                normals for non-intersecting or out-of-aperture rays are NaN.
                If True, invalid rays are dropped and a new, compact RayBundle
                is returned with origins at the intersection points and the
                same directions/fields/wavelengths for the surviving rays.

                Note: when dropping, the original leading batch structure
                is flattened into a single dimension.

        Returns:
            A dict with:
                - "points":    [..., N_rays, 3] or [M, 3]       (intersection points)
                - "normals":   [..., N_rays, 3] or [M, 3]       (surface normals)
                - "mask":      [..., N_rays]                    (valid mask in original layout)
                - "bundle":    RayBundle or None
        """
        inter = surface.intersect_rays(self)
        points = inter["points"]
        normals = inter["normals"]
        mask = inter["mask"]

        if not drop_invalid:
            return {
                "points": points,
                "normals": normals,
                "mask": mask,
                "bundle": None,
            }

        # Flatten everything and keep only valid rays
        mask_flat = mask.reshape(-1)  # [K]
        if mask_flat.sum() == 0:
            # No surviving rays -> return an empty bundle
            empty_orig = self.origins.reshape(-1, 3)[:0]
            bundle = RayBundle(
                origins=empty_orig,
                directions=empty_orig,
                fields=None if self.fields is None else self.fields.reshape(-1, self.fields.shape[-1])[:0],
                wavelengths=None if self.wavelengths is None else self.wavelengths_per_ray().reshape(-1)[:0],
            )
            return {
                "points": empty_orig,
                "normals": empty_orig,
                "mask": mask,
                "bundle": bundle,
            }

        points_flat = points.reshape(-1, points.shape[-1])[mask_flat]     # [M, 3]
        normals_flat = normals.reshape(-1, normals.shape[-1])[mask_flat]  # [M, 3]

        origins_flat = points_flat
        directions_flat = self.directions.reshape(-1, self.directions.shape[-1])[mask_flat]

        if self.fields is not None:
            fields_flat = self.fields.reshape(-1, self.fields.shape[-1])[mask_flat]
        else:
            fields_flat = None

        if self.wavelengths is not None:
            wl_full = self.wavelengths_per_ray().reshape(-1)[mask_flat]  # [M]
        else:
            wl_full = None

        new_bundle = RayBundle(
            origins=origins_flat,
            directions=directions_flat,
            fields=fields_flat,
            wavelengths=wl_full,
        )

        return {
            "points": points_flat,
            "normals": normals_flat,
            "mask": mask,
            "bundle": new_bundle,
        }

    def diffract(
        self,
        surface: Any,
        n1: float,
        n2: float,
        *,
        phase_gradient: Tensor | None = None,
        drop_invalid: bool = True,
    ) -> "RayBundle":
        """
        Apply refraction / diffraction at an interface defined by `surface`.

        - If `phase_gradient` is None:
            Reduces to standard Snell refraction using a vector form.

        - If `phase_gradient` is provided:
            Implements generalized Snell's law by shifting the tangential
            component of the wavevector:

                k2_t = k1_t + ∇φ

            where ∇φ is the phase gradient (in rad/m), supplied as a world-space
            vector (or broadcastable tensor). Only its tangential component is
            used; any normal component is automatically removed.

        Assumptions:
            - This RayBundle is in medium with refractive index n1.
            - Transmitted rays go into medium with refractive index n2.
            - `surface.intersect_rays(self)` returns intersection points &
            normals, as in PlaneSurface / SphericalSurface.
            - Surface normal points from medium 1 towards medium 2.
            - Wavelengths are set on the RayBundle.

        Currently, the E-field is NOT modified (no Fresnel amplitudes or
        polarization changes); only the ray directions are updated.
        For full field tracing, Fresnel equations should be used to update
        amplitude & polarization in the surface-aligned basis.

        Args:
            surface: a PlaneSurface, SphericalSurface, or custom object with
                    `intersect_rays(rays: RayBundle)` implemented.
            n1: refractive index of the incident medium.
            n2: refractive index of the transmitted medium.
            phase_gradient:
                Optional world-space phase gradient ∇φ, in rad/m, as:
                    - shape [3]
                    - or broadcastable to intersection points shape [..., N_rays, 3]
                Only the tangential component is used (normal component removed).
                If None, standard Snell refraction is used.
            drop_invalid:
                - If True: rays for which no transmitted propagating solution
                exists (e.g. total internal reflection, no intersection)
                are dropped and a compact RayBundle is returned.
                - If False: the original layout is preserved and invalid rays
                get NaN origins/directions.

        Returns:
            A new RayBundle with origins at the intersection points and
            directions updated according to refraction/diffraction.
        """
        if self.wavelengths is None:
            raise ValueError(
                "diffract() requires wavelengths to be set on the RayBundle."
            )

        # -------------------------
        # Step 1: intersect with the surface
        # -------------------------
        inter = surface.intersect_rays(self)
        points = inter["points"]   # [..., N, 3] (NaN where invalid)
        normals = inter["normals"] # [..., N, 3] (NaN where invalid)
        mask = inter["mask"]       # [..., N]

        # Build full wavelength tensor and directions in same layout
        lam_full = self.wavelengths_per_ray()        # [..., N]
        d_full = _normalize(self.directions)         # [..., N, 3]

        # Remove entries where there was no intersection
        valid = mask & (~torch.isnan(points[..., 0]))
        valid_flat = valid.reshape(-1)

        # ---------------------------------------------------
        # NEW: fallback if no intersection but rays are already
        #      sitting on the surface (e.g. after find_intersection)
        # ---------------------------------------------------
        if valid_flat.sum() == 0:
            # Try to detect "on-surface" rays for planar surfaces:
            # if origins are within a small distance of the plane, we
            # treat them as the intersection points.
            if hasattr(surface, "origin") and hasattr(surface, "normal"):
                o_world = self.origins                         # [..., N, 3]
                surf_origin = surface.origin.to(
                    device=o_world.device, dtype=o_world.dtype
                )
                surf_normal = surface.normal.to(
                    device=o_world.device, dtype=o_world.dtype
                )
                surf_normal = _normalize(surf_normal)          # [3]

                # Signed distance to the plane along the normal
                dist = ((o_world - surf_origin) * surf_normal).sum(dim=-1)  # [..., N]
                eps = 1e-6
                near = dist.abs() < eps                        # [..., N]

                if not near.any():
                    # Truly nothing to refract: behave as before (empty bundle)
                    empty = self.origins.reshape(-1, 3)[:0]
                    return RayBundle(
                        origins=empty,
                        directions=empty,
                        fields=None if self.fields is None else self.fields.reshape(-1, self.fields.shape[-1])[:0],
                        wavelengths=lam_full.reshape(-1)[:0],
                    )

                # Treat current origins as intersection points on the surface
                points = o_world
                # Constant normal broadcasted to all rays
                normals = surf_normal.view(*([1] * (points.ndim - 1)), 3).expand_as(points)
                mask = near

                valid = mask
                valid_flat = valid.reshape(-1)
            else:
                # Non-planar surface without "on-surface" handling:
                empty = self.origins.reshape(-1, 3)[:0]
                return RayBundle(
                    origins=empty,
                    directions=empty,
                    fields=None if self.fields is None else self.fields.reshape(-1, self.fields.shape[-1])[:0],
                    wavelengths=lam_full.reshape(-1)[:0],
                )

        # -------------------------
        # Continue as before, now with guaranteed valid_flat.sum() > 0
        # -------------------------
        pts_flat = points.reshape(-1, 3)[valid_flat]      # [M, 3]
        n_flat = normals.reshape(-1, 3)[valid_flat]       # [M, 3]
        d_in_flat = d_full.reshape(-1, 3)[valid_flat]     # [M, 3]
        lam_flat = lam_full.reshape(-1)[valid_flat]       # [M]

        if self.fields is not None:
            E_flat = self.fields.reshape(-1, self.fields.shape[-1])[valid_flat]
        else:
            E_flat = None

        n_hat = _normalize(n_flat)                        # [M, 3]
        i_hat = _normalize(d_in_flat)                     # [M, 3]

        # -------------------------
        # Step 2: wavevector magnitudes
        # -------------------------
        two_pi = 2.0 * torch.pi
        lam_flat = lam_flat.to(dtype=pts_flat.dtype, device=pts_flat.device)

        k1_mag = two_pi * float(n1) / lam_flat           # [M]
        k2_mag = two_pi * float(n2) / lam_flat           # [M]

        # Incident wavevector
        k1_vec = k1_mag[..., None] * i_hat               # [M, 3]

        # Decompose into normal + tangential components at the surface
        k1_n_scalar = (k1_vec * n_hat).sum(dim=-1, keepdim=True)  # [M, 1]
        k1_n = k1_n_scalar * n_hat                                  # [M, 3]
        k1_t = k1_vec - k1_n                                        # [M, 3]

        # -------------------------
        # Step 3: add phase gradient in tangential plane (if any)
        # -------------------------
        if phase_gradient is not None:
            g = phase_gradient.to(device=pts_flat.device, dtype=pts_flat.dtype)
            # Broadcast to [M, 3]
            g = torch.broadcast_to(g, k1_t.shape)
            # Remove any normal component: keep only tangential part
            g_n_scalar = (g * n_hat).sum(dim=-1, keepdim=True)
            g_t = g - g_n_scalar * n_hat
        else:
            g_t = torch.zeros_like(k1_t)

        # Generalized tangential component in medium 2
        k2_t = k1_t + g_t  # [M, 3]

        # -------------------------
        # Step 4: enforce dispersion in medium 2
        # -------------------------
        k2_t_sq = (k2_t * k2_t).sum(dim=-1)    # [M]
        k2_mag_sq = k2_mag * k2_mag            # [M]

        # For propagating transmitted rays, we need k2_mag_sq >= k2_t_sq.
        delta = k2_mag_sq - k2_t_sq            # [M]
        propagating = delta >= 0.0

        k2_n_mag = torch.sqrt(torch.clamp(delta, min=0.0))  # [M]
        # Choose sign so that transmitted ray goes into medium 2 (along +n_hat).
        k2_n = k2_n_mag[..., None] * n_hat                  # [M, 3]

        k2_vec = k2_t + k2_n                                 # [M, 3]

        # Final propagation direction in medium 2
        # (If phase_gradient is None, this reduces to Snell's refraction.)
        k2_mag_safe = k2_mag.clamp_min(1e-9)[..., None]
        d_out = k2_vec / k2_mag_safe                         # [M, 3]
        d_out = _normalize(d_out)

        # -------------------------
        # Step 5: handle non-propagating rays and return new bundle
        # -------------------------
        if drop_invalid:
            keep = propagating
        else:
            keep = torch.ones_like(propagating, dtype=torch.bool)

        if drop_invalid:
            pts_flat = pts_flat[keep]
            d_out = d_out[keep]
            lam_flat = lam_flat[keep]
            if E_flat is not None:
                E_flat = E_flat[keep]

            return RayBundle(
                origins=pts_flat,
                directions=d_out,
                fields=E_flat,
                wavelengths=lam_flat,
            )
        else:
            # Keep layout [M, ...] but set non-propagating rays to NaN
            nan_vec = torch.full_like(pts_flat, float("nan"))
            nan_dir = torch.full_like(d_out, float("nan"))
            pts_flat = torch.where(propagating[..., None], pts_flat, nan_vec)
            d_out = torch.where(propagating[..., None], d_out, nan_dir)

            return RayBundle(
                origins=pts_flat,
                directions=d_out,
                fields=E_flat,
                wavelengths=lam_flat,
            )


    # -------------------------
    # Device / dtype helpers
    # -------------------------

    def to(self, *args, **kwargs) -> "RayBundle":
        """
        Returns a new RayBundle moved to the given device/dtype.
        Mirrors Tensor.to() semantics.
        """
        origins = self.origins.to(*args, **kwargs)
        directions = self.directions.to(*args, **kwargs)
        fields = self.fields.to(*args, **kwargs) if self.fields is not None else None
        wavelengths = self.wavelengths.to(*args, **kwargs) if self.wavelengths is not None else None
        return RayBundle(
            origins=origins,
            directions=directions,
            fields=fields,
            wavelengths=wavelengths,
        )
    
    # -------------------------
    # Field transformations between ray and surface frames -- not used for now
    # -------------------------

    def _ray_frame(self) -> tuple[Tensor, Tensor, Tensor]:
        """
        Build an orthonormal frame aligned with each ray direction.

        Returns:
            e1: [..., N_rays, 3] – first transverse basis vector
            e2: [..., N_rays, 3] – second transverse basis vector
            ez: [..., N_rays, 3] – propagation direction (normalized)
        """
        return _build_orthonormal_frame(self.directions)

    def _surface_frame(self, normal: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Build an orthonormal frame aligned with the given surface normal.

        The input `normal` is broadcastable to `origins.shape` (i.e., [..., N_rays, 3]).

        Basis order:
            t1, t2: tangent directions on the surface
            n_hat:  surface normal (normalized)

        Args:
            normal: Tensor with last dim=3, broadcastable to self.origins.shape.

        Returns:
            t1:   [..., N_rays, 3]
            t2:   [..., N_rays, 3]
            n_hat:[..., N_rays, 3]
        """
        normal = normal.to(device=self.origins.device, dtype=self.origins.dtype)

        # Broadcast to match ray count & batch shape
        normal = torch.broadcast_to(normal, self.origins.shape)

        t1, t2, n_hat = _build_orthonormal_frame(normal)
        return t1, t2, n_hat

    def field_ray_to_surface(
        self,
        E_ray: Tensor,
        surface_normal: Tensor,
    ) -> Tensor:
        """
        Transform an E-field represented in the ray-aligned basis into
        a basis defined by a surface normal.

        Ray basis:
            (e1_ray, e2_ray, ez_ray) where ez_ray is along the propagation direction.

        Surface basis:
            (t1_surf, t2_surf, n_hat) where n_hat is the surface normal.

        E_ray and E_surface are 3-component vectors in their respective bases.

        Args:
            E_ray:
                E-field components in the ray basis, shape [..., N_rays, 3]
                where the leading dims match this RayBundle's base_shape.
            surface_normal:
                Surface normal(s), shape broadcastable to [..., N_rays, 3].
                Examples:
                    - [3]                  – one normal for all rays
                    - [..., 3]             – one normal per bundle or batch
                    - [..., N_rays, 3]     – per-ray normals

        Returns:
            E_surf: E-field components in surface basis, shape [..., N_rays, 3]
        """
        if E_ray.shape[:-1] != self.origins.shape[:-1] or E_ray.shape[-1] != 3:
            raise ValueError(
                f"E_ray must have shape {self.origins.shape[:-1] + (3,)}, "
                f"got {E_ray.shape}"
            )

        E_ray = E_ray.to(device=self.origins.device, dtype=self.origins.dtype)

        # Build frames
        e1_ray, e2_ray, ez_ray = self._ray_frame()             # [..., N_rays, 3]
        t1_surf, t2_surf, n_hat = self._surface_frame(surface_normal)

        # Convert from ray-basis components -> world vector
        # E_world = E1 * e1_ray + E2 * e2_ray + E3 * ez_ray
        E1 = E_ray[..., 0:1]   # [..., N_rays, 1]
        E2 = E_ray[..., 1:2]
        E3 = E_ray[..., 2:3]

        E_world = E1 * e1_ray + E2 * e2_ray + E3 * ez_ray   # [..., N_rays, 3]

        # Project world vector onto surface basis
        E_t1 = (E_world * t1_surf).sum(dim=-1, keepdim=True)
        E_t2 = (E_world * t2_surf).sum(dim=-1, keepdim=True)
        E_n  = (E_world * n_hat).sum(dim=-1, keepdim=True)

        E_surf = torch.cat([E_t1, E_t2, E_n], dim=-1)       # [..., N_rays, 3]
        return E_surf

    def field_surface_to_ray(
        self,
        E_surf: Tensor,
        surface_normal: Tensor,
    ) -> Tensor:
        """
        Transform an E-field represented in the surface-aligned basis back
        to the ray-aligned basis.

        Surface basis:
            (t1_surf, t2_surf, n_hat) where n_hat is the surface normal.

        Ray basis:
            (e1_ray, e2_ray, ez_ray) where ez_ray is along the propagation direction.

        Args:
            E_surf:
                E-field components in the surface basis, shape [..., N_rays, 3].
            surface_normal:
                Surface normal(s), broadcastable to [..., N_rays, 3].

        Returns:
            E_ray: E-field components in ray basis, shape [..., N_rays, 3].
        """
        if E_surf.shape[:-1] != self.origins.shape[:-1] or E_surf.shape[-1] != 3:
            raise ValueError(
                f"E_surf must have shape {self.origins.shape[:-1] + (3,)}, "
                f"got {E_surf.shape}"
            )

        E_surf = E_surf.to(device=self.origins.device, dtype=self.origins.dtype)

        # Build frames
        e1_ray, e2_ray, ez_ray = self._ray_frame()             # [..., N_rays, 3]
        t1_surf, t2_surf, n_hat = self._surface_frame(surface_normal)

        # Surface-basis components -> world vector
        E_t1 = E_surf[..., 0:1]
        E_t2 = E_surf[..., 1:2]
        E_n  = E_surf[..., 2:3]

        E_world = E_t1 * t1_surf + E_t2 * t2_surf + E_n * n_hat  # [..., N_rays, 3]

        # Project onto ray basis
        E1 = (E_world * e1_ray).sum(dim=-1, keepdim=True)
        E2 = (E_world * e2_ray).sum(dim=-1, keepdim=True)
        E3 = (E_world * ez_ray).sum(dim=-1, keepdim=True)

        E_ray = torch.cat([E1, E2, E3], dim=-1)  # [..., N_rays, 3]
        return E_ray

    # -------------------------
    # Example interaction with meta-lens surrogate (optional)
    # -------------------------

    def sample_metalens_response(
        self,
        model: "MetaLensModel",
        *,
        lens_coords: Tensor,
        cond: Tensor,
    ) -> dict[str, Tensor]:
        """
        Example helper that queries your meta-lens surrogate for
        amplitude/phase responses at given lens coordinates.

        This is *just* a convenience to show how the RayBundle could interact
        with the surrogate model; adapt as needed.

        lens_coords: [..., N_rays, coord_dim] — coordinates on the metalens plane
        cond:        [..., cond_dim] or [..., N_rays, cond_dim]

        returns: same dict as MetaLensModel.forward()
        """
        return model(lens_coords, cond)


# -------------------------
# Example usage
# -------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example: batch of B bundles, each with N rays in 3D
    B, N = 2, 1024

    origins = torch.zeros(B, N, 3, device=device)        # all rays start at z=0 plane
    directions = torch.randn(B, N, 3, device=device)     # random directions
    directions = directions / directions.norm(dim=-1, keepdim=True)  # normalize

    # Complex field packed as [Re, Im]
    fields = torch.ones(B, N, 2, device=device)  # unit amplitude, zero phase, for instance

    # --- Monochromatic bundle ---
    wl_mono = 550.0  # nm
    rays_mono = RayBundle(
        origins=origins,
        directions=directions,
        fields=fields,
        wavelengths=torch.tensor(wl_mono, device=device),
    )

    print("Monochromatic bundle:")
    print("  base_shape:", rays_mono.base_shape)
    print("  is_monochromatic:", rays_mono.is_monochromatic)
    print("  wavelengths_per_ray shape:", rays_mono.wavelengths_per_ray().shape)

    # --- Per-ray wavelengths (e.g. sampled spectrum) ---
    wl_per_ray = torch.linspace(500.0, 600.0, N, device=device)  # [N]
    wl_per_ray = wl_per_ray.expand(B, N)                         # [B, N]

    rays_poly = RayBundle(
        origins=origins,
        directions=directions,
        fields=fields,
        wavelengths=wl_per_ray,
    )

    print("Per-ray bundle:")
    print("  base_shape:", rays_poly.base_shape)
    print("  is_monochromatic:", rays_poly.is_monochromatic)
    print("  wavelengths_per_ray shape:", rays_poly.wavelengths_per_ray().shape)

    
    # Example: one surface normal for the whole bundle (e.g. z-plane)
    surf_normal = torch.tensor([0.0, 0.0, 1.0], device=device)

    # Transform fields from ray basis to surface basis
    E_ray = torch.randn(B, N, 3, device=device)
    rays = RayBundle(origins=origins, directions=directions, fields=E_ray, wavelengths=wl_per_ray)
    E_surf = rays.field_ray_to_surface(E_ray, surf_normal)

    # And back:
    E_ray_recovered = rays.field_surface_to_ray(E_surf, surf_normal)
    print("Field transformation error (should be close to 0):", (E_ray - E_ray_recovered).abs().max().item())

    # Wavevector:
    k_vec, k_mag = rays.wavevector(n=1.5)
    print("k_vec shape:", k_vec.shape)  # [B, N, 3]
    print("k_mag shape:", k_mag.shape)  # [B, N]
