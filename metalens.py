# metalens.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from rays import RayBundle
from surfaces import PlaneSurface
from param_field import ParameterField


@dataclass
class MetalensConfig:
    """
    Configuration for a Metalens object.

    - cond_dim: dimensionality of the condition vector fed to the surrogate.
                We assume the first entry is wavelength; the rest you can use
                for angles, polarization, etc.
    - n_in: refractive index of incident medium.
    - n_out: refractive index of transmitted medium.
    """
    cond_dim: int = 1
    n_in: float = 1.0
    n_out: float = 1.5


class Metalens(nn.Module):
    """
    Metalens wrapper that couples:

      1) A surface (PlaneSurface for now),
      2) A parameter field: coords -> w (structure parameters),
      3) A surrogate network: (w, cond) -> [amplitude, phase].

    It provides helpers to:
      - Intersect a ray bundle with the surface,
      - Evaluate phase at intersection points,
      - Compute phase gradients w.r.t. surface coordinates,
      - Build a phase gradient vector in world coordinates for generalized Snell,
      - Diffract rays using that phase gradient.

    The *design parameters* of the metalens are whatever lives inside
    `param_field.parameters()` (e.g. a neural network, a grid, or something else),
    while the surrogate is usually pre-trained and frozen.
    """

    def __init__(
        self,
        surface: PlaneSurface,
        param_field: ParameterField,
        surrogate_net: nn.Module,
        cfg: Optional[MetalensConfig] = None,
    ) -> None:
        super().__init__()
        self.surface = surface
        self.param_field = param_field
        self.surrogate_net = surrogate_net
        self.cfg = cfg or MetalensConfig()

    @property
    def n_in(self) -> float:
        return self.cfg.n_in

    @property
    def n_out(self) -> float:
        return self.cfg.n_out

    # ------------------------------------------------------------------
    # Core helper: intersect rays with lens surface and get valid bundle
    # ------------------------------------------------------------------

    def intersect_rays(self, rays: RayBundle) -> Dict[str, Any]:
        """
        Intersect rays with the metalens surface, dropping invalid rays.

        Returns dict with:
          - "bundle":  RayBundle of valid rays (origins at intersection points)
          - "points":  [M, 3] intersection points (world coords)
          - "mask":    original boolean mask [..., N_rays]
        """
        inter = rays.find_intersection(self.surface, drop_invalid=True)
        bundle: RayBundle = inter["bundle"]
        points: Tensor = inter["points"]
        mask: Tensor = inter["mask"]
        return {"bundle": bundle, "points": points, "mask": mask}

    # ------------------------------------------------------------------
    # Build condition vector for surrogate
    # ------------------------------------------------------------------

    def build_cond(
        self,
        wavelengths: Tensor,
        *,
        extra: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Build the condition vector for the surrogate.

        wavelengths: [M] or [..., M] in meters.
        extra:       optional extra condition tensor broadcastable to
                     [M, (cond_dim - 1)] if you want to encode angles,
                     polarization, etc.

        Returns:
            cond: [M, cond_dim]
                  cond[..., 0] = wavelength
        """
        wl = wavelengths
        if wl.ndim > 1:
            wl = wl.reshape(-1)  # flatten leading dims into one batch

        M = wl.shape[0]
        device = wl.device
        dtype = wl.dtype

        cond = torch.zeros(M, self.cfg.cond_dim, device=device, dtype=dtype)
        cond[:, 0] = wl

        if extra is not None:
            extra = extra.to(device=device, dtype=dtype)
            extra = extra.reshape(M, -1)
            if extra.shape[1] > self.cfg.cond_dim - 1:
                raise ValueError(
                    f"Extra condition has dim={extra.shape[1]}, "
                    f"but cond_dim-1={self.cfg.cond_dim - 1}."
                )
            cond[:, 1 : 1 + extra.shape[1]] = extra

        return cond  # [M, cond_dim]

    # ------------------------------------------------------------------
    # Phase evaluation + gradients at intersection points
    # ------------------------------------------------------------------

    def phase_and_gradients(
        self,
        coords_local_xy: Tensor,
        wavelengths: Tensor,
        *,
        require_higher_order: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Evaluate surrogate phase at surface-local coordinates and compute
        gradients:

            w = param_field(coords_local_xy)
            [amp, phase] = surrogate(w, cond)

            and ∂phase/∂coords_local_xy via autograd.

        Arguments:
            coords_local_xy: [M, 2]
                Local coordinates on the lens surface (x_local, y_local),
                in *meters*.
            wavelengths: [M] or [..., M]
                Wavelength per ray.
            require_higher_order: bool
                If True, constructs the graph needed for higher-order
                derivatives (create_graph=True). If you only need first-order
                gradients for training, you can set this to False to save
                memory.

        Returns dict with:
            - "phase":        [M]          scalar phase for each ray
            - "amplitude":    [M]          scalar amplitude (from surrogate)
            - "grad_coords":  [M, 2]       ∂phi/∂x_local, ∂phi/∂y_local
            - "w":            [M, w_dim]   structure parameters at these coords
        """
        # 🔑 Always make coords a leaf that requires grad so we can compute dφ/dcoords,
        # regardless of whether we keep a graph for higher-order stuff.
        coords_local_xy = coords_local_xy.clone().detach().requires_grad_(True)

        # Parameter field: coords -> w
        w = self.param_field(coords_local_xy)  # [M, w_dim]

        M = w.shape[0]
        wl = wavelengths
        if wl.ndim > 1:
            wl = wl.reshape(-1)

        if wl.shape[0] != M:
            raise ValueError(
                f"Mismatch between coords (M={M}) and wavelengths (len={wl.shape[0]})."
            )

        # Build condition vector for surrogate
        cond = self.build_cond(wl)  # [M, cond_dim]

        # Surrogate expects shapes [..., N, w_dim] and [..., N, cond_dim]
        w_in = w.unsqueeze(0)        # [1, M, w_dim]
        cond_in = cond.unsqueeze(0)  # [1, M, cond_dim]

        out = self.surrogate_net(w_in, cond_in).squeeze(0)  # [M, 2]

        amplitude = out[..., 0]  # [M]
        phase = out[..., 1]      # [M]

        # Compute gradient of phase wrt coords (x, y) at each ray
        grad_phi_coords = torch.autograd.grad(
            phase,
            coords_local_xy,
            grad_outputs=torch.ones_like(phase),
            create_graph=require_higher_order,   # 🔑 keep graph only if needed
        )[0]  # [M, 2]

        result: Dict[str, Tensor] = {
            "phase": phase,
            "amplitude": amplitude,
            "grad_coords": grad_phi_coords,
            "w": w,
        }
        return result

    # ------------------------------------------------------------------
    # Phase + gradients via coarse-grid FD + interpolation
    # ------------------------------------------------------------------

    def phase_and_gradients_fd_interp(
        self,
        coords_local_xy: Tensor,
        wavelengths: Tensor,
        *,
        grid_size: int = 32,
        margin: float = 0.05,
    ) -> Dict[str, Tensor]:
        """
        Compute phase and phase gradient using a coarse grid + finite differences.

        Idea:
          1) Build a regular grid over the region covered by coords_local_xy.
          2) Evaluate phase on that grid (via param_field + surrogate).
          3) Compute ∂phi/∂x, ∂phi/∂y on the grid via finite differences.
          4) Bilinearly interpolate the gradient at the actual ray positions.

        This enforces a larger spatial scale (controlled by `grid_size`) and
        avoids second-order autograd. Gradients still flow to param_field
        and surrogate parameters because all operations are differentiable.
        """
        device = coords_local_xy.device
        dtype = coords_local_xy.dtype

        coords = coords_local_xy  # [M, 2]
        x = coords[..., 0]
        y = coords[..., 1]

        # 1) Build bounding box with small margin
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        dx = x_max - x_min
        dy = y_max - y_min

        x_min = x_min - margin * dx
        x_max = x_max + margin * dx
        y_min = y_min - margin * dy
        y_max = y_max + margin * dy

        # 2) Regular grid over this box
        Nx = Ny = grid_size
        xs = torch.linspace(x_min, x_max, Nx, device=device, dtype=dtype)
        ys = torch.linspace(y_min, y_max, Ny, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [Ny, Nx]

        coords_grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # [Ny*Nx, 2]

        # For simplicity, use a single representative wavelength for the grid.
        # (If you want full multi-λ support later, you can build separate grids.)
        wl_flat = wavelengths.reshape(-1)
        wl0 = wl_flat.mean()
        wl_grid = wl0.expand(coords_grid.shape[0])  # [Ny*Nx]

        # 3) Evaluate phase/amplitude on the grid via param_field + surrogate
        w_grid = self.param_field(coords_grid)      # [Ng, w_dim]
        cond_grid = self.build_cond(wl_grid)        # [Ng, cond_dim]

        out_grid = self.surrogate_net(
            w_grid.unsqueeze(0),      # [1, Ng, w_dim]
            cond_grid.unsqueeze(0),   # [1, Ng, cond_dim]
        ).squeeze(0)                  # [Ng, 2]

        amp_grid = out_grid[..., 0].reshape(Ny, Nx)   # [Ny, Nx]
        phase_grid = out_grid[..., 1].reshape(Ny, Nx) # [Ny, Nx]

        # 4) Finite-difference gradients on the grid
        hx = (x_max - x_min) / (Nx - 1 + 1e-12)
        hy = (y_max - y_min) / (Ny - 1 + 1e-12)

        dphi_dx = torch.zeros_like(phase_grid)
        dphi_dy = torch.zeros_like(phase_grid)

        # Central differences (interior)
        dphi_dx[:, 1:-1] = (phase_grid[:, 2:] - phase_grid[:, :-2]) / (2.0 * hx)
        dphi_dy[1:-1, :] = (phase_grid[2:, :] - phase_grid[:-2, :]) / (2.0 * hy)

        # Forward/backward differences (boundaries)
        dphi_dx[:, 0] = (phase_grid[:, 1] - phase_grid[:, 0]) / hx
        dphi_dx[:, -1] = (phase_grid[:, -1] - phase_grid[:, -2]) / hx

        dphi_dy[0, :] = (phase_grid[1, :] - phase_grid[0, :]) / hy
        dphi_dy[-1, :] = (phase_grid[-1, :] - phase_grid[-2, :]) / hy

        # 5) Pack gradients into a 2-channel "image" and bilinearly sample
        #    at the ray intersection coords.
        # grad_grid: [1, 2, Ny, Nx]
        grad_grid = torch.stack([dphi_dx, dphi_dy], dim=0).unsqueeze(0)

        # Convert coords to [-1, 1] normalized for grid_sample
        x_norm = 2.0 * (x - x_min) / (x_max - x_min + 1e-12) - 1.0  # [M]
        y_norm = 2.0 * (y - y_min) / (y_max - y_min + 1e-12) - 1.0  # [M]

        # grid_sample expects [N, H_out, W_out, 2]
        # We'll treat each coord as (H_out=M, W_out=1)
        sample_grid = torch.stack([x_norm, y_norm], dim=-1)  # [M, 2]
        sample_grid = sample_grid.unsqueeze(0).unsqueeze(2)  # [1, M, 1, 2]

        grad_samples = F.grid_sample(
            grad_grid,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [1, 2, M, 1]

        grad_samples = grad_samples.squeeze(0).squeeze(-1).transpose(0, 1)  # [M, 2]

        # 6) For phase/amplitude at the actual ray coords, just evaluate directly
        #    with true per-ray wavelengths (no need to interpolate).
        wl = wavelengths
        if wl.ndim > 1:
            wl = wl.reshape(-1)

        w_points = self.param_field(coords_local_xy)   # [M, w_dim]
        cond_points = self.build_cond(wl)              # [M, cond_dim]
        out_points = self.surrogate_net(
            w_points.unsqueeze(0),
            cond_points.unsqueeze(0),
        ).squeeze(0)                                   # [M, 2]

        amp_points = out_points[..., 0]   # [M]
        phase_points = out_points[..., 1] # [M]

        result: Dict[str, Tensor] = {
            "phase": phase_points,
            "amplitude": amp_points,
            "grad_coords": grad_samples,  # smoothed ∂φ/∂x, ∂φ/∂y
            "w": w_points,
        }
        return result

    # ------------------------------------------------------------------
    # Phase gradient in world coordinates
    # ------------------------------------------------------------------

    def phase_gradient_world(
        self,
        points_world: Tensor,
        grad_coords_local_xy: Tensor,
    ) -> Tensor:
        """
        Convert phase gradient from local surface coordinates (x_local, y_local)
        to a world-space gradient vector ∇φ suitable for passing into
        RayBundle.diffract(..., phase_gradient=...).
        """
        device = points_world.device
        dtype = points_world.dtype

        gx = grad_coords_local_xy[..., 0]
        gy = grad_coords_local_xy[..., 1]

        g_local = torch.zeros(points_world.shape[0], 3, device=device, dtype=dtype)
        g_local[:, 0] = gx
        g_local[:, 1] = gy

        g_world = self.surface.local_normal_to_world(g_local)  # [M, 3]
        return g_world

    # ------------------------------------------------------------------
    # High-level helper: one step of "metalens → phase → ∇φ"
    # ------------------------------------------------------------------

    def evaluate_phase_on_rays(
        self,
        rays_in: RayBundle,
        *,
        require_higher_order: bool = True,
        grad_method: str = "autograd",   # "autograd" or "fd_interp"
        fd_grid_size: int = 32,
        fd_margin: float = 0.05,
    ) -> Dict[str, Any]:
        """
        High-level convenience method:

            1) Intersect rays with lens surface (drop invalid).
            2) Map intersection points into surface-local coords.
            3) Evaluate phase + gradients via parameter field + surrogate.
            4) Build world-space phase gradient vector.

        Returns dict with:
            - "rays_hit":          RayBundle of valid incident rays
            - "points_world":      [M, 3] intersection points
            - "coords_local_xy":   [M, 2] local coords
            - "phase":             [M]
            - "amplitude":         [M]
            - "grad_coords":       [M, 2]
            - "w":                 [M, w_dim]
            - "phase_grad_world":  [M, 3]
        """
        inter = self.intersect_rays(rays_in)
        rays_hit: RayBundle = inter["bundle"]
        pts_world: Tensor = inter["points"]  # [M, 3]

        pts_local = self.surface.world_to_local(pts_world)  # [M, 3]
        coords_local_xy = pts_local[..., :2]                # [M, 2]

        wl = rays_hit.wavelengths_per_ray()  # [M]

        if grad_method == "autograd":
            phase_data = self.phase_and_gradients(
                coords_local_xy,
                wl,
                require_higher_order=require_higher_order,
            )
        elif grad_method == "fd_interp":
            # finite-difference + interpolation; no higher-order graph
            phase_data = self.phase_and_gradients_fd_interp(
                coords_local_xy,
                wl,
                grid_size=fd_grid_size,
                margin=fd_margin,
            )
        else:
            raise ValueError(f"Unknown grad_method='{grad_method}'")

        grad_world = self.phase_gradient_world(
            pts_world,
            phase_data["grad_coords"],
        )

        result: Dict[str, Any] = dict(phase_data)
        result["rays_hit"] = rays_hit
        result["points_world"] = pts_world
        result["coords_local_xy"] = coords_local_xy
        result["phase_grad_world"] = grad_world
        return result

    # ------------------------------------------------------------------
    # Diffract rays using computed phase gradient
    # ------------------------------------------------------------------

    def diffract_rays(
        self,
        rays_in: RayBundle,
        phase_grad_world: Tensor,
        *,
        drop_invalid: bool = True,
    ) -> RayBundle:
        """
        Use the provided world-space phase gradient to diffract the rays.
        """
        return rays_in.diffract(
            surface=self.surface,
            n1=self.n_in,
            n2=self.n_out,
            phase_gradient=phase_grad_world,
            drop_invalid=drop_invalid,
        )
