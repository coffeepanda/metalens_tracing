# validate_two_param_fd.py
from __future__ import annotations

import torch
from torch import Tensor

from rays import RayBundle
from surfaces import PlaneSurface, CircularAperture
from metalens import Metalens, MetalensConfig
from networks import SurrogateNet, MetaLensConfig

from param_field import TwoParamRepNet  # from above


# Use double precision for better finite-difference stability
torch.set_default_dtype(torch.float64)


def build_test_ray_bundle(device: torch.device) -> RayBundle:
    """
    Build a small ray bundle aimed at the metalens:

      - Origins: a 5x5 grid at z = z0 < 0
      - Directions: all along +z
      - Wavelength: 550 nm

    Shapes:
      origins:    [1, N, 3]
      directions: [1, N, 3]
      wavelengths:[1, N]
      fields:     [1, N, 3]
    """
    Nx, Ny = 5, 5
    N = Nx * Ny

    z0 = -0.1  # 10 cm in front of the lens

    x = torch.linspace(-0.5e-3, 0.5e-3, Nx, device=device)
    y = torch.linspace(-0.5e-3, 0.5e-3, Ny, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    zz = torch.full_like(xx, z0)

    origins = torch.stack([xx, yy, zz], dim=-1).reshape(1, N, 3)

    d = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 3)
    directions = d.expand_as(origins)

    wavelengths = torch.full(
        (1, N),
        550e-9,
        device=device,
        dtype=torch.get_default_dtype(),
    )

    fields = torch.zeros(1, N, 3, device=device, dtype=torch.get_default_dtype())
    fields[..., 0] = 1.0

    return RayBundle(
        origins=origins,
        directions=directions,
        fields=fields,
        wavelengths=wavelengths,
    )


def build_metalens_two_param(device: torch.device) -> tuple[Metalens, RayBundle, PlaneSurface]:
    """
    Build a Metalens using TwoParamField as the design parameterization:

      - Plane metalens at z = 0, circular aperture of radius 0.5 mm
      - TwoParamField (alpha, beta) -> w(x,y)
      - SurrogateNet: maps (w, λ) -> [amp, phase]
      - Sensor plane at z = 5 cm
    """
    # --- Surrogate config ---
    base_cfg = MetaLensConfig(
        coord_dim=2,
        w_dim=1,          # one scalar w per (x,y)
        cond_dim=1,       # only wavelength
        hidden_dims_param=(64, 64, 64),
        hidden_dims_surrogate=(64, 64, 64),
    )

    surrogate_net = SurrogateNet(base_cfg).to(device).double()

    # Typically: FIX surrogate; metalens design is only in param_field
    for p in surrogate_net.parameters():
        p.requires_grad_(False)

    # --- Metalens surface (plane at z=0) ---
    aperture = CircularAperture(radius=0.5e-3)
    lens_surface = PlaneSurface(
        origin=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        aperture=aperture,
    )

    # 2-parameter parameter field
    param_field = TwoParamRepNet(w_dim=1, r0=0.5e-3).to(device).double()

    ml_cfg = MetalensConfig(
        cond_dim=base_cfg.cond_dim,
        n_in=1.0,   # air
        n_out=1.5,  # glass
    )

    metalens = Metalens(
        surface=lens_surface,
        param_field=param_field,
        surrogate_net=surrogate_net,
        cfg=ml_cfg,
    ).to(device).double()

    rays_in = build_test_ray_bundle(device)

    # Sensor plane at z = 5 cm, big aperture
    z_sensor = 0.05
    sensor_aperture = CircularAperture(radius=5e-3)
    sensor_plane = PlaneSurface(
        origin=(0.0, 0.0, z_sensor),
        normal=(0.0, 0.0, 1.0),
        aperture=sensor_aperture,
    )

    return metalens, rays_in, sensor_plane


def forward_loss(
    metalens: Metalens,
    rays_in: RayBundle,
    sensor_plane: PlaneSurface,
) -> Tensor:
    """
    Full forward pass:

        rays_in
          -> intersect metalens
          -> coords_local -> w(coords) via TwoParamField
          -> surrogate(w, λ) -> phase
          -> ∂φ/∂coords via autograd.grad
          -> ∇φ_world
          -> diffract()
          -> intersect sensor
          -> spot-size loss on the sensor plane

    Returns: scalar loss (mean squared radius at the sensor).
    """
    # 1) phase + gradients on the lens surface
    phase_data = metalens.evaluate_phase_on_rays(
        rays_in,
        require_higher_order=True,  # needed so grad_coords depends on (alpha, beta)
    )

    rays_hit = phase_data["rays_hit"]
    phase_grad_world = phase_data["phase_grad_world"]  # [M, 3]
    print("⚠️ phase_grad_world sample:", phase_grad_world.requires_grad)  # Should have grad_fn since it depends on param_field

    # 2) Diffract through the metalens
    rays_out = metalens.diffract_rays(
        rays_hit,
        phase_grad_world=phase_grad_world,
        drop_invalid=True,
    )

    # 3) Intersect with the sensor
    inter_sensor = rays_out.find_intersection(sensor_plane, drop_invalid=True)
    rays_at_sensor: RayBundle = inter_sensor["bundle"]
    sensor_points: Tensor = inter_sensor["points"]  # [K, 3]

    if rays_at_sensor.origins.shape[0] == 0:
        # Nothing hits the sensor: heavy penalty
        return torch.tensor(1.0, device=sensor_points.device, dtype=sensor_points.dtype)

    sensor_xy = sensor_points[..., :2]  # [K, 2]
    target_xy = torch.zeros_like(sensor_xy)

    # Spot-size loss
    loss = ((sensor_xy - target_xy) ** 2).sum(dim=-1).mean()
    return loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    metalens, rays_in, sensor_plane = build_metalens_two_param(device)

    param_field: TwoParamField = metalens.param_field  # type: ignore

    print("Initial alpha, beta:", param_field.alpha.item(), param_field.beta.item())

    # ------------------------------------------------------
    # 1) Autograd gradient d loss / d [alpha, beta]
    # ------------------------------------------------------
    print("\n🍏 Computing autograd gradient d(loss)/d [alpha, beta] ...")

    metalens.zero_grad(set_to_none=True)

    loss = forward_loss(metalens, rays_in, sensor_plane)
    print("Loss (autograd run):", float(loss))

    loss.backward()

    grad_alpha = param_field.alpha.grad.detach().clone()
    grad_beta = param_field.beta.grad.detach().clone()

    print("Autograd gradients:")
    print("  dL/d alpha =", grad_alpha.item())
    print("  dL/d beta  =", grad_beta.item())

    # ------------------------------------------------------
    # 2) Finite-difference gradients
    # ------------------------------------------------------
    print("\n🍎 Computing finite-difference gradients ...")

    eps = 1e-3  # FD step; works well with double precision

    with torch.no_grad():
        alpha0 = param_field.alpha.detach().clone()
        beta0 = param_field.beta.detach().clone()

    # dL/d alpha
    with torch.no_grad():
        param_field.alpha.copy_(alpha0 + eps)
    loss_plus = forward_loss(metalens, rays_in, sensor_plane).item()

    with torch.no_grad():
        param_field.alpha.copy_(alpha0 - eps)
    loss_minus = forward_loss(metalens, rays_in, sensor_plane).item()

    fd_grad_alpha = (loss_plus - loss_minus) / (2.0 * eps)

    # restore alpha
    with torch.no_grad():
        param_field.alpha.copy_(alpha0)

    # dL/d beta
    with torch.no_grad():
        param_field.beta.copy_(beta0 + eps)
    loss_plus = forward_loss(metalens, rays_in, sensor_plane).item()

    with torch.no_grad():
        param_field.beta.copy_(beta0 - eps)
    loss_minus = forward_loss(metalens, rays_in, sensor_plane).item()

    fd_grad_beta = (loss_plus - loss_minus) / (2.0 * eps)

    with torch.no_grad():
        param_field.beta.copy_(beta0)

    print("Finite-difference gradients:")
    print("  dL/d alpha ≈", fd_grad_alpha)
    print("  dL/d beta  ≈", fd_grad_beta)

    # ------------------------------------------------------
    # 3) Compare
    # ------------------------------------------------------
    abs_diff_alpha = abs(grad_alpha.item() - fd_grad_alpha)
    abs_diff_beta = abs(grad_beta.item() - fd_grad_beta)

    # Avoid div by zero for relative error
    rel_err_alpha = abs_diff_alpha / (abs(fd_grad_alpha) + 1e-12)
    rel_err_beta = abs_diff_beta / (abs(fd_grad_beta) + 1e-12)

    print("\n🔍 Comparison:")
    print(
        f"  alpha: auto = {grad_alpha.item(): .6e}, "
        f"fd = {fd_grad_alpha: .6e}, "
        f"|diff| = {abs_diff_alpha: .3e}, "
        f"rel err ~ {rel_err_alpha: .3e}"
    )
    print(
        f"  beta : auto = {grad_beta.item(): .6e}, "
        f"fd = {fd_grad_beta: .6e}, "
        f"|diff| = {abs_diff_beta: .3e}, "
        f"rel err ~ {rel_err_beta: .3e}"
    )

    print(
        "\n✅ If |diff| is small (say <1e-4) and relative error is modest "
        "(<1e-2), then gradients are propagating correctly from the sensor "
        "all the way back to the metalens parameters (alpha, beta) through "
        "your *current* Metalens + ParameterField architecture."
    )


if __name__ == "__main__":
    main()
