# metalens_siren_opt_example.py
from __future__ import annotations

import torch
from torch import Tensor

from rays import RayBundle
from surfaces import PlaneSurface, CircularAperture
from param_field import GridParameterField, NeuralParameterField
from metalens import Metalens, MetalensConfig
from networks import MetaLensConfig, ParamNet, SurrogateNet



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

    z0 = -0.002 # 0.5 cm in front of the lens

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




def build_siren_metalens(device: torch.device) -> tuple[Metalens, RayBundle, PlaneSurface]:
    """
    Build a Metalens using your SIREN-based ParamNet + SurrogateNet
    wrapped in a NeuralParameterField.
    """
    # ---------------------------
    # 1) MetaLensConfig for SIREN nets
    # ---------------------------
    # We only use wavelength as condition here, so cond_dim=1.
    siren_cfg = MetaLensConfig(
        coord_dim=2,
        w_dim=16,                     # structural parameter dimension
        cond_dim=1,                   # wavelength only
        hidden_dims_param=(64, 64, 64),
        hidden_dims_surrogate=(64, 64, 64),
    )

    # Representation network: coords -> w (SIREN-based)
    rep_net = ParamNet(siren_cfg).to(device)

    # Wrap as a ParameterField
    # param_field = NeuralParameterField(rep_net).to(device)
    param_field = GridParameterField(
        xy_min=(-0.5e-3, -0.5e-3),
        xy_max=(0.5e-3, 0.5e-3),
        grid_shape=(32, 32),
        w_dim=siren_cfg.w_dim,
        init_scale=1e-2,
    ).to(device)

    # Surrogate network: (w, cond) -> [amplitude, phase] (SIREN-based)
    surrogate = SurrogateNet(siren_cfg).to(device)

    # Usually you’ll freeze the surrogate if it’s pre-trained
    # for p in surrogate.parameters():
    #     p.requires_grad_(False)

    # ---------------------------
    # 2) Metalens surface
    # ---------------------------
    aperture = CircularAperture(radius=0.5e-3)
    lens_surface = PlaneSurface(
        origin=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        aperture=aperture,
    )

    # Make sure MetalensConfig.cond_dim matches siren_cfg.cond_dim
    ml_cfg = MetalensConfig(
        cond_dim=siren_cfg.cond_dim,
        n_in=1.0,   # air
        n_out=1.5,  # glass
    )

    metalens = Metalens(
        surface=lens_surface,
        param_field=param_field,
        surrogate_net=surrogate,
        cfg=ml_cfg,
    ).to(device)

    # ---------------------------
    # 3) Rays in and sensor plane
    # ---------------------------
    rays_in = build_test_ray_bundle(device)

    z_sensor = 0.002
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
    *,
    require_higher_order: bool = False,
) -> Tensor:
    """
    Full forward pass:

        rays_in
          -> intersect metalens
          -> coords_local -> w(coords) via SIREN
          -> surrogate(w, λ) -> phase
          -> ∂φ/∂coords via autograd.grad
          -> ∇φ_world
          -> diffract()
          -> intersect sensor
          -> spot-size loss on the sensor plane
    """
    phase_data = metalens.evaluate_phase_on_rays(
        rays_in,
        require_higher_order=require_higher_order,
        grad_method="fd_interp",
        fd_grid_size=32,
        fd_margin=0.00001,
    )

    rays_hit = phase_data["rays_hit"]
    phase_grad_world = phase_data["phase_grad_world"]  # [M, 3]

    # Diffract through the metalens
    rays_out = metalens.diffract_rays(
        rays_hit,
        phase_grad_world=phase_grad_world,
        drop_invalid=True,
    )

    # Intersect with the sensor
    inter_sensor = rays_out.find_intersection(sensor_plane, drop_invalid=True)
    rays_at_sensor: RayBundle = inter_sensor["bundle"]
    sensor_points: Tensor = inter_sensor["points"]  # [K, 3]

    if rays_at_sensor.origins.shape[0] == 0:
        # Nothing hits the sensor: heavy penalty
        return torch.tensor(
            1.0,
            device=sensor_points.device,
            dtype=sensor_points.dtype,
        )

    sensor_xy = sensor_points[..., :2]  # [K, 2]
    target_xy = torch.zeros_like(sensor_xy)

    # loss = ((sensor_xy - target_xy) ** 2).sum(dim=-1).mean()
    loss = ((sensor_xy - target_xy) ** 2).mean()
    return loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    metalens, rays_in, sensor_plane = build_siren_metalens(device)

    # Optimize only the SIREN representation (ParamNet via param_field)
    optimizer = torch.optim.Adam([{'params': metalens.param_field.parameters()}, {'params': metalens.surrogate_net.parameters()}], lr=1e-3)

    # --- Initial loss ---
    loss0 = forward_loss(metalens, rays_in, sensor_plane, require_higher_order=False)
    print(f"Initial loss: {loss0.item():.6e}")

    loss_history = []
    num_steps = 1000
    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = forward_loss(metalens, rays_in, sensor_plane, require_higher_order=True)
        loss.backward()

        max_norm = 0.0
        sample_info = None
        for name, p in metalens.param_field.named_parameters():
            if p.grad is not None:
                gnorm = p.grad.norm().item()
                if gnorm > max_norm:
                    max_norm = gnorm
                    sample_info = (name, gnorm)
        if sample_info is None:
            print(f"Step {step+1:3d} | loss = {loss.item():.6e} | ⚠️ NO GRAD on param_field")
        else:
            name, gnorm = sample_info
            if (step + 1) % 10 == 0 or step == 0:
                print(
                    f"Step {step+1:3d} | loss = {loss.item():.6e} "
                    f"| sample grad[{name}] norm = {gnorm:.3e}"
                )


        optimizer.step()
        loss_history.append(loss.item())

        # if (step + 1) % 10 == 0 or step == 0:
        #     print(f"Step {step+1:3d} | loss = {loss.item():.6e}")

    loss_final = forward_loss(metalens, rays_in, sensor_plane, require_higher_order=False)
    print(f"Final loss:   {loss_final.item():.6e}")

    from viz import plot_sideview_xz
    plot_sideview_xz(
        metalens,
        rays_in,
        sensor_plane,
        n_rays_plot=20,
        title="Metalens side view after optimization",
    )

    # export loss_hist to matplotlib image 
    import matplotlib.pyplot as plt
    plt.plot(loss_history)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.grid()
    plt.savefig("loss_history.png")
    print("⚠️ Saved loss history plot as loss_history.png")


if __name__ == "__main__":
    main()
