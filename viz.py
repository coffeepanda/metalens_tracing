# viz.py  (or inside metalens_siren_opt_example.py)
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
import matplotlib.pyplot as plt

from rays import RayBundle
from surfaces import PlaneSurface
from metalens import Metalens


def plot_sideview_xz(
    metalens: Metalens,
    rays_in: RayBundle,
    sensor_plane: PlaneSurface,
    *,
    n_rays_plot: int = 16,
    z_entry: Optional[float] = None,
    title: str = "Metalens side view (x–z)",
) -> None:
    """
    Plot a side view (x–z) of a subset of rays:

        entry plane  -> lens plane -> sensor plane

    using the same geometric pipeline as your simulation:
      - intersect rays with lens
      - compute phase gradient + diffract
      - intersect with sensor

    Assumptions:
      - metalens.surface is a PlaneSurface with normal ~ +z
      - rays propagate roughly along +z
    """
    device = rays_in.origins.device

    # ---------------------------
    # 1) Forward pass geometry (no gradients)
    # ---------------------------
    # Phase + gradients on lens
    phase_data = metalens.evaluate_phase_on_rays(
        rays_in,
        require_higher_order=False,  # no need for second-order here
    )

    rays_hit: RayBundle = phase_data["rays_hit"]          # [M, ...]
    pts_lens: Tensor = phase_data["points_world"]         # [M, 3]
    phase_grad_world: Tensor = phase_data["phase_grad_world"]  # [M, 3]

    # Diffract through lens
    rays_out = metalens.diffract_rays(
        rays_hit,
        phase_grad_world=phase_grad_world,
        drop_invalid=True,
    )

    # Intersect with sensor
    inter_sensor = rays_out.find_intersection(sensor_plane, drop_invalid=True)
    pts_sensor: Tensor = inter_sensor["points"]           # [K, 3]

    if pts_lens.numel() == 0 or pts_sensor.numel() == 0:
        print("No rays hit lens or sensor; nothing to plot.")
        return

    # ---------------------------
    # 2) Reconstruct an "entry plane" before the lens
    # ---------------------------
    # lens z coordinate:
    lens_z = float(metalens.surface.origin[2].item())  # PlaneSurface origin z

    # If not provided, use min z of *original* bundle as entry plane
    if z_entry is None:
        z_entry = float(rays_in.origins[..., 2].min().item())

    p_lens = pts_lens  # [M, 3]
    d_in = rays_hit.directions  # [M, 3]

    # Solve p_entry = p_lens - t * d_in such that p_entry_z = z_entry
    dz = d_in[..., 2]  # [M]
    # Avoid division by zero; mask out any weird rays
    eps = 1e-9
    dz_safe = torch.where(dz.abs() < eps, torch.full_like(dz, eps), dz)

    t_entry = (p_lens[..., 2] - z_entry) / dz_safe       # [M]
    p_entry = p_lens - d_in * t_entry.unsqueeze(-1)      # [M, 3]

    # ---------------------------
    # 3) Align counts / subsample rays for plotting
    # ---------------------------
    # Rays_out has shape [M_out, ...]; pts_sensor has [K, 3]
    # For typical configs, K == rays_out.origins.shape[0], but we guard anyway.
    M = min(p_entry.shape[0], p_lens.shape[0], pts_sensor.shape[0])
    if M == 0:
        print("No rays survive to sensor; nothing to plot.")
        return

    n_plot = min(n_rays_plot, M)
    # Uniformly sample indices across surviving rays
    idx = torch.linspace(0, M - 1, steps=n_plot, device=device).long()

    p_entry_sel = p_entry[idx]       # [n_plot, 3]
    p_lens_sel = p_lens[idx]         # [n_plot, 3]
    p_sensor_sel = pts_sensor[idx]   # [n_plot, 3]

    # ---------------------------
    # 4) Convert to CPU numpy for plotting
    # ---------------------------
    entry_x = p_entry_sel[:, 0].cpu().numpy()
    entry_z = p_entry_sel[:, 2].cpu().numpy()

    lens_x = p_lens_sel[:, 0].cpu().numpy()
    lens_z_arr = p_lens_sel[:, 2].cpu().numpy()

    sensor_x = p_sensor_sel[:, 0].cpu().numpy()
    sensor_z_arr = p_sensor_sel[:, 2].cpu().numpy()

    # Sensor z (should be constant)
    sensor_z = float(sensor_plane.origin[2].item())

    # ---------------------------
    # 5) Matplotlib plot
    # ---------------------------
    fig, ax = plt.subplots(figsize=(6, 4))

    for i in range(n_plot):
        zs = [entry_z[i], lens_z_arr[i], sensor_z_arr[i]]
        xs = [entry_x[i], lens_x[i], sensor_x[i]]
        ax.plot(zs, xs, "-")

    # Draw lens and sensor planes as vertical lines
    ax.axvline(lens_z, color="k", linestyle="--", label="Lens plane")
    ax.axvline(sensor_z, color="gray", linestyle=":", label="Sensor plane")

    ax.set_xlabel("z [m]")
    ax.set_ylabel("x [m]")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_aspect("equal", "box")

    plt.tight_layout()
    plt.show()
