from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn, Tensor


# -----------------------------
# SIREN building blocks
# -----------------------------

class SirenLayer(nn.Module):
    """
    Single SIREN layer with sine activation and SIREN-style initialization.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0

        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                # From SIREN paper
                bound = 1.0 / self.in_features
            else:
                bound = (6.0 / self.in_features) ** 0.5 / self.omega_0

            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., in_features]
        return torch.sin(self.omega_0 * self.linear(x))


class SirenMLP(nn.Module):
    """
    Generic SIREN MLP: input_dim -> hidden_dims -> output_dim.
    Can optionally have a linear last layer (no sine).
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (64, 64, 64),
        *,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        outermost_linear: bool = True,
        final_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()

        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []

        # First SIREN layer
        layers.append(
            SirenLayer(
                dims[0],
                dims[1],
                is_first=True,
                omega_0=first_omega_0,
            )
        )

        # Hidden SIREN layers
        for i in range(1, len(dims) - 1):
            layers.append(
                SirenLayer(
                    dims[i],
                    dims[i + 1],
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        # Output layer
        if outermost_linear:
            layers.append(nn.Linear(dims[-1], output_dim))
        else:
            layers.append(SirenLayer(dims[-1], output_dim, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*layers)
        self.final_activation = final_activation

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [..., input_dim]
        returns: [..., output_dim]
        """
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_dim)
        y_flat = self.net(x_flat)
        if self.final_activation is not None:
            y_flat = self.final_activation(y_flat)
        return y_flat.reshape(*orig_shape, self.output_dim)


# -----------------------------
# Metalens networks
# -----------------------------

@dataclass
class MetaLensConfig:
    coord_dim: int = 2               # (x, y) – can be extended
    w_dim: int = 16                  # dimensionality of nanostructure parameters
    cond_dim: int = 4                # e.g. [angle, polarization, wavelength, ...]
    hidden_dims_param: Iterable[int] = (64, 64, 64)
    hidden_dims_surrogate: Iterable[int] = (64, 64, 64)


class ParamNet(nn.Module):
    """
    First network:
        coords (x, y) -> structural parameters w.

    Accepts coords with shape [..., N, coord_dim] where leading dims are arbitrary.
    Produces w with shape [..., N, w_dim].
    """
    def __init__(self, cfg: MetaLensConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mlp = SirenMLP(
            input_dim=cfg.coord_dim,
            output_dim=cfg.w_dim,
            hidden_dims=tuple(cfg.hidden_dims_param),
            first_omega_0=30.0,
            hidden_omega_0=30.0,
            outermost_linear=True,
        )

    def forward(self, coords: Tensor) -> Tensor:
        """
        coords: [..., N, coord_dim]
        returns: [..., N, w_dim]
        """
        assert coords.shape[-1] == self.cfg.coord_dim, (
            f"Expected coords last dim={self.cfg.coord_dim}, got {coords.shape[-1]}"
        )
        return self.mlp(coords)


class SurrogateNet(nn.Module):
    """
    Second network:
        (w, conditions) -> [amplitude, phase].

    - w: [..., N, w_dim]
    - cond: either [..., cond_dim] or [..., N, cond_dim]
      (broadcasting handled automatically)

    Output:
      - amp_phase: [..., N, 2]  ([..., N, 0] = amplitude, [..., N, 1] = phase)
    """
    def __init__(self, cfg: MetaLensConfig) -> None:
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.w_dim + cfg.cond_dim
        self.mlp = SirenMLP(
            input_dim=input_dim,
            output_dim=2,  # amplitude & phase
            hidden_dims=tuple(cfg.hidden_dims_surrogate),
            first_omega_0=30.0,
            hidden_omega_0=30.0,
            outermost_linear=True,
        )

    def _broadcast_cond(self, w: Tensor, cond: Tensor) -> Tensor:
        """
        Make cond shape compatible with w for concatenation along the last dim.

        - w:   [..., N, w_dim]
        - cond: either [..., cond_dim] or [..., N, cond_dim]
        returns:
        - cond_broadcast: [..., N, cond_dim]
        """
        if cond.shape[-1] != self.cfg.cond_dim:
            raise ValueError(
                f"cond last dim must be {self.cfg.cond_dim}, got {cond.shape[-1]}"
            )

        # Case 1: cond already has point dimension N
        if cond.dim() == w.dim() and cond.shape[-2] == w.shape[-2]:
            return cond

        # Case 2: cond missing point dimension; treat as per-sample condition
        if cond.dim() == w.dim() - 1:
            # cond: [..., cond_dim] -> [..., 1, cond_dim] -> broadcast to [..., N, cond_dim]
            cond = cond.unsqueeze(-2)  # insert N dimension
            # Use expand so it doesn't copy data
            expand_shape = (*w.shape[:-1], self.cfg.cond_dim)
            cond = cond.expand(expand_shape)
            return cond

        raise ValueError(
            f"Incompatible cond shape {cond.shape} for w shape {w.shape}. "
            "Expected [..., cond_dim] or [..., N, cond_dim]."
        )

    def forward(self, w: Tensor, cond: Tensor) -> Tensor:
        """
        w:    [..., N, w_dim]
        cond: [..., cond_dim] or [..., N, cond_dim]

        returns:
        - amp_phase: [..., N, 2]
        """
        assert w.shape[-1] == self.cfg.w_dim, (
            f"Expected w last dim={self.cfg.w_dim}, got {w.shape[-1]}"
        )

        cond_b = self._broadcast_cond(w, cond)
        x = torch.cat([w, cond_b], dim=-1)  # [..., N, w_dim + cond_dim]
        return self.mlp(x)


class MetaLensModel(nn.Module):
    """
    Full chained model:
        coords -> w -> (w, cond) -> amplitude & phase.

    This keeps everything shape-agnostic over leading batch dims, as long as
    the last dimensions match the config.
    """
    def __init__(self, cfg: MetaLensConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.param_net = ParamNet(cfg)
        self.surrogate_net = SurrogateNet(cfg)

    def forward(self, coords: Tensor, cond: Tensor) -> dict[str, Tensor]:
        """
        coords: [..., N, coord_dim]
        cond:   [..., cond_dim] or [..., N, cond_dim]

        returns dict with:
            - "w":          [..., N, w_dim]
            - "amp_phase":  [..., N, 2]
            - "amplitude":  [..., N]
            - "phase":      [..., N]
        """
        w = self.param_net(coords)
        amp_phase = self.surrogate_net(w, cond)
        amplitude = amp_phase[..., 0]
        phase = amp_phase[..., 1]
        return {
            "w": w,
            "amp_phase": amp_phase,
            "amplitude": amplitude,
            "phase": phase,
        }


# -----------------------------
# Example usage
# -----------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = MetaLensConfig(
        coord_dim=2,
        w_dim=4,
        cond_dim=4,              # e.g. [theta, phi, pol, wavelength]
        hidden_dims_param=(64, 64, 64),
        hidden_dims_surrogate=(64, 64, 64),
    )

    model = MetaLensModel(cfg).to(device)

    # Example: batch of metalenses with arbitrary leading dims
    # Shape: [B, H, W, 2] where N = H * W points per lens
    B, H, W = 3, 32, 32
    coords = torch.rand(B, H * W, 2, device=device) * 2.0 - 1.0  # normalized grid

    # Conditions per *metalens* (broadcast to all points):
    # Shape: [B, cond_dim]
    cond = torch.tensor(
        [
            [0.0, 0.0, 0.0, 550.0],  # e.g. angle_x, angle_y, pol, wavelength
            [0.1, 0.0, 1.0, 550.0],
            [0.0, 0.1, 0.5, 600.0],
        ],
        device=device,
    )

    outputs = model(coords, cond)
    print("w shape:         ", outputs["w"].shape)          # [B, H*W, w_dim]
    print("amp_phase shape: ", outputs["amp_phase"].shape)  # [B, H*W, 2]
    print("amplitude shape: ", outputs["amplitude"].shape)  # [B, H*W]
    print("phase shape:     ", outputs["phase"].shape)      # [B, H*W]
