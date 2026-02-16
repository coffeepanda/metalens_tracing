# param_field.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ParameterField(nn.Module, ABC):
    """
    Abstract interface: map 2D surface coords -> structure parameters w.
    coords_local_xy: [M, 2] in meters.
    returns:         [M, w_dim]
    """
    @abstractmethod
    def forward(self, coords_local_xy: Tensor) -> Tensor:
        raise NotImplementedError


class NeuralParameterField(ParameterField):
    """
    Parameter field represented by a neural network.

    coords_local_xy: [M, 2] -> w: [M, w_dim]
    """
    def __init__(self, rep_net: nn.Module):
        super().__init__()
        self.rep_net = rep_net

    def forward(self, coords_local_xy: Tensor) -> Tensor:
        return self.rep_net(coords_local_xy)


class GridParameterField(ParameterField):
    """
    Parameter field represented by a learnable grid w(x_i, y_j).

    - w_grid: [1, w_dim, H, W] is the *physical* design parameter.
    - coords_local_xy: [M, 2] in meters.

    We map coords to [-1, 1] range and sample with bilinear interpolation
    via grid_sample.
    """
    def __init__(
        self,
        xy_min: Tuple[float, float],
        xy_max: Tuple[float, float],
        grid_shape: Tuple[int, int],
        w_dim: int,
        init_scale: float = 1e-2,
    ) -> None:
        super().__init__()
        self.xy_min = xy_min
        self.xy_max = xy_max
        self.H, self.W = grid_shape
        self.w_dim = w_dim

        # [1, w_dim, H, W]
        w_grid = init_scale * torch.randn(1, w_dim, self.H, self.W)
        self.w_grid = nn.Parameter(w_grid)

    def forward(self, coords_local_xy: Tensor) -> Tensor:
        """
        coords_local_xy: [M, 2] in meters.
        returns: [M, w_dim]
        """
        device = self.w_grid.device
        dtype = self.w_grid.dtype
        coords = coords_local_xy.to(device=device, dtype=dtype)

        # Map from [xy_min, xy_max] to normalized [-1, 1] for grid_sample.
        x_min, y_min = self.xy_min
        x_max, y_max = self.xy_max

        # Avoid division by zero if min==max
        scale_x = (x_max - x_min) if (x_max - x_min) != 0 else 1.0
        scale_y = (y_max - y_min) if (y_max - y_min) != 0 else 1.0

        # (x, y) -> (u, v) in [-1, 1]
        u = 2.0 * (coords[..., 0] - x_min) / scale_x - 1.0
        v = 2.0 * (coords[..., 1] - y_min) / scale_y - 1.0

        # grid_sample expects grid as [N, H_out, W_out, 2]
        # We'll treat M points as a 1D "image" of size H_out=M, W_out=1
        M = coords.shape[0]
        grid = torch.stack([u, v], dim=-1).view(1, M, 1, 2)  # [1, M, 1, 2]

        # w_grid: [1, w_dim, H, W] -> sample -> [1, w_dim, M, 1]
        sampled = F.grid_sample(
            self.w_grid,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [1, w_dim, M, 1]

        # -> [M, w_dim]
        sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)  # [M, w_dim]
        return sampled




class TwoParamRepNet(nn.Module):
    """
    Very simple 2-parameter representation:

        w(x, y) = alpha * (r^2 / r0^2) + beta,  where r^2 = x^2 + y^2

    - alpha: controls radial variation strength.
    - beta:  constant offset.

    Output: [M, w_dim], but w_dim=1 for this validation.
    """

    def __init__(self, w_dim: int = 1, r0: float = 0.5e-3) -> None:
        super().__init__()
        self.w_dim = w_dim
        self.r0 = r0

        # Two scalar parameters = "metalens parameters"
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, coords_local_xy: Tensor) -> Tensor:
        """
        coords_local_xy: [M, 2], in meters.
        returns:         [M, w_dim]
        """
        device = self.alpha.device
        dtype = self.alpha.dtype
        coords = coords_local_xy.to(device=device, dtype=dtype)

        x = coords[..., 0]
        y = coords[..., 1]
        r2 = x * x + y * y  # [M]

        # Normalize by r0^2 to keep values O(1)
        r0_sq = self.r0 * self.r0 + 1e-24
        r2_norm = r2 / r0_sq

        base = self.alpha * r2_norm + self.beta  # [M]

        w = base.unsqueeze(-1)                   # [M, 1]
        if self.w_dim > 1:
            # Broadcast same scalar into w_dim channels if needed
            w = w.expand(-1, self.w_dim)
        return w