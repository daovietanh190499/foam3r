"""Spherical harmonics (degree <= 3), interleaved RGB layout."""

from __future__ import annotations

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)

SH_DIM = {0: 1, 1: 4, 2: 9, 3: 16}


def sh_degree_from_num_coeffs(num_coeffs: int) -> int:
    for deg, dim in SH_DIM.items():
        if dim == num_coeffs:
            return deg
    raise ValueError(f"Unsupported SH coefficient count: {num_coeffs}")


def sh_basis(degree: int, dirs: torch.Tensor) -> torch.Tensor:
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    basis = [C0 * torch.ones_like(x)]
    if degree >= 1:
        basis.extend([-C1 * y, C1 * z, -C1 * x])
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        basis.extend(
            [
                C2[0] * xy,
                C2[1] * yz,
                C2[2] * (2.0 * zz - xx - yy),
                C2[3] * xz,
                C2[4] * (xx - yy),
            ]
        )
    if degree >= 3:
        basis.extend(
            [
                C3[0] * y * (3.0 * xx - yy),
                C3[1] * xy * z,
                C3[2] * y * (4.0 * zz - xx - yy),
                C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
                C3[4] * x * (4.0 * zz - xx - yy),
                C3[5] * z * (xx - yy),
                C3[6] * x * (xx - 3.0 * yy),
            ]
        )
    return torch.stack(basis, dim=-1)


def pack_sh_attributes(color_dc: torch.Tensor, color_sh: torch.Tensor) -> torch.Tensor:
    return torch.cat([color_dc, color_sh], dim=-1)


def eval_sh_rgb(sh_coeffs: torch.Tensor, sh_rgb_vals: torch.Tensor) -> torch.Tensor:
    k = sh_coeffs.shape[-1]
    vals = sh_rgb_vals.reshape(*sh_rgb_vals.shape[:-1], k, 3)
    rgb = 0.5 + (sh_coeffs.unsqueeze(-1) * vals).sum(dim=-2)
    return rgb.clamp(min=0.0)
