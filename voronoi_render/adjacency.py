"""CSR Delaunay adjacency -> padded neighbor tables for ray walking."""

from __future__ import annotations

import torch


def pad_adjacency(
    points: torch.Tensor,
    adjacency: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = points.shape[0]
    if n == 0:
        empty = points.new_zeros(0, 0, dtype=torch.long)
        return empty, points.new_zeros(0, 0, 3), empty.bool()

    degrees = offsets[1:] - offsets[:-1]
    max_deg = int(degrees.max().item()) if degrees.numel() else 0
    if max_deg == 0:
        return (
            torch.full((n, 1), -1, dtype=torch.long, device=points.device),
            torch.zeros(n, 1, 3, device=points.device, dtype=points.dtype),
            torch.zeros(n, 1, dtype=torch.bool, device=points.device),
        )

    nbr_idx = torch.full((n, max_deg), -1, dtype=torch.long, device=points.device)
    nbr_valid = torch.zeros(n, max_deg, dtype=torch.bool, device=points.device)

    adj_cpu = adjacency.detach().cpu()
    off_cpu = offsets.detach().cpu()
    for i in range(n):
        start = int(off_cpu[i].item())
        end = int(off_cpu[i + 1].item())
        deg = end - start
        if deg == 0:
            continue
        nbr_idx[i, :deg] = adj_cpu[start:end]
        nbr_valid[i, :deg] = True

    safe_idx = nbr_idx.clamp(min=0)
    nbr_diff = points[safe_idx] - points.unsqueeze(1)
    nbr_diff = nbr_diff * nbr_valid.unsqueeze(-1)
    return nbr_idx, nbr_diff, nbr_valid
