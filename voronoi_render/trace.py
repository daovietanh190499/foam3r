"""Voronoi cell walk + Beer-Lambert accumulation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .sh import eval_sh_rgb, sh_basis


def nearest_site_indices(
    origins: torch.Tensor,
    sites: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(sites.detach().float().cpu().numpy())
        q = origins.detach().float().cpu().numpy()
        _, idx = tree.query(q, k=1, workers=-1)
        return torch.from_numpy(idx.astype("int64")).to(origins.device)
    except Exception:
        return _nearest_site_indices_chunked(origins, sites, chunk_size)


class SiteIndexLookup:
    """Cached cKDTree for repeated nearest-site queries (CUDA eval)."""

    def __init__(self, sites: torch.Tensor):
        from scipy.spatial import cKDTree

        self._sites = sites.detach().float().cpu()
        self._tree = cKDTree(self._sites.numpy())

    def query(self, origins: torch.Tensor) -> torch.Tensor:
        q = origins.detach().float().cpu().numpy()
        _, idx = self._tree.query(q, k=1, workers=-1)
        return torch.from_numpy(idx.astype("int64"))


def _nearest_site_indices_chunked(
    origins: torch.Tensor,
    sites: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    n_rays = origins.shape[0]
    best_d = origins.new_full((n_rays,), float("inf"))
    best_i = torch.zeros(n_rays, dtype=torch.long, device=origins.device)
    for start in range(0, n_rays, chunk_size):
        end = min(start + chunk_size, n_rays)
        d = torch.cdist(origins[start:end], sites)
        local = d.min(dim=1)
        better = local.values < best_d[start:end]
        best_d[start:end] = torch.where(better, local.values, best_d[start:end])
        best_i[start:end] = torch.where(better, local.indices, best_i[start:end])
    return best_i


def trace_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    sites: torch.Tensor,
    sh_attrs: torch.Tensor,
    densities: torch.Tensor,
    nbr_idx: torch.Tensor,
    nbr_diff: torch.Tensor,
    nbr_valid: torch.Tensor,
    start_cells: torch.Tensor,
    *,
    sh_degree: int = 3,
    max_steps: int = 128,
    weight_threshold: float = 0.001,
) -> torch.Tensor:
    n_rays = origins.shape[0]
    device, dtype = origins.device, origins.dtype
    dirs = F.normalize(directions, dim=-1)
    sh_coeffs = sh_basis(sh_degree, dirs)

    t0 = torch.zeros(n_rays, device=device, dtype=dtype)
    transmittance = torch.ones(n_rays, device=device, dtype=dtype)
    rgb = torch.zeros(n_rays, 3, device=device, dtype=dtype)
    alive = torch.ones(n_rays, dtype=torch.bool, device=device)
    current = start_cells.long()

    if sites.shape[0] == 0:
        return torch.cat([rgb, torch.zeros(n_rays, 1, device=device, dtype=dtype)], dim=-1)

    for _ in range(max_steps):
        if not alive.any():
            break

        idx = current.clamp(min=0)
        diffs = nbr_diff[idx]
        nbr_ids = nbr_idx[idx]
        valid = nbr_valid[idx]
        primal = sites[idx]

        face_origins = primal.unsqueeze(1) + diffs * 0.5
        face_normals = diffs
        dp = (face_normals * dirs.unsqueeze(1)).sum(dim=-1)
        num = ((face_origins - origins.unsqueeze(1)) * face_normals).sum(dim=-1)
        t_hit = num / dp.clamp(min=1e-8)

        face_ok = valid & (dp > 0.0)
        t_hit = torch.where(face_ok, t_hit, torch.full_like(t_hit, float("inf")))

        t1, face_sel = t_hit.min(dim=1)
        next_cell = nbr_ids.gather(1, face_sel.unsqueeze(1)).squeeze(1)

        no_face = ~face_ok.any(dim=1)
        stop = no_face | ~alive
        advance = alive & ~stop
        contrib_seg = advance & (t1 > t0)

        site_rgb = eval_sh_rgb(sh_coeffs, sh_attrs[idx])
        delta_t = (t1 - t0).clamp(min=0.0)
        sigma = densities[idx].squeeze(-1)
        alpha = 1.0 - torch.exp(-sigma * delta_t)
        weight = transmittance * alpha

        rgb = rgb + weight.unsqueeze(-1) * site_rgb * contrib_seg.unsqueeze(-1).float()
        transmittance = torch.where(
            contrib_seg,
            transmittance * (1.0 - alpha),
            transmittance,
        )

        t0 = torch.where(advance, torch.maximum(t0, t1), t0)
        current = torch.where(advance, next_cell, current)
        alive_after = transmittance > weight_threshold
        alive = torch.where(contrib_seg, alive_after, advance)

    opacity = (1.0 - transmittance).clamp(0.0, 1.0)
    return torch.cat([rgb, opacity.unsqueeze(-1)], dim=-1)
