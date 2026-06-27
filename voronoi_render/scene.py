"""Voronoi scene container + ray trace entry point."""

from __future__ import annotations

from pathlib import Path

import torch

from .adjacency import pad_adjacency
from .io import load_model_pt, load_sites_npz
from .sh import pack_sh_attributes
from .trace import nearest_site_indices, trace_rays


class VoronoiScene:
    def __init__(
        self,
        *,
        max_intersections: int = 128,
        weight_threshold: float = 0.001,
        ray_chunk_size: int = 8192,
        nn_chunk_size: int = 4096,
        device: str | torch.device = "cuda",
    ):
        self.max_intersections = max_intersections
        self.weight_threshold = weight_threshold
        self.ray_chunk_size = ray_chunk_size
        self.nn_chunk_size = nn_chunk_size
        self.device = torch.device(device)
        self.sh_degree = 3
        self.sites: torch.Tensor | None = None
        self.sh_attrs: torch.Tensor | None = None
        self.densities: torch.Tensor | None = None
        self.nbr_idx: torch.Tensor | None = None
        self.nbr_diff: torch.Tensor | None = None
        self.nbr_valid: torch.Tensor | None = None

    def set_from_tensors(
        self,
        position: torch.Tensor,
        color_dc: torch.Tensor,
        color_sh: torch.Tensor,
        density: torch.Tensor,
        *,
        sh_degree: int = 3,
        adjacency: torch.Tensor | None = None,
        adjacency_offsets: torch.Tensor | None = None,
    ) -> None:
        if density.ndim == 1:
            density = density.unsqueeze(-1)
        if adjacency is None or adjacency_offsets is None:
            raise ValueError("Stored CSR adjacency is required")

        self.sites = position
        self.sh_attrs = pack_sh_attributes(color_dc, color_sh)
        self.densities = density
        self.sh_degree = sh_degree
        self.nbr_idx, self.nbr_diff, self.nbr_valid = pad_adjacency(
            position, adjacency, adjacency_offsets
        )

    def load_model_pt(self, path: str | Path, *, max_sites: int | None = None) -> None:
        data = load_model_pt(path, max_sites=max_sites, device=self.device)
        self.set_from_tensors(
            data["position"],
            data["color_dc"],
            data["color_sh"],
            data["density"],
            sh_degree=int(data["sh_degree"]),
            adjacency=data["adjacency"],
            adjacency_offsets=data["adjacency_offsets"],
        )

    def load_npz(self, path: str | Path) -> None:
        data = load_sites_npz(path, device=self.device)
        self.set_from_tensors(
            data["position"],
            data["color_dc"],
            data["color_sh"],
            data["density"],
            sh_degree=int(data["sh_degree"]),
            adjacency=data["adjacency"],
            adjacency_offsets=data["adjacency_offsets"],
        )

    def as_dict(self) -> dict:
        return {
            "position": self.sites,
            "sh_attrs": self.sh_attrs,
            "density": self.densities,
            "sh_degree": self.sh_degree,
            "adjacency": None,
            "adjacency_offsets": None,
        }

    def trace(self, rays: torch.Tensor, start_cells: torch.Tensor | None = None) -> torch.Tensor:
        assert self.sites is not None
        origins = rays[:, :3]
        directions = rays[:, 3:]
        parts = []
        for start in range(0, rays.shape[0], self.ray_chunk_size):
            end = min(start + self.ray_chunk_size, rays.shape[0])
            o, d = origins[start:end], directions[start:end]
            if start_cells is None:
                sc = nearest_site_indices(o, self.sites, self.nn_chunk_size)
            else:
                sc = start_cells[start:end]
            parts.append(
                trace_rays(
                    o,
                    d,
                    self.sites,
                    self.sh_attrs,
                    self.densities,
                    self.nbr_idx,
                    self.nbr_diff,
                    self.nbr_valid,
                    sc,
                    sh_degree=self.sh_degree,
                    max_steps=self.max_intersections,
                    weight_threshold=self.weight_threshold,
                )
            )
        return torch.cat(parts, dim=0)
