"""Load RadFoam checkpoints and export binary scenes for CUDA."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .adjacency import pad_adjacency
from .sh import pack_sh_attributes, sh_degree_from_num_coeffs

VORT_MAGIC = 0x54524F56  # "VORT" little-endian
VORT_VERSION = 1
VORT_VERSION_CSR = 2


def _activate_density(raw: torch.Tensor, activation_scale: float = 1.0) -> torch.Tensor:
    if raw.ndim == 1:
        raw = raw.unsqueeze(-1)
    return activation_scale * F.softplus(raw, beta=10.0)


def load_model_pt(
    pt_path: str | Path,
    *,
    max_sites: int | None = None,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> dict:
    scene = torch.load(pt_path, map_location="cpu", weights_only=False)
    device = torch.device(device)

    position = scene["xyz"].to(device=device, dtype=torch.float32)
    density_raw = scene["density"].to(device=device, dtype=torch.float32)
    color_dc = scene["color_dc"].to(device=device, dtype=torch.float32)
    color_sh = scene["color_sh"].to(device=device, dtype=torch.float32)
    sh_degree = sh_degree_from_num_coeffs(color_sh.shape[1] // 3 + 1)

    adjacency = scene.get("adjacency")
    adjacency_offsets = scene.get("adjacency_offsets")
    if adjacency is not None:
        adjacency = adjacency.to(device=device, dtype=torch.long)
    if adjacency_offsets is not None:
        adjacency_offsets = adjacency_offsets.to(device=device, dtype=torch.long)

    if max_sites is not None and position.shape[0] > max_sites:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        idx = torch.randperm(position.shape[0], generator=g)[:max_sites].to(device).sort().values
        position = position[idx]
        density_raw = density_raw[idx]
        color_dc = color_dc[idx]
        color_sh = color_sh[idx]
        adjacency = None
        adjacency_offsets = None

    density = _activate_density(density_raw)
    sh_attrs = pack_sh_attributes(color_dc, color_sh)
    return {
        "position": position,
        "density": density,
        "color_dc": color_dc,
        "color_sh": color_sh,
        "sh_attrs": sh_attrs,
        "sh_degree": sh_degree,
        "adjacency": adjacency,
        "adjacency_offsets": adjacency_offsets,
    }


def load_sites_npz(npz_path: str | Path, *, device: str | torch.device = "cpu") -> dict:
    data = np.load(npz_path)
    device = torch.device(device)
    position = torch.from_numpy(data["position"]).to(device=device, dtype=torch.float32)
    density_raw = torch.from_numpy(data["density"]).to(device=device, dtype=torch.float32)
    color_dc = torch.from_numpy(data["color_dc"]).to(device=device, dtype=torch.float32)
    color_sh = torch.from_numpy(data["color_sh"]).to(device=device, dtype=torch.float32)
    sh_degree = int(data["sh_degree"]) if "sh_degree" in data else 3
    adjacency = None
    adjacency_offsets = None
    if "adjacency" in data.files:
        adjacency = torch.from_numpy(data["adjacency"]).to(device=device, dtype=torch.long)
        adjacency_offsets = torch.from_numpy(data["adjacency_offsets"]).to(
            device=device, dtype=torch.long
        )
    density = _activate_density(density_raw)
    sh_attrs = pack_sh_attributes(color_dc, color_sh)
    return {
        "position": position,
        "density": density,
        "color_dc": color_dc,
        "color_sh": color_sh,
        "sh_attrs": sh_attrs,
        "sh_degree": sh_degree,
        "adjacency": adjacency,
        "adjacency_offsets": adjacency_offsets,
    }


def export_scene_bin(
    scene: dict,
    out_path: str | Path,
    *,
    max_steps: int = 128,
    weight_threshold: float = 0.001,
) -> Path:
    """
    Export padded Voronoi scene to a flat binary file for the CUDA renderer.

    Layout: header + positions + sh_attrs + density + nbr_idx + nbr_diff + nbr_valid
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    position = scene["position"].detach().float().cpu()
    sh_attrs = scene["sh_attrs"].detach().float().cpu()
    density = scene["density"].detach().float().cpu().reshape(-1)
    adjacency = scene["adjacency"]
    offsets = scene["adjacency_offsets"]
    if adjacency is None or offsets is None:
        raise ValueError("Scene must include CSR adjacency for CUDA export")

    nbr_idx, nbr_diff, nbr_valid = pad_adjacency(position, adjacency, offsets)
    n, max_deg = nbr_idx.shape
    sh_dim = sh_attrs.shape[1]
    sh_degree = int(scene["sh_degree"])

    header = struct.pack(
        "<IIIIIffI",
        VORT_MAGIC,
        VORT_VERSION,
        n,
        max_deg,
        sh_degree,
        weight_threshold,
        float(max_steps),
        sh_dim,
    )

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(position.numpy().astype(np.float32).tobytes())
        f.write(sh_attrs.numpy().astype(np.float32).tobytes())
        f.write(density.numpy().astype(np.float32).tobytes())
        f.write(nbr_idx.numpy().astype(np.int32).tobytes())
        f.write(nbr_diff.numpy().astype(np.float32).tobytes())
        f.write(nbr_valid.numpy().astype(np.uint8).tobytes())

    return out_path


def export_scene_bin_csr(
    scene: dict,
    out_path: str | Path,
    *,
    max_steps: int = 128,
    weight_threshold: float = 0.001,
) -> Path:
    """
    Export Voronoi scene with sparse CSR adjacency (VORT v2, RadFoam-style).

    Layout: header + positions + sh_attrs + density + adjacency + adjacency_offsets
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    position = scene["position"].detach().float().cpu()
    sh_attrs = scene["sh_attrs"].detach().float().cpu()
    density = scene["density"].detach().float().cpu().reshape(-1)
    adjacency = scene["adjacency"]
    offsets = scene["adjacency_offsets"]
    if adjacency is None or offsets is None:
        raise ValueError("Scene must include CSR adjacency for CUDA export")

    adjacency = adjacency.detach().cpu().numpy().astype(np.uint32)
    offsets = offsets.detach().cpu().numpy().astype(np.uint32)
    n = position.shape[0]
    adj_size = int(adjacency.shape[0])
    sh_dim = sh_attrs.shape[1]
    sh_degree = int(scene["sh_degree"])

    header = struct.pack(
        "<IIIIIffI",
        VORT_MAGIC,
        VORT_VERSION_CSR,
        n,
        adj_size,
        sh_degree,
        weight_threshold,
        float(max_steps),
        sh_dim,
    )

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(position.numpy().astype(np.float32).tobytes())
        f.write(sh_attrs.numpy().astype(np.float32).tobytes())
        f.write(density.numpy().astype(np.float32).tobytes())
        f.write(adjacency.tobytes())
        f.write(offsets.tobytes())

    return out_path
