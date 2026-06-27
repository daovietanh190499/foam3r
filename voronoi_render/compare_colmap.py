#!/usr/bin/env python3
"""Compare Python vs CUDA trace on real COLMAP rays (per-view PSNR parity)."""

from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from myresearch.paths import CUDA_BUILD, SCENE_VORT
from myresearch.voronoi_render.eval import load_colmap_split, psnr, rgba_to_display_rgb
from myresearch.voronoi_render.scene import VoronoiScene
from myresearch.voronoi_render.trace import nearest_site_indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("model.pt"))
    p.add_argument("--scene", type=Path, default=SCENE_VORT)
    p.add_argument("--data-dir", type=Path, default=Path("data/mipnerf360/counter"))
    p.add_argument("--view", type=int, default=1)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--cuda-build", type=Path, default=CUDA_BUILD)
    args = p.parse_args()

    sys.path.insert(0, str(args.cuda_build))
    import voronoi_render_cuda  # noqa: E402

    device = torch.device("cuda")
    _, rays, rgbs = load_colmap_split(args.data_dir, "test", 2, torch.device("cpu"))
    rays_hw = rays[args.view]
    h, w, _ = rays_hw.shape
    rh, rw = math.ceil(h / args.stride), math.ceil(w / args.stride)
    rays_s = rays_hw[:: args.stride, :: args.stride].reshape(-1, 6).float().to(device)

    scene = VoronoiScene(device=device)
    scene.load_model_pt(args.model)
    start = nearest_site_indices(rays_s[:, :3], scene.sites)
    py_rgba = scene.trace(rays_s, start).reshape(rh, rw, 4).cpu()
    py_rgb = rgba_to_display_rgb(py_rgba, h, w, white_background=True, stride=args.stride)
    py_psnr = psnr(py_rgb, rgbs[args.view])

    rays_np = rays_s.cpu().numpy().astype(np.float32)
    start_np = start.cpu().numpy().astype(np.int32)
    del scene, rays_s, start
    gc.collect()
    torch.cuda.empty_cache()

    cuda_scene = voronoi_render_cuda.CudaScene(str(args.scene))
    parts = []
    for s in range(0, rays_np.shape[0], 65536):
        e = min(s + 65536, rays_np.shape[0])
        parts.append(cuda_scene.trace(rays_np[s:e], start_np[s:e]))
    cu_rgba = torch.from_numpy(np.concatenate(parts, 0)).reshape(rh, rw, 4)
    cu_rgb = rgba_to_display_rgb(cu_rgba, h, w, white_background=True, stride=args.stride)
    cu_psnr = psnr(cu_rgb, rgbs[args.view])

    diff = (py_rgba.reshape(-1, 4).numpy() - cu_rgba.reshape(-1, 4).numpy())
    print(f"view {args.view} | py PSNR {py_psnr:.4f} | cuda PSNR {cu_psnr:.4f} | delta {cu_psnr - py_psnr:+.4f} dB")
    print(f"  rgba max diff {np.abs(diff).max():.6e} | mean {np.abs(diff).mean():.6e}")
    print(f"  rays with diff > 1e-3: {(np.abs(diff).reshape(-1,4).max(1) > 1e-3).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
