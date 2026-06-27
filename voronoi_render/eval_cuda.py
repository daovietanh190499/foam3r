#!/usr/bin/env python3
"""Render test views with CUDA Voronoi tracer."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from myresearch.paths import CUDA_BUILD, RENDER_CUDA, SCENE_VORT
from myresearch.voronoi_render.eval import (
    load_colmap_split,
    psnr,
    rgba_to_display_rgb,
    save_triptych,
)


def render_view_cuda(
    cuda_scene,
    rays_hw: torch.Tensor,
    *,
    white_background: bool,
    stride: int,
    ray_chunk: int,
) -> torch.Tensor:
    h, w, _ = rays_hw.shape
    rays = rays_hw[::stride, ::stride].reshape(-1, 6)
    rh, rw = math.ceil(h / stride), math.ceil(w / stride)
    rays_np = rays.cpu().numpy().astype(np.float32)

    parts = []
    for s in range(0, rays_np.shape[0], ray_chunk):
        e = min(s + ray_chunk, rays_np.shape[0])
        # start_cells=None → GPU NN on scene.positions (no extra 2GB buffer)
        parts.append(cuda_scene.trace(rays_np[s:e]))
    rgba = torch.from_numpy(np.concatenate(parts, axis=0)).reshape(rh, rw, 4)
    return rgba_to_display_rgb(rgba, h, w, white_background=white_background, stride=stride)


def main():
    p = argparse.ArgumentParser(description="CUDA Voronoi render vs GT")
    p.add_argument("--scene", type=Path, default=SCENE_VORT)
    p.add_argument("--data-dir", type=Path, default=Path("data/mipnerf360/counter"))
    p.add_argument("--out", type=Path, default=RENDER_CUDA)
    p.add_argument("--indices", type=str, default="0,1,2,3,4")
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--ray-chunk", type=int, default=65536)
    p.add_argument("--cuda-build", type=Path, default=CUDA_BUILD)
    args = p.parse_args()

    sys.path.insert(0, str(args.cuda_build))
    import voronoi_render_cuda  # noqa: E402

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading CUDA scene: {args.scene}")
    cuda_scene = voronoi_render_cuda.CudaScene(str(args.scene))
    print(f"  sites={cuda_scene.num_sites} max_degree={cuda_scene.max_degree}")

    print(f"Loading COLMAP test, downsample={args.downsample}")
    ds, rays, rgbs = load_colmap_split(args.data_dir, "test", args.downsample, torch.device("cpu"))
    view_ids = [int(x) for x in args.indices.split(",") if x.strip()]

    metrics = []
    for vid in view_ids:
        print(f"CUDA render view {vid:03d} ...")
        pred = render_view_cuda(
            cuda_scene,
            rays[vid],
            white_background=True,
            stride=args.stride,
            ray_chunk=args.ray_chunk,
        )
        gt = rgbs[vid]
        img_psnr = save_triptych(args.out / f"view_{vid:03d}_pred_gt_err.png", pred, gt)
        pred_only = (pred.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
        Image.fromarray(pred_only).save(args.out / f"view_{vid:03d}_pred.png")
        metrics.append({"view": vid, "psnr": img_psnr})
        print(f"  PSNR: {img_psnr:.3f} dB")

    summary = {
        "backend": "cuda",
        "nn_lookup": "gpu_spatial_grid",
        "scene": str(args.scene),
        "stride": args.stride,
        "downsample": args.downsample,
        "views": metrics,
        "avg_psnr": sum(m["psnr"] for m in metrics) / max(len(metrics), 1),
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Average PSNR: {summary['avg_psnr']:.3f} dB")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
