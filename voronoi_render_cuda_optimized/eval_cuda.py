#!/usr/bin/env python3
"""Render test views with optimized ``voronoi_cuda`` (fused NN+trace)."""

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

from myresearch.paths import RENDER_VORONOI, SCENE_VORT, VORONOI_BUILD
from myresearch.voronoi_render.eval import load_colmap_split, save_triptych, rgba_to_display_rgb


def render_view(cuda_scene, rays_hw, *, white_background: bool, stride: int, ray_chunk: int) -> torch.Tensor:
    h, w, _ = rays_hw.shape
    rays = rays_hw[::stride, ::stride].reshape(-1, 6)
    rh, rw = math.ceil(h / stride), math.ceil(w / stride)
    rays_np = rays.numpy().astype(np.float32)

    parts = []
    for s in range(0, rays_np.shape[0], ray_chunk):
        e = min(s + ray_chunk, rays_np.shape[0])
        parts.append(cuda_scene.trace(rays_np[s:e]))
    rgba = torch.from_numpy(np.concatenate(parts, axis=0)).reshape(rh, rw, 4)
    return rgba_to_display_rgb(rgba, h, w, white_background=white_background, stride=stride)


def main():
    p = argparse.ArgumentParser(description="Optimized CUDA Voronoi render")
    p.add_argument("--scene", type=Path, default=SCENE_VORT)
    p.add_argument("--data-dir", type=Path, default=Path("data/mipnerf360/counter"))
    p.add_argument("--out", type=Path, default=RENDER_VORONOI)
    p.add_argument("--indices", type=str, default="0,1,2,3,4")
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--ray-chunk", type=int, default=65536)
    p.add_argument("--cuda-build", type=Path, default=VORONOI_BUILD)
    args = p.parse_args()

    sys.path.insert(0, str(args.cuda_build))
    import voronoi_cuda  # noqa: E402

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Loading scene: {args.scene}")
    scene = voronoi_cuda.VoronoiScene(str(args.scene))
    print(f"  sites={scene.num_sites} max_degree={scene.max_degree}")

    _, rays, rgbs = load_colmap_split(args.data_dir, "test", args.downsample, torch.device("cpu"))
    view_ids = [int(x) for x in args.indices.split(",") if x.strip()]

    metrics = []
    for vid in view_ids:
        print(f"Render view {vid:03d} ...")
        pred = render_view(scene, rays[vid], white_background=True, stride=args.stride, ray_chunk=args.ray_chunk)
        img_psnr = save_triptych(args.out / f"view_{vid:03d}_pred_gt_err.png", pred, rgbs[vid])
        metrics.append({"view": vid, "psnr": img_psnr})
        print(f"  PSNR: {img_psnr:.3f} dB")

    summary = {
        "backend": "voronoi_cuda",
        "nn_lookup": "gpu_fused",
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
