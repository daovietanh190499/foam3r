#!/usr/bin/env python3
"""CUDA Voronoi render with CPU cKDTree nearest-site lookup (legacy pipeline)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from myresearch.paths import CUDA_BUILD, RENDER_CUDA_KDTREE, SCENE_VORT
from myresearch.voronoi_render.eval import (
    load_colmap_split,
    save_triptych,
)
from myresearch.voronoi_render.trace import SiteIndexLookup


def render_view_cuda(
    cuda_scene,
    site_lookup: SiteIndexLookup,
    rays_hw: torch.Tensor,
    *,
    white_background: bool,
    stride: int,
    ray_chunk: int,
) -> tuple[torch.Tensor, dict]:
    from myresearch.voronoi_render.eval import rgba_to_display_rgb

    h, w, _ = rays_hw.shape
    rays = rays_hw[::stride, ::stride].reshape(-1, 6)
    rh, rw = math.ceil(h / stride), math.ceil(w / stride)
    rays_np = rays.numpy().astype(np.float32)
    origins = torch.from_numpy(rays_np[:, :3])

    t0 = time.perf_counter()
    start_cells = site_lookup.query(origins).numpy().astype(np.int32)
    nn_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    parts = []
    for s in range(0, rays_np.shape[0], ray_chunk):
        e = min(s + ray_chunk, rays_np.shape[0])
        parts.append(cuda_scene.trace(rays_np[s:e], start_cells[s:e]))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    trace_s = time.perf_counter() - t0

    rgba = torch.from_numpy(np.concatenate(parts, axis=0)).reshape(rh, rw, 4)
    rgb = rgba_to_display_rgb(rgba, h, w, white_background=white_background, stride=stride)
    timing = {
        "nn_cpu_s": nn_s,
        "cuda_trace_s": trace_s,
        "rays": int(rays_np.shape[0]),
        "rays_per_sec_trace": rays_np.shape[0] / trace_s if trace_s > 0 else 0.0,
    }
    return rgb, timing


def main():
    p = argparse.ArgumentParser(description="CUDA render + CPU cKDTree NN lookup")
    p.add_argument("--model", type=Path, default=Path("model.pt"))
    p.add_argument("--scene", type=Path, default=SCENE_VORT)
    p.add_argument("--data-dir", type=Path, default=Path("data/mipnerf360/counter"))
    p.add_argument("--out", type=Path, default=RENDER_CUDA_KDTREE)
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

    print(f"Loading xyz from {args.model} (CPU cKDTree)")
    t0 = time.perf_counter()
    xyz = torch.load(args.model, map_location="cpu", weights_only=False)["xyz"].float()
    site_lookup = SiteIndexLookup(xyz)
    load_model_s = time.perf_counter() - t0
    print(f"  cKDTree ready in {load_model_s:.2f}s")

    print(f"Loading COLMAP test, downsample={args.downsample}")
    _, rays, rgbs = load_colmap_split(args.data_dir, "test", args.downsample, torch.device("cpu"))
    view_ids = [int(x) for x in args.indices.split(",") if x.strip()]

    metrics = []
    timings = []
    for vid in view_ids:
        print(f"CUDA (cKDTree) render view {vid:03d} ...")
        pred, view_timing = render_view_cuda(
            cuda_scene,
            site_lookup,
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
        timings.append({"view": vid, **view_timing})
        print(
            f"  PSNR: {img_psnr:.3f} dB | "
            f"NN {view_timing['nn_cpu_s']:.3f}s trace {view_timing['cuda_trace_s']:.3f}s"
        )

    nn_total = sum(t["nn_cpu_s"] for t in timings)
    trace_total = sum(t["cuda_trace_s"] for t in timings)
    n_views = len(view_ids)
    summary = {
        "backend": "cuda",
        "nn_lookup": "cpu_ckdtree",
        "model": str(args.model),
        "scene": str(args.scene),
        "stride": args.stride,
        "downsample": args.downsample,
        "load_model_ckdtree_s": load_model_s,
        "views": metrics,
        "timing": timings,
        "totals": {
            "nn_cpu_s": nn_total,
            "cuda_trace_s": trace_total,
            "render_s": nn_total + trace_total,
            "avg_per_view_s": (nn_total + trace_total) / max(n_views, 1),
            "avg_nn_per_view_s": nn_total / max(n_views, 1),
            "avg_trace_per_view_s": trace_total / max(n_views, 1),
        },
        "avg_psnr": sum(m["psnr"] for m in metrics) / max(len(metrics), 1),
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Average PSNR: {summary['avg_psnr']:.3f} dB")
    print(f"Render total: {summary['totals']['render_s']:.2f}s ({summary['totals']['avg_per_view_s']:.2f}s/view)")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
