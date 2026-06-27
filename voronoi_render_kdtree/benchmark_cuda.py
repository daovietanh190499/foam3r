#!/usr/bin/env python3
"""Benchmark CUDA render with CPU cKDTree NN lookup (legacy pipeline)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from myresearch.paths import CUDA_BUILD, RENDER_CUDA_KDTREE, SCENE_VORT
from myresearch.voronoi_render.benchmark_cuda import build_breakdown
from myresearch.voronoi_render.eval import composite_white_background, load_colmap_split
from myresearch.voronoi_render.trace import SiteIndexLookup


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("model.pt"))
    p.add_argument("--scene", type=Path, default=SCENE_VORT)
    p.add_argument("--data-dir", type=Path, default=Path("data/mipnerf360/counter"))
    p.add_argument("--out", type=Path, default=RENDER_CUDA_KDTREE / "timing.json")
    p.add_argument("--indices", type=str, default="0,1,2,3,4")
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--ray-chunk", type=int, default=65536)
    p.add_argument("--cuda-build", type=Path, default=CUDA_BUILD)
    args = p.parse_args()

    sys.path.insert(0, str(args.cuda_build))
    import voronoi_render_cuda  # noqa: E402

    view_ids = [int(x) for x in args.indices.split(",") if x.strip()]

    t0 = time.perf_counter()
    cuda_scene = voronoi_render_cuda.CudaScene(str(args.scene))
    sync()
    load_scene_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    xyz = torch.load(args.model, map_location="cpu", weights_only=False)["xyz"].float()
    site_lookup = SiteIndexLookup(xyz)
    load_model_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, rays, _ = load_colmap_split(args.data_dir, "test", args.downsample, torch.device("cpu"))
    load_colmap_s = time.perf_counter() - t0

    h, w, _ = rays[view_ids[0]].shape
    rh, rw = math.ceil(h / args.stride), math.ceil(w / args.stride)
    rays_per_view = rh * rw

    view_stats = []
    nn_total = trace_total = post_total = render_total = 0.0

    for vid in view_ids:
        rays_hw = rays[vid]
        rays_flat = rays_hw[:: args.stride, :: args.stride].reshape(-1, 6)
        rays_np = rays_flat.numpy().astype(np.float32)
        origins = torch.from_numpy(rays_np[:, :3])

        t_view = time.perf_counter()

        t0 = time.perf_counter()
        start = site_lookup.query(origins).numpy().astype(np.int32)
        nn_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        parts = []
        n_chunks = 0
        for s in range(0, rays_np.shape[0], args.ray_chunk):
            e = min(s + args.ray_chunk, rays_np.shape[0])
            parts.append(cuda_scene.trace(rays_np[s:e], start[s:e]))
            n_chunks += 1
        sync()
        trace_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        rgba = torch.from_numpy(np.concatenate(parts, axis=0)).reshape(rh, rw, 4)
        rgb = composite_white_background(rgba)
        if args.stride > 1:
            rgb = torch.nn.functional.interpolate(
                rgb.permute(2, 0, 1).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
        _ = rgb.numpy()
        post_s = time.perf_counter() - t0

        total_s = time.perf_counter() - t_view
        nn_total += nn_s
        trace_total += trace_s
        post_total += post_s
        render_total += total_s

        view_stats.append(
            {
                "view": vid,
                "nn_cpu_s": nn_s,
                "cuda_trace_s": trace_s,
                "postprocess_s": post_s,
                "total_view_s": total_s,
                "trace_chunks": n_chunks,
                "rays_traced": int(rays_np.shape[0]),
                "rays_per_sec_trace": rays_np.shape[0] / trace_s if trace_s > 0 else 0,
            }
        )

    n_views = len(view_ids)
    stats = {
        "backend": "cuda",
        "nn_lookup": "cpu_ckdtree",
        "config": {
            "stride": args.stride,
            "downsample": args.downsample,
            "ray_chunk": args.ray_chunk,
            "views": view_ids,
            "resolution_hw": [h, w],
            "rays_per_view_stride": rays_per_view,
        },
        "num_sites": cuda_scene.num_sites,
        "max_degree": cuda_scene.max_degree,
        "load_cuda_scene_s": load_scene_s,
        "load_model_ckdtree_s": load_model_s,
        "load_colmap_s": load_colmap_s,
        "views": view_stats,
        "totals": {
            "nn_cpu_s": nn_total,
            "cuda_trace_s": trace_total,
            "postprocess_s": post_total,
            "render_5_views_s": render_total,
            "avg_per_view_s": render_total / n_views,
            "avg_nn_per_view_s": nn_total / n_views,
            "avg_trace_per_view_s": trace_total / n_views,
        },
        "grand_total_s": load_scene_s + load_model_s + load_colmap_s + render_total,
        "breakdown": {
            **build_breakdown(
                load_scene_s=load_scene_s + load_model_s,
                load_colmap_s=load_colmap_s,
                views=view_stats,
                nn_key="nn_cpu_s",
                integrated_key=None,
            ),
            "init_one_time": {
                "load_scene_s": round(load_scene_s, 3),
                "load_model_ckdtree_s": round(load_model_s, 3),
                "load_colmap_s": round(load_colmap_s, 3),
                "total_s": round(load_scene_s + load_model_s + load_colmap_s, 3),
            },
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
