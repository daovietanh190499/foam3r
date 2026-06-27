#!/usr/bin/env python3
"""Benchmark CUDA render pipeline timing (GPU spatial-grid NN)."""

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

from myresearch.paths import CUDA_BUILD, RENDER_CUDA, SCENE_VORT
from myresearch.voronoi_render.eval import composite_white_background, load_colmap_split


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _pct(part: float, total: float) -> float:
    return round(100.0 * part / total, 1) if total > 0 else 0.0


def _avg_views(views: list[dict], keys: list[str], *, skip_first: bool = False) -> dict[str, float]:
    subset = views[1:] if skip_first and len(views) > 1 else views
    n = max(len(subset), 1)
    return {k: sum(v[k] for v in subset) / n for k in keys}


def build_breakdown(
    *,
    load_scene_s: float,
    load_colmap_s: float,
    views: list[dict],
    nn_key: str = "nn_gpu_s",
    trace_key: str = "cuda_trace_s",
    post_key: str = "postprocess_s",
    integrated_key: str | None = "trace_auto_nn_s",
) -> dict:
    nn_t = sum(v[nn_key] for v in views)
    trace_t = sum(v[trace_key] for v in views)
    post_t = sum(v[post_key] for v in views)
    render_t = nn_t + trace_t + post_t

    steady = _avg_views(views, [nn_key, trace_key, post_key], skip_first=True)
    steady_total = steady[nn_key] + steady[trace_key] + steady[post_key]

    out: dict = {
        "init_one_time": {
            "load_scene_s": round(load_scene_s, 3),
            "load_colmap_s": round(load_colmap_s, 3),
        },
        "render_5_views_split": {
            "nn_s": round(nn_t, 4),
            "trace_s": round(trace_t, 4),
            "postprocess_s": round(post_t, 4),
            "total_s": round(render_t, 4),
            "avg_per_view_s": round(render_t / max(len(views), 1), 4),
            "share_pct": {
                "nn": _pct(nn_t, render_t),
                "trace": _pct(trace_t, render_t),
                "postprocess": _pct(post_t, render_t),
            },
        },
        "render_steady_state_avg_view_1_4": {
            "nn_s": round(steady[nn_key], 4),
            "trace_s": round(steady[trace_key], 4),
            "postprocess_s": round(steady[post_key], 4),
            "total_s": round(steady_total, 4),
            "share_pct": {
                "nn": _pct(steady[nn_key], steady_total),
                "trace": _pct(steady[trace_key], steady_total),
                "postprocess": _pct(steady[post_key], steady_total),
            },
        },
    }

    if integrated_key:
        auto_t = sum(v[integrated_key] for v in views)
        auto_post_t = auto_t + post_t
        steady_auto = _avg_views(views, [integrated_key, post_key], skip_first=True)
        steady_auto_total = steady_auto[integrated_key] + steady_auto[post_key]
        out["render_5_views_integrated"] = {
            "trace_auto_nn_s": round(auto_t, 4),
            "postprocess_s": round(post_t, 4),
            "total_s": round(auto_post_t, 4),
            "avg_per_view_s": round(auto_post_t / max(len(views), 1), 4),
            "share_pct": {
                "trace_with_nn": _pct(auto_t, auto_post_t),
                "postprocess": _pct(post_t, auto_post_t),
            },
        }
        out["integrated_steady_state_avg_view_1_4"] = {
            "trace_auto_nn_s": round(steady_auto[integrated_key], 4),
            "postprocess_s": round(steady_auto[post_key], 4),
            "total_s": round(steady_auto_total, 4),
            "share_pct": {
                "trace_with_nn": _pct(steady_auto[integrated_key], steady_auto_total),
                "postprocess": _pct(steady_auto[post_key], steady_auto_total),
            },
        }

    out["end_to_end"] = {
        "init_s": round(load_scene_s + load_colmap_s, 3),
        "render_only_s": round(render_t, 4),
        "grand_total_s": round(load_scene_s + load_colmap_s + render_t, 3),
    }
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, default=SCENE_VORT)
    p.add_argument("--data-dir", type=Path, default=Path("data/mipnerf360/counter"))
    p.add_argument("--out", type=Path, default=RENDER_CUDA / "timing.json")
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
        origins_np = rays_np[:, :3]

        t_view = time.perf_counter()

        t0 = time.perf_counter()
        start = cuda_scene.nearest_site_indices(origins_np)
        sync()
        nn_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        parts = []
        n_chunks = 0
        start_np = start if isinstance(start, np.ndarray) else np.asarray(start)
        for s in range(0, rays_np.shape[0], args.ray_chunk):
            e = min(s + args.ray_chunk, rays_np.shape[0])
            parts.append(cuda_scene.trace(rays_np[s:e], start_np[s:e].astype(np.int32)))
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

        # Integrated path (NN on GPU inside trace)
        t0 = time.perf_counter()
        parts_auto = []
        for s in range(0, rays_np.shape[0], args.ray_chunk):
            e = min(s + args.ray_chunk, rays_np.shape[0])
            parts_auto.append(cuda_scene.trace(rays_np[s:e]))
        sync()
        auto_s = time.perf_counter() - t0

        total_s = time.perf_counter() - t_view
        nn_total += nn_s
        trace_total += trace_s
        post_total += post_s
        render_total += total_s

        view_stats.append(
            {
                "view": vid,
                "nn_gpu_s": nn_s,
                "cuda_trace_s": trace_s,
                "trace_auto_nn_s": auto_s,
                "postprocess_s": post_s,
                "total_view_s": total_s,
                "trace_chunks": n_chunks,
                "rays_traced": int(rays_np.shape[0]),
                "rays_per_sec_trace": rays_np.shape[0] / trace_s if trace_s > 0 else 0,
                "rays_per_sec_auto": rays_np.shape[0] / auto_s if auto_s > 0 else 0,
            }
        )

    n_views = len(view_ids)
    stats = {
        "backend": "cuda",
        "nn_lookup": "gpu_spatial_grid",
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
        "load_colmap_s": load_colmap_s,
        "extra_vram_note": "spatial grid ~8-10 MB (grid_offsets + grid_indices)",
        "views": view_stats,
        "totals": {
            "nn_gpu_s": nn_total,
            "cuda_trace_s": trace_total,
            "trace_auto_nn_s": sum(v["trace_auto_nn_s"] for v in view_stats),
            "postprocess_s": post_total,
            "render_5_views_s": render_total,
            "avg_per_view_s": render_total / n_views,
            "avg_nn_per_view_s": nn_total / n_views,
            "avg_trace_per_view_s": trace_total / n_views,
            "avg_trace_auto_per_view_s": sum(v["trace_auto_nn_s"] for v in view_stats) / n_views,
        },
        "grand_total_s": load_scene_s + load_colmap_s + render_total,
        "breakdown": build_breakdown(
            load_scene_s=load_scene_s,
            load_colmap_s=load_colmap_s,
            views=view_stats,
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
