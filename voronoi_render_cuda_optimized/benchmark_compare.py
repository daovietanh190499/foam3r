#!/usr/bin/env python3
"""Benchmark optimized voronoi_cuda vs legacy voronoi_render_cuda."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from myresearch.paths import CUDA_BUILD, RENDER_VORONOI, SCENE_VORT, VORONOI_BUILD
from myresearch.voronoi_render.eval import load_colmap_split


def bench_subprocess(module: str, build_dir: Path, scene: Path, rays_npy: Path, ray_chunk: int) -> dict:
    out_json = rays_npy.with_suffix(f".{module}.json")
    code = textwrap.dedent(
        f"""
        import json, sys, time
        import numpy as np
        import torch
        sys.path.insert(0, "{_ROOT}")
        sys.path.insert(0, "{build_dir}")
        rays_np = np.load("{rays_npy}")
        mod = __import__("{module}")
        cls = getattr(mod, "VoronoiScene", None) or mod.CudaScene

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        scene = cls("{scene}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_s = time.perf_counter() - t0

        parts = []
        chunk = {ray_chunk}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in range(0, rays_np.shape[0], chunk):
            e = min(s + chunk, rays_np.shape[0])
            parts.append(scene.trace(rays_np[s:e]))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        trace_s = time.perf_counter() - t0

        out = np.concatenate(parts, axis=0)
        stats = {{
            "module": "{module}",
            "load_scene_s": load_s,
            "trace_fused_s": trace_s,
            "total_render_s": trace_s,
            "rays": int(rays_np.shape[0]),
            "rays_per_sec": rays_np.shape[0] / trace_s if trace_s > 0 else 0,
        }}
        with open("{out_json}", "w") as f:
            json.dump(stats, f)
        """
    )
    subprocess.check_call([sys.executable, "-c", code])
    return json.loads(out_json.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, default=SCENE_VORT)
    p.add_argument("--data-dir", type=Path, default=Path("data/mipnerf360/counter"))
    p.add_argument("--out", type=Path, default=RENDER_VORONOI / "benchmark.json")
    p.add_argument("--view", type=int, default=1)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--ray-chunk", type=int, default=65536)
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args()

    import torch

    _, rays, _ = load_colmap_split(args.data_dir, "test", args.downsample, torch.device("cpu"))
    rays_np = rays[args.view][:: args.stride, :: args.stride].reshape(-1, 6).numpy().astype(np.float32)

    tmp = Path("/tmp/voronoi_bench")
    tmp.mkdir(exist_ok=True)
    rays_path = tmp / "rays.npy"
    np.save(rays_path, rays_np)

    for _ in range(args.warmup):
        bench_subprocess("voronoi_cuda", VORONOI_BUILD, args.scene, rays_path, args.ray_chunk)
        bench_subprocess("voronoi_render_cuda", CUDA_BUILD, args.scene, rays_path, args.ray_chunk)

    new = bench_subprocess("voronoi_cuda", VORONOI_BUILD, args.scene, rays_path, args.ray_chunk)
    old = bench_subprocess("voronoi_render_cuda", CUDA_BUILD, args.scene, rays_path, args.ray_chunk)

    speedup = old["trace_fused_s"] / max(new["trace_fused_s"], 1e-9)
    report = {"view": args.view, "rays": new["rays"], "voronoi_cuda": new, "voronoi_render_cuda": old, "speedup_trace": round(speedup, 3)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Speedup trace: {speedup:.2f}x")


if __name__ == "__main__":
    main()
