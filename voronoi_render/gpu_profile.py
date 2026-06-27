#!/usr/bin/env python3
"""Profile GPU VRAM and utilization for Python vs CUDA Voronoi render."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from myresearch.paths import CUDA_BUILD, RENDER_CUDA, SCENE_VORT
from myresearch.voronoi_render.eval import load_colmap_split
from myresearch.voronoi_render.scene import VoronoiScene
from myresearch.voronoi_render.trace import nearest_site_indices


def gpu_query() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        name, total, used, free, util_gpu, util_mem = [x.strip() for x in out.split(",")]
        return {
            "name": name,
            "vram_total_mib": int(total),
            "vram_used_mib": int(used),
            "vram_free_mib": int(free),
            "util_gpu_pct": int(util_gpu),
            "util_mem_pct": int(util_mem),
        }
    except Exception as e:
        return {"error": str(e)}


def torch_mem_mib() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mib": round(torch.cuda.memory_allocated() / (1024**2), 2),
        "reserved_mib": round(torch.cuda.memory_reserved() / (1024**2), 2),
        "max_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
    }


def mib(x: int | float) -> float:
    return round(float(x) / (1024**2), 2)


class GpuSampler:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self):
        while not self._stop.is_set():
            s = gpu_query()
            s["t"] = time.perf_counter()
            self.samples.append(s)
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        used = [s["vram_used_mib"] for s in self.samples if "vram_used_mib" in s]
        util = [s["util_gpu_pct"] for s in self.samples if "util_gpu_pct" in s]
        return {
            "n_samples": len(self.samples),
            "vram_used_mib_min": min(used),
            "vram_used_mib_max": max(used),
            "vram_used_mib_avg": round(sum(used) / len(used), 1),
            "util_gpu_pct_min": min(util),
            "util_gpu_pct_max": max(util),
            "util_gpu_pct_avg": round(sum(util) / len(util), 1),
        }


def profile_python(model: Path, rays_hw: torch.Tensor, stride: int, ray_chunk: int) -> dict:
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    time.sleep(0.2)

    baseline = gpu_query()
    baseline_torch = torch_mem_mib()

    t0 = time.perf_counter()
    scene = VoronoiScene(max_intersections=128, ray_chunk_size=ray_chunk, device=device)
    scene.load_model_pt(model)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    after_load = gpu_query()
    after_load_torch = torch_mem_mib()

    h, w, _ = rays_hw.shape
    rays = rays_hw[::stride, ::stride].reshape(-1, 6).to(device)
    rh, rw = math.ceil(h / stride), math.ceil(w / stride)

    with GpuSampler() as sampler:
        t0 = time.perf_counter()
        parts = []
        for s in range(0, rays.shape[0], ray_chunk):
            e = min(s + ray_chunk, rays.shape[0])
            parts.append(scene.trace(rays[s:e]))
        torch.cuda.synchronize()
        trace_s = time.perf_counter() - t0

    after_trace = gpu_query()
    after_trace_torch = torch_mem_mib()
    sample_summary = sampler.summary()

    del scene, rays, parts
    torch.cuda.empty_cache()

    return {
        "backend": "python_pytorch",
        "load_s": round(load_s, 3),
        "trace_s": round(trace_s, 3),
        "rays": int(rh * rw),
        "baseline": baseline,
        "after_load": after_load,
        "after_trace": after_trace,
        "torch_after_load": after_load_torch,
        "torch_after_trace": after_trace_torch,
        "vram_delta_load_mib": after_load.get("vram_used_mib", 0) - baseline.get("vram_used_mib", 0),
        "vram_delta_trace_mib": after_trace.get("vram_used_mib", 0) - after_load.get("vram_used_mib", 0),
        "trace_sampling": sample_summary,
    }


def profile_cuda(
    scene_path: Path,
    model: Path,
    rays_hw: torch.Tensor,
    stride: int,
    ray_chunk: int,
    cuda_build: Path,
) -> dict:
    sys.path.insert(0, str(cuda_build))
    import voronoi_render_cuda  # noqa: E402

    torch.cuda.empty_cache()
    time.sleep(0.2)
    baseline = gpu_query()

    t0 = time.perf_counter()
    cuda_scene = voronoi_render_cuda.CudaScene(str(scene_path))
    load_s = time.perf_counter() - t0
    after_load = gpu_query()

    xyz = torch.load(model, map_location="cpu", weights_only=False)["xyz"].float()
    h, w, _ = rays_hw.shape
    rays = rays_hw[::stride, ::stride].reshape(-1, 6).cpu()
    rh, rw = math.ceil(h / stride), math.ceil(w / stride)
    start = nearest_site_indices(rays[:, :3], xyz).numpy().astype(np.int32)
    rays_np = rays.numpy().astype(np.float32)

    with GpuSampler() as sampler:
        t0 = time.perf_counter()
        parts = []
        for s in range(0, rays_np.shape[0], ray_chunk):
            e = min(s + ray_chunk, rays_np.shape[0])
            parts.append(cuda_scene.trace(rays_np[s:e], start[s:e]))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        trace_s = time.perf_counter() - t0

    after_trace = gpu_query()
    sample_summary = sampler.summary()

    # per-chunk transient (rays + start + out on device in bindings)
    n = min(ray_chunk, rays_np.shape[0])
    transient_bytes = n * 6 * 4 + n * 4 + n * 4 * 4

    del cuda_scene, parts
    torch.cuda.empty_cache()
    after_free = gpu_query()

    return {
        "backend": "cuda_extension",
        "load_s": round(load_s, 3),
        "trace_s": round(trace_s, 3),
        "rays": int(rh * rw),
        "num_sites": 1985308,
        "baseline": baseline,
        "after_load": after_load,
        "after_trace": after_trace,
        "after_free": after_free,
        "vram_scene_mib": after_load.get("vram_used_mib", 0) - baseline.get("vram_used_mib", 0),
        "vram_trace_peak_delta_mib": after_trace.get("vram_used_mib", 0) - after_load.get("vram_used_mib", 0),
        "vram_released_after_free_mib": after_load.get("vram_used_mib", 0) - after_free.get("vram_used_mib", 0),
        "transient_per_chunk_theoretical_mib": mib(transient_bytes),
        "trace_sampling": sample_summary,
        "torch_mem": torch_mem_mib(),
    }


def theoretical_scene_bytes(num_sites: int, max_degree: int, sh_dim: int) -> dict:
    n, md, sd = num_sites, max_degree, sh_dim
    parts = {
        "positions": n * 3 * 4,
        "sh_attrs": n * sd * 4,
        "density": n * 4,
        "nbr_idx": n * md * 4,
        "nbr_diff": n * md * 3 * 4,
        "nbr_valid": n * md * 1,
    }
    total = sum(parts.values())
    return {k: mib(v) for k, v in parts.items()} | {"total_mib": mib(total)}


def main():
    p = argparse.ArgumentParser(description="GPU memory/util profile")
    p.add_argument("--model", type=Path, default=Path("model.pt"))
    p.add_argument("--scene", type=Path, default=SCENE_VORT)
    p.add_argument("--data-dir", type=Path, default=Path("data/mipnerf360/counter"))
    p.add_argument("--out", type=Path, default=RENDER_CUDA / "gpu_stats.json")
    p.add_argument("--view", type=int, default=0)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--ray-chunk", type=int, default=65536)
    p.add_argument("--cuda-build", type=Path, default=CUDA_BUILD)
    p.add_argument("--skip-python", action="store_true", help="skip PyTorch scene (saves VRAM for CUDA-only)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1

    _, rays, _ = load_colmap_split(args.data_dir, "test", 2, torch.device("cpu"))
    rays_hw = rays[args.view]

    stats = {
        "gpu": gpu_query(),
        "config": {
            "view": args.view,
            "stride": args.stride,
            "ray_chunk": args.ray_chunk,
            "resolution": list(rays_hw.shape[:2]),
        },
        "theoretical_scene_cuda_mib": theoretical_scene_bytes(1985308, 51, 48),
    }

    if not args.skip_python:
        print("Profiling Python/PyTorch GPU ...")
        stats["python"] = profile_python(args.model, rays_hw, args.stride, args.ray_chunk)
        torch.cuda.empty_cache()
        time.sleep(1)

    print("Profiling CUDA extension ...")
    stats["cuda"] = profile_cuda(
        args.scene, args.model, rays_hw, args.stride, args.ray_chunk, args.cuda_build
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
