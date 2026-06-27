#!/usr/bin/env python3
"""Compare Python vs CUDA Voronoi trace on identical rays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from myresearch.paths import CUDA_BUILD, SCENE_VORT
from myresearch.voronoi_render.io import export_scene_bin, load_model_pt
from myresearch.voronoi_render.scene import VoronoiScene
from myresearch.voronoi_render.trace import nearest_site_indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("model.pt"))
    p.add_argument("--scene", type=Path, default=SCENE_VORT)
    p.add_argument("--num-rays", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cuda-build", type=Path, default=CUDA_BUILD)
    args = p.parse_args()

    sys.path.insert(0, str(args.cuda_build))
    import voronoi_render_cuda  # noqa: E402

    device = torch.device("cuda")
    scene_py = VoronoiScene(device=device)
    scene_py.load_model_pt(args.model)

    g = torch.Generator(device=device)
    g.manual_seed(args.seed)
    rays = torch.randn(args.num_rays, 6, device=device, generator=g)
    rays[:, 3:] = torch.nn.functional.normalize(rays[:, 3:], dim=-1)
    start = nearest_site_indices(rays[:, :3], scene_py.sites)
    rays_np = rays.cpu().numpy().astype(np.float32)
    start_np = start.cpu().numpy().astype(np.int32)
    py_out = scene_py.trace(rays, start).cpu().numpy()

    del scene_py, rays, start
    torch.cuda.empty_cache()

    if not args.scene.exists():
        args.scene.parent.mkdir(parents=True, exist_ok=True)
        full = load_model_pt(args.model, device="cpu")
        export_scene_bin(full, args.scene)

    cuda_scene = voronoi_render_cuda.CudaScene(str(args.scene))
    cu_out = cuda_scene.trace(
        rays_np,
        start_np,
    )

    diff = np.abs(py_out - cu_out)
    print(f"Python vs CUDA | rays: {args.num_rays} | sites: {cuda_scene.num_sites}")
    print(f"  max abs diff:  {diff.max():.6e}")
    print(f"  mean abs diff: {diff.mean():.6e}")
    print(f"  RGB max diff:  {diff[:, :3].max():.6e}")
    print(f"  opacity diff:  {diff[:, 3].max():.6e}")
    ok = diff.max() < 1e-3
    print("PASS" if ok else "FAIL (tolerance 1e-3)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
