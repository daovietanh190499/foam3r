#!/usr/bin/env python3
"""Verify voronoi_cuda matches voronoi_render_cuda on random rays."""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from myresearch.paths import CUDA_BUILD, SCENE_VORT, VORONOI_BUILD


def _run_backend(module: str, build_dir: Path, scene: Path, rays_npy: Path, out_npy: Path) -> None:
    code = textwrap.dedent(
        f"""
        import sys, numpy as np
        sys.path.insert(0, "{_ROOT}")
        sys.path.insert(0, "{build_dir}")
        rays = np.load("{rays_npy}")
        mod = __import__("{module}")
        cls = getattr(mod, "VoronoiScene", None) or mod.CudaScene
        scene = cls("{scene}")
        out = scene.trace(rays)
        np.save("{out_npy}", out)
        """
    )
    subprocess.check_call([sys.executable, "-c", code])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, default=SCENE_VORT)
    p.add_argument("--num-rays", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import numpy as np
    import torch

    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed)
    rays = torch.randn(args.num_rays, 6, generator=g).numpy().astype(np.float32)
    rays[:, 3:] = rays[:, 3:] / np.linalg.norm(rays[:, 3:], axis=1, keepdims=True)

    tmp = Path("/tmp/voronoi_compare")
    tmp.mkdir(exist_ok=True)
    rays_path = tmp / "rays.npy"
    new_path = tmp / "new.npy"
    old_path = tmp / "old.npy"
    np.save(rays_path, rays)

    _run_backend("voronoi_cuda", VORONOI_BUILD, args.scene, rays_path, new_path)
    _run_backend("voronoi_render_cuda", CUDA_BUILD, args.scene, rays_path, old_path)

    out_new = np.load(new_path)
    out_old = np.load(old_path)
    diff = np.abs(out_new - out_old)
    print(f"rays={args.num_rays} max_diff={diff.max():.6e} mean_diff={diff.mean():.6e}")
    ok = diff.max() < 1e-2
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
