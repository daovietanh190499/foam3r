#!/usr/bin/env python3
"""Export ``model.pt`` to ``.vort`` binary for the CUDA renderer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from myresearch.paths import SCENE_VORT
from myresearch.voronoi_render.io import export_scene_bin, export_scene_bin_csr, load_model_pt
from myresearch.voronoi_render.adjacency import pad_adjacency


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("model.pt"))
    p.add_argument("--out", type=Path, default=SCENE_VORT)
    p.add_argument("--max-sites", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=128)
    p.add_argument(
        "--format",
        choices=("csr", "v1"),
        default="csr",
        help="csr: sparse VORT v2 (optimized CUDA); v1: padded legacy format",
    )
    args = p.parse_args()

    max_sites = args.max_sites if args.max_sites > 0 else None
    data = load_model_pt(args.model, max_sites=max_sites, device="cpu")
    n = data["position"].shape[0]
    offsets = data["adjacency_offsets"].cpu().numpy()
    max_deg = int((offsets[1:] - offsets[:-1]).max())
    avg_deg = (offsets[-1] / n) if n > 0 else 0.0
    print(f"Sites: {n} | max_degree: {max_deg} | avg_degree: {avg_deg:.1f}")

    if args.format == "csr":
        path = export_scene_bin_csr(data, args.out, max_steps=args.max_steps)
    else:
        path = export_scene_bin(data, args.out, max_steps=args.max_steps)
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB) format={args.format}")


if __name__ == "__main__":
    main()
