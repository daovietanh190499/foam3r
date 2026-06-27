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
from myresearch.voronoi_render.io import export_scene_bin, load_model_pt
from myresearch.voronoi_render.adjacency import pad_adjacency


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("model.pt"))
    p.add_argument("--out", type=Path, default=SCENE_VORT)
    p.add_argument("--max-sites", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=128)
    args = p.parse_args()

    max_sites = args.max_sites if args.max_sites > 0 else None
    data = load_model_pt(args.model, max_sites=max_sites, device="cpu")
    nbr_idx, _, _ = pad_adjacency(data["position"], data["adjacency"], data["adjacency_offsets"])
    print(f"Sites: {data['position'].shape[0]} | max_degree: {nbr_idx.shape[1]}")

    path = export_scene_bin(data, args.out, max_steps=args.max_steps)
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
