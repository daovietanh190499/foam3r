#!/usr/bin/env python3
"""Evaluate Voronoi renderer on Mip-NeRF 360 counter vs ground truth."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from myresearch.paths import EVAL_PYTHON
from myresearch.voronoi_render.scene import VoronoiScene


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = ((pred - gt) ** 2).mean().item()
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def composite_white_background(rgba: torch.Tensor) -> torch.Tensor:
    rgb = rgba[..., :3]
    opacity = rgba[..., 3:4]
    return (rgb + (1.0 - opacity)).clamp(0.0, 1.0)


def load_colmap_split(data_dir: Path, split: str, downsample: int, device: torch.device):
    sys.path.insert(0, str(_ROOT / "radfoam"))
    from data_loader.colmap import COLMAPDataset

    ds = COLMAPDataset(str(data_dir), split=split, downsample=downsample)
    return ds, ds.all_rays.to(device), ds.all_rgbs.to(device)


def render_image(
    scene: VoronoiScene,
    rays_hw: torch.Tensor,
    *,
    white_background: bool,
    stride: int,
) -> torch.Tensor:
    h, w, _ = rays_hw.shape
    rays = rays_hw[::stride, ::stride].reshape(-1, 6)
    rh, rw = math.ceil(h / stride), math.ceil(w / stride)
    rgba = scene.trace(rays).reshape(rh, rw, 4)
    return rgba_to_display_rgb(rgba, h, w, white_background=white_background, stride=stride)


def rgba_to_display_rgb(
    rgba: torch.Tensor,
    h: int,
    w: int,
    *,
    white_background: bool,
    stride: int,
) -> torch.Tensor:
    """Composite + optional bilinear upsample (shared by Python and CUDA eval)."""
    rgb = composite_white_background(rgba) if white_background else rgba[..., :3].clamp(0, 1)
    if stride > 1:
        rgb = torch.nn.functional.interpolate(
            rgb.permute(2, 0, 1).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
    return rgb


def save_triptych(path: Path, pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = pred.detach().cpu().clamp(0, 1)
    gt = gt.detach().cpu().clamp(0, 1)
    err = (pred - gt).abs()
    trip = torch.cat([pred, gt, err], dim=1)
    Image.fromarray((trip.numpy() * 255).astype("uint8")).save(path)
    return psnr(pred, gt)


def main():
    p = argparse.ArgumentParser(description="Eval counter scene vs GT")
    p.add_argument("--model", type=Path, default=Path("model.pt"))
    p.add_argument("--data-dir", type=Path, default=Path("data/mipnerf360/counter"))
    p.add_argument("--out", type=Path, default=EVAL_PYTHON)
    p.add_argument("--split", choices=["test", "train"], default="test")
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--indices", type=str, default="0,1,2,3,4")
    p.add_argument("--max-sites", type=int, default=0)
    p.add_argument("--max-intersections", type=int, default=128)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--ray-chunk", type=int, default=8192)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    max_sites = args.max_sites if args.max_sites > 0 else None

    scene = VoronoiScene(
        max_intersections=args.max_intersections,
        ray_chunk_size=args.ray_chunk,
        device=device,
    )
    scene.load_model_pt(args.model, max_sites=max_sites)
    print(f"Sites: {scene.sites.shape[0]}")

    ds, rays, rgbs = load_colmap_split(args.data_dir, args.split, args.downsample, device)
    view_ids = [int(x) for x in args.indices.split(",") if x.strip()]

    metrics = []
    for vid in view_ids:
        print(f"Rendering view {vid:03d} ...")
        pred = render_image(scene, rays[vid], white_background=True, stride=args.stride)
        img_psnr = save_triptych(args.out / f"view_{vid:03d}_pred_gt_err.png", pred, rgbs[vid])
        metrics.append({"view": vid, "psnr": img_psnr})
        print(f"  PSNR: {img_psnr:.3f} dB")

    summary = {
        "model": str(args.model),
        "num_sites": int(scene.sites.shape[0]),
        "views": metrics,
        "avg_psnr": sum(m["psnr"] for m in metrics) / max(len(metrics), 1),
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Average PSNR: {summary['avg_psnr']:.3f} dB")


if __name__ == "__main__":
    main()
