"""Default paths for experiment data and results (under ``myresearch/data/``)."""

from __future__ import annotations

from pathlib import Path

MYRESEARCH = Path(__file__).resolve().parent
DATA = MYRESEARCH / "data"

SCENE_VORT = DATA / "scene.vort"
EVAL_PYTHON = DATA / "eval_python"
RENDER_CUDA = DATA / "render_cuda"
RENDER_CUDA_KDTREE = DATA / "render_cuda_kdtree"
REPORT_FIGURES = DATA / "report_figures"
REPORT_FIGURES_PYTHON = REPORT_FIGURES / "python"
REPORT_FIGURES_CUDA = REPORT_FIGURES / "cuda"

CUDA_BUILD = MYRESEARCH / "voronoi_render_cuda" / "build"
VORONOI_BUILD = MYRESEARCH / "voronoi_render_cuda_optimized" / "build"
RENDER_VORONOI = DATA / "render_voronoi"
