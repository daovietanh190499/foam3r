# Voronoi Render (Python)

Reference implementation for RadFoam **counter** scene.

Paths: `myresearch/paths.py` → outputs under `myresearch/data/` (gitignored).

## Quick start

```bash
python -m myresearch.voronoi_render.eval --indices 0,1,2,3,4 --stride 4

python -m myresearch.voronoi_render.export_bin --model model.pt

python myresearch/voronoi_render/eval_cuda.py

python myresearch/voronoi_render_kdtree/eval_cuda.py
```
