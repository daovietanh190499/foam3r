# CUDA render — CPU cKDTree (legacy)

Output mặc định: `myresearch/data/render_cuda_kdtree/`

GPU NN pipeline: `myresearch/voronoi_render/eval_cuda.py` → `myresearch/data/render_cuda/`

```bash
.venv/bin/python myresearch/voronoi_render_kdtree/eval_cuda.py
.venv/bin/python myresearch/voronoi_render_kdtree/benchmark_cuda.py
```

Cần `model.pt` (xyz) và `myresearch/data/scene.vort`.
