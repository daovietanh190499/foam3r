# Optimized CUDA Voronoi renderer

Faster reimplementation vs `voronoi_render_cuda/`:

| Optimization | Detail |
|--------------|--------|
| **Sparse CSR adjacency** | RadFoam-style `adjacency` + `adjacency_offsets`; walk only real faces (~10–15 vs padded 51) |
| **Prefetch `adjacent_diff`** | GPU kernel at load: `positions[neighbor] - positions[site]` per edge |
| **Fused kernel** | GPU NN + Voronoi walk in one launch (no `start_cells` buffer) |
| **Buffer pool** | Reuse device rays/out buffers across `trace()` calls |
| **CUDA stream** | Async H2D + kernel + D2H |
| **Fast math** | `--use_fast_math`, `__expf`, `rsqrtf`, `__ldg` |

## VORT formats

- **v1 (padded):** legacy layout for `voronoi_render_cuda`; optimized loader converts to CSR on load.
- **v2 (CSR):** smaller on disk, faster load — export with `--format csr`.

```bash
python -m myresearch.voronoi_render.export_bin --model model.pt --format csr
python -m myresearch.voronoi_render.export_bin --model model.pt --format v1   # legacy CUDA
```

## Build

```bash
bash myresearch/voronoi_render_cuda_optimized/build.sh native
```

## Run

```bash
.venv/bin/python myresearch/voronoi_render_cuda_optimized/eval_cuda.py
.venv/bin/python myresearch/voronoi_render_cuda_optimized/benchmark_compare.py
.venv/bin/python myresearch/voronoi_render_cuda_optimized/compare_legacy.py
```

Output: `myresearch/data/render_voronoi/`

## Benchmark (counter, view 1, 101K rays, RTX 2060)

CSR optimized vs legacy padded CUDA: **~1.7×** faster trace (see `benchmark_compare.py`).
