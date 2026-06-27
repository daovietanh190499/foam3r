# Voronoi Render (CUDA)

C++/CUDA reimplementation of `myresearch/voronoi_render/` — same algorithm, one CUDA thread per ray.

## Build

```bash
cd myresearch/voronoi_render_cuda
chmod +x build.sh
./build.sh          # SM 75,80,86,89,90
./build.sh native   # only current GPU
```

Requires: CUDA toolkit, CMake 3.18+, pybind11 (already in project deps).

## GPU nearest-neighbor lookup

`CudaScene.trace(rays)` without `start_cells` runs **1-NN on GPU** using a uniform spatial grid
built at load time (~8 MB extra VRAM; reuses existing `positions`, no duplicate 2 GB buffer).

```bash
# Render (NN + trace fully on GPU)
python myresearch/voronoi_render/eval_cuda.py --indices 0,1,2
```

Optional API:

```python
import voronoi_render_cuda
scene = voronoi_render_cuda.CudaScene("scene.vort")
rgba = scene.trace(rays_np)                      # auto GPU NN
idx = scene.nearest_site_indices(origins_np)    # NN only
rgba = scene.trace(rays_np, start_cells=idx)     # explicit start cells
```


## Binary format (`.vort`)

| Field | Type |
|-------|------|
| magic | uint32 `VORT` |
| version | uint32 |
| num_sites, max_degree, sh_degree | uint32 |
| weight_threshold, max_steps | float |
| sh_dim | uint32 |
| positions | float32 N×3 |
| sh_attrs | float32 N×48 |
| density | float32 N |
| nbr_idx | int32 N×max_deg |
| nbr_diff | float32 N×max_deg×3 |
| nbr_valid | uint8 N×max_deg |

## Algorithm

Matches Python `trace.py`:

1. SH color from **ray direction** (not view-to-site)
2. Voronoi face walk with `dp > 0`, advance even when `t1 <= t0`
3. Beer-Lambert: `alpha = 1 - exp(-sigma * dt)`
