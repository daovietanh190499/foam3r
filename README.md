# Voronoi Render Experiment (counter)

Tái hiện RadFoam Voronoi renderer: Python, CUDA (GPU NN), CUDA + CPU cKDTree.

**Repo chỉ track code.** Data và kết quả nằm local tại `myresearch/data/` (gitignored).

## Cấu trúc code

| Thư mục | Nội dung |
|---------|----------|
| `paths.py` | Đường dẫn mặc định tới `data/` |
| `voronoi_render/` | Python renderer + eval/benchmark CUDA |
| `voronoi_render_cuda/` | C++/CUDA extension |
| `voronoi_render_kdtree/` | Pipeline legacy CPU cKDTree |
| `report_latex/` | Báo cáo LaTeX (PDF build local) |
| `REPORT_VORONOI_RENDER.md` | Báo cáo Markdown |
| `VORONOI_RENDER_GUIDE.md` | Hướng dẫn kỹ thuật |

## Cấu trúc `data/` (local, không commit)

```
data/
├── scene.vort              # Export từ model.pt (~2 GB)
├── eval_python/            # Kết quả Python (metrics, PNG)
├── render_cuda/            # Kết quả CUDA + GPU NN
├── render_cuda_kdtree/     # Timing/eval cKDTree
└── report_figures/         # Figures cho LaTeX (python/, cuda/)
```

## Phụ thuộc ngoài repo

- `model.pt` — checkpoint RadFoam (root)
- `data/mipnerf360/counter` — COLMAP dataset
- `radfoam/data_loader/colmap.py`

```bash
pip install -r myresearch/requirements.txt
```

## Setup data lần đầu

```bash
# Export scene binary
python -m myresearch.voronoi_render.export_bin --model model.pt

# Build CUDA
bash myresearch/voronoi_render_cuda/build.sh
```

## Lệnh chạy

**Python:**
```bash
.venv/bin/python -m myresearch.eval_counter --indices 0,1,2,3,4 --stride 4
```

**CUDA + GPU NN:**
```bash
.venv/bin/python myresearch/voronoi_render/eval_cuda.py --indices 0,1,2,3,4
```

**CUDA + cKDTree:**
```bash
.venv/bin/python myresearch/voronoi_render_kdtree/eval_cuda.py --indices 0,1,2,3,4
```

**LaTeX** (cần figures trong `data/report_figures/`):
```bash
make -C myresearch/report_latex
```

Sau khi render, copy figures cho báo cáo:
```bash
cp myresearch/data/eval_python/view_*_pred_gt_err.png myresearch/data/report_figures/python/
cp myresearch/data/render_cuda/view_*_pred_gt_err.png myresearch/data/render_cuda/view_*_pred.png \
   myresearch/data/report_figures/cuda/
```
