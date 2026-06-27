# Hướng dẫn đầy đủ: Tái hiện RadFoam Voronoi Renderer

Tài liệu này ghi lại toàn bộ quá trình nghiên cứu, debug, và triển khai pipeline render Voronoi từ checkpoint RadFoam (`model.pt`), bao gồm phiên bản Python tham chiếu và phiên bản C++/CUDA tối ưu. Mục tiêu cuối cùng: **render ảnh có chất lượng tương đương RadFoam gốc** trên scene **counter** (Mip-NeRF 360).

---

## Mục lục

1. [Tóm tắt công việc đã làm](#1-tóm-tắt-công-việc-đã-làm)
2. [Nguyên lý toán học](#2-nguyên-lý-toán-học)
3. [Cấu trúc checkpoint `model.pt`](#3-cấu-trúc-checkpoint-modelpt)
4. [Tải và tiền xử lý dữ liệu Mip-NeRF 360](#4-tải-và-tiền-xử-lý-dữ-liệu-mip-nerf-360)
5. [Cấu trúc thư mục dự án](#5-cấu-trúc-thư-mục-dự-án)
6. [Luồng xử lý end-to-end](#6-luồng-xử-lý-end-to-end)
7. [Chi tiết từng module Python](#7-chi-tiết-từng-module-python)
8. [Chi tiết implementation CUDA](#8-chi-tiết-implementation-cuda)
9. [Các lỗi đã gặp và cách sửa](#9-các-lỗi-đã-gặp-và-cách-sửa)
10. [Hướng dẫn chạy: Python](#10-hướng-dẫn-chạy-python)
11. [Hướng dẫn chạy: CUDA](#11-hướng-dẫn-chạy-cuda)
12. [Kết quả thực nghiệm](#12-kết-quả-thực-nghiệm)
13. [FAQ / Troubleshooting](#13-faq--troubleshooting)

---

## 1. Tóm tắt công việc đã làm

### Giai đoạn 1 — Khám phá `model.pt`

- Đọc `radfoam/viewer.py` và `radfoam/radfoam_model/scene.py` để hiểu cách RadFoam lưu scene.
- Phân tích file `model.pt` tại root repo: ~1.98M Voronoi sites, mỗi site có position, density, SH color, và **adjacency CSR** đã tính sẵn.

### Giai đoạn 2 — Prototype Python (ban đầu, chất lượng kém)

- Viết renderer Voronoi đơn giản trong `myresearch/renderer.py`.
- Trích xuất sites ra `.npz`, render bằng camera orbit tự tạo.
- **Kết quả:** mosaic polygon, không giống RadFoam — do nhiều lỗi thuật toán (xem mục 9).

### Giai đoạn 3 — Đọc lại thuật toán RadFoam CUDA

- Đọc `radfoam/src/tracing/tracing_utils.cuh` (Voronoi walk).
- Đọc `radfoam/src/tracing/pipeline.cu` (SH + Beer-Lambert).
- Đọc `radfoam/src/tracing/sh_utils.cuh` (layout hệ số SH).
- Implement lại chính xác trong Python (`radfoam_trace.py` → sau này gộp vào `voronoi_render/`).

### Giai đoạn 4 — Dataset và đánh giá đúng chuẩn

- Tải scene **counter** từ Mip-NeRF 360 (COLMAP format).
- Viết `eval_counter.py` dùng **camera COLMAP thật** + **white background compositing** như `radfoam/test.py`.
- Đạt PSNR ~26 dB trên 3 test views (stride=4).

### Giai đoạn 5 — Tổ chức lại code

- Gom code Python chạy tốt vào `myresearch/voronoi_render/`.
- Viết lại bằng C++/CUDA trong `myresearch/voronoi_render_cuda/`.
- Validate: Python vs CUDA max diff ≈ `2.7e-5` (PASS).

---

## 2. Nguyên lý toán học

### 2.1. Biểu diễn scene: Voronoi foam

RadFoam biểu diễn scene bằng tập **primal points** (sites) \(\{p_i\}_{i=1}^N\) trong \(\mathbb{R}^3\). Delaunay triangulation trên các điểm này tạo ra **Voronoi diagram** dual: mỗi site \(p_i\) sở hữu một **cell lồi** \(V_i\).

Hai site \(p_i, p_j\) là láng giềng Voronoi nếu chia sẻ một mặt phẳng — tương ứng cạnh trong Delaunay graph.

### 2.2. Bảng láng giềng (adjacency CSR)

Cho mỗi site \(i\), lưu danh sách neighbor indices trong mảng phẳng `adjacency`, với `adjacency_offsets` là CSR offsets:

```
neighbors(i) = adjacency[offsets[i] : offsets[i+1]]
```

Trong `model.pt` của counter:
- \(N \approx 1{,}985{,}308\) sites
- ~30.6M directed edges → ~15.4 neighbors/site trung bình
- `max_degree ≈ 51` sau khi pad

### 2.3. Ray tracing: đi bộ qua Voronoi cells

Mỗi tia có:
- Gốc \(o\), hướng đơn vị \(d\)
- Bắt đầu tại cell \(c_0\) (site gần nhất với \(o\))

Trong mỗi cell hiện tại (site \(p_i\)), với mỗi neighbor \(p_j\):

**Mặt phẳng bisector** giữa hai site:
\[
\text{face\_origin} = \frac{p_i + p_j}{2}, \quad \text{face\_normal} = p_j - p_i
\]

Giao điểm tia–mặt phẳng (tham số \(t\) dọc tia \(o + t d\)):
\[
t = \frac{(\text{face\_origin} - o) \cdot \text{face\_normal}}{d \cdot \text{face\_normal}}
\]

Chỉ xét mặt mà tia **đi ra** khỏi cell: \(d \cdot \text{face\_normal} > 0\).

Chọn \(t_1 = \min t\) trên tất cả mặt hợp lệ → cell kế tiếp.

**Quan trọng:** Nếu \(t_1 \le t_0\) (đã vượt qua mặt đó), **vẫn chuyển sang cell kế** nhưng **không tích lũy màu** trong segment đó. Đây là chi tiết dễ implement sai.

### 2.4. Tích lũy radiance: Beer–Lambert

Trong segment \([t_0, t_1]\) của cell \(i\), mật độ \(\sigma_i\) (đã activate):

\[
\Delta t = \max(t_1 - t_0, 0)
\]
\[
\alpha_i = 1 - e^{-\sigma_i \Delta t}
\]
\[
w_i = T \cdot \alpha_i
\]

Trong đó \(T\) là **transmittance** (độ trong suốt tích lũy). Sau mỗi segment:
\[
T \leftarrow T \cdot (1 - \alpha_i)
\]
\[
\text{RGB} \leftarrow \text{RGB} + w_i \cdot \text{color}_i
\]

Dừng khi \(T < \tau\) (`weight_threshold`, mặc định 0.001) hoặc không còn mặt thoát.

Opacity đầu ra: \(1 - T\).

### 2.5. Màu view-dependent: Spherical Harmonics (degree 3)

Mỗi site có hệ số SH cho RGB, layout **interleaved**:
```
[r₀, g₀, b₀, r₁, g₁, b₁, ..., r₁₅, g₁₅, b₁₅]   (48 giá trị, K=16 basis)
```

Trong đó \(r_0, g_0, b_0\) là DC coefficients (`color_dc`), phần còn lại là `color_sh`.

**Cực kỳ quan trọng:** Hướng đánh giá SH là **hướng tia** \(d\), **không phải** hướng từ camera đến site.

Basis functions \(Y_k(d)\) được tính theo công thức real SH (giống 3DGS / RadFoam CUDA).

Màu site:
\[
\text{RGB} = 0.5 + \sum_{k=0}^{15} Y_k(d) \cdot [r_k, g_k, b_k]
\]
\[
\text{RGB} \leftarrow \max(\text{RGB}, 0)
\]

### 2.6. Density activation

Raw density trong checkpoint được activate trước khi render:
\[
\sigma = \text{activation\_scale} \cdot \text{softplus}(\rho_{\text{raw}}, \beta=10)
\]

Với `activation_scale = 1.0` (theo `config.yaml`).

### 2.7. White background compositing

Scene counter dùng `white_background: true`. Sau khi trace ra RGBA:
\[
\text{RGB}_{\text{final}} = \text{RGB} + (1 - \text{opacity})
\]

Đây là bước **bắt buộc** để so sánh với ground truth Mip-NeRF 360.

---

## 3. Cấu trúc checkpoint `model.pt`

File `model.pt` là dict PyTorch do `RadFoamScene.save_pt()` tạo ra:

| Key | Shape | Dtype | Ý nghĩa |
|-----|-------|-------|---------|
| `xyz` | (N, 3) | float32 | Vị trí primal / Voronoi site |
| `density` | (N, 1) | float32 | Mật độ thô \(\rho\) (chưa softplus) |
| `color_dc` | (N, 3) | float32 | Hệ số SH bậc 0 (DC) |
| `color_sh` | (N, 45) | float32 | Hệ số SH bậc 1–3 (3×15) |
| `adjacency` | (E,) | int64 | CSR neighbor list |
| `adjacency_offsets` | (N+1,) | int64 | CSR offsets |

Với counter checkpoint:
```
N = 1,985,308
color_sh: 45 = 3 × ((3+1)² - 1) = 3 × 15  →  sh_degree = 3
E ≈ 30,657,740
```

### Config training (`config.yaml`)

```yaml
scene: counter
data_path: data/mipnerf360
dataset: colmap
sh_degree: 3
white_background: true
final_points: 2097152
activation_scale: 1.0
downsample: [4, 2]
```

---

## 4. Tải và tiền xử lý dữ liệu Mip-NeRF 360

### 4.1. Tải scene counter

Dataset nằm trong HuggingFace mirror (COLMAP format):

```bash
cd /mnt/data/foam3r
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'alexmkwizu/gaussian_training_datasets',
    repo_type='dataset',
    allow_patterns='mipnerf360/counter/*',
    local_dir='data'
)
"
```

Hoặc tải full Mip-NeRF 360 gốc:
```bash
wget http://storage.googleapis.com/gresearch/refraw/360_v2.zip
unzip 360_v2.zip -d data/mipnerf360
```

### 4.2. Cấu trúc thư mục sau khi tải

```
data/mipnerf360/counter/
├── images/          # full resolution
├── images_2/        # 2× downsampled
├── images_4/
├── images_8/
├── sparse/0/        # COLMAP: cameras.bin, images.bin, points3D.bin
└── poses_bounds.npy
```

### 4.3. COLMAP preprocessing (đã có sẵn)

Mip-NeRF 360 **đã chạy COLMAP** — không cần chạy lại. RadFoam đọc qua `radfoam/data_loader/colmap.py`:

1. Đọc reconstruction từ `sparse/0/`
2. Chọn split:
   - **train:** ảnh có index % 8 ≠ 0
   - **test:** ảnh có index % 8 == 0
3. Downsample: dùng thư mục `images_2` khi `downsample=2`
4. Với mỗi ảnh, tính **world rays** `(H, W, 6)` = `[origin(3), direction(3)]`
5. Load RGB ground truth `(H, W, 3)` normalized [0,1]

### 4.4. Thông số camera counter (test split)

- Downsample 2: **1558 × 1038** pixels
- 30 test views
- Focal length / poses lấy từ COLMAP (`cam_from_world.inverse()`)

---

## 5. Cấu trúc thư mục dự án

```
myresearch/
├── VORONOI_RENDER_GUIDE.md          ← tài liệu này
├── eval_counter.py                  ← wrapper → voronoi_render.eval
│
├── voronoi_render/                  ← PYTHON REFERENCE (chạy tốt)
│   ├── __init__.py
│   ├── README.md
│   ├── sh.py                        # Spherical harmonics
│   ├── adjacency.py                 # CSR → padded tables
│   ├── trace.py                     # Voronoi walk + Beer-Lambert
│   ├── scene.py                     # VoronoiScene
│   ├── io.py                        # load model.pt, export .vort
│   ├── eval.py                      # so sánh vs GT
│   ├── export_bin.py                # CLI export
│   ├── compare_cuda.py              # validate Python vs CUDA
│   └── output/
│       ├── scene.vort               # binary scene (~2.1 GB)
│       └── eval_counter/            # ảnh pred|gt|err + metrics.json
│
├── voronoi_render_cuda/             ← C++/CUDA
│   ├── README.md
│   ├── CMakeLists.txt
│   ├── build.sh
│   ├── include/voronoi_render/
│   │   ├── types.h
│   │   ├── scene.h
│   │   └── sh.cuh
│   ├── src/
│   │   ├── scene.cpp                # load .vort → GPU
│   │   ├── trace.cu                 # CUDA kernel
│   │   └── bindings.cpp             # pybind11
│   └── build/
│       └── voronoi_render_cuda*.so
│
├── data/                         # local results (gitignored)
│   └── eval_counter/
│       ├── metrics.json
│       └── view_*_pred_gt_err.png
│
data/
└── mipnerf360/counter/              ← dataset

model.pt                             ← RadFoam checkpoint (root repo)
config.yaml                          ← training config
```

---

## 6. Luồng xử lý end-to-end

### 6.1. Luồng eval Python (so với ground truth)

```
model.pt
    │
    ▼ load_model_pt()
    ├─ xyz, density, color_dc, color_sh
    ├─ adjacency, adjacency_offsets (CSR)
    ├─ density ← softplus(raw, β=10)
    └─ sh_attrs ← cat(color_dc, color_sh)  [interleaved 48]
    │
    ▼ pad_adjacency()
    ├─ nbr_idx    (N, max_deg)
    ├─ nbr_diff   (N, max_deg, 3)  = p_neighbor - p_i
    └─ nbr_valid  (N, max_deg)
    │
    ▼ VoronoiScene (GPU)

COLMAP dataset (counter, test, downsample=2)
    │
    ▼ COLMAPDataset
    ├─ rays  (num_views, H, W, 6)
    └─ rgbs  (num_views, H, W, 3)

Cho mỗi view:
    rays_flat = rays[view]  (có thể stride subsample)
    start_cells = nearest_site(origins)   # cKDTree
    rgba = trace_rays(...)
    rgb = rgba[..., :3] + (1 - rgba[..., 3])   # white bg
    PSNR(rgb, gt)
    save triptych [pred | gt | error]
```

### 6.2. Luồng CUDA

```
model.pt
    │
    ▼ export_scene_bin()  →  scene.vort (binary, ~2.1 GB)
    │
    ▼ CudaScene(path)  →  upload GPU
    │
    rays (N, 6) + start_cells (N,)
    │
    ▼ trace_rays_kernel  (1 CUDA thread / ray)
    │
    rgba (N, 4)
```

---

## 7. Chi tiết từng module Python

### 7.1. `voronoi_render/sh.py`

| Hàm | Mô tả |
|-----|-------|
| `sh_degree_from_num_coeffs(K)` | Suy ra degree từ số basis (16 → degree 3) |
| `sh_basis(degree, dirs)` | Tính 16 basis values \(Y_k(d)\) cho hướng tia `dirs` (...,3) |
| `pack_sh_attributes(dc, sh)` | `cat([dc, sh], dim=-1)` → (N, 48) interleaved |
| `eval_sh_rgb(sh_coeffs, sh_rgb_vals)` | `0.5 + sum_k Y_k * [r_k,g_k,b_k]`, clamp ≥ 0 |

**Layout interleaved** (khớp RadFoam CUDA `load_sh_as_rgb`):
```python
# sh_rgb_vals[i] với i % 3 = channel, i // 3 = basis index
rgb[ch] += sh_coeffs[bi] * sh_rgb_vals[i]
```

### 7.2. `voronoi_render/adjacency.py`

| Hàm | Mô tả |
|-----|-------|
| `pad_adjacency(points, adjacency, offsets)` | Chuyển CSR → ma trận padded cố định cho batch trace |

**Input CSR:**
```
offsets[i] .. offsets[i+1]  →  neighbor indices của site i
```

**Output:**
- `nbr_idx[i, f]` = index site láng giềng thứ f (hoặc -1 nếu pad)
- `nbr_diff[i, f] = points[nbr] - points[i]`
- `nbr_valid[i, f]` = True nếu láng giềng thật

### 7.3. `voronoi_render/trace.py`

| Hàm | Mô tả |
|-----|-------|
| `nearest_site_indices(origins, sites)` | Tìm site gần nhất cho mỗi ray origin (cKDTree hoặc chunked cdist) |
| `trace_rays(...)` | **Core algorithm** — Voronoi walk + SH color + Beer-Lambert |

**Vòng lặp `trace_rays` (mỗi step):**

```python
# 1. Tính giao điểm với tất cả mặt Voronoi của cell hiện tại
face_origins = primal + diffs * 0.5
face_normals = diffs
dp = dot(face_normals, ray_dir)
t_hit = dot(face_origins - origin, face_normals) / dp
face_ok = valid & (dp > 0)

# 2. Chọn mặt gần nhất
t1, face_sel = min(t_hit where face_ok)

# 3. Quyết định advance / contribute
advance = alive & ~no_face
contrib = advance & (t1 > t0)          # ← chỉ tích lũy nếu t1 > t0

# 4. Tích lũy màu (SH theo ray direction)
site_rgb = eval_sh_rgb(sh_basis(ray_dir), sh_attrs[cell])
alpha = 1 - exp(-sigma * (t1 - t0))
rgb += transmittance * alpha * site_rgb

# 5. Cập nhật state
t0 = max(t0, t1) if advance
current = next_cell if advance
alive = (transmittance > threshold) if contrib else advance
```

### 7.4. `voronoi_render/scene.py`

| Class / Method | Mô tả |
|----------------|-------|
| `VoronoiScene.__init__` | Cấu hình max_steps, weight_threshold, chunk sizes |
| `load_model_pt(path)` | Load checkpoint + build padded adjacency |
| `set_from_tensors(...)` | Set sites, SH, density, adjacency |
| `trace(rays)` | Chunk rays → `trace_rays()` |

### 7.5. `voronoi_render/io.py`

| Hàm | Mô tả |
|-----|-------|
| `_activate_density(raw)` | `softplus(raw, β=10) * activation_scale` |
| `load_model_pt(path)` | Load `model.pt` → dict tensors trên GPU/CPU |
| `load_sites_npz(path)` | Load từ `.npz` đã export trước đó |
| `export_scene_bin(scene, path)` | Export binary `.vort` cho CUDA |

**Format `.vort` (little-endian):**

| Offset | Field | Type |
|--------|-------|------|
| 0 | magic | uint32 = `0x54524F56` ("VORT") |
| 4 | version | uint32 = 1 |
| 8 | num_sites | uint32 |
| 12 | max_degree | uint32 |
| 16 | sh_degree | uint32 |
| 20 | weight_threshold | float32 |
| 24 | max_steps | float32 |
| 28 | sh_dim | uint32 |
| 32 | positions | float32 [N×3] |
| | sh_attrs | float32 [N×48] |
| | density | float32 [N] (đã activate) |
| | nbr_idx | int32 [N×max_deg] |
| | nbr_diff | float32 [N×max_deg×3] |
| | nbr_valid | uint8 [N×max_deg] |

### 7.6. `voronoi_render/eval.py`

Script đánh giá chính:

| Hàm | Mô tả |
|-----|-------|
| `load_colmap_split(...)` | Wrap `COLMAPDataset` từ radfoam |
| `render_image(scene, rays, stride)` | Trace + white bg + optional upscale |
| `save_triptych(pred, gt)` | Lưu [pred\|gt\|error], trả PSNR |
| `main()` | Loop qua test views, ghi `metrics.json` |

---

## 8. Chi tiết implementation CUDA

### 8.1. Kiến trúc

```
voronoi_render_cuda_core (static lib)
├── scene.cpp     — đọc .vort, cudaMalloc, cudaMemcpy H→D
└── trace.cu      — __global__ trace_rays_kernel

voronoi_render_cuda (pybind11 module)
└── bindings.cpp  — class CudaScene { trace(rays, start_cells) }
```

### 8.2. `trace.cu` — kernel chính

```cuda
__global__ void trace_rays_kernel(
    SceneGpu scene,      // pointers GPU
    float* rays,         // (num_rays, 6)
    int* start_cells,    // (num_rays,)
    int num_rays,
    float* out_rgba      // (num_rays, 4)
)
```

- **1 thread = 1 ray**
- Block size: 256 threads
- Logic **giống hệt** `trace.py` (đã validate numerically)

### 8.3. `sh.cuh` — device functions

| Hàm | Mô tả |
|-----|-------|
| `sh_basis(degree, dir, basis[16])` | Real SH basis trên GPU |
| `eval_sh_rgb(basis, sh_vals, rgb)` | `0.5 + Σ`, clamp ≥ 0 |

### 8.4. Build system

`CMakeLists.txt`:
- C++17 + CUDA 17
- `CMAKE_CUDA_ARCHITECTURES`: mặc định `75 80 86 89 90`
- Override: `./build.sh native` hoặc `./build.sh "86;89"`
- pybind11 từ venv: `python -m pybind11 --cmakedir`

### 8.5. Python bindings

```python
import voronoi_render_cuda
scene = voronoi_render_cuda.CudaScene("path/to/scene.vort")
rgba = scene.trace(rays_np, start_cells_np)  # → (N, 4) float32
```

---

## 9. Các lỗi đã gặp và cách sửa

Đây là phần quan trọng — giải thích tại sao các bản đầu cho kết quả **rất xấu**.

| # | Lỗi | Triệu chứng | Cách sửa |
|---|-----|-------------|----------|
| 1 | SH dùng hướng camera→site | Màu sai, mosaic | Dùng `sh_basis(ray_direction)` |
| 2 | SH layout reshape (3,15) thay vì interleaved | Màu hỗn loạn | `cat([dc, sh])` flat, eval theo `i%3`, `i//3` |
| 3 | Thiếu baseline `+0.5` trong SH | Ảnh tối/sai tone | `rgb = 0.5 + ...` |
| 4 | Rebuild Delaunay bằng scipy thay vì adjacency gốc | Topology sai hoàn toàn | Dùng `adjacency` từ `model.pt` |
| 5 | Subsample sites + induced subgraph | Mất hầu hết neighbors (~0.4/site) | Không subsample adjacency; dùng full model |
| 6 | Không advance cell khi `t1 ≤ t0` | Ray bị kẹt, mosaic polygon | `advance` tách biệt `contrib` |
| 7 | Camera orbit gần scene (dist≈16, scene≈130) | Nhìn thấy mặt Voronoi cell | Dùng COLMAP cameras thật |
| 8 | Thiếu white background compositing | PSNR thấp, nền đen | `rgb + (1 - opacity)` |
| 9 | Density chưa softplus | Sigma sai | `softplus(raw, β=10)` trước trace |
| 10 | Điều kiện `dp > 1e-6` thay vì `dp > 0` | Giao điểm sai nhẹ | Dùng `dp > 0` như CUDA gốc |

---

## 10. Hướng dẫn chạy: Python

### 10.1. Yêu cầu môi trường

```bash
cd /mnt/data/foam3r
# Dùng venv có sẵn
source .venv/bin/activate   # hoặc dùng .venv/bin/python trực tiếp

# Dependencies chính (đã có trong pyproject.toml):
# torch, scipy, pycolmap, pillow, huggingface_hub
```

### 10.2. Chuẩn bị dữ liệu

```bash
# Tải counter (nếu chưa có)
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('alexmkwizu/gaussian_training_datasets',
    repo_type='dataset', allow_patterns='mipnerf360/counter/*', local_dir='data')
"

# Kiểm tra
ls data/mipnerf360/counter/images_2/ | wc -l   # ~240 ảnh
ls data/mipnerf360/counter/sparse/0/          # cameras.bin, images.bin, ...
```

### 10.3. Kiểm tra checkpoint

```bash
.venv/bin/python -c "
import torch
d = torch.load('model.pt', map_location='cpu', weights_only=False)
print('keys:', list(d.keys()))
print('sites:', d['xyz'].shape[0])
print('adjacency:', d['adjacency'].shape[0], 'edges')
"
# Kỳ vọng: ~1,985,308 sites, ~30M edges
```

### 10.4. Chạy eval so với ground truth

```bash
cd /mnt/data/foam3r

# Eval 3 test views, stride=4 (nhanh, ~1558/4 × 1038/4 rays/view)
.venv/bin/python -m myresearch.voronoi_render.eval \
    --model model.pt \
    --data-dir data/mipnerf360/counter \
    --out myresearch/data/eval_python \
    --indices 0,1,2 \
    --stride 4 \
    --downsample 2 \
    --max-intersections 128 \
    --ray-chunk 8192 \
    --device cuda
```

**Output:**
```
myresearch/data/eval_python/
├── view_000_pred_gt_err.png    # [pred | GT | |error|]
├── view_001_pred_gt_err.png
├── view_002_pred_gt_err.png
└── metrics.json                # PSNR per view
```

### 10.5. Các tham số quan trọng

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--stride` | 4 | Subsample rays (1=full res, chậm) |
| `--downsample` | 2 | Dùng `images_2/` (khớp training) |
| `--max-intersections` | 128 | Số bước Voronoi walk tối đa / ray |
| `--ray-chunk` | 8192 | Rays mỗi batch (VRAM) |
| `--max-sites` | 0 (all) | **Không nên subsample** nếu muốn chất lượng đúng |
| `--indices` | 0,1,2,3,4 | Test view indices (0..29) |

### 10.6. Eval full resolution (chậm)

```bash
.venv/bin/python -m myresearch.voronoi_render.eval \
    --indices 0 \
    --stride 1 \
    --ray-chunk 4096 \
    --device cuda
# ~1.6M rays/view, cần GPU lớn, mất nhiều phút
```

### 10.7. Wrapper tương thích cũ

```bash
.venv/bin/python -m myresearch.eval_counter --indices 0,1,2 --stride 4
# → gọi cùng code voronoi_render.eval
```

---

## 11. Hướng dẫn chạy: CUDA

### 11.1. Build extension

```bash
cd /mnt/data/foam3r/myresearch/voronoi_render_cuda

# GPU hiện tại (nhanh nhất cho máy bạn)
./build.sh native

# Hoặc nhiều SM (portable binary)
./build.sh

# Hoặc chỉ định
./build.sh "86;89"
```

**Output:** `build/voronoi_render_cuda.cpython-312-x86_64-linux-gnu.so`

**Yêu cầu:** CUDA toolkit, CMake ≥ 3.18, pybind11 (có trong venv).

### 11.2. Export scene sang binary

```bash
cd /mnt/data/foam3r

# Bắt buộc dùng FULL model.pt (không --max-sites)
.venv/bin/python -m myresearch.voronoi_render.export_bin \
    --model model.pt \
    --out myresearch/data/scene.vort \
    --max-steps 128

# Output: ~2.1 GB, mất ~1-2 phút (pad adjacency 2M sites)
```

### 11.3. Validate Python vs CUDA

```bash
.venv/bin/python -m myresearch.voronoi_render.compare_cuda \
    --model model.pt \
    --scene myresearch/data/scene.vort \
    --num-rays 1024 \
    --cuda-build myresearch/voronoi_render_cuda/build
```

**Kỳ vọng:**
```
Python vs CUDA | rays: 1024 | sites: 1985308
  max abs diff:  2.73e-05
PASS
```

> **Lưu ý VRAM:** Script tự `del` Python scene trước khi load CUDA scene (tránh OOM khi giữ cả hai trên GPU).

### 11.4. Dùng CUDA từ Python

```python
import sys
import numpy as np
sys.path.insert(0, "myresearch/voronoi_render_cuda/build")
import voronoi_render_cuda

from myresearch.voronoi_render.trace import nearest_site_indices
from myresearch.voronoi_render.io import load_model_pt
import torch

# Rays từ COLMAP (ví dụ)
rays = ...  # (N, 6) float32
origins = torch.from_numpy(rays[:, :3]).cuda()
sites = load_model_pt("model.pt", device="cuda")["position"]
start = nearest_site_indices(origins, sites).cpu().numpy().astype(np.int32)

scene = voronoi_render_cuda.CudaScene("myresearch/data/scene.vort")
rgba = scene.trace(rays.astype(np.float32), start)

# White background
rgb = rgba[:, :3] + (1.0 - rgba[:, 3:4])
```

### 11.5. Rebuild sau khi sửa code CUDA

```bash
cd myresearch/voronoi_render_cuda/build
cmake --build . -j$(nproc)
```

---

## 12. Kết quả thực nghiệm

### 12.1. Eval counter (Python, stride=4, downsample=2)

| View | PSNR (dB) |
|------|-----------|
| 0 | 24.52 |
| 1 | 28.13 |
| 2 | 25.42 |
| **Avg** | **26.02** |

File: `myresearch/data/eval_python/metrics.json`

Ảnh so sánh: `view_*_pred_gt_err.png` — cột trái prediction, giữa ground truth, phải error map.

### 12.2. Python vs CUDA numerical match

| Metric | Value |
|--------|-------|
| Rays tested | 1024 |
| Sites | 1,985,308 |
| Max abs diff (RGBA) | 2.73e-05 |
| RGB max diff | 2.73e-05 |
| Opacity max diff | 5.48e-06 |
| Status | **PASS** |

### 12.3. Scene binary

| File | Size | Sites | max_degree |
|------|------|-------|------------|
| `scene.vort` | ~2.1 GB | 1,985,308 | 51 |

---

## 13. FAQ / Troubleshooting

### Q: Render ra mosaic polygon, không giống ảnh thật?

**A:** Kiểm tra lần lượt:
1. Có dùng adjacency từ `model.pt` không? (không rebuild scipy)
2. Có subsample sites không? (phải dùng full ~2M sites)
3. Có dùng COLMAP cameras không? (không dùng orbit camera tự tạo)
4. Có white background compositing không?
5. SH có dùng `ray.direction` không?

### Q: CUDA báo out of memory?

**A:**
- Scene ~2.1 GB chỉ trên GPU đã tốn ~2GB
- Không giữ Python scene + CUDA scene cùng lúc
- Giảm `--num-rays` khi test
- Cần GPU ≥ 8GB VRAM cho full scene

### Q: `export_bin` với `--max-sites` bị lỗi?

**A:** Subsample làm mất adjacency gốc. Export **luôn dùng full model** không `--max-sites`.

### Q: PSNR thấp hơn RadFoam paper (~29 dB)?

**A:** Có thể do:
- `stride > 1` khi eval (upsample bilinear không bù hết)
- Python trace chậm hơn CUDA gốc (nhưng thuật toán đã khớp)
- Khác biệt nhỏ về start_cell (cKDTree vs AABB tree của RadFoam)

### Q: Làm sao render view cụ thể bằng COLMAP camera?

```python
# Trong eval.py, --indices chọn view trong test split
# Test views = mỗi ảnh thứ 8 trong sorted COLMAP names
.venv/bin/python -m myresearch.voronoi_render.eval --indices 5 --stride 4
```

### Q: File cũ ở đâu?

| Cũ | Mới |
|----|-----|
| `myresearch/radfoam_trace.py` | `voronoi_render/trace.py` |
| `myresearch/radfoam_io.py` | `voronoi_render/io.py` |
| `myresearch/spherical_harmonics.py` | `voronoi_render/sh.py` |
| `myresearch/scene.py` (FoamScene radfoam) | `voronoi_render/scene.py` (VoronoiScene) |
| `myresearch/eval_counter.py` | wrapper → `voronoi_render/eval.py` |
| `myresearch/render_radfoam_views.py` | orbit camera demo (không dùng cho eval chuẩn) |

---

## Phụ lục A: Công thức SH basis (degree 3)

```
Y₀ = C₀
Y₁ = -C₁·y,  Y₂ = C₁·z,  Y₃ = -C₁·x
Y₄ = C₂[0]·xy,  Y₅ = C₂[1]·yz,  Y₆ = C₂[2]·(2zz-xx-yy),  ...
...
```

Với \(C_0 = 0.282095\), \(C_1 = 0.488603\), và \(C_2, C_3\) như trong `sh.py` / `sh.cuh`.

## Phụ lục B: Lệnh nhanh (cheat sheet)

```bash
# === PYTHON EVAL ===
cd /mnt/data/foam3r
.venv/bin/python -m myresearch.voronoi_render.eval \
    --model model.pt \
    --data-dir data/mipnerf360/counter \
    --indices 0,1,2 --stride 4 --device cuda

# === EXPORT CHO CUDA ===
.venv/bin/python -m myresearch.voronoi_render.export_bin \
    --model model.pt \
    --out myresearch/data/scene.vort

# === BUILD CUDA ===
cd myresearch/voronoi_render_cuda && ./build.sh native

# === VALIDATE CUDA ===
cd /mnt/data/foam3r
.venv/bin/python -m myresearch.voronoi_render.compare_cuda \
    --scene myresearch/data/scene.vort \
    --num-rays 1024
```

---

*Tài liệu được tạo: 2026-06-26. Pipeline validate trên scene counter, checkpoint `model.pt`, PSNR ~26 dB (stride=4), Python/CUDA numerical match PASS.*
