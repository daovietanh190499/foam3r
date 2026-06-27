# Tổng hợp I/O: Voronoi Render vs DUSt3R

Tài liệu mô tả **thành phần đầu vào / đầu ra**, **kích thước tensor**, **mối quan hệ**, và **xử lý background** của hai pipeline trong repo.

## Ký hiệu

| Ký hiệu | Ý nghĩa |
|---------|---------|
| **N** | Số Voronoi sites (cells) |
| **E** | Số cạnh Delaunay (CSR) = `adjacency_offsets[N]` |
| **D** | Max degree (padded), counter ≈ **51** |
| **d̄** | Degree trung bình, counter ≈ **15.4** |
| **R** | Số tia (rays) |
| **V** | Số ảnh đầu vào (DUSt3R) |
| **H×W** | Kích thước ảnh sau resize/crop |

**Ví dụ counter (`model.pt`):** N = 1,985,308 · E = 30,657,740 · R ≈ 101,400/view.

---

## 0. Cấu tạo `model.pt` (RadFoam checkpoint)

File checkpoint scene RadFoam sau training, lưu bằng `torch.save` — một `dict` gồm **6 tensor**, không có metadata / hyperparams kèm theo.

**Nguồn:** `radfoam/radfoam_model/scene.py` → `save_pt()` / `load_pt()`  
**Vị trí repo:** `model.pt` (root, ~674 MB cho scene counter)

### 0.1 Các thành phần

| Key | Shape | dtype | Ý nghĩa |
|-----|-------|-------|---------|
| `xyz` | `[N, 3]` | float32 | Tọa độ **primal points** (tâm Voronoi cell) |
| `density` | `[N, 1]` | float32 | Mật độ **thô** (raw log-space); khi render áp dụng `softplus` |
| `color_dc` | `[N, 3]` | float32 | SH band 0 — màu diffuse (3 kênh RGB) |
| `color_sh` | `[N, 45]` | float32 | SH band 1–3: `3 × ((L+1)² − 1)` với **L = 3** → 45 coeff |
| `adjacency` | `[E]` | int64 | CSR: index neighbor, flat list |
| `adjacency_offsets` | `[N+1]` | int64 | CSR offsets; `adjacency[offsets[i]:offsets[i+1]]` = neighbors của site `i` |

**Không có** trong file: camera, poses, AABB tree, `sh_degree` (suy ra từ `color_sh.shape[1]`), activation flags.

### 0.2 Kích thước (counter, N = 1,985,308)

| Key | Shape | Dung lượng tensor |
|-----|-------|-------------------|
| `xyz` | `[1,985,308, 3]` | ~23.8 MB |
| `density` | `[1,985,308, 1]` | ~7.9 MB |
| `color_dc` | `[1,985,308, 3]` | ~23.8 MB |
| `color_sh` | `[1,985,308, 45]` | ~357.4 MB |
| `adjacency` | `[30,657,740]` | ~245.3 MB |
| `adjacency_offsets` | `[1,985,309]` | ~15.9 MB |
| **Tổng payload** | | **~674 MB** (= kích thước file `.pt`) |

**Topology:**

```
E = adjacency_offsets[N] = 30,657,740
max_degree = 51
avg_degree ≈ 15.44
```

### 0.3 Quan hệ giữa các field

```
sh_degree = 3   (suy ra: color_sh.shape[1] = 3·((L+1)²−1) → L=3)

sh_attrs [N, 48] = cat(color_dc, color_sh)     # dùng khi render
density_render = softplus(density_raw)          # io.py / training activation

neighbor(i) = adjacency[ offsets[i] : offsets[i+1] ]
nbr_diff[i,j] = xyz[neighbor] - xyz[i]          # build lúc load hoặc prefetch GPU
```

### 0.4 Load trong `myresearch`

`voronoi_render/io.py` → `load_model_pt()`:

- Map `xyz` → `position`, giữ CSR `adjacency` / `adjacency_offsets`
- `density` → activate qua `softplus`
- `pack_sh_attributes(color_dc, color_sh)` → `sh_attrs [N, 48]`

Export binary cho CUDA:

| Format | File | Kích thước ~ |
|--------|------|--------------|
| VORT v1 (padded) | `data/scene.vort` | ~2134 MB |
| VORT v2 (CSR) | `data/scene_csr.vort` | ~544 MB |

```bash
python -m myresearch.voronoi_render.export_bin --model model.pt --format csr
python -m myresearch.voronoi_render.export_bin --model model.pt --format v1
```

---

## 1. Voronoi rendering (RadFoam / foam3r)

**Mục tiêu:** Từ scene đã train → render màu + opacity theo tia camera (volume rendering trên lưới Voronoi).

### 1.1 Scene tĩnh (load một lần)

| Thành phần | Shape | Kiểu | Mối quan hệ |
|------------|-------|------|-------------|
| `position` / `sites` | `[N, 3]` | float32 | Tọa độ tâm cell Voronoi |
| `density` | `[N, 1]` | float32 | σ sau `softplus` — hệ số hấp thụ Beer–Lambert |
| `color_dc` | `[N, 3]` | float32 | SH band 0 |
| `color_sh` | `[N, 45]` | float32 | SH band 1–3: 15 coeff × 3 kênh RGB |
| `sh_attrs` | `[N, 48]` | float32 | `cat(color_dc, color_sh)` |
| `sh_degree` | scalar | int | **3** (16 basis functions) |
| `adjacency` | `[E]` | int64 | CSR: danh sách phẳng index neighbor |
| `adjacency_offsets` | `[N+1]` | int64 | Neighbors site `i`: `adjacency[offsets[i]:offsets[i+1]]` |

**Quan hệ CSR:**

```
deg(i) = offsets[i+1] - offsets[i]     (1 ≤ deg(i) ≤ D)
E = Σ deg(i) ≈ N × d̄
```

**Dạng padded (Python / legacy CUDA):**

| Thành phần | Shape |
|------------|-------|
| `nbr_idx` | `[N, D]` |
| `nbr_diff` | `[N, D, 3]` — `position[neighbor] - position[i]` |
| `nbr_valid` | `[N, D]` bool |

**Dạng CSR GPU (optimized CUDA):**

| Thành phần | Shape |
|------------|-------|
| `adjacency` | `[E]` uint32 |
| `adjacency_offsets` | `[N+1]` uint32 |
| `adjacent_diff` | `[E, 3]` float32 — prefetch trên GPU |

**Hyperparams render:**

| Param | Mặc định |
|-------|----------|
| `max_steps` | 128 |
| `weight_threshold` | 0.001 |

**File binary VORT:** xem `voronoi_render/io.py` — v1 padded hoặc v2 CSR.

### 1.2 Đầu vào mỗi frame render

| Thành phần | Shape | Ghi chú |
|------------|-------|---------|
| `rays` | `[R, 6]` | `[ox, oy, oz, dx, dy, dz]` |
| `start_cells` *(optional)* | `[R]` int | Nếu `None` → 1-NN trên `position` (GPU grid hoặc cKDTree) |

### 1.3 Thuật toán (tóm tắt)

1. **Start cell:** 1-NN site gần ray origin.
2. **Walk:** Duyệt mặt Voronoi/Delaunay của cell hiện tại → tìm giao `t_hit` nhỏ nhất phía trước tia.
3. **Tích lũy:** Trong segment `[t0, t1]`: Beer–Lambert với `σ = density[i]`, màu từ SH theo **hướng tia** `dir`.
4. **Lặp** tối đa `max_steps` hoặc đến khi dừng (xem mục Background).

**Công thức segment:**

```
α = 1 - exp(-σ · (t1 - t0))
weight = T · α
rgb += weight · site_rgb
T *= (1 - α)
```

### 1.4 Đầu ra trace

| Thành phần | Shape | Ý nghĩa |
|------------|-------|---------|
| `rgba` | `[R, 4]` | RGB tích lũy + **opacity** = `1 - T` |

**Post-process eval:** composite nền trắng → ảnh `[H, W, 3]` (mục 3).

### 1.5 Xử lý background (Voronoi)

Background **không** được model hóa bằng geometry riêng (không có cell “vô cực” hay plane nền trong trace). Cơ chế gồm **3 tầng**:

#### Tầng 1 — Trong trace (Beer–Lambert + transmittance)

- Khởi tạo **transmittance** `T = 1` (tia chưa bị che).
- Mỗi cell góp phần opaque theo `α`; phần còn lại `T` = “nhìn xuyên” ra phía sau scene.
- **Dừng walk** khi:
  - Không còn mặt phía trước (`!any_face`) — tia thoát khỏi vùng có topology;
  - `T ≤ weight_threshold` (0.001) — coi như đủ trong suốt;
  - Đạt `max_steps` (128).
- **Opacity đầu ra:** `opacity = 1 - T` (clamp [0,1]).

→ Vùng trống / ngoài scene → `T ≈ 1` → **opacity ≈ 0** (trong suốt).

Code tham chiếu: `voronoi_render/trace.py`, `voronoi_render_cuda_optimized/src/trace_fused.cu`.

#### Tầng 2 — Composite nền trắng (eval / hiển thị)

Mip-NeRF 360 counter dùng **white background** (khớp GT). Sau trace:

```python
display_rgb = rgb + (1.0 - opacity)   # nền trắng (1,1,1)
```

Tương đương alpha compositing: `C = C_fg · α + C_bg · (1-α)` với `C_bg = white`, `α = opacity`.

Code: `voronoi_render/eval.py` → `composite_white_background()`.

#### Tầng 3 — Upsample

Stride > 1 → bilinear upsample về full resolution trước khi so PSNR với GT.

**Tóm lại:** Scene chỉ chứa vật thể (sites + density); background xuất hiện **implicit** qua transmittance còn lại, rồi được **blend trắng ở post-process** — không train/predict background riêng.

---

## 2. DUSt3R

**Mục tiêu:** Từ **V ảnh** không calibrated → pointmap per-pixel + global alignment → point cloud 3D (+ camera).

### 2.1 Đầu vào (sau `load_images`, model 512)

| Thành phần | Shape | Ghi chú |
|------------|-------|---------|
| `img` | `[1, 3, H_i, W_i]` | Normalize `(x-0.5)/0.5`, cạnh dài ≈ 512 |
| `true_shape` | `[2]` | `[H_i, W_i]` thực |
| `idx` | scalar | ID ảnh 0…V-1 |

**Cặp ảnh:** `make_pairs(..., scene_graph='complete', symmetrize=True)` → tối đa `V(V-1)` cặp × 2 chiều.

### 2.2 Đầu ra inference (per pair)

| Thành phần | Shape | Frame |
|------------|-------|-------|
| `pred1['pts3d']` | `[1, H_i, W_i, 3]` | Point pixel ảnh i, **frame ảnh i** |
| `pred2['pts3d_in_other_view']` | `[1, H_j, W_j, 3]` | Point pixel ảnh j, **frame ảnh i** |
| `pred1['conf']`, `pred2['conf']` | `[1, H, W]` | Confidence |

### 2.3 Sau global alignment

| Thành phần | Shape |
|------------|-------|
| `get_pts3d()[i]` | `[H_i, W_i, 3]` world frame |
| `get_masks()[i]` | `[H_i, W_i]` bool — `conf > min_conf_thr` |
| `get_im_poses()[i]` | `[4, 4]` cam2world |
| `get_focals()[i]` | scalar |

**Export:** PLY/GLB point cloud có màu (`dust3r/run_recon.py`).

**Background DUSt3R:** Không render tia. Pixel nền trời / vùng low-conf bị **lọc bởi mask confidence** khi export point cloud; không có bước composite nền trắng như Voronoi eval.

---

## 3. So sánh nhanh

| | **Voronoi render** | **DUSt3R** |
|--|-------------------|------------|
| Input chính | Scene `[N,·]` + rays `[R,6]` | V ảnh `[1,3,H,W]` |
| Representation | Lưới Voronoi 3D + CSR | Pointmap dense |
| Output | `[R,4]` RGBA → ảnh | Point cloud + poses |
| Background | Transmittance + composite trắng | Lọc conf, không composite |
| Camera | COLMAP rays cho sẵn | Ước lượng focal + pose |

---

## 4. Ví dụ số

**Voronoi counter:**

```
N = 1,985,308    E = 30,657,740    sh_attrs [N,48]
rays/view [101,400, 6]  →  rgba [101,400, 4]
```

**DUSt3R Chateau (2 ảnh, 512_dpt):**

```
img [1, 3, 512, 384] × 2
pts3d [1, 512, 384, 3]  →  PLY ~322k điểm (conf_thr=3)
```

---

## 5. File liên quan

| Pipeline | Code |
|----------|------|
| `model.pt` save/load | `radfoam/radfoam_model/scene.py` |
| Load + export VORT | `voronoi_render/io.py`, `export_bin.py` |
| Voronoi Python | `voronoi_render/trace.py`, `eval.py` |
| Voronoi CUDA CSR | `voronoi_render_cuda_optimized/src/trace_fused.cu` |
| DUSt3R | `dust3r/dust3r/model.py`, `inference.py`, `run_recon.py` |
