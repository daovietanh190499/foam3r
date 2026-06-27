#pragma once

#include "voronoi/math.cuh"

namespace voronoi {

__device__ inline void sh_basis(int degree, float3 dir, float basis[kShBasisSize]) {
    const float C0 = 0.28209479177387814f;
    const float C1 = 0.4886025119029199f;
    const float C2[5] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                         -1.0925484305920792f, 0.5462742152960396f};
    const float C3[7] = {-0.5900435899266435f,  2.890611442640554f,  -0.4570457994644658f,
                         0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
                         -0.5900435899266435f};

    const float x = dir.x, y = dir.y, z = dir.z;
    int k = 0;
    basis[k++] = C0;
    if (degree >= 1) {
        basis[k++] = -C1 * y;
        basis[k++] = C1 * z;
        basis[k++] = -C1 * x;
    }
    if (degree >= 2) {
        const float xx = x * x, yy = y * y, zz = z * z;
        const float xy = x * y, yz = y * z, xz = x * z;
        basis[k++] = C2[0] * xy;
        basis[k++] = C2[1] * yz;
        basis[k++] = C2[2] * (2.0f * zz - xx - yy);
        basis[k++] = C2[3] * xz;
        basis[k++] = C2[4] * (xx - yy);
    }
    if (degree >= 3) {
        const float xx = x * x, yy = y * y, zz = z * z;
        const float xy = x * y, yz = y * z, xz = x * z;
        basis[k++] = C3[0] * y * (3.0f * xx - yy);
        basis[k++] = C3[1] * xy * z;
        basis[k++] = C3[2] * y * (4.0f * zz - xx - yy);
        basis[k++] = C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
        basis[k++] = C3[4] * x * (4.0f * zz - xx - yy);
        basis[k++] = C3[5] * z * (xx - yy);
        basis[k++] = C3[6] * x * (xx - 3.0f * yy);
    }
}

__device__ inline void eval_sh_rgb(
    const float basis[kShBasisSize],
    const float* __restrict__ sh_rgb_vals,
    int sh_dim,
    float rgb[3]
) {
    rgb[0] = rgb[1] = rgb[2] = 0.5f;
#pragma unroll 4
    for (int i = 0; i < sh_dim; ++i) {
        const int ch = i % 3;
        const int bi = i / 3;
        rgb[ch] += basis[bi] * __ldg(&sh_rgb_vals[i]);
    }
    rgb[0] = fmaxf(rgb[0], 0.0f);
    rgb[1] = fmaxf(rgb[1], 0.0f);
    rgb[2] = fmaxf(rgb[2], 0.0f);
}

}  // namespace voronoi
