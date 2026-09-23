// M <= 8 tensor-core path for Q4_K / Q5_K, for Volta (sm_70).
//
// mul_mat_vec_q is limited by the number of instructions per weight byte, not by
// DRAM: its FMA pipe sits at ~47% while DRAM sits at ~23%. One mma.m8n8k4 does
// 256 MACs per warp, so the decode is paid once per weight instead of once per
// weight and per column.
//
// Layout. A warp owns 32 output columns and TC_SPLITK warps split K inside the
// CTA. Lane L holds weight column qp*8 + g and activation row g, with
// g = (lane&3) + 4*(lane&16) and qp = (lane>>2)&3. So a lane reads 32 bytes of
// its own column per 64-k chunk, which is exactly one 32 byte sector.
//
// No prepack. mma.m8n8k4 consumes four k values per slice and the order of those
// four is free, so A and B may agree on any permutation. This kernel pairs the
// codes at byte i and byte i+2 of a qs word, which one LOP3 extracts, and the A
// operand is written in the matching order by tc_convert_a.
//
// Codes become exact fp16 integers: "code | 0x6400" reads as 1024+code in fp16
// and stays exact for code <= 31, then one HADD2 removes the bias and one HFMA2
// applies (d*sc, -dmin*m).

#include "mmvq-tc.cuh"

#include <type_traits>

#define TC_NCOLS   32 // output columns per warp
#define TC_SPLITK   8 // warps per CTA, splitting K
#define TC_MPAD     8 // activation rows, padded

static_assert(sizeof(block_q4_K) == 144, "unexpected block_q4_K size");
static_assert(sizeof(block_q5_K) == 176, "unexpected block_q5_K size");

// A operand k order inside one 64-k chunk: for word i the low-nibble slice uses
// k = (4i, 4i+2, 4i+1, 4i+3) and the high-nibble slice follows at +32.
static __device__ __forceinline__ int tc_a_perm(const int k) {
    const int r = k & 63;
    const int i = r >> 3;
    const int t = r & 3;
    const int p = (t == 0) ? 0 : ((t == 1) ? 2 : ((t == 2) ? 1 : 3));
    return ((r & 4) == 0) ? (4*i + p) : (32 + 4*i + p);
}

#define TC_MMA(C, A0, A1, B0, B1)                                              \
    asm volatile(                                                              \
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "                     \
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "                      \
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                         \
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),          \
          "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                                   \
        : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// One exact half2 from two 5-bit codes already sitting at bits 0..4 / 16..20.
static __device__ __forceinline__ uint32_t tc_magic_scale(
        const uint32_t codes, const uint32_t h1024, const uint32_t fs, const uint32_t nm) {
    const half2 h = *reinterpret_cast<const half2 *>(&codes);
    const half2 t = __hsub2(h, *reinterpret_cast<const half2 *>(&h1024));
    const half2 r = __hfma2(t, *reinterpret_cast<const half2 *>(&fs),
                               *reinterpret_cast<const half2 *>(&nm));
    return *reinterpret_cast<const uint32_t *>(&r);
}

// fp32 activations -> fp16 A in the fragment k order above.
__global__ void tc_convert_a(const float * __restrict__ x, half * __restrict__ a,
                             const int64_t k, const int64_t m, const int64_t row_stride) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TC_MPAD * k) {
        return;
    }
    const int64_t row = idx / k;
    const int64_t p   = idx % k;
    const int64_t src = (p & ~(int64_t) 63) | (int64_t) tc_a_perm((int) p);
    a[idx] = row < m ? __float2half(x[row * row_stride + src]) : __float2half(0.0f);
}

// One CTA = one 32 column tile. Warp w takes super-blocks w, w+TC_SPLITK, ... so
// that the warps of a CTA sweep K together and keep the DRAM streams close.
template <bool Q5>
__global__ void __launch_bounds__(32 * TC_SPLITK, 2) tc_mul_mat_vec_q(
        const uint8_t * __restrict__ w, const half * __restrict__ a,
        float * __restrict__ y, const int64_t k, const int64_t m, const int64_t stride_row_y) {
#if !defined(VOLTA_MMA_AVAILABLE)
    NO_DEVICE_CODE;
#else
    using block_t = typename std::conditional<Q5, block_q5_K, block_q4_K>::type;

    __shared__ float partials[TC_SPLITK][256];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const int qp   = (lane >> 2) & 3;
    const int g    = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int col  = tile * TC_NCOLS + qp * 8 + g;

    const int      nblk = (int) (k / QK_K);
    const block_t * bcol = ((const block_t *) w) + (size_t) col * nblk;
    const half    * arow = a + (size_t) g * k;

    float acc[2][8];
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[c][i] = 0.0f;
        }

    const uint32_t h1024 = 0x64006400u;

    for (int b = warp; b < nblk; b += TC_SPLITK) {
        const block_t * blk = bcol + b;
        const half2 dm2 = blk->dm;
        const float d    = __low2float(dm2);
        const float dmin = __high2float(dm2);

        const uint32_t * scw = (const uint32_t *) blk->scales;
        const uint32_t sw0 = scw[0];
        const uint32_t sw1 = scw[1];
        const uint32_t sw2 = scw[2];

        uint32_t qh[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if constexpr (Q5) {
            const uint4 h0 = *(const uint4 *) (blk->qh);
            const uint4 h1 = *(const uint4 *) (blk->qh + 16);
            qh[0] = h0.x; qh[1] = h0.y; qh[2] = h0.z; qh[3] = h0.w;
            qh[4] = h1.x; qh[5] = h1.y; qh[6] = h1.z; qh[7] = h1.w;
        }

#pragma unroll
        for (int m4 = 0; m4 < 4; ++m4) {
            // Sub-blocks 2*m4 (low nibbles) and 2*m4+1 (high nibbles).
            uint32_t fs[2], nm[2];
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int j = 2 * m4 + t;
                uint8_t sc, mn;
                if (j < 4) {
                    sc = (uint8_t) ((sw0 >> (8 * j)) & 63u);
                    mn = (uint8_t) ((sw1 >> (8 * j)) & 63u);
                } else {
                    const int jj = j - 4;
                    sc = (uint8_t) (((sw2 >> (8 * jj)) & 0xFu) | (((sw0 >> (8 * jj)) & 0xC0u) >> 2));
                    mn = (uint8_t) (((sw2 >> (8 * jj + 4)) & 0xFu) | (((sw1 >> (8 * jj)) & 0xC0u) >> 2));
                }
                const half2 fs2 = __float2half2_rn(d * (float) sc);
                const half2 nm2 = __float2half2_rn(-dmin * (float) mn);
                fs[t] = *reinterpret_cast<const uint32_t *>(&fs2);
                nm[t] = *reinterpret_cast<const uint32_t *>(&nm2);
            }

            const uint4 q0 = *(const uint4 *) (blk->qs + 32 * m4);
            const uint4 q1 = *(const uint4 *) (blk->qs + 32 * m4 + 16);
            const uint32_t qw[8] = { q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w };
            const uint4 * av = (const uint4 *) (arow + (size_t) (4 * b + m4) * 64);

#pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint32_t b0, b1, b2, b3;
                if constexpr (Q5) {
                    const uint32_t hw = qh[i];
                    const uint32_t xl = (m4 <= 2) ? (hw << (4 - 2 * m4)) : (hw >> (2 * m4 - 4));
                    const uint32_t xh = (m4 <  2) ? (hw << (3 - 2 * m4)) : (hw >> (2 * m4 - 3));
                    const uint32_t cl = (xl & 0x10101010u) | ( qw[i]       & 0x0F0F0F0Fu);
                    const uint32_t ch = (xh & 0x10101010u) | ((qw[i] >> 4) & 0x0F0F0F0Fu);
                    b0 = tc_magic_scale(( cl        & 0x001F001Fu) | 0x64006400u, h1024, fs[0], nm[0]);
                    b1 = tc_magic_scale(((cl >>  8) & 0x001F001Fu) | 0x64006400u, h1024, fs[0], nm[0]);
                    b2 = tc_magic_scale(( ch        & 0x001F001Fu) | 0x64006400u, h1024, fs[1], nm[1]);
                    b3 = tc_magic_scale(((ch >>  8) & 0x001F001Fu) | 0x64006400u, h1024, fs[1], nm[1]);
                } else {
                    b0 = tc_magic_scale(( qw[i]        & 0x000F000Fu) | 0x64006400u, h1024, fs[0], nm[0]);
                    b1 = tc_magic_scale(((qw[i] >>  8) & 0x000F000Fu) | 0x64006400u, h1024, fs[0], nm[0]);
                    b2 = tc_magic_scale(((qw[i] >>  4) & 0x000F000Fu) | 0x64006400u, h1024, fs[1], nm[1]);
                    b3 = tc_magic_scale(((qw[i] >> 12) & 0x000F000Fu) | 0x64006400u, h1024, fs[1], nm[1]);
                }
                const uint4 ha = av[i];
                const uint32_t * au = (const uint32_t *) &ha;
                TC_MMA(acc[0], au[0], au[1], b0, b1);
                TC_MMA(acc[1], au[2], au[3], b2, b3);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc[0][i] += acc[1][i];
    }

    // C fragment: register i of lane L holds (row, col) below, inside the tile.
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        partials[warp][row * TC_NCOLS + qp * 8 + cl] = acc[0][i];
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 256; e += blockDim.x) {
        float v = 0.0f;
#pragma unroll
        for (int s = 0; s < TC_SPLITK; ++s) {
            v += partials[s][e];
        }
        const int row = e >> 5;
        const int cl  = e & 31;
        if (row < m) {
            y[row * stride_row_y + tile * TC_NCOLS + cl] = v;
        }
    }
#endif // !defined(VOLTA_MMA_AVAILABLE)
}

bool ggml_cuda_should_use_mmvq_tc(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc != GGML_CUDA_CC_VOLTA) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || dst->ne[1] != src1->ne[1]) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    // The weight read dominates and does not shrink with m, so only the full
    // 8-row verify batch beats mul_mat_vec_q.
    if (src1->ne[1] != TC_MPAD) {
        return false;
    }
    if (src0->ne[0] % QK_K != 0 || src0->ne[1] % TC_NCOLS != 0) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0]) {
        return false;
    }
    if (src0->nb[0] != ggml_type_size(src0->type) || src1->nb[0] != sizeof(float) ||
        dst->nb[0] != sizeof(float)) {
        return false;
    }
    return true;
}

void ggml_cuda_mul_mat_vec_q_tc(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne11 = src1->ne[1];

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<half> a_alloc(ctx.pool(), (size_t) TC_MPAD * ne00);
    half * a = a_alloc.get();

    const int64_t n_a = TC_MPAD * ne00;
    tc_convert_a<<<(unsigned) ((n_a + 255) / 256), 256, 0, stream>>>(
        (const float *) src1->data, a, ne00, ne11, src1->nb[1] / sizeof(float));

    const int64_t stride_row_y = dst->nb[1] / sizeof(float);
    const dim3 grid((unsigned) (ne01 / TC_NCOLS));
    const dim3 block(32 * TC_SPLITK);

    if (src0->type == GGML_TYPE_Q5_K) {
        tc_mul_mat_vec_q<true><<<grid, block, 0, stream>>>(
            (const uint8_t *) src0->data, a, (float *) dst->data, ne00, ne11, stride_row_y);
    } else {
        tc_mul_mat_vec_q<false><<<grid, block, 0, stream>>>(
            (const uint8_t *) src0->data, a, (float *) dst->data, ne00, ne11, stride_row_y);
    }
}
