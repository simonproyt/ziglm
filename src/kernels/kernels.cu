#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// Helper functions for Half / BF16 conversion on CUDA
// ============================================================================

__device__ __forceinline__ float f16_to_f32(unsigned short h) {
    int s = (h >> 15) & 0x0001;
    int e = (h >> 10) & 0x001f;
    int m = h & 0x03ff;
    if (e == 0) {
        return (s ? -1.0f : 1.0f) * ((float)m / 1024.0f) * (1.0f / 16384.0f);
    } else if (e == 31) {
        return m ? 0.0f : (s ? -1e30f : 1e30f);
    } else {
        float val = 1.0f + (float)m / 1024.0f;
        int exp = e - 15;
        float scale = (exp >= 0) ? (float)(1 << exp) : (1.0f / (float)(1 << (-exp)));
        return (s ? -1.0f : 1.0f) * val * scale;
    }
}

__device__ __forceinline__ float bf16_to_f32(unsigned short h) {
    unsigned int u32 = ((unsigned int)h) << 16;
    return __uint_as_float(u32);
}

// ============================================================================
// Quantized Matrix-Vector Multiplication (GEMV) Kernels
// ============================================================================

// Q4_0: Block of 32 weights stored in 18 bytes (2 bytes f16 delta + 16 bytes nibbles)
extern "C" __global__ void gemv_q4_0_f32(
    const void* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    int lane = threadIdx.x; // 0..31
    int n_blocks = cols / 32;
    int row_bytes = n_blocks * 18;
    const unsigned char* row_data = ((const unsigned char*)weights) + row * row_bytes;

    float sum = 0.0f;
    for (int b = lane; b < n_blocks; b += 32) {
        const unsigned char* blk = row_data + b * 18;
        unsigned short scale_u16 = *(const unsigned short*)blk;
        float d = f16_to_f32(scale_u16);
        const unsigned char* qs = blk + 2;
        const float* x_blk = x + b * 32;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            unsigned char byte_val = qs[i];
            int v0 = (int)(byte_val & 0x0F) - 8;
            int v1 = (int)(byte_val >> 4) - 8;
            sum += ((float)v0 * d) * x_blk[i] + ((float)v1 * d) * x_blk[i + 16];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[row] = sum;
    }
}

// Q8_0: Block of 32 weights stored in 34 bytes (2 bytes f16 delta + 32 bytes int8)
extern "C" __global__ void gemv_q8_0_f32(
    const void* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    int lane = threadIdx.x; // 0..31
    int n_blocks = cols / 32;
    int row_bytes = n_blocks * 34;
    const unsigned char* row_data = ((const unsigned char*)weights) + row * row_bytes;

    float sum = 0.0f;
    for (int b = lane; b < n_blocks; b += 32) {
        const unsigned char* blk = row_data + b * 34;
        unsigned short scale_u16 = *(const unsigned short*)blk;
        float d = f16_to_f32(scale_u16);
        const signed char* qs = (const signed char*)(blk + 2);
        const float* x_blk = x + b * 32;

        #pragma unroll
        for (int i = 0; i < 32; i++) {
            sum += ((float)qs[i] * d) * x_blk[i];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[row] = sum;
    }
}

// Q4_K: 256 weights in 144 bytes
extern "C" __global__ void gemv_q4_k_f32(
    const void* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    int lane = threadIdx.x; // 0..31
    int n_blocks = cols / 256;
    size_t row_bytes = (size_t)n_blocks * 144;
    const unsigned char* row_data = ((const unsigned char*)weights) + ((size_t)row) * row_bytes;

    float sum = 0.0f;
    for (int sb = 0; sb < n_blocks; sb++) {
        const unsigned char* blk = row_data + sb * 144;
        unsigned short d_u16 = *(const unsigned short*)blk;
        unsigned short dmin_u16 = *(const unsigned short*)(blk + 2);
        float d = f16_to_f32(d_u16);
        float min = f16_to_f32(dmin_u16);
        const unsigned char* sc_bytes = blk + 4;
        const unsigned char* qs = blk + 16;
        const float* x_sb = x + sb * 256;

        unsigned char scales[8];
        unsigned char mins[8];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            scales[j] = sc_bytes[j] & 63;
            mins[j] = sc_bytes[j + 4] & 63;
            scales[j + 4] = (sc_bytes[j + 8] & 0x0F) | ((sc_bytes[j] >> 6) << 4);
            mins[j + 4] = (sc_bytes[j + 8] >> 4) | ((sc_bytes[j + 4] >> 6) << 4);
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float d0 = d * (float)scales[2 * j];
            float m0 = min * (float)mins[2 * j];
            float d1 = d * (float)scales[2 * j + 1];
            float m1 = min * (float)mins[2 * j + 1];

            unsigned char q = qs[j * 32 + lane];
            float w0 = (float)(q & 0x0F) * d0 - m0;
            float w1 = (float)(q >> 4) * d1 - m1;

            sum += w0 * x_sb[j * 64 + lane];
            sum += w1 * x_sb[j * 64 + 32 + lane];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[row] = sum;
    }
}

// Q6_K: 256 weights in 210 bytes
extern "C" __global__ void gemv_q6_k_f32(
    const void* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    int lane = threadIdx.x; // 0..31
    int n_blocks = cols / 256;
    size_t row_bytes = (size_t)n_blocks * 210;
    const unsigned char* row_data = ((const unsigned char*)weights) + ((size_t)row) * row_bytes;

    float sum = 0.0f;
    for (int sb = 0; sb < n_blocks; sb++) {
        const unsigned char* blk = row_data + sb * 210;
        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;
        const signed char* sc = (const signed char*)(blk + 192);
        unsigned short d_u16 = *(const unsigned short*)(blk + 208);
        float d = f16_to_f32(d_u16);
        const float* x_sb = x + sb * 256;

        #pragma unroll
        for (int n = 0; n < 2; n++) {
            int ql_offset = n * 64;
            int qh_offset = n * 32;
            int sc_offset = n * 8;
            int dst_offset = n * 128;

            int is = lane / 16;
            unsigned char ql_l = ql[ql_offset + lane];
            unsigned char ql_l32 = ql[ql_offset + lane + 32];
            unsigned char qh_l = qh[qh_offset + lane];

            int q1 = (int)((ql_l & 0x0F) | (((qh_l >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql_l32 & 0x0F) | (((qh_l >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql_l >> 4) | (((qh_l >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql_l32 >> 4) | (((qh_l >> 6) & 3) << 4)) - 32;

            float sc0 = d * (float)sc[sc_offset + is + 0];
            float sc1 = d * (float)sc[sc_offset + is + 2];
            float sc2 = d * (float)sc[sc_offset + is + 4];
            float sc3 = d * (float)sc[sc_offset + is + 6];

            sum += sc0 * (float)q1 * x_sb[dst_offset + lane + 0];
            sum += sc1 * (float)q2 * x_sb[dst_offset + lane + 32];
            sum += sc2 * (float)q3 * x_sb[dst_offset + lane + 64];
            sum += sc3 * (float)q4 * x_sb[dst_offset + lane + 96];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[row] = sum;
    }
}

// F16: Float16 weights
extern "C" __global__ void gemv_f16_f32(
    const unsigned short* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    int lane = threadIdx.x; // 0..31
    const unsigned short* row_data = weights + row * cols;

    float sum = 0.0f;
    for (int col = lane; col < cols; col += 32) {
        sum += f16_to_f32(row_data[col]) * x[col];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[row] = sum;
    }
}

// BF16: Bfloat16 weights
extern "C" __global__ void gemv_bf16_f32(
    const unsigned short* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    int lane = threadIdx.x; // 0..31
    const unsigned short* row_data = weights + row * cols;

    float sum = 0.0f;
    for (int col = lane; col < cols; col += 32) {
        sum += bf16_to_f32(row_data[col]) * x[col];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[row] = sum;
    }
}

// F32: Standard Float32 weights
extern "C" __global__ void gemv_f32_f32(
    const float* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    int lane = threadIdx.x; // 0..31
    const float* row_data = weights + row * cols;

    float sum = 0.0f;
    for (int col = lane; col < cols; col += 32) {
        sum += row_data[col] * x[col];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[row] = sum;
    }
}

// ============================================================================
// Batched GEMM Kernels (for Prefill)
// ============================================================================

extern "C" __global__ void gemm_q4_0_f32(
    const void* __restrict__ weights,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int batch_size,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int b_idx = blockIdx.y;
    if (row >= rows || b_idx >= batch_size) return;

    int lane = threadIdx.x;
    int n_blocks = cols / 32;
    int row_bytes = n_blocks * 18;
    const unsigned char* row_data = ((const unsigned char*)weights) + row * row_bytes;
    const float* x_batch = X + b_idx * cols;

    float sum = 0.0f;
    for (int b = lane; b < n_blocks; b += 32) {
        const unsigned char* blk = row_data + b * 18;
        unsigned short scale_u16 = *(const unsigned short*)blk;
        float d = f16_to_f32(scale_u16);
        const unsigned char* qs = blk + 2;
        const float* x_blk = x_batch + b * 32;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            unsigned char byte_val = qs[i];
            int v0 = (int)(byte_val & 0x0F) - 8;
            int v1 = (int)(byte_val >> 4) - 8;
            sum += ((float)v0 * d) * x_blk[i] + ((float)v1 * d) * x_blk[i + 16];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        Y[b_idx * rows + row] = sum;
    }
}

// ============================================================================
// Layer Normalization & Positional Embedding Kernels
// ============================================================================

// RMSNorm Kernel: y = (x / sqrt(mean(x^2) + eps)) * (weight + (use_unit_offset ? 1.0 : 0.0))
extern "C" __global__ void rmsnorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int n,
    float eps,
    int use_unit_offset
) {
    __shared__ float s_sum[256];
    int tid = threadIdx.x;

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float val = x[i];
        local_sum += val * val;
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / (float)n;
    float inv_std = rsqrtf(mean + eps);

    for (int i = tid; i < n; i += blockDim.x) {
        float w = (weight != NULL) ? (weight[i] + (use_unit_offset ? 1.0f : 0.0f)) : 1.0f;
        out[i] = x[i] * inv_std * w;
    }
}

// Rotary Position Embedding (RoPE) Kernel
extern "C" __global__ void rope_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    int pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float freq_base
) {
    int h = blockIdx.x; // head index
    int i = threadIdx.x; // index in [0, half_dim)
    int half_dim = head_dim / 2;
    if (i >= half_dim) return;

    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
    float val = (float)pos * freq;
    float cos_val = cosf(val);
    float sin_val = sinf(val);

    if (h < num_heads && q != NULL) {
        int offset0 = h * head_dim + i;
        int offset1 = offset0 + half_dim;
        float q0 = q[offset0];
        float q1 = q[offset1];
        q[offset0] = q0 * cos_val - q1 * sin_val;
        q[offset1] = q0 * sin_val + q1 * cos_val;
    }

    if (h < num_kv_heads && k != NULL) {
        int offset0 = h * head_dim + i;
        int offset1 = offset0 + half_dim;
        float k0 = k[offset0];
        float k1 = k[offset1];
        k[offset0] = k0 * cos_val - k1 * sin_val;
        k[offset1] = k0 * sin_val + k1 * cos_val;
    }
}

// ============================================================================
// Activation & Elementwise Kernels
// ============================================================================

// GeGLU Activation: act = GELU(gate) * up = (0.5 * gate * (1.0 + tanh(sqrt(2/pi) * (gate + 0.044715 * gate^3)))) * up
__device__ __forceinline__ float gelu_f32(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coef = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + coef * x * x * x)));
}

extern "C" __global__ void geglu_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float u = up[idx];
        out[idx] = gelu_f32(g) * u;
    }
}

// SwiGLU Activation: act = SiLU(gate) * up = (gate / (1 + exp(-gate))) * up
extern "C" __global__ void swiglu_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float u = up[idx];
        float silu = g / (1.0f + expf(-g));
        out[idx] = silu * u;
    }
}

// SiLU In-place
extern "C" __global__ void silu_f32(
    float* __restrict__ x,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        x[idx] = v / (1.0f + expf(-v));
    }
}

// Residual Addition: x += residual
extern "C" __global__ void add_f32(
    float* __restrict__ x,
    const float* __restrict__ residual,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] += residual[idx];
    }
}

// In-place scaling: x *= scale
extern "C" __global__ void scale_f32(
    float* __restrict__ x,
    float scale,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= scale;
    }
}

// Softmax with optional Softcap
extern "C" __global__ void softmax_f32(
    float* __restrict__ scores,
    int n,
    float softcap
) {
    __shared__ float s_max[256];
    __shared__ float s_sum[256];
    int tid = threadIdx.x;

    // Apply softcap if specified
    if (softcap > 0.0f) {
        float inv_sc = 1.0f / softcap;
        for (int i = tid; i < n; i += blockDim.x) {
            scores[i] = softcap * tanhf(scores[i] * inv_sc);
        }
        __syncthreads();
    }

    // Step 1: Find Max
    float local_max = -1e30f;
    for (int i = tid; i < n; i += blockDim.x) {
        if (scores[i] > local_max) local_max = scores[i];
    }
    s_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s];
        }
        __syncthreads();
    }
    float max_val = s_max[0];
    __syncthreads();

    // Step 2: Exp and Sum
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float e = expf(scores[i] - max_val);
        scores[i] = e;
        local_sum += e;
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / (s_sum[0] + 1e-9f);

    // Step 3: Normalize
    for (int i = tid; i < n; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
}
