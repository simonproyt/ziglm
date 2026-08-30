#include "cuda_bridge.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define WARP_SIZE 32

// ============================================================================
// Device & Memory Management
// ============================================================================

extern "C" int cuda_device_get_info(int device_id, char* name, size_t name_len, size_t* total_vram_bytes) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) return (int)err;

    if (name && name_len > 0) {
        strncpy(name, prop.name, name_len - 1);
        name[name_len - 1] = '\0';
    }
    if (total_vram_bytes) {
        *total_vram_bytes = prop.totalGlobalMem;
    }
    return 0;
}

extern "C" int cuda_malloc(void** ptr, size_t bytes) {
    return (int)cudaMalloc(ptr, bytes);
}

extern "C" int cuda_free(void* ptr) {
    return (int)cudaFree(ptr);
}

extern "C" int cuda_memcpy_h2d(void* dst, const void* src, size_t bytes, CudaStream_t stream) {
    if (stream) {
        return (int)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream);
    } else {
        return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    }
}

extern "C" int cuda_memcpy_d2h(void* dst, const void* src, size_t bytes, CudaStream_t stream) {
    if (stream) {
        return (int)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    } else {
        return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    }
}

extern "C" int cuda_memcpy_d2d(void* dst, const void* src, size_t bytes, CudaStream_t stream) {
    if (stream) {
        return (int)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    } else {
        return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
    }
}

extern "C" int cuda_stream_create(CudaStream_t* stream) {
    cudaStream_t s;
    cudaError_t err = cudaStreamCreate(&s);
    if (err != cudaSuccess) return (int)err;
    *stream = (CudaStream_t)s;
    return 0;
}

extern "C" int cuda_stream_destroy(CudaStream_t stream) {
    if (!stream) return 0;
    return (int)cudaStreamDestroy((cudaStream_t)stream);
}

extern "C" int cuda_stream_sync(CudaStream_t stream) {
    if (!stream) return (int)cudaDeviceSynchronize();
    return (int)cudaStreamSynchronize((cudaStream_t)stream);
}

// ============================================================================
// Float Conversions
// ============================================================================

__device__ __forceinline__ float f16_to_f32(unsigned short h) {
    return __half2float(*(const __half*)&h);
}

__device__ __forceinline__ float bf16_to_f32(unsigned short b) {
    unsigned int u = ((unsigned int)b) << 16;
    return __int_as_float(u);
}

// ============================================================================
// Quantized GEMV Kernels
// ============================================================================

// Q4_0: 18-byte blocks (2-byte f16 scale + 16-byte nibbles = 32 weights, 100% coalesced 32-thread parallelization)
__global__ void k_gemv_q4_0(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    int lane = threadIdx.x; // 0..31
    int num_blocks = cols / 32;
    const unsigned char* row_weights = weights + (size_t)row * num_blocks * 18;
    int total_bytes = num_blocks * 16;

    float sum = 0.0f;
    for (int k = lane; k < total_bytes; k += 32) {
        int b = k / 16;
        int i = k % 16;

        const unsigned char* block_ptr = row_weights + b * 18;
        float d = f16_to_f32(*(const unsigned short*)block_ptr);
        unsigned char byte = block_ptr[2 + i];
        int q0 = (int)(byte & 0x0F) - 8;
        int q1 = (int)(byte >> 4) - 8;
        const float* x_block = x + b * 32;

        sum += d * ((float)q0 * x_block[i] + (float)q1 * x_block[i + 16]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[row] = sum;
    }
}

// Q8_0: 34-byte blocks (2-byte f16 scale + 32 int8 weights, 100% coalesced 32-thread parallelization)
__global__ void k_gemv_q8_0(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    int lane = threadIdx.x;
    int num_blocks = cols / 32;
    const unsigned char* row_weights = weights + (size_t)row * num_blocks * 34;

    float sum = 0.0f;
    for (int k = lane; k < cols; k += 32) {
        int b = k / 32;
        int i = k % 32;

        const unsigned char* block_ptr = row_weights + b * 34;
        float d = f16_to_f32(*(const unsigned short*)block_ptr);
        const signed char* qs = (const signed char*)(block_ptr + 2);

        sum += d * ((float)qs[i] * x[k]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[row] = sum;
    }
}

// Q4_K GEMV (256-element superblocks = 144 bytes)
__global__ void k_gemv_q4_k(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    int lane = threadIdx.x;
    int num_superblocks = cols / 256;
    const unsigned char* row_weights = weights + (size_t)row * num_superblocks * 144;

    float sum = 0.0f;
    for (int sb = lane; sb < num_superblocks; sb += 32) {
        const unsigned char* block = row_weights + sb * 144;
        float d = f16_to_f32(*(const unsigned short*)(block + 0));
        float min = f16_to_f32(*(const unsigned short*)(block + 2));
        const unsigned char* scales_raw = block + 4;
        const unsigned char* qs = block + 16;
        const float* x_sb = x + sb * 256;

        unsigned char scales[8];
        unsigned char mins[8];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            scales[j] = scales_raw[j] & 63;
            mins[j] = scales_raw[j + 4] & 63;
            scales[j + 4] = (scales_raw[j + 8] & 0x0F) | ((scales_raw[j] >> 6) << 4);
            mins[j + 4] = (scales_raw[j + 8] >> 4) | ((scales_raw[j + 4] >> 6) << 4);
        }

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float d0 = d * (float)scales[2 * j];
            float m0 = min * (float)mins[2 * j];
            float d1 = d * (float)scales[2 * j + 1];
            float m1 = min * (float)mins[2 * j + 1];

            const unsigned char* q_sub = qs + j * 32;
            const float* x_sub0 = x_sb + j * 64;
            const float* x_sub1 = x_sb + j * 64 + 32;

            float acc_q0 = 0.0f;
            float acc_x0 = 0.0f;
            float acc_q1 = 0.0f;
            float acc_x1 = 0.0f;

            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                unsigned char q = q_sub[l];
                float q0 = (float)(q & 0x0F);
                float q1 = (float)(q >> 4);
                float x0 = x_sub0[l];
                float x1 = x_sub1[l];

                acc_q0 += q0 * x0;
                acc_x0 += x0;
                acc_q1 += q1 * x1;
                acc_x1 += x1;
            }

            sum += (acc_q0 * d0 - acc_x0 * m0) + (acc_q1 * d1 - acc_x1 * m1);
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

// Q6_K GEMV (256-element superblocks = 210 bytes, 100% coalesced 32-thread parallelization)
__global__ void k_gemv_q6_k(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    int lane = threadIdx.x; // 0..31
    int num_superblocks = cols / 256;
    const unsigned char* row_weights = weights + (size_t)row * num_superblocks * 210;
    int total_l_steps = num_superblocks * 64; // 64 elements of (q1,q2,q3,q4) per superblock

    float sum = 0.0f;
    for (int k = lane; k < total_l_steps; k += 32) {
        int sb = k / 64;
        int rem = k % 64;
        int n = rem / 32;
        int l = rem % 32;
        int is = l / 16;

        const unsigned char* block = row_weights + sb * 210;
        const unsigned char* ql = block + 0 + n * 64;
        const unsigned char* qh = block + 128 + n * 32;
        const signed char* scales = (const signed char*)(block + 192 + n * 8);
        float d = f16_to_f32(*(const unsigned short*)(block + 208));
        const float* x_sb = x + sb * 256 + n * 128;

        unsigned char ql_l = ql[l];
        unsigned char ql_l32 = ql[l + 32];
        unsigned char qh_l = qh[l];

        int q1 = (int)((ql_l & 0x0F) | (((qh_l >> 0) & 3) << 4)) - 32;
        int q2 = (int)((ql_l32 & 0x0F) | (((qh_l >> 2) & 3) << 4)) - 32;
        int q3 = (int)((ql_l >> 4) | (((qh_l >> 4) & 3) << 4)) - 32;
        int q4 = (int)((ql_l32 >> 4) | (((qh_l >> 6) & 3) << 4)) - 32;

        float d_sc0 = d * (float)scales[is + 0];
        float d_sc1 = d * (float)scales[is + 2];
        float d_sc2 = d * (float)scales[is + 4];
        float d_sc3 = d * (float)scales[is + 6];

        sum += (float)q1 * d_sc0 * x_sb[l + 0];
        sum += (float)q2 * d_sc1 * x_sb[l + 32];
        sum += (float)q3 * d_sc2 * x_sb[l + 64];
        sum += (float)q4 * d_sc3 * x_sb[l + 96];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[row] = sum;
    }
}

// F16 GEMV
__global__ void k_gemv_f16(
    const unsigned short* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    int lane = threadIdx.x;
    const unsigned short* row_data = weights + (size_t)row * cols;

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

// BF16 GEMV
__global__ void k_gemv_bf16(
    const unsigned short* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    int lane = threadIdx.x;
    const unsigned short* row_data = weights + (size_t)row * cols;

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

// F32 GEMV
__global__ void k_gemv_f32(
    const float* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    int lane = threadIdx.x;
    const float* row_data = weights + (size_t)row * cols;

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

// GEMV Host Dispatchers (8 warps = 256 threads per block)
extern "C" void cuda_gemv_q4_0(const void* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((rows + 7) / 8);
    k_gemv_q4_0<<<grid, block, 0, (cudaStream_t)stream>>>((const unsigned char*)weights, x, y, rows, cols);
}

extern "C" void cuda_gemv_q8_0(const void* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((rows + 7) / 8);
    k_gemv_q8_0<<<grid, block, 0, (cudaStream_t)stream>>>((const unsigned char*)weights, x, y, rows, cols);
}

extern "C" void cuda_gemv_q4_k(const void* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((rows + 7) / 8);
    k_gemv_q4_k<<<grid, block, 0, (cudaStream_t)stream>>>((const unsigned char*)weights, x, y, rows, cols);
}

extern "C" void cuda_gemv_q6_k(const void* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((rows + 7) / 8);
    k_gemv_q6_k<<<grid, block, 0, (cudaStream_t)stream>>>((const unsigned char*)weights, x, y, rows, cols);
}

extern "C" void cuda_gemv_f16(const void* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((rows + 7) / 8);
    k_gemv_f16<<<grid, block, 0, (cudaStream_t)stream>>>((const unsigned short*)weights, x, y, rows, cols);
}

extern "C" void cuda_gemv_bf16(const void* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((rows + 7) / 8);
    k_gemv_bf16<<<grid, block, 0, (cudaStream_t)stream>>>((const unsigned short*)weights, x, y, rows, cols);
}

extern "C" void cuda_gemv_f32(const float* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((rows + 7) / 8);
    k_gemv_f32<<<grid, block, 0, (cudaStream_t)stream>>>((const float*)weights, x, y, rows, cols);
}

// ============================================================================
// Normalization & Elementwise Kernels
// ============================================================================

__global__ void k_rmsnorm(
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

extern "C" void cuda_rmsnorm(const float* x, const float* weight, float* out, int n, float eps, int use_unit_offset, CudaStream_t stream) {
    k_rmsnorm<<<1, 256, 0, (cudaStream_t)stream>>>(x, weight, out, n, eps, use_unit_offset);
}

__global__ void k_rope(
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

extern "C" void cuda_rope(float* q, float* k, int pos, int num_heads, int num_kv_heads, int head_dim, float freq_base, CudaStream_t stream) {
    int max_heads = (num_heads > num_kv_heads) ? num_heads : num_kv_heads;
    int half_dim = head_dim / 2;
    int threads = (half_dim < 256) ? half_dim : 256;
    k_rope<<<max_heads, threads, 0, (cudaStream_t)stream>>>(q, k, pos, num_heads, num_kv_heads, head_dim, freq_base);
}

__device__ __forceinline__ float gelu_f32(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coef = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + coef * x * x * x)));
}

__global__ void k_geglu(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = gelu_f32(gate[idx]) * up[idx];
    }
}

extern "C" void cuda_geglu(const float* gate, const float* up, float* out, int n, CudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    k_geglu<<<blocks, threads, 0, (cudaStream_t)stream>>>(gate, up, out, n);
}

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void k_swiglu(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = silu_f32(gate[idx]) * up[idx];
    }
}

extern "C" void cuda_swiglu(const float* gate, const float* up, float* out, int n, CudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    k_swiglu<<<blocks, threads, 0, (cudaStream_t)stream>>>(gate, up, out, n);
}

__global__ void k_add(float* __restrict__ x, const float* __restrict__ residual, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] += residual[idx];
    }
}

extern "C" void cuda_add(float* x, const float* residual, int n, CudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    k_add<<<blocks, threads, 0, (cudaStream_t)stream>>>(x, residual, n);
}

__global__ void k_scale(float* __restrict__ x, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= scale;
    }
}

extern "C" void cuda_scale(float* x, float scale, int n, CudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    k_scale<<<blocks, threads, 0, (cudaStream_t)stream>>>(x, scale, n);
}

__global__ void k_tanh_softcap(float* __restrict__ x, float cap, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = cap * tanhf(x[idx] / cap);
    }
}

extern "C" void cuda_tanh_softcap(float* x, float cap, int n, CudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    k_tanh_softcap<<<blocks, threads, 0, (cudaStream_t)stream>>>(x, cap, n);
}

// ============================================================================
// GPU-Resident KV Cache
// ============================================================================

__global__ void k_kv_cache_put(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int layer_idx,
    int pos,
    int max_seq,
    int kv_dim,
    int max_kv_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < kv_dim) {
        size_t offset = ((size_t)layer_idx * max_seq + pos) * max_kv_dim + idx;
        k_cache[offset] = k[idx];
        v_cache[offset] = v[idx];
    }
}

extern "C" void cuda_kv_cache_put(
    float* k_cache,
    float* v_cache,
    const float* k,
    const float* v,
    int layer_idx,
    int pos,
    int max_seq,
    int n_kv_heads,
    int head_dim,
    CudaStream_t stream
) {
    int kv_dim = n_kv_heads * head_dim;
    int threads = 256;
    int blocks = (kv_dim + threads - 1) / threads;
    k_kv_cache_put<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        k_cache, v_cache, k, v, layer_idx, pos, max_seq, kv_dim, kv_dim
    );
}

// ============================================================================
// GPU-Resident Multi-Head Attention Forward
// ============================================================================

__global__ void k_attention_forward(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    int donor_layer,
    int pos,
    int max_seq,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float attn_scale,
    float softcap,
    int sliding_window
) {
    int h = blockIdx.x; // Query head index (0..n_heads-1)
    if (h >= n_heads) return;

    int gqa_group = (n_kv_heads > 0) ? (n_heads / n_kv_heads) : 1;
    int kv_h = h / gqa_group;
    int kv_dim = n_kv_heads * head_dim;

    const float* q_head = q + h * head_dim;
    float* out_head = out + h * head_dim;

    int seq_len = pos + 1;
    int start_t = (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0;
    int valid_tokens = seq_len - start_t;

    extern __shared__ float s_mem[];
    // Memory layout:
    // s_scores: valid_tokens floats
    // s_q: head_dim floats
    // s_red: 64 floats (blockDim.x)
    float* s_scores = s_mem;
    float* s_q = s_scores + valid_tokens;
    float* s_red = s_q + head_dim;

    int tid = threadIdx.x;

    // Load Q into shared memory
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = q_head[d];
    }
    __syncthreads();

    // 1. Compute dot products: Q_head . K[t]
    for (int i = tid; i < valid_tokens; i += blockDim.x) {
        int t = start_t + i;
        size_t k_offset = ((size_t)donor_layer * max_seq + t) * kv_dim + kv_h * head_dim;
        const float* k_vec = k_cache + k_offset;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += s_q[d] * k_vec[d];
        }
        dot *= attn_scale;

        if (softcap > 0.0f) {
            dot = softcap * tanhf(dot / softcap);
        }

        s_scores[i] = dot;
    }
    __syncthreads();

    // 2. Softmax: Find max score across valid_tokens
    float local_max = -1e30f;
    for (int i = tid; i < valid_tokens; i += blockDim.x) {
        if (s_scores[i] > local_max) local_max = s_scores[i];
    }
    s_red[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_red[tid + s] > s_red[tid]) s_red[tid] = s_red[tid + s];
        }
        __syncthreads();
    }
    float max_val = s_red[0];

    // 3. Softmax: Exp & sum
    float local_sum = 0.0f;
    for (int i = tid; i < valid_tokens; i += blockDim.x) {
        float ex = expf(s_scores[i] - max_val);
        s_scores[i] = ex;
        local_sum += ex;
    }
    s_red[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_red[tid] += s_red[tid + s];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / (s_red[0] + 1e-9f);

    for (int i = tid; i < valid_tokens; i += blockDim.x) {
        s_scores[i] *= inv_sum;
    }
    __syncthreads();

    // 4. Weighted accumulation: Out[d] = sum_t (score[t] * V[t, d])
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < valid_tokens; ++i) {
            int t = start_t + i;
            size_t v_offset = ((size_t)donor_layer * max_seq + t) * kv_dim + kv_h * head_dim;
            acc += s_scores[i] * v_cache[v_offset + d];
        }
        out_head[d] = acc;
    }
}

extern "C" void cuda_attention_forward(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    float* out,
    int donor_layer,
    int pos,
    int max_seq,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float attn_scale,
    float softcap,
    int sliding_window,
    CudaStream_t stream
) {
    int seq_len = pos + 1;
    int start_t = (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0;
    int valid_tokens = seq_len - start_t;

    int threads = 64;
    size_t shared_bytes = (valid_tokens + head_dim + threads) * sizeof(float);
    k_attention_forward<<<n_heads, threads, shared_bytes, (cudaStream_t)stream>>>(
        q, k_cache, v_cache, out, donor_layer, pos, max_seq, n_heads, n_kv_heads, head_dim, attn_scale, softcap, sliding_window
    );
}

// ============================================================================
// Gemma Per-Layer Embedding Gate & Fusion
// ============================================================================

__global__ void k_ple_gate_gelu(
    const float* __restrict__ ple_gate_in,
    const float* __restrict__ ple_slice,
    float* __restrict__ ple_buf_out,
    int ple_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ple_dim) {
        ple_buf_out[idx] = gelu_f32(ple_gate_in[idx]) * ple_slice[idx];
    }
}

extern "C" void cuda_ple_gate_gelu(
    const float* ple_gate_in,
    const float* ple_slice,
    float* ple_buf_out,
    int ple_dim,
    CudaStream_t stream
) {
    int threads = 256;
    int blocks = (ple_dim + threads - 1) / threads;
    k_ple_gate_gelu<<<blocks, threads, 0, (cudaStream_t)stream>>>(ple_gate_in, ple_slice, ple_buf_out, ple_dim);
}

__global__ void k_ple_ctx_fuse(
    float* __restrict__ ctx_ple_buf,
    const float* __restrict__ ctx_scratch,
    int n,
    int add_token_embd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float scratch = ctx_scratch[idx];
        if (add_token_embd) {
            const float inv_sqrt_2 = 0.70710678118f;
            ctx_ple_buf[idx] = (ctx_ple_buf[idx] + scratch) * inv_sqrt_2;
        } else {
            ctx_ple_buf[idx] = scratch;
        }
    }
}

extern "C" void cuda_ple_ctx_fuse(
    float* ctx_ple_buf,
    const float* ctx_scratch,
    int n,
    int add_token_embd,
    CudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    k_ple_ctx_fuse<<<blocks, threads, 0, (cudaStream_t)stream>>>(ctx_ple_buf, ctx_scratch, n, add_token_embd);
}
