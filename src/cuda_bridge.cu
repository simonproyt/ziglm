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
    cudaError_t err;
    if (stream) {
        err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream);
    } else {
        err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cuda_memcpy_h2d failed (dst=%p, src=%p, bytes=%zu): %s (%d)\n", dst, src, bytes, cudaGetErrorString(err), (int)err);
    }
    return (int)err;
}

extern "C" int cuda_memcpy_d2h(void* dst, const void* src, size_t bytes, CudaStream_t stream) {
    cudaError_t err;
    if (stream) {
        err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    } else {
        err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cuda_memcpy_d2h failed (dst=%p, src=%p, bytes=%zu): %s (%d)\n", dst, src, bytes, cudaGetErrorString(err), (int)err);
    }
    return (int)err;
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
// Fast Hardware SIMD Utilities (DP4A & VSUBSS4)
// ============================================================================

static __device__ __forceinline__ int ziglm_dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const signed char* a8 = (const signed char*)&a;
    const signed char* b8 = (const signed char*)&b;
    return c + (int)a8[0]*(int)b8[0] + (int)a8[1]*(int)b8[1] + (int)a8[2]*(int)b8[2] + (int)a8[3]*(int)b8[3];
#endif
}

static __device__ __forceinline__ int ziglm_vsubss4(int a, int b) {
#if __CUDA_ARCH__ >= 300
    return __vsubss4(a, b);
#else
    const signed char* a8 = (const signed char*)&a;
    const signed char* b8 = (const signed char*)&b;
    int res = 0;
    signed char* r8 = (signed char*)&res;
    r8[0] = (signed char)(a8[0] - b8[0]);
    r8[1] = (signed char)(a8[1] - b8[1]);
    r8[2] = (signed char)(a8[2] - b8[2]);
    r8[3] = (signed char)(a8[3] - b8[3]);
    return res;
#endif
}

static __device__ __forceinline__ int get_int_b2(const void* x, int i32) {
    const unsigned short* x16 = (const unsigned short*)x;
    int x32 = ((int)x16[2 * i32 + 0]) << 0;
    x32 |= ((int)x16[2 * i32 + 1]) << 16;
    return x32;
}

static __device__ __forceinline__ int get_int_b4(const void* x, int i32) {
    return ((const int*)x)[i32];
}

// ============================================================================
// Quantized Activation (Q8_1)
// ============================================================================

struct __align__(4) block_q8_1 {
    half2 ds; // ds.x = d, ds.y = sum
    int8_t qs[32];
};

__device__ block_q8_1 g_q8_1_buf[2048]; // supports up to 65,536 columns

__global__ void k_quantize_q8_1(
    const float* __restrict__ x,
    block_q8_1* __restrict__ y,
    int cols
) {
    int num_blocks = (cols + 31) / 32;
    int b = blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= num_blocks) return;

    int lane = threadIdx.x; // 0..31
    int idx = b * 32 + lane;
    float xi = (idx < cols) ? x[idx] : 0.0f;

    float amax = fabsf(xi);
    float sum = xi;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    float d = amax / 127.0f;
    float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int8_t q = (int8_t)__float2int_rn(xi * id);

    y[b].qs[lane] = q;

    if (lane == 0) {
        y[b].ds = __floats2half2_rn(d, sum);
    }
}

// ============================================================================
// Quantized GEMV Kernels (MMVQ + DP4A Hardware SIMD)
// ============================================================================

// Q4_0 MMVQ: 128 threads (4 warps) per block, 2 rows per block (shared activation loads)
__global__ void k_gemv_q4_0(
    const unsigned char* __restrict__ weights,
    const block_q8_1* __restrict__ y_q8_1,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row0 = blockIdx.x * 2;
    int row1 = row0 + 1;
    if (row0 >= rows) return;

    int tid = threadIdx.y * 32 + threadIdx.x; // 0..127
    int num_blocks = cols / 32;
    size_t row_stride = (size_t)num_blocks * 18;
    const unsigned char* row_w0 = weights + (size_t)row0 * row_stride;
    const unsigned char* row_w1 = (row1 < rows) ? (weights + (size_t)row1 * row_stride) : NULL;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    int kbx_base = tid >> 1;     // 0..63
    int half_block = tid & 1;    // 0 or 1
    int kqs = half_block << 1;   // 0 or 2

    for (int kbx = kbx_base; kbx < num_blocks; kbx += 64) {
        // Load activation block ONCE for both rows
        const block_q8_1* bq8 = y_q8_1 + kbx;
        float2 ds8 = __half22float2(bq8->ds);

        int u0 = get_int_b4(bq8->qs, kqs + 0);
        int u1 = get_int_b4(bq8->qs, kqs + 0 + 4);
        int u2 = get_int_b4(bq8->qs, kqs + 1);
        int u3 = get_int_b4(bq8->qs, kqs + 1 + 4);

        // Row 0
        const unsigned char* b_ptr0 = row_w0 + (size_t)kbx * 18;
        float d4_0 = f16_to_f32(*(const unsigned short*)b_ptr0);
        int v0_0 = get_int_b2(b_ptr0 + 2, kqs + 0);
        int v1_0 = get_int_b2(b_ptr0 + 2, kqs + 1);

        int sumi0 = 0;
        sumi0 = ziglm_dp4a((v0_0 >> 0) & 0x0F0F0F0F, u0, sumi0);
        sumi0 = ziglm_dp4a((v0_0 >> 4) & 0x0F0F0F0F, u1, sumi0);
        sumi0 = ziglm_dp4a((v1_0 >> 0) & 0x0F0F0F0F, u2, sumi0);
        sumi0 = ziglm_dp4a((v1_0 >> 4) & 0x0F0F0F0F, u3, sumi0);
        sum0 += d4_0 * ((float)sumi0 * ds8.x - 4.0f * ds8.y);

        // Row 1
        if (row_w1) {
            const unsigned char* b_ptr1 = row_w1 + (size_t)kbx * 18;
            float d4_1 = f16_to_f32(*(const unsigned short*)b_ptr1);
            int v0_1 = get_int_b2(b_ptr1 + 2, kqs + 0);
            int v1_1 = get_int_b2(b_ptr1 + 2, kqs + 1);

            int sumi1 = 0;
            sumi1 = ziglm_dp4a((v0_1 >> 0) & 0x0F0F0F0F, u0, sumi1);
            sumi1 = ziglm_dp4a((v0_1 >> 4) & 0x0F0F0F0F, u1, sumi1);
            sumi1 = ziglm_dp4a((v1_1 >> 0) & 0x0F0F0F0F, u2, sumi1);
            sumi1 = ziglm_dp4a((v1_1 >> 4) & 0x0F0F0F0F, u3, sumi1);
            sum1 += d4_1 * ((float)sumi1 * ds8.x - 4.0f * ds8.y);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum0 += __shfl_down_sync(0xffffffff, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffff, sum1, offset);
    }

    __shared__ float s_warp_sums0[4];
    __shared__ float s_warp_sums1[4];
    int warp_id = threadIdx.y;
    int lane = threadIdx.x;

    if (lane == 0) {
        s_warp_sums0[warp_id] = sum0;
        s_warp_sums1[warp_id] = sum1;
    }
    __syncthreads();

    if (warp_id == 0 && lane == 0) {
        y[row0] = s_warp_sums0[0] + s_warp_sums0[1] + s_warp_sums0[2] + s_warp_sums0[3];
        if (row1 < rows) {
            y[row1] = s_warp_sums1[0] + s_warp_sums1[1] + s_warp_sums1[2] + s_warp_sums1[3];
        }
    }
}

// Fused GEMV Gate + Up + GEGLU (Q4_0, 2 rows per block, eliminates 2 kernel launches and intermediate DRAM roundtrips)
__global__ void k_gemv_geglu_q4_0(
    const unsigned char* __restrict__ gate_w,
    const unsigned char* __restrict__ up_w,
    const block_q8_1* __restrict__ y_q8_1,
    float* __restrict__ act_out,
    int rows,
    int cols
) {
    int row0 = blockIdx.x * 2;
    int row1 = row0 + 1;
    if (row0 >= rows) return;

    int tid = threadIdx.y * 32 + threadIdx.x; // 0..127
    int num_blocks = cols / 32;
    size_t row_stride = (size_t)num_blocks * 18;
    const unsigned char* rg0 = gate_w + (size_t)row0 * row_stride;
    const unsigned char* ru0 = up_w + (size_t)row0 * row_stride;
    const unsigned char* rg1 = (row1 < rows) ? (gate_w + (size_t)row1 * row_stride) : NULL;
    const unsigned char* ru1 = (row1 < rows) ? (up_w + (size_t)row1 * row_stride) : NULL;

    float sum_g0 = 0.0f, sum_u0 = 0.0f;
    float sum_g1 = 0.0f, sum_u1 = 0.0f;

    int kbx_base = tid >> 1;     // 0..63
    int half_block = tid & 1;    // 0 or 1
    int kqs = half_block << 1;   // 0 or 2

    for (int kbx = kbx_base; kbx < num_blocks; kbx += 64) {
        const block_q8_1* bq8 = y_q8_1 + kbx;
        float2 ds8 = __half22float2(bq8->ds);

        int u0 = get_int_b4(bq8->qs, kqs + 0);
        int u1 = get_int_b4(bq8->qs, kqs + 0 + 4);
        int u2 = get_int_b4(bq8->qs, kqs + 1);
        int u3 = get_int_b4(bq8->qs, kqs + 1 + 4);

        // Row 0 Gate & Up
        const unsigned char* bg0 = rg0 + (size_t)kbx * 18;
        float dg0 = f16_to_f32(*(const unsigned short*)bg0);
        int vg0_0 = get_int_b2(bg0 + 2, kqs + 0);
        int vg0_1 = get_int_b2(bg0 + 2, kqs + 1);

        int sig0 = 0;
        sig0 = ziglm_dp4a((vg0_0 >> 0) & 0x0F0F0F0F, u0, sig0);
        sig0 = ziglm_dp4a((vg0_0 >> 4) & 0x0F0F0F0F, u1, sig0);
        sig0 = ziglm_dp4a((vg0_1 >> 0) & 0x0F0F0F0F, u2, sig0);
        sig0 = ziglm_dp4a((vg0_1 >> 4) & 0x0F0F0F0F, u3, sig0);
        sum_g0 += dg0 * ((float)sig0 * ds8.x - 4.0f * ds8.y);

        const unsigned char* bu0 = ru0 + (size_t)kbx * 18;
        float du0 = f16_to_f32(*(const unsigned short*)bu0);
        int vu0_0 = get_int_b2(bu0 + 2, kqs + 0);
        int vu0_1 = get_int_b2(bu0 + 2, kqs + 1);

        int siu0 = 0;
        siu0 = ziglm_dp4a((vu0_0 >> 0) & 0x0F0F0F0F, u0, siu0);
        siu0 = ziglm_dp4a((vu0_0 >> 4) & 0x0F0F0F0F, u1, siu0);
        siu0 = ziglm_dp4a((vu0_1 >> 0) & 0x0F0F0F0F, u2, siu0);
        siu0 = ziglm_dp4a((vu0_1 >> 4) & 0x0F0F0F0F, u3, siu0);
        sum_u0 += du0 * ((float)siu0 * ds8.x - 4.0f * ds8.y);

        // Row 1 Gate & Up
        if (rg1) {
            const unsigned char* bg1 = rg1 + (size_t)kbx * 18;
            float dg1 = f16_to_f32(*(const unsigned short*)bg1);
            int vg1_0 = get_int_b2(bg1 + 2, kqs + 0);
            int vg1_1 = get_int_b2(bg1 + 2, kqs + 1);

            int sig1 = 0;
            sig1 = ziglm_dp4a((vg1_0 >> 0) & 0x0F0F0F0F, u0, sig1);
            sig1 = ziglm_dp4a((vg1_0 >> 4) & 0x0F0F0F0F, u1, sig1);
            sig1 = ziglm_dp4a((vg1_1 >> 0) & 0x0F0F0F0F, u2, sig1);
            sig1 = ziglm_dp4a((vg1_1 >> 4) & 0x0F0F0F0F, u3, sig1);
            sum_g1 += dg1 * ((float)sig1 * ds8.x - 4.0f * ds8.y);

            const unsigned char* bu1 = ru1 + (size_t)kbx * 18;
            float du1 = f16_to_f32(*(const unsigned short*)bu1);
            int vu1_0 = get_int_b2(bu1 + 2, kqs + 0);
            int vu1_1 = get_int_b2(bu1 + 2, kqs + 1);

            int siu1 = 0;
            siu1 = ziglm_dp4a((vu1_0 >> 0) & 0x0F0F0F0F, u0, siu1);
            siu1 = ziglm_dp4a((vu1_0 >> 4) & 0x0F0F0F0F, u1, siu1);
            siu1 = ziglm_dp4a((vu1_1 >> 0) & 0x0F0F0F0F, u2, siu1);
            siu1 = ziglm_dp4a((vu1_1 >> 4) & 0x0F0F0F0F, u3, siu1);
            sum_u1 += du1 * ((float)siu1 * ds8.x - 4.0f * ds8.y);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_g0 += __shfl_down_sync(0xffffffff, sum_g0, offset);
        sum_u0 += __shfl_down_sync(0xffffffff, sum_u0, offset);
        sum_g1 += __shfl_down_sync(0xffffffff, sum_g1, offset);
        sum_u1 += __shfl_down_sync(0xffffffff, sum_u1, offset);
    }

    __shared__ float s_g0[4], s_u0[4], s_g1[4], s_u1[4];
    int warp_id = threadIdx.y;
    int lane = threadIdx.x;

    if (lane == 0) {
        s_g0[warp_id] = sum_g0;
        s_u0[warp_id] = sum_u0;
        s_g1[warp_id] = sum_g1;
        s_u1[warp_id] = sum_u1;
    }
    __syncthreads();

    if (warp_id == 0 && lane == 0) {
        float g0 = s_g0[0] + s_g0[1] + s_g0[2] + s_g0[3];
        float u0 = s_u0[0] + s_u0[1] + s_u0[2] + s_u0[3];
        float gelu0 = 0.5f * g0 * (1.0f + tanhf(0.7978845608f * (g0 + 0.044715f * g0 * g0 * g0)));
        act_out[row0] = gelu0 * u0;

        if (row1 < rows) {
            float g1 = s_g1[0] + s_g1[1] + s_g1[2] + s_g1[3];
            float u1 = s_u1[0] + s_u1[1] + s_u1[2] + s_u1[3];
            float gelu1 = 0.5f * g1 * (1.0f + tanhf(0.7978845608f * (g1 + 0.044715f * g1 * g1 * g1)));
            act_out[row1] = gelu1 * u1;
        }
    }
}

// Fused GEMV QKV (Q4_0, computes Q, K, V projections in 1 kernel launch with 2 rows per block)
__global__ void k_gemv_qkv_q4_0(
    const unsigned char* __restrict__ q_w,
    const unsigned char* __restrict__ k_w,
    const unsigned char* __restrict__ v_w,
    const block_q8_1* __restrict__ y_q8_1,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    int q_rows,
    int k_rows,
    int v_rows,
    int cols
) {
    int total_rows = q_rows + k_rows + v_rows;
    int row0 = blockIdx.x * 2;
    int row1 = row0 + 1;
    if (row0 >= total_rows) return;

    int tid = threadIdx.y * 32 + threadIdx.x; // 0..127
    int num_blocks = cols / 32;
    size_t row_stride = (size_t)num_blocks * 18;

    const unsigned char* r_w0 = NULL;
    float* out_ptr0 = NULL;
    if (row0 < q_rows) {
        r_w0 = q_w + (size_t)row0 * row_stride;
        out_ptr0 = q_out + row0;
    } else if (row0 < q_rows + k_rows) {
        int r = row0 - q_rows;
        r_w0 = k_w + (size_t)r * row_stride;
        out_ptr0 = k_out + r;
    } else {
        int r = row0 - q_rows - k_rows;
        r_w0 = v_w + (size_t)r * row_stride;
        out_ptr0 = v_out + r;
    }

    const unsigned char* r_w1 = NULL;
    float* out_ptr1 = NULL;
    if (row1 < total_rows) {
        if (row1 < q_rows) {
            r_w1 = q_w + (size_t)row1 * row_stride;
            out_ptr1 = q_out + row1;
        } else if (row1 < q_rows + k_rows) {
            int r = row1 - q_rows;
            r_w1 = k_w + (size_t)r * row_stride;
            out_ptr1 = k_out + r;
        } else {
            int r = row1 - q_rows - k_rows;
            r_w1 = v_w + (size_t)r * row_stride;
            out_ptr1 = v_out + r;
        }
    }

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    int kbx_base = tid >> 1;     // 0..63
    int half_block = tid & 1;    // 0 or 1
    int kqs = half_block << 1;   // 0 or 2

    for (int kbx = kbx_base; kbx < num_blocks; kbx += 64) {
        const block_q8_1* bq8 = y_q8_1 + kbx;
        float2 ds8 = __half22float2(bq8->ds);

        int u0 = get_int_b4(bq8->qs, kqs + 0);
        int u1 = get_int_b4(bq8->qs, kqs + 0 + 4);
        int u2 = get_int_b4(bq8->qs, kqs + 1);
        int u3 = get_int_b4(bq8->qs, kqs + 1 + 4);

        // Row 0
        const unsigned char* b0 = r_w0 + (size_t)kbx * 18;
        float d4_0 = f16_to_f32(*(const unsigned short*)b0);
        int v0_0 = get_int_b2(b0 + 2, kqs + 0);
        int v1_0 = get_int_b2(b0 + 2, kqs + 1);

        int sumi0 = 0;
        sumi0 = ziglm_dp4a((v0_0 >> 0) & 0x0F0F0F0F, u0, sumi0);
        sumi0 = ziglm_dp4a((v0_0 >> 4) & 0x0F0F0F0F, u1, sumi0);
        sumi0 = ziglm_dp4a((v1_0 >> 0) & 0x0F0F0F0F, u2, sumi0);
        sumi0 = ziglm_dp4a((v1_0 >> 4) & 0x0F0F0F0F, u3, sumi0);
        sum0 += d4_0 * ((float)sumi0 * ds8.x - 4.0f * ds8.y);

        // Row 1
        if (r_w1) {
            const unsigned char* b1 = r_w1 + (size_t)kbx * 18;
            float d4_1 = f16_to_f32(*(const unsigned short*)b1);
            int v0_1 = get_int_b2(b1 + 2, kqs + 0);
            int v1_1 = get_int_b2(b1 + 2, kqs + 1);

            int sumi1 = 0;
            sumi1 = ziglm_dp4a((v0_1 >> 0) & 0x0F0F0F0F, u0, sumi1);
            sumi1 = ziglm_dp4a((v0_1 >> 4) & 0x0F0F0F0F, u1, sumi1);
            sumi1 = ziglm_dp4a((v1_1 >> 0) & 0x0F0F0F0F, u2, sumi1);
            sumi1 = ziglm_dp4a((v1_1 >> 4) & 0x0F0F0F0F, u3, sumi1);
            sum1 += d4_1 * ((float)sumi1 * ds8.x - 4.0f * ds8.y);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum0 += __shfl_down_sync(0xffffffff, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffff, sum1, offset);
    }

    __shared__ float s_warp_sums0[4];
    __shared__ float s_warp_sums1[4];
    int warp_id = threadIdx.y;
    int lane = threadIdx.x;

    if (lane == 0) {
        s_warp_sums0[warp_id] = sum0;
        s_warp_sums1[warp_id] = sum1;
    }
    __syncthreads();

    if (warp_id == 0 && lane == 0) {
        *out_ptr0 = s_warp_sums0[0] + s_warp_sums0[1] + s_warp_sums0[2] + s_warp_sums0[3];
        if (out_ptr1) {
            *out_ptr1 = s_warp_sums1[0] + s_warp_sums1[1] + s_warp_sums1[2] + s_warp_sums1[3];
        }
    }
}

// Q8_0 MMVQ: 128 threads (4 warps) per row, DP4A SIMD
__global__ void k_gemv_q8_0(
    const unsigned char* __restrict__ weights,
    const block_q8_1* __restrict__ y_q8_1,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.y * 32 + threadIdx.x; // 0..127
    int num_blocks = cols / 32;
    const unsigned char* row_w = weights + (size_t)row * num_blocks * 34;

    float sum = 0.0f;

    int kbx_base = tid >> 2;     // 0..31
    int iqs = (tid & 3) << 1;    // 0, 2, 4, 6

    for (int kbx = kbx_base; kbx < num_blocks; kbx += 32) {
        const unsigned char* b_ptr = row_w + (size_t)kbx * 34;
        float d8_0 = f16_to_f32(*(const unsigned short*)b_ptr);

        const block_q8_1* bq8 = y_q8_1 + kbx;
        float d8_1 = __low2float(bq8->ds);

        int v0 = get_int_b4(b_ptr + 2, iqs + 0);
        int v1 = get_int_b4(b_ptr + 2, iqs + 1);

        int u0 = get_int_b4(bq8->qs, iqs + 0);
        int u1 = get_int_b4(bq8->qs, iqs + 1);

        int sumi = 0;
        sumi = ziglm_dp4a(v0, u0, sumi);
        sumi = ziglm_dp4a(v1, u1, sumi);

        sum += d8_0 * d8_1 * (float)sumi;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float s_warp_sums[4];
    int warp_id = threadIdx.y;
    int lane = threadIdx.x;

    if (lane == 0) {
        s_warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0 && lane == 0) {
        y[row] = s_warp_sums[0] + s_warp_sums[1] + s_warp_sums[2] + s_warp_sums[3];
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

// Q6_K MMVQ: 64 threads (2 warps) per block, 2 rows per block (shared activation loads)
__global__ void k_gemv_q6_k(
    const unsigned char* __restrict__ weights,
    const block_q8_1* __restrict__ y_q8_1,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row0 = blockIdx.x * 2;
    int row1 = row0 + 1;
    if (row0 >= rows) return;

    int tid = threadIdx.y * 32 + threadIdx.x; // 0..63
    int num_superblocks = cols / 256;
    size_t row_stride = (size_t)num_superblocks * 210;
    const unsigned char* row_w0 = weights + (size_t)row0 * row_stride;
    const unsigned char* row_w1 = (row1 < rows) ? (weights + (size_t)row1 * row_stride) : NULL;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    int kbx_base = tid >> 5;     // 0 for warp 0, 1 for warp 1
    int iqs = tid & 31;          // 0..31

    for (int kbx = kbx_base; kbx < num_superblocks; kbx += 2) {
        const block_q8_1* bq8_1_base = y_q8_1 + (size_t)kbx * 8; // 256 weights = 8 Q8_1 blocks

        const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);

        int u[2];
        float d8[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[i] = get_int_b4(bq8_1_base[bq8_offset + 2 * i].qs, iqs % 8);
            d8[i] = __low2float(bq8_1_base[bq8_offset + 2 * i].ds);
        }

        // Row 0
        const unsigned char* sb_ptr0 = row_w0 + (size_t)kbx * 210;
        const unsigned char* ql0 = sb_ptr0;
        const unsigned char* qh0 = sb_ptr0 + 128;
        const signed char* scales0 = (const signed char*)(sb_ptr0 + 192) + scale_offset;
        float d0 = f16_to_f32(*(const unsigned short*)(sb_ptr0 + 208));

        int vl0 = get_int_b2(ql0, iqs);
        int vh0 = get_int_b2(qh0, 8 * (iqs / 16) + (iqs % 8)) >> vh_shift;

        float sumf0 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int sc = (int)scales0[4 * i];
            int vil = (vl0 >> (4 * i)) & 0x0F0F0F0F;
            int vih = ((vh0 >> (4 * i)) << 4) & 0x30303030;
            int vi = ziglm_vsubss4(vil | vih, 0x20202020);
            sumf0 += d8[i] * ((float)ziglm_dp4a(vi, u[i], 0) * (float)sc);
        }
        sum0 += d0 * sumf0;

        // Row 1
        if (row_w1) {
            const unsigned char* sb_ptr1 = row_w1 + (size_t)kbx * 210;
            const unsigned char* ql1 = sb_ptr1;
            const unsigned char* qh1 = sb_ptr1 + 128;
            const signed char* scales1 = (const signed char*)(sb_ptr1 + 192) + scale_offset;
            float d1 = f16_to_f32(*(const unsigned short*)(sb_ptr1 + 208));

            int vl1 = get_int_b2(ql1, iqs);
            int vh1 = get_int_b2(qh1, 8 * (iqs / 16) + (iqs % 8)) >> vh_shift;

            float sumf1 = 0.0f;
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int sc = (int)scales1[4 * i];
                int vil = (vl1 >> (4 * i)) & 0x0F0F0F0F;
                int vih = ((vh1 >> (4 * i)) << 4) & 0x30303030;
                int vi = ziglm_vsubss4(vil | vih, 0x20202020);
                sumf1 += d8[i] * ((float)ziglm_dp4a(vi, u[i], 0) * (float)sc);
            }
            sum1 += d1 * sumf1;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum0 += __shfl_down_sync(0xffffffff, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffff, sum1, offset);
    }

    __shared__ float s_warp_sums0[2];
    __shared__ float s_warp_sums1[2];
    int warp_id = threadIdx.y;
    int lane = threadIdx.x;

    if (lane == 0) {
        s_warp_sums0[warp_id] = sum0;
        s_warp_sums1[warp_id] = sum1;
    }
    __syncthreads();

    if (warp_id == 0 && lane == 0) {
        y[row0] = s_warp_sums0[0] + s_warp_sums0[1];
        if (row1 < rows) {
            y[row1] = s_warp_sums1[0] + s_warp_sums1[1];
        }
    }
}

// F16 GEMV (4 rows per warp, 32-bit coalesced loads)
__global__ void k_gemv_f16(
    const unsigned short* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    int row_base = (blockIdx.x * blockDim.y + threadIdx.y) * 4;
    if (row_base >= rows) return;

    int lane = threadIdx.x;

    const unsigned short* r_w[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        r_w[r] = (row_base + r < rows) ? (weights + (size_t)(row_base + r) * cols) : NULL;
    }

    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int col_step = 64;
    int col_base = lane * 2;
    for (int col = col_base; col < cols; col += col_step) {
        float2 x_val = (col + 1 < cols) ? *(const float2*)(x + col) : make_float2(x[col], 0.0f);
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            if (r_w[r]) {
                unsigned int w_pair = (col + 1 < cols) ? *(const unsigned int*)(r_w[r] + col) : (unsigned int)r_w[r][col];
                float w0 = f16_to_f32((unsigned short)(w_pair & 0xFFFF));
                float w1 = (col + 1 < cols) ? f16_to_f32((unsigned short)(w_pair >> 16)) : 0.0f;
                sum[r] += w0 * x_val.x + w1 * x_val.y;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum[r] += __shfl_down_sync(0xffffffff, sum[r], offset);
        }
        if (lane == 0 && row_base + r < rows) {
            y[row_base + r] = sum[r];
        }
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

// GEMV Host Dispatchers
extern "C" void cuda_gemv_q4_0(const void* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    int num_q8_blocks = (cols + 31) / 32;
    int q_threads = 256;
    int q_blocks = (num_q8_blocks + (q_threads / 32) - 1) / (q_threads / 32);
    dim3 q_grid(q_blocks);
    dim3 q_dim(32, q_threads / 32);
    k_quantize_q8_1<<<q_grid, q_dim, 0, (cudaStream_t)stream>>>(x, g_q8_1_buf, cols);

    dim3 block(32, 4);
    dim3 grid((rows + 1) / 2);
    k_gemv_q4_0<<<grid, block, 0, (cudaStream_t)stream>>>((const unsigned char*)weights, g_q8_1_buf, y, rows, cols);
}

extern "C" void cuda_gemv_q8_0(const void* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    int num_q8_blocks = (cols + 31) / 32;
    int q_threads = 256;
    int q_blocks = (num_q8_blocks + (q_threads / 32) - 1) / (q_threads / 32);
    dim3 q_grid(q_blocks);
    dim3 q_dim(32, q_threads / 32);
    k_quantize_q8_1<<<q_grid, q_dim, 0, (cudaStream_t)stream>>>(x, g_q8_1_buf, cols);

    dim3 block(32, 4);
    dim3 grid(rows);
    k_gemv_q8_0<<<grid, block, 0, (cudaStream_t)stream>>>((const unsigned char*)weights, g_q8_1_buf, y, rows, cols);
}

extern "C" void cuda_gemv_q4_k(const void* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((rows + 7) / 8);
    k_gemv_q4_k<<<grid, block, 0, (cudaStream_t)stream>>>((const unsigned char*)weights, x, y, rows, cols);
}

extern "C" void cuda_gemv_q6_k(const void* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    int num_q8_blocks = (cols + 31) / 32;
    int q_threads = 256;
    int q_blocks = (num_q8_blocks + (q_threads / 32) - 1) / (q_threads / 32);
    dim3 q_grid(q_blocks);
    dim3 q_dim(32, q_threads / 32);
    k_quantize_q8_1<<<q_grid, q_dim, 0, (cudaStream_t)stream>>>(x, g_q8_1_buf, cols);

    dim3 block(32, 2);
    dim3 grid((rows + 1) / 2);
    k_gemv_q6_k<<<grid, block, 0, (cudaStream_t)stream>>>((const unsigned char*)weights, g_q8_1_buf, y, rows, cols);
}

extern "C" void cuda_gemv_geglu_q4_0(
    const void* gate_w,
    const void* up_w,
    const float* x,
    float* act_out,
    int rows,
    int cols,
    CudaStream_t stream
) {
    if (rows <= 0 || cols <= 0) return;
    int num_q8_blocks = (cols + 31) / 32;
    int q_threads = 256;
    int q_blocks = (num_q8_blocks + (q_threads / 32) - 1) / (q_threads / 32);
    dim3 q_grid(q_blocks);
    dim3 q_dim(32, q_threads / 32);
    k_quantize_q8_1<<<q_grid, q_dim, 0, (cudaStream_t)stream>>>(x, g_q8_1_buf, cols);

    dim3 block(32, 4);
    dim3 grid((rows + 1) / 2);
    k_gemv_geglu_q4_0<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)gate_w,
        (const unsigned char*)up_w,
        g_q8_1_buf,
        act_out,
        rows,
        cols
    );
}

extern "C" void cuda_gemv_qkv_q4_0(
    const void* q_w,
    const void* k_w,
    const void* v_w,
    const float* x,
    float* q_out,
    float* k_out,
    float* v_out,
    int q_rows,
    int k_rows,
    int v_rows,
    int cols,
    CudaStream_t stream
) {
    int total_rows = q_rows + k_rows + v_rows;
    if (total_rows <= 0 || cols <= 0) return;

    int num_q8_blocks = (cols + 31) / 32;
    int q_threads = 256;
    int q_blocks = (num_q8_blocks + (q_threads / 32) - 1) / (q_threads / 32);
    dim3 q_grid(q_blocks);
    dim3 q_dim(32, q_threads / 32);
    k_quantize_q8_1<<<q_grid, q_dim, 0, (cudaStream_t)stream>>>(x, g_q8_1_buf, cols);

    dim3 block(32, 4);
    dim3 grid((total_rows + 1) / 2);
    k_gemv_qkv_q4_0<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)q_w,
        (const unsigned char*)k_w,
        (const unsigned char*)v_w,
        g_q8_1_buf,
        q_out,
        k_out,
        v_out,
        q_rows,
        k_rows,
        v_rows,
        cols
    );
}

extern "C" void cuda_gemv_f16(const void* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((rows + 31) / 32);
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

extern "C" void cuda_gemv(int qtype, const void* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream) {
    if (qtype == 2) {
        cuda_gemv_q4_0(weights, x, y, rows, cols, stream);
    } else if (qtype == 8) {
        cuda_gemv_q8_0(weights, x, y, rows, cols, stream);
    } else if (qtype == 12) {
        cuda_gemv_q4_k(weights, x, y, rows, cols, stream);
    } else if (qtype == 14) {
        cuda_gemv_q6_k(weights, x, y, rows, cols, stream);
    } else if (qtype == 1) {
        cuda_gemv_f16(weights, x, y, rows, cols, stream);
    } else if (qtype == 30) {
        cuda_gemv_bf16(weights, x, y, rows, cols, stream);
    } else {
        cuda_gemv_f32((const float*)weights, x, y, rows, cols, stream);
    }
}

// ============================================================================
// Batched GEMM Operations (Matrix * Matrix)
// ============================================================================

__global__ void k_gemm_q4_0(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int batch_size,
    int rows,
    int cols
) {
    int row_base = blockIdx.x * 2;
    int b_base = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    if (b_base >= batch_size) return;

    int lane = threadIdx.x;
    int num_blocks = cols / 32;
    int total_bytes = num_blocks * 16;

    const unsigned char* row0_weights = (row_base < rows) ? (weights + (size_t)row_base * num_blocks * 18) : NULL;
    const unsigned char* row1_weights = (row_base + 1 < rows) ? (weights + (size_t)(row_base + 1) * num_blocks * 18) : NULL;

    float sum0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sum1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k = lane; k < total_bytes; k += 32) {
        int blk = k / 16;
        int i = k % 16;

        float x_vals[8];
        #pragma unroll
        for (int bi = 0; bi < 4; bi++) {
            if (b_base + bi < batch_size) {
                const float* x_block = X + (size_t)(b_base + bi) * cols + blk * 32;
                x_vals[bi * 2 + 0] = x_block[i];
                x_vals[bi * 2 + 1] = x_block[i + 16];
            }
        }

        if (row0_weights) {
            const unsigned char* block_ptr = row0_weights + blk * 18;
            float d0 = f16_to_f32(*(const unsigned short*)block_ptr);
            unsigned char byte0 = block_ptr[2 + i];
            int q0_0 = (int)(byte0 & 0x0F) - 8;
            int q0_1 = (int)(byte0 >> 4) - 8;
            #pragma unroll
            for (int bi = 0; bi < 4; bi++) {
                if (b_base + bi < batch_size) {
                    sum0[bi] += d0 * ((float)q0_0 * x_vals[bi * 2 + 0] + (float)q0_1 * x_vals[bi * 2 + 1]);
                }
            }
        }

        if (row1_weights) {
            const unsigned char* block_ptr = row1_weights + blk * 18;
            float d1 = f16_to_f32(*(const unsigned short*)block_ptr);
            unsigned char byte1 = block_ptr[2 + i];
            int q1_0 = (int)(byte1 & 0x0F) - 8;
            int q1_1 = (int)(byte1 >> 4) - 8;
            #pragma unroll
            for (int bi = 0; bi < 4; bi++) {
                if (b_base + bi < batch_size) {
                    sum1[bi] += d1 * ((float)q1_0 * x_vals[bi * 2 + 0] + (float)q1_1 * x_vals[bi * 2 + 1]);
                }
            }
        }
    }

    #pragma unroll
    for (int bi = 0; bi < 4; bi++) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum0[bi] += __shfl_down_sync(0xffffffff, sum0[bi], offset);
            sum1[bi] += __shfl_down_sync(0xffffffff, sum1[bi], offset);
        }

        if (lane == 0 && (b_base + bi < batch_size)) {
            if (row_base < rows) {
                Y[(size_t)(b_base + bi) * rows + row_base] = sum0[bi];
            }
            if (row_base + 1 < rows) {
                Y[(size_t)(b_base + bi) * rows + row_base + 1] = sum1[bi];
            }
        }
    }
}

__global__ void k_gemm_q8_0(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int batch_size,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int b_base = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    if (b_base >= batch_size) return;

    int lane = threadIdx.x;
    int num_blocks = cols / 32;
    const unsigned char* row_weights = weights + (size_t)row * num_blocks * 34;

    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int blk = lane; blk < num_blocks; blk += 32) {
        const unsigned char* block_ptr = row_weights + blk * 34;
        float d = f16_to_f32(*(const unsigned short*)block_ptr);
        const signed char* qs = (const signed char*)(block_ptr + 2);

        #pragma unroll
        for (int bi = 0; bi < 4; bi++) {
            if (b_base + bi < batch_size) {
                const float* x_blk = X + (size_t)(b_base + bi) * cols + blk * 32;
                float local_sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    local_sum += (float)qs[i] * x_blk[i];
                }
                sum[bi] += d * local_sum;
            }
        }
    }

    #pragma unroll
    for (int bi = 0; bi < 4; bi++) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum[bi] += __shfl_down_sync(0xffffffff, sum[bi], offset);
        }
        if (lane == 0 && (b_base + bi < batch_size)) {
            Y[(size_t)(b_base + bi) * rows + row] = sum[bi];
        }
    }
}

__global__ void k_gemm_f16(
    const unsigned short* __restrict__ weights,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int batch_size,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int b_base = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    if (b_base >= batch_size) return;

    int lane = threadIdx.x;
    const unsigned short* row_data = weights + (size_t)row * cols;

    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int col = lane; col < cols; col += 32) {
        float w = f16_to_f32(row_data[col]);
        #pragma unroll
        for (int bi = 0; bi < 4; bi++) {
            if (b_base + bi < batch_size) {
                sum[bi] += w * X[(size_t)(b_base + bi) * cols + col];
            }
        }
    }

    #pragma unroll
    for (int bi = 0; bi < 4; bi++) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum[bi] += __shfl_down_sync(0xffffffff, sum[bi], offset);
        }
        if (lane == 0 && (b_base + bi < batch_size)) {
            Y[(size_t)(b_base + bi) * rows + row] = sum[bi];
        }
    }
}

extern "C" void cuda_gemm_q4_0(
    const void* weights,
    const float* x,
    float* y,
    int batch_size,
    int rows,
    int cols,
    CudaStream_t stream
) {
    if (batch_size <= 0 || rows <= 0 || cols <= 0) return;
    if (batch_size == 1) {
        cuda_gemv_q4_0(weights, x, y, rows, cols, stream);
        return;
    }
    dim3 block(32, 4);
    dim3 grid((rows + 1) / 2, (batch_size + 15) / 16);
    k_gemm_q4_0<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)weights, x, y, batch_size, rows, cols
    );
}

extern "C" void cuda_gemm_q8_0(
    const void* weights,
    const float* x,
    float* y,
    int batch_size,
    int rows,
    int cols,
    CudaStream_t stream
) {
    if (batch_size <= 0 || rows <= 0 || cols <= 0) return;
    if (batch_size == 1) {
        cuda_gemv_q8_0(weights, x, y, rows, cols, stream);
        return;
    }
    dim3 block(32, 4);
    dim3 grid(rows, (batch_size + 15) / 16);
    k_gemm_q8_0<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const unsigned char*)weights, x, y, batch_size, rows, cols
    );
}

extern "C" void cuda_gemm_f16(
    const void* weights,
    const float* x,
    float* y,
    int batch_size,
    int rows,
    int cols,
    CudaStream_t stream
) {
    if (batch_size <= 0 || rows <= 0 || cols <= 0) return;
    if (batch_size == 1) {
        cuda_gemv_f16(weights, x, y, rows, cols, stream);
        return;
    }
    dim3 block(32, 4);
    dim3 grid(rows, (batch_size + 15) / 16);
    k_gemm_f16<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const unsigned short*)weights, x, y, batch_size, rows, cols
    );
}

extern "C" void cuda_gemm(
    int qtype,
    const void* weights,
    const float* x,
    float* y,
    int batch_size,
    int rows,
    int cols,
    CudaStream_t stream
) {
    if (batch_size <= 0 || rows <= 0 || cols <= 0) return;
    if (batch_size == 1) {
        cuda_gemv(qtype, weights, x, y, rows, cols, stream);
        return;
    }
    if (qtype == 2) { // Q4_0
        cuda_gemm_q4_0(weights, x, y, batch_size, rows, cols, stream);
    } else if (qtype == 8) { // Q8_0
        cuda_gemm_q8_0(weights, x, y, batch_size, rows, cols, stream);
    } else if (qtype == 1) { // F16
        cuda_gemm_f16(weights, x, y, batch_size, rows, cols, stream);
    } else {
        // Safe fallback for other types: run GEMV for each vector in batch
        for (int b = 0; b < batch_size; b++) {
            cuda_gemv(qtype, weights, x + (size_t)b * cols, y + (size_t)b * rows, rows, cols, stream);
        }
    }
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

__global__ void k_rmsnorm_batched(
    float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int head_dim,
    int count,
    float eps,
    int use_unit_offset
) {
    int h = blockIdx.x;
    if (h >= count) return;

    int tid = threadIdx.x;
    float* x_head = x + (size_t)h * head_dim;
    float* out_head = out + (size_t)h * head_dim;

    __shared__ float s_sum[256];
    float local_sum = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = x_head[i];
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

    float mean = s_sum[0] / (float)head_dim;
    float inv_std = rsqrtf(mean + eps);

    for (int i = tid; i < head_dim; i += blockDim.x) {
        float w = (weight != NULL) ? (weight[i] + (use_unit_offset ? 1.0f : 0.0f)) : 1.0f;
        out_head[i] = x_head[i] * inv_std * w;
    }
}

extern "C" void cuda_rmsnorm_batched(
    float* x,
    const float* weight,
    float* out,
    int head_dim,
    int count,
    float eps,
    int use_unit_offset,
    CudaStream_t stream
) {
    if (count <= 0 || head_dim <= 0) return;
    int threads = (head_dim < 256) ? 32 * ((head_dim + 31) / 32) : 256;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    k_rmsnorm_batched<<<count, threads, 0, (cudaStream_t)stream>>>(
        x, weight, out, head_dim, count, eps, use_unit_offset
    );
}

// ============================================================================
// Embedding Lookup Kernels
// ============================================================================

__global__ void k_embed_lookup_q4_0(
    const unsigned char* __restrict__ weights,
    int token_id,
    float* __restrict__ out,
    int dim,
    float scale
) {
    int num_blocks = dim / 32;
    size_t row_offset = (size_t)token_id * (size_t)num_blocks * 18;
    const unsigned char* row_weights = weights + row_offset;

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < num_blocks) {
        const unsigned char* block_ptr = row_weights + b * 18;
        float d = f16_to_f32(*(const unsigned short*)block_ptr) * scale;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            unsigned char byte = block_ptr[2 + i];
            int q0 = (int)(byte & 0x0F) - 8;
            int q1 = (int)(byte >> 4) - 8;
            out[b * 32 + i] = (float)q0 * d;
            out[b * 32 + i + 16] = (float)q1 * d;
        }
    }
}

__global__ void k_embed_lookup_q8_0(
    const unsigned char* __restrict__ weights,
    int token_id,
    float* __restrict__ out,
    int dim,
    float scale
) {
    int num_blocks = dim / 32;
    size_t row_offset = (size_t)token_id * (size_t)num_blocks * 34;
    const unsigned char* row_weights = weights + row_offset;

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < num_blocks) {
        const unsigned char* block_ptr = row_weights + b * 34;
        float d = f16_to_f32(*(const unsigned short*)block_ptr) * scale;
        const signed char* qs = (const signed char*)(block_ptr + 2);
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            out[b * 32 + i] = (float)qs[i] * d;
        }
    }
}

__global__ void k_embed_lookup_f32(
    const float* __restrict__ weights,
    int token_id,
    float* __restrict__ out,
    int dim,
    float scale
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        out[i] = weights[(size_t)token_id * dim + i] * scale;
    }
}

__global__ void k_embed_lookup_bf16(
    const unsigned short* __restrict__ weights,
    int token_id,
    float* __restrict__ out,
    int dim,
    float scale
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        out[i] = bf16_to_f32(weights[(size_t)token_id * dim + i]) * scale;
    }
}

__global__ void k_embed_lookup_f16(
    const unsigned short* __restrict__ weights,
    int token_id,
    float* __restrict__ out,
    int dim,
    float scale
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        out[i] = f16_to_f32(weights[(size_t)token_id * dim + i]) * scale;
    }
}

__global__ void k_embed_lookup_q6_k(
    const unsigned char* __restrict__ weights,
    int token_id,
    float* __restrict__ out,
    int dim,
    float scale
) {
    int num_sb = dim / 256;
    int sb = blockIdx.x;
    if (sb >= num_sb) return;

    size_t row_offset = (size_t)token_id * (size_t)num_sb * 210;
    const unsigned char* block = weights + row_offset + (size_t)sb * 210;
    float d = f16_to_f32(*(const unsigned short*)(block + 208)) * scale;

    int tid = threadIdx.x; // 0..63 (64 threads per superblock)
    int n = tid / 32;      // 0 or 1
    int l = tid % 32;      // 0..31
    int is = l / 16;

    const unsigned char* ql = block + 0 + n * 64;
    const unsigned char* qh = block + 128 + n * 32;
    const signed char* scales = (const signed char*)(block + 192 + n * 8);

    unsigned char ql_l = ql[l];
    unsigned char ql_l32 = ql[l + 32];
    unsigned char qh_l = qh[l];

    int q1 = (int)((ql_l & 0x0F) | (((qh_l >> 0) & 3) << 4)) - 32;
    int q2 = (int)((ql_l32 & 0x0F) | (((qh_l >> 2) & 3) << 4)) - 32;
    int q3 = (int)((ql_l >> 4) | (((qh_l >> 4) & 3) << 4)) - 32;
    int q4 = (int)((ql_l32 >> 4) | (((qh_l >> 6) & 3) << 4)) - 32;

    float* dst = out + (size_t)sb * 256 + (size_t)n * 128;
    dst[l + 0]  = d * (float)scales[is + 0] * (float)q1;
    dst[l + 32] = d * (float)scales[is + 2] * (float)q2;
    dst[l + 64] = d * (float)scales[is + 4] * (float)q3;
    dst[l + 96] = d * (float)scales[is + 6] * (float)q4;
}

extern "C" void cuda_embed_lookup(
    const void* emb_weights,
    int qtype,
    int token_id,
    float* out,
    int dim,
    float scale,
    CudaStream_t stream
) {
    if (!emb_weights || dim <= 0) return;
    if (qtype == 2) { // Q4_0
        int num_blocks = dim / 32;
        int threads = (num_blocks < 256) ? 32 * ((num_blocks + 31) / 32) : 256;
        if (threads < 32) threads = 32;
        int blocks = (num_blocks + threads - 1) / threads;
        k_embed_lookup_q4_0<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            (const unsigned char*)emb_weights, token_id, out, dim, scale
        );
    } else if (qtype == 8) { // Q8_0
        int num_blocks = dim / 32;
        int threads = (num_blocks < 256) ? 32 * ((num_blocks + 31) / 32) : 256;
        if (threads < 32) threads = 32;
        int blocks = (num_blocks + threads - 1) / threads;
        k_embed_lookup_q8_0<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            (const unsigned char*)emb_weights, token_id, out, dim, scale
        );
    } else if (qtype == 14) { // Q6_K
        int num_sb = dim / 256;
        k_embed_lookup_q6_k<<<num_sb, 64, 0, (cudaStream_t)stream>>>(
            (const unsigned char*)emb_weights, token_id, out, dim, scale
        );
    } else if (qtype == 30) { // BF16
        int threads = 256;
        int blocks = (dim + threads - 1) / threads;
        k_embed_lookup_bf16<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            (const unsigned short*)emb_weights, token_id, out, dim, scale
        );
    } else if (qtype == 1) { // F16
        int threads = 256;
        int blocks = (dim + threads - 1) / threads;
        k_embed_lookup_f16<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            (const unsigned short*)emb_weights, token_id, out, dim, scale
        );
    } else if (qtype == 0) { // F32
        int threads = 256;
        int blocks = (dim + threads - 1) / threads;
        k_embed_lookup_f32<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            (const float*)emb_weights, token_id, out, dim, scale
        );
    }
}

__global__ void k_embed_lookup_batch_q4_0(
    const unsigned char* __restrict__ weights,
    const int* __restrict__ token_ids,
    float* __restrict__ out,
    int n_tokens,
    int dim,
    float scale
) {
    int num_blocks = dim / 32;
    int global_b = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_b < n_tokens * num_blocks) {
        int t = global_b / num_blocks;
        int b = global_b % num_blocks;
        int token_id = token_ids[t];
        size_t row_offset = (size_t)token_id * (size_t)num_blocks * 18;
        const unsigned char* block_ptr = weights + row_offset + (size_t)b * 18;
        float d = f16_to_f32(*(const unsigned short*)block_ptr) * scale;
        float* out_ptr = out + (size_t)t * dim + (size_t)b * 32;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            unsigned char byte = block_ptr[2 + i];
            int q0 = (int)(byte & 0x0F) - 8;
            int q1 = (int)(byte >> 4) - 8;
            out_ptr[i] = (float)q0 * d;
            out_ptr[i + 16] = (float)q1 * d;
        }
    }
}

__global__ void k_embed_lookup_batch_q8_0(
    const unsigned char* __restrict__ weights,
    const int* __restrict__ token_ids,
    float* __restrict__ out,
    int n_tokens,
    int dim,
    float scale
) {
    int num_blocks = dim / 32;
    int global_b = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_b < n_tokens * num_blocks) {
        int t = global_b / num_blocks;
        int b = global_b % num_blocks;
        int token_id = token_ids[t];
        size_t row_offset = (size_t)token_id * (size_t)num_blocks * 34;
        const unsigned char* block_ptr = weights + row_offset + (size_t)b * 34;
        float d = f16_to_f32(*(const unsigned short*)block_ptr) * scale;
        const signed char* qs = (const signed char*)(block_ptr + 2);
        float* out_ptr = out + (size_t)t * dim + (size_t)b * 32;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            out_ptr[i] = (float)qs[i] * d;
        }
    }
}

__global__ void k_embed_lookup_batch_bf16(
    const unsigned short* __restrict__ weights,
    const int* __restrict__ token_ids,
    float* __restrict__ out,
    int n_tokens,
    int dim,
    float scale
) {
    int global_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_i < n_tokens * dim) {
        int t = global_i / dim;
        int i = global_i % dim;
        int token_id = token_ids[t];
        out[(size_t)t * dim + i] = bf16_to_f32(weights[(size_t)token_id * dim + i]) * scale;
    }
}

__global__ void k_embed_lookup_batch_f16(
    const unsigned short* __restrict__ weights,
    const int* __restrict__ token_ids,
    float* __restrict__ out,
    int n_tokens,
    int dim,
    float scale
) {
    int global_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_i < n_tokens * dim) {
        int t = global_i / dim;
        int i = global_i % dim;
        int token_id = token_ids[t];
        out[(size_t)t * dim + i] = f16_to_f32(weights[(size_t)token_id * dim + i]) * scale;
    }
}

__global__ void k_embed_lookup_batch_f32(
    const float* __restrict__ weights,
    const int* __restrict__ token_ids,
    float* __restrict__ out,
    int n_tokens,
    int dim,
    float scale
) {
    int global_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_i < n_tokens * dim) {
        int t = global_i / dim;
        int i = global_i % dim;
        int token_id = token_ids[t];
        out[(size_t)t * dim + i] = weights[(size_t)token_id * dim + i] * scale;
    }
}

__global__ void k_embed_lookup_batch_q6_k(
    const unsigned char* __restrict__ weights,
    const int* __restrict__ token_ids,
    float* __restrict__ out,
    int n_tokens,
    int dim,
    float scale
) {
    int num_sb = dim / 256;
    int global_sb = blockIdx.x;
    if (global_sb >= n_tokens * num_sb) return;

    int t = global_sb / num_sb;
    int sb = global_sb % num_sb;
    int token_id = token_ids[t];

    size_t row_offset = (size_t)token_id * (size_t)num_sb * 210;
    const unsigned char* block = weights + row_offset + (size_t)sb * 210;
    float d = f16_to_f32(*(const unsigned short*)(block + 208)) * scale;

    int tid = threadIdx.x; // 0..63
    int n = tid / 32;
    int l = tid % 32;
    int is = l / 16;

    const unsigned char* ql = block + 0 + n * 64;
    const unsigned char* qh = block + 128 + n * 32;
    const signed char* scales = (const signed char*)(block + 192 + n * 8);

    unsigned char ql_l = ql[l];
    unsigned char ql_l32 = ql[l + 32];
    unsigned char qh_l = qh[l];

    int q1 = (int)((ql_l & 0x0F) | (((qh_l >> 0) & 3) << 4)) - 32;
    int q2 = (int)((ql_l32 & 0x0F) | (((qh_l >> 2) & 3) << 4)) - 32;
    int q3 = (int)((ql_l >> 4) | (((qh_l >> 4) & 3) << 4)) - 32;
    int q4 = (int)((ql_l32 >> 4) | (((qh_l >> 6) & 3) << 4)) - 32;

    float* dst = out + (size_t)t * dim + (size_t)sb * 256 + (size_t)n * 128;
    dst[l + 0]  = d * (float)scales[is + 0] * (float)q1;
    dst[l + 32] = d * (float)scales[is + 2] * (float)q2;
    dst[l + 64] = d * (float)scales[is + 4] * (float)q3;
    dst[l + 96] = d * (float)scales[is + 6] * (float)q4;
}

extern "C" void cuda_embed_lookup_batch(
    const void* emb_weights,
    int qtype,
    const int* token_ids,
    float* out,
    int n_tokens,
    int dim,
    float scale,
    CudaStream_t stream
) {
    if (!emb_weights || !token_ids || n_tokens <= 0 || dim <= 0) return;
    if (qtype == 2) { // Q4_0
        int num_blocks = dim / 32;
        int total = n_tokens * num_blocks;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        k_embed_lookup_batch_q4_0<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            (const unsigned char*)emb_weights, token_ids, out, n_tokens, dim, scale
        );
    } else if (qtype == 8) { // Q8_0
        int num_blocks = dim / 32;
        int total = n_tokens * num_blocks;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        k_embed_lookup_batch_q8_0<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            (const unsigned char*)emb_weights, token_ids, out, n_tokens, dim, scale
        );
    } else if (qtype == 14) { // Q6_K
        int num_sb = dim / 256;
        k_embed_lookup_batch_q6_k<<<n_tokens * num_sb, 64, 0, (cudaStream_t)stream>>>(
            (const unsigned char*)emb_weights, token_ids, out, n_tokens, dim, scale
        );
    } else if (qtype == 30) { // BF16
        int total = n_tokens * dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        k_embed_lookup_batch_bf16<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            (const unsigned short*)emb_weights, token_ids, out, n_tokens, dim, scale
        );
    } else if (qtype == 1) { // F16
        int total = n_tokens * dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        k_embed_lookup_batch_f16<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            (const unsigned short*)emb_weights, token_ids, out, n_tokens, dim, scale
        );
    } else if (qtype == 0) { // F32
        int total = n_tokens * dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        k_embed_lookup_batch_f32<<<blocks, threads, 0, (cudaStream_t)stream>>>(
            (const float*)emb_weights, token_ids, out, n_tokens, dim, scale
        );
    }
}

__global__ void k_add_rmsnorm(
    float* __restrict__ x,
    const float* __restrict__ residual,
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
        float val = x[i] + residual[i];
        x[i] = val; // in-place update of residual
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

__global__ void k_add_rmsnorm_batched(
    float* __restrict__ x,
    const float* __restrict__ residual_add,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int n,
    float eps,
    int use_unit_offset
) {
    int row = blockIdx.x;
    float* x_row = x + (size_t)row * n;
    const float* res_row = residual_add + (size_t)row * n;
    float* out_row = out + (size_t)row * n;

    __shared__ float s_sum[256];
    int tid = threadIdx.x;

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float val = x_row[i] + res_row[i];
        x_row[i] = val; // in-place update of residual
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
        out_row[i] = x_row[i] * inv_std * w;
    }
}

extern "C" void cuda_add_rmsnorm(
    float* x,
    const float* residual,
    const float* weight,
    float* out,
    int n,
    float eps,
    int use_unit_offset,
    CudaStream_t stream
) {
    k_add_rmsnorm<<<1, 256, 0, (cudaStream_t)stream>>>(x, residual, weight, out, n, eps, use_unit_offset);
}

extern "C" void cuda_add_rmsnorm_batched(
    float* x,
    const float* residual,
    const float* weight,
    float* out,
    int n,
    int batch_size,
    float eps,
    int use_unit_offset,
    CudaStream_t stream
) {
    if (batch_size <= 0) return;
    if (batch_size == 1) {
        cuda_add_rmsnorm(x, residual, weight, out, n, eps, use_unit_offset, stream);
        return;
    }
    k_add_rmsnorm_batched<<<batch_size, 256, 0, (cudaStream_t)stream>>>(x, residual, weight, out, n, eps, use_unit_offset);
}

__global__ void k_rope(
    float* __restrict__ q,
    float* __restrict__ k,
    int pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float freq_base
) {
    int h = blockIdx.x; // head index
    int i = threadIdx.x; // index in [0, half_dim)
    int eff_rotary = (rotary_dim == 0 || rotary_dim > head_dim) ? head_dim : rotary_dim;
    int half_dim = eff_rotary / 2;
    if (i >= half_dim) return;

    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)eff_rotary);
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

extern "C" void cuda_rope(float* q, float* k, int pos, int num_heads, int num_kv_heads, int head_dim, int rotary_dim, float freq_base, CudaStream_t stream) {
    int max_heads = (num_heads > num_kv_heads) ? num_heads : num_kv_heads;
    int eff_rotary = (rotary_dim == 0 || rotary_dim > head_dim) ? head_dim : rotary_dim;
    int half_dim = eff_rotary / 2;
    int threads = (half_dim < 256) ? half_dim : 256;
    k_rope<<<max_heads, threads, 0, (cudaStream_t)stream>>>(q, k, pos, num_heads, num_kv_heads, head_dim, rotary_dim, freq_base);
}

__global__ void k_rope_batched(
    float* __restrict__ q,
    float* __restrict__ k,
    int pos,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float freq_base
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    if (b >= batch_size) return;

    int i = threadIdx.x;
    int eff_rotary = (rotary_dim == 0 || rotary_dim > head_dim) ? head_dim : rotary_dim;
    int half_dim = eff_rotary / 2;
    if (i >= half_dim) return;

    int token_pos = pos + b;
    float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)eff_rotary);
    float val = (float)token_pos * freq;
    float cos_val = cosf(val);
    float sin_val = sinf(val);

    if (h < num_heads && q != NULL) {
        size_t b_offset = (size_t)b * (num_heads * head_dim);
        int offset0 = b_offset + h * head_dim + i;
        int offset1 = offset0 + half_dim;
        float q0 = q[offset0];
        float q1 = q[offset1];
        q[offset0] = q0 * cos_val - q1 * sin_val;
        q[offset1] = q0 * sin_val + q1 * cos_val;
    }

    if (h < num_kv_heads && k != NULL) {
        size_t b_offset = (size_t)b * (num_kv_heads * head_dim);
        int offset0 = b_offset + h * head_dim + i;
        int offset1 = offset0 + half_dim;
        float k0 = k[offset0];
        float k1 = k[offset1];
        k[offset0] = k0 * cos_val - k1 * sin_val;
        k[offset1] = k0 * sin_val + k1 * cos_val;
    }
}

extern "C" void cuda_rope_batched(
    float* q,
    float* k,
    int pos,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float freq_base,
    CudaStream_t stream
) {
    if (batch_size <= 0) return;
    int max_heads = (num_heads > num_kv_heads) ? num_heads : num_kv_heads;
    int eff_rotary = (rotary_dim == 0 || rotary_dim > head_dim) ? head_dim : rotary_dim;
    int half_dim = eff_rotary / 2;
    int threads = (half_dim < 256) ? half_dim : 256;
    dim3 grid(max_heads, batch_size);
    k_rope_batched<<<grid, threads, 0, (cudaStream_t)stream>>>(
        q, k, pos, batch_size, num_heads, num_kv_heads, head_dim, rotary_dim, freq_base
    );
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
    int max_kv_dim,
    CudaStream_t stream
) {
    int kv_dim = n_kv_heads * head_dim;
    int threads = 256;
    int blocks = (kv_dim + threads - 1) / threads;
    k_kv_cache_put<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        k_cache, v_cache, k, v, layer_idx, pos, max_seq, kv_dim, max_kv_dim
    );
}

__global__ void k_kv_cache_put_batched(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int layer_idx,
    int pos,
    int batch_size,
    int max_seq,
    int kv_dim,
    int max_kv_dim
) {
    int b = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size && idx < kv_dim) {
        size_t cache_offset = ((size_t)layer_idx * max_seq + (pos + b)) * max_kv_dim + idx;
        size_t src_offset = (size_t)b * kv_dim + idx;
        k_cache[cache_offset] = k[src_offset];
        v_cache[cache_offset] = v[src_offset];
    }
}

extern "C" void cuda_kv_cache_put_batched(
    float* k_cache,
    float* v_cache,
    const float* k,
    const float* v,
    int layer_idx,
    int pos,
    int batch_size,
    int max_seq,
    int n_kv_heads,
    int head_dim,
    int max_kv_dim,
    CudaStream_t stream
) {
    if (batch_size <= 0) return;
    int kv_dim = n_kv_heads * head_dim;
    int threads = 256;
    dim3 grid((kv_dim + threads - 1) / threads, batch_size);
    k_kv_cache_put_batched<<<grid, threads, 0, (cudaStream_t)stream>>>(
        k_cache, v_cache, k, v, layer_idx, pos, batch_size, max_seq, kv_dim, max_kv_dim
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
    int max_kv_dim,
    float attn_scale,
    float softcap,
    int sliding_window
) {
    int h = blockIdx.x; // Query head index (0..n_heads-1)
    if (h >= n_heads) return;

    int gqa_group = (n_kv_heads > 0) ? (n_heads / n_kv_heads) : 1;
    int kv_h = h / gqa_group;

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
        size_t k_offset = ((size_t)donor_layer * max_seq + t) * max_kv_dim + kv_h * head_dim;
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
            size_t v_offset = ((size_t)donor_layer * max_seq + t) * max_kv_dim + kv_h * head_dim;
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
    int max_kv_dim,
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
        q, k_cache, v_cache, out, donor_layer, pos, max_seq, n_heads, n_kv_heads, head_dim, max_kv_dim, attn_scale, softcap, sliding_window
    );
}

__global__ void k_attention_batched(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    int donor_layer,
    int pos,
    int batch_size,
    int max_seq,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_kv_dim,
    float attn_scale,
    float softcap,
    int sliding_window,
    int max_valid_tokens
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    if (h >= n_heads || b >= batch_size) return;

    int gqa_group = (n_kv_heads > 0) ? (n_heads / n_kv_heads) : 1;
    int kv_h = h / gqa_group;

    const float* q_head = q + (size_t)b * (n_heads * head_dim) + h * head_dim;
    float* out_head = out + (size_t)b * (n_heads * head_dim) + h * head_dim;

    int token_pos = pos + b;
    int seq_len = token_pos + 1;
    int start_t = (sliding_window > 0 && seq_len > sliding_window) ? (seq_len - sliding_window) : 0;
    int valid_tokens = seq_len - start_t;

    extern __shared__ float s_mem[];
    float* s_scores = s_mem;
    float* s_q = s_scores + max_valid_tokens;
    float* s_red = s_q + head_dim;

    int tid = threadIdx.x;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = q_head[d];
    }
    __syncthreads();

    for (int i = tid; i < valid_tokens; i += blockDim.x) {
        int t = start_t + i;
        size_t k_offset = ((size_t)donor_layer * max_seq + t) * max_kv_dim + kv_h * head_dim;
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

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < valid_tokens; ++i) {
            int t = start_t + i;
            size_t v_offset = ((size_t)donor_layer * max_seq + t) * max_kv_dim + kv_h * head_dim;
            acc += s_scores[i] * v_cache[v_offset + d];
        }
        out_head[d] = acc;
    }
}

extern "C" void cuda_attention_batched(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    float* out,
    int donor_layer,
    int pos,
    int batch_size,
    int max_seq,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_kv_dim,
    float attn_scale,
    float softcap,
    int sliding_window,
    CudaStream_t stream
) {
    if (batch_size <= 0) return;
    if (batch_size == 1) {
        cuda_attention_forward(q, k_cache, v_cache, out, donor_layer, pos, max_seq, n_heads, n_kv_heads, head_dim, max_kv_dim, attn_scale, softcap, sliding_window, stream);
        return;
    }
    int max_valid_tokens = pos + batch_size;
    int threads = 64;
    size_t shared_bytes = (max_valid_tokens + head_dim + threads) * sizeof(float);
    dim3 grid(n_heads, batch_size);
    k_attention_batched<<<grid, threads, shared_bytes, (cudaStream_t)stream>>>(
        q, k_cache, v_cache, out, donor_layer, pos, batch_size, max_seq, n_heads, n_kv_heads, head_dim, max_kv_dim, attn_scale, softcap, sliding_window, max_valid_tokens
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

// ============================================================================
// Argmax Reduction Kernels
// ============================================================================

__device__ float g_argmax_tmp_vals[256];
__device__ unsigned int g_argmax_tmp_idxs[256];

__global__ void k_argmax_stage1(
    const float* __restrict__ logits,
    int n,
    float* __restrict__ block_vals,
    unsigned int* __restrict__ block_idxs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float max_val = -1e30f;
    unsigned int max_idx = 0;

    for (int i = tid; i < n; i += stride) {
        float val = logits[i];
        if (val > max_val) {
            max_val = val;
            max_idx = (unsigned int)i;
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
        unsigned int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }

    __shared__ float s_val[32];
    __shared__ unsigned int s_idx[32];

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        s_val[warp_id] = max_val;
        s_idx[warp_id] = max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x >> 5;
        float v = (lane < num_warps) ? s_val[lane] : -1e30f;
        unsigned int idx = (lane < num_warps) ? s_idx[lane] : 0;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_v = __shfl_down_sync(0xffffffff, v, offset);
            unsigned int other_i = __shfl_down_sync(0xffffffff, idx, offset);
            if (other_v > v) {
                v = other_v;
                idx = other_i;
            }
        }

        if (lane == 0) {
            block_vals[blockIdx.x] = v;
            block_idxs[blockIdx.x] = idx;
        }
    }
}

__global__ void k_argmax_stage2(
    const float* __restrict__ block_vals,
    const unsigned int* __restrict__ block_idxs,
    int num_blocks,
    unsigned int* __restrict__ out_idx
) {
    int lane = threadIdx.x; // 0..255

    float v = (lane < num_blocks) ? block_vals[lane] : -1e30f;
    unsigned int idx = (lane < num_blocks) ? block_idxs[lane] : 0;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_v = __shfl_down_sync(0xffffffff, v, offset);
        unsigned int other_i = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_v > v) {
            v = other_v;
            idx = other_i;
        }
    }

    __shared__ float s_val[32];
    __shared__ unsigned int s_idx[32];

    int warp_id = threadIdx.x >> 5;
    int lane_in_warp = threadIdx.x & 31;

    if (lane_in_warp == 0) {
        s_val[warp_id] = v;
        s_idx[warp_id] = idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float fv = (lane_in_warp < 8) ? s_val[lane_in_warp] : -1e30f;
        unsigned int fi = (lane_in_warp < 8) ? s_idx[lane_in_warp] : 0;

        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            float other_v = __shfl_down_sync(0xffffffff, fv, offset);
            unsigned int other_i = __shfl_down_sync(0xffffffff, fi, offset);
            if (other_v > fv) {
                fv = other_v;
                fi = other_i;
            }
        }

        if (lane_in_warp == 0) {
            *out_idx = fi;
        }
    }
}

extern "C" void cuda_argmax(
    const float* logits,
    int n,
    unsigned int* out_idx,
    CudaStream_t stream
) {
    int num_blocks = 256;
    if (n < 256 * 256) {
        num_blocks = (n + 255) / 256;
        if (num_blocks < 1) num_blocks = 1;
    }
    k_argmax_stage1<<<num_blocks, 256, 0, (cudaStream_t)stream>>>(logits, n, g_argmax_tmp_vals, g_argmax_tmp_idxs);
    k_argmax_stage2<<<1, 256, 0, (cudaStream_t)stream>>>(g_argmax_tmp_vals, g_argmax_tmp_idxs, num_blocks, out_idx);
}
