#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* CudaStream_t;

// Device & Memory Management
int cuda_device_get_info(int device_id, char* name, size_t name_len, size_t* total_vram_bytes);
int cuda_malloc(void** ptr, size_t bytes);
int cuda_free(void* ptr);
int cuda_memcpy_h2d(void* dst, const void* src, size_t bytes, CudaStream_t stream);
int cuda_memcpy_d2h(void* dst, const void* src, size_t bytes, CudaStream_t stream);
int cuda_memcpy_d2d(void* dst, const void* src, size_t bytes, CudaStream_t stream);
int cuda_stream_create(CudaStream_t* stream);
int cuda_stream_destroy(CudaStream_t stream);
int cuda_stream_sync(CudaStream_t stream);

// Quantized & Float GEMV Operations (Matrix * Vector)
void cuda_gemv_q4_0(const void* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream);
void cuda_gemv_q8_0(const void* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream);
void cuda_gemv_q4_k(const void* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream);
void cuda_gemv_q6_k(const void* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream);
void cuda_gemv_f16(const void* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream);
void cuda_gemv_bf16(const void* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream);
void cuda_gemv_f32(const float* weights, const float* x, float* y, int rows, int cols, CudaStream_t stream);

// Normalization & Embeddings
void cuda_rmsnorm(const float* x, const float* weight, float* out, int n, float eps, int use_unit_offset, CudaStream_t stream);
void cuda_rope(float* q, float* k, int pos, int num_heads, int num_kv_heads, int head_dim, float freq_base, CudaStream_t stream);

// Activations & Elementwise Math
void cuda_geglu(const float* gate, const float* up, float* out, int n, CudaStream_t stream);
void cuda_swiglu(const float* gate, const float* up, float* out, int n, CudaStream_t stream);
void cuda_add(float* x, const float* residual, int n, CudaStream_t stream);
void cuda_scale(float* x, float scale, int n, CudaStream_t stream);
void cuda_tanh_softcap(float* x, float cap, int n, CudaStream_t stream);

// GPU-Resident KV Cache
void cuda_kv_cache_put(
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
);

// GPU-Resident Multi-Head / Grouped-Query Attention
void cuda_attention_forward(
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
);

// Gemma Per-Layer Embedding Gate & Fusion
void cuda_ple_gate_gelu(const float* ple_gate_in, const float* ple_slice, float* ple_buf_out, int ple_dim, CudaStream_t stream);
void cuda_ple_ctx_fuse(float* ctx_ple_buf, const float* ctx_scratch, int n, int add_token_embd, CudaStream_t stream);

#ifdef __cplusplus
}
#endif
