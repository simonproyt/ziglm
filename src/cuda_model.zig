const std = @import("std");
const types = @import("types.zig");
const ModelParams = types.ModelParams;
const GGMLType = types.GGMLType;
const Tensor = types.Tensor;
const quant = @import("quant.zig");
const math = @import("math.zig");
const cuda = @import("cuda.zig");
const CudaDevice = cuda.CudaDevice;
const CudaBuffer = cuda.CudaBuffer;
const model_mod = @import("model.zig");
const TransformerModel = model_mod.TransformerModel;
const ModelBuffers = model_mod.ModelBuffers;
const LayerWeights = model_mod.LayerWeights;
const KVCache = @import("kv_cache.zig").KVCache;

pub const CudaGpuTensor = struct {
    qtype: GGMLType,
    rows: usize,
    cols: usize,
    buf: CudaBuffer,
    size_bytes: usize,

    pub fn upload(device: *const CudaDevice, t: Tensor, rows: usize, cols: usize) !CudaGpuTensor {
        const size_bytes = t.data.len;
        const buf = try device.alloc(size_bytes);
        try buf.upload(t.data, device.stream);

        return CudaGpuTensor{
            .qtype = t.type,
            .rows = rows,
            .cols = cols,
            .buf = buf,
            .size_bytes = size_bytes,
        };
    }

    pub fn deinit(self: *CudaGpuTensor) void {
        self.buf.deinit();
    }
};

pub const CudaGpuNorm = struct {
    buf: CudaBuffer,
    len: usize,

    pub fn upload(device: *const CudaDevice, slice: []const f32) !CudaGpuNorm {
        const size_bytes = slice.len * @sizeOf(f32);
        const buf = try device.alloc(size_bytes);
        const slice_bytes: []const u8 = std.mem.sliceAsBytes(slice);
        try buf.upload(slice_bytes, device.stream);

        return CudaGpuNorm{
            .buf = buf,
            .len = slice.len,
        };
    }

    pub fn deinit(self: *CudaGpuNorm) void {
        self.buf.deinit();
    }
};

pub const CudaGpuLayer = struct {
    input_layernorm: ?CudaGpuNorm = null,
    post_attention_layernorm: ?CudaGpuNorm = null,
    pre_feedforward_layernorm: ?CudaGpuNorm = null,
    post_feedforward_layernorm: ?CudaGpuNorm = null,
    post_per_layer_input_norm: ?CudaGpuNorm = null,

    attn_q: ?CudaGpuTensor = null,
    attn_k: ?CudaGpuTensor = null,
    attn_v: ?CudaGpuTensor = null,
    attn_output: ?CudaGpuTensor = null,

    attn_q_norm: ?CudaGpuNorm = null,
    attn_k_norm: ?CudaGpuNorm = null,

    ffn_gate: ?CudaGpuTensor = null,
    ffn_up: ?CudaGpuTensor = null,
    ffn_down: ?CudaGpuTensor = null,

    per_layer_input_gate: ?CudaGpuTensor = null,
    per_layer_projection: ?CudaGpuTensor = null,

    scale: f32 = 1.0,
    head_dim: usize = 0,
    n_heads: usize = 0,
    n_kv_heads: usize = 0,
    rope_theta: f32 = 10000.0,
    sliding_window: usize = 0,

    pub fn deinit(self: *CudaGpuLayer) void {
        if (self.input_layernorm) |*n| n.deinit();
        if (self.post_attention_layernorm) |*n| n.deinit();
        if (self.pre_feedforward_layernorm) |*n| n.deinit();
        if (self.post_feedforward_layernorm) |*n| n.deinit();
        if (self.post_per_layer_input_norm) |*n| n.deinit();

        if (self.attn_q) |*t| t.deinit();
        if (self.attn_k) |*t| t.deinit();
        if (self.attn_v) |*t| t.deinit();
        if (self.attn_output) |*t| t.deinit();

        if (self.attn_q_norm) |*n| n.deinit();
        if (self.attn_k_norm) |*n| n.deinit();

        if (self.ffn_gate) |*t| t.deinit();
        if (self.ffn_up) |*t| t.deinit();
        if (self.ffn_down) |*t| t.deinit();

        if (self.per_layer_input_gate) |*t| t.deinit();
        if (self.per_layer_projection) |*t| t.deinit();
    }
};

pub const CudaGpuModel = struct {
    allocator: std.mem.Allocator,
    device: *CudaDevice,
    params: ModelParams,
    layers: []CudaGpuLayer,
    output: ?CudaGpuTensor = null,
    output_norm: ?CudaGpuNorm = null,

    // Per-Layer Embedding Context Projection weights on GPU
    per_layer_model_projection: ?CudaGpuTensor = null,

    // Device execution buffers
    d_x: CudaBuffer,
    d_xb: CudaBuffer,
    d_q: CudaBuffer,
    d_k: CudaBuffer,
    d_v: CudaBuffer,
    d_attn_out: CudaBuffer,
    d_gate: CudaBuffer,
    d_up: CudaBuffer,
    d_act: CudaBuffer,
    d_ffn_out: CudaBuffer,
    d_ple_gate: CudaBuffer,
    d_ple_buf: CudaBuffer,
    d_ctx_ple_buf: CudaBuffer,
    d_logits: CudaBuffer,

    // GPU-Resident KV Cache
    d_k_cache: CudaBuffer,
    d_v_cache: CudaBuffer,
    max_seq_len: usize = 4096,

    // Host staging buffers
    host_x: []f32,
    host_logits: []f32,

    pub fn init(allocator: std.mem.Allocator, device: *CudaDevice, cpu_model: *const TransformerModel) !*CudaGpuModel {
        const p = cpu_model.params;
        const dim = p.embedding_length;
        const max_seq: usize = 4096;

        var max_heads: usize = if (p.head_count > 0) p.head_count else 1;
        var max_kv_heads: usize = if (p.head_count_kv > 0) p.head_count_kv else 1;
        var max_head_dim: usize = if (p.head_size > 0) p.head_size else 64;
        var max_inter: usize = dim * 4;

        for (cpu_model.layers) |l| {
            if (l.n_heads > max_heads) max_heads = l.n_heads;
            if (l.n_kv_heads > max_kv_heads) max_kv_heads = l.n_kv_heads;
            if (l.head_dim > max_head_dim) max_head_dim = l.head_dim;
            if (l.intermediate_size > max_inter) max_inter = l.intermediate_size;
        }

        // Allocate device buffers
        const d_x = try device.alloc(dim * @sizeOf(f32));
        const d_xb = try device.alloc(dim * @sizeOf(f32));
        const d_q = try device.alloc(max_heads * max_head_dim * @sizeOf(f32));
        const d_k = try device.alloc(max_kv_heads * max_head_dim * @sizeOf(f32));
        const d_v = try device.alloc(max_kv_heads * max_head_dim * @sizeOf(f32));
        const d_attn_out = try device.alloc(max_heads * max_head_dim * @sizeOf(f32));
        const d_gate = try device.alloc(max_inter * @sizeOf(f32));
        const d_up = try device.alloc(max_inter * @sizeOf(f32));
        const d_act = try device.alloc(max_inter * @sizeOf(f32));
        const d_ffn_out = try device.alloc(dim * @sizeOf(f32));
        const d_ple_gate = try device.alloc(256 * @sizeOf(f32));
        const d_ple_buf = try device.alloc(256 * @sizeOf(f32));
        const d_ctx_ple_buf = try device.alloc(cpu_model.layers.len * 256 * @sizeOf(f32));
        const d_logits = try device.alloc(p.vocab_size * @sizeOf(f32));

        // GPU KV Cache: [num_layers, max_seq, max_kv_heads * max_head_dim]
        const kv_total_floats = cpu_model.layers.len * max_seq * max_kv_heads * max_head_dim;
        const d_k_cache = try device.alloc(kv_total_floats * @sizeOf(f32));
        const d_v_cache = try device.alloc(kv_total_floats * @sizeOf(f32));

        // Host staging buffers
        const host_x = try allocator.alloc(f32, dim);
        const host_logits = try allocator.alloc(f32, p.vocab_size);

        // Upload transformer layers
        const gpu_layers = try allocator.alloc(CudaGpuLayer, cpu_model.layers.len);
        for (cpu_model.layers, 0..) |l, i| {
            const head_size = if (l.head_dim > 0) l.head_dim else p.head_size;
            const n_heads = if (l.n_heads > 0) l.n_heads else p.head_count;
            const n_kv_heads = if (l.n_kv_heads > 0) l.n_kv_heads else p.head_count_kv;
            const inter_size = if (l.intermediate_size > 0) l.intermediate_size else dim * 4;

            gpu_layers[i] = .{
                .scale = l.layer_scalar,
                .head_dim = head_size,
                .n_heads = n_heads,
                .n_kv_heads = n_kv_heads,
                .rope_theta = l.rope_theta,
                .sliding_window = l.sliding_window,
            };

            if (l.input_layernorm) |norm| {
                gpu_layers[i].input_layernorm = try CudaGpuNorm.upload(device, norm);
            }
            if (l.post_attention_layernorm) |norm| {
                gpu_layers[i].post_attention_layernorm = try CudaGpuNorm.upload(device, norm);
            }
            if (l.pre_feedforward_layernorm) |norm| {
                gpu_layers[i].pre_feedforward_layernorm = try CudaGpuNorm.upload(device, norm);
            }
            if (l.post_feedforward_layernorm) |norm| {
                gpu_layers[i].post_feedforward_layernorm = try CudaGpuNorm.upload(device, norm);
            }
            if (l.post_per_layer_input_norm) |norm| {
                gpu_layers[i].post_per_layer_input_norm = try CudaGpuNorm.upload(device, norm);
            }

            if (l.attn_q) |t| gpu_layers[i].attn_q = try CudaGpuTensor.upload(device, t, n_heads * head_size, dim);
            if (l.attn_k) |t| gpu_layers[i].attn_k = try CudaGpuTensor.upload(device, t, n_kv_heads * head_size, dim);
            if (l.attn_v) |t| gpu_layers[i].attn_v = try CudaGpuTensor.upload(device, t, n_kv_heads * head_size, dim);
            if (l.attn_output) |t| gpu_layers[i].attn_output = try CudaGpuTensor.upload(device, t, dim, n_heads * head_size);

            if (l.attn_q_norm) |norm| gpu_layers[i].attn_q_norm = try CudaGpuNorm.upload(device, norm);
            if (l.attn_k_norm) |norm| gpu_layers[i].attn_k_norm = try CudaGpuNorm.upload(device, norm);

            if (l.ffn_gate) |t| gpu_layers[i].ffn_gate = try CudaGpuTensor.upload(device, t, inter_size, dim);
            if (l.ffn_up) |t| gpu_layers[i].ffn_up = try CudaGpuTensor.upload(device, t, inter_size, dim);
            if (l.ffn_down) |t| gpu_layers[i].ffn_down = try CudaGpuTensor.upload(device, t, dim, inter_size);

            if (l.per_layer_input_gate) |t| gpu_layers[i].per_layer_input_gate = try CudaGpuTensor.upload(device, t, 256, dim);
            if (l.per_layer_projection) |t| gpu_layers[i].per_layer_projection = try CudaGpuTensor.upload(device, t, dim, 256);
        }

        // Upload Output Norm
        var gpu_output_norm: ?CudaGpuNorm = null;
        if (cpu_model.output_norm.len > 0) {
            gpu_output_norm = try CudaGpuNorm.upload(device, cpu_model.output_norm);
        }

        // Upload Output Projection
        var gpu_output: ?CudaGpuTensor = null;
        if (cpu_model.output) |t| {
            gpu_output = try CudaGpuTensor.upload(device, t, p.vocab_size, dim);
        } else {
            gpu_output = try CudaGpuTensor.upload(device, cpu_model.token_embd, p.vocab_size, dim);
        }

        var gpu_ctx_proj: ?CudaGpuTensor = null;
        if (cpu_model.per_layer_model_projection) |t| {
            gpu_ctx_proj = try CudaGpuTensor.upload(device, t, cpu_model.layers.len * 256, dim);
        }

        device.sync();

        const self = try allocator.create(CudaGpuModel);
        self.* = .{
            .allocator = allocator,
            .device = device,
            .params = p,
            .layers = gpu_layers,
            .output = gpu_output,
            .output_norm = gpu_output_norm,
            .per_layer_model_projection = gpu_ctx_proj,
            .d_x = d_x,
            .d_xb = d_xb,
            .d_q = d_q,
            .d_k = d_k,
            .d_v = d_v,
            .d_attn_out = d_attn_out,
            .d_gate = d_gate,
            .d_up = d_up,
            .d_act = d_act,
            .d_ffn_out = d_ffn_out,
            .d_ple_gate = d_ple_gate,
            .d_ple_buf = d_ple_buf,
            .d_ctx_ple_buf = d_ctx_ple_buf,
            .d_logits = d_logits,
            .d_k_cache = d_k_cache,
            .d_v_cache = d_v_cache,
            .max_seq_len = max_seq,
            .host_x = host_x,
            .host_logits = host_logits,
        };

        std.debug.print("⚡ Successfully initialized pure CUDA C GPU acceleration ({d} layers in VRAM)!\n", .{gpu_layers.len});
        return self;
    }

    pub fn deinit(self: *CudaGpuModel) void {
        self.device.sync();
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);

        if (self.output) |*t| t.deinit();
        if (self.output_norm) |*n| n.deinit();
        if (self.per_layer_model_projection) |*t| t.deinit();

        self.d_x.deinit();
        self.d_xb.deinit();
        self.d_q.deinit();
        self.d_k.deinit();
        self.d_v.deinit();
        self.d_attn_out.deinit();
        self.d_gate.deinit();
        self.d_up.deinit();
        self.d_act.deinit();
        self.d_ffn_out.deinit();
        self.d_ple_gate.deinit();
        self.d_ple_buf.deinit();
        self.d_ctx_ple_buf.deinit();
        self.d_logits.deinit();
        self.d_k_cache.deinit();
        self.d_v_cache.deinit();

        self.allocator.free(self.host_x);
        self.allocator.free(self.host_logits);
        self.allocator.destroy(self);
    }

    pub fn forward(
        self: *CudaGpuModel,
        cpu_model: *const TransformerModel,
        token_id: u32,
        pos: usize,
        kv_cache: *KVCache,
        bufs: *ModelBuffers,
        custom_embedding: ?[]const f32,
        is_last_token: bool,
    ) ![]const f32 {
        _ = kv_cache;
        const p = self.params;
        const dim = p.embedding_length;

        // 1. Embedding lookup & scaling
        if (custom_embedding) |emb| {
            @memcpy(self.host_x[0..@min(dim, emb.len)], emb[0..@min(dim, emb.len)]);
            if (emb.len < dim) {
                @memset(self.host_x[emb.len..dim], 0.0);
            }
        } else {
            const token_row = cpu_model.token_embd.getRow(token_id);
            quant.dequantizeRow(cpu_model.token_embd.type, token_row, self.host_x, dim);
            const token_scale = if (p.arch == .gemma or p.arch == .gemma2 or p.arch == .gemma4)
                @sqrt(@as(f32, @floatFromInt(dim)))
            else
                1.0;
            for (self.host_x) |*v| v.* *= token_scale;
        }

        // Upload embedding to GPU d_x
        const host_x_bytes: []const u8 = std.mem.sliceAsBytes(self.host_x);
        try self.d_x.upload(host_x_bytes, self.device.stream);

        // PLE precomputation (Token identity + Context projection)
        if (cpu_model.embed_tokens_per_layer) |ple_tab| {
            const ple_dim: usize = 256;
            const total_ple_dim = self.layers.len * ple_dim;
            if (custom_embedding == null and token_id < p.vocab_size) {
                const ple_row = ple_tab.getRow(token_id);
                quant.dequantizeRow(ple_tab.type, ple_row, bufs.ctx_ple_buf[0..total_ple_dim], total_ple_dim);
                const token_scale = @sqrt(@as(f32, @floatFromInt(ple_dim)));
                for (bufs.ctx_ple_buf[0..total_ple_dim]) |*v| v.* *= token_scale;
            } else {
                @memset(bufs.ctx_ple_buf[0..total_ple_dim], 0.0);
            }

            if (cpu_model.per_layer_model_projection) |ctx_proj_t| {
                const inv_sqrt_dim = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
                for (0..dim) |d| bufs.xb[d] = self.host_x[d] * inv_sqrt_dim;
                math.gemv(null, ctx_proj_t.type, ctx_proj_t.data, bufs.xb, bufs.ctx_scratch[0..total_ple_dim], total_ple_dim, dim);
                if (cpu_model.per_layer_projection_norm) |norm_slice| {
                    const inv_sqrt_2: f32 = 1.0 / @sqrt(2.0);
                    for (0..self.layers.len) |l_idx| {
                        const slice = bufs.ctx_scratch[l_idx * ple_dim .. (l_idx + 1) * ple_dim];
                        math.rmsNorm(slice, norm_slice, slice, p.layer_norm_rms_epsilon, false);
                        if (custom_embedding == null) {
                            for (0..ple_dim) |d| {
                                bufs.ctx_ple_buf[l_idx * ple_dim + d] = (bufs.ctx_ple_buf[l_idx * ple_dim + d] + slice[d]) * inv_sqrt_2;
                            }
                        } else {
                            for (0..ple_dim) |d| {
                                bufs.ctx_ple_buf[l_idx * ple_dim + d] = slice[d];
                            }
                        }
                    }
                }
            }

            const ple_bytes: []const u8 = std.mem.sliceAsBytes(bufs.ctx_ple_buf[0..total_ple_dim]);
            try self.d_ctx_ple_buf.upload(ple_bytes, self.device.stream);
        }

        const d_x_ptr: [*]f32 = @ptrCast(@alignCast(self.d_x.ptr));
        const d_xb_ptr: [*]f32 = @ptrCast(@alignCast(self.d_xb.ptr));
        const d_q_ptr: [*]f32 = @ptrCast(@alignCast(self.d_q.ptr));
        const d_k_ptr: [*]f32 = @ptrCast(@alignCast(self.d_k.ptr));
        const d_v_ptr: [*]f32 = @ptrCast(@alignCast(self.d_v.ptr));
        const d_attn_out_ptr: [*]f32 = @ptrCast(@alignCast(self.d_attn_out.ptr));
        const d_gate_ptr: [*]f32 = @ptrCast(@alignCast(self.d_gate.ptr));
        const d_up_ptr: [*]f32 = @ptrCast(@alignCast(self.d_up.ptr));
        const d_act_ptr: [*]f32 = @ptrCast(@alignCast(self.d_act.ptr));
        const d_ffn_out_ptr: [*]f32 = @ptrCast(@alignCast(self.d_ffn_out.ptr));
        const d_ple_gate_ptr: [*]f32 = @ptrCast(@alignCast(self.d_ple_gate.ptr));
        const d_ple_buf_ptr: [*]f32 = @ptrCast(@alignCast(self.d_ple_buf.ptr));
        const d_ctx_ple_ptr: [*]f32 = @ptrCast(@alignCast(self.d_ctx_ple_buf.ptr));
        const d_k_cache_ptr: [*]f32 = @ptrCast(@alignCast(self.d_k_cache.ptr));
        const d_v_cache_ptr: [*]f32 = @ptrCast(@alignCast(self.d_v_cache.ptr));

        // 2. Transformer layers forward (100% on GPU, ZERO host transfers!)
        for (self.layers, 0..) |layer, layer_idx| {
            const head_size = layer.head_dim;
            const n_heads = layer.n_heads;
            const n_kv_heads = layer.n_kv_heads;

            // A. Pre-attention norm
            if (layer.input_layernorm) |norm| {
                const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                self.device.rmsNorm(d_x_ptr, norm_ptr, d_xb_ptr, dim, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            } else {
                _ = cuda.cuda_memcpy_d2d(self.d_xb.ptr, self.d_x.ptr, dim * @sizeOf(f32), self.device.stream);
            }

            // Q Projection
            if (layer.attn_q) |t_q| {
                self.device.gemv(t_q.qtype, t_q.buf.ptr, d_xb_ptr, d_q_ptr, n_heads * head_size, dim);
            }

            // Q Norm
            if (layer.attn_q_norm) |q_norm| {
                const q_norm_ptr: [*]const f32 = @ptrCast(@alignCast(q_norm.buf.ptr));
                for (0..n_heads) |h| {
                    const q_head_ptr = d_q_ptr + h * head_size;
                    self.device.rmsNorm(q_head_ptr, q_norm_ptr, q_head_ptr, head_size, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
                }
            }

            // RoPE on Q (split half)
            self.device.rope(d_q_ptr, null, pos, n_heads, 0, head_size, layer.rope_theta);

            // KV Cache Handling
            const is_kv_shared = (p.arch == .gemma4 and layer_idx >= 15);
            const donor_layer: usize = if (is_kv_shared) (if (layer.head_dim >= 512) 14 else 13) else layer_idx;

            if (!is_kv_shared) {
                if (layer.attn_k) |t_k| {
                    self.device.gemv(t_k.qtype, t_k.buf.ptr, d_xb_ptr, d_k_ptr, n_kv_heads * head_size, dim);
                }
                if (layer.attn_v) |t_v| {
                    self.device.gemv(t_v.qtype, t_v.buf.ptr, d_xb_ptr, d_v_ptr, n_kv_heads * head_size, dim);
                }

                if (layer.attn_k_norm) |k_norm| {
                    const k_norm_ptr: [*]const f32 = @ptrCast(@alignCast(k_norm.buf.ptr));
                    for (0..n_kv_heads) |h| {
                        const k_head_ptr = d_k_ptr + h * head_size;
                        self.device.rmsNorm(k_head_ptr, k_norm_ptr, k_head_ptr, head_size, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
                    }
                }

                if (p.arch == .gemma4) {
                    for (0..n_kv_heads) |h| {
                        const v_head_ptr = d_v_ptr + h * head_size;
                        self.device.rmsNorm(v_head_ptr, null, v_head_ptr, head_size, p.layer_norm_rms_epsilon, false);
                    }
                }

                // RoPE on K (split half)
                self.device.rope(null, d_k_ptr, pos, 0, n_kv_heads, head_size, layer.rope_theta);

                // Store K and V directly into GPU KV Cache
                self.device.kvCachePut(d_k_cache_ptr, d_v_cache_ptr, d_k_ptr, d_v_ptr, layer_idx, pos, self.max_seq_len, n_kv_heads, head_size);
            }

            // GPU-Resident Multi-Head Attention Forward
            const attn_scale: f32 = if (p.arch == .gemma4) 1.0 else 1.0 / @sqrt(@as(f32, @floatFromInt(head_size)));
            self.device.attentionForward(
                d_q_ptr,
                d_k_cache_ptr,
                d_v_cache_ptr,
                d_attn_out_ptr,
                donor_layer,
                pos,
                self.max_seq_len,
                n_heads,
                n_kv_heads,
                head_size,
                attn_scale,
                p.attn_logit_softcapping,
                layer.sliding_window,
            );

            // Output projection GEMV: attn_out -> xb
            if (layer.attn_output) |t_out| {
                self.device.gemv(t_out.qtype, t_out.buf.ptr, d_attn_out_ptr, d_xb_ptr, dim, n_heads * head_size);
            }

            // Post-attention norm (applied to xb with dim)
            if (layer.post_attention_layernorm) |norm| {
                const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                self.device.rmsNorm(d_xb_ptr, norm_ptr, d_xb_ptr, dim, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            }

            // Residual add: x += xb
            self.device.add(d_x_ptr, d_xb_ptr, dim);

            // B. Pre-feedforward norm
            if (layer.pre_feedforward_layernorm) |norm| {
                const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                self.device.rmsNorm(d_x_ptr, norm_ptr, d_xb_ptr, dim, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            } else {
                _ = cuda.cuda_memcpy_d2d(self.d_xb.ptr, self.d_x.ptr, dim * @sizeOf(f32), self.device.stream);
            }

            // FFN Gate & Up projections
            const inter_dim = if (layer.ffn_gate) |g| g.rows else dim * 4;
            if (layer.ffn_gate) |t_gate| {
                self.device.gemv(t_gate.qtype, t_gate.buf.ptr, d_xb_ptr, d_gate_ptr, inter_dim, dim);
            }
            if (layer.ffn_up) |t_up| {
                self.device.gemv(t_up.qtype, t_up.buf.ptr, d_xb_ptr, d_up_ptr, inter_dim, dim);
            }

            // GeGLU activation: act = gelu(gate) * up
            self.device.geglu(d_gate_ptr, d_up_ptr, d_act_ptr, inter_dim);

            // FFN Down projection: act -> ffn_out
            if (layer.ffn_down) |t_down| {
                self.device.gemv(t_down.qtype, t_down.buf.ptr, d_act_ptr, d_ffn_out_ptr, dim, inter_dim);
            }

            // Post-feedforward norm (if any)
            if (layer.post_feedforward_layernorm) |norm| {
                const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                self.device.rmsNorm(d_ffn_out_ptr, norm_ptr, d_ffn_out_ptr, dim, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            }

            // Residual add: x += ffn_out
            self.device.add(d_x_ptr, d_ffn_out_ptr, dim);

            // C. Per-Layer Embedding Injection (PLE) on GPU
            if (layer.per_layer_input_gate != null and layer.per_layer_projection != null) {
                const ple_dim: usize = 256;
                const ple_slice_ptr = d_ctx_ple_ptr + layer_idx * ple_dim;

                const t_gate = layer.per_layer_input_gate.?;
                self.device.gemv(t_gate.qtype, t_gate.buf.ptr, d_x_ptr, d_ple_gate_ptr, ple_dim, dim);

                self.device.pleGateGelu(d_ple_gate_ptr, ple_slice_ptr, d_ple_buf_ptr, ple_dim);

                const t_proj = layer.per_layer_projection.?;
                self.device.gemv(t_proj.qtype, t_proj.buf.ptr, d_ple_buf_ptr, d_xb_ptr, dim, ple_dim);

                if (layer.post_per_layer_input_norm) |norm| {
                    const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                    self.device.rmsNorm(d_xb_ptr, norm_ptr, d_xb_ptr, dim, p.layer_norm_rms_epsilon, false);
                }

                self.device.add(d_x_ptr, d_xb_ptr, dim);
            }

            // D. End-of-layer scaling
            if (layer.scale != 1.0) {
                self.device.scale(d_x_ptr, layer.scale, dim);
            }
        }

        // 3. Final Output Norm
        if (self.output_norm) |norm| {
            const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
            self.device.rmsNorm(d_x_ptr, norm_ptr, d_xb_ptr, dim, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
        } else {
            _ = cuda.cuda_memcpy_d2d(self.d_xb.ptr, self.d_x.ptr, dim * @sizeOf(f32), self.device.stream);
        }

        // If not last token in prefill, return empty without downloading logits
        if (!is_last_token) {
            return self.host_logits[0..0];
        }

        // 4. Output Logits Projection (LM Head)
        const d_logits_ptr: [*]f32 = @ptrCast(@alignCast(self.d_logits.ptr));
        if (self.output) |t_out| {
            self.device.gemv(t_out.qtype, t_out.buf.ptr, d_xb_ptr, d_logits_ptr, p.vocab_size, dim);
        }

        // Final Logit Softcapping
        if (p.final_logit_softcapping > 0.0) {
            const cap = p.final_logit_softcapping;
            self.device.scale(d_logits_ptr, 1.0 / cap, p.vocab_size);
        }

        // Download logits to host
        const logits_bytes: []u8 = std.mem.sliceAsBytes(self.host_logits);
        try self.d_logits.download(logits_bytes, self.device.stream);
        self.device.sync();

        if (p.final_logit_softcapping > 0.0) {
            const cap = p.final_logit_softcapping;
            for (self.host_logits) |*v| {
                v.* = cap * std.math.tanh(v.*);
            }
        }

        return self.host_logits;
    }
};

test "CudaGpuModel C ABI vs CPU TransformerModel forward numerical parity" {
    const allocator = std.testing.allocator;
    const model_path = "/home/simonuwu/models/gemma4-q4/gemma-4-E2B_q4_0-it.gguf";

    const fd = std.posix.openat(std.posix.AT.FDCWD, model_path, .{ .ACCMODE = .RDONLY }, 0) catch return;
    _ = std.posix.system.close(fd);

    const Engine = @import("engine.zig").Engine;

    var cpu_eng = try Engine.load(allocator, model_path, .{ .use_gpu = false });
    defer cpu_eng.deinit();

    var gpu_eng = try Engine.load(allocator, model_path, .{ .use_gpu = true });
    defer gpu_eng.deinit();

    const test_tok: u32 = 100;
    const pos: usize = 0;

    cpu_eng.reset();
    const cpu_logits = try cpu_eng.model.forward(test_tok, pos, cpu_eng.kv_cache, cpu_eng.buffers, cpu_eng.thread_pool, true);

    gpu_eng.reset();
    const gpu_logits = try gpu_eng.gpu_model.?.forward(gpu_eng.model, test_tok, pos, gpu_eng.kv_cache, gpu_eng.buffers, null, true);

    var max_diff: f32 = 0.0;
    var max_idx: usize = 0;
    for (0..cpu_logits.len) |i| {
        const diff = @abs(cpu_logits[i] - gpu_logits[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }
    std.debug.print("\n[CUDA C ABI Real Model Parity] Max diff: {d:.6} at index {d} (CPU: {d:.4}, GPU: {d:.4})\n", .{ max_diff, max_idx, cpu_logits[max_idx], gpu_logits[max_idx] });

    var cpu_max_val: f32 = -1e9;
    var cpu_argmax: u32 = 0;
    var gpu_max_val: f32 = -1e9;
    var gpu_argmax: u32 = 0;
    for (0..cpu_logits.len) |i| {
        if (cpu_logits[i] > cpu_max_val) {
            cpu_max_val = cpu_logits[i];
            cpu_argmax = @intCast(i);
        }
        if (gpu_logits[i] > gpu_max_val) {
            gpu_max_val = gpu_logits[i];
            gpu_argmax = @intCast(i);
        }
    }
    std.debug.print("CPU argmax: {d} ('{s}'), GPU argmax: {d} ('{s}')\n", .{ cpu_argmax, cpu_eng.tokenizer.decode(cpu_argmax), gpu_argmax, gpu_eng.tokenizer.decode(gpu_argmax) });
    try std.testing.expectEqual(cpu_argmax, gpu_argmax);
}
