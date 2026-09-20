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

    pub inline fn asConstF32(self: *const CudaGpuNorm) [*]const f32 {
        return self.buf.asConstF32();
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
    rotary_dim: usize = 0,
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
    }
};

pub const CudaGpuModel = struct {
    allocator: std.mem.Allocator,
    device: *CudaDevice,
    params: ModelParams,
    layers: []CudaGpuLayer,
    output: ?CudaGpuTensor = null,
    output_norm: ?CudaGpuNorm = null,
    token_embd: ?CudaGpuTensor = null,
    owns_token_embd: bool = false,

    // Per-Layer Embedding Context Projection weights on GPU
    per_layer_model_projection: ?CudaGpuTensor = null,
    per_layer_projection_norm: ?CudaGpuNorm = null,
    embed_tokens_per_layer: ?CudaGpuTensor = null,

    // Device temporary buffers
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
    d_ctx_scratch: CudaBuffer,
    d_logits: CudaBuffer,
    d_argmax: CudaBuffer,
    d_tokens: CudaBuffer,
    d_params: CudaBuffer,

    // GPU-Resident KV Cache
    d_k_cache: CudaBuffer,
    d_v_cache: CudaBuffer,
    max_seq_len: usize = 4096,
    max_kv_dim: usize = 0,

    // Host staging buffers
    host_x: []f32,
    host_ple: []f32,
    host_logits: []f32,
    host_params: [*]c_int,
    host_out_tok: *c_uint,

    // CUDA Graph state for near-zero dispatch overhead decode
    graph_exec: ?cuda.CudaGraphExec_t = null,
    graph: ?cuda.CudaGraph_t = null,

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
        const max_batch: usize = 512;
        const total_ple_dim = cpu_model.layers.len * 256;

        const d_x = try device.alloc(max_batch * dim * @sizeOf(f32));
        const d_xb = try device.alloc(max_batch * dim * @sizeOf(f32));
        const d_q = try device.alloc(max_batch * max_heads * max_head_dim * @sizeOf(f32));
        const d_k = try device.alloc(max_batch * max_kv_heads * max_head_dim * @sizeOf(f32));
        const d_v = try device.alloc(max_batch * max_kv_heads * max_head_dim * @sizeOf(f32));
        const d_attn_out = try device.alloc(max_batch * max_heads * max_head_dim * @sizeOf(f32));
        const d_gate = try device.alloc(max_batch * max_inter * @sizeOf(f32));
        const d_up = try device.alloc(max_batch * max_inter * @sizeOf(f32));
        const d_act = try device.alloc(max_batch * max_inter * @sizeOf(f32));
        const d_ffn_out = try device.alloc(max_batch * dim * @sizeOf(f32));
        const d_ple_gate = try device.alloc(max_batch * 256 * @sizeOf(f32));
        const d_ple_buf = try device.alloc(max_batch * 256 * @sizeOf(f32));
        const d_ctx_ple_buf = try device.alloc(max_batch * total_ple_dim * @sizeOf(f32));
        const d_ctx_scratch = try device.alloc(max_batch * total_ple_dim * @sizeOf(f32));
        const d_logits = try device.alloc(p.vocab_size * @sizeOf(f32));
        const d_argmax = try device.alloc(@sizeOf(c_uint));
        const d_tokens = try device.alloc(max_batch * @sizeOf(c_int));
        const d_params = try device.alloc(2 * @sizeOf(c_int));

        // GPU KV Cache: [num_layers, max_seq, max_kv_heads * max_head_dim]
        const max_kv_dim = max_kv_heads * max_head_dim;
        const kv_total_floats = cpu_model.layers.len * max_seq * max_kv_dim;
        const d_k_cache = try device.alloc(kv_total_floats * @sizeOf(f32));
        const d_v_cache = try device.alloc(kv_total_floats * @sizeOf(f32));

        // Host staging buffers
        const host_x = try allocator.alloc(f32, max_batch * dim);
        const host_ple = try allocator.alloc(f32, max_batch * total_ple_dim);
        const host_logits = try allocator.alloc(f32, p.vocab_size);

        var h_params_raw: ?*anyopaque = null;
        if (cuda.cuda_malloc_host(&h_params_raw, 2 * @sizeOf(c_int)) != 0 or h_params_raw == null) {
            return error.CudaHostAllocFailed;
        }
        const host_params: [*]c_int = @ptrCast(@alignCast(h_params_raw.?));

        var h_out_raw: ?*anyopaque = null;
        if (cuda.cuda_malloc_host(&h_out_raw, @sizeOf(c_uint)) != 0 or h_out_raw == null) {
            return error.CudaHostAllocFailed;
        }
        const host_out_tok: *c_uint = @ptrCast(@alignCast(h_out_raw.?));

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
                .rotary_dim = l.rotary_dim,
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

            if (l.attn_q_norm) |norm| gpu_layers[i].attn_q_norm = try CudaGpuNorm.upload(device, norm);
            if (l.attn_k_norm) |norm| gpu_layers[i].attn_k_norm = try CudaGpuNorm.upload(device, norm);

            if (l.attn_q) |t| gpu_layers[i].attn_q = try CudaGpuTensor.upload(device, t, n_heads * head_size, dim);
            if (l.attn_k) |t| gpu_layers[i].attn_k = try CudaGpuTensor.upload(device, t, n_kv_heads * head_size, dim);
            if (l.attn_v) |t| gpu_layers[i].attn_v = try CudaGpuTensor.upload(device, t, n_kv_heads * head_size, dim);
            if (l.attn_output) |t| gpu_layers[i].attn_output = try CudaGpuTensor.upload(device, t, dim, n_heads * head_size);

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

        // Upload Output Projection and Token Embeddings
        const can_gpu_embed = switch (cpu_model.token_embd.type) {
            .F32, .F16, .BF16, .Q4_0, .Q8_0, .Q4_K, .Q6_K => true,
            else => false,
        };

        var gpu_output: ?CudaGpuTensor = null;
        var gpu_token_embd: ?CudaGpuTensor = null;
        var owns_token_embd = false;
        if (cpu_model.output) |t| {
            gpu_output = try CudaGpuTensor.upload(device, t, p.vocab_size, dim);
            if (can_gpu_embed) {
                gpu_token_embd = CudaGpuTensor.upload(device, cpu_model.token_embd, p.vocab_size, dim) catch null;
                owns_token_embd = (gpu_token_embd != null);
            }
        } else {
            gpu_output = try CudaGpuTensor.upload(device, cpu_model.token_embd, p.vocab_size, dim);
            if (can_gpu_embed) {
                gpu_token_embd = gpu_output;
            }
            owns_token_embd = false;
        }

        var gpu_ctx_proj: ?CudaGpuTensor = null;
        if (cpu_model.per_layer_model_projection) |t| {
            gpu_ctx_proj = try CudaGpuTensor.upload(device, t, cpu_model.layers.len * 256, dim);
        }

        var gpu_ctx_norm: ?CudaGpuNorm = null;
        if (cpu_model.per_layer_projection_norm) |norm| {
            gpu_ctx_norm = try CudaGpuNorm.upload(device, norm);
        }

        var gpu_embed_tokens_per_layer: ?CudaGpuTensor = null;
        if (cpu_model.embed_tokens_per_layer) |t| {
            const can_gpu_ple = switch (t.type) {
                .F32, .F16, .BF16, .Q4_0, .Q8_0, .Q4_K, .Q6_K => true,
                else => false,
            };
            if (can_gpu_ple) {
                gpu_embed_tokens_per_layer = CudaGpuTensor.upload(device, t, p.vocab_size, cpu_model.layers.len * 256) catch null;
                if (gpu_embed_tokens_per_layer != null) {
                    std.debug.print("⚡ Gemma-4 Per-Layer Embedding table offloaded to GPU VRAM!\n", .{});
                }
            }
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
            .token_embd = gpu_token_embd,
            .owns_token_embd = owns_token_embd,
            .per_layer_model_projection = gpu_ctx_proj,
            .per_layer_projection_norm = gpu_ctx_norm,
            .embed_tokens_per_layer = gpu_embed_tokens_per_layer,
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
            .d_ctx_scratch = d_ctx_scratch,
            .d_logits = d_logits,
            .d_argmax = d_argmax,
            .d_tokens = d_tokens,
            .d_params = d_params,
            .d_k_cache = d_k_cache,
            .d_v_cache = d_v_cache,
            .max_seq_len = max_seq,
            .max_kv_dim = max_kv_dim,
            .host_x = host_x,
            .host_ple = host_ple,
            .host_logits = host_logits,
            .host_params = host_params,
            .host_out_tok = host_out_tok,
        };

        std.debug.print("⚡ Successfully initialized pure CUDA C GPU acceleration ({d} layers in VRAM)!\n", .{gpu_layers.len});
        return self;
    }

    pub fn deinit(self: *CudaGpuModel) void {
        self.device.sync();
        if (self.graph_exec) |exec| {
            _ = cuda.cuda_graph_exec_destroy(exec);
            self.graph_exec = null;
        }
        if (self.graph) |g| {
            _ = cuda.cuda_graph_destroy(g);
            self.graph = null;
        }
        _ = cuda.cuda_free_host(self.host_params);
        _ = cuda.cuda_free_host(self.host_out_tok);

        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);

        if (self.output) |*t| t.deinit();
        if (self.owns_token_embd) {
            if (self.token_embd) |*t| t.deinit();
        }
        if (self.output_norm) |*n| n.deinit();
        if (self.per_layer_model_projection) |*t| t.deinit();
        if (self.per_layer_projection_norm) |*n| n.deinit();
        if (self.embed_tokens_per_layer) |*t| t.deinit();

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
        self.d_ctx_scratch.deinit();
        self.d_logits.deinit();
        self.d_argmax.deinit();
        self.d_tokens.deinit();
        self.d_params.deinit();
        self.d_k_cache.deinit();
        self.d_v_cache.deinit();

        self.allocator.free(self.host_x);
        self.allocator.free(self.host_ple);
        self.allocator.free(self.host_logits);
        self.allocator.destroy(self);
    }

    pub fn forward(
        self: *CudaGpuModel,
        cpu_model: *const TransformerModel,
        token_id: u32,
        pos: usize,
        _: *KVCache,
        bufs: *ModelBuffers,
        custom_embedding: ?[]const f32,
        is_last_token: bool,
    ) ![]const f32 {
        const p = self.params;
        const dim = p.embedding_length;

        const d_x_ptr: [*]f32 = @ptrCast(@alignCast(self.d_x.ptr));
        const d_xb_ptr: [*]f32 = @ptrCast(@alignCast(self.d_xb.ptr));

        // 1. Embedding lookup (100% on GPU if token_embd resident)
        if (custom_embedding) |embd| {
            @memcpy(self.host_x[0..dim], embd[0..dim]);
            const host_x_bytes: []const u8 = std.mem.sliceAsBytes(self.host_x);
            try self.d_x.upload(host_x_bytes, self.device.stream);
        } else if (self.token_embd) |emb_tensor| {
            const embd_scale: f32 = if (p.arch == .gemma or p.arch == .gemma2 or p.arch == .gemma4)
                @sqrt(@as(f32, @floatFromInt(dim)))
            else
                1.0;
            self.device.embedLookup(
                emb_tensor.buf.ptr,
                emb_tensor.qtype,
                token_id,
                d_x_ptr,
                dim,
                embd_scale,
            );
        } else {
            const row_bytes = cpu_model.token_embd.getRow(token_id);
            quant.dequantizeRow(cpu_model.token_embd.type, row_bytes, self.host_x[0..dim], dim);
            const embd_scale = if (p.arch == .gemma or p.arch == .gemma2 or p.arch == .gemma4)
                @sqrt(@as(f32, @floatFromInt(dim)))
            else
                1.0;
            for (self.host_x[0..dim]) |*v| v.* *= embd_scale;
            const host_x_bytes: []const u8 = std.mem.sliceAsBytes(self.host_x);
            try self.d_x.upload(host_x_bytes, self.device.stream);
        }

        // PLE precomputation on GPU
        if (self.embed_tokens_per_layer) |ple_tensor| {
            const ple_dim: usize = 256;
            const total_ple_dim = self.layers.len * ple_dim;
            const d_ctx_ple_ptr: [*]f32 = @ptrCast(@alignCast(self.d_ctx_ple_buf.ptr));

            if (custom_embedding == null and token_id < p.vocab_size) {
                const token_scale = @sqrt(@as(f32, @floatFromInt(ple_dim)));
                self.device.embedLookup(
                    ple_tensor.buf.ptr,
                    ple_tensor.qtype,
                    token_id,
                    d_ctx_ple_ptr,
                    total_ple_dim,
                    token_scale,
                );
            } else {
                self.device.scale(d_ctx_ple_ptr, 0.0, total_ple_dim);
            }

            if (self.per_layer_model_projection) |ctx_proj| {
                const inv_sqrt_dim: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
                const d_ctx_scratch_ptr: [*]f32 = @ptrCast(@alignCast(self.d_ctx_scratch.ptr));

                // Scale d_x into d_xb: d_xb = d_x * inv_sqrt_dim on GPU
                _ = cuda.cuda_memcpy_d2d(self.d_xb.ptr, self.d_x.ptr, dim * @sizeOf(f32), self.device.stream);
                self.device.scale(d_xb_ptr, inv_sqrt_dim, dim);

                // GPU GEMV: ctx_scratch = ctx_proj * d_xb (8960 rows x 1536 cols in parallel on GPU)
                self.device.gemv(ctx_proj.qtype, ctx_proj.buf.ptr, d_xb_ptr, d_ctx_scratch_ptr, total_ple_dim, dim);

                // Normalization on GPU (batched across all 35 layers in ONE kernel launch!)
                if (self.per_layer_projection_norm) |norm| {
                    const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                    self.device.rmsNormBatched(d_ctx_scratch_ptr, norm_ptr, d_ctx_scratch_ptr, ple_dim, self.layers.len, p.layer_norm_rms_epsilon, false);
                }

                // Fusion on GPU
                self.device.pleCtxFuse(d_ctx_ple_ptr, d_ctx_scratch_ptr, total_ple_dim, custom_embedding == null and token_id < p.vocab_size);
            }
        } else if (cpu_model.embed_tokens_per_layer) |ple_tab| {
            const ple_dim: usize = 256;
            const total_ple_dim = self.layers.len * ple_dim;
            const d_ctx_ple_ptr: [*]f32 = @ptrCast(@alignCast(self.d_ctx_ple_buf.ptr));

            if (custom_embedding == null and token_id < p.vocab_size) {
                const ple_row = ple_tab.getRow(token_id);
                quant.dequantizeRow(ple_tab.type, ple_row, bufs.ctx_ple_buf[0..total_ple_dim], total_ple_dim);
                const token_scale = @sqrt(@as(f32, @floatFromInt(ple_dim)));
                for (bufs.ctx_ple_buf[0..total_ple_dim]) |*v| v.* *= token_scale;
            } else {
                @memset(bufs.ctx_ple_buf[0..total_ple_dim], 0.0);
            }

            const ple_bytes: []const u8 = std.mem.sliceAsBytes(bufs.ctx_ple_buf[0..total_ple_dim]);
            try self.d_ctx_ple_buf.upload(ple_bytes, self.device.stream);

            if (self.per_layer_model_projection) |ctx_proj| {
                const inv_sqrt_dim: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
                const d_ctx_scratch_ptr: [*]f32 = @ptrCast(@alignCast(self.d_ctx_scratch.ptr));

                // Scale d_x into d_xb: d_xb = d_x * inv_sqrt_dim on GPU
                _ = cuda.cuda_memcpy_d2d(self.d_xb.ptr, self.d_x.ptr, dim * @sizeOf(f32), self.device.stream);
                self.device.scale(d_xb_ptr, inv_sqrt_dim, dim);

                // GPU GEMV: ctx_scratch = ctx_proj * d_xb (8960 rows x 1536 cols in parallel on GPU)
                self.device.gemv(ctx_proj.qtype, ctx_proj.buf.ptr, d_xb_ptr, d_ctx_scratch_ptr, total_ple_dim, dim);

                // Normalization on GPU (batched across all 35 layers in ONE kernel launch!)
                if (self.per_layer_projection_norm) |norm| {
                    const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                    self.device.rmsNormBatched(d_ctx_scratch_ptr, norm_ptr, d_ctx_scratch_ptr, ple_dim, self.layers.len, p.layer_norm_rms_epsilon, false);
                }

                // Fusion on GPU
                self.device.pleCtxFuse(d_ctx_ple_ptr, d_ctx_scratch_ptr, total_ple_dim, custom_embedding == null);
            }
        }

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
            // A. Pre-Attention Norm
            if (layer.input_layernorm) |norm| {
                const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                self.device.rmsNorm(d_x_ptr, norm_ptr, d_xb_ptr, dim, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            } else {
                _ = cuda.cuda_memcpy_d2d(self.d_xb.ptr, self.d_x.ptr, dim * @sizeOf(f32), self.device.stream);
            }

            const head_size = layer.head_dim;
            const n_heads = layer.n_heads;
            const n_kv_heads = layer.n_kv_heads;

            // KV Cache Handling (with Cross-Layer Sharing for Gemma 4 & hybrid architectures)
            const unshared_count: usize = if (p.num_kv_shared_layers > 0)
                (p.block_count - p.num_kv_shared_layers)
            else if (p.arch == .gemma4)
                (if (p.block_count == 42) 24 else 15)
            else
                p.block_count;

            const is_kv_shared = (p.arch == .gemma4 and (layer.attn_k == null or layer_idx >= unshared_count));
            var donor_layer: usize = layer_idx;
            if (is_kv_shared) {
                var l = unshared_count;
                while (l > 0) {
                    l -= 1;
                    if (self.layers[l].head_dim == layer.head_dim) {
                        donor_layer = l;
                        break;
                    }
                }
            }

            if (!is_kv_shared and layer.attn_q != null and layer.attn_k != null and layer.attn_v != null and
                layer.attn_q.?.qtype == .Q4_0 and layer.attn_k.?.qtype == .Q4_0 and layer.attn_v.?.qtype == .Q4_0)
            {
                self.device.gemvQkvQ4_0(
                    layer.attn_q.?.buf.ptr,
                    layer.attn_k.?.buf.ptr,
                    layer.attn_v.?.buf.ptr,
                    d_xb_ptr,
                    d_q_ptr,
                    d_k_ptr,
                    d_v_ptr,
                    n_heads * head_size,
                    n_kv_heads * head_size,
                    n_kv_heads * head_size,
                    dim,
                );
            } else {
                if (layer.attn_q) |t_q| {
                    self.device.gemv(t_q.qtype, t_q.buf.ptr, d_xb_ptr, d_q_ptr, n_heads * head_size, dim);
                }
                if (!is_kv_shared) {
                    if (layer.attn_k) |t_k| {
                        self.device.gemv(t_k.qtype, t_k.buf.ptr, d_xb_ptr, d_k_ptr, n_kv_heads * head_size, dim);
                    }
                    if (layer.attn_v) |t_v| {
                        self.device.gemv(t_v.qtype, t_v.buf.ptr, d_xb_ptr, d_v_ptr, n_kv_heads * head_size, dim);
                    }
                }
            }

            if (layer.attn_q_norm) |q_norm| {
                const q_norm_ptr: [*]const f32 = @ptrCast(@alignCast(q_norm.buf.ptr));
                self.device.rmsNormBatched(d_q_ptr, q_norm_ptr, d_q_ptr, head_size, n_heads, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            }

            if (!is_kv_shared) {
                if (layer.attn_k_norm) |k_norm| {
                    const k_norm_ptr: [*]const f32 = @ptrCast(@alignCast(k_norm.buf.ptr));
                    self.device.rmsNormBatched(d_k_ptr, k_norm_ptr, d_k_ptr, head_size, n_kv_heads, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
                }

                if (p.arch == .gemma4) {
                    self.device.rmsNormBatched(d_v_ptr, null, d_v_ptr, head_size, n_kv_heads, p.layer_norm_rms_epsilon, false);
                }

                self.device.rope(null, d_k_ptr, pos, 0, n_kv_heads, head_size, layer.rotary_dim, layer.rope_theta);
                self.device.kvCachePut(d_k_cache_ptr, d_v_cache_ptr, d_k_ptr, d_v_ptr, layer_idx, pos, self.max_seq_len, n_kv_heads, head_size, self.max_kv_dim);
            }

            self.device.rope(d_q_ptr, null, pos, n_heads, 0, head_size, layer.rotary_dim, layer.rope_theta);

            const donor_kv_heads = self.layers[donor_layer].n_kv_heads;
            const attn_scale: f32 = if (p.arch == .gemma4) 1.0 else 1.0 / @sqrt(@as(f32, @floatFromInt(head_size)));
            self.device.attentionForward(d_q_ptr, d_k_cache_ptr, d_v_cache_ptr, d_attn_out_ptr, donor_layer, pos, self.max_seq_len, n_heads, donor_kv_heads, head_size, self.max_kv_dim, attn_scale, p.attn_logit_softcapping, layer.sliding_window);

            if (layer.attn_output) |t_out| {
                self.device.gemv(t_out.qtype, t_out.buf.ptr, d_attn_out_ptr, d_xb_ptr, dim, n_heads * head_size);
            }

            if (layer.post_attention_layernorm) |norm| {
                const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                self.device.rmsNorm(d_xb_ptr, norm_ptr, d_xb_ptr, dim, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            }

            const d_ffn_in_ptr = if (layer.pre_feedforward_layernorm) |norm| blk: {
                const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                self.device.addRmsNorm(d_x_ptr, d_xb_ptr, norm_ptr, d_xb_ptr, dim, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
                break :blk d_xb_ptr;
            } else blk: {
                self.device.add(d_x_ptr, d_xb_ptr, dim);
                break :blk d_x_ptr;
            };

            const inter_dim = if (layer.ffn_gate) |g| g.rows else dim * 4;

            if (layer.ffn_gate) |t_gate| {
                if (layer.ffn_up) |t_up| {
                    if (t_gate.qtype == .Q4_0 and t_up.qtype == .Q4_0) {
                        self.device.gemvGegluQ4_0(t_gate.buf.ptr, t_up.buf.ptr, d_ffn_in_ptr, d_act_ptr, inter_dim, dim);
                    } else {
                        self.device.gemv(t_gate.qtype, t_gate.buf.ptr, d_ffn_in_ptr, d_gate_ptr, inter_dim, dim);
                        self.device.gemv(t_up.qtype, t_up.buf.ptr, d_ffn_in_ptr, d_up_ptr, inter_dim, dim);
                        self.device.geglu(d_gate_ptr, d_up_ptr, d_act_ptr, inter_dim);
                    }
                }
            }

            if (layer.ffn_down) |t_down| {
                self.device.gemv(t_down.qtype, t_down.buf.ptr, d_act_ptr, d_ffn_out_ptr, dim, inter_dim);
            }

            if (layer.post_feedforward_layernorm) |norm| {
                const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                self.device.rmsnormAdd(d_x_ptr, d_ffn_out_ptr, norm_ptr, dim, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            } else {
                self.device.add(d_x_ptr, d_ffn_out_ptr, dim);
            }

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
                    self.device.rmsnormAdd(d_x_ptr, d_xb_ptr, norm_ptr, dim, p.layer_norm_rms_epsilon, false);
                } else {
                    self.device.add(d_x_ptr, d_xb_ptr, dim);
                }
            }

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

        if (!is_last_token) {
            return self.host_logits[0..0];
        }

        // 4. Output Logits Projection (LM Head)
        const d_logits_ptr: [*]f32 = @ptrCast(@alignCast(self.d_logits.ptr));
        if (self.output) |t_out| {
            self.device.gemv(t_out.qtype, t_out.buf.ptr, d_xb_ptr, d_logits_ptr, p.vocab_size, dim);
        }

        if (p.final_logit_softcapping > 0.0) {
            self.device.tanhSoftcap(d_logits_ptr, p.final_logit_softcapping, p.vocab_size);
        }

        const logits_bytes: []u8 = std.mem.sliceAsBytes(self.host_logits);
        try self.d_logits.download(logits_bytes, self.device.stream);
        self.device.sync();

        return self.host_logits;
    }

    /// Pure GPU decode step: dispatches all kernels for 1 token into the stream.
    /// Reads token_id and pos from self.d_params (device memory).
    /// Writes argmax index to self.d_argmax.
    /// Zero CPU synchronization, zero host allocations - 100% graphable!
    pub fn forwardGpuDecodeStep(self: *CudaGpuModel) void {
        const p = self.params;
        const dim = p.embedding_length;
        const d_x_ptr: [*]f32 = @ptrCast(@alignCast(self.d_x.ptr));
        const d_xb_ptr: [*]f32 = @ptrCast(@alignCast(self.d_xb.ptr));
        const d_tokens_ptr: [*]const c_int = @ptrCast(@alignCast(self.d_params.ptr));
        const d_pos_ptr: [*]const c_int = @ptrCast(@alignCast(@as([*]u8, @ptrCast(self.d_params.ptr.?)) + @sizeOf(c_int)));

        // 1. Embedding lookup from device token ID
        if (self.token_embd) |emb_tensor| {
            const embd_scale: f32 = if (p.arch == .gemma or p.arch == .gemma2 or p.arch == .gemma4)
                @sqrt(@as(f32, @floatFromInt(dim)))
            else
                1.0;
            self.device.embedLookupBatch(
                emb_tensor.buf.ptr,
                emb_tensor.qtype,
                d_tokens_ptr,
                d_x_ptr,
                1,
                dim,
                embd_scale,
            );
        }

        // Gemma 4 PLE precomputation on GPU
        const ple_dim: usize = 256;
        const total_ple_dim = self.layers.len * ple_dim;
        const d_ctx_ple_ptr: [*]f32 = @ptrCast(@alignCast(self.d_ctx_ple_buf.ptr));

        if (self.embed_tokens_per_layer) |ple_tensor| {
            const token_scale = @sqrt(@as(f32, @floatFromInt(ple_dim)));
            self.device.embedLookupBatch(
                ple_tensor.buf.ptr,
                ple_tensor.qtype,
                d_tokens_ptr,
                d_ctx_ple_ptr,
                1,
                total_ple_dim,
                token_scale,
            );
        }

        if (self.per_layer_model_projection) |ctx_proj| {
            const inv_sqrt_dim: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
            const d_ctx_scratch_ptr: [*]f32 = @ptrCast(@alignCast(self.d_ctx_scratch.ptr));

            _ = cuda.cuda_memcpy_d2d(self.d_xb.ptr, self.d_x.ptr, dim * @sizeOf(f32), self.device.stream);
            self.device.scale(d_xb_ptr, inv_sqrt_dim, dim);

            self.device.gemv(ctx_proj.qtype, ctx_proj.buf.ptr, d_xb_ptr, d_ctx_scratch_ptr, total_ple_dim, dim);

            if (self.per_layer_projection_norm) |norm| {
                const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                self.device.rmsNormBatched(d_ctx_scratch_ptr, norm_ptr, d_ctx_scratch_ptr, ple_dim, self.layers.len, p.layer_norm_rms_epsilon, false);
            }

            self.device.pleCtxFuse(d_ctx_ple_ptr, d_ctx_scratch_ptr, total_ple_dim, true);
        }

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
        const d_k_cache_ptr: [*]f32 = @ptrCast(@alignCast(self.d_k_cache.ptr));
        const d_v_cache_ptr: [*]f32 = @ptrCast(@alignCast(self.d_v_cache.ptr));

        // 2. Transformer layers forward
        for (self.layers, 0..) |layer, layer_idx| {
            if (layer.input_layernorm) |norm| {
                const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                self.device.rmsNorm(d_x_ptr, norm_ptr, d_xb_ptr, dim, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            } else {
                _ = cuda.cuda_memcpy_d2d(self.d_xb.ptr, self.d_x.ptr, dim * @sizeOf(f32), self.device.stream);
            }

            const head_size = layer.head_dim;
            const n_heads = layer.n_heads;
            const n_kv_heads = layer.n_kv_heads;

            const unshared_count: usize = if (p.num_kv_shared_layers > 0)
                (p.block_count - p.num_kv_shared_layers)
            else if (p.arch == .gemma4)
                (if (p.block_count == 42) 24 else 15)
            else
                p.block_count;

            const is_kv_shared = (p.arch == .gemma4 and (layer.attn_k == null or layer_idx >= unshared_count));
            var donor_layer: usize = layer_idx;
            if (is_kv_shared) {
                var l = unshared_count;
                while (l > 0) {
                    l -= 1;
                    if (self.layers[l].head_dim == layer.head_dim) {
                        donor_layer = l;
                        break;
                    }
                }
            }

            if (!is_kv_shared and layer.attn_q != null and layer.attn_k != null and layer.attn_v != null and
                layer.attn_q.?.qtype == .Q4_0 and layer.attn_k.?.qtype == .Q4_0 and layer.attn_v.?.qtype == .Q4_0)
            {
                self.device.gemvQkvQ4_0(
                    layer.attn_q.?.buf.ptr,
                    layer.attn_k.?.buf.ptr,
                    layer.attn_v.?.buf.ptr,
                    d_xb_ptr,
                    d_q_ptr,
                    d_k_ptr,
                    d_v_ptr,
                    n_heads * head_size,
                    n_kv_heads * head_size,
                    n_kv_heads * head_size,
                    dim,
                );
            } else {
                if (layer.attn_q) |t_q| {
                    self.device.gemv(t_q.qtype, t_q.buf.ptr, d_xb_ptr, d_q_ptr, n_heads * head_size, dim);
                }
                if (!is_kv_shared) {
                    if (layer.attn_k) |t_k| {
                        self.device.gemv(t_k.qtype, t_k.buf.ptr, d_xb_ptr, d_k_ptr, n_kv_heads * head_size, dim);
                    }
                    if (layer.attn_v) |t_v| {
                        self.device.gemv(t_v.qtype, t_v.buf.ptr, d_xb_ptr, d_v_ptr, n_kv_heads * head_size, dim);
                    }
                }
            }

            if (layer.attn_q_norm) |q_norm| {
                const q_norm_ptr: [*]const f32 = @ptrCast(@alignCast(q_norm.buf.ptr));
                self.device.rmsNormBatched(d_q_ptr, q_norm_ptr, d_q_ptr, head_size, n_heads, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            }

            if (!is_kv_shared) {
                if (layer.attn_k_norm) |k_norm| {
                    const k_norm_ptr: [*]const f32 = @ptrCast(@alignCast(k_norm.buf.ptr));
                    self.device.rmsNormBatched(d_k_ptr, k_norm_ptr, d_k_ptr, head_size, n_kv_heads, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
                }

                if (p.arch == .gemma4) {
                    self.device.rmsNormBatched(d_v_ptr, null, d_v_ptr, head_size, n_kv_heads, p.layer_norm_rms_epsilon, false);
                }

                self.device.ropeInd(null, d_k_ptr, d_pos_ptr, 0, n_kv_heads, head_size, layer.rotary_dim, layer.rope_theta);
                self.device.kvCachePutInd(d_k_cache_ptr, d_v_cache_ptr, d_k_ptr, d_v_ptr, layer_idx, d_pos_ptr, self.max_seq_len, n_kv_heads, head_size, self.max_kv_dim);
            }

            self.device.ropeInd(d_q_ptr, null, d_pos_ptr, n_heads, 0, head_size, layer.rotary_dim, layer.rope_theta);

            const donor_kv_heads = self.layers[donor_layer].n_kv_heads;
            const attn_scale: f32 = if (p.arch == .gemma4) 1.0 else 1.0 / @sqrt(@as(f32, @floatFromInt(head_size)));
            self.device.attentionForwardInd(d_q_ptr, d_k_cache_ptr, d_v_cache_ptr, d_attn_out_ptr, donor_layer, d_pos_ptr, self.max_seq_len, n_heads, donor_kv_heads, head_size, self.max_kv_dim, attn_scale, p.attn_logit_softcapping, layer.sliding_window);

            if (layer.attn_output) |t_out| {
                self.device.gemv(t_out.qtype, t_out.buf.ptr, d_attn_out_ptr, d_xb_ptr, dim, n_heads * head_size);
            }

            if (layer.post_attention_layernorm) |norm| {
                const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                self.device.rmsNorm(d_xb_ptr, norm_ptr, d_xb_ptr, dim, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            }

            const d_ffn_in_ptr = if (layer.pre_feedforward_layernorm) |norm| blk: {
                const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                self.device.addRmsNorm(d_x_ptr, d_xb_ptr, norm_ptr, d_xb_ptr, dim, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
                break :blk d_xb_ptr;
            } else blk: {
                self.device.add(d_x_ptr, d_xb_ptr, dim);
                break :blk d_x_ptr;
            };

            const inter_dim = if (layer.ffn_gate) |g| g.rows else dim * 4;

            if (layer.ffn_gate) |t_gate| {
                if (layer.ffn_up) |t_up| {
                    if (t_gate.qtype == .Q4_0 and t_up.qtype == .Q4_0) {
                        self.device.gemvGegluQ4_0(t_gate.buf.ptr, t_up.buf.ptr, d_ffn_in_ptr, d_act_ptr, inter_dim, dim);
                    } else {
                        self.device.gemv(t_gate.qtype, t_gate.buf.ptr, d_ffn_in_ptr, d_gate_ptr, inter_dim, dim);
                        self.device.gemv(t_up.qtype, t_up.buf.ptr, d_ffn_in_ptr, d_up_ptr, inter_dim, dim);
                        self.device.geglu(d_gate_ptr, d_up_ptr, d_act_ptr, inter_dim);
                    }
                }
            }

            if (layer.ffn_down) |t_down| {
                self.device.gemv(t_down.qtype, t_down.buf.ptr, d_act_ptr, d_ffn_out_ptr, dim, inter_dim);
            }

            if (layer.post_feedforward_layernorm) |norm| {
                const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                self.device.rmsnormAdd(d_x_ptr, d_ffn_out_ptr, norm_ptr, dim, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            } else {
                self.device.add(d_x_ptr, d_ffn_out_ptr, dim);
            }

            if (layer.per_layer_input_gate != null and layer.per_layer_projection != null) {
                const ple_slice_ptr = d_ctx_ple_ptr + layer_idx * ple_dim;
                const t_gate = layer.per_layer_input_gate.?;
                self.device.gemv(t_gate.qtype, t_gate.buf.ptr, d_x_ptr, d_ple_gate_ptr, ple_dim, dim);
                self.device.pleGateGelu(d_ple_gate_ptr, ple_slice_ptr, d_ple_buf_ptr, ple_dim);
                const t_proj = layer.per_layer_projection.?;
                self.device.gemv(t_proj.qtype, t_proj.buf.ptr, d_ple_buf_ptr, d_xb_ptr, dim, ple_dim);
                if (layer.post_per_layer_input_norm) |norm| {
                    const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                    self.device.rmsnormAdd(d_x_ptr, d_xb_ptr, norm_ptr, dim, p.layer_norm_rms_epsilon, false);
                } else {
                    self.device.add(d_x_ptr, d_xb_ptr, dim);
                }
            }

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

        // 4. Output Logits Projection (LM Head)
        const d_logits_ptr: [*]f32 = @ptrCast(@alignCast(self.d_logits.ptr));
        if (self.output) |t_out| {
            self.device.gemv(t_out.qtype, t_out.buf.ptr, d_xb_ptr, d_logits_ptr, p.vocab_size, dim);
        }

        // 5. Argmax on GPU
        const d_argmax_ptr: [*]c_uint = @ptrCast(@alignCast(self.d_argmax.ptr));
        self.device.argmax(d_logits_ptr, p.vocab_size, d_argmax_ptr);
    }

    pub fn forwardArgmax(
        self: *CudaGpuModel,
        cpu_model: *const TransformerModel,
        token_id: u32,
        pos: usize,
        kv_cache: ?*KVCache,
        bufs: *ModelBuffers,
        custom_embedding: ?[]const f32,
    ) !u32 {
        // Fast-path: CUDA Graph Execution for standard autoregressive decoding
        const can_use_graph = (custom_embedding == null and self.token_embd != null);
        if (can_use_graph) {
            // Upload dynamic decode parameters (token_id and pos)
            self.host_params[0] = @intCast(token_id);
            self.host_params[1] = @intCast(pos);
            _ = cuda.cuda_memcpy_h2d(self.d_params.ptr, self.host_params, 2 * @sizeOf(c_int), self.device.stream);

            // Gemma 4: If PLE table is stored in host memory, dequantize and upload the token's PLE row
            if (self.embed_tokens_per_layer == null) {
                if (cpu_model.embed_tokens_per_layer) |ple_tab| {
                    const ple_dim: usize = 256;
                    const total_ple_dim = self.layers.len * ple_dim;
                    if (token_id < self.params.vocab_size) {
                        const ple_row = ple_tab.getRow(token_id);
                        quant.dequantizeRow(ple_tab.type, ple_row, bufs.ctx_ple_buf[0..total_ple_dim], total_ple_dim);
                        const token_scale = @sqrt(@as(f32, @floatFromInt(ple_dim)));
                        for (bufs.ctx_ple_buf[0..total_ple_dim]) |*v| v.* *= token_scale;
                    } else {
                        @memset(bufs.ctx_ple_buf[0..total_ple_dim], 0.0);
                    }
                    const ple_bytes = std.mem.sliceAsBytes(bufs.ctx_ple_buf[0..total_ple_dim]);
                    _ = cuda.cuda_memcpy_h2d(self.d_ctx_ple_buf.ptr, ple_bytes.ptr, ple_bytes.len, self.device.stream);
                }
            }

            // If already captured, launch graph directly!
            if (self.graph_exec) |exec| {
                _ = cuda.cuda_graph_launch(exec, self.device.stream);
                _ = cuda.cuda_memcpy_d2h(self.host_out_tok, self.d_argmax.ptr, @sizeOf(c_uint), self.device.stream);
                self.device.sync();
                return @intCast(self.host_out_tok.*);
            }

            // If not yet captured, capture graph on stream

            const capture_res = cuda.cuda_graph_begin_capture(self.device.stream);
            if (capture_res == 0) {
                self.forwardGpuDecodeStep();
                var g: cuda.CudaGraph_t = null;
                const end_res = cuda.cuda_graph_end_capture(self.device.stream, &g);
                if (end_res == 0 and g != null) {
                    var exec: cuda.CudaGraphExec_t = null;
                    const inst_res = cuda.cuda_graph_instantiate(&exec, g);
                    if (inst_res == 0 and exec != null) {
                        self.graph = g;
                        self.graph_exec = exec;

                        // First launch using the newly instantiated graph!
                        _ = cuda.cuda_graph_launch(exec, self.device.stream);
                        _ = cuda.cuda_memcpy_d2h(self.host_out_tok, self.d_argmax.ptr, @sizeOf(c_uint), self.device.stream);
                        self.device.sync();
                        return @intCast(self.host_out_tok.*);
                    } else {
                        std.debug.print("⚠️ CUDA Graph instantiate failed: res={d}\n", .{inst_res});
                        if (g != null) _ = cuda.cuda_graph_destroy(g);
                    }
                } else {
                    std.debug.print("⚠️ CUDA Graph end_capture failed: res={d}\n", .{end_res});
                }
            } else {
                std.debug.print("⚠️ CUDA Graph begin_capture failed: res={d}\n", .{capture_res});
            }
        }

        // Fallback: regular non-graphed forward
        const p = self.params;
        const dim = p.embedding_length;
        _ = try self.forward(cpu_model, token_id, pos, kv_cache.?, bufs, custom_embedding, false);

        const d_xb_ptr: [*]f32 = @ptrCast(@alignCast(self.d_xb.ptr));
        const d_logits_ptr: [*]f32 = @ptrCast(@alignCast(self.d_logits.ptr));
        if (self.output) |t_out| {
            self.device.gemv(t_out.qtype, t_out.buf.ptr, d_xb_ptr, d_logits_ptr, p.vocab_size, dim);
        }
        if (p.final_logit_softcapping > 0.0) {
            self.device.tanhSoftcap(d_logits_ptr, p.final_logit_softcapping, p.vocab_size);
        }

        const d_argmax_ptr: [*]c_uint = @ptrCast(@alignCast(self.d_argmax.ptr));
        self.device.argmax(d_logits_ptr, p.vocab_size, d_argmax_ptr);
        var out_tok: c_uint = 0;
        try self.d_argmax.download(std.mem.asBytes(&out_tok), self.device.stream);
        self.device.sync();
        return @intCast(out_tok);
    }

    pub fn forwardBatch(
        self: *CudaGpuModel,
        cpu_model: *const TransformerModel,
        tokens: []const u32,
        embs: ?[]const f32,
        pos: usize,
        kv_cache: ?*KVCache,
        bufs: *ModelBuffers,
    ) ![]const f32 {
        if (tokens.len == 0) return self.host_logits[0..0];
        if (tokens.len == 1) {
            const emb_single = if (embs != null) embs.?[0..self.params.embedding_length] else null;
            return self.forward(cpu_model, tokens[0], pos, kv_cache.?, bufs, emb_single, true);
        }

        const B = tokens.len;
        const p = self.params;
        const dim = p.embedding_length;
        const d_x_ptr: [*]f32 = @ptrCast(@alignCast(self.d_x.ptr));
        const d_xb_ptr: [*]f32 = @ptrCast(@alignCast(self.d_xb.ptr));

        // 1. Embeddings lookup and batch upload
        const embd_scale: f32 = if (p.arch == .gemma or p.arch == .gemma2 or p.arch == .gemma4)
            @sqrt(@as(f32, @floatFromInt(dim)))
        else
            1.0;

        if (embs) |precomputed_embs| {
            // Use provided dense embeddings directly (caller must apply scaling if needed)
            @memcpy(self.host_x[0 .. B * dim], precomputed_embs[0 .. B * dim]);
            try self.d_x.upload(std.mem.sliceAsBytes(self.host_x[0 .. B * dim]), self.device.stream);
        } else if (self.token_embd) |emb_tensor| {
            // Fast 100% GPU Batched Embedding Lookup!
            var token_ids_buf: [512]c_int = undefined;
            for (tokens, 0..) |tok, b| {
                token_ids_buf[b] = @intCast(tok);
            }
            try self.d_tokens.upload(std.mem.sliceAsBytes(token_ids_buf[0..B]), self.device.stream);
            const d_tokens_ptr: [*]const c_int = @ptrCast(@alignCast(self.d_tokens.ptr));
            self.device.embedLookupBatch(
                emb_tensor.buf.ptr,
                emb_tensor.qtype,
                d_tokens_ptr,
                d_x_ptr,
                B,
                dim,
                embd_scale,
            );
        } else {
            // Lookup from token embedding table (CPU fallback)
            for (tokens, 0..) |tok, b| {
                const dst = self.host_x[b * dim .. (b + 1) * dim];
                const row_bytes = cpu_model.token_embd.getRow(tok);
                quant.dequantizeRow(cpu_model.token_embd.type, row_bytes, dst, dim);
                for (dst) |*v| v.* *= embd_scale;
            }
            try self.d_x.upload(std.mem.sliceAsBytes(self.host_x[0 .. B * dim]), self.device.stream);
        }

        // PLE precomputation for Gemma 4
        const ple_dim: usize = 256;
        const total_ple_dim = self.layers.len * ple_dim;
        const d_ctx_ple_ptr: [*]f32 = @ptrCast(@alignCast(self.d_ctx_ple_buf.ptr));
        const d_ctx_scratch_ptr: [*]f32 = @ptrCast(@alignCast(self.d_ctx_scratch.ptr));

        if (self.embed_tokens_per_layer) |ple_tensor| {
            const token_scale = @sqrt(@as(f32, @floatFromInt(ple_dim)));
            var token_ids_buf: [512]c_int = undefined;
            for (tokens, 0..) |tok, b| {
                token_ids_buf[b] = @intCast(tok);
            }
            try self.d_tokens.upload(std.mem.sliceAsBytes(token_ids_buf[0..B]), self.device.stream);
            const d_tokens_ptr: [*]const c_int = @ptrCast(@alignCast(self.d_tokens.ptr));
            self.device.embedLookupBatch(
                ple_tensor.buf.ptr,
                ple_tensor.qtype,
                d_tokens_ptr,
                d_ctx_ple_ptr,
                B,
                total_ple_dim,
                token_scale,
            );

            if (self.per_layer_model_projection) |ctx_proj| {
                const inv_sqrt_dim: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
                _ = cuda.cuda_memcpy_d2d(self.d_xb.ptr, self.d_x.ptr, B * dim * @sizeOf(f32), self.device.stream);
                self.device.scale(d_xb_ptr, inv_sqrt_dim, B * dim);

                // GPU Batched GEMM: ctx_scratch = ctx_proj * d_xb (B vectors in parallel on GPU)
                self.device.gemm(ctx_proj.qtype, ctx_proj.buf.ptr, d_xb_ptr, d_ctx_scratch_ptr, B, total_ple_dim, dim);

                // Batched RMSNorm across all B * layers.len
                if (self.per_layer_projection_norm) |norm| {
                    const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                    self.device.rmsNormBatched(d_ctx_scratch_ptr, norm_ptr, d_ctx_scratch_ptr, ple_dim, B * self.layers.len, p.layer_norm_rms_epsilon, false);
                }

                // Fusion on GPU across B * total_ple_dim
                self.device.pleCtxFuse(d_ctx_ple_ptr, d_ctx_scratch_ptr, B * total_ple_dim, true);
            }
        } else if (cpu_model.embed_tokens_per_layer) |ple_tab| {
            const token_scale = @sqrt(@as(f32, @floatFromInt(ple_dim)));
            for (tokens, 0..) |tok, b| {
                const dst = self.host_ple[b * total_ple_dim .. (b + 1) * total_ple_dim];
                if (tok < p.vocab_size) {
                    const ple_row = ple_tab.getRow(tok);
                    quant.dequantizeRow(ple_tab.type, ple_row, dst, total_ple_dim);
                    for (dst) |*v| v.* *= token_scale;
                } else {
                    @memset(dst, 0.0);
                }
            }
            try self.d_ctx_ple_buf.upload(std.mem.sliceAsBytes(self.host_ple[0 .. B * total_ple_dim]), self.device.stream);

            if (self.per_layer_model_projection) |ctx_proj| {
                const inv_sqrt_dim: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
                _ = cuda.cuda_memcpy_d2d(self.d_xb.ptr, self.d_x.ptr, B * dim * @sizeOf(f32), self.device.stream);
                self.device.scale(d_xb_ptr, inv_sqrt_dim, B * dim);

                // GPU Batched GEMM: ctx_scratch = ctx_proj * d_xb (B vectors in parallel on GPU)
                self.device.gemm(ctx_proj.qtype, ctx_proj.buf.ptr, d_xb_ptr, d_ctx_scratch_ptr, B, total_ple_dim, dim);

                // Batched RMSNorm across all B * layers.len
                if (self.per_layer_projection_norm) |norm| {
                    const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                    self.device.rmsNormBatched(d_ctx_scratch_ptr, norm_ptr, d_ctx_scratch_ptr, ple_dim, B * self.layers.len, p.layer_norm_rms_epsilon, false);
                }

                // Fusion on GPU across B * total_ple_dim
                self.device.pleCtxFuse(d_ctx_ple_ptr, d_ctx_scratch_ptr, B * total_ple_dim, true);
            }
        }

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
        const d_k_cache_ptr: [*]f32 = @ptrCast(@alignCast(self.d_k_cache.ptr));
        const d_v_cache_ptr: [*]f32 = @ptrCast(@alignCast(self.d_v_cache.ptr));

        // 2. Transformer layers forward (100% on GPU, ZERO host transfers!)
        for (self.layers, 0..) |layer, layer_idx| {
            // A. Pre-Attention Norm
            if (layer.input_layernorm) |norm| {
                const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                self.device.rmsNormBatched(d_x_ptr, norm_ptr, d_xb_ptr, dim, B, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            } else {
                _ = cuda.cuda_memcpy_d2d(self.d_xb.ptr, self.d_x.ptr, B * dim * @sizeOf(f32), self.device.stream);
            }

            const head_size = layer.head_dim;
            const n_heads = layer.n_heads;
            const n_kv_heads = layer.n_kv_heads;

            if (layer.attn_q) |t_q| {
                self.device.gemm(t_q.qtype, t_q.buf.ptr, d_xb_ptr, d_q_ptr, B, n_heads * head_size, dim);
            }

            if (layer.attn_q_norm) |q_norm| {
                const q_norm_ptr: [*]const f32 = @ptrCast(@alignCast(q_norm.buf.ptr));
                self.device.rmsNormBatched(d_q_ptr, q_norm_ptr, d_q_ptr, head_size, B * n_heads, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            }

            // KV Cache Handling (with Cross-Layer Sharing for Gemma 4 & hybrid architectures)
            const unshared_count: usize = if (p.num_kv_shared_layers > 0)
                (p.block_count - p.num_kv_shared_layers)
            else if (p.arch == .gemma4)
                (if (p.block_count == 42) 24 else 15)
            else
                p.block_count;

            const is_kv_shared = (p.arch == .gemma4 and (layer.attn_k == null or layer_idx >= unshared_count));
            var donor_layer: usize = layer_idx;
            if (is_kv_shared) {
                var l = unshared_count;
                while (l > 0) {
                    l -= 1;
                    if (self.layers[l].head_dim == layer.head_dim) {
                        donor_layer = l;
                        break;
                    }
                }
            } else {
                if (layer.attn_k) |t_k| {
                    self.device.gemm(t_k.qtype, t_k.buf.ptr, d_xb_ptr, d_k_ptr, B, n_kv_heads * head_size, dim);
                }
                if (layer.attn_v) |t_v| {
                    self.device.gemm(t_v.qtype, t_v.buf.ptr, d_xb_ptr, d_v_ptr, B, n_kv_heads * head_size, dim);
                }

                if (layer.attn_k_norm) |k_norm| {
                    const k_norm_ptr: [*]const f32 = @ptrCast(@alignCast(k_norm.buf.ptr));
                    self.device.rmsNormBatched(d_k_ptr, k_norm_ptr, d_k_ptr, head_size, B * n_kv_heads, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
                }

                if (p.arch == .gemma4) {
                    self.device.rmsNormBatched(d_v_ptr, null, d_v_ptr, head_size, B * n_kv_heads, p.layer_norm_rms_epsilon, false);
                }

                self.device.ropeBatched(null, d_k_ptr, pos, B, 0, n_kv_heads, head_size, layer.rotary_dim, layer.rope_theta);

                // Store K and V into KV Cache for all B tokens
                self.device.kvCachePutBatched(
                    d_k_cache_ptr,
                    d_v_cache_ptr,
                    d_k_ptr,
                    d_v_ptr,
                    layer_idx,
                    pos,
                    B,
                    self.max_seq_len,
                    n_kv_heads,
                    head_size,
                    self.max_kv_dim,
                );
            }

            self.device.ropeBatched(d_q_ptr, null, pos, B, n_heads, 0, head_size, layer.rotary_dim, layer.rope_theta);

            // C. Multi-Head Attention (Batched across heads AND batch tokens)
            const donor_kv_heads = self.layers[donor_layer].n_kv_heads;
            const attn_scale: f32 = if (p.arch == .gemma4) 1.0 else 1.0 / @sqrt(@as(f32, @floatFromInt(head_size)));
            self.device.attentionBatched(
                d_q_ptr,
                d_k_cache_ptr,
                d_v_cache_ptr,
                d_attn_out_ptr,
                donor_layer,
                pos,
                B,
                self.max_seq_len,
                n_heads,
                donor_kv_heads,
                head_size,
                self.max_kv_dim,
                attn_scale,
                p.attn_logit_softcapping,
                layer.sliding_window,
            );

            // D. Attention Output Projection (GEMM)
            if (layer.attn_output) |t_out| {
                self.device.gemm(t_out.qtype, t_out.buf.ptr, d_attn_out_ptr, d_xb_ptr, B, dim, n_heads * head_size);
            }

            // Post-attention layernorm
            if (layer.post_attention_layernorm) |norm| {
                const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                self.device.rmsNormBatched(d_xb_ptr, norm_ptr, d_xb_ptr, dim, B, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            }

            // E. Feed-Forward Network
            if (layer.pre_feedforward_layernorm) |norm| {
                const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                self.device.addRmsNormBatched(d_x_ptr, d_xb_ptr, norm_ptr, d_xb_ptr, dim, B, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            } else {
                self.device.add(d_x_ptr, d_xb_ptr, B * dim);
                _ = cuda.cuda_memcpy_d2d(self.d_xb.ptr, self.d_x.ptr, B * dim * @sizeOf(f32), self.device.stream);
            }

            const inter_size = if (layer.ffn_gate) |g| g.rows else dim * 4;

            if (layer.ffn_gate) |t_gate| {
                self.device.gemm(t_gate.qtype, t_gate.buf.ptr, d_xb_ptr, d_gate_ptr, B, inter_size, dim);
            }
            if (layer.ffn_up) |t_up| {
                self.device.gemm(t_up.qtype, t_up.buf.ptr, d_xb_ptr, d_up_ptr, B, inter_size, dim);
            }

            self.device.geglu(d_gate_ptr, d_up_ptr, d_act_ptr, B * inter_size);

            if (layer.ffn_down) |t_down| {
                self.device.gemm(t_down.qtype, t_down.buf.ptr, d_act_ptr, d_ffn_out_ptr, B, dim, inter_size);
            }

            if (layer.post_feedforward_layernorm) |norm| {
                const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                self.device.rmsNormBatched(d_ffn_out_ptr, norm_ptr, d_ffn_out_ptr, dim, B, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            }

            // Residual Add: X = X + FFN_Out
            self.device.add(d_x_ptr, d_ffn_out_ptr, B * dim);

            // Gemma Per-Layer Embedding Branch (if present)
            if (layer.per_layer_input_gate != null and layer.per_layer_projection != null) {
                const t_gate = layer.per_layer_input_gate.?;
                self.device.gemm(t_gate.qtype, t_gate.buf.ptr, d_x_ptr, d_ple_gate_ptr, B, 256, dim);

                for (0..B) |b| {
                    const ple_slice = d_ctx_ple_ptr + (b * self.layers.len + layer_idx) * 256;
                    const ple_gate_b = d_ple_gate_ptr + b * 256;
                    const ple_buf_b = d_ple_buf_ptr + b * 256;
                    self.device.pleGateGelu(ple_gate_b, ple_slice, ple_buf_b, 256);
                }

                const t_proj = layer.per_layer_projection.?;
                self.device.gemm(t_proj.qtype, t_proj.buf.ptr, d_ple_buf_ptr, d_xb_ptr, B, dim, 256);

                if (layer.post_per_layer_input_norm) |norm| {
                    const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
                    self.device.rmsNormBatched(d_xb_ptr, norm_ptr, d_xb_ptr, dim, B, p.layer_norm_rms_epsilon, false);
                }

                self.device.add(d_x_ptr, d_xb_ptr, B * dim);
            }

            if (layer.scale != 1.0) {
                self.device.scale(d_x_ptr, layer.scale, B * dim);
            }
        }

        // 3. Final Output Norm on the LAST token in the batch (at d_x + (B - 1) * dim)
        const last_token_x_ptr = d_x_ptr + (B - 1) * dim;
        if (self.output_norm) |norm| {
            const norm_ptr: [*]const f32 = @ptrCast(@alignCast(norm.buf.ptr));
            self.device.rmsNorm(last_token_x_ptr, norm_ptr, d_xb_ptr, dim, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
        } else {
            _ = cuda.cuda_memcpy_d2d(self.d_xb.ptr, last_token_x_ptr, dim * @sizeOf(f32), self.device.stream);
        }

        // 4. Output Logits Projection (LM Head) for the last token
        const d_logits_ptr: [*]f32 = @ptrCast(@alignCast(self.d_logits.ptr));
        if (self.output) |t_out| {
            self.device.gemv(t_out.qtype, t_out.buf.ptr, d_xb_ptr, d_logits_ptr, p.vocab_size, dim);
        }

        if (p.final_logit_softcapping > 0.0) {
            self.device.tanhSoftcap(d_logits_ptr, p.final_logit_softcapping, p.vocab_size);
        }

        const logits_bytes: []u8 = std.mem.sliceAsBytes(self.host_logits);
        try self.d_logits.download(logits_bytes, self.device.stream);
        self.device.sync();

        return self.host_logits;
    }

    pub fn forwardBatchArgmax(
        self: *CudaGpuModel,
        cpu_model: *const TransformerModel,
        tokens: []const u32,
        embs: ?[]const f32,
        pos: usize,
        kv_cache: ?*KVCache,
        bufs: *ModelBuffers,
    ) !u32 {
        if (tokens.len == 0) return 0;
        if (tokens.len == 1) {
            const emb_single = if (embs != null) embs.?[0..self.params.embedding_length] else null;
            return self.forwardArgmax(cpu_model, tokens[0], pos, kv_cache, bufs, emb_single);
        }

        _ = try self.forwardBatch(cpu_model, tokens, embs, pos, kv_cache, bufs);

        const p = self.params;
        const d_logits_ptr: [*]f32 = @ptrCast(@alignCast(self.d_logits.ptr));
        const d_argmax_ptr: [*]c_uint = @ptrCast(@alignCast(self.d_argmax.ptr));
        self.device.argmax(d_logits_ptr, p.vocab_size, d_argmax_ptr);
        var out_tok: c_uint = 0;
        try self.d_argmax.download(std.mem.asBytes(&out_tok), self.device.stream);
        self.device.sync();
        return @intCast(out_tok);
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

    // Test forwardBatch vs sequential forward
    const test_batch = [_]u32{ 100, 200, 300, 400 };
    gpu_eng.reset();
    const seq_final = try allocator.alloc(f32, cpu_eng.params.vocab_size);
    defer allocator.free(seq_final);
    for (test_batch, 0..) |tok, p_idx| {
        const l = try gpu_eng.gpu_model.?.forward(gpu_eng.model, tok, p_idx, gpu_eng.kv_cache, gpu_eng.buffers, null, p_idx == test_batch.len - 1);
        if (p_idx == test_batch.len - 1) {
            @memcpy(seq_final[0..l.len], l);
        }
    }

    gpu_eng.reset();
    const bat_logits = try gpu_eng.gpu_model.?.forwardBatch(gpu_eng.model, &test_batch, null, 0, gpu_eng.kv_cache, gpu_eng.buffers);

    var batch_diff: f32 = 0.0;
    var b_max_idx: usize = 0;
    for (0..bat_logits.len) |i| {
        const diff = @abs(seq_final[i] - bat_logits[i]);
        if (diff > batch_diff) {
            batch_diff = diff;
            b_max_idx = i;
        }
    }
    std.debug.print("[CUDA C ABI Batch Parity] Max diff: {d:.6} at index {d} (Seq: {d:.4}, Bat: {d:.4})\n", .{ batch_diff, b_max_idx, seq_final[b_max_idx], bat_logits[b_max_idx] });
    try std.testing.expect(batch_diff < 1.0);

    // Test forwardArgmax and forwardBatchArgmax
    gpu_eng.reset();
    const gpu_argmax_kernel = try gpu_eng.gpu_model.?.forwardArgmax(gpu_eng.model, test_tok, pos, gpu_eng.kv_cache, gpu_eng.buffers, null);
    std.debug.print("GPU direct argmax: {d} ('{s}')\n", .{ gpu_argmax_kernel, gpu_eng.tokenizer.decode(gpu_argmax_kernel) });
    try std.testing.expectEqual(cpu_argmax, gpu_argmax_kernel);

    gpu_eng.reset();
    const batch_argmax_kernel = try gpu_eng.gpu_model.?.forwardBatchArgmax(gpu_eng.model, &test_batch, null, 0, gpu_eng.kv_cache, gpu_eng.buffers);
    var bat_max_val: f32 = -1e9;
    var bat_argmax: u32 = 0;
    for (0..bat_logits.len) |i| {
        if (bat_logits[i] > bat_max_val) {
            bat_max_val = bat_logits[i];
            bat_argmax = @intCast(i);
        }
    }
    std.debug.print("GPU batch argmax: {d}, kernel batch argmax: {d}\n", .{ bat_argmax, batch_argmax_kernel });
    try std.testing.expectEqual(bat_argmax, batch_argmax_kernel);
}
