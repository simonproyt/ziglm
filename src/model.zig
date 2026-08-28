const std = @import("std");
const types = @import("types.zig");
const GGMLType = types.GGMLType;
const Tensor = types.Tensor;
const ModelParams = types.ModelParams;
const Architecture = types.Architecture;
const GGUFFile = @import("gguf.zig").GGUFFile;
const SafeTensorsFile = @import("safetensors.zig").SafeTensorsFile;
const KVCache = @import("kv_cache.zig").KVCache;
const ThreadPool = @import("thread_pool.zig").ThreadPool;
const math = @import("math.zig");
const quant = @import("quant.zig");
const vision = @import("vision.zig");
const audio = @import("audio.zig");
const VisionEncoder = vision.VisionEncoder;
const VisionLayerWeights = vision.VisionLayerWeights;

pub const LayerWeights = struct {
    // Layernorms
    input_layernorm: ?[]const f32 = null,
    post_attention_layernorm: ?[]const f32 = null,
    pre_feedforward_layernorm: ?[]const f32 = null,
    post_feedforward_layernorm: ?[]const f32 = null,
    post_per_layer_input_norm: ?[]const f32 = null,

    // Gating / Scaling
    layer_scalar: f32 = 1.0,

    // Per-layer embeddings projections
    per_layer_input_gate: ?Tensor = null,
    per_layer_projection: ?Tensor = null,

    // Attention weights
    attn_q: ?Tensor = null,
    attn_k: ?Tensor = null,
    attn_v: ?Tensor = null,
    attn_output: ?Tensor = null,
    attn_q_norm: ?[]const f32 = null,
    attn_k_norm: ?[]const f32 = null,

    // Attention parameters for this layer
    head_dim: usize = 0,
    n_heads: usize = 0,
    n_kv_heads: usize = 0,
    rope_theta: f32 = 10000.0,
    rotary_dim: usize = 0,
    sliding_window: usize = 0,

    // FFN projections
    ffn_gate: ?Tensor = null,
    ffn_up: ?Tensor = null,
    ffn_down: ?Tensor = null,
    intermediate_size: usize = 0,
};

pub const ModelBuffers = struct {
    allocator: std.mem.Allocator,
    x: []f32, // [embedding_length]
    xb: []f32, // [embedding_length]
    q: []f32, // [max_q_dim]
    k: []f32, // [max_kv_dim]
    v: []f32, // [max_kv_dim]
    attn_scores: []f32, // [max_seq_len]
    attn_out: []f32, // [max_q_dim]
    gate: []f32, // [max_ffn]
    up: []f32, // [max_ffn]
    ffn_out: []f32, // [embedding_length]
    ple_buf: []f32, // [256]
    ple_gate: []f32, // [256]
    ctx_ple_buf: []f32, // [num_layers * 256]
    ctx_scratch: []f32, // [num_layers * 256]
    logits: []f32, // [vocab_size]

    pub fn init(allocator: std.mem.Allocator, params: *const ModelParams, max_seq_len: usize) !*ModelBuffers {
        const self = try allocator.create(ModelBuffers);
        const q_len = @max(4096, params.head_count * params.head_size);
        const kv_len = @max(1024, params.head_count_kv * params.head_size);
        const max_ffn = @max(16384, params.feed_forward_length * 2);

        const ple_total = @max(256, params.block_count * 256);
        self.* = .{
            .allocator = allocator,
            .x = try allocator.alloc(f32, params.embedding_length),
            .xb = try allocator.alloc(f32, params.embedding_length),
            .q = try allocator.alloc(f32, q_len),
            .k = try allocator.alloc(f32, kv_len),
            .v = try allocator.alloc(f32, kv_len),
            .attn_scores = try allocator.alloc(f32, max_seq_len),
            .attn_out = try allocator.alloc(f32, q_len),
            .gate = try allocator.alloc(f32, max_ffn),
            .up = try allocator.alloc(f32, max_ffn),
            .ffn_out = try allocator.alloc(f32, params.embedding_length),
            .ple_buf = try allocator.alloc(f32, 256),
            .ple_gate = try allocator.alloc(f32, 256),
            .ctx_ple_buf = try allocator.alloc(f32, ple_total),
            .ctx_scratch = try allocator.alloc(f32, ple_total),
            .logits = try allocator.alloc(f32, params.vocab_size),
        };
        return self;
    }

    pub fn deinit(self: *ModelBuffers) void {
        self.allocator.free(self.x);
        self.allocator.free(self.xb);
        self.allocator.free(self.q);
        self.allocator.free(self.k);
        self.allocator.free(self.v);
        self.allocator.free(self.attn_scores);
        self.allocator.free(self.attn_out);
        self.allocator.free(self.gate);
        self.allocator.free(self.up);
        self.allocator.free(self.ffn_out);
        self.allocator.free(self.ple_buf);
        self.allocator.free(self.ple_gate);
        self.allocator.free(self.ctx_ple_buf);
        self.allocator.free(self.ctx_scratch);
        self.allocator.free(self.logits);
        self.allocator.destroy(self);
    }
};

pub const TransformerModel = struct {
    allocator: std.mem.Allocator,
    params: ModelParams,
    token_embd: Tensor,
    embed_tokens_per_layer: ?Tensor = null,
    per_layer_model_projection: ?Tensor = null,
    per_layer_projection_norm: ?[]const f32 = null,
    output_norm: []const f32,
    output: ?Tensor = null,
    layers: []LayerWeights,
    audio_encoder: ?audio.AudioEncoder = null,
    vision_encoder: ?VisionEncoder = null,

    fn loadNorm(allocator: std.mem.Allocator, t_opt: ?Tensor, len: usize) !?[]const f32 {
        if (t_opt) |t| {
            const act_len = if (t.elements() > 0) t.elements() else len;
            const slice = try allocator.alloc(f32, act_len);
            quant.dequantizeRow(t.type, t.data, slice, act_len);
            return slice;
        }
        return null;
    }

    fn loadScalar(t_opt: ?Tensor) f32 {
        if (t_opt) |t_scalar| {
            if (t_scalar.type == .BF16 and t_scalar.data.len >= 2) {
                const u = std.mem.readInt(u16, t_scalar.data[0..2], .little);
                return @as(f32, @bitCast(@as(u32, u) << 16));
            } else if (t_scalar.type == .F16 and t_scalar.data.len >= 2) {
                const u = std.mem.readInt(u16, t_scalar.data[0..2], .little);
                return @floatCast(@as(f16, @bitCast(u)));
            } else if (t_scalar.type == .F32 and t_scalar.data.len >= 4) {
                const u = std.mem.readInt(u32, t_scalar.data[0..4], .little);
                return @bitCast(u);
            }
        }
        return 1.0;
    }

    pub fn load(allocator: std.mem.Allocator, gguf: *const GGUFFile) !*TransformerModel {
        const token_embd = gguf.getTensor("token_embd.weight") orelse return error.MissingTokenEmbeddingTensor;
        const out_norm_t = gguf.getTensor("output_norm.weight") orelse return error.MissingOutputNormTensor;

        const dim = gguf.params.embedding_length;
        const head_size = gguf.params.head_size;

        const out_norm_slice = try allocator.alloc(f32, dim);
        errdefer allocator.free(out_norm_slice);
        quant.dequantizeRow(out_norm_t.type, out_norm_t.data, out_norm_slice, dim);

        const self = try allocator.create(TransformerModel);
        self.* = .{
            .allocator = allocator,
            .params = gguf.params,
            .token_embd = token_embd,
            .embed_tokens_per_layer = gguf.getTensor("per_layer_token_embd.weight") orelse gguf.getTensor("embed_tokens_per_layer.weight"),
            .per_layer_model_projection = gguf.getTensor("per_layer_model_proj.weight") orelse gguf.getTensor("per_layer_model_projection.weight"),
            .per_layer_projection_norm = try loadNorm(allocator, gguf.getTensor("per_layer_proj_norm.weight") orelse gguf.getTensor("per_layer_projection_norm.weight"), 256),
            .output_norm = out_norm_slice,
            .output = gguf.getTensor("output.weight"),
            .layers = try allocator.alloc(LayerWeights, gguf.params.block_count),
        };
        errdefer self.deinit();

        var name_buf: [128]u8 = undefined;

        for (0..gguf.params.block_count) |i| {
            var layer = LayerWeights{};

            // 1. Attention Norm
            const in_norm_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm.weight", .{i}) catch continue;
            layer.input_layernorm = try loadNorm(allocator, gguf.getTensor(in_norm_name), dim);

            const post_attn_name = std.fmt.bufPrint(&name_buf, "blk.{d}.post_attention_norm.weight", .{i}) catch continue;
            layer.post_attention_layernorm = try loadNorm(allocator, gguf.getTensor(post_attn_name) orelse blk: {
                const alt_name = std.fmt.bufPrint(&name_buf, "blk.{d}.post_attn_norm.weight", .{i}) catch break :blk null;
                break :blk gguf.getTensor(alt_name);
            }, dim);

            // 2. Pre-FFN Norm & Post-FFN Norm
            const pre_ffn_name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_norm.weight", .{i}) catch continue;
            layer.pre_feedforward_layernorm = try loadNorm(allocator, gguf.getTensor(pre_ffn_name), dim);

            const post_ffn_name = std.fmt.bufPrint(&name_buf, "blk.{d}.post_ffw_norm.weight", .{i}) catch continue;
            layer.post_feedforward_layernorm = try loadNorm(allocator, gguf.getTensor(post_ffn_name) orelse blk: {
                const alt_name = std.fmt.bufPrint(&name_buf, "blk.{d}.post_ffn_norm.weight", .{i}) catch break :blk null;
                break :blk gguf.getTensor(alt_name);
            }, dim);

            // 3. PLE Tensors
            const post_ple_name = std.fmt.bufPrint(&name_buf, "blk.{d}.post_norm.weight", .{i}) catch continue;
            layer.post_per_layer_input_norm = try loadNorm(allocator, gguf.getTensor(post_ple_name) orelse blk: {
                const alt_name = std.fmt.bufPrint(&name_buf, "blk.{d}.post_per_layer_input_norm.weight", .{i}) catch break :blk null;
                break :blk gguf.getTensor(alt_name);
            }, dim);

            const ple_gate_name = std.fmt.bufPrint(&name_buf, "blk.{d}.inp_gate.weight", .{i}) catch continue;
            layer.per_layer_input_gate = gguf.getTensor(ple_gate_name) orelse blk: {
                const alt_name = std.fmt.bufPrint(&name_buf, "blk.{d}.per_layer_input_gate.weight", .{i}) catch break :blk null;
                break :blk gguf.getTensor(alt_name);
            };

            const ple_proj_name = std.fmt.bufPrint(&name_buf, "blk.{d}.proj.weight", .{i}) catch continue;
            layer.per_layer_projection = gguf.getTensor(ple_proj_name) orelse blk: {
                const alt_name = std.fmt.bufPrint(&name_buf, "blk.{d}.per_layer_projection.weight", .{i}) catch break :blk null;
                break :blk gguf.getTensor(alt_name);
            };

            // 4. Layer scalar
            const scalar_name = std.fmt.bufPrint(&name_buf, "blk.{d}.layer_output_scale.weight", .{i}) catch continue;
            layer.layer_scalar = loadScalar(gguf.getTensor(scalar_name) orelse blk: {
                const alt_name = std.fmt.bufPrint(&name_buf, "blk.{d}.layer_scalar", .{i}) catch break :blk null;
                break :blk gguf.getTensor(alt_name);
            });

            // 5. Q, K, V, Output
            const q_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_q.weight", .{i}) catch continue;
            layer.attn_q = gguf.getTensor(q_name);

            const k_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_k.weight", .{i}) catch continue;
            layer.attn_k = gguf.getTensor(k_name);

            const v_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_v.weight", .{i}) catch continue;
            layer.attn_v = gguf.getTensor(v_name);

            const out_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_output.weight", .{i}) catch continue;
            layer.attn_output = gguf.getTensor(out_name);

            // 6. Q / K Norms & Head Dim
            const q_norm_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_q_norm.weight", .{i}) catch continue;
            const q_norm_t = gguf.getTensor(q_norm_name);
            if (q_norm_t) |qnt| {
                layer.head_dim = qnt.elements();
                layer.attn_q_norm = try loadNorm(allocator, q_norm_t, layer.head_dim);
            } else {
                layer.head_dim = head_size;
            }

            const k_norm_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_k_norm.weight", .{i}) catch continue;
            layer.attn_k_norm = try loadNorm(allocator, gguf.getTensor(k_norm_name), layer.head_dim);

            if (layer.attn_q) |t_q| {
                layer.n_heads = t_q.elements() / dim / layer.head_dim;
            } else {
                layer.n_heads = gguf.params.head_count;
            }

            if (layer.attn_k) |t_k| {
                layer.n_kv_heads = t_k.elements() / dim / layer.head_dim;
            } else {
                layer.n_kv_heads = if (gguf.params.head_count_kv > 0) gguf.params.head_count_kv else layer.n_heads;
            }

            if (layer.head_dim >= 512) {
                layer.rope_theta = 1000000.0;
                layer.rotary_dim = 128; // 0.25 * 512
                layer.sliding_window = 0;
            } else {
                layer.rope_theta = 10000.0;
                layer.rotary_dim = layer.head_dim;
                layer.sliding_window = 512;
            }

            // 7. FFN Projections
            const gate_name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_gate.weight", .{i}) catch continue;
            layer.ffn_gate = gguf.getTensor(gate_name);

            const up_name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_up.weight", .{i}) catch continue;
            layer.ffn_up = gguf.getTensor(up_name);

            const down_name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_down.weight", .{i}) catch continue;
            layer.ffn_down = gguf.getTensor(down_name);

            if (layer.ffn_gate) |t_gate| {
                layer.intermediate_size = t_gate.elements() / dim;
            } else {
                layer.intermediate_size = gguf.params.feed_forward_length;
            }

            self.layers[i] = layer;
        }

        const patch_proj = gguf.getTensor("v.patch_embedder.input_proj.weight") orelse gguf.getTensor("vision.patch_embedder.input_proj.weight");
        const pos_emb = gguf.getTensor("v.patch_embedder.position_embedding_table") orelse gguf.getTensor("vision.patch_embedder.position_embedding_table");
        const emb_proj = gguf.getTensor("v.embedding_projection.weight") orelse gguf.getTensor("vision.embedding_projection.weight");

        self.vision_encoder = if (patch_proj != null or emb_proj != null)
            VisionEncoder.init(allocator, patch_proj, pos_emb, emb_proj, &[_]vision.VisionLayerWeights{})
        else
            null;

        return self;
    }

    pub fn loadMMPROJ(self: *TransformerModel, allocator: std.mem.Allocator, gguf: *const GGUFFile) !void {
        const patch_proj = gguf.getTensor("v.patch_embedder.input_proj.weight") orelse
            gguf.getTensor("v.patch_embd.weight") orelse
            gguf.getTensor("mm.patch_embd.weight");

        const pos_emb = gguf.getTensor("v.position_embd.weight") orelse
            gguf.getTensor("v.patch_embedder.position_embedding_table") orelse
            gguf.getTensor("v.position_embedding_table");

        const emb_proj = gguf.getTensor("mm.input_projection.weight") orelse
            gguf.getTensor("v.embedding_projection.weight") orelse
            gguf.getTensor("mm.0.weight");

        var vision_layers: std.ArrayList(vision.VisionLayerWeights) = .empty;
        errdefer vision_layers.deinit(allocator);

        for (0..32) |l_idx| {
            var buf: [128]u8 = undefined;
            const q_name = std.fmt.bufPrint(&buf, "v.blk.{d}.attn_q.weight", .{l_idx}) catch break;
            const q_proj = gguf.getTensor(q_name);
            if (q_proj == null) break;

            var layer = vision.VisionLayerWeights{};
            layer.q_proj = q_proj;

            var name_buf: [128]u8 = undefined;
            layer.input_layernorm = try loadNorm(allocator, gguf.getTensor(try std.fmt.bufPrint(&name_buf, "v.blk.{d}.ln1.weight", .{l_idx})), 768);
            layer.post_attention_layernorm = try loadNorm(allocator, gguf.getTensor(try std.fmt.bufPrint(&name_buf, "v.blk.{d}.ln2.weight", .{l_idx})), 768);
            layer.q_proj = gguf.getTensor(try std.fmt.bufPrint(&name_buf, "v.blk.{d}.attn_q.weight", .{l_idx}));
            layer.k_proj = gguf.getTensor(try std.fmt.bufPrint(&name_buf, "v.blk.{d}.attn_k.weight", .{l_idx}));
            layer.v_proj = gguf.getTensor(try std.fmt.bufPrint(&name_buf, "v.blk.{d}.attn_v.weight", .{l_idx}));
            layer.o_proj = gguf.getTensor(try std.fmt.bufPrint(&name_buf, "v.blk.{d}.attn_out.weight", .{l_idx}));
            layer.gate_proj = gguf.getTensor(try std.fmt.bufPrint(&name_buf, "v.blk.{d}.ffn_gate.weight", .{l_idx}));
            layer.up_proj = gguf.getTensor(try std.fmt.bufPrint(&name_buf, "v.blk.{d}.ffn_up.weight", .{l_idx}));
            layer.down_proj = gguf.getTensor(try std.fmt.bufPrint(&name_buf, "v.blk.{d}.ffn_down.weight", .{l_idx}));
            layer.q_norm = try loadNorm(allocator, gguf.getTensor(try std.fmt.bufPrint(&name_buf, "v.blk.{d}.attn_q_norm.weight", .{l_idx})), 72);
            layer.k_norm = try loadNorm(allocator, gguf.getTensor(try std.fmt.bufPrint(&name_buf, "v.blk.{d}.attn_k_norm.weight", .{l_idx})), 72);
            layer.pre_feedforward_layernorm = try loadNorm(allocator, gguf.getTensor(try std.fmt.bufPrint(&name_buf, "v.blk.{d}.ffn_pre_norm.weight", .{l_idx})), 768);
            layer.post_feedforward_layernorm = try loadNorm(allocator, gguf.getTensor(try std.fmt.bufPrint(&name_buf, "v.blk.{d}.ffn_post_norm.weight", .{l_idx})), 768);

            try vision_layers.append(allocator, layer);
        }

        const v_layers_slice = vision_layers.toOwnedSlice(allocator) catch |e| {
            vision_layers.deinit(allocator);
            return e;
        };

        if (patch_proj != null) {
            self.vision_encoder = vision.VisionEncoder.init(
                allocator,
                patch_proj,
                pos_emb,
                emb_proj,
                v_layers_slice,
            );
        }

        const a_conv0_w = gguf.getTensor("a.conv1d.0.weight");
        const a_conv0_n = try loadNorm(allocator, gguf.getTensor("a.conv1d.0.norm.weight"), 128);
        const a_conv1_w = gguf.getTensor("a.conv1d.1.weight");
        const a_conv1_n = try loadNorm(allocator, gguf.getTensor("a.conv1d.1.norm.weight"), 32);
        const a_inp_proj = gguf.getTensor("a.input_projection.weight");
        const a_pre_out = gguf.getTensor("a.pre_encode.out.weight");
        const a_pre_bias = try loadNorm(allocator, gguf.getTensor("a.pre_encode.out.bias"), 1536);
        const a_mm_proj = gguf.getTensor("mm.a.input_projection.weight");

        var audio_layers: std.ArrayList(audio.AudioLayerWeights) = .empty;
        errdefer audio_layers.deinit(allocator);

        for (0..32) |l_idx| {
            var buf: [128]u8 = undefined;
            const q_name = std.fmt.bufPrint(&buf, "a.blk.{d}.attn_q.weight", .{l_idx}) catch break;
            const q_proj = gguf.getTensor(q_name);
            if (q_proj == null) break;

            var layer = audio.AudioLayerWeights{};
            layer.attn_q = q_proj;

            var name_buf: [128]u8 = undefined;
            layer.ffn_norm = try loadNorm(allocator, gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.ffn_norm.weight", .{l_idx})), 1024);
            layer.ffn_up = gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.ffn_up.weight", .{l_idx}));
            layer.ffn_down = gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.ffn_down.weight", .{l_idx}));
            layer.ffn_post_norm = try loadNorm(allocator, gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.ffn_post_norm.weight", .{l_idx})), 1024);
            
            layer.attn_pre_norm = try loadNorm(allocator, gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.attn_pre_norm.weight", .{l_idx})), 1024);
            layer.attn_k = gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.attn_k.weight", .{l_idx}));
            layer.attn_v = gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.attn_v.weight", .{l_idx}));
            layer.attn_k_rel = gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.attn_k_rel.weight", .{l_idx}));
            layer.per_dim_scale = try loadNorm(allocator, gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.per_dim_scale.weight", .{l_idx})), 128);
            layer.attn_out = gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.attn_out.weight", .{l_idx}));
            layer.attn_post_norm = try loadNorm(allocator, gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.attn_post_norm.weight", .{l_idx})), 1024);
            
            layer.norm_conv = try loadNorm(allocator, gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.norm_conv.weight", .{l_idx})), 1024);
            layer.conv_pw1 = gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.conv_pw1.weight", .{l_idx}));
            layer.conv_dw = gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.conv_dw.weight", .{l_idx}));
            layer.conv_norm = try loadNorm(allocator, gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.conv_norm.weight", .{l_idx})), 1024);
            layer.conv_pw2 = gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.conv_pw2.weight", .{l_idx}));
            
            layer.ffn_norm_1 = try loadNorm(allocator, gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.ffn_norm_1.weight", .{l_idx})), 1024);
            layer.ffn_up_1 = gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.ffn_up_1.weight", .{l_idx}));
            layer.ffn_down_1 = gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.ffn_down_1.weight", .{l_idx}));
            layer.ffn_post_norm_1 = try loadNorm(allocator, gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.ffn_post_norm_1.weight", .{l_idx})), 1024);
            
            layer.ln2 = try loadNorm(allocator, gguf.getTensor(try std.fmt.bufPrint(&name_buf, "a.blk.{d}.ln2.weight", .{l_idx})), 1024);

            try audio_layers.append(allocator, layer);
        }

        const a_layers_slice = audio_layers.toOwnedSlice(allocator) catch |e| {
            audio_layers.deinit(allocator);
            return e;
        };

        if (a_conv0_w != null) {
            self.audio_encoder = audio.AudioEncoder.init(
                allocator,
                a_conv0_w,
                a_conv0_n,
                a_conv1_w,
                a_conv1_n,
                a_inp_proj,
                a_pre_out,
                a_pre_bias,
                a_mm_proj,
                a_layers_slice,
            );
        }
    }

    pub fn loadFromSafeTensors(allocator: std.mem.Allocator, params: ModelParams, st: *const SafeTensorsFile) !*TransformerModel {
        const token_embd = st.getTensor("token_embd.weight") orelse return error.MissingTokenEmbeddingTensor;
        const out_norm_t = st.getTensor("output_norm.weight") orelse return error.MissingOutputNormTensor;

        const dim = params.embedding_length;
        const head_size = params.head_size;

        const out_norm_slice = try allocator.alloc(f32, dim);
        errdefer allocator.free(out_norm_slice);
        quant.dequantizeRow(out_norm_t.type, out_norm_t.data, out_norm_slice, dim);

        const self = try allocator.create(TransformerModel);
        self.* = .{
            .allocator = allocator,
            .params = params,
            .token_embd = token_embd,
            .embed_tokens_per_layer = st.getTensor("embed_tokens_per_layer.weight"),
            .per_layer_model_projection = st.getTensor("per_layer_model_projection.weight"),
            .per_layer_projection_norm = try loadNorm(allocator, st.getTensor("per_layer_projection_norm.weight"), 256),
            .output_norm = out_norm_slice,
            .output = st.getTensor("output.weight"),
            .layers = try allocator.alloc(LayerWeights, params.block_count),
        };
        errdefer self.deinit();

        var name_buf: [128]u8 = undefined;

        for (0..params.block_count) |i| {
            var layer = LayerWeights{};

            // Layernorms
            const in_norm_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_norm.weight", .{i}) catch continue;
            layer.input_layernorm = try loadNorm(allocator, st.getTensor(in_norm_name), dim);

            const post_attn_name = std.fmt.bufPrint(&name_buf, "blk.{d}.post_attn_norm.weight", .{i}) catch continue;
            layer.post_attention_layernorm = try loadNorm(allocator, st.getTensor(post_attn_name), dim);

            const pre_ffn_name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_norm.weight", .{i}) catch continue;
            layer.pre_feedforward_layernorm = try loadNorm(allocator, st.getTensor(pre_ffn_name), dim);

            const post_ffn_name = std.fmt.bufPrint(&name_buf, "blk.{d}.post_ffn_norm.weight", .{i}) catch continue;
            layer.post_feedforward_layernorm = try loadNorm(allocator, st.getTensor(post_ffn_name), dim);

            const post_ple_name = std.fmt.bufPrint(&name_buf, "blk.{d}.post_per_layer_input_norm", .{i}) catch continue;
            layer.post_per_layer_input_norm = try loadNorm(allocator, st.getTensor(post_ple_name), dim);

            // Per-Layer Input Gate and Projection
            const ple_gate_name = std.fmt.bufPrint(&name_buf, "blk.{d}.per_layer_input_gate", .{i}) catch continue;
            layer.per_layer_input_gate = st.getTensor(ple_gate_name);

            const ple_proj_name = std.fmt.bufPrint(&name_buf, "blk.{d}.per_layer_projection", .{i}) catch continue;
            layer.per_layer_projection = st.getTensor(ple_proj_name);

            // Layer Scalar
            const scalar_name = std.fmt.bufPrint(&name_buf, "blk.{d}.layer_scalar", .{i}) catch continue;
            if (st.getTensor(scalar_name)) |t_scalar| {
                if (t_scalar.type == .BF16 and t_scalar.data.len >= 2) {
                    const u = std.mem.readInt(u16, t_scalar.data[0..2], .little);
                    layer.layer_scalar = @as(f32, @bitCast(@as(u32, u) << 16));
                } else if (t_scalar.type == .F32 and t_scalar.data.len >= 4) {
                    const u = std.mem.readInt(u32, t_scalar.data[0..4], .little);
                    layer.layer_scalar = @bitCast(u);
                }
            }

            // Q, K, V
            const q_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_q.weight", .{i}) catch continue;
            layer.attn_q = st.getTensor(q_name);

            const k_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_k.weight", .{i}) catch continue;
            layer.attn_k = st.getTensor(k_name);

            const v_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_v.weight", .{i}) catch continue;
            layer.attn_v = st.getTensor(v_name);

            const out_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_output.weight", .{i}) catch continue;
            layer.attn_output = st.getTensor(out_name);

            // Q / K Norms & Head Dim
            const q_norm_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_q_norm.weight", .{i}) catch continue;
            const q_norm_t = st.getTensor(q_norm_name);
            if (q_norm_t) |qnt| {
                layer.head_dim = qnt.elements();
                layer.attn_q_norm = try loadNorm(allocator, q_norm_t, layer.head_dim);
            } else {
                layer.head_dim = head_size;
            }

            const k_norm_name = std.fmt.bufPrint(&name_buf, "blk.{d}.attn_k_norm.weight", .{i}) catch continue;
            layer.attn_k_norm = try loadNorm(allocator, st.getTensor(k_norm_name), layer.head_dim);

            if (layer.attn_q) |t_q| {
                layer.n_heads = t_q.elements() / dim / layer.head_dim;
            } else {
                layer.n_heads = params.head_count;
            }

            if (layer.attn_k) |t_k| {
                layer.n_kv_heads = t_k.elements() / dim / layer.head_dim;
            } else {
                layer.n_kv_heads = if (params.head_count_kv > 0) params.head_count_kv else layer.n_heads;
            }

            if (layer.head_dim >= 512) {
                layer.rope_theta = 1000000.0;
                layer.rotary_dim = 128; // 0.25 * 512
                layer.sliding_window = 0;
            } else {
                layer.rope_theta = 10000.0;
                layer.rotary_dim = layer.head_dim;
                layer.sliding_window = 512;
            }

            // FFN projections
            const gate_name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_gate.weight", .{i}) catch continue;
            layer.ffn_gate = st.getTensor(gate_name);

            const up_name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_up.weight", .{i}) catch continue;
            layer.ffn_up = st.getTensor(up_name);

            const down_name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_down.weight", .{i}) catch continue;
            layer.ffn_down = st.getTensor(down_name);

            if (layer.ffn_gate) |t_gate| {
                layer.intermediate_size = t_gate.elements() / dim;
            } else {
                layer.intermediate_size = params.feed_forward_length;
            }

            self.layers[i] = layer;
        }

        const patch_proj = st.getTensor("model.vision_tower.patch_embedder.input_proj.weight") orelse st.getTensor("vision_tower.patch_embedder.input_proj.weight");
        const pos_emb = st.getTensor("model.vision_tower.patch_embedder.position_embedding_table") orelse st.getTensor("vision_tower.patch_embedder.position_embedding_table");
        const emb_proj = st.getTensor("model.embed_vision.embedding_projection.weight") orelse st.getTensor("embed_vision.embedding_projection.weight");

        // Load 16 Vision Transformer layers from SafeTensors
        var vision_layers = std.ArrayList(vision.VisionLayerWeights).empty;
        for (0..16) |l_idx| {
            var v_name: [128]u8 = undefined;
            const q_proj_name = std.fmt.bufPrint(&v_name, "model.vision_tower.encoder.layers.{d}.self_attn.q_proj.linear.weight", .{l_idx}) catch break;
            const q_proj = st.getTensor(q_proj_name) orelse break;

            const in_ln_name = std.fmt.bufPrint(&v_name, "model.vision_tower.encoder.layers.{d}.input_layernorm.weight", .{l_idx}) catch break;
            const post_attn_name = std.fmt.bufPrint(&v_name, "model.vision_tower.encoder.layers.{d}.post_attention_layernorm.weight", .{l_idx}) catch break;
            const pre_ffn_name = std.fmt.bufPrint(&v_name, "model.vision_tower.encoder.layers.{d}.pre_feedforward_layernorm.weight", .{l_idx}) catch break;
            const post_ffn_name = std.fmt.bufPrint(&v_name, "model.vision_tower.encoder.layers.{d}.post_feedforward_layernorm.weight", .{l_idx}) catch break;

            const k_proj_name = std.fmt.bufPrint(&v_name, "model.vision_tower.encoder.layers.{d}.self_attn.k_proj.linear.weight", .{l_idx}) catch break;
            const v_proj_name = std.fmt.bufPrint(&v_name, "model.vision_tower.encoder.layers.{d}.self_attn.v_proj.linear.weight", .{l_idx}) catch break;
            const o_proj_name = std.fmt.bufPrint(&v_name, "model.vision_tower.encoder.layers.{d}.self_attn.o_proj.linear.weight", .{l_idx}) catch break;

            const q_norm_n = std.fmt.bufPrint(&v_name, "model.vision_tower.encoder.layers.{d}.self_attn.q_norm.weight", .{l_idx}) catch break;
            const k_norm_n = std.fmt.bufPrint(&v_name, "model.vision_tower.encoder.layers.{d}.self_attn.k_norm.weight", .{l_idx}) catch break;

            const gate_proj_n = std.fmt.bufPrint(&v_name, "model.vision_tower.encoder.layers.{d}.mlp.gate_proj.linear.weight", .{l_idx}) catch break;
            const up_proj_n = std.fmt.bufPrint(&v_name, "model.vision_tower.encoder.layers.{d}.mlp.up_proj.linear.weight", .{l_idx}) catch break;
            const down_proj_n = std.fmt.bufPrint(&v_name, "model.vision_tower.encoder.layers.{d}.mlp.down_proj.linear.weight", .{l_idx}) catch break;

            try vision_layers.append(allocator, .{
                .input_layernorm = try loadNorm(allocator, st.getTensor(in_ln_name), 768),
                .post_attention_layernorm = try loadNorm(allocator, st.getTensor(post_attn_name), 768),
                .pre_feedforward_layernorm = try loadNorm(allocator, st.getTensor(pre_ffn_name), 768),
                .post_feedforward_layernorm = try loadNorm(allocator, st.getTensor(post_ffn_name), 768),
                .q_proj = q_proj,
                .k_proj = st.getTensor(k_proj_name),
                .v_proj = st.getTensor(v_proj_name),
                .o_proj = st.getTensor(o_proj_name),
                .q_norm = try loadNorm(allocator, st.getTensor(q_norm_n), 64),
                .k_norm = try loadNorm(allocator, st.getTensor(k_norm_n), 64),
                .gate_proj = st.getTensor(gate_proj_n),
                .up_proj = st.getTensor(up_proj_n),
                .down_proj = st.getTensor(down_proj_n),
            });
        }
        const v_layers_slice = try vision_layers.toOwnedSlice(allocator);
        if (v_layers_slice.len > 0) {
            std.debug.print("  Loaded {d} Vision Transformer layers from SafeTensors\n", .{v_layers_slice.len});
        }

        self.vision_encoder = if (patch_proj != null or emb_proj != null or v_layers_slice.len > 0)
            VisionEncoder.init(allocator, patch_proj, pos_emb, emb_proj, v_layers_slice)
        else
            null;

        return self;
    }

    pub fn deinit(self: *TransformerModel) void {
        if (self.vision_encoder) |*ve| ve.deinit();
        if (self.audio_encoder) |*ae| ae.deinit();
        for (self.layers) |layer| {
            if (layer.input_layernorm) |n| self.allocator.free(n);
            if (layer.post_attention_layernorm) |n| self.allocator.free(n);
            if (layer.pre_feedforward_layernorm) |n| self.allocator.free(n);
            if (layer.post_feedforward_layernorm) |n| self.allocator.free(n);
            if (layer.post_per_layer_input_norm) |n| self.allocator.free(n);
            if (layer.attn_q_norm) |n| self.allocator.free(n);
            if (layer.attn_k_norm) |n| self.allocator.free(n);
        }
        self.allocator.free(self.output_norm);
        if (self.per_layer_projection_norm) |n| self.allocator.free(n);
        self.allocator.free(self.layers);
        self.allocator.destroy(self);
    }

    pub fn forward(
        self: *const TransformerModel,
        token_id: u32,
        pos: usize,
        kv_cache: *KVCache,
        bufs: *ModelBuffers,
        pool: ?*ThreadPool,
        compute_logits: bool,
    ) ![]const f32 {
        return self.forwardWithEmbedding(null, token_id, pos, kv_cache, bufs, pool, compute_logits);
    }

    pub fn forwardWithEmbedding(
        self: *const TransformerModel,
        custom_embedding: ?[]const f32,
        token_id: u32,
        pos: usize,
        kv_cache: *KVCache,
        bufs: *ModelBuffers,
        pool: ?*ThreadPool,
        compute_logits: bool,
    ) ![]const f32 {
        const p = &self.params;
        const dim = p.embedding_length;

        if (custom_embedding) |emb| {
            @memcpy(bufs.x[0..@min(dim, emb.len)], emb[0..@min(dim, emb.len)]);
            if (emb.len < dim) {
                @memset(bufs.x[emb.len..dim], 0.0);
            }
        } else {
            // 1. Embedding lookup
            if (token_id >= p.vocab_size) return error.TokenOutOfBounds;
            const embd_row = self.token_embd.getRow(token_id);
            quant.dequantizeRow(self.token_embd.type, embd_row, bufs.x, dim);

            // Gemma embedding scaling
            if (p.arch == .gemma or p.arch == .gemma2 or p.arch == .gemma4) {
                const scale = @sqrt(@as(f32, @floatFromInt(dim)));
                for (bufs.x) |*v| v.* *= scale;
            }
        }

        // Per-layer embedding precomputation (Token identity + Context-aware projection)
        if (self.embed_tokens_per_layer) |ple_tab| {
            const ple_dim: usize = 256;
            const total_ple_dim = self.layers.len * ple_dim;
            if (token_id < p.vocab_size) {
                const ple_row = ple_tab.getRow(token_id);
                quant.dequantizeRow(ple_tab.type, ple_row, bufs.ctx_ple_buf[0..total_ple_dim], total_ple_dim);
            } else {
                @memset(bufs.ctx_ple_buf[0..total_ple_dim], 0.0);
            }
            const token_scale = @sqrt(@as(f32, @floatFromInt(ple_dim)));
            for (bufs.ctx_ple_buf[0..total_ple_dim]) |*v| v.* *= token_scale;

            if (self.per_layer_model_projection) |ctx_proj_t| {
                const inv_sqrt_dim = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
                for (0..dim) |d| bufs.xb[d] = bufs.x[d] * inv_sqrt_dim;
                math.gemv(pool, ctx_proj_t.type, ctx_proj_t.data, bufs.xb, bufs.ctx_scratch[0..total_ple_dim], total_ple_dim, dim);
                if (self.per_layer_projection_norm) |norm_slice| {
                    const inv_sqrt_2: f32 = 1.0 / @sqrt(2.0);
                    for (0..self.layers.len) |l_idx| {
                        const slice = bufs.ctx_scratch[l_idx * ple_dim .. (l_idx + 1) * ple_dim];
                        math.rmsNorm(slice, norm_slice, slice, p.layer_norm_rms_epsilon, false);
                        for (0..ple_dim) |d| {
                            bufs.ctx_ple_buf[l_idx * ple_dim + d] = (bufs.ctx_ple_buf[l_idx * ple_dim + d] + slice[d]) * inv_sqrt_2;
                        }
                    }
                }
            }
        }

        // 2. Transformer Layers Forward Pass
        for (self.layers, 0..) |layer, layer_idx| {
            // A. Pre-Attention Norm
            if (layer.input_layernorm) |norm_slice| {
                math.rmsNorm(bufs.x, norm_slice, bufs.xb, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            } else {
                @memcpy(bufs.xb, bufs.x);
            }

            const head_size = if (layer.head_dim > 0) layer.head_dim else p.head_size;
            const n_heads = if (layer.n_heads > 0) layer.n_heads else p.head_count;
            const n_kv_heads = if (layer.n_kv_heads > 0) layer.n_kv_heads else p.head_count_kv;
            const gqa_group = if (n_kv_heads > 0) n_heads / n_kv_heads else 1;

            // Q Projection
            if (layer.attn_q) |t_q| {
                math.gemv(pool, t_q.type, t_q.data, bufs.xb, bufs.q, n_heads * head_size, dim);
            }

            // Optional Q RMSNorm
            if (layer.attn_q_norm) |q_norm_slice| {
                for (0..n_heads) |h| {
                    const q_head = bufs.q[h * head_size .. (h + 1) * head_size];
                    math.rmsNorm(q_head, q_norm_slice, q_head, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
                }
            }

            // RoPE for Query and Key with precomputed table (avoid 50,000+ transcendental pow/cos/sin calls)
            const pos_f32 = @as(f32, @floatFromInt(pos)) * p.rope_freq_scale;
            const eff_rotary = if (layer.rotary_dim == 0 or layer.rotary_dim > head_size) head_size else layer.rotary_dim;
            const half_dim = eff_rotary / 2;
            const log_theta = @log(layer.rope_theta);
            const inv_rotary = 1.0 / @as(f32, @floatFromInt(eff_rotary));

            var cos_tab: [256]f32 = undefined;
            var sin_tab: [256]f32 = undefined;
            const safe_half_dim = @min(half_dim, 256);
            for (0..safe_half_dim) |i| {
                const exponent = @as(f32, @floatFromInt(2 * i)) * inv_rotary;
                const freq = @exp(-exponent * log_theta);
                const theta = pos_f32 * freq;
                cos_tab[i] = @cos(theta);
                sin_tab[i] = @sin(theta);
            }

            for (0..n_heads) |h| {
                const q_head = bufs.q[h * head_size .. (h + 1) * head_size];
                for (0..safe_half_dim) |i| {
                    const cos_t = cos_tab[i];
                    const sin_t = sin_tab[i];
                    const v0 = q_head[i];
                    const v1 = q_head[i + half_dim];
                    q_head[i] = v0 * cos_t - v1 * sin_t;
                    q_head[i + half_dim] = v0 * sin_t + v1 * cos_t;
                }
            }

            // KV Cache Handling (with Cross-Layer Sharing for Gemma 4)
            const is_kv_shared = (p.arch == .gemma4 and layer_idx >= 15);
            const donor_layer: usize = if (is_kv_shared) (if (layer.head_dim >= 512) 14 else 13) else layer_idx;

            if (!is_kv_shared) {
                if (layer.attn_k) |t_k| {
                    math.gemv(pool, t_k.type, t_k.data, bufs.xb, bufs.k, n_kv_heads * head_size, dim);
                }
                if (layer.attn_v) |t_v| {
                    math.gemv(pool, t_v.type, t_v.data, bufs.xb, bufs.v, n_kv_heads * head_size, dim);
                }

                if (layer.attn_k_norm) |k_norm_slice| {
                    for (0..n_kv_heads) |h| {
                        const k_head = bufs.k[h * head_size .. (h + 1) * head_size];
                        math.rmsNorm(k_head, k_norm_slice, k_head, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
                    }
                }

                if (p.arch == .gemma4) {
                    for (0..n_kv_heads) |h| {
                        const v_head = bufs.v[h * head_size .. (h + 1) * head_size];
                        math.rmsNormNoScale(v_head, v_head, p.layer_norm_rms_epsilon);
                    }
                }

                // RoPE for Key (reusing precomputed cos/sin table)
                for (0..n_kv_heads) |h| {
                    const k_head = bufs.k[h * head_size .. (h + 1) * head_size];
                    for (0..safe_half_dim) |i| {
                        const cos_t = cos_tab[i];
                        const sin_t = sin_tab[i];
                        const v0 = k_head[i];
                        const v1 = k_head[i + half_dim];
                        k_head[i] = v0 * cos_t - v1 * sin_t;
                        k_head[i + half_dim] = v0 * sin_t + v1 * cos_t;
                    }
                }

                kv_cache.put(layer_idx, pos, bufs.k[0 .. n_kv_heads * head_size], bufs.v[0 .. n_kv_heads * head_size]);
            }

            // Scaled Dot-Product Attention with GQA & sliding window
            const attn_scale: f32 = if (p.arch == .gemma4) 1.0 else 1.0 / @sqrt(@as(f32, @floatFromInt(head_size)));
            const seq_len = pos + 1;
            const start_t = if (layer.sliding_window > 0 and seq_len > layer.sliding_window) seq_len - layer.sliding_window else 0;

            for (0..n_heads) |h| {
                const kv_h = h / gqa_group;
                const q_head = bufs.q[h * head_size .. (h + 1) * head_size];
                const head_scores = bufs.attn_scores[start_t..seq_len];

                for (start_t..seq_len) |t| {
                    const k_vec = kv_cache.getKey(donor_layer, t, kv_h);
                    var score = math.dotF32F32(q_head, k_vec[0..head_size]) * attn_scale;

                    if (p.attn_logit_softcapping > 0.0) {
                        const cap = p.attn_logit_softcapping;
                        score = cap * std.math.tanh(score / cap);
                    }

                    head_scores[t - start_t] = score;
                }

                math.softmax(head_scores);

                // Accumulate weighted values into attn_out using SIMD
                const out_head = bufs.attn_out[h * head_size .. (h + 1) * head_size];
                @memset(out_head, 0.0);

                const Vec = @Vector(8, f32);
                const n_vec = head_size / 8;

                for (start_t..seq_len) |t| {
                    const weight = head_scores[t - start_t];
                    const v_weight: Vec = @splat(weight);
                    const v_vec = kv_cache.getValue(donor_layer, t, kv_h);

                    for (0..n_vec) |chunk| {
                        const idx = chunk * 8;
                        const v_val: Vec = v_vec[idx..][0..8].*;
                        const cur: Vec = out_head[idx..][0..8].*;
                        out_head[idx..][0..8].* = cur + v_weight * v_val;
                    }
                }
            }

            // Attention Output Projection
            if (layer.attn_output) |t_out| {
                math.gemv(pool, t_out.type, t_out.data, bufs.attn_out[0 .. n_heads * head_size], bufs.xb, dim, n_heads * head_size);
            }

            // Post-attention norm
            if (layer.post_attention_layernorm) |norm_slice| {
                math.rmsNorm(bufs.xb, norm_slice, bufs.xb, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            }

            // Residual Add
            for (0..dim) |d| {
                bufs.x[d] += bufs.xb[d];
            }

            // B. Pre-FFN Norm
            if (layer.pre_feedforward_layernorm) |norm_slice| {
                math.rmsNorm(bufs.x, norm_slice, bufs.xb, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            } else {
                @memcpy(bufs.xb, bufs.x);
            }

            const ffn_len = if (layer.intermediate_size > 0) layer.intermediate_size else p.feed_forward_length;

            // FFN Projections
            if (layer.ffn_gate) |t_gate| {
                math.gemv(pool, t_gate.type, t_gate.data, bufs.xb, bufs.gate, ffn_len, dim);
            }
            if (layer.ffn_up) |t_up| {
                math.gemv(pool, t_up.type, t_up.data, bufs.xb, bufs.up, ffn_len, dim);
            }

            // Activation: SwiGLU or GeGLU
            if (p.arch == .gemma or p.arch == .gemma2 or p.arch == .gemma4) {
                math.geglu(bufs.gate[0..ffn_len], bufs.up[0..ffn_len], bufs.gate[0..ffn_len]);
            } else {
                math.swiglu(bufs.gate[0..ffn_len], bufs.up[0..ffn_len], bufs.gate[0..ffn_len]);
            }

            // Down Projection
            if (layer.ffn_down) |t_down| {
                math.gemv(pool, t_down.type, t_down.data, bufs.gate[0..ffn_len], bufs.ffn_out, dim, ffn_len);
            }

            // Post-FFN norm
            if (layer.post_feedforward_layernorm) |norm_slice| {
                math.rmsNorm(bufs.ffn_out, norm_slice, bufs.ffn_out, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);
            }

            // Residual Add
            for (0..dim) |d| {
                bufs.x[d] += bufs.ffn_out[d];
            }

            // C. Per-Layer Embedding Injection (PLE)
            if (layer.per_layer_input_gate != null and layer.per_layer_projection != null and bufs.ctx_ple_buf.len >= (layer_idx + 1) * 256) {
                const ple_dim: usize = 256;
                const ple_slice = bufs.ctx_ple_buf[layer_idx * ple_dim .. (layer_idx + 1) * ple_dim];

                // gate = gelu(W_gate * x)
                const t_gate = layer.per_layer_input_gate.?;
                math.gemv(pool, t_gate.type, t_gate.data, bufs.x, bufs.ple_gate[0..ple_dim], ple_dim, dim);
                for (bufs.ple_gate[0..ple_dim]) |*g| g.* = math.gelu(g.*);

                // ple = ple_slice * gate
                for (0..ple_dim) |d| bufs.ple_buf[d] = ple_slice[d] * bufs.ple_gate[d];

                // delta_x = W_proj * ple
                const t_proj = layer.per_layer_projection.?;
                math.gemv(pool, t_proj.type, t_proj.data, bufs.ple_buf[0..ple_dim], bufs.xb, dim, ple_dim);

                // norm(delta_x)
                if (layer.post_per_layer_input_norm) |norm_slice| {
                    math.rmsNorm(bufs.xb, norm_slice, bufs.xb, p.layer_norm_rms_epsilon, false);
                }

                // x = x + delta_x
                for (0..dim) |d| bufs.x[d] += bufs.xb[d];
            }

            // D. End-of-layer scaling
            const scalar = layer.layer_scalar;
            if (p.arch == .gemma4) {
                for (bufs.x) |*v| v.* *= scalar;
            }
        }

        if (!compute_logits) return bufs.logits;

        // 3. Final Norm
        math.rmsNorm(bufs.x, self.output_norm, bufs.xb, p.layer_norm_rms_epsilon, p.use_gemma_rms_unit_offset);

        // 4. LM Head Output Projection
        if (self.output) |head| {
            math.gemv(pool, head.type, head.data, bufs.xb, bufs.logits, p.vocab_size, dim);
        } else {
            // Weight tying: token_embd is reused as LM head
            math.gemv(pool, self.token_embd.type, self.token_embd.data, bufs.xb, bufs.logits, p.vocab_size, dim);
        }

        // Final logit softcapping (Gemma 2 / Gemma 4)
        if (p.final_logit_softcapping > 0.0) {
            const cap = p.final_logit_softcapping;
            for (bufs.logits) |*logit| {
                logit.* = cap * std.math.tanh(logit.* / cap);
            }
        }

        return bufs.logits;
    }
};
