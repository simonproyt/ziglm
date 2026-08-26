const std = @import("std");
const types = @import("types.zig");
const Tensor = types.Tensor;
const math = @import("math.zig");
const quant = @import("quant.zig");
const Image = @import("image.zig").Image;
const ThreadPool = @import("thread_pool.zig").ThreadPool;

pub const VisionLayerWeights = struct {
    input_layernorm: ?[]const f32 = null,
    post_attention_layernorm: ?[]const f32 = null,
    pre_feedforward_layernorm: ?[]const f32 = null,
    post_feedforward_layernorm: ?[]const f32 = null,
    q_proj: ?Tensor = null,
    k_proj: ?Tensor = null,
    v_proj: ?Tensor = null,
    o_proj: ?Tensor = null,
    q_norm: ?[]const f32 = null,
    k_norm: ?[]const f32 = null,
    gate_proj: ?Tensor = null,
    up_proj: ?Tensor = null,
    down_proj: ?Tensor = null,
};

pub const VisionEncoder = struct {
    allocator: std.mem.Allocator,
    patch_proj: ?Tensor = null,
    position_embeddings: ?Tensor = null,
    embedding_projection: ?Tensor = null,
    layers: []VisionLayerWeights,
    hidden_size: usize = 768,
    intermediate_size: usize = 3072,
    num_heads: usize = 12,
    head_dim: usize = 64,
    llm_dim: usize = 1536,
    patch_size: usize = 16,

    pub fn init(
        allocator: std.mem.Allocator,
        patch_proj: ?Tensor,
        pos_emb: ?Tensor,
        embed_proj: ?Tensor,
        layers: []VisionLayerWeights,
    ) VisionEncoder {
        return VisionEncoder{
            .allocator = allocator,
            .patch_proj = patch_proj,
            .position_embeddings = pos_emb,
            .embedding_projection = embed_proj,
            .layers = layers,
            .hidden_size = 768,
            .intermediate_size = 3072,
            .num_heads = 12,
            .head_dim = 64,
            .llm_dim = if (embed_proj) |ep| (if (ep.n_dims >= 2) ep.shape[1] else ep.shape[0]) else 1536,
            .patch_size = 16,
        };
    }

    pub fn deinit(self: *VisionEncoder) void {
        for (self.layers) |l| {
            if (l.input_layernorm) |n| self.allocator.free(n);
            if (l.post_attention_layernorm) |n| self.allocator.free(n);
            if (l.pre_feedforward_layernorm) |n| self.allocator.free(n);
            if (l.post_feedforward_layernorm) |n| self.allocator.free(n);
            if (l.q_norm) |n| self.allocator.free(n);
            if (l.k_norm) |n| self.allocator.free(n);
        }
        if (self.layers.len > 0) {
            self.allocator.free(self.layers);
        }
    }

    /// Encode an image through the 16-layer Vision Transformer into LLM embeddings [num_patches, llm_dim]
    pub fn encodeImage(
        self: *const VisionEncoder,
        allocator: std.mem.Allocator,
        image: *const Image,
        pool: ?*ThreadPool,
    ) ![]f32 {
        const raw_patches = try image.extractPatches(allocator, self.patch_size);
        defer allocator.free(raw_patches);

        const patch_dim = self.patch_size * self.patch_size * 3; // 768
        const num_patches = raw_patches.len / patch_dim;
        const patches_x = image.width / self.patch_size;
        const patches_y = image.height / self.patch_size;
        _ = patches_y;

        // Sequence states for all patches: [num_patches, hidden_size]
        const states = try allocator.alloc(f32, num_patches * self.hidden_size);
        defer allocator.free(states);

        // 1. Initial Patch Projection & 2D Positional Embeddings
        var y_pos_buf: [768]f32 = undefined;
        var x_pos_buf: [768]f32 = undefined;

        for (0..num_patches) |p| {
            const p_in = raw_patches[p * patch_dim .. (p + 1) * patch_dim];
            const p_state = states[p * self.hidden_size .. (p + 1) * self.hidden_size];

            // Linear patch embedding
            if (self.patch_proj) |proj| {
                math.gemv(pool, proj.type, proj.data, p_in, p_state, self.hidden_size, patch_dim);
            } else {
                @memcpy(p_state[0..@min(patch_dim, self.hidden_size)], p_in[0..@min(patch_dim, self.hidden_size)]);
            }

            // 2D Spatial Positional Embeddings: [2, 10240, 768] -> table[0, py, :] + table[1, px, :]
            const py = if (patches_x > 0) p / patches_x else 0;
            const px = if (patches_x > 0) p % patches_x else 0;

            if (self.position_embeddings) |pos_t| {
                const pos_stride: usize = 10240;
                const row_bytes = self.hidden_size * pos_t.type.typeSize() / pos_t.type.blockSize();

                // Y embedding from slice 0
                if (py < pos_stride and (py + 1) * row_bytes <= pos_t.data.len) {
                    const y_row = pos_t.data[py * row_bytes .. (py + 1) * row_bytes];
                    quant.dequantizeRow(pos_t.type, y_row, &y_pos_buf, self.hidden_size);
                    for (0..self.hidden_size) |d| p_state[d] += y_pos_buf[d];
                }

                // X embedding from slice 1 (offset by pos_stride)
                const x_idx = pos_stride + px;
                if (px < pos_stride and (x_idx + 1) * row_bytes <= pos_t.data.len) {
                    const x_row = pos_t.data[x_idx * row_bytes .. (x_idx + 1) * row_bytes];
                    quant.dequantizeRow(pos_t.type, x_row, &x_pos_buf, self.hidden_size);
                    for (0..self.hidden_size) |d| p_state[d] += x_pos_buf[d];
                }
            }
        }

        // 2. Vision Transformer Encoder Layers (16 Layers)
        if (self.layers.len > 0) {
            // Temporary working buffers for all patches
            const norm_buf = try allocator.alloc(f32, num_patches * self.hidden_size);
            defer allocator.free(norm_buf);

            const q_buf = try allocator.alloc(f32, num_patches * self.hidden_size);
            defer allocator.free(q_buf);

            const k_buf = try allocator.alloc(f32, num_patches * self.hidden_size);
            defer allocator.free(k_buf);

            const v_buf = try allocator.alloc(f32, num_patches * self.hidden_size);
            defer allocator.free(v_buf);

            const attn_ctx = try allocator.alloc(f32, num_patches * self.hidden_size);
            defer allocator.free(attn_ctx);

            const attn_out = try allocator.alloc(f32, num_patches * self.hidden_size);
            defer allocator.free(attn_out);

            const gate_buf = try allocator.alloc(f32, self.intermediate_size);
            defer allocator.free(gate_buf);

            const up_buf = try allocator.alloc(f32, self.intermediate_size);
            defer allocator.free(up_buf);

            const act_buf = try allocator.alloc(f32, self.intermediate_size);
            defer allocator.free(act_buf);

            const down_buf = try allocator.alloc(f32, self.hidden_size);
            defer allocator.free(down_buf);

            const scores = try allocator.alloc(f32, num_patches);
            defer allocator.free(scores);

            const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));

            for (self.layers) |layer| {
                // A. Input Layernorm
                for (0..num_patches) |p| {
                    const p_in = states[p * self.hidden_size .. (p + 1) * self.hidden_size];
                    const p_norm = norm_buf[p * self.hidden_size .. (p + 1) * self.hidden_size];
                    if (layer.input_layernorm) |ln| {
                        math.rmsNorm(p_in, ln, p_norm, 1e-6, false);
                    } else {
                        @memcpy(p_norm, p_in);
                    }
                }

                // B. Q, K, V Projections
                for (0..num_patches) |p| {
                    const p_norm = norm_buf[p * self.hidden_size .. (p + 1) * self.hidden_size];
                    const p_q = q_buf[p * self.hidden_size .. (p + 1) * self.hidden_size];
                    const p_k = k_buf[p * self.hidden_size .. (p + 1) * self.hidden_size];
                    const p_v = v_buf[p * self.hidden_size .. (p + 1) * self.hidden_size];

                    if (layer.q_proj) |t| math.gemv(pool, t.type, t.data, p_norm, p_q, self.hidden_size, self.hidden_size);
                    if (layer.k_proj) |t| math.gemv(pool, t.type, t.data, p_norm, p_k, self.hidden_size, self.hidden_size);
                    if (layer.v_proj) |t| math.gemv(pool, t.type, t.data, p_norm, p_v, self.hidden_size, self.hidden_size);

                    // Head norms for Q and K
                    for (0..self.num_heads) |h| {
                        const q_h = p_q[h * self.head_dim .. (h + 1) * self.head_dim];
                        const k_h = p_k[h * self.head_dim .. (h + 1) * self.head_dim];
                        if (layer.q_norm) |qn| math.rmsNorm(q_h, qn, q_h, 1e-6, false);
                        if (layer.k_norm) |kn| math.rmsNorm(k_h, kn, k_h, 1e-6, false);
                    }
                }

                // C. Bidirectional Multi-Head Self-Attention across all patches
                @memset(attn_ctx, 0.0);
                for (0..self.num_heads) |h| {
                    for (0..num_patches) |p1| {
                        const q1 = q_buf[p1 * self.hidden_size + h * self.head_dim .. p1 * self.hidden_size + (h + 1) * self.head_dim];

                        for (0..num_patches) |p2| {
                            const k2 = k_buf[p2 * self.hidden_size + h * self.head_dim .. p2 * self.hidden_size + (h + 1) * self.head_dim];
                            scores[p2] = math.dotF32F32(q1, k2) * scale;
                        }

                        math.softmax(scores);

                        const out_h = attn_ctx[p1 * self.hidden_size + h * self.head_dim .. p1 * self.hidden_size + (h + 1) * self.head_dim];
                        for (0..num_patches) |p2| {
                            const weight = scores[p2];
                            const v2 = v_buf[p2 * self.hidden_size + h * self.head_dim .. p2 * self.hidden_size + (h + 1) * self.head_dim];
                            for (0..self.head_dim) |d| {
                                out_h[d] += weight * v2[d];
                            }
                        }
                    }
                }

                // D. Attention Output Projection & Residual
                for (0..num_patches) |p| {
                    const p_ctx = attn_ctx[p * self.hidden_size .. (p + 1) * self.hidden_size];
                    const p_out = attn_out[p * self.hidden_size .. (p + 1) * self.hidden_size];
                    const p_state = states[p * self.hidden_size .. (p + 1) * self.hidden_size];

                    if (layer.o_proj) |t| {
                        math.gemv(pool, t.type, t.data, p_ctx, p_out, self.hidden_size, self.hidden_size);
                    } else {
                        @memcpy(p_out, p_ctx);
                    }

                    if (layer.post_attention_layernorm) |pn| {
                        math.rmsNorm(p_out, pn, p_out, 1e-6, false);
                    }

                    for (0..self.hidden_size) |d| {
                        p_state[d] += p_out[d];
                    }
                }

                // E. FeedForward (GEGLU MLP) & Residual
                for (0..num_patches) |p| {
                    const p_state = states[p * self.hidden_size .. (p + 1) * self.hidden_size];
                    const p_norm = norm_buf[p * self.hidden_size .. (p + 1) * self.hidden_size];

                    if (layer.pre_feedforward_layernorm) |fn_norm| {
                        math.rmsNorm(p_state, fn_norm, p_norm, 1e-6, false);
                    } else {
                        @memcpy(p_norm, p_state);
                    }

                    if (layer.gate_proj) |t_gate| {
                        math.gemv(pool, t_gate.type, t_gate.data, p_norm, gate_buf, self.intermediate_size, self.hidden_size);
                    }
                    if (layer.up_proj) |t_up| {
                        math.gemv(pool, t_up.type, t_up.data, p_norm, up_buf, self.intermediate_size, self.hidden_size);
                    }

                    math.geglu(gate_buf, up_buf, act_buf);

                    if (layer.down_proj) |t_down| {
                        math.gemv(pool, t_down.type, t_down.data, act_buf, down_buf, self.hidden_size, self.intermediate_size);
                    } else {
                        @memcpy(down_buf, act_buf[0..self.hidden_size]);
                    }

                    if (layer.post_feedforward_layernorm) |pfn| {
                        math.rmsNorm(down_buf, pfn, down_buf, 1e-6, false);
                    }

                    for (0..self.hidden_size) |d| {
                        p_state[d] += down_buf[d];
                    }
                }
            }
        }

        // 3. Final Projection to LLM Embedding Space: [num_patches, 768] -> [num_patches, llm_dim]
        const out_embeddings = try allocator.alloc(f32, num_patches * self.llm_dim);
        defer allocator.free(out_embeddings);

        for (0..num_patches) |p| {
            const p_state = states[p * self.hidden_size .. (p + 1) * self.hidden_size];
            const p_out = out_embeddings[p * self.llm_dim .. (p + 1) * self.llm_dim];

            if (self.embedding_projection) |emb_proj| {
                math.gemv(pool, emb_proj.type, emb_proj.data, p_state, p_out, self.llm_dim, self.hidden_size);
            } else {
                @memset(p_out, 0.0);
                @memcpy(p_out[0..@min(self.hidden_size, self.llm_dim)], p_state[0..@min(self.hidden_size, self.llm_dim)]);
            }
        }

        // 4. Resample / Pool to exactly 280 soft tokens (Gemma 4 vision token count)
        const target_tokens: usize = 280;
        const final_embeddings = try allocator.alloc(f32, target_tokens * self.llm_dim);
        errdefer allocator.free(final_embeddings);

        if (num_patches == target_tokens) {
            @memcpy(final_embeddings, out_embeddings);
        } else if (num_patches > 0) {
            for (0..target_tokens) |t| {
                const src_idx = (t * num_patches) / target_tokens;
                const p_src = out_embeddings[src_idx * self.llm_dim .. (src_idx + 1) * self.llm_dim];
                const p_dst = final_embeddings[t * self.llm_dim .. (t + 1) * self.llm_dim];
                @memcpy(p_dst, p_src);
            }
        } else {
            @memset(final_embeddings, 0.0);
        }

        return final_embeddings;
    }
};

test "VisionEncoder test" {
    const allocator = std.testing.allocator;
    const encoder = VisionEncoder.init(allocator, null, null, null, &[_]VisionLayerWeights{});
    var img = try Image.createSynthetic(allocator, 32, 32);
    defer img.deinit();

    const embeddings = try encoder.encodeImage(allocator, &img, null);
    defer allocator.free(embeddings);

    try std.testing.expectEqual(280 * 1536, embeddings.len);
}
