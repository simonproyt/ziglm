const std = @import("std");
const types = @import("types.zig");
const Tensor = types.Tensor;
const math = @import("math.zig");
const quant = @import("quant.zig");
const Image = @import("image.zig").Image;
const ThreadPool = @import("thread_pool.zig").ThreadPool;

pub const VisionEncoder = struct {
    allocator: std.mem.Allocator,
    patch_proj: ?Tensor = null,
    position_embeddings: ?Tensor = null,
    embedding_projection: ?Tensor = null,
    hidden_size: usize = 768,
    llm_dim: usize = 1536,
    patch_size: usize = 16,

    pub fn init(allocator: std.mem.Allocator, patch_proj: ?Tensor, pos_emb: ?Tensor, embed_proj: ?Tensor) VisionEncoder {
        return VisionEncoder{
            .allocator = allocator,
            .patch_proj = patch_proj,
            .position_embeddings = pos_emb,
            .embedding_projection = embed_proj,
            .hidden_size = 768,
            .llm_dim = if (embed_proj) |ep| ep.shape[0] else 1536,
            .patch_size = 16,
        };
    }

    pub fn deinit(_: *VisionEncoder) void {}

    /// Encode an image into a sequence of LLM embedding vectors [num_patches, llm_dim]
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

        // Output buffer: num_patches * llm_dim
        const out_embeddings = try allocator.alloc(f32, num_patches * self.llm_dim);
        errdefer allocator.free(out_embeddings);

        var patch_feat_buf = try allocator.alloc(f32, self.hidden_size);
        defer allocator.free(patch_feat_buf);

        for (0..num_patches) |p| {
            const p_in = raw_patches[p * patch_dim .. (p + 1) * patch_dim];
            const p_out = out_embeddings[p * self.llm_dim .. (p + 1) * self.llm_dim];

            // 1. Patch projection: [768, 768] * [768] -> [768]
            if (self.patch_proj) |proj| {
                math.gemv(pool, proj.type, proj.data, p_in, patch_feat_buf, self.hidden_size, patch_dim);
            } else {
                @memcpy(patch_feat_buf[0..@min(patch_dim, self.hidden_size)], p_in[0..@min(patch_dim, self.hidden_size)]);
            }

            // 2. Add position embedding if available
            if (self.position_embeddings) |pos_t| {
                if (pos_t.data.len >= (p + 1) * self.hidden_size * @sizeOf(f16)) {
                    var pos_buf: [768]f32 = undefined;
                    const pos_row = pos_t.getRow(@intCast(p));
                    quant.dequantizeRow(pos_t.type, pos_row, &pos_buf, self.hidden_size);
                    for (0..self.hidden_size) |d| {
                        patch_feat_buf[d] += pos_buf[d];
                    }
                }
            }

            // 3. Project to LLM hidden dim: [1536, 768] * [768] -> [1536]
            if (self.embedding_projection) |emb_proj| {
                math.gemv(pool, emb_proj.type, emb_proj.data, patch_feat_buf, p_out, self.llm_dim, self.hidden_size);
            } else {
                @memset(p_out, 0.0);
                @memcpy(p_out[0..@min(self.hidden_size, self.llm_dim)], patch_feat_buf[0..@min(self.hidden_size, self.llm_dim)]);
            }
        }

        return out_embeddings;
    }
};

test "VisionEncoder test" {
    const allocator = std.testing.allocator;
    const encoder = VisionEncoder.init(allocator, null, null, null);
    var img = try Image.createSynthetic(allocator, 32, 32);
    defer img.deinit();

    const embeddings = try encoder.encodeImage(allocator, &img, null);
    defer allocator.free(embeddings);

    try std.testing.expectEqual(4 * 1536, embeddings.len);
}
