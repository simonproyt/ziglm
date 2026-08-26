const std = @import("std");

pub const KVCache = struct {
    allocator: std.mem.Allocator,
    block_count: usize,
    head_count_kv: usize,
    head_size: usize,
    max_seq_len: usize,
    current_pos: usize = 0,

    // Contiguous memory buffer: [block_count * 2 * max_seq_len * head_count_kv * head_size]
    buffer: []f32,
    layer_stride: usize, // 2 * max_seq_len * head_count_kv * head_size
    pos_stride: usize, // head_count_kv * head_size
    head_stride: usize, // head_size

    pub fn init(
        allocator: std.mem.Allocator,
        block_count: usize,
        head_count_kv: usize,
        head_size: usize,
        max_seq_len: usize,
    ) !*KVCache {
        const self = try allocator.create(KVCache);
        const head_stride = @max(512, head_size);
        const pos_stride = head_count_kv * head_stride;
        const kv_block_stride = max_seq_len * pos_stride;
        const layer_stride = 2 * kv_block_stride;
        const total_floats = block_count * layer_stride;

        const buffer = try allocator.alloc(f32, total_floats);
        @memset(buffer, 0.0);

        self.* = .{
            .allocator = allocator,
            .block_count = block_count,
            .head_count_kv = head_count_kv,
            .head_size = head_size,
            .max_seq_len = max_seq_len,
            .current_pos = 0,
            .buffer = buffer,
            .layer_stride = layer_stride,
            .pos_stride = pos_stride,
            .head_stride = head_stride,
        };

        return self;
    }

    pub fn deinit(self: *KVCache) void {
        self.allocator.free(self.buffer);
        self.allocator.destroy(self);
    }

    pub fn reset(self: *KVCache) void {
        self.current_pos = 0;
    }

    pub fn setPos(self: *KVCache, pos: usize) void {
        self.current_pos = @min(pos, self.max_seq_len);
    }

    pub inline fn put(
        self: *KVCache,
        layer_idx: usize,
        pos: usize,
        k_src: []const f32, // [head_count_kv * head_size]
        v_src: []const f32, // [head_count_kv * head_size]
    ) void {
        std.debug.assert(layer_idx < self.block_count);
        std.debug.assert(pos < self.max_seq_len);
        std.debug.assert(k_src.len <= self.pos_stride);
        std.debug.assert(v_src.len <= self.pos_stride);

        const layer_offset = layer_idx * self.layer_stride;
        const kv_block_stride = self.max_seq_len * self.pos_stride;

        // Keys
        const k_start = layer_offset + pos * self.pos_stride;
        @memcpy(self.buffer[k_start .. k_start + k_src.len], k_src);

        // Values
        const v_start = layer_offset + kv_block_stride + pos * self.pos_stride;
        @memcpy(self.buffer[v_start .. v_start + v_src.len], v_src);
    }

    pub inline fn getKey(self: *const KVCache, layer_idx: usize, pos: usize, kv_head_idx: usize) []const f32 {
        const layer_offset = layer_idx * self.layer_stride;
        const start = layer_offset + pos * self.pos_stride + kv_head_idx * self.head_stride;
        return self.buffer[start .. start + self.head_stride];
    }

    pub inline fn getValue(self: *const KVCache, layer_idx: usize, pos: usize, kv_head_idx: usize) []const f32 {
        const layer_offset = layer_idx * self.layer_stride;
        const kv_block_stride = self.max_seq_len * self.pos_stride;
        const start = layer_offset + kv_block_stride + pos * self.pos_stride + kv_head_idx * self.head_stride;
        return self.buffer[start .. start + self.head_stride];
    }
};

test "KVCache put and retrieve" {
    const allocator = std.testing.allocator;
    var cache = try KVCache.init(allocator, 2, 4, 8, 16);
    defer cache.deinit();

    var k = [_]f32{ 1.0 } ** 32;
    var v = [_]f32{ 2.0 } ** 32;
    k[0] = 42.0;
    v[0] = 99.0;

    cache.put(0, 5, &k, &v);

    const k_slice = cache.getKey(0, 5, 0);
    const v_slice = cache.getValue(0, 5, 0);

    try std.testing.expectEqual(@as(f32, 42.0), k_slice[0]);
    try std.testing.expectEqual(@as(f32, 99.0), v_slice[0]);
}
