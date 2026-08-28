const std = @import("std");
const types = @import("types.zig");
const GGMLType = types.GGMLType;
const RoPEType = types.RoPEType;
const math = @import("math.zig");
const ThreadPool = @import("thread_pool.zig").ThreadPool;
const KVCache = @import("kv_cache.zig").KVCache;

pub const DeviceType = enum {
    cpu,
    cuda,
    vulkan,
};

pub const Backend = struct {
    device_type: DeviceType = .cpu,
    thread_pool: ?*ThreadPool = null,
    allocator: std.mem.Allocator,

    pub fn initCpu(allocator: std.mem.Allocator, thread_pool: ?*ThreadPool) Backend {
        return .{
            .device_type = .cpu,
            .thread_pool = thread_pool,
            .allocator = allocator,
        };
    }

    pub inline fn gemv(
        self: *const Backend,
        qtype: GGMLType,
        weight_data: []const u8,
        x: []const f32,
        y: []f32,
        rows: usize,
        cols: usize,
    ) void {
        math.gemv(self.thread_pool, qtype, weight_data, x, y, rows, cols);
    }

    pub inline fn gemm(
        self: *const Backend,
        qtype: GGMLType,
        weight_data: []const u8,
        X: []const f32,
        Y: []f32,
        batch_size: usize,
        rows: usize,
        cols: usize,
    ) void {
        math.gemm(self.thread_pool, qtype, weight_data, X, Y, batch_size, rows, cols);
    }

    pub inline fn rmsNorm(
        self: *const Backend,
        x: []const f32,
        weight: []const f32,
        out: []f32,
        eps: f32,
        use_unit_offset: bool,
    ) void {
        _ = self;
        math.rmsNorm(x, weight, out, eps, use_unit_offset);
    }

    pub inline fn rmsNormNoScale(
        self: *const Backend,
        x: []const f32,
        out: []f32,
        eps: f32,
    ) void {
        _ = self;
        math.rmsNormNoScale(x, out, eps);
    }

    pub inline fn geglu(
        self: *const Backend,
        gate: []const f32,
        up: []const f32,
        out: []f32,
    ) void {
        _ = self;
        math.geglu(gate, up, out);
    }

    pub inline fn swiglu(
        self: *const Backend,
        gate: []const f32,
        up: []const f32,
        out: []f32,
    ) void {
        _ = self;
        math.swiglu(gate, up, out);
    }

    /// High-throughput SIMD Attention
    pub inline fn attention(
        self: *const Backend,
        q: []const f32,
        kv_cache: *const KVCache,
        layer_idx: usize,
        start_t: usize,
        seq_len: usize,
        head_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        attn_scale: f32,
        softcap: f32,
        scores_buf: []f32,
        out: []f32,
    ) void {
        _ = self;
        math.fastAttention(
            q,
            kv_cache,
            layer_idx,
            start_t,
            seq_len,
            head_size,
            n_heads,
            n_kv_heads,
            attn_scale,
            softcap,
            scores_buf,
            out,
        );
    }
};
