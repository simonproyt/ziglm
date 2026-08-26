const std = @import("std");
const types = @import("types.zig");
const GGMLType = types.GGMLType;
const RoPEType = types.RoPEType;
const BlockQ4_0 = types.BlockQ4_0;
const BlockQ4_1 = types.BlockQ4_1;
const BlockQ5_0 = types.BlockQ5_0;
const BlockQ5_1 = types.BlockQ5_1;
const BlockQ8_0 = types.BlockQ8_0;
const BlockQ8_1 = types.BlockQ8_1;
const BlockQ2_K = types.BlockQ2_K;
const BlockQ3_K = types.BlockQ3_K;
const BlockQ4_K = types.BlockQ4_K;
const BlockQ5_K = types.BlockQ5_K;
const BlockQ6_K = types.BlockQ6_K;
const BlockQ8_K = types.BlockQ8_K;
const BlockIQ4_NL = types.BlockIQ4_NL;
const quant = @import("quant.zig");
const ThreadPool = @import("thread_pool.zig").ThreadPool;

// ============================================================================
// Vectorized SIMD Dot Products
// ============================================================================

pub fn dotF32F32(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    const n = a.len;
    const Vec = @Vector(8, f32);
    const vec_len = 8;
    const n_vec = n / vec_len;

    var sum_vec: Vec = @splat(0.0);
    var i: usize = 0;

    while (i < n_vec * vec_len) : (i += vec_len) {
        const va: Vec = a[i..][0..vec_len].*;
        const vb: Vec = b[i..][0..vec_len].*;
        sum_vec += va * vb;
    }

    var total = @reduce(.Add, sum_vec);

    // Scalar tail
    while (i < n) : (i += 1) {
        total += a[i] * b[i];
    }

    return total;
}

pub fn dotF16F32(a_bytes: []const u8, b: []const f32, n: usize) f32 {
    if (@intFromPtr(a_bytes.ptr) % @alignOf(f16) == 0) {
        const a: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, a_bytes[0 .. n * @sizeOf(f16)]));
        var total: f32 = 0.0;
        const Vec = @Vector(8, f32);
        const vec_len = 8;
        const n_vec = n / vec_len;

        var sum_vec: Vec = @splat(0.0);
        var i: usize = 0;

        while (i < n_vec * vec_len) : (i += vec_len) {
            var af32: [8]f32 = undefined;
            inline for (0..8) |k| {
                af32[k] = quant.f16ToF32(a[i + k]);
            }
            const va: Vec = af32;
            const vb: Vec = b[i..][0..vec_len].*;
            sum_vec += va * vb;
        }

        total = @reduce(.Add, sum_vec);

        while (i < n) : (i += 1) {
            total += quant.f16ToF32(a[i]) * b[i];
        }

        return total;
    } else {
        var total: f32 = 0.0;
        for (0..n) |i| {
            const u = std.mem.readInt(u16, a_bytes[i * 2 ..][0..2], .little);
            const f: f16 = @bitCast(u);
            total += quant.f16ToF32(f) * b[i];
        }
        return total;
    }
}

pub fn dotBf16F32(a_bytes: []const u8, b: []const f32, n: usize) f32 {
    if (@intFromPtr(a_bytes.ptr) % @alignOf(u16) == 0) {
        const a: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, a_bytes[0 .. n * @sizeOf(u16)]));
        var total: f32 = 0.0;
        const Vec = @Vector(8, f32);
        const vec_len = 8;
        const n_vec = n / vec_len;

        var sum_vec: Vec = @splat(0.0);
        var i: usize = 0;

        while (i < n_vec * vec_len) : (i += vec_len) {
            var af32: [8]f32 = undefined;
            inline for (0..8) |k| {
                af32[k] = quant.bf16ToF32(a[i + k]);
            }
            const va: Vec = af32;
            const vb: Vec = b[i..][0..vec_len].*;
            sum_vec += va * vb;
        }

        total = @reduce(.Add, sum_vec);

        while (i < n) : (i += 1) {
            total += quant.bf16ToF32(a[i]) * b[i];
        }

        return total;
    } else {
        var total: f32 = 0.0;
        for (0..n) |i| {
            const u = std.mem.readInt(u16, a_bytes[i * 2 ..][0..2], .little);
            total += quant.bf16ToF32(u) * b[i];
        }
        return total;
    }
}

pub fn dotQ8_0F32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 32;
    const block_size = @sizeOf(BlockQ8_0);
    var total: f32 = 0.0;

    for (0..n_blocks) |block_idx| {
        const blk: *const BlockQ8_0 = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        const d: f32 = @floatCast(blk.d);
        const b_slice = b[block_idx * 32 .. (block_idx + 1) * 32];

        var sum: f32 = 0.0;
        const Vec = @Vector(8, f32);
        inline for (0..4) |chunk| {
            var q_f32: [8]f32 = undefined;
            inline for (0..8) |k| {
                q_f32[k] = @floatFromInt(blk.qs[chunk * 8 + k]);
            }
            const q_vec: Vec = q_f32;
            const b_vec: Vec = b_slice[chunk * 8 ..][0..8].*;
            sum += @reduce(.Add, q_vec * b_vec);
        }

        total += sum * d;
    }

    return total;
}

pub fn dotQ8_1F32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 32;
    const block_size = @sizeOf(BlockQ8_1);
    var total: f32 = 0.0;

    for (0..n_blocks) |block_idx| {
        const blk: *const BlockQ8_1 = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        const d: f32 = @floatCast(blk.d);
        const b_slice = b[block_idx * 32 .. (block_idx + 1) * 32];

        var sum: f32 = 0.0;
        const Vec = @Vector(8, f32);
        inline for (0..4) |chunk| {
            var q_f32: [8]f32 = undefined;
            inline for (0..8) |k| {
                q_f32[k] = @floatFromInt(blk.qs[chunk * 8 + k]);
            }
            const q_vec: Vec = q_f32;
            const b_vec: Vec = b_slice[chunk * 8 ..][0..8].*;
            sum += @reduce(.Add, q_vec * b_vec);
        }

        total += sum * d;
    }

    return total;
}

pub fn dotQ4_0F32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 32;
    const block_size = @sizeOf(BlockQ4_0);
    var total: f32 = 0.0;

    for (0..n_blocks) |block_idx| {
        const blk: *const BlockQ4_0 = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        const d: f32 = @floatCast(blk.d);
        const b_slice = b[block_idx * 32 .. (block_idx + 1) * 32];

        var sum: f32 = 0.0;
        for (0..16) |i| {
            const qs = blk.qs[i];
            const v0: f32 = @floatFromInt(@as(i32, @intCast(qs & 0x0F)) - 8);
            const v1: f32 = @floatFromInt(@as(i32, @intCast((qs >> 4) & 0x0F)) - 8);
            sum += v0 * b_slice[i] + v1 * b_slice[i + 16];
        }

        total += sum * d;
    }

    return total;
}

pub fn dotQ4_1F32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 32;
    const block_size = @sizeOf(BlockQ4_1);
    var total: f32 = 0.0;

    for (0..n_blocks) |block_idx| {
        const blk: *const BlockQ4_1 = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        const d: f32 = @floatCast(blk.d);
        const m: f32 = @floatCast(blk.m);
        const b_slice = b[block_idx * 32 .. (block_idx + 1) * 32];

        var sum_q: f32 = 0.0;
        var sum_b: f32 = 0.0;
        for (0..16) |i| {
            const qs = blk.qs[i];
            const v0: f32 = @floatFromInt(qs & 0x0F);
            const v1: f32 = @floatFromInt((qs >> 4) & 0x0F);
            sum_q += v0 * b_slice[i] + v1 * b_slice[i + 16];
            sum_b += b_slice[i] + b_slice[i + 16];
        }

        total += sum_q * d + sum_b * m;
    }

    return total;
}

pub fn dotQ5_0F32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 32;
    const block_size = @sizeOf(BlockQ5_0);
    var total: f32 = 0.0;

    var temp_buf: [32]f32 = undefined;
    for (0..n_blocks) |block_idx| {
        const blk: *const BlockQ5_0 = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        quant.dequantizeQ5_0(blk, &temp_buf);
        total += dotF32F32(&temp_buf, b[block_idx * 32 .. (block_idx + 1) * 32]);
    }

    return total;
}

pub fn dotQ5_1F32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 32;
    const block_size = @sizeOf(BlockQ5_1);
    var total: f32 = 0.0;

    var temp_buf: [32]f32 = undefined;
    for (0..n_blocks) |block_idx| {
        const blk: *const BlockQ5_1 = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        quant.dequantizeQ5_1(blk, &temp_buf);
        total += dotF32F32(&temp_buf, b[block_idx * 32 .. (block_idx + 1) * 32]);
    }

    return total;
}

pub fn dotQ2_KF32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 256;
    const block_size = @sizeOf(BlockQ2_K);
    var total: f32 = 0.0;

    var temp_buf: [256]f32 = undefined;
    for (0..n_blocks) |block_idx| {
        const blk: *const BlockQ2_K = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        quant.dequantizeQ2_K(blk, &temp_buf);
        total += dotF32F32(&temp_buf, b[block_idx * 256 .. (block_idx + 1) * 256]);
    }

    return total;
}

pub fn dotQ3_KF32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 256;
    const block_size = @sizeOf(BlockQ3_K);
    var total: f32 = 0.0;

    var temp_buf: [256]f32 = undefined;
    for (0..n_blocks) |block_idx| {
        const blk: *const BlockQ3_K = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        quant.dequantizeQ3_K(blk, &temp_buf);
        total += dotF32F32(&temp_buf, b[block_idx * 256 .. (block_idx + 1) * 256]);
    }

    return total;
}

pub fn dotQ4_KF32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 256;
    const block_size = @sizeOf(BlockQ4_K);
    var total: f32 = 0.0;

    var temp_buf: [256]f32 = undefined;
    for (0..n_blocks) |block_idx| {
        const blk: *const BlockQ4_K = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        quant.dequantizeQ4_K(blk, &temp_buf);
        total += dotF32F32(&temp_buf, b[block_idx * 256 .. (block_idx + 1) * 256]);
    }

    return total;
}

pub fn dotQ5_KF32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 256;
    const block_size = @sizeOf(BlockQ5_K);
    var total: f32 = 0.0;

    var temp_buf: [256]f32 = undefined;
    for (0..n_blocks) |block_idx| {
        const blk: *const BlockQ5_K = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        quant.dequantizeQ5_K(blk, &temp_buf);
        total += dotF32F32(&temp_buf, b[block_idx * 256 .. (block_idx + 1) * 256]);
    }

    return total;
}

pub fn dotQ6_KF32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 256;
    const block_size = @sizeOf(BlockQ6_K);
    var total: f32 = 0.0;

    var temp_buf: [256]f32 = undefined;
    for (0..n_blocks) |block_idx| {
        const blk: *const BlockQ6_K = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        quant.dequantizeQ6_K(blk, &temp_buf);
        total += dotF32F32(&temp_buf, b[block_idx * 256 .. (block_idx + 1) * 256]);
    }

    return total;
}

pub fn dotQ8_KF32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 256;
    const block_size = @sizeOf(BlockQ8_K);
    var total: f32 = 0.0;

    var temp_buf: [256]f32 = undefined;
    for (0..n_blocks) |block_idx| {
        const blk: *const BlockQ8_K = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        quant.dequantizeQ8_K(blk, &temp_buf);
        total += dotF32F32(&temp_buf, b[block_idx * 256 .. (block_idx + 1) * 256]);
    }

    return total;
}

pub fn dotIQ4_NLF32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 32;
    const block_size = @sizeOf(BlockIQ4_NL);
    var total: f32 = 0.0;

    var temp_buf: [32]f32 = undefined;
    for (0..n_blocks) |block_idx| {
        const blk: *const BlockIQ4_NL = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        quant.dequantizeIQ4_NL(blk, &temp_buf);
        total += dotF32F32(&temp_buf, b[block_idx * 32 .. (block_idx + 1) * 32]);
    }

    return total;
}

pub fn dotRow(qtype: GGMLType, row_bytes: []const u8, x: []const f32, cols: usize) f32 {
    return switch (qtype) {
        .F32 => blk: {
            const bytes = row_bytes[0 .. cols * @sizeOf(f32)];
            if (@intFromPtr(bytes.ptr) % @alignOf(f32) == 0) {
                break :blk dotF32F32(@alignCast(std.mem.bytesAsSlice(f32, bytes)), x[0..cols]);
            } else {
                var total: f32 = 0.0;
                for (0..cols) |i| {
                    const u = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
                    const f: f32 = @bitCast(u);
                    total += f * x[i];
                }
                break :blk total;
            }
        },
        .F16 => dotF16F32(row_bytes, x, cols),
        .BF16 => dotBf16F32(row_bytes, x, cols),
        .Q8_0 => dotQ8_0F32(row_bytes, x, cols),
        .Q8_1 => dotQ8_1F32(row_bytes, x, cols),
        .Q4_0 => dotQ4_0F32(row_bytes, x, cols),
        .Q4_1 => dotQ4_1F32(row_bytes, x, cols),
        .Q5_0 => dotQ5_0F32(row_bytes, x, cols),
        .Q5_1 => dotQ5_1F32(row_bytes, x, cols),
        .Q2_K => dotQ2_KF32(row_bytes, x, cols),
        .Q3_K => dotQ3_KF32(row_bytes, x, cols),
        .Q4_K => dotQ4_KF32(row_bytes, x, cols),
        .Q5_K => dotQ5_KF32(row_bytes, x, cols),
        .Q6_K => dotQ6_KF32(row_bytes, x, cols),
        .Q8_K => dotQ8_KF32(row_bytes, x, cols),
        .IQ4_NL => dotIQ4_NLF32(row_bytes, x, cols),
        else => blk: {
            var temp_buf: [256]f32 = undefined;
            const blk_size = qtype.blockSize();
            const type_size = qtype.typeSize();
            const n_blocks = cols / blk_size;
            var total: f32 = 0.0;

            for (0..n_blocks) |b| {
                const b_start = b * blk_size;
                const src_block = row_bytes[b * type_size .. (b + 1) * type_size];
                quant.dequantizeRow(qtype, src_block, temp_buf[0..blk_size], blk_size);
                total += dotF32F32(temp_buf[0..blk_size], x[b_start .. b_start + blk_size]);
            }
            break :blk total;
        },
    };
}

// ============================================================================
// Matrix-Vector Multiplication (GEMV)
// ============================================================================

pub const GemvContext = struct {
    qtype: GGMLType,
    weight_data: []const u8,
    row_bytes: usize,
    x: []const f32,
    y: []f32,
    cols: usize,
};

fn gemvWorker(ctx_ptr: ?*anyopaque, row_idx: usize, _: usize) void {
    const ctx: *const GemvContext = @ptrCast(@alignCast(ctx_ptr.?));
    const row_start = row_idx * ctx.row_bytes;
    const row_data = ctx.weight_data[row_start .. row_start + ctx.row_bytes];
    ctx.y[row_idx] = dotRow(ctx.qtype, row_data, ctx.x, ctx.cols);
}

pub fn gemv(
    pool: ?*ThreadPool,
    qtype: GGMLType,
    weight_data: []const u8,
    x: []const f32,
    y: []f32,
    rows: usize,
    cols: usize,
) void {
    const blk_size = qtype.blockSize();
    const type_size = qtype.typeSize();
    const row_bytes = (cols / blk_size) * type_size;

    var ctx = GemvContext{
        .qtype = qtype,
        .weight_data = weight_data,
        .row_bytes = row_bytes,
        .x = x,
        .y = y,
        .cols = cols,
    };

    if (pool) |p| {
        p.parallelFor(rows, &ctx, gemvWorker);
    } else {
        for (0..rows) |r| {
            gemvWorker(&ctx, r, 0);
        }
    }
}

pub fn gemm(
    pool: ?*ThreadPool,
    qtype: GGMLType,
    weight_data: []const u8,
    X: []const f32,
    Y: []f32,
    batch_size: usize,
    rows: usize,
    cols: usize,
) void {
    for (0..batch_size) |b| {
        const x_slice = X[b * cols .. (b + 1) * cols];
        const y_slice = Y[b * rows .. (b + 1) * rows];
        gemv(pool, qtype, weight_data, x_slice, y_slice, rows, cols);
    }
}

// ============================================================================
// Normalization Layers
// ============================================================================

pub fn rmsNorm(
    x: []const f32,
    weight: []const f32,
    out: []f32,
    eps: f32,
    use_unit_offset: bool,
) void {
    std.debug.assert(x.len == weight.len and x.len == out.len);
    const n = x.len;

    var sum_sq: f32 = 0.0;
    const Vec = @Vector(8, f32);
    const vec_len = 8;
    const n_vec = n / vec_len;

    var sum_vec: Vec = @splat(0.0);
    var i: usize = 0;

    while (i < n_vec * vec_len) : (i += vec_len) {
        const vx: Vec = x[i..][0..vec_len].*;
        sum_vec += vx * vx;
    }

    sum_sq = @reduce(.Add, sum_vec);

    while (i < n) : (i += 1) {
        sum_sq += x[i] * x[i];
    }

    const mean_sq = sum_sq / @as(f32, @floatFromInt(n));
    const inv_std = 1.0 / @sqrt(mean_sq + eps);

    i = 0;
    const inv_std_vec: Vec = @splat(inv_std);
    const one_vec: Vec = @splat(1.0);

    while (i < n_vec * vec_len) : (i += vec_len) {
        const vx: Vec = x[i..][0..vec_len].*;
        const vw: Vec = weight[i..][0..vec_len].*;
        const norm_x = vx * inv_std_vec;

        const res = if (use_unit_offset)
            norm_x * (one_vec + vw)
        else
            norm_x * vw;

        out[i..][0..vec_len].* = res;
    }

    while (i < n) : (i += 1) {
        const norm_x = x[i] * inv_std;
        out[i] = if (use_unit_offset)
            norm_x * (1.0 + weight[i])
        else
            norm_x * weight[i];
    }
}

pub fn rmsNormNoScale(
    x: []const f32,
    out: []f32,
    eps: f32,
) void {
    std.debug.assert(x.len == out.len);
    const n = x.len;

    var sum_sq: f32 = 0.0;
    const Vec = @Vector(8, f32);
    const vec_len = 8;
    const n_vec = n / vec_len;

    var sum_vec: Vec = @splat(0.0);
    var i: usize = 0;

    while (i < n_vec * vec_len) : (i += vec_len) {
        const vx: Vec = x[i..][0..vec_len].*;
        sum_vec += vx * vx;
    }

    sum_sq = @reduce(.Add, sum_vec);

    while (i < n) : (i += 1) {
        sum_sq += x[i] * x[i];
    }

    const mean_sq = sum_sq / @as(f32, @floatFromInt(n));
    const inv_std = 1.0 / @sqrt(mean_sq + eps);

    i = 0;
    const inv_std_vec: Vec = @splat(inv_std);

    while (i < n_vec * vec_len) : (i += vec_len) {
        const vx: Vec = x[i..][0..vec_len].*;
        out[i..][0..vec_len].* = vx * inv_std_vec;
    }

    while (i < n) : (i += 1) {
        out[i] = x[i] * inv_std;
    }
}

pub fn layerNorm(
    x: []const f32,
    weight: []const f32,
    bias: ?[]const f32,
    out: []f32,
    eps: f32,
) void {
    const n = x.len;
    var sum: f32 = 0.0;
    for (x) |v| sum += v;
    const mean = sum / @as(f32, @floatFromInt(n));

    var sum_sq: f32 = 0.0;
    for (x) |v| {
        const diff = v - mean;
        sum_sq += diff * diff;
    }
    const inv_std = 1.0 / @sqrt((sum_sq / @as(f32, @floatFromInt(n))) + eps);

    for (0..n) |i| {
        const norm_val = (x[i] - mean) * inv_std * weight[i];
        out[i] = if (bias) |b| norm_val + b[i] else norm_val;
    }
}

// ============================================================================
// Positional Embeddings: RoPE (Rotary Position Embeddings)
// ============================================================================

pub fn applyRoPE(
    q: []f32,
    k: []f32,
    pos: usize,
    head_size: usize,
    rotary_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    freq_base: f32,
    freq_scale: f32,
    rope_type: RoPEType,
) void {
    const pos_f32 = @as(f32, @floatFromInt(pos)) * freq_scale;
    const eff_rotary = if (rotary_dim == 0 or rotary_dim > head_size) head_size else rotary_dim;
    const half_dim = eff_rotary / 2;

    // Apply to Q
    for (0..n_heads) |h| {
        const q_head = q[h * head_size .. (h + 1) * head_size];
        applyRoPEHead(q_head[0..eff_rotary], pos_f32, eff_rotary, half_dim, freq_base, rope_type);
    }

    // Apply to K
    for (0..n_kv_heads) |h| {
        const k_head = k[h * head_size .. (h + 1) * head_size];
        applyRoPEHead(k_head[0..eff_rotary], pos_f32, eff_rotary, half_dim, freq_base, rope_type);
    }
}

fn applyRoPEHead(
    head: []f32,
    pos_f32: f32,
    head_size: usize,
    half_dim: usize,
    freq_base: f32,
    rope_type: RoPEType,
) void {
    var i: usize = 0;
    while (i < half_dim) : (i += 1) {
        const exponent = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_size));
        const theta = pos_f32 / std.math.pow(f32, freq_base, exponent);
        const cos_theta = @cos(theta);
        const sin_theta = @sin(theta);

        switch (rope_type) {
            .normal => {
                const v0 = head[i];
                const v1 = head[i + half_dim];
                head[i] = v0 * cos_theta - v1 * sin_theta;
                head[i + half_dim] = v0 * sin_theta + v1 * cos_theta;
            },
            .neox => {
                const v0 = head[2 * i];
                const v1 = head[2 * i + 1];
                head[2 * i] = v0 * cos_theta - v1 * sin_theta;
                head[2 * i + 1] = v0 * sin_theta + v1 * cos_theta;
            },
        }
    }
}

// ============================================================================
// Activation Functions & Softmax
// ============================================================================

pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

pub fn softmax(x: []f32) void {
    if (x.len == 0) return;
    var max_val = x[0];
    for (x[1..]) |v| {
        if (v > max_val) max_val = v;
    }

    var sum: f32 = 0.0;
    for (x) |*v| {
        const exp_v = @exp(v.* - max_val);
        v.* = exp_v;
        sum += exp_v;
    }

    const inv_sum = 1.0 / sum;
    for (x) |*v| {
        v.* *= inv_sum;
    }
}

pub fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

pub fn swiglu(gate: []const f32, up: []const f32, out: []f32) void {
    std.debug.assert(gate.len == up.len and gate.len == out.len);
    for (0..gate.len) |i| {
        out[i] = silu(gate[i]) * up[i];
    }
}

pub fn gelu(x: f32) f32 {
    const sqrt_2_over_pi: f32 = 0.7978845608;
    const coef: f32 = 0.044715;
    return 0.5 * x * (1.0 + std.math.tanh(sqrt_2_over_pi * (x + coef * x * x * x)));
}

pub fn geglu(gate: []const f32, up: []const f32, out: []f32) void {
    std.debug.assert(gate.len == up.len and gate.len == out.len);
    for (0..gate.len) |i| {
        out[i] = gelu(gate[i]) * up[i];
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

test "SIMD dot product correctness" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    const b = [_]f32{ 2.0, 1.0, 0.5, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0 };

    var expected: f32 = 0.0;
    for (0..a.len) |i| {
        expected += a[i] * b[i];
    }

    const actual = dotF32F32(&a, &b);
    try std.testing.expectApproxEqAbs(expected, actual, 1e-5);
}

test "RMSNorm computation" {
    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var out: [4]f32 = undefined;

    rmsNorm(&x, &weight, &out, 1e-5, false);

    var sum_sq: f32 = 0.0;
    for (out) |v| sum_sq += v * v;
    const rms_out = @sqrt(sum_sq / 4.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), rms_out, 1e-4);
}

test "Softmax sum to 1.0" {
    var scores = [_]f32{ 2.0, 1.0, 0.1, -1.0, 5.0 };
    softmax(&scores);

    var sum: f32 = 0.0;
    for (scores) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "RoPE positional embedding" {
    var q = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var k = [_]f32{ 1.0, 0.0, 0.0, 1.0 };

    applyRoPE(&q, &k, 1, 4, 4, 1, 1, 10000.0, 1.0, .normal);
    try std.testing.expect(q[0] != 1.0 or q[1] != 0.0);
}
