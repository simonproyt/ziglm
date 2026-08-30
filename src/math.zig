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
    const n_unroll = n / 32;

    var sum0: Vec = @splat(0.0);
    var sum1: Vec = @splat(0.0);
    var sum2: Vec = @splat(0.0);
    var sum3: Vec = @splat(0.0);

    var i: usize = 0;
    while (i < n_unroll * 32) : (i += 32) {
        const va0: Vec = a[i..][0..8].*;
        const vb0: Vec = b[i..][0..8].*;
        const va1: Vec = a[i + 8 ..][0..8].*;
        const vb1: Vec = b[i + 8 ..][0..8].*;
        const va2: Vec = a[i + 16 ..][0..8].*;
        const vb2: Vec = b[i + 16 ..][0..8].*;
        const va3: Vec = a[i + 24 ..][0..8].*;
        const vb3: Vec = b[i + 24 ..][0..8].*;

        sum0 += va0 * vb0;
        sum1 += va1 * vb1;
        sum2 += va2 * vb2;
        sum3 += va3 * vb3;
    }

    while (i + vec_len <= n) : (i += vec_len) {
        const va: Vec = a[i..][0..vec_len].*;
        const vb: Vec = b[i..][0..vec_len].*;
        sum0 += va * vb;
    }

    var total = @reduce(.Add, (sum0 + sum1) + (sum2 + sum3));

    while (i < n) : (i += 1) {
        total += a[i] * b[i];
    }

    return total;
}

pub fn dotF16F32(a_bytes: []const u8, b: []const f32, n: usize) f32 {
    const Vec = @Vector(8, f32);
    const Vec16 = @Vector(8, f16);
    const vec_len = 8;
    const n_unroll = n / 32;

    var sum0: Vec = @splat(0.0);
    var sum1: Vec = @splat(0.0);
    var sum2: Vec = @splat(0.0);
    var sum3: Vec = @splat(0.0);
    var i: usize = 0;

    while (i < n_unroll * 32) : (i += 32) {
        const raw0: Vec16 = @bitCast(@as(*align(1) const [16]u8, @ptrCast(a_bytes[i * 2 ..][0..16])).*[0..16].*);
        const raw1: Vec16 = @bitCast(@as(*align(1) const [16]u8, @ptrCast(a_bytes[(i + 8) * 2 ..][0..16])).*[0..16].*);
        const raw2: Vec16 = @bitCast(@as(*align(1) const [16]u8, @ptrCast(a_bytes[(i + 16) * 2 ..][0..16])).*[0..16].*);
        const raw3: Vec16 = @bitCast(@as(*align(1) const [16]u8, @ptrCast(a_bytes[(i + 24) * 2 ..][0..16])).*[0..16].*);

        const va0: Vec = @floatCast(raw0);
        const va1: Vec = @floatCast(raw1);
        const va2: Vec = @floatCast(raw2);
        const va3: Vec = @floatCast(raw3);

        const vb0: Vec = b[i..][0..8].*;
        const vb1: Vec = b[i + 8 ..][0..8].*;
        const vb2: Vec = b[i + 16 ..][0..8].*;
        const vb3: Vec = b[i + 24 ..][0..8].*;

        sum0 += va0 * vb0;
        sum1 += va1 * vb1;
        sum2 += va2 * vb2;
        sum3 += va3 * vb3;
    }

    while (i + vec_len <= n) : (i += vec_len) {
        const raw: Vec16 = @bitCast(@as(*align(1) const [16]u8, @ptrCast(a_bytes[i * 2 ..][0..16])).*[0..16].*);
        const va: Vec = @floatCast(raw);
        const vb: Vec = b[i..][0..8].*;
        sum0 += va * vb;
    }

    var total = @reduce(.Add, (sum0 + sum1) + (sum2 + sum3));

    while (i < n) : (i += 1) {
        const u = std.mem.readInt(u16, a_bytes[i * 2 ..][0..2], .little);
        const f: f16 = @bitCast(u);
        total += quant.f16ToF32(f) * b[i];
    }

    return total;
}

pub fn dotBf16F32(a_bytes: []const u8, b: []const f32, n: usize) f32 {
    const Vec = @Vector(8, f32);
    const VecU = @Vector(8, u32);
    const vec_len = 8;
    const n_unroll = n / 32;

    var sum0: Vec = @splat(0.0);
    var sum1: Vec = @splat(0.0);
    var sum2: Vec = @splat(0.0);
    var sum3: Vec = @splat(0.0);
    var i: usize = 0;

    while (i < n_unroll * 32) : (i += 32) {
        const raw0: @Vector(8, u16) = @bitCast(@as(*align(1) const [16]u8, @ptrCast(a_bytes[i * 2 ..][0..16])).*[0..16].*);
        const raw1: @Vector(8, u16) = @bitCast(@as(*align(1) const [16]u8, @ptrCast(a_bytes[(i + 8) * 2 ..][0..16])).*[0..16].*);
        const raw2: @Vector(8, u16) = @bitCast(@as(*align(1) const [16]u8, @ptrCast(a_bytes[(i + 16) * 2 ..][0..16])).*[0..16].*);
        const raw3: @Vector(8, u16) = @bitCast(@as(*align(1) const [16]u8, @ptrCast(a_bytes[(i + 24) * 2 ..][0..16])).*[0..16].*);

        const val0: VecU = @as(VecU, raw0) << @as(VecU, @splat(16));
        const val1: VecU = @as(VecU, raw1) << @as(VecU, @splat(16));
        const val2: VecU = @as(VecU, raw2) << @as(VecU, @splat(16));
        const val3: VecU = @as(VecU, raw3) << @as(VecU, @splat(16));

        const va0: Vec = @bitCast(val0);
        const va1: Vec = @bitCast(val1);
        const va2: Vec = @bitCast(val2);
        const va3: Vec = @bitCast(val3);

        const vb0: Vec = b[i..][0..8].*;
        const vb1: Vec = b[i + 8 ..][0..8].*;
        const vb2: Vec = b[i + 16 ..][0..8].*;
        const vb3: Vec = b[i + 24 ..][0..8].*;

        sum0 += va0 * vb0;
        sum1 += va1 * vb1;
        sum2 += va2 * vb2;
        sum3 += va3 * vb3;
    }

    while (i + vec_len <= n) : (i += vec_len) {
        const raw: @Vector(8, u16) = @bitCast(@as(*align(1) const [16]u8, @ptrCast(a_bytes[i * 2 ..][0..16])).*[0..16].*);
        const val: VecU = @as(VecU, raw) << @as(VecU, @splat(16));
        const va: Vec = @bitCast(val);
        const vb: Vec = b[i..][0..8].*;
        sum0 += va * vb;
    }

    var total = @reduce(.Add, (sum0 + sum1) + (sum2 + sum3));

    while (i < n) : (i += 1) {
        const u = std.mem.readInt(u16, a_bytes[i * 2 ..][0..2], .little);
        total += quant.bf16ToF32(u) * b[i];
    }

    return total;
}

pub fn dotQ8_0F32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 32;
    const block_size = @sizeOf(BlockQ8_0);
    const Vec = @Vector(16, f32);
    var sum0: Vec = @splat(0.0);
    var sum1: Vec = @splat(0.0);

    var block_idx: usize = 0;
    while (block_idx + 2 <= n_blocks) : (block_idx += 2) {
        const blk0: *const BlockQ8_0 = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        const blk1: *const BlockQ8_0 = @ptrCast(@alignCast(a_bytes[(block_idx + 1) * block_size .. (block_idx + 2) * block_size].ptr));

        const vd0: Vec = @splat(@as(f32, @floatCast(blk0.d)));
        const vd1: Vec = @splat(@as(f32, @floatCast(blk1.d)));

        const b_ptr0 = b.ptr + block_idx * 32;
        const b_ptr1 = b.ptr + (block_idx + 1) * 32;

        const vb0_0: Vec = b_ptr0[0..16].*;
        const vb0_1: Vec = b_ptr0[16..32].*;
        const vb1_0: Vec = b_ptr1[0..16].*;
        const vb1_1: Vec = b_ptr1[16..32].*;

        const qs0_0: @Vector(16, i8) = blk0.qs[0..16].*;
        const qs0_1: @Vector(16, i8) = blk0.qs[16..32].*;
        const qs1_0: @Vector(16, i8) = blk1.qs[0..16].*;
        const qs1_1: @Vector(16, i8) = blk1.qs[16..32].*;
        
        const vq0_0: Vec = @floatFromInt(@as(@Vector(16, i32), qs0_0));
        const vq0_1: Vec = @floatFromInt(@as(@Vector(16, i32), qs0_1));
        const vq1_0: Vec = @floatFromInt(@as(@Vector(16, i32), qs1_0));
        const vq1_1: Vec = @floatFromInt(@as(@Vector(16, i32), qs1_1));

        sum0 += (vq0_0 * vb0_0 + vq0_1 * vb0_1) * vd0;
        sum1 += (vq1_0 * vb1_0 + vq1_1 * vb1_1) * vd1;
    }

    while (block_idx < n_blocks) : (block_idx += 1) {
        const blk: *const BlockQ8_0 = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        const vd: Vec = @splat(@as(f32, @floatCast(blk.d)));
        const b_ptr = b.ptr + block_idx * 32;

        const vb0: Vec = b_ptr[0..16].*;
        const vb1: Vec = b_ptr[16..32].*;

        var q0: [16]f32 = undefined;
        var q1: [16]f32 = undefined;
        inline for (0..16) |k| {
            q0[k] = @floatFromInt(blk.qs[k]);
            q1[k] = @floatFromInt(blk.qs[k + 16]);
        }
        const vq0: Vec = q0;
        const vq1: Vec = q1;
        sum0 += (vq0 * vb0 + vq1 * vb1) * vd;
    }

    return @reduce(.Add, sum0 + sum1);
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
    const Vec = @Vector(16, f32);
    const eight: Vec = @splat(8.0);
    var sum0: Vec = @splat(0.0);
    var sum1: Vec = @splat(0.0);

    var block_idx: usize = 0;
    while (block_idx + 2 <= n_blocks) : (block_idx += 2) {
        const blk0: *const BlockQ4_0 = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        const blk1: *const BlockQ4_0 = @ptrCast(@alignCast(a_bytes[(block_idx + 1) * block_size .. (block_idx + 2) * block_size].ptr));

        const vd0: Vec = @splat(@as(f32, @floatCast(blk0.d)));
        const vd1: Vec = @splat(@as(f32, @floatCast(blk1.d)));

        const b_ptr0 = b.ptr + block_idx * 32;
        const b_ptr1 = b.ptr + (block_idx + 1) * 32;

        const vb0_0: Vec = b_ptr0[0..16].*;
        const vb0_1: Vec = b_ptr0[16..32].*;
        const vb1_0: Vec = b_ptr1[0..16].*;
        const vb1_1: Vec = b_ptr1[16..32].*;

        const byte0: @Vector(16, u8) = blk0.qs;
        const byte1: @Vector(16, u8) = blk1.qs;
        const mask: @Vector(16, u8) = @splat(0x0F);
        const shift: @Vector(16, u3) = @splat(4);

        const vq0_0: Vec = @floatFromInt(@as(@Vector(16, i32), byte0 & mask));
        const vq0_1: Vec = @floatFromInt(@as(@Vector(16, i32), byte0 >> shift));
        const vq1_0: Vec = @floatFromInt(@as(@Vector(16, i32), byte1 & mask));
        const vq1_1: Vec = @floatFromInt(@as(@Vector(16, i32), byte1 >> shift));

        const acc0 = (vq0_0 * vb0_0 + vq0_1 * vb0_1) - eight * (vb0_0 + vb0_1);
        const acc1 = (vq1_0 * vb1_0 + vq1_1 * vb1_1) - eight * (vb1_0 + vb1_1);

        sum0 += acc0 * vd0;
        sum1 += acc1 * vd1;
    }

    while (block_idx < n_blocks) : (block_idx += 1) {
        const blk: *const BlockQ4_0 = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        const vd: Vec = @splat(@as(f32, @floatCast(blk.d)));
        const b_ptr = b.ptr + block_idx * 32;

        const vb0: Vec = b_ptr[0..16].*;
        const vb1: Vec = b_ptr[16..32].*;

        const byte: @Vector(16, u8) = blk.qs;
        const mask: @Vector(16, u8) = @splat(0x0F);
        const shift: @Vector(16, u3) = @splat(4);

        const vq0: Vec = @floatFromInt(@as(@Vector(16, i32), byte & mask));
        const vq1: Vec = @floatFromInt(@as(@Vector(16, i32), byte >> shift));

        const acc = (vq0 * vb0 + vq1 * vb1) - eight * (vb0 + vb1);
        sum0 += acc * vd;
    }

    return @reduce(.Add, sum0 + sum1);
}

pub fn dotQ4_1F32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 32;
    const block_size = @sizeOf(BlockQ4_1);
    const Vec = @Vector(16, f32);
    var total: f32 = 0.0;

    for (0..n_blocks) |block_idx| {
        const blk: *const BlockQ4_1 = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        const d: f32 = @floatCast(blk.d);
        const m: f32 = @floatCast(blk.m);
        const b_ptr = b.ptr + block_idx * 32;

        const vb0: Vec = b_ptr[0..16].*;
        const vb1: Vec = b_ptr[16..32].*;

        const byte: @Vector(16, u8) = blk.qs;
        const mask: @Vector(16, u8) = @splat(0x0F);
        const shift: @Vector(16, u3) = @splat(4);

        const vq0: Vec = @floatFromInt(@as(@Vector(16, i32), byte & mask));
        const vq1: Vec = @floatFromInt(@as(@Vector(16, i32), byte >> shift));

        const sum_q = @reduce(.Add, vq0 * vb0 + vq1 * vb1);
        const sum_b = @reduce(.Add, vb0 + vb1);

        total += sum_q * d + sum_b * m;
    }

    return total;
}

pub fn dotQ5_0F32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 32;
    const block_size = @sizeOf(BlockQ5_0);
    const Vec = @Vector(16, f32);
    const sixteen: Vec = @splat(16.0);
    var total: f32 = 0.0;

    for (0..n_blocks) |block_idx| {
        const blk: *const BlockQ5_0 = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        const d: f32 = @floatCast(blk.d);
        const b_ptr = b.ptr + block_idx * 32;
        const qh = std.mem.readInt(u32, blk.qh[0..4], .little);

        const vb0: Vec = b_ptr[0..16].*;
        const vb1: Vec = b_ptr[16..32].*;

        var v0_arr: [16]f32 = undefined;
        var v1_arr: [16]f32 = undefined;
        inline for (0..16) |k| {
            const qs = blk.qs[k];
            const h0: u8 = if ((qh & (@as(u32, 1) << @intCast(k))) != 0) 16 else 0;
            const h1: u8 = if ((qh & (@as(u32, 1) << @intCast(k + 16))) != 0) 16 else 0;

            v0_arr[k] = @floatFromInt((qs & 0x0F) | h0);
            v1_arr[k] = @floatFromInt((qs >> 4) | h1);
        }

        const vq0: Vec = v0_arr;
        const vq1: Vec = v1_arr;

        total += @reduce(.Add, (vq0 - sixteen) * vb0 + (vq1 - sixteen) * vb1) * d;
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
    const Vec = @Vector(16, f32);
    var total_sum: Vec = @splat(0.0);

    for (0..n_blocks) |block_idx| {
        const blk: *const BlockQ4_K = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        const d: f32 = @floatCast(blk.d);
        const min: f32 = @floatCast(blk.dmin);
        const b_ptr = b.ptr + block_idx * 256;

        var scales: [8]u8 = undefined;
        var mins: [8]u8 = undefined;

        inline for (0..4) |j| {
            scales[j] = blk.scales[j] & 63;
            mins[j] = blk.scales[j + 4] & 63;
            scales[j + 4] = (blk.scales[j + 8] & 0x0F) | ((blk.scales[j] >> 6) << 4);
            mins[j + 4] = (blk.scales[j + 8] >> 4) | ((blk.scales[j + 4] >> 6) << 4);
        }

        var qs_idx: usize = 0;
        var b_idx: usize = 0;

        for (0..4) |j| {
            const vd0: Vec = @splat(d * @as(f32, @floatFromInt(scales[2 * j])));
            const vm0: Vec = @splat(min * @as(f32, @floatFromInt(mins[2 * j])));
            const vd1: Vec = @splat(d * @as(f32, @floatFromInt(scales[2 * j + 1])));
            const vm1: Vec = @splat(min * @as(f32, @floatFromInt(mins[2 * j + 1])));

            for (0..2) |half| {
                const off = half * 16;
                var q0_arr: [16]f32 = undefined;
                var q1_arr: [16]f32 = undefined;

                inline for (0..16) |k| {
                    const q = blk.qs[qs_idx + off + k];
                    q0_arr[k] = @floatFromInt(q & 0x0F);
                    q1_arr[k] = @floatFromInt(q >> 4);
                }

                const vq0: Vec = q0_arr;
                const vq1: Vec = q1_arr;
                const vb0: Vec = b_ptr[b_idx + off ..][0..16].*;
                const vb1: Vec = b_ptr[b_idx + 32 + off ..][0..16].*;

                total_sum += (vq0 * vd0 - vm0) * vb0 + (vq1 * vd1 - vm1) * vb1;
            }

            qs_idx += 32;
            b_idx += 64;
        }
    }

    return @reduce(.Add, total_sum);
}

pub fn dotQ5_KF32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 256;
    const block_size = @sizeOf(BlockQ5_K);
    var total: f32 = 0.0;

    for (0..n_blocks) |block_idx| {
        const blk: *const BlockQ5_K = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        const d: f32 = @floatCast(blk.d);
        const min: f32 = @floatCast(blk.dmin);
        const b_ptr = b.ptr + block_idx * 256;

        var scales: [8]u8 = undefined;
        var mins: [8]u8 = undefined;

        inline for (0..4) |j| {
            scales[j] = blk.scales[j] & 63;
            mins[j] = blk.scales[j + 4] & 63;
            scales[j + 4] = (blk.scales[j + 8] & 0x0F) | ((blk.scales[j] >> 6) << 4);
            mins[j + 4] = (blk.scales[j + 8] >> 4) | ((blk.scales[j + 4] >> 6) << 4);
        }

        var ql_idx: usize = 0;
        var b_idx: usize = 0;
        var mask1: u8 = 1;
        var mask2: u8 = 2;
        var block_sum: f32 = 0.0;

        for (0..4) |j| {
            const d0 = d * @as(f32, @floatFromInt(scales[2 * j]));
            const m0 = min * @as(f32, @floatFromInt(mins[2 * j]));
            const d1 = d * @as(f32, @floatFromInt(scales[2 * j + 1]));
            const m1 = min * @as(f32, @floatFromInt(mins[2 * j + 1]));

            for (0..32) |l| {
                const ql = blk.qs[ql_idx + l];
                const qh = blk.qh[l];

                const h0: u8 = if ((qh & mask1) != 0) 16 else 0;
                const h1: u8 = if ((qh & mask2) != 0) 16 else 0;

                const v0 = @as(f32, @floatFromInt((ql & 0x0F) | h0)) * d0 - m0;
                const v1 = @as(f32, @floatFromInt(((ql >> 4) & 0x0F) | h1)) * d1 - m1;

                block_sum += v0 * b_ptr[b_idx + l] + v1 * b_ptr[b_idx + 32 + l];
            }

            ql_idx += 32;
            b_idx += 64;
            mask1 <<= 2;
            mask2 <<= 2;
        }

        total += block_sum;
    }

    return total;
}

pub fn dotQ6_KF32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 256;
    const block_size = @sizeOf(BlockQ6_K);
    const Vec = @Vector(16, f32);
    var total_sum: Vec = @splat(0.0);

    for (0..n_blocks) |block_idx| {
        const blk: *const BlockQ6_K = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        const d: f32 = @floatCast(blk.d);
        const vd: Vec = @splat(d);
        const b_ptr = b.ptr + block_idx * 256;

        for (0..2) |n| {
            const ql_offset = n * 64;
            const qh_offset = n * 32;
            const sc_offset = n * 8;
            const b_offset = n * 128;

            for (0..2) |chunk| {
                const l_start = chunk * 16;
                const is = chunk;

                var q1_arr: [16]f32 = undefined;
                var q2_arr: [16]f32 = undefined;
                var q3_arr: [16]f32 = undefined;
                var q4_arr: [16]f32 = undefined;

                inline for (0..16) |k| {
                    const l = l_start + k;
                    const ql_l = blk.ql[ql_offset + l];
                    const ql_l32 = blk.ql[ql_offset + l + 32];
                    const qh_l = blk.qh[qh_offset + l];

                    q1_arr[k] = @floatFromInt(@as(i32, @intCast((ql_l & 0x0F) | (((qh_l >> 0) & 3) << 4))) - 32);
                    q2_arr[k] = @floatFromInt(@as(i32, @intCast((ql_l32 & 0x0F) | (((qh_l >> 2) & 3) << 4))) - 32);
                    q3_arr[k] = @floatFromInt(@as(i32, @intCast((ql_l >> 4) | (((qh_l >> 4) & 3) << 4))) - 32);
                    q4_arr[k] = @floatFromInt(@as(i32, @intCast((ql_l32 >> 4) | (((qh_l >> 6) & 3) << 4))) - 32);
                }

                const s1: Vec = @splat(@as(f32, @floatFromInt(blk.scales[sc_offset + is + 0])));
                const s2: Vec = @splat(@as(f32, @floatFromInt(blk.scales[sc_offset + is + 2])));
                const s3: Vec = @splat(@as(f32, @floatFromInt(blk.scales[sc_offset + is + 4])));
                const s4: Vec = @splat(@as(f32, @floatFromInt(blk.scales[sc_offset + is + 6])));

                const vb1: Vec = b_ptr[b_offset + l_start + 0 ..][0..16].*;
                const vb2: Vec = b_ptr[b_offset + l_start + 32 ..][0..16].*;
                const vb3: Vec = b_ptr[b_offset + l_start + 64 ..][0..16].*;
                const vb4: Vec = b_ptr[b_offset + l_start + 96 ..][0..16].*;

                const vq1: Vec = q1_arr;
                const vq2: Vec = q2_arr;
                const vq3: Vec = q3_arr;
                const vq4: Vec = q4_arr;

                total_sum += (vq1 * s1 * vb1 + vq2 * s2 * vb2 + vq3 * s3 * vb3 + vq4 * s4 * vb4) * vd;
            }
        }
    }

    return @reduce(.Add, total_sum);
}

pub fn dotQ8_KF32(a_bytes: []const u8, b: []const f32, n_elements: usize) f32 {
    const n_blocks = n_elements / 256;
    const block_size = @sizeOf(BlockQ8_K);
    var total: f32 = 0.0;

    for (0..n_blocks) |block_idx| {
        const blk: *const BlockQ8_K = @ptrCast(@alignCast(a_bytes[block_idx * block_size .. (block_idx + 1) * block_size].ptr));
        const d = blk.d;
        const b_ptr = b.ptr + block_idx * 256;
        var sum: f32 = 0.0;
        for (0..256) |i| {
            sum += @as(f32, @floatFromInt(blk.qs[i])) * b_ptr[i];
        }
        total += sum * d;
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

fn gemvRangeWorker(ctx_ptr: ?*anyopaque, start: usize, end: usize, _: usize) void {
    const ctx: *const GemvContext = @ptrCast(@alignCast(ctx_ptr.?));
    const row_bytes = ctx.row_bytes;
    const qtype = ctx.qtype;
    const weight_data = ctx.weight_data;
    const x = ctx.x;
    const y = ctx.y;
    const cols = ctx.cols;

    for (start..end) |r| {
        const row_start = r * row_bytes;
        const row_data = weight_data[row_start .. row_start + row_bytes];
        y[r] = dotRow(qtype, row_data, x, cols);
    }
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

    // For smaller matrices or projections, avoid thread dispatch overhead
    if (pool != null and rows >= 64 and (rows * cols) >= 32768) {
        pool.?.parallelForRange(rows, &ctx, gemvRangeWorker);
    } else {
        gemvRangeWorker(&ctx, 0, rows, 0);
    }
}

pub const GemmContext = struct {
    qtype: GGMLType,
    weight_data: []const u8,
    row_bytes: usize,
    X: []const f32,
    Y: []f32,
    batch_size: usize,
    rows: usize,
    cols: usize,
};

fn gemmRangeWorker(ctx_ptr: ?*anyopaque, start_row: usize, end_row: usize, _: usize) void {
    const ctx: *const GemmContext = @ptrCast(@alignCast(ctx_ptr.?));
    const qtype = ctx.qtype;
    const weight_data = ctx.weight_data;
    const row_bytes = ctx.row_bytes;
    const X = ctx.X;
    const Y = ctx.Y;
    const batch_size = ctx.batch_size;
    const rows = ctx.rows;
    const cols = ctx.cols;

    for (start_row..end_row) |r| {
        const row_start = r * row_bytes;
        const row_data = weight_data[row_start .. row_start + row_bytes];
        for (0..batch_size) |b| {
            const x_slice = X[b * cols .. (b + 1) * cols];
            Y[b * rows + r] = dotRow(qtype, row_data, x_slice, cols);
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
    if (batch_size == 0 or rows == 0 or cols == 0) return;
    if (batch_size == 1) {
        gemv(pool, qtype, weight_data, X, Y, rows, cols);
        return;
    }

    const blk_size = qtype.blockSize();
    const type_size = qtype.typeSize();
    const row_bytes = (cols / blk_size) * type_size;

    var ctx = GemmContext{
        .qtype = qtype,
        .weight_data = weight_data,
        .row_bytes = row_bytes,
        .X = X,
        .Y = Y,
        .batch_size = batch_size,
        .rows = rows,
        .cols = cols,
    };

    if (pool != null and rows >= 8 and (rows * cols * batch_size) >= 8192) {
        pool.?.parallelForRange(rows, &ctx, gemmRangeWorker);
    } else {
        gemmRangeWorker(&ctx, 0, rows, 0);
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
    const n_unroll = n / 32;

    var sum0: Vec = @splat(0.0);
    var sum1: Vec = @splat(0.0);
    var sum2: Vec = @splat(0.0);
    var sum3: Vec = @splat(0.0);
    var i: usize = 0;

    while (i < n_unroll * 32) : (i += 32) {
        const vx0: Vec = x[i..][0..8].*;
        const vx1: Vec = x[i + 8 ..][0..8].*;
        const vx2: Vec = x[i + 16 ..][0..8].*;
        const vx3: Vec = x[i + 24 ..][0..8].*;

        sum0 += vx0 * vx0;
        sum1 += vx1 * vx1;
        sum2 += vx2 * vx2;
        sum3 += vx3 * vx3;
    }

    while (i + vec_len <= n) : (i += vec_len) {
        const vx: Vec = x[i..][0..vec_len].*;
        sum0 += vx * vx;
    }

    sum_sq = @reduce(.Add, (sum0 + sum1) + (sum2 + sum3));

    while (i < n) : (i += 1) {
        sum_sq += x[i] * x[i];
    }

    const mean_sq = sum_sq / @as(f32, @floatFromInt(n));
    const inv_std = 1.0 / @sqrt(mean_sq + eps);

    i = 0;
    const inv_std_vec: Vec = @splat(inv_std);
    const one_vec: Vec = @splat(1.0);

    while (i < n_unroll * 32) : (i += 32) {
        const vx0: Vec = x[i..][0..8].*;
        const vx1: Vec = x[i + 8 ..][0..8].*;
        const vx2: Vec = x[i + 16 ..][0..8].*;
        const vx3: Vec = x[i + 24 ..][0..8].*;

        const vw0: Vec = weight[i..][0..8].*;
        const vw1: Vec = weight[i + 8 ..][0..8].*;
        const vw2: Vec = weight[i + 16 ..][0..8].*;
        const vw3: Vec = weight[i + 24 ..][0..8].*;

        if (use_unit_offset) {
            out[i..][0..8].* = (vx0 * inv_std_vec) * (one_vec + vw0);
            out[i + 8 ..][0..8].* = (vx1 * inv_std_vec) * (one_vec + vw1);
            out[i + 16 ..][0..8].* = (vx2 * inv_std_vec) * (one_vec + vw2);
            out[i + 24 ..][0..8].* = (vx3 * inv_std_vec) * (one_vec + vw3);
        } else {
            out[i..][0..8].* = (vx0 * inv_std_vec) * vw0;
            out[i + 8 ..][0..8].* = (vx1 * inv_std_vec) * vw1;
            out[i + 16 ..][0..8].* = (vx2 * inv_std_vec) * vw2;
            out[i + 24 ..][0..8].* = (vx3 * inv_std_vec) * vw3;
        }
    }

    while (i + vec_len <= n) : (i += vec_len) {
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

pub fn meanAbs(arr: []const f32) f32 {
    var sum: f32 = 0;
    for (arr) |v| sum += @abs(v);
    return sum / @as(f32, @floatFromInt(arr.len));
}
