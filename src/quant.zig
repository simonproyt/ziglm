const std = @import("std");
const types = @import("types.zig");
const GGMLType = types.GGMLType;
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

// ============================================================================
// Float Conversions
// ============================================================================

pub inline fn f16ToF32(val: f16) f32 {
    return @floatCast(val);
}

pub inline fn f32ToF16(val: f32) f16 {
    return @floatCast(val);
}

pub inline fn bf16ToF32(u: u16) f32 {
    const bits: u32 = @as(u32, u) << 16;
    return @bitCast(bits);
}

pub inline fn f32ToBf16(f: f32) u16 {
    const bits: u32 = @bitCast(f);
    const rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    const rounded = bits + rounding_bias;
    return @as(u16, @truncate(rounded >> 16));
}

// Non-linear lookup table for IQ4_NL
pub const kvalues_iq4nl = [16]f32{
    -127.0 / 127.0, -104.0 / 127.0, -83.0 / 127.0, -65.0 / 127.0,
    -49.0 / 127.0,  -35.0 / 127.0,  -22.0 / 127.0, -10.0 / 127.0,
    1.0 / 127.0,    13.0 / 127.0,   25.0 / 127.0,  38.0 / 127.0,
    53.0 / 127.0,   69.0 / 127.0,   89.0 / 127.0,  113.0 / 127.0,
};

// ============================================================================
// Q8_0 & Q8_1
// ============================================================================

pub fn quantizeQ8_0(src: []const f32, dst: *BlockQ8_0) void {
    std.debug.assert(src.len >= 32);
    var amax: f32 = 0.0;
    for (src[0..32]) |v| {
        const a = @abs(v);
        if (a > amax) amax = a;
    }

    const d = amax / 127.0;
    dst.d = @floatCast(d);
    const id = if (d != 0.0) 1.0 / d else 0.0;

    for (0..32) |i| {
        const x = src[i] * id;
        const v = @round(x);
        dst.qs[i] = @intFromFloat(std.math.clamp(v, -128.0, 127.0));
    }
}

pub fn dequantizeQ8_0(block: *const BlockQ8_0, dst: []f32) void {
    std.debug.assert(dst.len >= 32);
    const d: f32 = @floatCast(block.d);
    for (0..32) |i| {
        dst[i] = @as(f32, @floatFromInt(block.qs[i])) * d;
    }
}

pub fn dequantizeQ8_1(block: *const BlockQ8_1, dst: []f32) void {
    std.debug.assert(dst.len >= 32);
    const d: f32 = @floatCast(block.d);
    for (0..32) |i| {
        dst[i] = @as(f32, @floatFromInt(block.qs[i])) * d;
    }
}

// ============================================================================
// Q4_0 & Q4_1
// ============================================================================

pub fn quantizeQ4_0(src: []const f32, dst: *BlockQ4_0) void {
    std.debug.assert(src.len >= 32);
    var amax: f32 = 0.0;
    var max: f32 = 0.0;

    for (src[0..32]) |v| {
        const a = @abs(v);
        if (a > amax) {
            amax = a;
            max = v;
        }
    }

    const d = -max / 8.0;
    dst.d = @floatCast(d);
    const id = if (d != 0.0) 1.0 / d else 0.0;

    for (0..16) |i| {
        const x0 = src[i] * id;
        const x1 = src[i + 16] * id;
        const v0: u8 = @intFromFloat(std.math.clamp(@round(x0) + 8.0, 0.0, 15.0));
        const v1: u8 = @intFromFloat(std.math.clamp(@round(x1) + 8.0, 0.0, 15.0));
        dst.qs[i] = (v0 & 0x0F) | ((v1 & 0x0F) << 4);
    }
}

pub fn dequantizeQ4_0(block: *const BlockQ4_0, dst: []f32) void {
    std.debug.assert(dst.len >= 32);
    const d: f32 = @floatCast(block.d);

    for (0..16) |i| {
        const qs = block.qs[i];
        const v0: i32 = @as(i32, @intCast(qs & 0x0F)) - 8;
        const v1: i32 = @as(i32, @intCast((qs >> 4) & 0x0F)) - 8;
        dst[i] = @as(f32, @floatFromInt(v0)) * d;
        dst[i + 16] = @as(f32, @floatFromInt(v1)) * d;
    }
}

pub fn dequantizeQ4_1(block: *const BlockQ4_1, dst: []f32) void {
    std.debug.assert(dst.len >= 32);
    const d: f32 = @floatCast(block.d);
    const m: f32 = @floatCast(block.m);

    for (0..16) |i| {
        const qs = block.qs[i];
        const v0: f32 = @floatFromInt(qs & 0x0F);
        const v1: f32 = @floatFromInt((qs >> 4) & 0x0F);
        dst[i] = v0 * d + m;
        dst[i + 16] = v1 * d + m;
    }
}

// ============================================================================
// Q5_0 & Q5_1
// ============================================================================

pub fn dequantizeQ5_0(block: *const BlockQ5_0, dst: []f32) void {
    std.debug.assert(dst.len >= 32);
    const d: f32 = @floatCast(block.d);
    const qh_val = std.mem.readInt(u32, &block.qh, .little);

    for (0..16) |i| {
        const qs = block.qs[i];
        const h0: u5 = @intCast(((qh_val >> @intCast(i)) & 1) << 4);
        const h1: u5 = @intCast(((qh_val >> @intCast(i + 16)) & 1) << 4);
        const v0: i32 = @as(i32, @intCast((qs & 0x0F) | h0)) - 16;
        const v1: i32 = @as(i32, @intCast(((qs >> 4) & 0x0F) | h1)) - 16;
        dst[i] = @as(f32, @floatFromInt(v0)) * d;
        dst[i + 16] = @as(f32, @floatFromInt(v1)) * d;
    }
}

pub fn dequantizeQ5_1(block: *const BlockQ5_1, dst: []f32) void {
    std.debug.assert(dst.len >= 32);
    const d: f32 = @floatCast(block.d);
    const m: f32 = @floatCast(block.m);
    const qh_val = std.mem.readInt(u32, &block.qh, .little);

    for (0..16) |i| {
        const qs = block.qs[i];
        const h0: u5 = @intCast(((qh_val >> @intCast(i)) & 1) << 4);
        const h1: u5 = @intCast(((qh_val >> @intCast(i + 16)) & 1) << 4);
        const v0: u32 = (qs & 0x0F) | h0;
        const v1: u32 = ((qs >> 4) & 0x0F) | h1;
        dst[i] = @as(f32, @floatFromInt(v0)) * d + m;
        dst[i + 16] = @as(f32, @floatFromInt(v1)) * d + m;
    }
}

// ============================================================================
// Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
// ============================================================================

pub fn dequantizeQ2_K(block: *const BlockQ2_K, dst: []f32) void {
    std.debug.assert(dst.len >= 256);
    const d: f32 = @floatCast(block.d);
    const min: f32 = @floatCast(block.dmin);

    var y_idx: usize = 0;
    var q_offset: usize = 0;

    for (0..2) |n| {
        var shift: usize = 0;
        for (0..4) |j| {
            const sc = block.scales[n * 8 + j];
            const dl = d * @as(f32, @floatFromInt(sc & 0x0F));
            const ml = min * @as(f32, @floatFromInt(sc >> 4));

            const sc1 = block.scales[n * 8 + j + 4];
            const dl1 = d * @as(f32, @floatFromInt(sc1 & 0x0F));
            const ml1 = min * @as(f32, @floatFromInt(sc1 >> 4));

            const shift_u3: u3 = @intCast(shift);
            for (0..16) |l| {
                const q_val = (block.qs[q_offset + l] >> shift_u3) & 3;
                dst[y_idx] = dl * @as(f32, @floatFromInt(q_val)) - ml;
                y_idx += 1;
            }
            for (0..16) |l| {
                const q_val = (block.qs[q_offset + 16 + l] >> shift_u3) & 3;
                dst[y_idx] = dl1 * @as(f32, @floatFromInt(q_val)) - ml1;
                y_idx += 1;
            }
            shift += 2;
        }
        q_offset += 32;
    }
}

pub fn dequantizeQ3_K(block: *const BlockQ3_K, dst: []f32) void {
    std.debug.assert(dst.len >= 256);
    const d: f32 = @floatCast(block.d);

    var scales: [16]u8 = undefined;
    for (0..4) |j| {
        scales[j] = (block.scales[j] & 0x0F) | ((block.scales[j + 8] & 0x03) << 4);
        scales[j + 4] = (block.scales[j + 4] & 0x0F) | (((block.scales[j + 8] >> 2) & 0x03) << 4);
        scales[j + 8] = ((block.scales[j] >> 4) & 0x0F) | (((block.scales[j + 8] >> 4) & 0x03) << 4);
        scales[j + 12] = ((block.scales[j + 4] >> 4) & 0x0F) | (((block.scales[j + 8] >> 6) & 0x03) << 4);
    }

    var y_idx: usize = 0;
    var q_offset: usize = 0;
    var is: usize = 0;

    for (0..2) |n| {
        var shift: usize = 0;
        for (0..4) |j| {
            const sc0: i32 = @as(i32, @intCast(scales[is])) - 32;
            is += 1;
            const dl0 = d * @as(f32, @floatFromInt(sc0));

            const bit_mask: u8 = @as(u8, 1) << @intCast(2 * (j % 2));
            const hm_shift: u3 = @intCast(2 * n + (j / 2));
            const shift_u3: u3 = @intCast(shift);

            for (0..16) |l| {
                const q_bits = (block.qs[q_offset + l] >> shift_u3) & 3;
                const hm_bit = (block.hmask[l] >> hm_shift) & bit_mask;
                const q_val: i32 = @as(i32, @intCast(q_bits)) - if (hm_bit == 0) @as(i32, 4) else @as(i32, 0);
                dst[y_idx] = dl0 * @as(f32, @floatFromInt(q_val));
                y_idx += 1;
            }

            const sc1: i32 = @as(i32, @intCast(scales[is])) - 32;
            is += 1;
            const dl1 = d * @as(f32, @floatFromInt(sc1));

            for (0..16) |l| {
                const q_bits = (block.qs[q_offset + 16 + l] >> shift_u3) & 3;
                const hm_bit = (block.hmask[l + 16] >> hm_shift) & bit_mask;
                const q_val: i32 = @as(i32, @intCast(q_bits)) - if (hm_bit == 0) @as(i32, 4) else @as(i32, 0);
                dst[y_idx] = dl1 * @as(f32, @floatFromInt(q_val));
                y_idx += 1;
            }

            shift += 2;
        }
        q_offset += 32;
    }
}

pub fn dequantizeQ4_K(block: *const BlockQ4_K, dst: []f32) void {
    std.debug.assert(dst.len >= 256);
    const d: f32 = @floatCast(block.d);
    const min: f32 = @floatCast(block.dmin);

    var scales: [8]u8 = undefined;
    var mins: [8]u8 = undefined;

    for (0..4) |j| {
        scales[j] = block.scales[j] & 63;
        mins[j] = block.scales[j + 4] & 63;
        scales[j + 4] = (block.scales[j + 8] & 0x0F) | ((block.scales[j] >> 6) << 4);
        mins[j + 4] = (block.scales[j + 8] >> 4) | ((block.scales[j + 4] >> 6) << 4);
    }

    var qs_idx: usize = 0;
    var dst_idx: usize = 0;

    for (0..4) |j| {
        const d0 = d * @as(f32, @floatFromInt(scales[2 * j]));
        const m0 = min * @as(f32, @floatFromInt(mins[2 * j]));
        const d1 = d * @as(f32, @floatFromInt(scales[2 * j + 1]));
        const m1 = min * @as(f32, @floatFromInt(mins[2 * j + 1]));

        for (0..32) |l| {
            const q = block.qs[qs_idx + l];
            dst[dst_idx + l] = @as(f32, @floatFromInt(q & 0x0F)) * d0 - m0;
            dst[dst_idx + 32 + l] = @as(f32, @floatFromInt((q >> 4) & 0x0F)) * d1 - m1;
        }

        qs_idx += 32;
        dst_idx += 64;
    }
}

pub fn dequantizeQ5_K(block: *const BlockQ5_K, dst: []f32) void {
    std.debug.assert(dst.len >= 256);
    const d: f32 = @floatCast(block.d);
    const min: f32 = @floatCast(block.dmin);

    var scales: [8]u8 = undefined;
    var mins: [8]u8 = undefined;

    for (0..4) |j| {
        scales[j] = block.scales[j] & 63;
        mins[j] = block.scales[j + 4] & 63;
        scales[j + 4] = (block.scales[j + 8] & 0x0F) | ((block.scales[j] >> 6) << 4);
        mins[j + 4] = (block.scales[j + 8] >> 4) | ((block.scales[j + 4] >> 6) << 4);
    }

    var ql_idx: usize = 0;
    var dst_idx: usize = 0;
    var mask1: u8 = 1;
    var mask2: u8 = 2;

    for (0..4) |j| {
        const d0 = d * @as(f32, @floatFromInt(scales[2 * j]));
        const m0 = min * @as(f32, @floatFromInt(mins[2 * j]));
        const d1 = d * @as(f32, @floatFromInt(scales[2 * j + 1]));
        const m1 = min * @as(f32, @floatFromInt(mins[2 * j + 1]));

        for (0..32) |l| {
            const qh_val = block.qh[l];
            const h0: f32 = if ((qh_val & mask1) != 0) 16.0 else 0.0;
            const h1: f32 = if ((qh_val & mask2) != 0) 16.0 else 0.0;

            const ql_byte = block.qs[ql_idx + l];
            const v0 = @as(f32, @floatFromInt(ql_byte & 0x0F)) + h0;
            const v1 = @as(f32, @floatFromInt(ql_byte >> 4)) + h1;

            dst[dst_idx + l] = d0 * v0 - m0;
            dst[dst_idx + 32 + l] = d1 * v1 - m1;
        }

        ql_idx += 32;
        dst_idx += 64;
        mask1 <<= 2;
        mask2 <<= 2;
    }
}

pub fn dequantizeQ6_K(block: *const BlockQ6_K, dst: []f32) void {
    std.debug.assert(dst.len >= 256);
    const d: f32 = @floatCast(block.d);

    for (0..2) |n| {
        const ql_offset = n * 64;
        const qh_offset = n * 32;
        const sc_offset = n * 8;
        const dst_offset = n * 128;

        for (0..32) |l| {
            const is = l / 16;
            const ql_l = block.ql[ql_offset + l];
            const ql_l32 = block.ql[ql_offset + l + 32];
            const qh_l = block.qh[qh_offset + l];

            const q1: i32 = @as(i32, @intCast((ql_l & 0x0F) | (((qh_l >> 0) & 3) << 4))) - 32;
            const q2: i32 = @as(i32, @intCast((ql_l32 & 0x0F) | (((qh_l >> 2) & 3) << 4))) - 32;
            const q3: i32 = @as(i32, @intCast((ql_l >> 4) | (((qh_l >> 4) & 3) << 4))) - 32;
            const q4: i32 = @as(i32, @intCast((ql_l32 >> 4) | (((qh_l >> 6) & 3) << 4))) - 32;

            dst[dst_offset + l + 0] = d * @as(f32, @floatFromInt(block.scales[sc_offset + is + 0])) * @as(f32, @floatFromInt(q1));
            dst[dst_offset + l + 32] = d * @as(f32, @floatFromInt(block.scales[sc_offset + is + 2])) * @as(f32, @floatFromInt(q2));
            dst[dst_offset + l + 64] = d * @as(f32, @floatFromInt(block.scales[sc_offset + is + 4])) * @as(f32, @floatFromInt(q3));
            dst[dst_offset + l + 96] = d * @as(f32, @floatFromInt(block.scales[sc_offset + is + 6])) * @as(f32, @floatFromInt(q4));
        }
    }
}

pub fn dequantizeQ8_K(block: *const BlockQ8_K, dst: []f32) void {
    std.debug.assert(dst.len >= 256);
    const d = block.d;
    for (0..256) |i| {
        dst[i] = @as(f32, @floatFromInt(block.qs[i])) * d;
    }
}

// ============================================================================
// IQ4_NL (Importance Matrix Non-Linear)
// ============================================================================

pub fn dequantizeIQ4_NL(block: *const BlockIQ4_NL, dst: []f32) void {
    std.debug.assert(dst.len >= 32);
    const d: f32 = @floatCast(block.d);
    for (0..16) |i| {
        const q0 = block.qs[i] & 0x0F;
        const q1 = (block.qs[i] >> 4) & 0x0F;
        dst[i] = d * kvalues_iq4nl[q0];
        dst[i + 16] = d * kvalues_iq4nl[q1];
    }
}

// ============================================================================
// Generic Row Dequantization & Quantization
// ============================================================================

pub fn dequantizeRow(qtype: GGMLType, src_bytes: []const u8, dst: []f32, n_elements: usize) void {
    std.debug.assert(dst.len >= n_elements);

    switch (qtype) {
        .F32 => {
            const bytes = src_bytes[0 .. n_elements * @sizeOf(f32)];
            if (@intFromPtr(bytes.ptr) % @alignOf(f32) == 0) {
                const src_f32: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, bytes));
                @memcpy(dst[0..n_elements], src_f32);
            } else {
                for (0..n_elements) |i| {
                    const u = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
                    dst[i] = @bitCast(u);
                }
            }
        },
        .F16 => {
            const bytes = src_bytes[0 .. n_elements * @sizeOf(f16)];
            if (@intFromPtr(bytes.ptr) % @alignOf(f16) == 0) {
                const src_f16: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, bytes));
                for (0..n_elements) |i| {
                    dst[i] = f16ToF32(src_f16[i]);
                }
            } else {
                for (0..n_elements) |i| {
                    const u = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
                    const f: f16 = @bitCast(u);
                    dst[i] = f16ToF32(f);
                }
            }
        },
        .BF16 => {
            const bytes = src_bytes[0 .. n_elements * @sizeOf(u16)];
            for (0..n_elements) |i| {
                const u = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
                dst[i] = bf16ToF32(u);
            }
        },
        .F64 => {
            const bytes = src_bytes[0 .. n_elements * @sizeOf(f64)];
            for (0..n_elements) |i| {
                const u = std.mem.readInt(u64, bytes[i * 8 ..][0..8], .little);
                const f: f64 = @bitCast(u);
                dst[i] = @floatCast(f);
            }
        },
        .I8 => {
            const src_i8: []const i8 = @alignCast(std.mem.bytesAsSlice(i8, src_bytes[0 .. n_elements * @sizeOf(i8)]));
            for (0..n_elements) |i| {
                dst[i] = @as(f32, @floatFromInt(src_i8[i]));
            }
        },
        .I16 => {
            const bytes = src_bytes[0 .. n_elements * @sizeOf(i16)];
            for (0..n_elements) |i| {
                const val = std.mem.readInt(i16, bytes[i * 2 ..][0..2], .little);
                dst[i] = @as(f32, @floatFromInt(val));
            }
        },
        .I32 => {
            const bytes = src_bytes[0 .. n_elements * @sizeOf(i32)];
            for (0..n_elements) |i| {
                const val = std.mem.readInt(i32, bytes[i * 4 ..][0..4], .little);
                dst[i] = @as(f32, @floatFromInt(val));
            }
        },
        .I64 => {
            const bytes = src_bytes[0 .. n_elements * @sizeOf(i64)];
            for (0..n_elements) |i| {
                const val = std.mem.readInt(i64, bytes[i * 8 ..][0..8], .little);
                dst[i] = @as(f32, @floatFromInt(val));
            }
        },
        .Q8_0 => {
            const n_blocks = n_elements / 32;
            const block_size = @sizeOf(BlockQ8_0);
            for (0..n_blocks) |b| {
                const blk_bytes = src_bytes[b * block_size .. (b + 1) * block_size];
                const block: *const BlockQ8_0 = @ptrCast(@alignCast(blk_bytes.ptr));
                dequantizeQ8_0(block, dst[b * 32 .. (b + 1) * 32]);
            }
        },
        .Q8_1 => {
            const n_blocks = n_elements / 32;
            const block_size = @sizeOf(BlockQ8_1);
            for (0..n_blocks) |b| {
                const blk_bytes = src_bytes[b * block_size .. (b + 1) * block_size];
                const block: *const BlockQ8_1 = @ptrCast(@alignCast(blk_bytes.ptr));
                dequantizeQ8_1(block, dst[b * 32 .. (b + 1) * 32]);
            }
        },
        .Q4_0 => {
            const n_blocks = n_elements / 32;
            const block_size = @sizeOf(BlockQ4_0);
            for (0..n_blocks) |b| {
                const blk_bytes = src_bytes[b * block_size .. (b + 1) * block_size];
                const block: *const BlockQ4_0 = @ptrCast(@alignCast(blk_bytes.ptr));
                dequantizeQ4_0(block, dst[b * 32 .. (b + 1) * 32]);
            }
        },
        .Q4_1 => {
            const n_blocks = n_elements / 32;
            const block_size = @sizeOf(BlockQ4_1);
            for (0..n_blocks) |b| {
                const blk_bytes = src_bytes[b * block_size .. (b + 1) * block_size];
                const block: *const BlockQ4_1 = @ptrCast(@alignCast(blk_bytes.ptr));
                dequantizeQ4_1(block, dst[b * 32 .. (b + 1) * 32]);
            }
        },
        .Q5_0 => {
            const n_blocks = n_elements / 32;
            const block_size = @sizeOf(BlockQ5_0);
            for (0..n_blocks) |b| {
                const blk_bytes = src_bytes[b * block_size .. (b + 1) * block_size];
                const block: *const BlockQ5_0 = @ptrCast(@alignCast(blk_bytes.ptr));
                dequantizeQ5_0(block, dst[b * 32 .. (b + 1) * 32]);
            }
        },
        .Q5_1 => {
            const n_blocks = n_elements / 32;
            const block_size = @sizeOf(BlockQ5_1);
            for (0..n_blocks) |b| {
                const blk_bytes = src_bytes[b * block_size .. (b + 1) * block_size];
                const block: *const BlockQ5_1 = @ptrCast(@alignCast(blk_bytes.ptr));
                dequantizeQ5_1(block, dst[b * 32 .. (b + 1) * 32]);
            }
        },
        .Q2_K => {
            const n_blocks = n_elements / 256;
            const block_size = @sizeOf(BlockQ2_K);
            for (0..n_blocks) |b| {
                const blk_bytes = src_bytes[b * block_size .. (b + 1) * block_size];
                const block: *const BlockQ2_K = @ptrCast(@alignCast(blk_bytes.ptr));
                dequantizeQ2_K(block, dst[b * 256 .. (b + 1) * 256]);
            }
        },
        .Q3_K => {
            const n_blocks = n_elements / 256;
            const block_size = @sizeOf(BlockQ3_K);
            for (0..n_blocks) |b| {
                const blk_bytes = src_bytes[b * block_size .. (b + 1) * block_size];
                const block: *const BlockQ3_K = @ptrCast(@alignCast(blk_bytes.ptr));
                dequantizeQ3_K(block, dst[b * 256 .. (b + 1) * 256]);
            }
        },
        .Q4_K => {
            const n_blocks = n_elements / 256;
            const block_size = @sizeOf(BlockQ4_K);
            for (0..n_blocks) |b| {
                const blk_bytes = src_bytes[b * block_size .. (b + 1) * block_size];
                const block: *const BlockQ4_K = @ptrCast(@alignCast(blk_bytes.ptr));
                dequantizeQ4_K(block, dst[b * 256 .. (b + 1) * 256]);
            }
        },
        .Q5_K => {
            const n_blocks = n_elements / 256;
            const block_size = @sizeOf(BlockQ5_K);
            for (0..n_blocks) |b| {
                const blk_bytes = src_bytes[b * block_size .. (b + 1) * block_size];
                const block: *const BlockQ5_K = @ptrCast(@alignCast(blk_bytes.ptr));
                dequantizeQ5_K(block, dst[b * 256 .. (b + 1) * 256]);
            }
        },
        .Q6_K => {
            const n_blocks = n_elements / 256;
            const block_size = @sizeOf(BlockQ6_K);
            for (0..n_blocks) |b| {
                const blk_bytes = src_bytes[b * block_size .. (b + 1) * block_size];
                const block: *const BlockQ6_K = @ptrCast(@alignCast(blk_bytes.ptr));
                dequantizeQ6_K(block, dst[b * 256 .. (b + 1) * 256]);
            }
        },
        .Q8_K => {
            const n_blocks = n_elements / 256;
            const block_size = @sizeOf(BlockQ8_K);
            for (0..n_blocks) |b| {
                const blk_bytes = src_bytes[b * block_size .. (b + 1) * block_size];
                const block: *const BlockQ8_K = @ptrCast(@alignCast(blk_bytes.ptr));
                dequantizeQ8_K(block, dst[b * 256 .. (b + 1) * 256]);
            }
        },
        .IQ4_NL => {
            const n_blocks = n_elements / 32;
            const block_size = @sizeOf(BlockIQ4_NL);
            for (0..n_blocks) |b| {
                const blk_bytes = src_bytes[b * block_size .. (b + 1) * block_size];
                const block: *const BlockIQ4_NL = @ptrCast(@alignCast(blk_bytes.ptr));
                dequantizeIQ4_NL(block, dst[b * 32 .. (b + 1) * 32]);
            }
        },
        else => {
            @memset(dst[0..n_elements], 0.0);
        },
    }
}

pub fn quantizeRow(qtype: GGMLType, src: []const f32, dst_bytes: []u8, n_elements: usize) void {
    switch (qtype) {
        .F32 => {
            const dst_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, dst_bytes[0 .. n_elements * @sizeOf(f32)]));
            @memcpy(dst_f32, src[0..n_elements]);
        },
        .F16 => {
            const dst_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, dst_bytes[0 .. n_elements * @sizeOf(f16)]));
            for (0..n_elements) |i| {
                dst_f16[i] = f32ToF16(src[i]);
            }
        },
        .BF16 => {
            const dst_bf16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, dst_bytes[0 .. n_elements * @sizeOf(u16)]));
            for (0..n_elements) |i| {
                dst_bf16[i] = f32ToBf16(src[i]);
            }
        },
        .Q8_0 => {
            const n_blocks = n_elements / 32;
            const block_size = @sizeOf(BlockQ8_0);
            for (0..n_blocks) |b| {
                const blk_bytes = dst_bytes[b * block_size .. (b + 1) * block_size];
                const block: *BlockQ8_0 = @ptrCast(@alignCast(blk_bytes.ptr));
                quantizeQ8_0(src[b * 32 .. (b + 1) * 32], block);
            }
        },
        .Q4_0 => {
            const n_blocks = n_elements / 32;
            const block_size = @sizeOf(BlockQ4_0);
            for (0..n_blocks) |b| {
                const blk_bytes = dst_bytes[b * block_size .. (b + 1) * block_size];
                const block: *BlockQ4_0 = @ptrCast(@alignCast(blk_bytes.ptr));
                quantizeQ4_0(src[b * 32 .. (b + 1) * 32], block);
            }
        },
        else => {
            @memset(dst_bytes, 0);
        },
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

test "bfloat16 conversions" {
    const original: f32 = 3.14159265;
    const bf = f32ToBf16(original);
    const converted = bf16ToF32(bf);
    try std.testing.expectApproxEqRel(original, converted, 0.01);
}

test "Q8_0 quantize and dequantize" {
    var original: [32]f32 = undefined;
    for (0..32) |i| {
        original[i] = @as(f32, @floatFromInt(i)) * 0.1 - 1.5;
    }

    var block: BlockQ8_0 = undefined;
    quantizeQ8_0(&original, &block);

    var recovered: [32]f32 = undefined;
    dequantizeQ8_0(&block, &recovered);

    for (0..32) |i| {
        try std.testing.expectApproxEqAbs(original[i], recovered[i], 0.05);
    }
}

test "Q4_0 quantize and dequantize" {
    var original: [32]f32 = undefined;
    for (0..32) |i| {
        original[i] = (@as(f32, @floatFromInt(i)) - 16.0) * 0.2;
    }

    var block: BlockQ4_0 = undefined;
    quantizeQ4_0(&original, &block);

    var recovered: [32]f32 = undefined;
    dequantizeQ4_0(&block, &recovered);

    for (0..32) |i| {
        try std.testing.expectApproxEqAbs(original[i], recovered[i], 0.5);
    }
}

test "Q5_0 dequantize" {
    var block: BlockQ5_0 = undefined;
    block.d = @floatCast(@as(f32, 0.5));
    block.qh = [4]u8{ 0x55, 0xAA, 0x55, 0xAA };
    @memset(&block.qs, 0x12);

    var recovered: [32]f32 = undefined;
    dequantizeQ5_0(&block, &recovered);
    try std.testing.expect(recovered[0] != 0.0);
}

test "Q5_K dequantize" {
    var block: BlockQ5_K = undefined;
    block.d = @floatCast(@as(f32, 0.02));
    block.dmin = @floatCast(@as(f32, 0.01));
    @memset(&block.scales, 10);
    @memset(&block.qh, 0);
    @memset(&block.qs, 0x22);

    var recovered: [256]f32 = undefined;
    dequantizeQ5_K(&block, &recovered);
    try std.testing.expect(recovered[0] != 0.0);
}

test "Q2_K dequantize" {
    var block: BlockQ2_K = undefined;
    block.d = @floatCast(@as(f32, 0.02));
    block.dmin = @floatCast(@as(f32, 0.01));
    @memset(&block.scales, 0x24);
    @memset(&block.qs, 0x55);

    var recovered: [256]f32 = undefined;
    dequantizeQ2_K(&block, &recovered);
    try std.testing.expect(recovered[0] != 0.0);
}

test "Q3_K dequantize" {
    var block: BlockQ3_K = undefined;
    block.d = @floatCast(@as(f32, 0.02));
    @memset(&block.scales, 0x24);
    @memset(&block.hmask, 0x55);
    @memset(&block.qs, 0x55);

    var recovered: [256]f32 = undefined;
    dequantizeQ3_K(&block, &recovered);
    try std.testing.expect(recovered[0] != 0.0);
}

test "IQ4_NL dequantize" {
    var block: BlockIQ4_NL = undefined;
    block.d = @floatCast(@as(f32, 1.0));
    @memset(&block.qs, 0x08);

    var recovered: [32]f32 = undefined;
    dequantizeIQ4_NL(&block, &recovered);
    try std.testing.expect(recovered[0] != 0.0);
}
