const std = @import("std");

pub const Image = struct {
    width: usize,
    height: usize,
    channels: usize,
    data: []f32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Image) void {
        self.allocator.free(self.data);
    }

    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !Image {
        const fd = try std.posix.openat(std.posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY }, 0);
        defer _ = std.posix.system.close(fd);

        const end_pos = std.posix.system.lseek(fd, 0, 2);
        if (end_pos < 0) return error.SeekFailed;
        _ = std.posix.system.lseek(fd, 0, 0);
        const file_size: usize = @intCast(end_pos);

        const buf = try allocator.alloc(u8, file_size);
        defer allocator.free(buf);

        var total_read: usize = 0;
        while (total_read < file_size) {
            const n = try std.posix.read(fd, buf[total_read..]);
            if (n == 0) break;
            total_read += n;
        }

        if (buf.len >= 2 and buf[0] == 'B' and buf[1] == 'M') {
            return loadBMP(allocator, buf);
        } else if (buf.len >= 2 and buf[0] == 'P' and buf[1] == '6') {
            return loadPPM(allocator, buf);
        } else {
            return error.UnsupportedImageFormat;
        }
    }

    pub fn createSynthetic(allocator: std.mem.Allocator, width: usize, height: usize) !Image {
        const total_pixels = width * height * 3;
        const data = try allocator.alloc(f32, total_pixels);
        for (0..height) |y| {
            for (0..width) |x| {
                const idx = (y * width + x) * 3;
                data[idx + 0] = @as(f32, @floatFromInt(x)) / @as(f32, @floatFromInt(width));
                data[idx + 1] = @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(height));
                data[idx + 2] = 0.5;
            }
        }
        return Image{
            .width = width,
            .height = height,
            .channels = 3,
            .data = data,
            .allocator = allocator,
        };
    }

    fn loadBMP(allocator: std.mem.Allocator, bytes: []const u8) !Image {
        if (bytes.len < 54) return error.InvalidBMP;
        const offset = std.mem.readInt(u32, bytes[10..14], .little);
        const width = std.mem.readInt(i32, bytes[18..22], .little);
        const height = std.mem.readInt(i32, bytes[22..26], .little);
        const bpp = std.mem.readInt(u16, bytes[28..30], .little);

        if (width <= 0 or height <= 0) return error.InvalidBMPDimensions;
        const w: usize = @intCast(width);
        const h: usize = @intCast(height);

        const out_data = try allocator.alloc(f32, w * h * 3);
        errdefer allocator.free(out_data);

        const row_stride = (w * @as(usize, bpp / 8) + 3) & ~@as(usize, 3);

        for (0..h) |y| {
            const src_y = h - 1 - y;
            const src_row_start = offset + src_y * row_stride;
            if (src_row_start >= bytes.len) break;

            for (0..w) |x| {
                const dst_idx = (y * w + x) * 3;
                if (bpp == 24) {
                    const src_idx = src_row_start + x * 3;
                    if (src_idx + 2 < bytes.len) {
                        const b_val = bytes[src_idx];
                        const g_val = bytes[src_idx + 1];
                        const r_val = bytes[src_idx + 2];
                        out_data[dst_idx + 0] = @as(f32, @floatFromInt(r_val)) / 255.0;
                        out_data[dst_idx + 1] = @as(f32, @floatFromInt(g_val)) / 255.0;
                        out_data[dst_idx + 2] = @as(f32, @floatFromInt(b_val)) / 255.0;
                    }
                } else if (bpp == 32) {
                    const src_idx = src_row_start + x * 4;
                    if (src_idx + 3 < bytes.len) {
                        const b_val = bytes[src_idx];
                        const g_val = bytes[src_idx + 1];
                        const r_val = bytes[src_idx + 2];
                        out_data[dst_idx + 0] = @as(f32, @floatFromInt(r_val)) / 255.0;
                        out_data[dst_idx + 1] = @as(f32, @floatFromInt(g_val)) / 255.0;
                        out_data[dst_idx + 2] = @as(f32, @floatFromInt(b_val)) / 255.0;
                    }
                }
            }
        }

        return Image{
            .width = w,
            .height = h,
            .channels = 3,
            .data = out_data,
            .allocator = allocator,
        };
    }

    fn loadPPM(allocator: std.mem.Allocator, bytes: []const u8) !Image {
        var it = std.mem.tokenizeAny(u8, bytes, " \t\r\n");
        const magic = it.next() orelse return error.InvalidPPM;
        if (!std.mem.eql(u8, magic, "P6")) return error.InvalidPPM;

        const w_str = it.next() orelse return error.InvalidPPM;
        const h_str = it.next() orelse return error.InvalidPPM;
        const max_str = it.next() orelse return error.InvalidPPM;

        const w = try std.fmt.parseInt(usize, w_str, 10);
        const h = try std.fmt.parseInt(usize, h_str, 10);
        const max_val = try std.fmt.parseFloat(f32, max_str);

        const header_end = it.index;
        const pixel_bytes = bytes[header_end..];
        if (pixel_bytes.len < w * h * 3) return error.UnexpectedEOF;

        const out_data = try allocator.alloc(f32, w * h * 3);
        errdefer allocator.free(out_data);

        for (0..w * h * 3) |i| {
            out_data[i] = @as(f32, @floatFromInt(pixel_bytes[i])) / max_val;
        }

        return Image{
            .width = w,
            .height = h,
            .channels = 3,
            .data = out_data,
            .allocator = allocator,
        };
    }

    pub fn extractPatches(self: *const Image, allocator: std.mem.Allocator, patch_size: usize) ![]f32 {
        const patch_dim = patch_size * patch_size * 3;
        const patches_x = self.width / patch_size;
        const patches_y = self.height / patch_size;
        const num_patches = patches_x * patches_y;

        if (num_patches == 0) return error.ImageSmallerThanPatch;

        const out = try allocator.alloc(f32, num_patches * patch_dim);
        errdefer allocator.free(out);

        var p_idx: usize = 0;
        for (0..patches_y) |py| {
            for (0..patches_x) |px| {
                const patch_out = out[p_idx * patch_dim .. (p_idx + 1) * patch_dim];
                var k: usize = 0;
                for (0..patch_size) |dy| {
                    for (0..patch_size) |dx| {
                        const img_x = px * patch_size + dx;
                        const img_y = py * patch_size + dy;
                        const src_idx = (img_y * self.width + img_x) * 3;
                        patch_out[k + 0] = self.data[src_idx + 0];
                        patch_out[k + 1] = self.data[src_idx + 1];
                        patch_out[k + 2] = self.data[src_idx + 2];
                        k += 3;
                    }
                }
                p_idx += 1;
            }
        }

        return out;
    }
};

test "Image synthetic create and extract patches" {
    const allocator = std.testing.allocator;
    var img = try Image.createSynthetic(allocator, 32, 32);
    defer img.deinit();

    const patches = try img.extractPatches(allocator, 16);
    defer allocator.free(patches);

    try std.testing.expectEqual(4 * 16 * 16 * 3, patches.len);
}
