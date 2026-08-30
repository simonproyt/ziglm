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
        // Expand ~ to HOME if path starts with ~/
        var resolved_path_buf: [1024]u8 = undefined;
        var actual_path = path;
        if (std.mem.startsWith(u8, path, "~/")) {
            const env_fd = std.posix.openat(std.posix.AT.FDCWD, "/proc/self/environ", .{ .ACCMODE = .RDONLY }, 0) catch null;
            if (env_fd) |efd| {
                defer _ = std.posix.system.close(efd);
                var env_buf: [4096]u8 = undefined;
                const env_len = std.posix.read(efd, &env_buf) catch 0;
                var iter = std.mem.splitScalar(u8, env_buf[0..env_len], 0);
                while (iter.next()) |var_str| {
                    if (std.mem.startsWith(u8, var_str, "HOME=")) {
                        const home = var_str["HOME=".len..];
                        actual_path = try std.fmt.bufPrint(&resolved_path_buf, "{s}/{s}", .{ home, path[2..] });
                        break;
                    }
                }
            }
        }

        // Standardized on ImageMagick (magick / convert / ffmpeg) pipeline for all image formats
        return loadViaExternalConverter(allocator, actual_path);
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

    pub fn loadPPMWithConsumedBytes(allocator: std.mem.Allocator, bytes: []const u8) !struct { image: Image, bytes_consumed: usize } {
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
        const is_16bit = max_val > 255.0;
        const sample_bytes: usize = if (is_16bit) 2 else 1;
        const pixel_len = w * h * 3 * sample_bytes;
        if (pixel_bytes.len < pixel_len) return error.UnexpectedEOF;

        const out_data = try allocator.alloc(f32, w * h * 3);
        errdefer allocator.free(out_data);

        if (is_16bit) {
            for (0..w * h * 3) |i| {
                const sample = std.mem.readInt(u16, pixel_bytes[i * 2 ..][0..2], .big);
                out_data[i] = @as(f32, @floatFromInt(sample)) / max_val;
            }
        } else {
            for (0..w * h * 3) |i| {
                out_data[i] = @as(f32, @floatFromInt(pixel_bytes[i])) / max_val;
            }
        }

        return .{
            .image = Image{
                .width = w,
                .height = h,
                .channels = 3,
                .data = out_data,
                .allocator = allocator,
            },
            .bytes_consumed = header_end + pixel_len,
        };
    }

    pub fn loadPPM(allocator: std.mem.Allocator, bytes: []const u8) !Image {
        const res = try loadPPMWithConsumedBytes(allocator, bytes);
        return res.image;
    }

    fn loadViaExternalConverter(allocator: std.mem.Allocator, actual_path: []const u8) !Image {
        var pipe_fds: [2]std.posix.fd_t = undefined;
        if (std.posix.system.pipe(&pipe_fds) != 0) return error.PipeFailed;
        const read_fd = pipe_fds[0];
        const write_fd = pipe_fds[1];

        const pid = std.posix.system.fork();
        if (pid < 0) {
            _ = std.posix.system.close(read_fd);
            _ = std.posix.system.close(write_fd);
            return error.ForkFailed;
        }

        if (pid == 0) {
            // Child process
            _ = std.posix.system.close(read_fd);
            _ = std.posix.system.dup2(write_fd, std.posix.STDOUT_FILENO);
            _ = std.posix.system.close(write_fd);

            // Redirect stderr to /dev/null
            const dev_null_fd = std.posix.openat(std.posix.AT.FDCWD, "/dev/null", .{ .ACCMODE = .WRONLY }, 0) catch -1;
            if (dev_null_fd >= 0) {
                _ = std.posix.system.dup2(dev_null_fd, std.posix.STDERR_FILENO);
                _ = std.posix.system.close(dev_null_fd);
            }

            var path_z_buf: [1024]u8 = undefined;
            const path_z = std.fmt.bufPrintZ(&path_z_buf, "{s}", .{actual_path}) catch std.posix.system.exit(1);

            const argv_magick = [_:null]?[*:0]const u8{
                "magick",
                path_z.ptr,
                "-depth",
                "8",
                "ppm:-",
                null,
            };

            const argv_convert = [_:null]?[*:0]const u8{
                "convert",
                path_z.ptr,
                "-depth",
                "8",
                "ppm:-",
                null,
            };

            const argv_ffmpeg = [_:null]?[*:0]const u8{
                "ffmpeg",
                "-y",
                "-i",
                path_z.ptr,
                "-f",
                "image2pipe",
                "-vcodec",
                "ppm",
                "-",
                null,
            };

            const envp = [_:null]?[*:0]const u8{ "PATH=/usr/bin:/bin:/usr/local/bin", null };
            // Try ImageMagick ('magick' or 'convert') first, then fallback to ffmpeg
            _ = std.posix.system.execve("/usr/bin/magick", &argv_magick, &envp);
            _ = std.posix.system.execve("/usr/bin/convert", &argv_convert, &envp);
            _ = std.posix.system.execve("/usr/bin/ffmpeg", &argv_ffmpeg, &envp);
            _ = std.posix.system.execve("/bin/ffmpeg", &argv_ffmpeg, &envp);
            std.posix.system.exit(1);
        }

        // Parent process
        _ = std.posix.system.close(write_fd);
        defer _ = std.posix.system.close(read_fd);

        var out_list: std.ArrayList(u8) = .empty;
        defer out_list.deinit(allocator);

        var buf: [32768]u8 = undefined;
        while (true) {
            const n = std.posix.read(read_fd, &buf) catch break;
            if (n == 0) break;
            try out_list.appendSlice(allocator, buf[0..n]);
        }

        var status: u32 = 0;
        _ = std.posix.system.waitpid(@intCast(pid), &status, 0);

        if (out_list.items.len >= 2 and out_list.items[0] == 'P' and out_list.items[1] == '6') {
            return loadPPM(allocator, out_list.items);
        }

        return error.UnsupportedImageFormat;
    }

    /// Resize image to target dimension using bilinear interpolation
    pub fn resize(self: *const Image, allocator: std.mem.Allocator, target_w: usize, target_h: usize) !Image {
        const out_data = try allocator.alloc(f32, target_w * target_h * 3);
        errdefer allocator.free(out_data);

        const scale_x = @as(f32, @floatFromInt(self.width)) / @as(f32, @floatFromInt(target_w));
        const scale_y = @as(f32, @floatFromInt(self.height)) / @as(f32, @floatFromInt(target_h));

        for (0..target_h) |ty| {
            const src_y = @as(f32, @floatFromInt(ty)) * scale_y;
            const y0: usize = @intFromFloat(src_y);
            const y1: usize = @min(y0 + 1, self.height - 1);
            const dy = src_y - @as(f32, @floatFromInt(y0));

            for (0..target_w) |tx| {
                const src_x = @as(f32, @floatFromInt(tx)) * scale_x;
                const x0: usize = @intFromFloat(src_x);
                const x1: usize = @min(x0 + 1, self.width - 1);
                const dx = src_x - @as(f32, @floatFromInt(x0));

                const dst_idx = (ty * target_w + tx) * 3;

                inline for (0..3) |c| {
                    const p00 = self.data[(y0 * self.width + x0) * 3 + c];
                    const p10 = self.data[(y0 * self.width + x1) * 3 + c];
                    const p01 = self.data[(y1 * self.width + x0) * 3 + c];
                    const p11 = self.data[(y1 * self.width + x1) * 3 + c];

                    const top = p00 * (1.0 - dx) + p10 * dx;
                    const bot = p01 * (1.0 - dx) + p11 * dx;
                    out_data[dst_idx + c] = top * (1.0 - dy) + bot * dy;
                }
            }
        }

        return Image{
            .width = target_w,
            .height = target_h,
            .channels = 3,
            .data = out_data,
            .allocator = allocator,
        };
    }

    /// Extract 16x16 pixel patches flattened as [N_patches, 16*16*3] = [N_patches, 768]
    /// Resizes image to standard resolution (max 256x256) if too large
    pub fn extractPatches(self: *const Image, allocator: std.mem.Allocator, patch_size: usize) ![]f32 {
        var processed_img: Image = undefined;
        var needs_free = false;

        const max_dim: usize = 256;
        if ((self.width > max_dim or self.height > max_dim) or (self.width % patch_size != 0) or (self.height % patch_size != 0)) {
            const target_w = ((@min(self.width, max_dim) + patch_size - 1) / patch_size) * patch_size;
            const target_h = ((@min(self.height, max_dim) + patch_size - 1) / patch_size) * patch_size;
            processed_img = try self.resize(allocator, target_w, target_h);
            needs_free = true;
        } else {
            processed_img = self.*;
        }
        defer if (needs_free) processed_img.deinit();

        const patch_dim = patch_size * patch_size * 3;
        const patches_x = processed_img.width / patch_size;
        const patches_y = processed_img.height / patch_size;
        const num_patches = patches_x * patches_y;

        if (num_patches == 0) return error.ImageSmallerThanPatch;

        const out = try allocator.alloc(f32, num_patches * patch_dim);
        errdefer allocator.free(out);

        var p_idx: usize = 0;
        for (0..patches_y) |py| {
            for (0..patches_x) |px| {
                const patch_out = out[p_idx * patch_dim .. (p_idx + 1) * patch_dim];
                for (0..3) |c| {
                    for (0..patch_size) |dy| {
                        for (0..patch_size) |dx| {
                            const img_x = px * patch_size + dx;
                            const img_y = py * patch_size + dy;
                            const src_idx = (img_y * processed_img.width + img_x) * 3;
                            patch_out[c * (patch_size * patch_size) + dy * patch_size + dx] = processed_img.data[src_idx + c];
                        }
                    }
                }
                p_idx += 1;
            }
        }

        return out;
    }
};

test "Image synthetic create, resize, and extract patches" {
    const allocator = std.testing.allocator;
    var img = try Image.createSynthetic(allocator, 100, 100);
    defer img.deinit();

    const patches = try img.extractPatches(allocator, 16);
    defer allocator.free(patches);

    try std.testing.expect(patches.len > 0);
}
