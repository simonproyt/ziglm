const std = @import("std");
const Image = @import("image.zig").Image;
const vision = @import("vision.zig");
const VisionEncoder = vision.VisionEncoder;

pub const VideoFrame = struct {
    image: Image,
    timestamp_sec: f32,

    pub fn deinit(self: *VideoFrame) void {
        self.image.deinit();
    }
};

pub const Video = struct {
    allocator: std.mem.Allocator,
    frames: []VideoFrame,
    fps: f32 = 1.0,
    duration_sec: f32 = 0.0,

    pub fn deinit(self: *Video) void {
        for (self.frames) |*frame| {
            frame.deinit();
        }
        self.allocator.free(self.frames);
        self.allocator.destroy(self);
    }

    /// Extract N evenly spaced frames from a video file
    pub fn load(allocator: std.mem.Allocator, video_path: []const u8, max_frames: usize) !*Video {
        const self = try allocator.create(Video);
        errdefer allocator.destroy(self);

        var pipe_fds: [2]std.posix.fd_t = undefined;
        if (std.posix.system.pipe(&pipe_fds) != 0) {
            // Fallback to single image load
            const img = try Image.loadFromFile(allocator, video_path);
            const frames = try allocator.alloc(VideoFrame, 1);
            frames[0] = .{ .image = img, .timestamp_sec = 0.0 };
            self.* = .{
                .allocator = allocator,
                .frames = frames,
                .fps = 1.0,
                .duration_sec = 1.0,
            };
            return self;
        }
        const read_fd = pipe_fds[0];
        const write_fd = pipe_fds[1];

        const pid = std.posix.system.fork();
        if (pid < 0) {
            _ = std.posix.system.close(read_fd);
            _ = std.posix.system.close(write_fd);
            const img = try Image.loadFromFile(allocator, video_path);
            const frames = try allocator.alloc(VideoFrame, 1);
            frames[0] = .{ .image = img, .timestamp_sec = 0.0 };
            self.* = .{
                .allocator = allocator,
                .frames = frames,
                .fps = 1.0,
                .duration_sec = 1.0,
            };
            return self;
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
            const path_z = std.fmt.bufPrintZ(&path_z_buf, "{s}", .{video_path}) catch std.posix.system.exit(1);

            var vframes_buf: [16]u8 = undefined;
            const vframes_str = std.fmt.bufPrintZ(&vframes_buf, "{d}", .{max_frames}) catch "8";

            const argv = [_:null]?[*:0]const u8{
                "ffmpeg",
                "-y",
                "-i",
                path_z.ptr,
                "-vf",
                "fps=1,scale=256:256:force_original_aspect_ratio=decrease,pad=256:256:(ow-iw)/2:(oh-ih)/2",
                "-vframes",
                vframes_str.ptr,
                "-f",
                "image2pipe",
                "-vcodec",
                "ppm",
                "-",
                null,
            };

            const envp = [_:null]?[*:0]const u8{ "PATH=/usr/bin:/bin:/usr/local/bin", null };
            _ = std.posix.system.execve("/usr/bin/ffmpeg", &argv, &envp);
            _ = std.posix.system.execve("/bin/ffmpeg", &argv, &envp);
            std.posix.system.exit(1);
        }

        // Parent process
        _ = std.posix.system.close(write_fd);
        defer _ = std.posix.system.close(read_fd);

        var out_list: std.ArrayList(u8) = .empty;
        defer out_list.deinit(allocator);

        var buf: [65536]u8 = undefined;
        while (true) {
            const n = std.posix.read(read_fd, &buf) catch break;
            if (n == 0) break;
            try out_list.appendSlice(allocator, buf[0..n]);
        }

        var status: c_int = 0;
        _ = std.posix.system.waitpid(@intCast(pid), &status, 0);

        var frames_list: std.ArrayList(VideoFrame) = .empty;
        errdefer {
            for (frames_list.items) |*f| f.deinit();
            frames_list.deinit(allocator);
        }

        var offset: usize = 0;
        const bytes = out_list.items;
        var frame_idx: usize = 0;

        while (offset + 10 < bytes.len and frame_idx < max_frames) {
            if (bytes[offset] != 'P' or bytes[offset + 1] != '6') {
                offset += 1;
                continue;
            }

            const slice = bytes[offset..];
            if (Image.loadPPMWithConsumedBytes(allocator, slice)) |res| {
                try frames_list.append(allocator, .{
                    .image = res.image,
                    .timestamp_sec = @as(f32, @floatFromInt(frame_idx)),
                });
                frame_idx += 1;
                offset += res.bytes_consumed;
            } else |_| {
                offset += 1;
            }
        }

        if (frames_list.items.len == 0) {
            // Single image/fallback load
            if (Image.loadFromFile(allocator, video_path)) |img| {
                try frames_list.append(allocator, .{
                    .image = img,
                    .timestamp_sec = 0.0,
                });
            } else |err| return err;
        }

        self.* = .{
            .allocator = allocator,
            .frames = try frames_list.toOwnedSlice(allocator),
            .fps = 1.0,
            .duration_sec = @as(f32, @floatFromInt(frames_list.items.len)),
        };
        return self;
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "Video structure creation and memory management" {
    const allocator = std.testing.allocator;

    // Create synthetic video with 2 test frames
    const data1 = try allocator.alloc(f32, 16 * 16 * 3);
    const data2 = try allocator.alloc(f32, 16 * 16 * 3);
    @memset(data1, 0.5);
    @memset(data2, 0.8);
    const img1 = Image{ .allocator = allocator, .width = 16, .height = 16, .channels = 3, .data = data1 };
    const img2 = Image{ .allocator = allocator, .width = 16, .height = 16, .channels = 3, .data = data2 };

    const frames = try allocator.alloc(VideoFrame, 2);
    frames[0] = .{ .image = img1, .timestamp_sec = 0.0 };
    frames[1] = .{ .image = img2, .timestamp_sec = 1.0 };

    const video = try allocator.create(Video);
    video.* = .{
        .allocator = allocator,
        .frames = frames,
        .fps = 1.0,
        .duration_sec = 2.0,
    };
    defer video.deinit();

    try std.testing.expectEqual(@as(usize, 2), video.frames.len);
}
