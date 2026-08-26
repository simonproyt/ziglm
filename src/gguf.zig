const std = @import("std");
const types = @import("types.zig");
const GGMLType = types.GGMLType;
const Tensor = types.Tensor;
const ModelParams = types.ModelParams;
const Architecture = types.Architecture;
const RoPEType = types.RoPEType;

pub const GGUF_MAGIC = 0x46554747; // "GGUF" in little endian

pub const GGUFValueType = enum(u32) {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
    _,
};

pub const GGUFArray = struct {
    type: GGUFValueType,
    len: usize,
    data: []const u8, // points directly into buffer
};

pub const GGUFValue = union(GGUFValueType) {
    UINT8: u8,
    INT8: i8,
    UINT16: u16,
    INT16: i16,
    UINT32: u32,
    INT32: i32,
    FLOAT32: f32,
    BOOL: bool,
    STRING: []const u8,
    ARRAY: GGUFArray,
    UINT64: u64,
    INT64: i64,
    FLOAT64: f64,
};

pub const GGUFMetadataKV = struct {
    key: []const u8,
    value: GGUFValue,
};

pub const GGUFFile = struct {
    allocator: std.mem.Allocator,
    raw_buffer: []const u8,
    is_mmap: bool,
    owns_buffer: bool = false,
    fd: ?std.posix.fd_t = null,

    version: u32,
    tensor_count: u64,
    metadata_count: u64,
    metadata: []GGUFMetadataKV,
    tensors: []Tensor,
    tensor_map: std.StringHashMap(usize),
    alignment: usize = 32,
    data_offset: usize = 0,

    params: ModelParams,

    pub fn open(allocator: std.mem.Allocator, file_path: []const u8) !*GGUFFile {
        const fd = try std.posix.openat(std.posix.AT.FDCWD, file_path, .{ .ACCMODE = .RDONLY }, 0);
        errdefer _ = std.posix.system.close(fd);

        const end_pos = std.posix.system.lseek(fd, 0, 2); // SEEK_END = 2
        if (end_pos < 0) return error.SeekFailed;
        _ = std.posix.system.lseek(fd, 0, 0); // SEEK_SET = 0
        const file_size: usize = @intCast(end_pos);
        if (file_size < 24) return error.InvalidGGUFFile;

        // Use mmap for zero-copy file mapping
        const mapped_slice = std.posix.mmap(
            null,
            file_size,
            .{ .READ = true },
            .{ .TYPE = .SHARED },
            fd,
            0,
        ) catch {
            // If mmap fails, read entire file into allocated memory
            const buffer = try allocator.alloc(u8, file_size);
            errdefer allocator.free(buffer);

            var total_read: usize = 0;
            while (total_read < file_size) {
                const n = try std.posix.read(fd, buffer[total_read..]);
                if (n == 0) break;
                total_read += n;
            }

            var f = try initFromBuffer(allocator, buffer, false, null);
            f.owns_buffer = true;
            _ = std.posix.system.close(fd);
            return f;
        };

        return initFromBuffer(allocator, mapped_slice, true, fd);
    }

    pub fn initFromBuffer(
        allocator: std.mem.Allocator,
        buffer: []const u8,
        is_mmap: bool,
        fd: ?std.posix.fd_t,
    ) !*GGUFFile {
        var self = try allocator.create(GGUFFile);
        self.* = .{
            .allocator = allocator,
            .raw_buffer = buffer,
            .is_mmap = is_mmap,
            .owns_buffer = false,
            .fd = fd,
            .version = 0,
            .tensor_count = 0,
            .metadata_count = 0,
            .metadata = &[_]GGUFMetadataKV{},
            .tensors = &[_]Tensor{},
            .tensor_map = std.StringHashMap(usize).init(allocator),
            .params = .{},
        };
        errdefer self.deinit();

        try self.parse();
        return self;
    }

    pub fn deinit(self: *GGUFFile) void {
        self.tensor_map.deinit();
        if (self.metadata.len > 0) {
            self.allocator.free(self.metadata);
        }
        if (self.tensors.len > 0) {
            self.allocator.free(self.tensors);
        }
        if (self.is_mmap) {
            std.posix.munmap(@alignCast(self.raw_buffer));
        } else if (self.owns_buffer) {
            self.allocator.free(self.raw_buffer);
        }
        if (self.fd) |fd| {
            _ = std.posix.system.close(fd);
        }
        self.allocator.destroy(self);
    }

    fn readU8(self: *GGUFFile, offset: *usize) !u8 {
        if (offset.* + 1 > self.raw_buffer.len) return error.UnexpectedEOF;
        const val = self.raw_buffer[offset.*];
        offset.* += 1;
        return val;
    }

    fn readU16(self: *GGUFFile, offset: *usize) !u16 {
        if (offset.* + 2 > self.raw_buffer.len) return error.UnexpectedEOF;
        const val = std.mem.readInt(u16, self.raw_buffer[offset.*..][0..2], .little);
        offset.* += 2;
        return val;
    }

    fn readU32(self: *GGUFFile, offset: *usize) !u32 {
        if (offset.* + 4 > self.raw_buffer.len) return error.UnexpectedEOF;
        const val = std.mem.readInt(u32, self.raw_buffer[offset.*..][0..4], .little);
        offset.* += 4;
        return val;
    }

    fn readU64(self: *GGUFFile, offset: *usize) !u64 {
        if (offset.* + 8 > self.raw_buffer.len) return error.UnexpectedEOF;
        const val = std.mem.readInt(u64, self.raw_buffer[offset.*..][0..8], .little);
        offset.* += 8;
        return val;
    }

    fn readI32(self: *GGUFFile, offset: *usize) !i32 {
        if (offset.* + 4 > self.raw_buffer.len) return error.UnexpectedEOF;
        const val = std.mem.readInt(i32, self.raw_buffer[offset.*..][0..4], .little);
        offset.* += 4;
        return val;
    }

    fn readF32(self: *GGUFFile, offset: *usize) !f32 {
        const u = try self.readU32(offset);
        return @bitCast(u);
    }

    fn readString(self: *GGUFFile, offset: *usize) ![]const u8 {
        const len = try self.readU64(offset);
        if (offset.* + len > self.raw_buffer.len) return error.UnexpectedEOF;
        const str = self.raw_buffer[offset.* .. offset.* + len];
        offset.* += len;
        return str;
    }

    fn parseValue(self: *GGUFFile, offset: *usize, val_type: GGUFValueType) !GGUFValue {
        return switch (val_type) {
            .UINT8 => .{ .UINT8 = try self.readU8(offset) },
            .INT8 => .{ .INT8 = @bitCast(try self.readU8(offset)) },
            .UINT16 => .{ .UINT16 = try self.readU16(offset) },
            .INT16 => .{ .INT16 = @bitCast(try self.readU16(offset)) },
            .UINT32 => .{ .UINT32 = try self.readU32(offset) },
            .INT32 => .{ .INT32 = try self.readI32(offset) },
            .FLOAT32 => .{ .FLOAT32 = try self.readF32(offset) },
            .BOOL => .{ .BOOL = (try self.readU8(offset)) != 0 },
            .STRING => .{ .STRING = try self.readString(offset) },
            .UINT64 => .{ .UINT64 = try self.readU64(offset) },
            .INT64 => .{ .INT64 = @bitCast(try self.readU64(offset)) },
            .FLOAT64 => .{ .FLOAT64 = @bitCast(try self.readU64(offset)) },
            .ARRAY => blk: {
                const arr_type_u32 = try self.readU32(offset);
                const arr_type: GGUFValueType = @enumFromInt(arr_type_u32);
                const arr_len = try self.readU64(offset);
                const start_data = offset.*;

                for (0..arr_len) |_| {
                    _ = try self.parseValue(offset, arr_type);
                }

                break :blk .{
                    .ARRAY = .{
                        .type = arr_type,
                        .len = arr_len,
                        .data = self.raw_buffer[start_data..offset.*],
                    },
                };
            },
            _ => error.UnsupportedGGUFValueType,
        };
    }

    fn parse(self: *GGUFFile) !void {
        var offset: usize = 0;

        // Magic
        const magic = try self.readU32(&offset);
        if (magic != GGUF_MAGIC) return error.InvalidGGUFMagic;

        // Version (supports v2 and v3)
        self.version = try self.readU32(&offset);
        if (self.version != 2 and self.version != 3) {
            return error.UnsupportedGGUFVersion;
        }

        // Counts
        self.tensor_count = try self.readU64(&offset);
        self.metadata_count = try self.readU64(&offset);

        // Parse Metadata Key-Value pairs
        self.metadata = try self.allocator.alloc(GGUFMetadataKV, self.metadata_count);
        for (0..self.metadata_count) |i| {
            const key = try self.readString(&offset);
            const val_type_u32 = try self.readU32(&offset);
            const val_type: GGUFValueType = @enumFromInt(val_type_u32);
            const value = try self.parseValue(&offset, val_type);
            self.metadata[i] = .{ .key = key, .value = value };

            if (std.mem.eql(u8, key, "general.alignment")) {
                if (value == .UINT32) self.alignment = value.UINT32;
            }
        }

        // Parse Tensor Information
        self.tensors = try self.allocator.alloc(Tensor, self.tensor_count);
        for (0..self.tensor_count) |i| {
            const name = try self.readString(&offset);
            const n_dims = try self.readU32(&offset);
            if (n_dims > 4) return error.InvalidTensorDimensions;

            var shape = [4]usize{ 1, 1, 1, 1 };
            for (0..n_dims) |d| {
                shape[d] = try self.readU64(&offset);
            }

            const qtype_u32 = try self.readU32(&offset);
            const qtype: GGMLType = @enumFromInt(qtype_u32);
            const tensor_offset = try self.readU64(&offset);

            self.tensors[i] = .{
                .name = name,
                .type = qtype,
                .n_dims = n_dims,
                .shape = shape,
                .offset = tensor_offset,
                .data = &[_]u8{},
            };

            try self.tensor_map.put(name, i);
        }

        // Align offset to alignment boundary for tensor data
        const align_mask = self.alignment - 1;
        self.data_offset = (offset + align_mask) & ~align_mask;

        // Set tensor data slices
        for (self.tensors) |*t| {
            const abs_offset = self.data_offset + t.offset;
            const size = t.sizeBytes();
            if (abs_offset + size > self.raw_buffer.len) {
                return error.TensorDataOutOfBounds;
            }
            t.data = self.raw_buffer[abs_offset .. abs_offset + size];
        }

        // Extract Model Architecture & Parameters
        self.extractParams();
    }

    pub fn getMetadata(self: *const GGUFFile, key: []const u8) ?GGUFValue {
        for (self.metadata) |kv| {
            if (std.mem.eql(u8, kv.key, key)) return kv.value;
        }
        return null;
    }

    pub fn getU32(self: *const GGUFFile, key: []const u8) ?u32 {
        if (self.getMetadata(key)) |val| {
            return switch (val) {
                .UINT32 => |v| v,
                .UINT64 => |v| @truncate(v),
                .INT32 => |v| @intCast(v),
                else => null,
            };
        }
        return null;
    }

    pub fn getU64(self: *const GGUFFile, key: []const u8) ?u64 {
        if (self.getMetadata(key)) |val| {
            return switch (val) {
                .UINT64 => |v| v,
                .UINT32 => |v| v,
                .INT64 => |v| @intCast(v),
                .INT32 => |v| @intCast(v),
                .ARRAY => |arr| blk: {
                    if (arr.len > 0) {
                        if (arr.type == .UINT32 and arr.data.len >= 4) {
                            break :blk std.mem.readInt(u32, arr.data[0..4], .little);
                        } else if (arr.type == .UINT64 and arr.data.len >= 8) {
                            break :blk std.mem.readInt(u64, arr.data[0..8], .little);
                        } else if (arr.type == .INT32 and arr.data.len >= 4) {
                            break :blk @intCast(std.mem.readInt(i32, arr.data[0..4], .little));
                        } else if (arr.type == .INT64 and arr.data.len >= 8) {
                            break :blk @intCast(std.mem.readInt(i64, arr.data[0..8], .little));
                        }
                    }
                    break :blk null;
                },
                else => null,
            };
        }
        return null;
    }

    pub fn getF32(self: *const GGUFFile, key: []const u8) ?f32 {
        if (self.getMetadata(key)) |val| {
            return switch (val) {
                .FLOAT32 => |v| v,
                .FLOAT64 => |v| @floatCast(v),
                .ARRAY => |arr| blk: {
                    if (arr.len > 0) {
                        if (arr.type == .FLOAT32 and arr.data.len >= 4) {
                            const u = std.mem.readInt(u32, arr.data[0..4], .little);
                            break :blk @bitCast(u);
                        }
                    }
                    break :blk null;
                },
                else => null,
            };
        }
        return null;
    }

    pub fn getString(self: *const GGUFFile, key: []const u8) ?[]const u8 {
        if (self.getMetadata(key)) |val| {
            if (val == .STRING) return val.STRING;
        }
        return null;
    }

    pub fn getTensor(self: *const GGUFFile, name: []const u8) ?Tensor {
        if (self.tensor_map.get(name)) |idx| {
            return self.tensors[idx];
        }
        return null;
    }

    fn extractParams(self: *GGUFFile) void {
        const arch_name = self.getString("general.architecture") orelse "llama";
        self.params.arch = Architecture.fromString(arch_name);

        var key_buf: [128]u8 = undefined;

        if (self.findParamU64(arch_name, "context_length", &key_buf)) |v| {
            self.params.context_length = v;
        }
        if (self.findParamU64(arch_name, "embedding_length", &key_buf)) |v| {
            self.params.embedding_length = v;
        }
        if (self.findParamU64(arch_name, "feed_forward_length", &key_buf)) |v| {
            self.params.feed_forward_length = v;
        }
        if (self.findParamU64(arch_name, "block_count", &key_buf)) |v| {
            self.params.block_count = v;
        }
        if (self.findParamU64(arch_name, "attention.head_count", &key_buf)) |v| {
            self.params.head_count = v;
        }
        if (self.findParamU64(arch_name, "attention.head_count_kv", &key_buf)) |v| {
            self.params.head_count_kv = v;
        } else {
            self.params.head_count_kv = self.params.head_count;
        }
        if (self.findParamU64(arch_name, "attention.key_length", &key_buf)) |v| {
            self.params.head_size = v;
        }
        if (self.findParamU64(arch_name, "attention.sliding_window", &key_buf)) |v| {
            self.params.sliding_window = v;
        }
        if (self.findParamF32(arch_name, "attention.layer_norm_rms_epsilon", &key_buf)) |v| {
            self.params.layer_norm_rms_epsilon = v;
        }
        if (self.findParamF32(arch_name, "rope.freq_base", &key_buf)) |v| {
            self.params.rope_freq_base = v;
        }
        if (self.findParamF32(arch_name, "rope.freq_scale", &key_buf)) |v| {
            self.params.rope_freq_scale = v;
        }
        if (self.findParamU64(arch_name, "rope.dimension_count", &key_buf)) |v| {
            self.params.rope_dim_count = v;
        }
        if (self.findParamF32(arch_name, "final_logit_softcapping", &key_buf)) |v| {
            self.params.final_logit_softcapping = v;
        }
        if (self.getTensor("token_embd.weight")) |t| {
            self.params.vocab_size = t.shape[1];
        }

        self.params.initComputed();
    }

    fn findParamU64(self: *const GGUFFile, arch: []const u8, name: []const u8, buf: *[128]u8) ?usize {
        const full_key = std.fmt.bufPrint(buf, "{s}.{s}", .{ arch, name }) catch return null;
        if (self.getU64(full_key)) |v| return @intCast(v);
        return null;
    }

    fn findParamF32(self: *const GGUFFile, arch: []const u8, name: []const u8, buf: *[128]u8) ?f32 {
        const full_key = std.fmt.bufPrint(buf, "{s}.{s}", .{ arch, name }) catch return null;
        return self.getF32(full_key);
    }
};

pub const BufferWriter = struct {
    buffer: []u8,
    pos: usize = 0,

    pub fn writeBytes(self: *BufferWriter, bytes: []const u8) !void {
        if (self.pos + bytes.len > self.buffer.len) return error.NoSpaceLeft;
        @memcpy(self.buffer[self.pos .. self.pos + bytes.len], bytes);
        self.pos += bytes.len;
    }

    pub fn writeU8(self: *BufferWriter, v: u8) !void {
        try self.writeBytes(&[_]u8{v});
    }

    pub fn writeU16(self: *BufferWriter, v: u16) !void {
        var b: [2]u8 = undefined;
        std.mem.writeInt(u16, &b, v, .little);
        try self.writeBytes(&b);
    }

    pub fn writeU32(self: *BufferWriter, v: u32) !void {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, v, .little);
        try self.writeBytes(&b);
    }

    pub fn writeU64(self: *BufferWriter, v: u64) !void {
        var b: [8]u8 = undefined;
        std.mem.writeInt(u64, &b, v, .little);
        try self.writeBytes(&b);
    }

    pub fn writeF32(self: *BufferWriter, v: f32) !void {
        const u: u32 = @bitCast(v);
        try self.writeU32(u);
    }

    pub fn writeString(self: *BufferWriter, str: []const u8) !void {
        try self.writeU64(str.len);
        try self.writeBytes(str);
    }
};

test "GGUF parsing in-memory validation" {
    var buffer: [512]u8 = undefined;
    var bw = BufferWriter{ .buffer = &buffer };

    try bw.writeU32(GGUF_MAGIC);
    try bw.writeU32(3);
    try bw.writeU64(0);
    try bw.writeU64(1);

    try bw.writeString("general.architecture");
    try bw.writeU32(@intFromEnum(GGUFValueType.STRING));
    try bw.writeString("llama");

    const written_bytes = buffer[0..bw.pos];
    var file = try GGUFFile.initFromBuffer(std.testing.allocator, written_bytes, false, null);
    defer file.deinit();

    try std.testing.expectEqual(@as(u32, 3), file.version);
    try std.testing.expectEqual(Architecture.llama, file.params.arch);
}
