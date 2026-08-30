const std = @import("std");
const types = @import("types.zig");
const GGMLType = types.GGMLType;
const gguf = @import("gguf.zig");
const GGUF_MAGIC = gguf.GGUF_MAGIC;
const GGUFValueType = gguf.GGUFValueType;
const BufferWriter = gguf.BufferWriter;

pub const SampleModelConfig = struct {
    arch: []const u8 = "llama",
    block_count: usize = 2,
    embedding_length: usize = 64,
    feed_forward_length: usize = 128,
    head_count: usize = 4,
    head_count_kv: usize = 2,
    vocab_size: usize = 32,
    context_length: usize = 128,
};

fn writeAllBytes(fd: std.posix.fd_t, bytes: []const u8) !void {
    var written: usize = 0;
    while (written < bytes.len) {
        const rc = std.posix.system.write(fd, bytes[written..].ptr, bytes.len - written);
        const err = std.posix.errno(rc);
        if (err != .SUCCESS) return error.WriteFailed;
        written += @as(usize, @intCast(rc));
    }
}

pub fn generateSampleGGUF(allocator: std.mem.Allocator, file_path: []const u8, config: SampleModelConfig) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    const fd = try std.posix.openat(
        std.posix.AT.FDCWD,
        file_path,
        .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true },
        0o644,
    );
    defer _ = std.posix.system.close(fd);

    var meta_buf: [16384]u8 = undefined;
    var bw = BufferWriter{ .buffer = &meta_buf };

    // 1. Header
    try bw.writeU32(GGUF_MAGIC);
    try bw.writeU32(3); // version

    const tensor_count: u64 = 3 + config.block_count * 9;
    try bw.writeU64(tensor_count);

    const meta_count: u64 = 15;
    try bw.writeU64(meta_count);

    // 2. Metadata Key-Values
    try bw.writeString("general.architecture");
    try bw.writeU32(@intFromEnum(GGUFValueType.STRING));
    try bw.writeString(config.arch);

    var key_buf: [64]u8 = undefined;
    try bw.writeString(try std.fmt.bufPrint(&key_buf, "{s}.block_count", .{config.arch}));
    try bw.writeU32(@intFromEnum(GGUFValueType.UINT32));
    try bw.writeU32(@intCast(config.block_count));

    try bw.writeString(try std.fmt.bufPrint(&key_buf, "{s}.context_length", .{config.arch}));
    try bw.writeU32(@intFromEnum(GGUFValueType.UINT32));
    try bw.writeU32(@intCast(config.context_length));

    try bw.writeString(try std.fmt.bufPrint(&key_buf, "{s}.embedding_length", .{config.arch}));
    try bw.writeU32(@intFromEnum(GGUFValueType.UINT32));
    try bw.writeU32(@intCast(config.embedding_length));

    try bw.writeString(try std.fmt.bufPrint(&key_buf, "{s}.feed_forward_length", .{config.arch}));
    try bw.writeU32(@intFromEnum(GGUFValueType.UINT32));
    try bw.writeU32(@intCast(config.feed_forward_length));

    try bw.writeString(try std.fmt.bufPrint(&key_buf, "{s}.attention.head_count", .{config.arch}));
    try bw.writeU32(@intFromEnum(GGUFValueType.UINT32));
    try bw.writeU32(@intCast(config.head_count));

    try bw.writeString(try std.fmt.bufPrint(&key_buf, "{s}.attention.head_count_kv", .{config.arch}));
    try bw.writeU32(@intFromEnum(GGUFValueType.UINT32));
    try bw.writeU32(@intCast(config.head_count_kv));

    try bw.writeString(try std.fmt.bufPrint(&key_buf, "{s}.attention.layer_norm_rms_epsilon", .{config.arch}));
    try bw.writeU32(@intFromEnum(GGUFValueType.FLOAT32));
    try bw.writeF32(1e-5);

    try bw.writeString(try std.fmt.bufPrint(&key_buf, "{s}.rope.freq_base", .{config.arch}));
    try bw.writeU32(@intFromEnum(GGUFValueType.FLOAT32));
    try bw.writeF32(10000.0);

    try bw.writeString(try std.fmt.bufPrint(&key_buf, "{s}.rope.dimension_count", .{config.arch}));
    try bw.writeU32(@intFromEnum(GGUFValueType.UINT32));
    try bw.writeU32(@intCast(config.embedding_length / config.head_count));

    try bw.writeString("tokenizer.ggml.model");
    try bw.writeU32(@intFromEnum(GGUFValueType.STRING));
    try bw.writeString("gpt2");

    try bw.writeString("tokenizer.ggml.bos_token_id");
    try bw.writeU32(@intFromEnum(GGUFValueType.UINT32));
    try bw.writeU32(0);

    try bw.writeString("tokenizer.ggml.eos_token_id");
    try bw.writeU32(@intFromEnum(GGUFValueType.UINT32));
    try bw.writeU32(1);

    try bw.writeString("tokenizer.ggml.tokens");
    try bw.writeU32(@intFromEnum(GGUFValueType.ARRAY));
    try bw.writeU32(@intFromEnum(GGUFValueType.STRING));
    try bw.writeU64(config.vocab_size);

    const sample_tokens = [_][]const u8{
        "<s>", "</s>", "<unk>", "Hello", "world", "!",
        "The", "capital", "of", "France", "is", "Paris",
        "A", "B", "C", "D", "E", "F", "G", "H",
        "I", "J", "K", "L", "M", "N", "O", "P",
        "Q", "R", "S", "T",
    };

    for (0..config.vocab_size) |i| {
        if (i < sample_tokens.len) {
            try bw.writeString(sample_tokens[i]);
        } else {
            var tok_buf: [32]u8 = undefined;
            const tok_str = try std.fmt.bufPrint(&tok_buf, "tok_{d}", .{i});
            try bw.writeString(tok_str);
        }
    }

    try bw.writeString("tokenizer.ggml.merges");
    try bw.writeU32(@intFromEnum(GGUFValueType.ARRAY));
    try bw.writeU32(@intFromEnum(GGUFValueType.STRING));
    try bw.writeU64(2);
    try bw.writeString("t h");
    try bw.writeString("Ġ t");

    // 3. Tensor Information
    const dim = config.embedding_length;
    const ffn_dim = config.feed_forward_length;
    const head_size = dim / config.head_count;
    const kv_dim = config.head_count_kv * head_size;
    const vocab = config.vocab_size;

    const TensorMeta = struct {
        name: []const u8,
        shape: [2]u64,
        qtype: GGMLType,
        size_bytes: usize,
    };

    var tensors_meta: std.ArrayList(TensorMeta) = .empty;
    defer tensors_meta.deinit(arena_allocator);

    try tensors_meta.append(arena_allocator, .{
        .name = "token_embd.weight",
        .shape = .{ dim, vocab },
        .qtype = .F32,
        .size_bytes = vocab * dim * @sizeOf(f32),
    });

    for (0..config.block_count) |i| {
        const norm1_name = try std.fmt.allocPrint(arena_allocator, "blk.{d}.attn_norm.weight", .{i});
        try tensors_meta.append(arena_allocator, .{ .name = norm1_name, .shape = .{ dim, 1 }, .qtype = .F32, .size_bytes = dim * @sizeOf(f32) });

        const q_name = try std.fmt.allocPrint(arena_allocator, "blk.{d}.attn_q.weight", .{i});
        try tensors_meta.append(arena_allocator, .{ .name = q_name, .shape = .{ dim, dim }, .qtype = .F32, .size_bytes = dim * dim * @sizeOf(f32) });

        const k_name = try std.fmt.allocPrint(arena_allocator, "blk.{d}.attn_k.weight", .{i});
        try tensors_meta.append(arena_allocator, .{ .name = k_name, .shape = .{ dim, kv_dim }, .qtype = .F32, .size_bytes = kv_dim * dim * @sizeOf(f32) });

        const v_name = try std.fmt.allocPrint(arena_allocator, "blk.{d}.attn_v.weight", .{i});
        try tensors_meta.append(arena_allocator, .{ .name = v_name, .shape = .{ dim, kv_dim }, .qtype = .F32, .size_bytes = kv_dim * dim * @sizeOf(f32) });

        const o_name = try std.fmt.allocPrint(arena_allocator, "blk.{d}.attn_output.weight", .{i});
        try tensors_meta.append(arena_allocator, .{ .name = o_name, .shape = .{ dim, dim }, .qtype = .F32, .size_bytes = dim * dim * @sizeOf(f32) });

        const norm2_name = try std.fmt.allocPrint(arena_allocator, "blk.{d}.ffn_norm.weight", .{i});
        try tensors_meta.append(arena_allocator, .{ .name = norm2_name, .shape = .{ dim, 1 }, .qtype = .F32, .size_bytes = dim * @sizeOf(f32) });

        const gate_name = try std.fmt.allocPrint(arena_allocator, "blk.{d}.ffn_gate.weight", .{i});
        try tensors_meta.append(arena_allocator, .{ .name = gate_name, .shape = .{ dim, ffn_dim }, .qtype = .F32, .size_bytes = ffn_dim * dim * @sizeOf(f32) });

        const up_name = try std.fmt.allocPrint(arena_allocator, "blk.{d}.ffn_up.weight", .{i});
        try tensors_meta.append(arena_allocator, .{ .name = up_name, .shape = .{ dim, ffn_dim }, .qtype = .F32, .size_bytes = ffn_dim * dim * @sizeOf(f32) });

        const down_name = try std.fmt.allocPrint(arena_allocator, "blk.{d}.ffn_down.weight", .{i});
        try tensors_meta.append(arena_allocator, .{ .name = down_name, .shape = .{ ffn_dim, dim }, .qtype = .F32, .size_bytes = dim * ffn_dim * @sizeOf(f32) });
    }

    try tensors_meta.append(arena_allocator, .{
        .name = "output_norm.weight",
        .shape = .{ dim, 1 },
        .qtype = .F32,
        .size_bytes = dim * @sizeOf(f32),
    });

    try tensors_meta.append(arena_allocator, .{
        .name = "output.weight",
        .shape = .{ dim, vocab },
        .qtype = .F32,
        .size_bytes = vocab * dim * @sizeOf(f32),
    });

    // Write Tensor Info headers
    var current_offset: u64 = 0;
    for (tensors_meta.items) |t| {
        try bw.writeString(t.name);
        try bw.writeU32(2);
        try bw.writeU64(t.shape[0]);
        try bw.writeU64(t.shape[1]);
        try bw.writeU32(@intFromEnum(t.qtype));
        try bw.writeU64(current_offset);
        current_offset += t.size_bytes;
    }

    // Write header bytes to file
    try writeAllBytes(fd, meta_buf[0..bw.pos]);

    // Align to 32 bytes
    const align_boundary: usize = 32;
    const pad = (align_boundary - (bw.pos % align_boundary)) % align_boundary;
    if (pad > 0) {
        const zeros = [_]u8{0} ** 32;
        try writeAllBytes(fd, zeros[0..pad]);
    }

    // Write deterministic weights
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    for (tensors_meta.items) |t| {
        const num_floats = t.size_bytes / @sizeOf(f32);
        const float_buf = try arena_allocator.alloc(f32, num_floats);

        const scale: f32 = 0.02;
        for (float_buf) |*v| {
            v.* = (rand.float(f32) - 0.5) * scale;
        }

        if (std.mem.endsWith(u8, t.name, "norm.weight")) {
            for (float_buf) |*v| v.* = 1.0;
        }

        try writeAllBytes(fd, std.mem.sliceAsBytes(float_buf));
    }
}

pub fn generateSampleSafeTensors(allocator: std.mem.Allocator, dir_path: []const u8, config: SampleModelConfig) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var dir_z: [1024:0]u8 = undefined;
    if (dir_path.len < dir_z.len) {
        @memcpy(dir_z[0..dir_path.len], dir_path);
        dir_z[dir_path.len] = 0;
        _ = std.posix.system.mkdir(&dir_z, 0o755);
    }

    // 1. Write config.json
    var cfg_path_buf: [1024]u8 = undefined;
    const cfg_path = try std.fmt.bufPrint(&cfg_path_buf, "{s}/config.json", .{dir_path});
    const cfg_fd = try std.posix.openat(std.posix.AT.FDCWD, cfg_path, .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true }, 0o644);
    defer _ = std.posix.system.close(cfg_fd);

    const config_json = try std.fmt.allocPrint(arena_allocator,
        \\{{
        \\  "architectures": ["LlamaForCausalLM"],
        \\  "hidden_size": {d},
        \\  "intermediate_size": {d},
        \\  "num_attention_heads": {d},
        \\  "num_key_value_heads": {d},
        \\  "num_hidden_layers": {d},
        \\  "vocab_size": {d},
        \\  "max_position_embeddings": {d},
        \\  "rms_norm_eps": 1e-5,
        \\  "rope_theta": 10000.0
        \\}}
    , .{
        config.embedding_length,
        config.feed_forward_length,
        config.head_count,
        config.head_count_kv,
        config.block_count,
        config.vocab_size,
        config.context_length,
    });
    try writeAllBytes(cfg_fd, config_json);

    // 2. Write tokenizer.json
    var tok_path_buf: [1024]u8 = undefined;
    const tok_path = try std.fmt.bufPrint(&tok_path_buf, "{s}/tokenizer.json", .{dir_path});
    const tok_fd = try std.posix.openat(std.posix.AT.FDCWD, tok_path, .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true }, 0o644);
    defer _ = std.posix.system.close(tok_fd);

    const tokenizer_json =
        \\{
        \\  "version": "1.0",
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "<s>": 0, "</s>": 1, "<unk>": 2,
        \\      "Hello": 3, "world": 4, "!": 5,
        \\      "The": 6, "capital": 7, "of": 8, "France": 9, "is": 10, "Paris": 11
        \\    },
        \\    "merges": ["t h", "Ġ t"]
        \\  },
        \\  "added_tokens": [
        \\    {"id": 0, "content": "<s>", "special": true},
        \\    {"id": 1, "content": "</s>", "special": true}
        \\  ]
        \\}
    ;
    try writeAllBytes(tok_fd, tokenizer_json);

    // 3. Write model.safetensors
    var st_path_buf: [1024]u8 = undefined;
    const st_path = try std.fmt.bufPrint(&st_path_buf, "{s}/model.safetensors", .{dir_path});
    const st_fd = try std.posix.openat(std.posix.AT.FDCWD, st_path, .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true }, 0o644);
    defer _ = std.posix.system.close(st_fd);

    const dim = config.embedding_length;
    const ffn_dim = config.feed_forward_length;
    const head_size = dim / config.head_count;
    const kv_dim = config.head_count_kv * head_size;
    const vocab = config.vocab_size;

    const TensorMeta = struct {
        name: []const u8,
        shape: [2]usize,
        size_bytes: usize,
    };

    var tensors_meta: std.ArrayList(TensorMeta) = .empty;
    defer tensors_meta.deinit(arena_allocator);

    try tensors_meta.append(arena_allocator, .{
        .name = "model.embed_tokens.weight",
        .shape = .{ vocab, dim },
        .size_bytes = vocab * dim * @sizeOf(f32),
    });

    for (0..config.block_count) |i| {
        const norm1_name = try std.fmt.allocPrint(arena_allocator, "model.layers.{d}.input_layernorm.weight", .{i});
        try tensors_meta.append(arena_allocator, .{ .name = norm1_name, .shape = .{ dim, 1 }, .size_bytes = dim * @sizeOf(f32) });

        const q_name = try std.fmt.allocPrint(arena_allocator, "model.layers.{d}.self_attn.q_proj.weight", .{i});
        try tensors_meta.append(arena_allocator, .{ .name = q_name, .shape = .{ dim, dim }, .size_bytes = dim * dim * @sizeOf(f32) });

        const k_name = try std.fmt.allocPrint(arena_allocator, "model.layers.{d}.self_attn.k_proj.weight", .{i});
        try tensors_meta.append(arena_allocator, .{ .name = k_name, .shape = .{ kv_dim, dim }, .size_bytes = kv_dim * dim * @sizeOf(f32) });

        const v_name = try std.fmt.allocPrint(arena_allocator, "model.layers.{d}.self_attn.v_proj.weight", .{i});
        try tensors_meta.append(arena_allocator, .{ .name = v_name, .shape = .{ kv_dim, dim }, .size_bytes = kv_dim * dim * @sizeOf(f32) });

        const o_name = try std.fmt.allocPrint(arena_allocator, "model.layers.{d}.self_attn.o_proj.weight", .{i});
        try tensors_meta.append(arena_allocator, .{ .name = o_name, .shape = .{ dim, dim }, .size_bytes = dim * dim * @sizeOf(f32) });

        const norm2_name = try std.fmt.allocPrint(arena_allocator, "model.layers.{d}.post_attention_layernorm.weight", .{i});
        try tensors_meta.append(arena_allocator, .{ .name = norm2_name, .shape = .{ dim, 1 }, .size_bytes = dim * @sizeOf(f32) });

        const gate_name = try std.fmt.allocPrint(arena_allocator, "model.layers.{d}.mlp.gate_proj.weight", .{i});
        try tensors_meta.append(arena_allocator, .{ .name = gate_name, .shape = .{ ffn_dim, dim }, .size_bytes = ffn_dim * dim * @sizeOf(f32) });

        const up_name = try std.fmt.allocPrint(arena_allocator, "model.layers.{d}.mlp.up_proj.weight", .{i});
        try tensors_meta.append(arena_allocator, .{ .name = up_name, .shape = .{ ffn_dim, dim }, .size_bytes = ffn_dim * dim * @sizeOf(f32) });

        const down_name = try std.fmt.allocPrint(arena_allocator, "model.layers.{d}.mlp.down_proj.weight", .{i});
        try tensors_meta.append(arena_allocator, .{ .name = down_name, .shape = .{ dim, ffn_dim }, .size_bytes = dim * ffn_dim * @sizeOf(f32) });
    }

    try tensors_meta.append(arena_allocator, .{
        .name = "model.norm.weight",
        .shape = .{ dim, 1 },
        .size_bytes = dim * @sizeOf(f32),
    });

    try tensors_meta.append(arena_allocator, .{
        .name = "lm_head.weight",
        .shape = .{ vocab, dim },
        .size_bytes = vocab * dim * @sizeOf(f32),
    });

    // Build JSON Header
    var json_buf: std.ArrayList(u8) = .empty;
    defer json_buf.deinit(arena_allocator);

    try json_buf.appendSlice(arena_allocator, "{\n");
    var current_offset: usize = 0;
    for (tensors_meta.items, 0..) |t, idx| {
        const next_offset = current_offset + t.size_bytes;
        const entry_str = try std.fmt.allocPrint(arena_allocator,
            \\  "{s}": {{"dtype": "F32", "shape": [{d}, {d}], "data_offsets": [{d}, {d}]}}{s}
            \\
        , .{
            t.name,
            t.shape[0],
            t.shape[1],
            current_offset,
            next_offset,
            if (idx + 1 < tensors_meta.items.len) "," else "",
        });
        try json_buf.appendSlice(arena_allocator, entry_str);
        current_offset = next_offset;
    }
    try json_buf.appendSlice(arena_allocator, "}");

    // Pad JSON with spaces to satisfy 32-byte alignment for tensors
    const total_header_len = 8 + json_buf.items.len;
    const align_boundary: usize = 32;
    const pad = (align_boundary - (total_header_len % align_boundary)) % align_boundary;
    for (0..pad) |_| {
        try json_buf.append(arena_allocator, ' ');
    }

    // Write 8-byte header length
    const header_len: u64 = json_buf.items.len;
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header_len, .little);
    try writeAllBytes(st_fd, &len_buf);

    // Write JSON header
    try writeAllBytes(st_fd, json_buf.items);

    // Write random tensor weights
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    for (tensors_meta.items) |t| {
        const num_floats = t.size_bytes / @sizeOf(f32);
        const float_buf = try arena_allocator.alloc(f32, num_floats);

        const scale: f32 = 0.02;
        for (float_buf) |*v| {
            v.* = (rand.float(f32) - 0.5) * scale;
        }

        if (std.mem.endsWith(u8, t.name, "norm.weight")) {
            for (float_buf) |*v| v.* = 1.0;
        }

        try writeAllBytes(st_fd, std.mem.sliceAsBytes(float_buf));
    }
}

test "generate and load sample GGUF" {
    const allocator = std.testing.allocator;
    const sample_path = "/tmp/test_sample.gguf";
    defer _ = std.posix.system.unlink(sample_path);

    try generateSampleGGUF(allocator, sample_path, .{});

    var gguf_file = try gguf.GGUFFile.open(allocator, sample_path);
    defer gguf_file.deinit();

    try std.testing.expectEqual(types.Architecture.llama, gguf_file.params.arch);
    try std.testing.expectEqual(@as(usize, 2), gguf_file.params.block_count);
    try std.testing.expectEqual(@as(usize, 64), gguf_file.params.embedding_length);
}

test "generate and load sample SafeTensors" {
    const allocator = std.testing.allocator;
    const sample_dir = "/tmp/test_safetensors_model";
    defer {
        _ = std.posix.system.unlink("/tmp/test_safetensors_model/config.json");
        _ = std.posix.system.unlink("/tmp/test_safetensors_model/tokenizer.json");
        _ = std.posix.system.unlink("/tmp/test_safetensors_model/model.safetensors");
        _ = std.posix.system.rmdir("/tmp/test_safetensors_model");
    }

    try generateSampleSafeTensors(allocator, sample_dir, .{});

    const safetensors = @import("safetensors.zig");
    var st_file = try safetensors.SafeTensorsFile.open(allocator, sample_dir);
    defer st_file.deinit();

    try std.testing.expect(st_file.params != null);
    try std.testing.expectEqual(@as(usize, 2), st_file.params.?.block_count);
    try std.testing.expectEqual(@as(usize, 64), st_file.params.?.embedding_length);

    const embd = st_file.getTensor("token_embd.weight");
    try std.testing.expect(embd != null);
}
