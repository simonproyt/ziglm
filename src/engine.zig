const std = @import("std");
const types = @import("types.zig");
const ModelParams = types.ModelParams;
const GenerationOptions = types.GenerationOptions;
const GenerationStats = types.GenerationStats;
const gguf = @import("gguf.zig");
const GGUFFile = gguf.GGUFFile;
const safetensors = @import("safetensors.zig");
const SafeTensorsFile = safetensors.SafeTensorsFile;
const tokenizer_mod = @import("tokenizer.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const ChatMessage = tokenizer_mod.ChatMessage;
const ChatTemplateType = tokenizer_mod.ChatTemplateType;
const model_mod = @import("model.zig");
const TransformerModel = model_mod.TransformerModel;
const ModelBuffers = model_mod.ModelBuffers;
const KVCache = @import("kv_cache.zig").KVCache;
const Sampler = @import("sampler.zig").Sampler;
const ThreadPool = @import("thread_pool.zig").ThreadPool;
const Image = @import("image.zig").Image;
const VisionEncoder = @import("vision.zig").VisionEncoder;

pub const ModelFormat = enum {
    gguf,
    safetensors,
};

pub const EngineOptions = struct {
    num_threads: ?usize = null,
    max_seq_len: ?usize = null,
    seed: u64 = 42,
};

pub const TokenCallback = *const fn (ctx: ?*anyopaque, token_str: []const u8, token_id: u32) bool;

pub inline fn getTimestampNs() i64 {
    var ts: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    return @as(i64, ts.sec) * 1_000_000_000 + @as(i64, ts.nsec);
}

pub const Engine = struct {
    allocator: std.mem.Allocator,
    format: ModelFormat,
    gguf_file: ?*GGUFFile = null,
    safetensors_file: ?*SafeTensorsFile = null,
    tokenizer: *Tokenizer,
    model: *TransformerModel,
    kv_cache: *KVCache,
    buffers: *ModelBuffers,
    sampler: *Sampler,
    thread_pool: *ThreadPool,
    max_seq_len: usize,
    params: ModelParams,

    pub fn load(allocator: std.mem.Allocator, model_path: []const u8, options: EngineOptions) !*Engine {
        var is_safetensors = std.mem.endsWith(u8, model_path, ".safetensors");
        if (!is_safetensors) {
            const test_dfd = std.posix.openat(std.posix.AT.FDCWD, model_path, .{ .ACCMODE = .RDONLY, .DIRECTORY = true }, 0) catch null;
            if (test_dfd) |dfd| {
                _ = std.posix.system.close(dfd);
                is_safetensors = true;
            }
        }

        if (is_safetensors) {
            return loadSafeTensorsEngine(allocator, model_path, options);
        } else {
            return loadGGUFEngine(allocator, model_path, options);
        }
    }

    fn loadGGUFEngine(allocator: std.mem.Allocator, model_path: []const u8, options: EngineOptions) !*Engine {
        const gguf_file = try GGUFFile.open(allocator, model_path);
        errdefer gguf_file.deinit();

        const tokenizer = try Tokenizer.loadFromGGUF(allocator, gguf_file);
        errdefer tokenizer.deinit();

        const model = try TransformerModel.load(allocator, gguf_file);
        errdefer model.deinit();

        const max_seq = options.max_seq_len orelse @min(gguf_file.params.context_length, 4096);
        const max_head_size = @max(512, gguf_file.params.head_size);
        const kv_cache = try KVCache.init(
            allocator,
            gguf_file.params.block_count,
            gguf_file.params.head_count_kv,
            max_head_size,
            max_seq,
        );
        errdefer kv_cache.deinit();

        const buffers = try ModelBuffers.init(allocator, &gguf_file.params, max_seq);
        errdefer buffers.deinit();

        const sampler = try Sampler.init(allocator, gguf_file.params.vocab_size, options.seed);
        errdefer sampler.deinit();

        const thread_pool = try ThreadPool.init(allocator, options.num_threads);
        errdefer thread_pool.deinit();

        const self = try allocator.create(Engine);
        self.* = .{
            .allocator = allocator,
            .format = .gguf,
            .gguf_file = gguf_file,
            .safetensors_file = null,
            .tokenizer = tokenizer,
            .model = model,
            .kv_cache = kv_cache,
            .buffers = buffers,
            .sampler = sampler,
            .thread_pool = thread_pool,
            .max_seq_len = max_seq,
            .params = gguf_file.params,
        };

        return self;
    }

    fn loadSafeTensorsEngine(allocator: std.mem.Allocator, model_path: []const u8, options: EngineOptions) !*Engine {
        const st_file = try SafeTensorsFile.open(allocator, model_path);
        errdefer st_file.deinit();

        var params = st_file.params orelse ModelParams{};
        params.initComputed();

        var is_dir = false;
        const test_dfd = std.posix.openat(std.posix.AT.FDCWD, model_path, .{ .ACCMODE = .RDONLY, .DIRECTORY = true }, 0) catch null;
        if (test_dfd) |dfd| {
            _ = std.posix.system.close(dfd);
            is_dir = true;
        }

        const dir_path: []const u8 = if (is_dir) model_path else (std.fs.path.dirname(model_path) orelse ".");
        var tok_path_buf: [1024]u8 = undefined;
        const tok_json_path = std.fmt.bufPrint(&tok_path_buf, "{s}/tokenizer.json", .{dir_path}) catch null;

        var tokenizer: *Tokenizer = undefined;
        var tok_loaded = false;
        if (tok_json_path) |tp| {
            const tok_test_fd = std.posix.openat(std.posix.AT.FDCWD, tp, .{ .ACCMODE = .RDONLY }, 0) catch null;
            if (tok_test_fd) |tfd| {
                _ = std.posix.system.close(tfd);
                if (Tokenizer.loadFromHFJson(allocator, tp)) |t| {
                    tokenizer = t;
                    tok_loaded = true;
                } else |_| {}
            }
        }

        if (!tok_loaded) {
            tokenizer = try allocator.create(Tokenizer);
            tokenizer.* = .{
                .allocator = allocator,
                .model_type = .bpe,
                .tokens = try allocator.alloc(tokenizer_mod.Token, params.vocab_size),
                .token_to_id = std.StringHashMap(u32).init(allocator),
                .merges = std.StringHashMap(u32).init(allocator),
                .owns_strings = false,
            };
            for (tokenizer.tokens, 0..) |*t, i| {
                t.* = .{ .id = @intCast(i), .text = "", .score = 0.0 };
            }
        }
        errdefer tokenizer.deinit();

        const model = try TransformerModel.loadFromSafeTensors(allocator, params, st_file);
        errdefer model.deinit();

        const max_seq = options.max_seq_len orelse @min(params.context_length, 4096);
        const max_head_size = @max(512, params.head_size);
        const kv_cache = try KVCache.init(
            allocator,
            params.block_count,
            params.head_count_kv,
            max_head_size,
            max_seq,
        );
        errdefer kv_cache.deinit();

        const buffers = try ModelBuffers.init(allocator, &params, max_seq);
        errdefer buffers.deinit();

        const sampler = try Sampler.init(allocator, params.vocab_size, options.seed);
        errdefer sampler.deinit();

        const thread_pool = try ThreadPool.init(allocator, options.num_threads);
        errdefer thread_pool.deinit();

        const self = try allocator.create(Engine);
        self.* = .{
            .allocator = allocator,
            .format = .safetensors,
            .gguf_file = null,
            .safetensors_file = st_file,
            .tokenizer = tokenizer,
            .model = model,
            .kv_cache = kv_cache,
            .buffers = buffers,
            .sampler = sampler,
            .thread_pool = thread_pool,
            .max_seq_len = max_seq,
            .params = params,
        };

        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.thread_pool.deinit();
        self.sampler.deinit();
        self.buffers.deinit();
        self.kv_cache.deinit();
        self.model.deinit();
        self.tokenizer.deinit();
        if (self.gguf_file) |gf| gf.deinit();
        if (self.safetensors_file) |sf| sf.deinit();
        self.allocator.destroy(self);
    }

    pub fn reset(self: *Engine) void {
        self.kv_cache.reset();
    }

    pub fn prefill(self: *Engine, tokens: []const u32) ![]const f32 {
        if (tokens.len == 0) return error.EmptyPrompt;
        if (tokens.len > self.max_seq_len) return error.PromptExceedsContext;

        var last_logits: []const f32 = undefined;
        for (tokens, 0..) |tok, pos| {
            const is_last = (pos == tokens.len - 1);
            last_logits = try self.model.forward(
                tok,
                pos,
                self.kv_cache,
                self.buffers,
                self.thread_pool,
                is_last,
            );
        }
        return last_logits;
    }

    pub fn decode(self: *Engine, token: u32, pos: usize) ![]const f32 {
        if (pos >= self.max_seq_len) return error.ContextFull;
        return try self.model.forward(
            token,
            pos,
            self.kv_cache,
            self.buffers,
            self.thread_pool,
            true,
        );
    }

    pub fn encodeImage(self: *Engine, image_path: []const u8) ![]f32 {
        var img = try Image.loadFromFile(self.allocator, image_path);
        defer img.deinit();

        if (self.model.vision_encoder) |*v_enc| {
            return v_enc.encodeImage(self.allocator, &img, self.thread_pool);
        } else {
            const enc = VisionEncoder.init(self.allocator, null, null, null);
            return enc.encodeImage(self.allocator, &img, self.thread_pool);
        }
    }

    pub fn prefillMultimodal(
        self: *Engine,
        tokens: []const u32,
        image_embeddings: ?[]const f32,
        image_patch_count: usize,
    ) ![]const f32 {
        if (tokens.len == 0 and image_patch_count == 0) return error.EmptyPrompt;
        const total_len = tokens.len + image_patch_count;
        if (total_len > self.max_seq_len) return error.PromptExceedsContext;

        const dim = self.model.params.embedding_length;
        var last_logits: []const f32 = undefined;
        var pos: usize = 0;

        if (image_embeddings) |emb| {
            for (0..image_patch_count) |p| {
                const patch_emb = emb[p * dim .. (p + 1) * dim];
                const is_last = (pos == total_len - 1);
                last_logits = try self.model.forwardWithEmbedding(
                    patch_emb,
                    258880,
                    pos,
                    self.kv_cache,
                    self.buffers,
                    self.thread_pool,
                    is_last,
                );
                pos += 1;
            }
        }

        for (tokens) |tok| {
            const is_last = (pos == total_len - 1);
            last_logits = try self.model.forwardWithEmbedding(
                null,
                tok,
                pos,
                self.kv_cache,
                self.buffers,
                self.thread_pool,
                is_last,
            );
            pos += 1;
        }

        return last_logits;
    }

    pub fn generate(
        self: *Engine,
        prompt: []const u8,
        options: GenerationOptions,
        callback_ctx: ?*anyopaque,
        callback: ?TokenCallback,
    ) !GenerationStats {
        return self.generateWithImage(prompt, null, options, callback_ctx, callback);
    }

    pub fn generateWithImage(
        self: *Engine,
        prompt: []const u8,
        image_path: ?[]const u8,
        options: GenerationOptions,
        callback_ctx: ?*anyopaque,
        callback: ?TokenCallback,
    ) !GenerationStats {
        var stats = GenerationStats{};
        const t_start = getTimestampNs();

        var image_embeddings: ?[]f32 = null;
        var image_patches: usize = 0;
        if (image_path) |img_p| {
            image_embeddings = try self.encodeImage(img_p);
            image_patches = image_embeddings.?.len / self.model.params.embedding_length;
        }
        defer if (image_embeddings) |emb| self.allocator.free(emb);

        // 1. Tokenize prompt
        const prompt_tokens = try self.tokenizer.encode(self.allocator, prompt, true);
        defer self.allocator.free(prompt_tokens);

        stats.prompt_tokens = prompt_tokens.len + image_patches;
        if (stats.prompt_tokens == 0) return stats;

        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(self.allocator);
        try history.appendSlice(self.allocator, prompt_tokens);

        self.reset();

        // 2. Prefill phase
        const t_prefill_start = getTimestampNs();
        var logits = try self.prefillMultimodal(prompt_tokens, image_embeddings, image_patches);
        const t_prefill_end = getTimestampNs();
        stats.prefill_time_ms = @as(f64, @floatFromInt(t_prefill_end - t_prefill_start)) / 1_000_000.0;

        // Optionally echo prompt tokens
        if (options.echo_prompt) {
            if (callback) |cb| {
                for (prompt_tokens) |tok| {
                    const token_str = self.tokenizer.decode(tok);
                    _ = cb(callback_ctx, token_str, tok);
                }
            }
        }

        // 3. Autoregressive Generation phase
        const t_gen_start = getTimestampNs();
        var cur_pos = prompt_tokens.len + image_patches;
        var completion_count: usize = 0;

        var prev_token = self.sampler.sample(logits, history.items, options.sampler);
        try history.append(self.allocator, prev_token);
        completion_count += 1;

        if (callback) |cb| {
            const token_str = self.tokenizer.decode(prev_token);
            if (!cb(callback_ctx, token_str, prev_token)) {
                stats.completion_tokens = completion_count;
                const t_now = getTimestampNs();
                stats.generation_time_ms = @as(f64, @floatFromInt(t_now - t_gen_start)) / 1_000_000.0;
                stats.total_time_ms = @as(f64, @floatFromInt(t_now - t_start)) / 1_000_000.0;
                return stats;
            }
        }

        const eos_id = self.tokenizer.eos_token_id;

        while (completion_count < options.max_tokens and cur_pos < self.max_seq_len) {
            if (eos_id != null and prev_token == eos_id.?) break;

            var is_stop = false;
            for (options.stop_tokens) |stop_id| {
                if (prev_token == stop_id) {
                    is_stop = true;
                    break;
                }
            }
            if (is_stop) break;

            logits = try self.decode(prev_token, cur_pos);
            cur_pos += 1;

            const next_token = self.sampler.sample(logits, history.items, options.sampler);
            try history.append(self.allocator, next_token);
            completion_count += 1;
            prev_token = next_token;

            if (callback) |cb| {
                const token_str = self.tokenizer.decode(next_token);
                if (!cb(callback_ctx, token_str, next_token)) break;
            }
        }

        const t_gen_end = getTimestampNs();
        stats.completion_tokens = completion_count;
        stats.generation_time_ms = @as(f64, @floatFromInt(t_gen_end - t_gen_start)) / 1_000_000.0;
        stats.total_time_ms = @as(f64, @floatFromInt(t_gen_end - t_start)) / 1_000_000.0;

        return stats;
    }

    pub fn chat(
        self: *Engine,
        messages: []const ChatMessage,
        options: GenerationOptions,
        callback_ctx: ?*anyopaque,
        callback: ?TokenCallback,
    ) !GenerationStats {
        const formatted_prompt = try self.tokenizer.applyChatTemplate(self.allocator, messages, null);
        defer self.allocator.free(formatted_prompt);

        return try self.generate(formatted_prompt, options, callback_ctx, callback);
    }
};

test "Engine end-to-end inference on synthetic GGUF and SafeTensors" {
    const allocator = std.testing.allocator;
    const synthetic = @import("synthetic.zig");

    // 1. Test GGUF Engine
    const gguf_path = "/tmp/test_engine_model.gguf";
    defer _ = std.posix.system.unlink(gguf_path);
    try synthetic.generateSampleGGUF(allocator, gguf_path, .{});

    var engine_gguf = try Engine.load(allocator, gguf_path, .{ .max_seq_len = 64 });
    defer engine_gguf.deinit();

    try std.testing.expectEqual(ModelFormat.gguf, engine_gguf.format);

    const stats_gguf = try engine_gguf.generate("Hello world", .{ .max_tokens = 5, .sampler = .{ .greedy = true } }, null, null);
    try std.testing.expect(stats_gguf.prompt_tokens > 0);
    try std.testing.expect(stats_gguf.completion_tokens > 0);

    // 2. Test SafeTensors Engine
    const st_dir = "/tmp/test_engine_st_model";
    defer {
        _ = std.posix.system.unlink("/tmp/test_engine_st_model/config.json");
        _ = std.posix.system.unlink("/tmp/test_engine_st_model/tokenizer.json");
        _ = std.posix.system.unlink("/tmp/test_engine_st_model/model.safetensors");
        _ = std.posix.system.rmdir(st_dir);
    }
    try synthetic.generateSampleSafeTensors(allocator, st_dir, .{});

    var engine_st = try Engine.load(allocator, st_dir, .{ .max_seq_len = 64 });
    defer engine_st.deinit();

    try std.testing.expectEqual(ModelFormat.safetensors, engine_st.format);

    const stats_st = try engine_st.generate("Hello world", .{ .max_tokens = 5, .sampler = .{ .greedy = true } }, null, null);
    try std.testing.expect(stats_st.prompt_tokens > 0);
    try std.testing.expect(stats_st.completion_tokens > 0);
}
