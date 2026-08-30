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
const vision = @import("vision.zig");
const VisionEncoder = vision.VisionEncoder;
const audio = @import("audio.zig");
const video = @import("video.zig");
const cuda_mod = @import("cuda.zig");
const CudaDevice = cuda_mod.CudaDevice;
const cuda_model = @import("cuda_model.zig");
const CudaGpuModel = cuda_model.CudaGpuModel;

pub const ModelFormat = enum {
    gguf,
    safetensors,
};

pub const EngineOptions = struct {
    num_threads: ?usize = null,
    max_seq_len: ?usize = null,
    seed: u64 = 42,
    mmproj_path: ?[]const u8 = null,
    use_gpu: bool = false,
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
    mmproj_gguf_file: ?*GGUFFile = null,
    safetensors_file: ?*SafeTensorsFile = null,
    tokenizer: *Tokenizer,
    model: *TransformerModel,
    kv_cache: *KVCache,
    buffers: *ModelBuffers,
    sampler: *Sampler,
    thread_pool: *ThreadPool,
    cuda_device: ?*CudaDevice = null,
    gpu_model: ?*CudaGpuModel = null,
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

        // Auto-load companion multimodal projector if present
        var mmproj_to_open: ?[]const u8 = options.mmproj_path;
        var auto_mmproj_buf: [512]u8 = undefined;

        if (mmproj_to_open == null) {
            if (std.mem.lastIndexOfScalar(u8, model_path, '/')) |last_slash| {
                const dir = model_path[0 .. last_slash + 1];
                const cand1 = std.fmt.bufPrint(&auto_mmproj_buf, "{s}gemma-4-E2B-it-mmproj.gguf", .{dir}) catch null;
                if (cand1) |cand| {
                    if (std.posix.openat(std.posix.AT.FDCWD, cand, .{}, 0)) |fd| {
                        _ = std.posix.system.close(fd);
                        mmproj_to_open = cand;
                    } else |_| {}
                }
            }
        }

        var mmproj_gguf_file: ?*GGUFFile = null;
        if (mmproj_to_open) |mm_path| {
            if (GGUFFile.open(allocator, mm_path)) |mm_gf| {
                mmproj_gguf_file = mm_gf;
                model.loadMMPROJ(allocator, mm_gf) catch |err| {
                    std.debug.print("Warning: Failed to load mmproj weights: {any}\n", .{err});
                };
            } else |_| {}
        }

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

        var cuda_device: ?*CudaDevice = null;
        var gpu_model: ?*CudaGpuModel = null;
        if (options.use_gpu) {
            cuda_device = CudaDevice.init(allocator) catch |err| blk: {
                std.debug.print("⚠️  CUDA GPU initialization failed ({s}), falling back to CPU.\n", .{@errorName(err)});
                break :blk null;
            };
            if (cuda_device) |cd| {
                std.debug.print("🚀 CUDA GPU Acceleration active: {s} ({d} MB VRAM)\n", .{ cd.getName(), cd.total_vram_bytes / 1024 / 1024 });
                gpu_model = CudaGpuModel.init(allocator, cd, model) catch |err| blk: {
                    std.debug.print("⚠️  Failed to offload Transformer weights to GPU ({s}), falling back to CPU.\n", .{@errorName(err)});
                    break :blk null;
                };
                if (gpu_model != null) {
                    std.debug.print("⚡ Successfully offloaded all Transformer layers to GPU VRAM!\n", .{});
                }
            }
        }

        const self = try allocator.create(Engine);
        self.* = .{
            .allocator = allocator,
            .format = .gguf,
            .gguf_file = gguf_file,
            .mmproj_gguf_file = mmproj_gguf_file,
            .safetensors_file = null,
            .tokenizer = tokenizer,
            .model = model,
            .kv_cache = kv_cache,
            .buffers = buffers,
            .sampler = sampler,
            .thread_pool = thread_pool,
            .cuda_device = cuda_device,
            .gpu_model = gpu_model,
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

        var cuda_device: ?*CudaDevice = null;
        var gpu_model: ?*CudaGpuModel = null;
        if (options.use_gpu) {
            cuda_device = CudaDevice.init(allocator) catch |err| blk: {
                std.debug.print("⚠️  CUDA GPU initialization failed ({s}), falling back to CPU.\n", .{@errorName(err)});
                break :blk null;
            };
            if (cuda_device) |cd| {
                std.debug.print("🚀 CUDA GPU Acceleration active: {s} ({d} MB VRAM)\n", .{ cd.getName(), cd.total_vram_bytes / 1024 / 1024 });
                gpu_model = CudaGpuModel.init(allocator, cd, model) catch |err| blk: {
                    std.debug.print("⚠️  Failed to offload Transformer weights to GPU ({s}), falling back to CPU.\n", .{@errorName(err)});
                    break :blk null;
                };
                if (gpu_model != null) {
                    std.debug.print("⚡ Successfully offloaded all Transformer layers to GPU VRAM!\n", .{});
                }
            }
        }

        const self = try allocator.create(Engine);
        self.* = .{
            .allocator = allocator,
            .format = .safetensors,
            .gguf_file = null,
            .mmproj_gguf_file = null,
            .safetensors_file = st_file,
            .tokenizer = tokenizer,
            .model = model,
            .kv_cache = kv_cache,
            .buffers = buffers,
            .sampler = sampler,
            .thread_pool = thread_pool,
            .cuda_device = cuda_device,
            .gpu_model = gpu_model,
            .max_seq_len = max_seq,
            .params = params,
        };

        return self;
    }

    pub fn deinit(self: *Engine) void {
        if (self.gpu_model) |gm| gm.deinit();
        if (self.cuda_device) |cd| cd.deinit();
        self.thread_pool.deinit();
        self.sampler.deinit();
        self.buffers.deinit();
        self.kv_cache.deinit();
        self.model.deinit();
        self.tokenizer.deinit();
        if (self.gguf_file) |gf| gf.deinit();
        if (self.mmproj_gguf_file) |mgf| mgf.deinit();
        if (self.safetensors_file) |sf| sf.deinit();
        self.allocator.destroy(self);
    }

    pub fn reset(self: *Engine) void {
        self.kv_cache.reset();
    }

    pub fn forwardTokenOrEmbedding(
        self: *Engine,
        custom_embedding: ?[]const f32,
        token_id: u32,
        pos: usize,
        is_last_token: bool,
    ) ![]const f32 {
        if (self.gpu_model) |gm| {
            return try gm.forward(
                self.model,
                token_id,
                pos,
                self.kv_cache,
                self.buffers,
                custom_embedding,
                is_last_token,
            );
        } else {
            return try self.model.forwardWithEmbedding(
                custom_embedding,
                token_id,
                pos,
                self.kv_cache,
                self.buffers,
                self.thread_pool,
                is_last_token,
            );
        }
    }

    pub fn prefill(self: *Engine, tokens: []const u32) ![]const f32 {
        if (tokens.len == 0) return error.EmptyPrompt;
        if (tokens.len > self.max_seq_len) return error.PromptExceedsContext;

        var last_logits: []const f32 = undefined;
        for (tokens, 0..) |tok, pos| {
            const is_last = (pos == tokens.len - 1);
            last_logits = try self.forwardTokenOrEmbedding(null, tok, pos, is_last);
        }
        return last_logits;
    }

    pub fn decode(self: *Engine, token: u32, pos: usize) ![]const f32 {
        if (pos >= self.max_seq_len) return error.ContextFull;
        return try self.forwardTokenOrEmbedding(null, token, pos, true);
    }

    pub fn encodeImage(self: *Engine, image_path: []const u8) ![]f32 {
        var img = try Image.loadFromFile(self.allocator, image_path);
        defer img.deinit();

        if (self.model.vision_encoder) |*v_enc| {
            return v_enc.encodeImage(self.allocator, &img, self.thread_pool);
        } else {
            const enc = vision.VisionEncoder.init(self.allocator, null, null, null, &[_]vision.VisionLayerWeights{});
            return enc.encodeImage(self.allocator, &img, self.thread_pool);
        }
    }

    pub fn encodeAudio(self: *Engine, audio_path: []const u8) ![]f32 {
        var audio_data = try audio.loadWav(self.allocator, audio_path);
        defer audio_data.deinit(self.allocator);

        var mel_gen = try audio.LogMelSpectrogram.init(self.allocator, 128, 512, 160, 16000);
        defer mel_gen.deinit();

        const spec = try mel_gen.compute(audio_data.samples);
        defer self.allocator.free(spec);

        const dim = self.model.params.embedding_length;
        if (self.model.audio_encoder) |*ae| {
            return try ae.encode(self.allocator, spec, self.thread_pool);
        }

        const n_frames = spec.len / 80;
        const embeddings = try self.allocator.alloc(f32, n_frames * dim);
        @memset(embeddings, 0.0);

        for (0..n_frames) |f| {
            const frame_spec = spec[f * 80 .. (f + 1) * 80];
            const emb_frame = embeddings[f * dim .. (f + 1) * dim];
            for (0..dim) |d| {
                emb_frame[d] = frame_spec[d % 80] * 0.1;
            }
        }

        return embeddings;
    }

    pub fn encodeVideo(self: *Engine, video_path: []const u8, max_frames: usize) !struct { embeddings: []f32, num_frames: usize } {
        const vid = try video.Video.load(self.allocator, video_path, max_frames);
        defer vid.deinit();

        var total_embeddings: std.ArrayList(f32) = .empty;
        errdefer total_embeddings.deinit(self.allocator);

        for (vid.frames) |*frame| {
            const frame_emb = if (self.model.vision_encoder) |*v_enc|
                try v_enc.encodeImageWithTokens(self.allocator, &frame.image, self.thread_pool, 70)
            else blk: {
                const enc = vision.VisionEncoder.init(self.allocator, null, null, null, &[_]vision.VisionLayerWeights{});
                break :blk try enc.encodeImageWithTokens(self.allocator, &frame.image, self.thread_pool, 70);
            };
            defer self.allocator.free(frame_emb);
            try total_embeddings.appendSlice(self.allocator, frame_emb);
        }

        const embs = try total_embeddings.toOwnedSlice(self.allocator);
        return .{
            .embeddings = embs,
            .num_frames = vid.frames.len,
        };
    }

    pub const PrefillResult = struct {
        logits: []const f32,
        pos: usize,
    };

    pub fn generateWithMediaEmbeddings(
        self: *Engine,
        prompt: []const u8,
        embeddings: ?[]const f32,
        item_count: usize,
        kind: types.MultimodalKind,
        num_frames: usize,
    ) !PrefillResult {
        const tokens = try self.tokenizer.encode(self.allocator, prompt, true);
        defer self.allocator.free(tokens);

        if (tokens.len == 0) return error.EmptyPrompt;

        var placeholder_pos: ?usize = null;
        for (tokens, 0..) |tok, idx| {
            const s = self.tokenizer.decode(tok);
            switch (kind) {
                .image => {
                    if (tok == 258880 or tok == 255999 or std.mem.eql(u8, s, "<|image|>") or std.mem.eql(u8, s, "<|image>")) {
                        placeholder_pos = idx;
                        break;
                    }
                },
                .audio => {
                    if (tok == 258881 or tok == 256000 or std.mem.eql(u8, s, "<|audio|>") or std.mem.eql(u8, s, "<|audio>")) {
                        placeholder_pos = idx;
                        break;
                    }
                },
                .video => {
                    if (tok == 258884 or tok == 258880 or tok == 255999 or std.mem.eql(u8, s, "<|video|>") or std.mem.eql(u8, s, "<|image|>")) {
                        placeholder_pos = idx;
                        break;
                    }
                },
            }
        }

        const dim = self.model.params.embedding_length;
        var last_logits: []const f32 = undefined;
        var pos: usize = 0;
        var inserted = false;

        const total_tokens_est = tokens.len + item_count + num_frames * 2 + 10;
        if (total_tokens_est > self.max_seq_len) return error.PromptExceedsContext;

        for (tokens, 0..) |tok, idx| {
            if (embeddings != null and !inserted and (placeholder_pos == null or idx == placeholder_pos.?)) {
                const emb = embeddings.?;
                switch (kind) {
                    .image => {
                        // 1. Beginning of Image: <|image> (255999)
                        last_logits = try self.forwardTokenOrEmbedding(null, 255999, pos, false);
                        pos += 1;

                        // 2. Image patch embeddings (258880)
                        for (0..item_count) |p| {
                            const patch_emb = emb[p * dim .. (p + 1) * dim];
                            last_logits = try self.forwardTokenOrEmbedding(patch_emb, 258880, pos, false);
                            pos += 1;
                        }

                        // 3. End of Image: <image|> (258882)
                        last_logits = try self.forwardTokenOrEmbedding(null, 258882, pos, false);
                        pos += 1;
                    },
                    .audio => {
                        // 1. Beginning of Audio: <|audio> (256000)
                        last_logits = try self.forwardTokenOrEmbedding(null, 256000, pos, false);
                        pos += 1;

                        // 2. Audio frame embeddings (258881)
                        for (0..item_count) |f| {
                            const frame_emb = emb[f * dim .. (f + 1) * dim];
                            last_logits = try self.forwardTokenOrEmbedding(frame_emb, 258881, pos, false);
                            pos += 1;
                        }

                        // 3. End of Audio: <audio|> (258883)
                        last_logits = try self.forwardTokenOrEmbedding(null, 258883, pos, false);
                        pos += 1;
                    },
                    .video => {
                        const patches_per_frame = if (num_frames > 0) item_count / num_frames else item_count;
                        for (0..num_frames) |f_idx| {
                            // Frame timestamp: e.g. "00:00 "
                            const ts_min = (f_idx * 2) / 60;
                            const ts_sec = (f_idx * 2) % 60;
                            var ts_buf: [32]u8 = undefined;
                            const ts_str = std.fmt.bufPrint(&ts_buf, "{d:0>2}:{d:0>2} ", .{ ts_min, ts_sec }) catch "00:00 ";
                            const ts_toks = self.tokenizer.encode(self.allocator, ts_str, false) catch &[_]u32{};
                            defer if (ts_toks.len > 0) self.allocator.free(ts_toks);
                            for (ts_toks) |t_tok| {
                                last_logits = try self.forwardTokenOrEmbedding(null, t_tok, pos, false);
                                pos += 1;
                            }

                            // Frame start: <|image> (255999)
                            last_logits = try self.forwardTokenOrEmbedding(null, 255999, pos, false);
                            pos += 1;

                            for (0..patches_per_frame) |p| {
                                const p_idx = f_idx * patches_per_frame + p;
                                if (p_idx < item_count) {
                                    const patch_emb = emb[p_idx * dim .. (p_idx + 1) * dim];
                                    last_logits = try self.forwardTokenOrEmbedding(patch_emb, 258884, pos, false);
                                    pos += 1;
                                }
                            }

                            // Frame end: <image|> (258882)
                            last_logits = try self.forwardTokenOrEmbedding(null, 258882, pos, false);
                            pos += 1;
                        }
                    },
                }

                inserted = true;
                if (placeholder_pos != null) {
                    continue; // Skip placeholder token
                }
            }

            const is_last = (idx == tokens.len - 1);
            last_logits = try self.forwardTokenOrEmbedding(
                null,
                tok,
                pos,
                is_last,
            );
            pos += 1;
        }

        return PrefillResult{
            .logits = last_logits,
            .pos = pos,
        };
    }

    pub fn generate(
        self: *Engine,
        prompt: []const u8,
        options: GenerationOptions,
        callback_ctx: ?*anyopaque,
        callback: ?*const fn (ctx: ?*anyopaque, token_str: []const u8, token_id: u32) bool,
    ) !GenerationStats {
        return self.generateWithImage(prompt, null, options, callback_ctx, callback);
    }

    pub fn generateWithImage(
        self: *Engine,
        prompt: []const u8,
        image_path: ?[]const u8,
        options: GenerationOptions,
        callback_ctx: ?*anyopaque,
        callback: ?*const fn (ctx: ?*anyopaque, token_str: []const u8, token_id: u32) bool,
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

        // 1. Tokenize prompt with turn wrapper if needed
        const formatted_prompt = if (std.mem.indexOf(u8, prompt, "<|turn>") == null and std.mem.indexOf(u8, prompt, "<start_of_turn>") == null)
            (if (image_path != null)
                try std.fmt.allocPrint(self.allocator, "<|turn>user\n<|image|>{s}<turn|>\n<|turn>model\n", .{prompt})
            else
                try std.fmt.allocPrint(self.allocator, "<|turn>user\n{s}<turn|>\n<|turn>model\n", .{prompt}))
        else
            try self.allocator.dupe(u8, prompt);
        defer self.allocator.free(formatted_prompt);

        const prompt_tokens = try self.tokenizer.encode(self.allocator, formatted_prompt, true);
        defer self.allocator.free(prompt_tokens);

        stats.prompt_tokens = prompt_tokens.len + image_patches;
        if (stats.prompt_tokens == 0) return stats;

        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(self.allocator);
        try history.appendSlice(self.allocator, prompt_tokens);

        self.reset();

        // 2. Prefill phase
        const t_prefill_start = getTimestampNs();
        const prefill_res = try self.generateWithMediaEmbeddings(formatted_prompt, image_embeddings, image_patches, .image, 1);
        var logits = prefill_res.logits;
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
        var cur_pos = prefill_res.pos;
        var completion_count: usize = 0;

        var prev_token = self.sampler.sample(logits, history.items, options.sampler);
        if (self.tokenizer.isEosToken(prev_token)) {
            const t_now = getTimestampNs();
            stats.completion_tokens = 0;
            stats.generation_time_ms = @as(f64, @floatFromInt(t_now - t_gen_start)) / 1_000_000.0;
            stats.total_time_ms = @as(f64, @floatFromInt(t_now - t_start)) / 1_000_000.0;
            return stats;
        }

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

        while (completion_count < options.max_tokens and cur_pos < self.max_seq_len) {
            logits = try self.decode(prev_token, cur_pos);
            cur_pos += 1;

            const next_token = self.sampler.sample(logits, history.items, options.sampler);
            if (self.tokenizer.isEosToken(next_token)) break;

            var is_stop = false;
            for (options.stop_tokens) |stop_id| {
                if (next_token == stop_id) {
                    is_stop = true;
                    break;
                }
            }
            if (is_stop) break;

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

    pub fn generateWithVideo(
        self: *Engine,
        prompt: []const u8,
        video_path: ?[]const u8,
        max_frames: usize,
        options: GenerationOptions,
        callback_ctx: ?*anyopaque,
        callback: ?*const fn (ctx: ?*anyopaque, token_str: []const u8, token_id: u32) bool,
    ) !GenerationStats {
        if (video_path == null) return self.generate(prompt, options, callback_ctx, callback);

        var stats = GenerationStats{};
        const t_start = getTimestampNs();
        const dim = self.model.params.embedding_length;

        const vid_enc = try self.encodeVideo(video_path.?, if (max_frames > 0) max_frames else 2);
        const video_embeddings = vid_enc.embeddings;
        const video_patches = vid_enc.embeddings.len / dim;
        defer self.allocator.free(video_embeddings);

        const formatted_prompt = if (std.mem.indexOf(u8, prompt, "<|turn>") == null and std.mem.indexOf(u8, prompt, "<start_of_turn>") == null)
            try std.fmt.allocPrint(self.allocator, "<|turn>user\n<|video|>{s}<turn|>\n<|turn>model\n", .{prompt})
        else
            try self.allocator.dupe(u8, prompt);
        defer self.allocator.free(formatted_prompt);

        const prompt_tokens = try self.tokenizer.encode(self.allocator, formatted_prompt, true);
        defer self.allocator.free(prompt_tokens);

        stats.prompt_tokens = prompt_tokens.len + video_patches;
        if (stats.prompt_tokens == 0) return stats;

        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(self.allocator);
        try history.appendSlice(self.allocator, prompt_tokens);

        self.reset();

        const t_prefill_start = getTimestampNs();
        const prefill_res = try self.generateWithMediaEmbeddings(formatted_prompt, video_embeddings, video_patches, .video, vid_enc.num_frames);
        var logits = prefill_res.logits;
        const t_prefill_end = getTimestampNs();
        stats.prefill_time_ms = @as(f64, @floatFromInt(t_prefill_end - t_prefill_start)) / 1_000_000.0;

        if (options.echo_prompt) {
            if (callback) |cb| {
                for (prompt_tokens) |tok| {
                    const token_str = self.tokenizer.decode(tok);
                    _ = cb(callback_ctx, token_str, tok);
                }
            }
        }

        const t_gen_start = getTimestampNs();
        var cur_pos = prefill_res.pos;
        var completion_count: usize = 0;

        var prev_token = self.sampler.sample(logits, history.items, options.sampler);
        if (self.tokenizer.isEosToken(prev_token)) {
            const t_now = getTimestampNs();
            stats.completion_tokens = 0;
            stats.generation_time_ms = @as(f64, @floatFromInt(t_now - t_gen_start)) / 1_000_000.0;
            stats.total_time_ms = @as(f64, @floatFromInt(t_now - t_start)) / 1_000_000.0;
            return stats;
        }

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

        while (completion_count < options.max_tokens and cur_pos < self.max_seq_len) {
            logits = try self.decode(prev_token, cur_pos);
            cur_pos += 1;

            const next_token = self.sampler.sample(logits, history.items, options.sampler);
            if (self.tokenizer.isEosToken(next_token)) break;

            var is_stop = false;
            for (options.stop_tokens) |stop_id| {
                if (next_token == stop_id) {
                    is_stop = true;
                    break;
                }
            }
            if (is_stop) break;

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

    pub fn generateWithAudio(
        self: *Engine,
        prompt: []const u8,
        audio_path: ?[]const u8,
        options: GenerationOptions,
        callback_ctx: ?*anyopaque,
        callback: ?*const fn (ctx: ?*anyopaque, token_str: []const u8, token_id: u32) bool,
    ) !GenerationStats {
        if (audio_path == null) return self.generate(prompt, options, callback_ctx, callback);

        var stats = GenerationStats{};
        const t_start = getTimestampNs();
        const dim = self.model.params.embedding_length;

        const audio_embeddings = try self.encodeAudio(audio_path.?);
        const audio_frames = audio_embeddings.len / dim;
        defer self.allocator.free(audio_embeddings);

        const formatted_prompt = if (std.mem.indexOf(u8, prompt, "<|turn>") == null and std.mem.indexOf(u8, prompt, "<start_of_turn>") == null)
            try std.fmt.allocPrint(self.allocator, "<|turn>user\n<|audio|>{s}<turn|>\n<|turn>model\n", .{prompt})
        else
            try self.allocator.dupe(u8, prompt);
        defer self.allocator.free(formatted_prompt);

        const prompt_tokens = try self.tokenizer.encode(self.allocator, formatted_prompt, true);
        defer self.allocator.free(prompt_tokens);

        stats.prompt_tokens = prompt_tokens.len + audio_frames;
        if (stats.prompt_tokens == 0) return stats;

        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(self.allocator);
        try history.appendSlice(self.allocator, prompt_tokens);

        self.reset();

        const t_prefill_start = getTimestampNs();
        const prefill_res = try self.generateWithMediaEmbeddings(formatted_prompt, audio_embeddings, audio_frames, .audio, 1);
        var logits = prefill_res.logits;
        const t_prefill_end = getTimestampNs();
        stats.prefill_time_ms = @as(f64, @floatFromInt(t_prefill_end - t_prefill_start)) / 1_000_000.0;

        if (options.echo_prompt) {
            if (callback) |cb| {
                for (prompt_tokens) |tok| {
                    const token_str = self.tokenizer.decode(tok);
                    _ = cb(callback_ctx, token_str, tok);
                }
            }
        }

        const t_gen_start = getTimestampNs();
        var cur_pos = prefill_res.pos;
        var completion_count: usize = 0;

        var prev_token = self.sampler.sample(logits, history.items, options.sampler);
        if (self.tokenizer.isEosToken(prev_token)) {
            const t_now = getTimestampNs();
            stats.completion_tokens = 0;
            stats.generation_time_ms = @as(f64, @floatFromInt(t_now - t_gen_start)) / 1_000_000.0;
            stats.total_time_ms = @as(f64, @floatFromInt(t_now - t_start)) / 1_000_000.0;
            return stats;
        }

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

        while (completion_count < options.max_tokens and cur_pos < self.max_seq_len) {
            logits = try self.decode(prev_token, cur_pos);
            cur_pos += 1;

            const next_token = self.sampler.sample(logits, history.items, options.sampler);
            if (self.tokenizer.isEosToken(next_token)) break;

            var is_stop = false;
            for (options.stop_tokens) |stop_id| {
                if (next_token == stop_id) {
                    is_stop = true;
                    break;
                }
            }
            if (is_stop) break;

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
