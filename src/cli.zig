const std = @import("std");
const types = @import("types.zig");
const GenerationOptions = types.GenerationOptions;
const Engine = @import("engine.zig").Engine;
const Server = @import("server.zig").Server;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const ChatMessage = @import("tokenizer.zig").ChatMessage;
const synthetic = @import("synthetic.zig");
const GGUFFile = @import("gguf.zig").GGUFFile;

pub fn printUsage() void {
    std.debug.print(
        \\
        \\  ⚡ ziglm - High-Performance LLM Inference Engine in Pure Zig ⚡
        \\
        \\  USAGE:
        \\    ziglm <subcommand> [options]
        \\
        \\  SUBCOMMANDS:
        \\    run          Run one-shot prompt inference
        \\    chat         Start interactive multi-turn chat session
        \\    serve        Start OpenAI-compatible HTTP REST API server
        \\    bench        Run performance and throughput benchmark
        \\    info         Inspect GGUF model metadata and tensors
        \\    test-model   Generate a sample GGUF model for testing
        \\
        \\  GENERAL OPTIONS:
        \\    -m, --model <path>      Path to GGUF model file
        \\    -p, --prompt <text>     Input prompt for text generation
        \\    -t, --temp <float>      Sampling temperature (default: 0.7, 0 = greedy)
        \\        --top-p <float>     Nucleus sampling top-p (default: 0.9)
        \\        --top-k <int>       Top-K sampling (default: 40)
        \\        --min-p <float>     Min-P sampling (default: 0.05)
        \\    -n, --max-tokens <int>  Maximum tokens to generate (default: 256)
        \\    -j, --threads <int>     Number of worker threads (default: CPU cores)
        \\        --image <path>      Path to input image (PNG, JPG, BMP, PPM) for vision
        \\        --video <path>      Path to input video (MP4, MKV, frames) for video reasoning
        \\        --audio <path>      Path to input audio (WAV 16kHz) for speech/audio reasoning
        \\        --port <int>        HTTP server port (default: 8080)
        \\        --seed <int>        Random seed for reproducibility (default: 42)
        \\        --system <text>     System prompt for chat mode
        \\    -h, --help              Print this help menu
        \\
        \\  EXAMPLES:
        \\    ziglm test-model --create-sample model.gguf
        \\    ziglm run -m model.gguf -p "Hello world"
        \\    ziglm run -m model.gguf --video clip.mp4 -p "Describe the video action."
        \\    ziglm run -m model.gguf --audio speech.wav -p "Transcribe what you hear."
        \\    ziglm chat -m model.gguf
        \\    ziglm serve -m model.gguf --port 8080
        \\
        \\
    , .{});
}

pub const CliArgs = struct {
    subcommand: []const u8 = "help",
    model_path: ?[]const u8 = null,
    prompt: []const u8 = "Hello! Who are you?",
    system_prompt: ?[]const u8 = null,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    top_k: usize = 40,
    min_p: f32 = 0.05,
    max_tokens: usize = 256,
    threads: ?usize = null,
    image_path: ?[]const u8 = null,
    video_path: ?[]const u8 = null,
    audio_path: ?[]const u8 = null,
    mmproj_path: ?[]const u8 = null,
    port: u16 = 8080,
    seed: u64 = 42,
    sample_out: []const u8 = "sample_model.gguf",
};

pub fn parseArgsFromIterator(arg_it: *std.process.Args.Iterator) CliArgs {
    var args = CliArgs{};
    _ = arg_it.skip(); // skip binary name

    if (arg_it.next()) |sub| {
        args.subcommand = sub;
    } else {
        return args;
    }

    while (arg_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            args.model_path = arg_it.next();
        } else if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--prompt")) {
            if (arg_it.next()) |p| args.prompt = p;
        } else if (std.mem.eql(u8, arg, "--system")) {
            args.system_prompt = arg_it.next();
        } else if (std.mem.eql(u8, arg, "-t") or std.mem.eql(u8, arg, "--temp")) {
            if (arg_it.next()) |v| args.temperature = std.fmt.parseFloat(f32, v) catch 0.7;
        } else if (std.mem.eql(u8, arg, "--top-p")) {
            if (arg_it.next()) |v| args.top_p = std.fmt.parseFloat(f32, v) catch 0.9;
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            if (arg_it.next()) |v| args.top_k = std.fmt.parseInt(usize, v, 10) catch 40;
        } else if (std.mem.eql(u8, arg, "--min-p")) {
            if (arg_it.next()) |v| args.min_p = std.fmt.parseFloat(f32, v) catch 0.05;
        } else if (std.mem.eql(u8, arg, "-n") or std.mem.eql(u8, arg, "--max-tokens")) {
            if (arg_it.next()) |v| args.max_tokens = std.fmt.parseInt(usize, v, 10) catch 256;
        } else if (std.mem.eql(u8, arg, "-j") or std.mem.eql(u8, arg, "--threads")) {
            if (arg_it.next()) |v| args.threads = std.fmt.parseInt(usize, v, 10) catch null;
        } else if (std.mem.eql(u8, arg, "--greedy")) {
            args.temperature = 0.0;
        } else if (std.mem.eql(u8, arg, "--image")) {
            args.image_path = arg_it.next();
        } else if (std.mem.eql(u8, arg, "--video")) {
            args.video_path = arg_it.next();
        } else if (std.mem.eql(u8, arg, "--audio")) {
            args.audio_path = arg_it.next();
        } else if (std.mem.eql(u8, arg, "--mmproj")) {
            args.mmproj_path = arg_it.next();
        } else if (std.mem.eql(u8, arg, "--port")) {
            if (arg_it.next()) |v| args.port = std.fmt.parseInt(u16, v, 10) catch 8080;
        } else if (std.mem.eql(u8, arg, "--seed")) {
            if (arg_it.next()) |v| args.seed = std.fmt.parseInt(u64, v, 10) catch 42;
        } else if (std.mem.eql(u8, arg, "--create-sample")) {
            if (arg_it.next()) |v| args.sample_out = v;
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            args.subcommand = "help";
        }
    }

    return args;
}

fn unescapePrompt(allocator: std.mem.Allocator, input: []const u8) []const u8 {
    if (std.mem.indexOf(u8, input, "\\n") == null and std.mem.indexOf(u8, input, "\\t") == null) {
        return input;
    }
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < input.len) {
        if (input[i] == '\\' and i + 1 < input.len) {
            if (input[i + 1] == 'n') {
                out.append(allocator, '\n') catch return input;
                i += 2;
                continue;
            } else if (input[i + 1] == 't') {
                out.append(allocator, '\t') catch return input;
                i += 2;
                continue;
            } else if (input[i + 1] == '\\') {
                out.append(allocator, '\\') catch return input;
                i += 2;
                continue;
            }
        }
        out.append(allocator, input[i]) catch return input;
        i += 1;
    }
    return out.toOwnedSlice(allocator) catch input;
}

fn printTokenStdout(_: ?*anyopaque, token_str: []const u8, _: u32) bool {
    const stdout = std.posix.STDOUT_FILENO;
    _ = std.posix.system.write(stdout, token_str.ptr, token_str.len);
    return true;
}

pub fn runCli(allocator: std.mem.Allocator, args: CliArgs) !void {
    if (std.mem.eql(u8, args.subcommand, "help")) {
        printUsage();
        return;
    }

    // Subcommand: test-model
    if (std.mem.eql(u8, args.subcommand, "test-model")) {
        std.debug.print("Generating sample GGUF test model: {s} ...\n", .{args.sample_out});
        try synthetic.generateSampleGGUF(allocator, args.sample_out, .{});
        std.debug.print("✓ Successfully generated sample model: {s}\n", .{args.sample_out});
        return;
    }

    // Model path check
    const model_path = args.model_path orelse {
        std.debug.print("Error: --model <path.gguf> argument required.\n", .{});
        printUsage();
        return;
    };

    // Subcommand: info
    if (std.mem.eql(u8, args.subcommand, "info")) {
        std.debug.print("Loading model metadata: {s} ...\n", .{model_path});

        var is_safetensors = std.mem.endsWith(u8, model_path, ".safetensors");
        if (!is_safetensors) {
            const test_dfd = std.posix.openat(std.posix.AT.FDCWD, model_path, .{ .ACCMODE = .RDONLY, .DIRECTORY = true }, 0) catch null;
            if (test_dfd) |dfd| {
                _ = std.posix.system.close(dfd);
                is_safetensors = true;
            }
        }

        if (is_safetensors) {
            const safetensors = @import("safetensors.zig");
            var st_file = try safetensors.SafeTensorsFile.open(allocator, model_path);
            defer st_file.deinit();

            std.debug.print("\n=== Model Format: SafeTensors ===\n", .{});
            if (st_file.params) |p| {
                std.debug.print("  Architecture:        {s}\n", .{p.arch.asString()});
                std.debug.print("  Context Length:      {d}\n", .{p.context_length});
                std.debug.print("  Embedding Length:    {d}\n", .{p.embedding_length});
                std.debug.print("  Feed Forward Length: {d}\n", .{p.feed_forward_length});
                std.debug.print("  Layers (Blocks):     {d}\n", .{p.block_count});
                std.debug.print("  Attention Heads:     {d}\n", .{p.head_count});
                std.debug.print("  KV Heads (GQA):      {d}\n", .{p.head_count_kv});
                std.debug.print("  Head Size:           {d}\n", .{p.head_size});
                std.debug.print("  Vocab Size:          {d}\n", .{p.vocab_size});
                std.debug.print("  RMS Epsilon:         {e}\n", .{p.layer_norm_rms_epsilon});
                std.debug.print("  RoPE Base Freq:      {d:.1}\n", .{p.rope_freq_base});
            }

            std.debug.print("\n=== Tensors ({d} total) ===\n", .{st_file.tensors.count()});
            var t_it = st_file.tensors.valueIterator();
            var count: usize = 0;
            while (t_it.next()) |t| {
                if (count < 15) {
                    std.debug.print("  {s:<36} {s:<8} [{d}, {d}] ({d} bytes)\n", .{
                        t.name,
                        t.type.name(),
                        t.shape[0],
                        t.shape[1],
                        t.sizeBytes(),
                    });
                }
                count += 1;
            }
            if (count > 15) {
                std.debug.print("  ... and {d} more tensors\n", .{count - 15});
            }
            return;
        }

        var gguf_file = try GGUFFile.open(allocator, model_path);
        defer gguf_file.deinit();

        std.debug.print("\n=== Model Architecture (GGUF) ===\n", .{});
        std.debug.print("  Architecture:        {s}\n", .{gguf_file.params.arch.asString()});
        std.debug.print("  Context Length:      {d}\n", .{gguf_file.params.context_length});
        std.debug.print("  Embedding Length:    {d}\n", .{gguf_file.params.embedding_length});
        std.debug.print("  Feed Forward Length: {d}\n", .{gguf_file.params.feed_forward_length});
        std.debug.print("  Layers (Blocks):     {d}\n", .{gguf_file.params.block_count});
        std.debug.print("  Attention Heads:     {d}\n", .{gguf_file.params.head_count});
        std.debug.print("  KV Heads (GQA):      {d}\n", .{gguf_file.params.head_count_kv});
        std.debug.print("  Head Size:           {d}\n", .{gguf_file.params.head_size});
        std.debug.print("  Vocab Size:          {d}\n", .{gguf_file.params.vocab_size});
        std.debug.print("  RMS Epsilon:         {e}\n", .{gguf_file.params.layer_norm_rms_epsilon});
        std.debug.print("  RoPE Base Freq:      {d:.1}\n", .{gguf_file.params.rope_freq_base});

        std.debug.print("\n=== Tensors ({d} total) ===\n", .{gguf_file.tensors.len});
        var a_count: usize = 0;
        var v_count: usize = 0;
        var l_count: usize = 0;
        for (gguf_file.tensors) |t| {
            if (std.mem.indexOf(u8, t.name, "blk") == null) {
                std.debug.print("  [GLOBAL] {s:<42} {s:<8} (id={d}) [{d}, {d}] ({d} bytes)\n", .{ t.name, t.type.name(), @intFromEnum(t.type), t.shape[0], t.shape[1], t.sizeBytes() });
            }
            if (std.mem.startsWith(u8, t.name, "a.")) {
                a_count += 1;
            } else if (std.mem.startsWith(u8, t.name, "v.") or std.mem.startsWith(u8, t.name, "mm.") or std.mem.startsWith(u8, t.name, "patch_embd")) {
                v_count += 1;
            } else {
                l_count += 1;
            }
        }
        std.debug.print("\nSummary: {d} Audio tensors, {d} Vision tensors, {d} Language tensors\n", .{ a_count, v_count, l_count });
        if (gguf_file.tensors.len > 15) {
            std.debug.print("  ... and {d} more tensors\n", .{gguf_file.tensors.len - 15});
        }
        return;
    }

    // Subcommand: run
    if (std.mem.eql(u8, args.subcommand, "run")) {
        std.debug.print("Loading model: {s} ...\n", .{model_path});
        var engine = try Engine.load(allocator, model_path, .{
            .num_threads = args.threads,
            .seed = args.seed,
            .mmproj_path = args.mmproj_path,
        });
        defer engine.deinit();

        std.debug.print("\nPrompt: {s}\n\n", .{args.prompt});

        const options = GenerationOptions{
            .max_tokens = args.max_tokens,
            .sampler = .{
                .temperature = args.temperature,
                .top_p = args.top_p,
                .top_k = args.top_k,
                .min_p = args.min_p,
                .seed = args.seed,
            },
        };

        const clean_prompt = unescapePrompt(allocator, args.prompt);

        var stats: types.GenerationStats = undefined;
        if (args.video_path) |vid| {
            std.debug.print("Processing multimodal video: {s} ...\n", .{vid});
            stats = try engine.generateWithVideo(clean_prompt, vid, options, null, printTokenStdout);
        } else if (args.audio_path) |aud| {
            std.debug.print("Processing multimodal audio: {s} ...\n", .{aud});
            stats = try engine.generateWithAudio(clean_prompt, aud, options, null, printTokenStdout);
        } else {
            if (args.image_path) |img| {
                std.debug.print("Processing multimodal image: {s} ...\n", .{img});
            }
            stats = try engine.generateWithImage(clean_prompt, args.image_path, options, null, printTokenStdout);
        }

        std.debug.print("\n\n────────────────────────────────────────\n", .{});
        std.debug.print("⚡ Prefill:    {d:.1} tok/s ({d} tokens in {d:.1} ms)\n", .{
            stats.prefillTokensPerSec(),
            stats.prompt_tokens,
            stats.prefill_time_ms,
        });
        std.debug.print("⚡ Generation: {d:.1} tok/s ({d} tokens in {d:.1} ms)\n", .{
            stats.generationTokensPerSec(),
            stats.completion_tokens,
            stats.generation_time_ms,
        });
        std.debug.print("⚡ Total time: {d:.1} ms\n", .{stats.total_time_ms});
        return;
    }

    // Subcommand: chat
    if (std.mem.eql(u8, args.subcommand, "chat")) {
        std.debug.print("Loading model for chat: {s} ...\n", .{model_path});
        var engine = try Engine.load(allocator, model_path, .{
            .num_threads = args.threads,
            .seed = args.seed,
            .mmproj_path = args.mmproj_path,
        });
        defer engine.deinit();

        std.debug.print("\n💬 Interactive Chat Session Started (/reset to clear, /exit to quit)\n", .{});

        var messages: std.ArrayList(ChatMessage) = .empty;
        defer messages.deinit(allocator);

        if (args.system_prompt) |sys| {
            try messages.append(allocator, .{ .role = "system", .content = sys });
        }

        var line_buf: [4096]u8 = undefined;

        while (true) {
            std.debug.print("\n\x1b[1;34mUser > \x1b[0m", .{});

            var line_len: usize = 0;
            while (line_len < line_buf.len) {
                const n = std.posix.read(std.posix.STDIN_FILENO, line_buf[line_len .. line_len + 1]) catch break;
                if (n == 0) break;
                if (line_buf[line_len] == '\n') break;
                line_len += 1;
            }
            if (line_len == 0) break;

            const user_input = std.mem.trim(u8, line_buf[0..line_len], " \r\t\n");
            if (std.mem.eql(u8, user_input, "/exit") or std.mem.eql(u8, user_input, "/quit")) break;
            if (std.mem.eql(u8, user_input, "/reset")) {
                messages.clearRetainingCapacity();
                engine.reset();
                std.debug.print("✓ Conversation history cleared.\n", .{});
                continue;
            }

            const user_dup = try allocator.dupe(u8, user_input);
            try messages.append(allocator, .{ .role = "user", .content = user_dup });

            std.debug.print("\x1b[1;32mAssistant > \x1b[0m", .{});

            const options = GenerationOptions{
                .max_tokens = args.max_tokens,
                .sampler = .{
                    .temperature = args.temperature,
                    .top_p = args.top_p,
                    .top_k = args.top_k,
                    .min_p = args.min_p,
                    .seed = args.seed,
                },
            };

            var assistant_response: std.ArrayList(u8) = .empty;
            defer assistant_response.deinit(allocator);

            const CollectChatContext = struct {
                list: *std.ArrayList(u8),
                alloc: std.mem.Allocator,
            };
            var cc_ctx = CollectChatContext{ .list = &assistant_response, .alloc = allocator };

            const chatCallback = struct {
                fn run(ctx_ptr: ?*anyopaque, token_str: []const u8, _: u32) bool {
                    const ctx: *CollectChatContext = @ptrCast(@alignCast(ctx_ptr.?));
                    const stdout = std.posix.STDOUT_FILENO;
                    _ = std.posix.system.write(stdout, token_str.ptr, token_str.len);
                    ctx.list.appendSlice(ctx.alloc, token_str) catch {};
                    return true;
                }
            }.run;

            _ = try engine.chat(messages.items, options, &cc_ctx, chatCallback);
            std.debug.print("\n", .{});

            const assist_dup = try assistant_response.toOwnedSlice(allocator);
            try messages.append(allocator, .{ .role = "assistant", .content = assist_dup });
        }
        return;
    }

    // Subcommand: serve
    if (std.mem.eql(u8, args.subcommand, "serve")) {
        var engine = try Engine.load(allocator, model_path, .{
            .num_threads = args.threads,
            .seed = args.seed,
            .mmproj_path = args.mmproj_path,
        });
        defer engine.deinit();

        var server = Server.init(allocator, engine, .{ .port = args.port });
        defer server.deinit();

        try server.listenAndServe();
        return;
    }

    // Subcommand: bench
    if (std.mem.eql(u8, args.subcommand, "bench")) {
        std.debug.print("Running benchmark on: {s} ...\n", .{model_path});
        var engine = try Engine.load(allocator, model_path, .{
            .num_threads = args.threads,
            .seed = args.seed,
            .mmproj_path = args.mmproj_path,
        });
        defer engine.deinit();

        const bench_prompt = "The quick brown fox jumps over the lazy dog and explores the universe with fast mathematical computations";
        const options = GenerationOptions{
            .max_tokens = 32,
            .sampler = .{ .greedy = true },
        };

        std.debug.print("Running 3 warmup & benchmark iterations ...\n", .{});
        for (0..3) |iter| {
            const stats = try engine.generate(bench_prompt, options, null, null);
            std.debug.print("  Iteration {d}: Prefill = {d:.1} tok/s | Generation = {d:.1} tok/s | Total = {d:.1} ms\n", .{
                iter + 1,
                stats.prefillTokensPerSec(),
                stats.generationTokensPerSec(),
                stats.total_time_ms,
            });
        }
        return;
    }

    std.debug.print("Unknown subcommand: {s}\n", .{args.subcommand});
    printUsage();
}
