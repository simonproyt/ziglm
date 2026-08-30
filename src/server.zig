const std = @import("std");
const Engine = @import("engine.zig").Engine;
const ChatMessage = @import("tokenizer.zig").ChatMessage;
const GenerationOptions = @import("types.zig").GenerationOptions;

pub const ServerConfig = struct {
    port: u16 = 8080,
    host: []const u8 = "127.0.0.1",
};

pub const Server = struct {
    allocator: std.mem.Allocator,
    engine: *Engine,
    config: ServerConfig,
    server_fd: std.posix.fd_t = 0,
    running: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    pub fn init(allocator: std.mem.Allocator, engine: *Engine, config: ServerConfig) *Server {
        const self = allocator.create(Server) catch unreachable;
        self.* = .{
            .allocator = allocator,
            .engine = engine,
            .config = config,
        };
        return self;
    }

    pub fn deinit(self: *Server) void {
        self.stop();
        self.allocator.destroy(self);
    }

    pub fn stop(self: *Server) void {
        if (self.running.swap(false, .release)) {
            if (self.server_fd > 0) {
                _ = std.posix.system.close(self.server_fd);
                self.server_fd = 0;
            }
        }
    }

    fn writeSocket(fd: std.posix.fd_t, bytes: []const u8) !void {
        var written: usize = 0;
        while (written < bytes.len) {
            const rc = std.posix.system.write(fd, bytes[written..].ptr, bytes.len - written);
            const err = std.posix.errno(rc);
            if (err != .SUCCESS) return error.SocketWriteFailed;
            written += @as(usize, @intCast(rc));
        }
    }

    pub fn listenAndServe(self: *Server) !void {
        // Create TCP socket
        const rc = std.posix.system.socket(
            std.posix.AF.INET,
            std.posix.SOCK.STREAM,
            std.posix.IPPROTO.TCP,
        );
        const err = std.posix.errno(rc);
        if (err != .SUCCESS) return error.SocketCreateFailed;
        const sock_fd: std.posix.fd_t = @intCast(rc);
        errdefer _ = std.posix.system.close(sock_fd);

        // Allow SO_REUSEADDR
        const enable: u32 = 1;
        _ = std.posix.system.setsockopt(
            sock_fd,
            std.posix.SOL.SOCKET,
            std.posix.SO.REUSEADDR,
            @ptrCast(&enable),
            @sizeOf(u32),
        );

        // Bind address
        var addr: std.posix.sockaddr.in = .{
            .family = std.posix.AF.INET,
            .port = std.mem.nativeToBig(u16, self.config.port),
            .addr = 0x0100007F, // 127.0.0.1 in network byte order
        };

        const bind_rc = std.posix.system.bind(
            sock_fd,
            @ptrCast(&addr),
            @sizeOf(std.posix.sockaddr.in),
        );
        if (std.posix.errno(bind_rc) != .SUCCESS) return error.BindFailed;

        const listen_rc = std.posix.system.listen(sock_fd, 128);
        if (std.posix.errno(listen_rc) != .SUCCESS) return error.ListenFailed;

        self.server_fd = sock_fd;
        self.running.store(true, .release);

        std.debug.print("\n🚀 ziglm HTTP Server listening on http://{s}:{d}\n", .{ self.config.host, self.config.port });
        std.debug.print("Endpoints:\n", .{});
        std.debug.print("  GET  /health\n", .{});
        std.debug.print("  GET  /v1/models\n", .{});
        std.debug.print("  POST /v1/chat/completions (OpenAI Compatible)\n", .{});
        std.debug.print("  POST /v1/completions\n\n", .{});

        var req_buffer: [65536]u8 = undefined;

        while (self.running.load(.acquire)) {
            var client_addr: std.posix.sockaddr.in = undefined;
            var addr_len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr.in);

            const client_rc = std.posix.system.accept(
                sock_fd,
                @ptrCast(&client_addr),
                &addr_len,
            );
            if (std.posix.errno(client_rc) != .SUCCESS) {
                if (!self.running.load(.acquire)) break;
                continue;
            }
            const client_fd: std.posix.fd_t = @intCast(client_rc);
            defer _ = std.posix.system.close(client_fd);

            const n_read = std.posix.read(client_fd, &req_buffer) catch continue;
            if (n_read == 0) continue;

            const req_str = req_buffer[0..n_read];
            self.handleRequest(client_fd, req_str) catch |req_err| {
                std.debug.print("Error handling client request: {}\n", .{req_err});
            };
        }
    }

    fn handleRequest(self: *Server, client_fd: std.posix.fd_t, request: []const u8) !void {
        // Parse HTTP line
        var line_iter = std.mem.splitSequence(u8, request, "\r\n");
        const first_line = line_iter.next() orelse return;

        var part_iter = std.mem.splitScalar(u8, first_line, ' ');
        const method = part_iter.next() orelse return;
        const path = part_iter.next() orelse return;

        // Health check
        if (std.mem.eql(u8, method, "GET") and (std.mem.eql(u8, path, "/health") or std.mem.eql(u8, path, "/"))) {
            const body = "{\"status\":\"ok\",\"engine\":\"ziglm\",\"version\":\"1.0\"}";
            try self.sendJsonResponse(client_fd, 200, body);
            return;
        }

        // Models list
        if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/v1/models")) {
            const body = "{\"object\":\"list\",\"data\":[{\"id\":\"ziglm-model\",\"object\":\"model\",\"created\":1700000000,\"owned_by\":\"ziglm\"}]}";
            try self.sendJsonResponse(client_fd, 200, body);
            return;
        }

        // Chat completions
        if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/chat/completions")) {
            // Find body (after \r\n\r\n)
            const body_start_idx = std.mem.indexOf(u8, request, "\r\n\r\n");
            if (body_start_idx == null) {
                try self.sendJsonResponse(client_fd, 400, "{\"error\":\"Missing HTTP body\"}");
                return;
            }
            const body_str = request[body_start_idx.? + 4 ..];

            try self.handleChatCompletions(client_fd, body_str);
            return;
        }

        // Fallback 404
        try self.sendJsonResponse(client_fd, 404, "{\"error\":\"Not Found\"}");
    }

    fn sendJsonResponse(self: *Server, client_fd: std.posix.fd_t, status: u16, body: []const u8) !void {
        _ = self;
        var header_buf: [512]u8 = undefined;
        const status_text = if (status == 200) "OK" else if (status == 400) "Bad Request" else "Not Found";
        const header = try std.fmt.bufPrint(&header_buf, "HTTP/1.1 {d} {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", .{
            status,
            status_text,
            body.len,
        });

        try writeSocket(client_fd, header);
        try writeSocket(client_fd, body);
    }

    fn handleChatCompletions(self: *Server, client_fd: std.posix.fd_t, body_json: []const u8) !void {
        // Parse stream flag, temperature, max_tokens, and messages
        const is_stream = std.mem.indexOf(u8, body_json, "\"stream\":true") != null or
            std.mem.indexOf(u8, body_json, "\"stream\": true") != null;

        var max_tokens: usize = 256;
        if (std.mem.indexOf(u8, body_json, "\"max_tokens\":")) |idx| {
            const sub = body_json[idx + 13 ..];
            var end: usize = 0;
            while (end < sub.len and (sub[end] >= '0' and sub[end] <= '9')) : (end += 1) {}
            if (end > 0) {
                max_tokens = std.fmt.parseInt(usize, sub[0..end], 10) catch 256;
            }
        }

        var temperature: f32 = 0.7;
        if (std.mem.indexOf(u8, body_json, "\"temperature\":")) |idx| {
            const sub = body_json[idx + 14 ..];
            var end: usize = 0;
            while (end < sub.len and ((sub[end] >= '0' and sub[end] <= '9') or sub[end] == '.')) : (end += 1) {}
            if (end > 0) {
                temperature = std.fmt.parseFloat(f32, sub[0..end]) catch 0.7;
            }
        }

        // Simple prompt extraction from message content
        var prompt_buf: [4096]u8 = undefined;
        var prompt_len: usize = 0;

        if (std.mem.indexOf(u8, body_json, "\"content\":")) |idx| {
            const sub = body_json[idx + 10 ..];
            if (std.mem.indexOf(u8, sub, "\"")) |q1| {
                const after_q1 = sub[q1 + 1 ..];
                if (std.mem.indexOf(u8, after_q1, "\"")) |q2| {
                    const content = after_q1[0..q2];
                    @memcpy(prompt_buf[0..content.len], content);
                    prompt_len = content.len;
                }
            }
        }

        const prompt = if (prompt_len > 0) prompt_buf[0..prompt_len] else "Hello";

        const options = GenerationOptions{
            .max_tokens = max_tokens,
            .sampler = .{ .temperature = temperature },
        };

        if (is_stream) {
            // Stream SSE response
            const sse_headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
            try writeSocket(client_fd, sse_headers);

            const StreamContext = struct {
                fd: std.posix.fd_t,
                chunk_buf: [1024]u8,
            };
            var s_ctx = StreamContext{ .fd = client_fd, .chunk_buf = undefined };

            const streamCallback = struct {
                fn send(ctx_ptr: ?*anyopaque, token_str: []const u8, _: u32) bool {
                    const ctx: *StreamContext = @ptrCast(@alignCast(ctx_ptr.?));
                    var escaped_buf: [512]u8 = undefined;
                    var esc_len: usize = 0;
                    for (token_str) |c| {
                        if (c == '"') {
                            escaped_buf[esc_len] = '\\';
                            escaped_buf[esc_len + 1] = '"';
                            esc_len += 2;
                        } else if (c == '\n') {
                            escaped_buf[esc_len] = '\\';
                            escaped_buf[esc_len + 1] = 'n';
                            esc_len += 2;
                        } else {
                            escaped_buf[esc_len] = c;
                            esc_len += 1;
                        }
                    }
                    const sse_line = std.fmt.bufPrint(
                        &ctx.chunk_buf,
                        "data: {{\"id\":\"chatcmpl-zig\",\"object\":\"chat.completion.chunk\",\"choices\":[{{\"delta\":{{\"content\":\"{s}\"}}}}]}}\n\n",
                        .{escaped_buf[0..esc_len]},
                    ) catch return false;

                    writeSocket(ctx.fd, sse_line) catch return false;
                    return true;
                }
            }.send;

            _ = try self.engine.generate(prompt, options, &s_ctx, streamCallback);
            try writeSocket(client_fd, "data: [DONE]\n\n");
        } else {
            // Collect full response
            var response_buf: [16384]u8 = undefined;
            var response_len: usize = 0;

            const CollectContext = struct {
                buf: []u8,
                len: *usize,
            };
            var c_ctx = CollectContext{ .buf = &response_buf, .len = &response_len };

            const collectCallback = struct {
                fn collect(ctx_ptr: ?*anyopaque, token_str: []const u8, _: u32) bool {
                    const ctx: *CollectContext = @ptrCast(@alignCast(ctx_ptr.?));
                    if (ctx.len.* + token_str.len <= ctx.buf.len) {
                        @memcpy(ctx.buf[ctx.len.* .. ctx.len.* + token_str.len], token_str);
                        ctx.len.* += token_str.len;
                    }
                    return true;
                }
            }.collect;

            const stats = try self.engine.generate(prompt, options, &c_ctx, collectCallback);

            var json_buf: [32768]u8 = undefined;
            const json_resp = try std.fmt.bufPrint(&json_buf,
                \\{{"id":"chatcmpl-zig","object":"chat.completion","created":1700000000,"model":"ziglm-model","choices":[{{"index":0,"message":{{"role":"assistant","content":"{s}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
            , .{
                response_buf[0..response_len],
                stats.prompt_tokens,
                stats.completion_tokens,
                stats.prompt_tokens + stats.completion_tokens,
            });

            try self.sendJsonResponse(client_fd, 200, json_resp);
        }
    }
};
