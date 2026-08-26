const std = @import("std");
const GGUFFile = @import("gguf.zig").GGUFFile;

pub const TokenType = enum {
    normal,
    unknown,
    control,
    user_defined,
    unused,
    byte,
};

pub const Token = struct {
    id: u32,
    text: []const u8,
    score: f32 = 0.0,
    token_type: TokenType = .normal,
};

pub const TokenizerModelType = enum {
    bpe,
    spm, // SentencePiece
    wpm, // WordPiece
};

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    model_type: TokenizerModelType,
    tokens: []Token,
    token_to_id: std.StringHashMap(u32),
    merges: std.StringHashMap(u32), // merge pair -> rank
    bos_token_id: ?u32 = null,
    eos_token_id: ?u32 = null,
    unk_token_id: ?u32 = null,
    pad_token_id: ?u32 = null,
    owns_strings: bool = false,

    threadlocal var decode_buf: [4096]u8 = undefined;

    pub fn loadFromGGUF(allocator: std.mem.Allocator, gguf: *const GGUFFile) !*Tokenizer {
        const tok = try allocator.create(Tokenizer);
        tok.* = .{
            .allocator = allocator,
            .model_type = .bpe,
            .tokens = &[_]Token{},
            .token_to_id = std.StringHashMap(u32).init(allocator),
            .merges = std.StringHashMap(u32).init(allocator),
            .owns_strings = false,
        };
        errdefer tok.deinit();

        // Detect tokenizer model type
        if (gguf.getString("tokenizer.ggml.model")) |model_name| {
            if (std.mem.eql(u8, model_name, "llama") or std.mem.eql(u8, model_name, "spm")) {
                tok.model_type = .spm;
            } else if (std.mem.eql(u8, model_name, "gpt2") or std.mem.eql(u8, model_name, "bpe")) {
                tok.model_type = .bpe;
            }
        }

        // Special tokens
        tok.bos_token_id = gguf.getU32("tokenizer.ggml.bos_token_id");
        tok.eos_token_id = gguf.getU32("tokenizer.ggml.eos_token_id");
        tok.unk_token_id = gguf.getU32("tokenizer.ggml.unknown_token_id");
        tok.pad_token_id = gguf.getU32("tokenizer.ggml.padding_token_id");

        // Parse tokens array
        const tokens_val = gguf.getMetadata("tokenizer.ggml.tokens") orelse return error.MissingTokenizerTokens;
        if (tokens_val != .ARRAY or tokens_val.ARRAY.type != .STRING) {
            return error.InvalidTokenizerTokensMetadata;
        }

        const arr = tokens_val.ARRAY;
        tok.tokens = try allocator.alloc(Token, arr.len);

        // Read scores if available
        var scores_data: ?[]const u8 = null;
        if (gguf.getMetadata("tokenizer.ggml.scores")) |val| {
            if (val == .ARRAY and val.ARRAY.type == .FLOAT32 and val.ARRAY.len == arr.len) {
                scores_data = val.ARRAY.data;
            }
        }

        // Read token types if available
        var tok_types_data: ?[]const u8 = null;
        if (gguf.getMetadata("tokenizer.ggml.token_type")) |val| {
            if (val == .ARRAY and val.ARRAY.type == .INT32 and val.ARRAY.len == arr.len) {
                tok_types_data = val.ARRAY.data;
            }
        }

        var offset: usize = 0;
        for (0..arr.len) |i| {
            if (offset + 8 > arr.data.len) return error.UnexpectedEOF;
            const str_len = std.mem.readInt(u64, arr.data[offset..][0..8], .little);
            offset += 8;

            if (offset + str_len > arr.data.len) return error.UnexpectedEOF;
            const str = arr.data[offset .. offset + str_len];
            offset += str_len;

            var score: f32 = 0.0;
            if (scores_data) |sd| {
                if (i * 4 + 4 <= sd.len) {
                    const u = std.mem.readInt(u32, sd[i * 4 ..][0..4], .little);
                    score = @bitCast(u);
                }
            }

            var tok_type: TokenType = .normal;
            if (tok_types_data) |ttd| {
                if (i * 4 + 4 <= ttd.len) {
                    const t_val = std.mem.readInt(i32, ttd[i * 4 ..][0..4], .little);
                    tok_type = switch (t_val) {
                        1 => .normal,
                        2 => .unknown,
                        3 => .control,
                        4 => .user_defined,
                        5 => .unused,
                        6 => .byte,
                        else => .normal,
                    };
                }
            }

            tok.tokens[i] = .{
                .id = @intCast(i),
                .text = str,
                .score = score,
                .token_type = tok_type,
            };

            try tok.token_to_id.put(str, @intCast(i));
        }

        // Parse merges array if available
        if (gguf.getMetadata("tokenizer.ggml.merges")) |val| {
            if (val == .ARRAY and val.ARRAY.type == .STRING) {
                const merges_arr = val.ARRAY;
                var m_offset: usize = 0;
                for (0..merges_arr.len) |i| {
                    if (m_offset + 8 > merges_arr.data.len) break;
                    const str_len = std.mem.readInt(u64, merges_arr.data[m_offset..][0..8], .little);
                    m_offset += 8;
                    const str = merges_arr.data[m_offset .. m_offset + str_len];
                    m_offset += str_len;

                    try tok.merges.put(str, @intCast(i));
                }
            }
        }

        return tok;
    }

    pub fn loadFromHFJson(allocator: std.mem.Allocator, tokenizer_json_path: []const u8) !*Tokenizer {
        const fd = try std.posix.openat(std.posix.AT.FDCWD, tokenizer_json_path, .{ .ACCMODE = .RDONLY }, 0);
        defer _ = std.posix.system.close(fd);

        const end_pos = std.posix.system.lseek(fd, 0, 2);
        if (end_pos < 0) return error.SeekFailed;
        _ = std.posix.system.lseek(fd, 0, 0);
        const file_size: usize = @intCast(end_pos);

        const content = try allocator.alloc(u8, file_size);
        defer allocator.free(content);

        var total_read: usize = 0;
        while (total_read < file_size) {
            const n = try std.posix.read(fd, content[total_read..]);
            if (n == 0) break;
            total_read += n;
        }

        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{ .allocate = .alloc_always });
        defer parsed.deinit();

        if (parsed.value != .object) return error.InvalidTokenizerJson;
        const root = parsed.value.object;

        const tok = try allocator.create(Tokenizer);
        tok.* = .{
            .allocator = allocator,
            .model_type = .bpe,
            .tokens = &[_]Token{},
            .token_to_id = std.StringHashMap(u32).init(allocator),
            .merges = std.StringHashMap(u32).init(allocator),
            .owns_strings = true,
        };
        errdefer tok.deinit();

        // Model vocab & merges
        if (root.get("model")) |m_val| {
            if (m_val == .object) {
                const m_obj = m_val.object;
                if (m_obj.get("type")) |t_val| {
                    if (t_val == .string and std.mem.eql(u8, t_val.string, "BPE")) {
                        tok.model_type = .bpe;
                    }
                }

                if (m_obj.get("vocab")) |v_val| {
                    if (v_val == .object) {
                        const vocab_obj = v_val.object;
                        var max_id: usize = 0;
                        var it = vocab_obj.iterator();
                        while (it.next()) |entry| {
                            if (entry.value_ptr.* == .integer) {
                                const id: usize = @intCast(entry.value_ptr.*.integer);
                                if (id > max_id) max_id = id;
                            }
                        }

                        tok.tokens = try allocator.alloc(Token, max_id + 1);
                        for (tok.tokens, 0..) |*t, i| {
                            t.* = .{
                                .id = @intCast(i),
                                .text = "",
                                .score = 0.0,
                            };
                        }

                        var it2 = vocab_obj.iterator();
                        while (it2.next()) |entry| {
                            if (entry.value_ptr.* == .integer) {
                                const id: usize = @intCast(entry.value_ptr.*.integer);
                                const token_str = try allocator.dupe(u8, entry.key_ptr.*);
                                tok.tokens[id] = .{
                                    .id = @intCast(id),
                                    .text = token_str,
                                    .score = 0.0,
                                };
                                try tok.token_to_id.put(token_str, @intCast(id));
                            }
                        }
                    }
                }

                if (m_obj.get("merges")) |merges_val| {
                    if (merges_val == .array) {
                        for (merges_val.array.items, 0..) |item, i| {
                            if (item == .string) {
                                const m_str = try allocator.dupe(u8, item.string);
                                try tok.merges.put(m_str, @intCast(i));
                            } else if (item == .array and item.array.items.len == 2) {
                                const p1 = item.array.items[0];
                                const p2 = item.array.items[1];
                                if (p1 == .string and p2 == .string) {
                                    const m_str = try std.fmt.allocPrint(allocator, "{s} {s}", .{ p1.string, p2.string });
                                    try tok.merges.put(m_str, @intCast(i));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Added tokens
        if (root.get("added_tokens")) |added_val| {
            if (added_val == .array) {
                for (added_val.array.items) |item| {
                    if (item == .object) {
                        const obj = item.object;
                        const id_v = obj.get("id");
                        const content_v = obj.get("content");
                        if (id_v != null and content_v != null and id_v.? == .integer and content_v.? == .string) {
                            const id: usize = @intCast(id_v.?.integer);
                            const str = content_v.?.string;
                            if (id < tok.tokens.len) {
                                if (tok.tokens[id].text.len == 0) {
                                    const duped = try allocator.dupe(u8, str);
                                    tok.tokens[id].text = duped;
                                    try tok.token_to_id.put(duped, @intCast(id));
                                }
                            }
                            if (std.mem.indexOf(u8, str, "bos") != null or std.mem.eql(u8, str, "<s>") or std.mem.eql(u8, str, "<|begin_of_text|>")) {
                                tok.bos_token_id = @intCast(id);
                            }
                            if (std.mem.indexOf(u8, str, "eos") != null or std.mem.eql(u8, str, "</s>") or std.mem.eql(u8, str, "<|end_of_text|>") or std.mem.eql(u8, str, "<|eot_id|>") or std.mem.eql(u8, str, "<turn|>")) {
                                tok.eos_token_id = @intCast(id);
                            }
                        }
                    }
                }
            }
        }

        // Detect SPM model type if vocab contains SentencePiece space " "
        if (tok.token_to_id.get(" ") != null or tok.token_to_id.get(" user") != null or tok.token_to_id.get(" What") != null) {
            tok.model_type = .spm;
        }

        return tok;
    }

    pub fn deinit(self: *Tokenizer) void {
        if (self.owns_strings) {
            for (self.tokens) |t| {
                if (t.text.len > 0) {
                    self.allocator.free(t.text);
                }
            }
            var m_it = self.merges.iterator();
            while (m_it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
            }
        }
        self.token_to_id.deinit();
        self.merges.deinit();
        if (self.tokens.len > 0) {
            self.allocator.free(self.tokens);
        }
        self.allocator.destroy(self);
    }

    pub fn encode(self: *const Tokenizer, caller_allocator: std.mem.Allocator, text: []const u8, add_bos: bool) ![]u32 {
        var arena = std.heap.ArenaAllocator.init(caller_allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        var tokens_list: std.ArrayList(u32) = .empty;

        if (add_bos) {
            if (self.bos_token_id) |bos| {
                try tokens_list.append(allocator, bos);
            }
        }

        if (text.len == 0) {
            return caller_allocator.dupe(u32, tokens_list.items);
        }

        // Check exact match first
        if (self.token_to_id.get(text)) |id| {
            try tokens_list.append(allocator, id);
            return caller_allocator.dupe(u32, tokens_list.items);
        }

        const special_tokens = [_][]const u8{
            "<bos>",
            "<eos>",
            "<|turn>",
            "<turn|>",
            "<|channel>",
            "<channel|>",
            "<start_of_turn>",
            "<end_of_turn>",
            "<|tool>",
            "<tool|>",
            "<|tool_call>",
            "<tool_call|>",
            "<|tool_response>",
            "<tool_response|>",
        };

        var text_idx: usize = 0;
        while (text_idx < text.len) {
            // Check for special tokens at current position
            var matched_special: ?[]const u8 = null;
            for (special_tokens) |st| {
                if (std.mem.startsWith(u8, text[text_idx..], st)) {
                    matched_special = st;
                    break;
                }
            }

            if (matched_special) |st| {
                if (self.token_to_id.get(st)) |id| {
                    try tokens_list.append(allocator, id);
                }
                text_idx += st.len;
                continue;
            }

            // Find next special token
            var next_special_idx = text.len;
            for (special_tokens) |st| {
                if (std.mem.indexOf(u8, text[text_idx..], st)) |pos| {
                    if (text_idx + pos < next_special_idx) {
                        next_special_idx = text_idx + pos;
                    }
                }
            }

            const chunk = text[text_idx..next_special_idx];
            text_idx = next_special_idx;

            if (chunk.len == 0) continue;

            // BPE encoding with byte fallback
            var pieces: std.ArrayList([]const u8) = .empty;
            defer pieces.deinit(allocator);

            // Initial piece generation
            const is_spm = (self.model_type == .spm or self.token_to_id.get("\xe2\x96\x81") != null or self.token_to_id.get("\xe2\x96\x81user") != null or self.token_to_id.get(" ") != null);
            if (std.unicode.Utf8View.init(chunk)) |utf8_view| {
                var it = utf8_view.iterator();
                while (it.nextCodepointSlice()) |cp_slice| {
                    if (std.mem.eql(u8, cp_slice, " ")) {
                        if (is_spm) {
                            try pieces.append(allocator, "\xe2\x96\x81");
                        } else {
                            try pieces.append(allocator, "Ġ");
                        }
                    } else if (std.mem.eql(u8, cp_slice, "\n")) {
                        if (is_spm) {
                            try pieces.append(allocator, "\n");
                        } else {
                            try pieces.append(allocator, "Ċ");
                        }
                    } else {
                        try pieces.append(allocator, cp_slice);
                    }
                }
            } else |_| {
                for (0..chunk.len) |idx| {
                    const b = chunk[idx];
                    if (b == ' ') {
                        if (is_spm) {
                            try pieces.append(allocator, "\xe2\x96\x81");
                        } else {
                            try pieces.append(allocator, "Ġ");
                        }
                    } else if (b == '\n') {
                        if (is_spm) {
                            try pieces.append(allocator, "\n");
                        } else {
                            try pieces.append(allocator, "Ċ");
                        }
                    } else {
                        try pieces.append(allocator, chunk[idx .. idx + 1]);
                    }
                }
            }

            // Iterative merge loop based on merge ranks
            if (self.merges.count() > 0) {
                var merge_buf: [512]u8 = undefined;

                while (pieces.items.len >= 2) {
                    var best_rank: u32 = std.math.maxInt(u32);
                    var best_idx: ?usize = null;

                    for (0..pieces.items.len - 1) |j| {
                        const pair = std.fmt.bufPrint(&merge_buf, "{s} {s}", .{
                            pieces.items[j],
                            pieces.items[j + 1],
                        }) catch continue;

                        if (self.merges.get(pair)) |rank| {
                            if (rank < best_rank) {
                                best_rank = rank;
                                best_idx = j;
                            }
                        }
                    }

                    if (best_idx) |idx| {
                        const merged = try std.fmt.allocPrint(allocator, "{s}{s}", .{
                            pieces.items[idx],
                            pieces.items[idx + 1],
                        });
                        pieces.items[idx] = merged;
                        _ = pieces.orderedRemove(idx + 1);
                    } else {
                        break;
                    }
                }
            }

            // Map final pieces to token IDs
            for (pieces.items) |piece| {
                if (self.token_to_id.get(piece)) |id| {
                    try tokens_list.append(allocator, id);
                } else {
                    for (piece) |b| {
                        var byte_buf: [16]u8 = undefined;
                        const byte_repr = std.fmt.bufPrint(&byte_buf, "<0x{X:0>2}>", .{b}) catch "";
                        if (self.token_to_id.get(byte_repr)) |id| {
                            try tokens_list.append(allocator, id);
                        } else if (self.unk_token_id) |unk| {
                            try tokens_list.append(allocator, unk);
                        }
                    }
                }
            }
        }

        return caller_allocator.dupe(u32, tokens_list.items);
    }

    pub fn decode(self: *const Tokenizer, token_id: u32) []const u8 {
        if (token_id >= self.tokens.len) return "";
        const raw_text = self.tokens[token_id].text;

        // Handle byte token format <0xXX>
        if (raw_text.len == 6 and std.mem.startsWith(u8, raw_text, "<0x") and raw_text[5] == '>') {
            if (std.fmt.parseInt(u8, raw_text[3..5], 16)) |b| {
                decode_buf[0] = b;
                return decode_buf[0..1];
            } else |_| {}
        }

        var out_idx: usize = 0;
        var i: usize = 0;
        while (i < raw_text.len and out_idx + 4 < decode_buf.len) {
            if (i + 3 <= raw_text.len and raw_text[i] == 0xe2 and raw_text[i + 1] == 0x96 and raw_text[i + 2] == 0x81) {
                decode_buf[out_idx] = ' ';
                out_idx += 1;
                i += 3;
            } else if (i + 2 <= raw_text.len and raw_text[i] == 0xc4 and raw_text[i + 1] == 0xa0) {
                decode_buf[out_idx] = ' ';
                out_idx += 1;
                i += 2;
            } else if (i + 2 <= raw_text.len and raw_text[i] == 0xc4 and raw_text[i + 1] == 0x8a) {
                decode_buf[out_idx] = '\n';
                out_idx += 1;
                i += 2;
            } else {
                decode_buf[out_idx] = raw_text[i];
                out_idx += 1;
                i += 1;
            }
        }

        return decode_buf[0..out_idx];
    }

    pub fn applyChatTemplate(
        self: *const Tokenizer,
        allocator: std.mem.Allocator,
        messages: []const ChatMessage,
        template_type_opt: ?ChatTemplateType,
    ) ![]u8 {
        const template = template_type_opt orelse ChatTemplateType.detect(self);
        var buffer: std.ArrayList(u8) = .empty;
        errdefer buffer.deinit(allocator);

        switch (template) {
            .llama3 => {
                try buffer.appendSlice(allocator, "<|begin_of_text|>");
                for (messages) |msg| {
                    try buffer.appendSlice(allocator, "<|start_header_id|>");
                    try buffer.appendSlice(allocator, msg.role);
                    try buffer.appendSlice(allocator, "<|end_header_id|>\n\n");
                    try buffer.appendSlice(allocator, msg.content);
                    try buffer.appendSlice(allocator, "<|eot_id|>");
                }
                try buffer.appendSlice(allocator, "<|start_header_id|>assistant<|end_header_id|>\n\n");
            },
            .chatml => {
                for (messages) |msg| {
                    try buffer.appendSlice(allocator, "<|im_start|>");
                    try buffer.appendSlice(allocator, msg.role);
                    try buffer.appendSlice(allocator, "\n");
                    try buffer.appendSlice(allocator, msg.content);
                    try buffer.appendSlice(allocator, "<|im_end|>\n");
                }
                try buffer.appendSlice(allocator, "<|im_start|>assistant\n");
            },
            .mistral => {
                for (messages) |msg| {
                    if (std.mem.eql(u8, msg.role, "user")) {
                        try buffer.appendSlice(allocator, "[INST] ");
                        try buffer.appendSlice(allocator, msg.content);
                        try buffer.appendSlice(allocator, " [/INST]");
                    } else if (std.mem.eql(u8, msg.role, "assistant")) {
                        try buffer.appendSlice(allocator, msg.content);
                    }
                }
            },
            .gemma4 => {
                for (messages) |msg| {
                    try buffer.appendSlice(allocator, "<|turn>");
                    const role = if (std.mem.eql(u8, msg.role, "assistant")) "model" else msg.role;
                    try buffer.appendSlice(allocator, role);
                    try buffer.appendSlice(allocator, "\n");
                    try buffer.appendSlice(allocator, msg.content);
                    try buffer.appendSlice(allocator, "<turn|>\n");
                }
                try buffer.appendSlice(allocator, "<|turn>model\n");
            },
            .gemma => {
                for (messages) |msg| {
                    try buffer.appendSlice(allocator, "<start_of_turn>");
                    try buffer.appendSlice(allocator, msg.role);
                    try buffer.appendSlice(allocator, "\n");
                    try buffer.appendSlice(allocator, msg.content);
                    try buffer.appendSlice(allocator, "<end_of_turn>\n");
                }
                try buffer.appendSlice(allocator, "<start_of_turn>model\n");
            },
            .llama2 => {
                try buffer.appendSlice(allocator, "<s>[INST] ");
                var has_system = false;
                for (messages) |msg| {
                    if (std.mem.eql(u8, msg.role, "system")) {
                        try buffer.appendSlice(allocator, "<<SYS>>\n");
                        try buffer.appendSlice(allocator, msg.content);
                        try buffer.appendSlice(allocator, "\n<</SYS>>\n\n");
                        has_system = true;
                    } else if (std.mem.eql(u8, msg.role, "user")) {
                        try buffer.appendSlice(allocator, msg.content);
                        try buffer.appendSlice(allocator, " [/INST] ");
                    } else if (std.mem.eql(u8, msg.role, "assistant")) {
                        try buffer.appendSlice(allocator, msg.content);
                        try buffer.appendSlice(allocator, " </s><s>[INST] ");
                    }
                }
            },
            .alpaca => {
                for (messages) |msg| {
                    if (std.mem.eql(u8, msg.role, "system")) {
                        try buffer.appendSlice(allocator, msg.content);
                        try buffer.appendSlice(allocator, "\n\n");
                    } else if (std.mem.eql(u8, msg.role, "user")) {
                        try buffer.appendSlice(allocator, "### Instruction:\n");
                        try buffer.appendSlice(allocator, msg.content);
                        try buffer.appendSlice(allocator, "\n\n### Response:\n");
                    }
                }
            },
        }

        return buffer.toOwnedSlice(allocator);
    }
};

pub const ChatMessage = struct {
    role: []const u8,
    content: []const u8,
};

pub const ChatTemplateType = enum {
    llama3,
    chatml,
    llama2,
    mistral,
    gemma,
    gemma4,
    alpaca,

    pub fn detect(tokenizer: *const Tokenizer) ChatTemplateType {
        if (tokenizer.token_to_id.contains("<|turn>")) {
            return .gemma4;
        } else if (tokenizer.token_to_id.contains("<|start_header_id|>")) {
            return .llama3;
        } else if (tokenizer.token_to_id.contains("<|im_start|>")) {
            return .chatml;
        } else if (tokenizer.token_to_id.contains("<start_of_turn>")) {
            return .gemma;
        } else if (tokenizer.token_to_id.contains("[INST]")) {
            return .mistral;
        }
        return .chatml;
    }
};

test "Tokenizer basic encoding and decoding" {
    const allocator = std.testing.allocator;
    var tok = try allocator.create(Tokenizer);
    tok.* = .{
        .allocator = allocator,
        .model_type = .bpe,
        .tokens = try allocator.alloc(Token, 5),
        .token_to_id = std.StringHashMap(u32).init(allocator),
        .merges = std.StringHashMap(u32).init(allocator),
    };
    defer tok.deinit();

    const sample = [_][]const u8{ "<s>", "</s>", "Hello", "world", "!" };
    for (sample, 0..) |s, i| {
        tok.tokens[i] = .{ .id = @intCast(i), .text = s, .score = 0.0 };
        try tok.token_to_id.put(s, @intCast(i));
    }
    tok.bos_token_id = 0;
    tok.eos_token_id = 1;

    const encoded = try tok.encode(allocator, "Hello", true);
    defer allocator.free(encoded);

    try std.testing.expectEqual(@as(usize, 2), encoded.len);
    try std.testing.expectEqual(@as(u32, 0), encoded[0]); // BOS
    try std.testing.expectEqual(@as(u32, 2), encoded[1]); // Hello

    const decoded = tok.decode(2);
    try std.testing.expectEqualStrings("Hello", decoded);
}

test "Tokenizer HuggingFace JSON parser" {
    const allocator = std.testing.allocator;
    const json_sample =
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "<bos>": 2,
        \\      "<|turn>": 105,
        \\      "user": 2364,
        \\      "\n": 107
        \\    },
        \\    "merges": []
        \\  },
        \\  "added_tokens": [
        \\    {"id": 2, "content": "<bos>"}
        \\  ]
        \\}
    ;

    const path = "/tmp/test_tokenizer_sample.json";
    const fd = try std.posix.openat(std.posix.AT.FDCWD, path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
    _ = std.posix.system.write(fd, json_sample.ptr, json_sample.len);
    _ = std.posix.system.close(fd);
    defer _ = std.posix.system.unlink(path);

    const tok = try Tokenizer.loadFromHFJson(allocator, path);
    defer tok.deinit();

    try std.testing.expectEqual(@as(u32, 2), tok.bos_token_id.?);
    try std.testing.expectEqual(@as(u32, 2364), tok.token_to_id.get("user").?);
}
