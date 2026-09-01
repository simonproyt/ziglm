const std = @import("std");
const types = @import("types.zig");
const GGMLType = types.GGMLType;
const Tensor = types.Tensor;
const ModelParams = types.ModelParams;
const Architecture = types.Architecture;

pub const SafeTensorsFile = struct {
    allocator: std.mem.Allocator,
    fd: ?std.posix.fd_t = null,
    mmap_data: []const u8,
    is_mmap: bool,
    header_json: []const u8,
    parsed_json: std.json.Parsed(std.json.Value),
    tensors: std.StringHashMap(Tensor),
    params: ?ModelParams = null,

    pub fn open(allocator: std.mem.Allocator, path: []const u8) !*SafeTensorsFile {
        var safetensors_path_buf: [1024]u8 = undefined;

        // Check if path is a directory
        var actual_path = path;
        var dir_path = path;
        const test_dir_fd = std.posix.openat(std.posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY, .DIRECTORY = true }, 0) catch null;
        if (test_dir_fd) |dfd| {
            _ = std.posix.system.close(dfd);
            actual_path = try std.fmt.bufPrint(&safetensors_path_buf, "{s}/model.safetensors", .{path});
            dir_path = path;
        } else {
            if (std.fs.path.dirname(path)) |dp| {
                dir_path = dp;
            } else {
                dir_path = ".";
            }
        }

        const fd = try std.posix.openat(std.posix.AT.FDCWD, actual_path, .{ .ACCMODE = .RDONLY }, 0);
        errdefer _ = std.posix.system.close(fd);

        const end_pos = std.posix.system.lseek(fd, 0, 2); // SEEK_END
        if (end_pos < 0) return error.SeekFailed;
        _ = std.posix.system.lseek(fd, 0, 0); // SEEK_SET
        const file_size: usize = @intCast(end_pos);
        if (file_size < 8) return error.InvalidSafeTensorsFile;

        var is_mmap = true;
        const data = std.posix.mmap(
            null,
            file_size,
            .{ .READ = true },
            .{ .TYPE = .SHARED },
            fd,
            0,
        ) catch {
            is_mmap = false;
            const buf = try allocator.alloc(u8, file_size);
            errdefer allocator.free(buf);

            var total_read: usize = 0;
            while (total_read < file_size) {
                const n = try std.posix.read(fd, buf[total_read..]);
                if (n == 0) break;
                total_read += n;
            }
            return initFromBuffer(allocator, buf, false, null, dir_path);
        };

        return initFromBuffer(allocator, data, is_mmap, fd, dir_path);
    }

    pub fn initFromBuffer(
        allocator: std.mem.Allocator,
        data: []const u8,
        is_mmap: bool,
        fd: ?std.posix.fd_t,
        dir_path: []const u8,
    ) !*SafeTensorsFile {
        const file_size = data.len;
        if (file_size < 8) return error.InvalidSafeTensorsFile;

        // Read 8-byte little-endian header length N
        const header_len_u64 = std.mem.readInt(u64, data[0..8], .little);
        if (header_len_u64 == 0 or header_len_u64 + 8 > file_size) {
            return error.InvalidSafeTensorsHeaderSize;
        }
        const header_len: usize = @intCast(header_len_u64);
        const header_json = data[8 .. 8 + header_len];
        const data_base_offset = 8 + header_len;

        // Parse JSON header
        var parsed = try std.json.parseFromSlice(
            std.json.Value,
            allocator,
            header_json,
            .{ .allocate = .alloc_always },
        );
        errdefer parsed.deinit();

        if (parsed.value != .object) {
            return error.InvalidSafeTensorsJson;
        }

        const self = try allocator.create(SafeTensorsFile);
        self.* = .{
            .allocator = allocator,
            .fd = fd,
            .mmap_data = data,
            .is_mmap = is_mmap,
            .header_json = header_json,
            .parsed_json = parsed,
            .tensors = std.StringHashMap(Tensor).init(allocator),
            .params = null,
        };
        errdefer self.deinit();

        // Index all tensors
        var iter = parsed.value.object.iterator();
        while (iter.next()) |entry| {
            const tensor_name = entry.key_ptr.*;
            if (std.mem.eql(u8, tensor_name, "__metadata__")) continue;

            const t_obj = entry.value_ptr.*;
            if (t_obj != .object) continue;

            // Extract dtype
            const dtype_val = t_obj.object.get("dtype") orelse continue;
            if (dtype_val != .string) continue;
            const dtype = parseDtype(dtype_val.string);

            // Extract shape
            const shape_val = t_obj.object.get("shape") orelse continue;
            if (shape_val != .array) continue;

            var shape: [4]usize = [_]usize{ 1, 1, 1, 1 };
            const n_dims = @min(shape_val.array.items.len, 4);

            for (0..n_dims) |idx| {
                const item = shape_val.array.items[n_dims - 1 - idx];
                if (item == .integer) {
                    shape[idx] = @intCast(item.integer);
                }
            }

            // Extract data_offsets
            const offsets_val = t_obj.object.get("data_offsets") orelse continue;
            if (offsets_val != .array or offsets_val.array.items.len < 2) continue;
            const begin_val = offsets_val.array.items[0];
            const end_val = offsets_val.array.items[1];
            if (begin_val != .integer or end_val != .integer) continue;

            const begin: usize = @intCast(begin_val.integer);
            const end: usize = @intCast(end_val.integer);

            if (data_base_offset + end > file_size or begin > end) {
                return error.InvalidTensorOffsets;
            }

            const raw_slice = data[data_base_offset + begin .. data_base_offset + end];

            const tensor = Tensor{
                .name = tensor_name,
                .type = dtype,
                .n_dims = n_dims,
                .shape = shape,
                .offset = data_base_offset + begin,
                .data = raw_slice,
            };

            try self.tensors.put(tensor_name, tensor);
        }

        // Try reading config.json if located in the same directory
        var config_path_buf: [1024]u8 = undefined;
        const config_path = std.fmt.bufPrint(&config_path_buf, "{s}/config.json", .{dir_path}) catch null;
        if (config_path) |cfg_p| {
            if (parseHFConfig(allocator, cfg_p)) |p| {
                self.params = p;
            } else |_| {}
        }

        // Reconcile and infer model parameters from actual tensors in the file
        var p = self.params orelse ModelParams{};

        if (self.getTensor("output_norm.weight")) |norm| {
            const norm_dim = if (norm.n_dims == 1) norm.shape[0] else @max(norm.shape[0], norm.shape[1]);
            if (norm_dim > 0) {
                p.embedding_length = norm_dim;
            }
        } else if (self.getTensor("blk.0.attn_norm.weight")) |norm| {
            const norm_dim = if (norm.n_dims == 1) norm.shape[0] else @max(norm.shape[0], norm.shape[1]);
            if (norm_dim > 0) {
                p.embedding_length = norm_dim;
            }
        }

        if (self.getTensor("token_embd.weight")) |embd| {
            if (p.embedding_length > 0) {
                p.vocab_size = embd.elements() / p.embedding_length;
            } else {
                p.embedding_length = @min(embd.shape[0], embd.shape[1]);
                p.vocab_size = @max(embd.shape[0], embd.shape[1]);
            }
        }

        // Count actual blocks present in the file
        var max_layer: usize = 0;
        var found_any_layer = false;
        var layer_buf: [128]u8 = undefined;
        for (0..256) |i| {
            const blk_norm = std.fmt.bufPrint(&layer_buf, "blk.{d}.attn_norm.weight", .{i}) catch break;
            if (self.getTensor(blk_norm) != null) {
                max_layer = i + 1;
                found_any_layer = true;
            }
        }
        if (found_any_layer and max_layer > 0) {
            p.block_count = max_layer;
        }

        // Infer FFN length
        if (self.getTensor("blk.0.ffn_gate.weight")) |gate| {
            if (p.embedding_length > 0) {
                p.feed_forward_length = gate.elements() / p.embedding_length;
            }
        } else if (self.getTensor("blk.0.ffn_up.weight")) |up| {
            if (p.embedding_length > 0) {
                p.feed_forward_length = up.elements() / p.embedding_length;
            }
        }

        // Infer head count & head size
        if (self.getTensor("blk.0.attn_q.weight")) |q_t| {
            const q_dim = if (p.embedding_length > 0) q_t.elements() / p.embedding_length else q_t.shape[0];
            if (p.head_size == 0 or q_dim % p.head_size != 0) {
                p.head_size = if (q_dim % 256 == 0 and q_dim / 256 <= 64) 256 else (if (q_dim % 128 == 0) 128 else (if (q_dim % 64 == 0) 64 else 32));
            }
            if (p.head_size > 0) {
                p.head_count = q_dim / p.head_size;
            }
        }

        if (self.getTensor("blk.0.attn_k.weight")) |k_t| {
            const k_dim = if (p.embedding_length > 0) k_t.elements() / p.embedding_length else k_t.shape[0];
            if (p.head_size > 0 and k_dim % p.head_size == 0) {
                p.head_count_kv = k_dim / p.head_size;
            }
        } else {
            p.head_count_kv = p.head_count;
        }

        p.initComputed();
        self.params = p;

        return self;
    }

    pub fn deinit(self: *SafeTensorsFile) void {
        self.tensors.deinit();
        self.parsed_json.deinit();

        if (self.is_mmap) {
            std.posix.munmap(@alignCast(self.mmap_data));
        } else {
            self.allocator.free(self.mmap_data);
        }

        if (self.fd) |f| {
            _ = std.posix.system.close(f);
        }

        self.allocator.destroy(self);
    }

    pub fn getTensorDirect(self: *const SafeTensorsFile, name: []const u8) ?Tensor {
        return self.tensors.get(name);
    }

    /// Retrieve a tensor by name, supporting both HuggingFace and GGUF canonical naming conventions
    pub fn getTensor(self: *const SafeTensorsFile, name: []const u8) ?Tensor {
        if (self.tensors.get(name)) |t| return t;

        // Try mapping GGUF name -> HuggingFace name
        var hf_name_buf: [256]u8 = undefined;

        if (std.mem.eql(u8, name, "token_embd.weight")) {
            const emb_names = [_][]const u8{
                "model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight",
                "language_model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
                "embed_tokens.weight",
                "transformer.wte.weight",
                "model.wte.weight",
            };
            for (emb_names) |n| {
                if (self.tensors.get(n)) |t| return t;
            }
        } else if (std.mem.eql(u8, name, "output_norm.weight")) {
            const norm_names = [_][]const u8{
                "model.norm.weight",
                "model.language_model.norm.weight",
                "language_model.norm.weight",
                "language_model.model.norm.weight",
                "norm.weight",
                "transformer.ln_f.weight",
                "model.ln_f.weight",
            };
            for (norm_names) |n| {
                if (self.tensors.get(n)) |t| return t;
            }
        } else if (std.mem.eql(u8, name, "output.weight")) {
            const out_names = [_][]const u8{
                "lm_head.weight",
                "model.lm_head.weight",
                "model.language_model.lm_head.weight",
                "language_model.lm_head.weight",
                "model.language_model.embed_tokens.weight",
                "language_model.embed_tokens.weight",
                "model.embed_tokens.weight",
                "embed_tokens.weight",
            };
            for (out_names) |n| {
                if (self.tensors.get(n)) |t| return t;
            }
        } else if (std.mem.eql(u8, name, "embed_tokens_per_layer.weight")) {
            const ple_names = [_][]const u8{
                "model.language_model.embed_tokens_per_layer.weight",
                "language_model.embed_tokens_per_layer.weight",
                "model.embed_tokens_per_layer.weight",
                "embed_tokens_per_layer.weight",
            };
            for (ple_names) |n| {
                if (self.tensors.get(n)) |t| return t;
            }
        } else if (std.mem.eql(u8, name, "per_layer_model_projection.weight")) {
            const ctx_names = [_][]const u8{
                "model.language_model.per_layer_model_projection.weight",
                "language_model.per_layer_model_projection.weight",
                "model.per_layer_model_projection.weight",
                "per_layer_model_projection.weight",
            };
            for (ctx_names) |n| {
                if (self.tensors.get(n)) |t| return t;
            }
        } else if (std.mem.eql(u8, name, "per_layer_projection_norm.weight")) {
            const ctx_norm_names = [_][]const u8{
                "model.language_model.per_layer_projection_norm.weight",
                "language_model.per_layer_projection_norm.weight",
                "model.per_layer_projection_norm.weight",
                "per_layer_projection_norm.weight",
            };
            for (ctx_norm_names) |n| {
                if (self.tensors.get(n)) |t| return t;
            }
        } else if (std.mem.startsWith(u8, name, "blk.")) {
            // Pattern: blk.{N}.{component}.weight
            var it = std.mem.splitScalar(u8, name, '.');
            _ = it.next(); // blk
            const idx_str = it.next() orelse return null;
            const comp = it.next() orelse return null;

            const layer_prefixes = [_][]const u8{
                "model.layers",
                "model.language_model.layers",
                "language_model.layers",
                "language_model.model.layers",
                "layers",
                "transformer.h",
            };

            for (layer_prefixes) |prefix| {
                if (std.mem.eql(u8, comp, "attn_norm") or std.mem.eql(u8, comp, "input_layernorm")) {
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.input_layernorm.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.ln_1.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                } else if (std.mem.eql(u8, comp, "attn_q")) {
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.self_attn.q_proj.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.attn.q_proj.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                } else if (std.mem.eql(u8, comp, "attn_k")) {
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.self_attn.k_proj.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.attn.k_proj.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                } else if (std.mem.eql(u8, comp, "attn_v")) {
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.self_attn.v_proj.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.attn.v_proj.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                } else if (std.mem.eql(u8, comp, "attn_output")) {
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.self_attn.o_proj.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.attn.o_proj.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                } else if (std.mem.eql(u8, comp, "attn_q_norm")) {
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.self_attn.q_norm.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                } else if (std.mem.eql(u8, comp, "attn_k_norm")) {
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.self_attn.k_norm.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                } else if (std.mem.eql(u8, comp, "post_attn_norm") or std.mem.eql(u8, comp, "post_attention_layernorm")) {
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.post_attention_layernorm.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                } else if (std.mem.eql(u8, comp, "ffn_norm") or std.mem.eql(u8, comp, "pre_feedforward_layernorm")) {
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.pre_feedforward_layernorm.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.post_attention_layernorm.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.ln_2.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                } else if (std.mem.eql(u8, comp, "post_ffn_norm") or std.mem.eql(u8, comp, "post_feedforward_layernorm")) {
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.post_feedforward_layernorm.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                } else if (std.mem.eql(u8, comp, "post_per_layer_input_norm")) {
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.post_per_layer_input_norm.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                } else if (std.mem.eql(u8, comp, "per_layer_input_gate")) {
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.per_layer_input_gate.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                } else if (std.mem.eql(u8, comp, "per_layer_projection")) {
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.per_layer_projection.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                } else if (std.mem.eql(u8, comp, "layer_scalar")) {
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.layer_scalar", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                } else if (std.mem.eql(u8, comp, "ffn_gate")) {
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.mlp.gate_proj.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                } else if (std.mem.eql(u8, comp, "ffn_up")) {
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.mlp.up_proj.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                } else if (std.mem.eql(u8, comp, "ffn_down")) {
                    if (std.fmt.bufPrint(&hf_name_buf, "{s}.{s}.mlp.down_proj.weight", .{ prefix, idx_str }) catch null) |n| {
                        if (self.tensors.get(n)) |t| return t;
                    }
                }
            }
        }

        return null;
    }
};

pub fn parseDtype(dtype_str: []const u8) GGMLType {
    if (std.mem.eql(u8, dtype_str, "F32")) return .F32;
    if (std.mem.eql(u8, dtype_str, "F16")) return .F16;
    if (std.mem.eql(u8, dtype_str, "BF16") or std.mem.eql(u8, dtype_str, "bfloat16")) return .BF16;
    if (std.mem.eql(u8, dtype_str, "F64")) return .F64;
    if (std.mem.eql(u8, dtype_str, "I8") or std.mem.eql(u8, dtype_str, "U8")) return .I8;
    if (std.mem.eql(u8, dtype_str, "I16") or std.mem.eql(u8, dtype_str, "U16")) return .I16;
    if (std.mem.eql(u8, dtype_str, "I32") or std.mem.eql(u8, dtype_str, "U32")) return .I32;
    if (std.mem.eql(u8, dtype_str, "I64") or std.mem.eql(u8, dtype_str, "U64")) return .I64;
    return .F32;
}

/// Parse HuggingFace config.json to ModelParams
pub fn parseHFConfig(allocator: std.mem.Allocator, config_path: []const u8) !ModelParams {
    const fd = try std.posix.openat(std.posix.AT.FDCWD, config_path, .{ .ACCMODE = .RDONLY }, 0);
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

    if (parsed.value != .object) return error.InvalidConfigJson;
    const obj = parsed.value.object;

    var target_obj = obj;
    if (obj.get("text_config")) |tc| {
        if (tc == .object) {
            target_obj = tc.object;
        }
    }

    var params = ModelParams{};

    // Architecture
    if (obj.get("architectures")) |arch_val| {
        if (arch_val == .array and arch_val.array.items.len > 0 and arch_val.array.items[0] == .string) {
            const arch_str = arch_val.array.items[0].string;
            if (std.mem.indexOf(u8, arch_str, "Llama") != null) params.arch = .llama;
            if (std.mem.indexOf(u8, arch_str, "Qwen2") != null) params.arch = .qwen2;
            if (std.mem.indexOf(u8, arch_str, "Mistral") != null) params.arch = .mistral;
            if (std.mem.indexOf(u8, arch_str, "Gemma4") != null or std.mem.indexOf(u8, arch_str, "gemma4") != null) {
                params.arch = .gemma4;
            } else if (std.mem.indexOf(u8, arch_str, "Gemma2") != null) {
                params.arch = .gemma2;
            } else if (std.mem.indexOf(u8, arch_str, "Gemma") != null) {
                params.arch = .gemma;
            }
            if (std.mem.indexOf(u8, arch_str, "Phi") != null) params.arch = .phi3;
        }
    }
    if (obj.get("model_type")) |mt| {
        if (mt == .string) {
            if (std.mem.indexOf(u8, mt.string, "gemma4") != null or std.mem.indexOf(u8, mt.string, "gemma_4") != null) {
                params.arch = .gemma4;
            } else if (std.mem.indexOf(u8, mt.string, "gemma") != null) {
                params.arch = .gemma;
            }
            if (std.mem.indexOf(u8, mt.string, "llama") != null) params.arch = .llama;
            if (std.mem.indexOf(u8, mt.string, "qwen") != null) params.arch = .qwen2;
            if (std.mem.indexOf(u8, mt.string, "mistral") != null) params.arch = .mistral;
        }
    }

    const cfg_sources = [_]std.json.ObjectMap{ target_obj, obj };
    for (cfg_sources) |cfg| {
        if (cfg.get("hidden_size")) |v| {
            if (v == .integer and v.integer > 0) params.embedding_length = @intCast(v.integer);
        }
        if (cfg.get("intermediate_size")) |v| {
            if (v == .integer and v.integer > 0) params.feed_forward_length = @intCast(v.integer);
        }
        if (cfg.get("num_attention_heads")) |v| {
            if (v == .integer and v.integer > 0) params.head_count = @intCast(v.integer);
        }
        if (cfg.get("num_key_value_heads")) |v| {
            if (v == .integer and v.integer > 0) params.head_count_kv = @intCast(v.integer);
        }
        if (cfg.get("num_hidden_layers")) |v| {
            if (v == .integer and v.integer > 0) params.block_count = @intCast(v.integer);
        }
        if (cfg.get("vocab_size")) |v| {
            if (v == .integer and v.integer > 0) params.vocab_size = @intCast(v.integer);
        }
        if (cfg.get("rms_norm_eps")) |v| {
            if (v == .float) params.layer_norm_rms_epsilon = @floatCast(v.float);
        } else if (cfg.get("layer_norm_eps")) |v| {
            if (v == .float) params.layer_norm_rms_epsilon = @floatCast(v.float);
        }
        if (cfg.get("rope_theta")) |v| {
            if (v == .float) params.rope_freq_base = @floatCast(v.float);
            if (v == .integer) params.rope_freq_base = @floatFromInt(v.integer);
        }
        if (cfg.get("max_position_embeddings")) |v| {
            if (v == .integer and v.integer > 0) params.context_length = @intCast(v.integer);
        }
        if (cfg.get("head_dim")) |v| {
            if (v == .integer and v.integer > 0) params.head_size = @intCast(v.integer);
        }
        if (cfg.get("num_kv_shared_layers")) |v| {
            if (v == .integer and v.integer > 0) params.num_kv_shared_layers = @intCast(v.integer);
        }
        if (cfg.get("final_logit_softcapping")) |v| {
            if (v == .float) params.final_logit_softcapping = @floatCast(v.float);
            if (v == .integer) params.final_logit_softcapping = @floatFromInt(v.integer);
        }
        if (cfg.get("attn_logit_softcapping")) |v| {
            if (v == .float) params.attn_logit_softcapping = @floatCast(v.float);
            if (v == .integer) params.attn_logit_softcapping = @floatFromInt(v.integer);
        }
    }

    params.initComputed();
    return params;
}

test "parseDtype support" {
    try std.testing.expectEqual(GGMLType.F32, parseDtype("F32"));
    try std.testing.expectEqual(GGMLType.F16, parseDtype("F16"));
    try std.testing.expectEqual(GGMLType.BF16, parseDtype("BF16"));
    try std.testing.expectEqual(GGMLType.I8, parseDtype("I8"));
}
