const std = @import("std");

/// GGML Tensor Data Types matching the GGUF specification
pub const GGMLType = enum(u32) {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    I64 = 19,
    F64 = 20,
    IQ1_S = 21,
    IQ4_NL = 22,
    IQ3_S = 23,
    IQ2_S = 24,
    IQ4_XS = 25,
    I8_0 = 26,
    BF16 = 30,
    _,

    pub fn blockSize(self: GGMLType) usize {
        return switch (self) {
            .F32, .I32, .F64, .I64, .I16, .I8, .F16, .BF16 => 1,
            .Q4_0, .Q4_1, .Q5_0, .Q5_1, .Q8_0, .Q8_1, .IQ4_NL => 32,
            .Q2_K, .Q3_K, .Q4_K, .Q5_K, .Q6_K, .Q8_K, .IQ4_XS, .IQ3_S, .IQ2_S, .IQ1_S => 256,
            else => 32,
        };
    }

    pub fn typeSize(self: GGMLType) usize {
        return switch (self) {
            .F32 => @sizeOf(f32),
            .F16 => @sizeOf(f16),
            .BF16 => 2,
            .I8 => @sizeOf(i8),
            .I16 => @sizeOf(i16),
            .I32 => @sizeOf(i32),
            .I64 => @sizeOf(i64),
            .F64 => @sizeOf(f64),
            .Q4_0 => @sizeOf(BlockQ4_0),
            .Q4_1 => @sizeOf(BlockQ4_1),
            .Q5_0 => @sizeOf(BlockQ5_0),
            .Q5_1 => @sizeOf(BlockQ5_1),
            .Q8_0 => @sizeOf(BlockQ8_0),
            .Q8_1 => @sizeOf(BlockQ8_1),
            .Q2_K => @sizeOf(BlockQ2_K),
            .Q3_K => @sizeOf(BlockQ3_K),
            .Q4_K => @sizeOf(BlockQ4_K),
            .Q5_K => @sizeOf(BlockQ5_K),
            .Q6_K => @sizeOf(BlockQ6_K),
            .Q8_K => @sizeOf(BlockQ8_K),
            .IQ4_NL => @sizeOf(BlockIQ4_NL),
            else => 1,
        };
    }

    pub fn name(self: GGMLType) []const u8 {
        return switch (self) {
            .F32 => "f32",
            .F16 => "f16",
            .BF16 => "bf16",
            .Q4_0 => "q4_0",
            .Q4_1 => "q4_1",
            .Q5_0 => "q5_0",
            .Q5_1 => "q5_1",
            .Q8_0 => "q8_0",
            .Q8_1 => "q8_1",
            .Q2_K => "q2_k",
            .Q3_K => "q3_k",
            .Q4_K => "q4_k",
            .Q5_K => "q5_k",
            .Q6_K => "q6_k",
            .Q8_K => "q8_k",
            .I8 => "i8",
            .I16 => "i16",
            .I32 => "i32",
            .I64 => "i64",
            .F64 => "f64",
            .IQ4_NL => "iq4_nl",
            .IQ4_XS => "iq4_xs",
            .IQ3_S => "iq3_s",
            .IQ2_S => "iq2_s",
            .IQ1_S => "iq1_s",
            else => "unknown",
        };
    }

    pub fn isQuantized(self: GGMLType) bool {
        return switch (self) {
            .Q4_0, .Q4_1, .Q5_0, .Q5_1, .Q8_0, .Q8_1, .Q2_K, .Q3_K, .Q4_K, .Q5_K, .Q6_K, .Q8_K, .IQ4_NL, .IQ4_XS, .IQ3_S, .IQ2_S, .IQ1_S => true,
            else => false,
        };
    }
};

// ============================================================================
// Quantization Block Structures
// ============================================================================

/// Q4_0 block: 32 4-bit values + 1 16-bit float scale (18 bytes total)
pub const BlockQ4_0 = extern struct {
    d: f16, // scale
    qs: [16]u8, // 32 nibbles
};

/// Q4_1 block: 32 4-bit values + scale + min (20 bytes total)
pub const BlockQ4_1 = extern struct {
    d: f16, // scale
    m: f16, // min
    qs: [16]u8, // 32 nibbles
};

/// Q5_0 block: 32 5-bit values + 1 16-bit float scale (22 bytes total)
pub const BlockQ5_0 = extern struct {
    d: f16, // scale
    qh: [4]u8, // 32 high bits (1 bit per weight)
    qs: [16]u8, // 32 low 4-bit nibbles
};

/// Q5_1 block: 32 5-bit values + scale + min (24 bytes total)
pub const BlockQ5_1 = extern struct {
    d: f16, // scale
    m: f16, // min
    qh: [4]u8, // 32 high bits
    qs: [16]u8, // 32 low 4-bit nibbles
};

/// Q8_0 block: 32 8-bit signed ints + 1 16-bit float scale (34 bytes total)
pub const BlockQ8_0 = extern struct {
    d: f16, // scale
    qs: [32]i8, // 32 8-bit quantized values
};

/// Q8_1 block: 32 8-bit signed ints + scale + sum (36 bytes total)
pub const BlockQ8_1 = extern struct {
    d: f16, // scale
    s: f16, // sum
    qs: [32]i8, // 32 values
};

/// Q2_K block: 256 values with super-blocks (84 bytes total)
pub const BlockQ2_K = extern struct {
    scales: [16]u8, // 16 4-bit scales and 4-bit mins
    qs: [64]u8, // 256 2-bit values
    d: f16, // super-block scale
    dmin: f16, // super-block min
};

/// Q3_K block: 256 values with super-blocks (110 bytes total)
pub const BlockQ3_K = extern struct {
    hmask: [32]u8, // 256 high bits
    qs: [64]u8, // 256 2-bit low values
    scales: [12]u8, // 16 6-bit scales
    d: f16, // super-block scale
};

/// Q4_K block: 256 values with super-blocks (144 bytes total)
pub const BlockQ4_K = extern struct {
    d: f16, // super-block scale
    dmin: f16, // super-block min
    scales: [12]u8, // 6-bit scales and mins
    qs: [128]u8, // 4-bit nibbles for 256 values
};

/// Q5_K block: 256 values with super-blocks (176 bytes total)
pub const BlockQ5_K = extern struct {
    d: f16, // super-block scale
    dmin: f16, // super-block min
    scales: [12]u8, // 8 6-bit scales and 8 6-bit mins
    qh: [32]u8, // 256 1-bit high bits
    qs: [128]u8, // 256 4-bit low values
};

/// Q6_K block: 256 values (210 bytes total)
pub const BlockQ6_K = extern struct {
    ql: [128]u8, // low 4-bits
    qh: [64]u8, // high 2-bits
    scales: [16]i8, // 8-bit scales
    d: f16, // super-block scale
};

/// Q8_K block: 256 values (292 bytes total)
pub const BlockQ8_K = extern struct {
    d: f32, // delta / scale
    qs: [256]i8, // 256 8-bit values
    bsums: [16]i16, // block sums
};

/// IQ4_NL block: 32 values with non-linear lookup table (18 bytes total)
pub const BlockIQ4_NL = extern struct {
    d: f16, // scale
    qs: [16]u8, // 32 4-bit indices
};

// ============================================================================
// Model Architecture Types
// ============================================================================

pub const Architecture = enum {
    llama,
    qwen2,
    mistral,
    gemma,
    gemma2,
    gemma4,
    phi,
    phi3,
    deepseek,
    starcoder2,
    gpt2,
    unknown,

    pub fn fromString(s: []const u8) Architecture {
        if (std.mem.eql(u8, s, "llama")) return .llama;
        if (std.mem.eql(u8, s, "qwen2") or std.mem.eql(u8, s, "qwen2.5")) return .qwen2;
        if (std.mem.eql(u8, s, "mistral")) return .mistral;
        if (std.mem.eql(u8, s, "gemma")) return .gemma;
        if (std.mem.eql(u8, s, "gemma2")) return .gemma2;
        if (std.mem.eql(u8, s, "gemma4") or std.mem.eql(u8, s, "gemma_4")) return .gemma4;
        if (std.mem.eql(u8, s, "phi") or std.mem.eql(u8, s, "phi2")) return .phi;
        if (std.mem.eql(u8, s, "phi3")) return .phi3;
        if (std.mem.eql(u8, s, "deepseek")) return .deepseek;
        if (std.mem.eql(u8, s, "starcoder2")) return .starcoder2;
        if (std.mem.eql(u8, s, "gpt2")) return .gpt2;
        return .unknown;
    }

    pub fn asString(self: Architecture) []const u8 {
        return @tagName(self);
    }
};

/// Model Hyperparameters extracted from GGUF metadata
pub const ModelParams = struct {
    arch: Architecture = .llama,
    context_length: usize = 2048,
    embedding_length: usize = 4096, // d_model / hidden_size
    feed_forward_length: usize = 11008, // intermediate_size
    head_count: usize = 32, // num_attention_heads
    head_count_kv: usize = 32, // num_key_value_heads (for GQA)
    block_count: usize = 32, // num_hidden_layers
    head_size: usize = 0, // embedding_length / head_count (or custom)
    vocab_size: usize = 32000,
    layer_norm_rms_epsilon: f32 = 1e-5,
    rope_freq_base: f32 = 10000.0,
    rope_freq_scale: f32 = 1.0,
    rope_dim_count: usize = 0, // rope head dim
    rope_type: RoPEType = .normal,
    use_gemma_rms_unit_offset: bool = false, // Gemma 2: norm * (1.0 + weight)
    attn_logit_softcapping: f32 = 0.0, // Gemma 2
    final_logit_softcapping: f32 = 0.0, // Gemma 2
    expert_count: usize = 0, // MoE
    expert_used_count: usize = 0, // MoE
    sliding_window: usize = 0, // Sliding window attention

    pub fn initComputed(self: *ModelParams) void {
        if (self.head_count > 0 and self.head_size == 0) {
            self.head_size = self.embedding_length / self.head_count;
        }
        if (self.rope_dim_count == 0) {
            self.rope_dim_count = self.head_size;
        }
        if (self.head_count_kv == 0) {
            self.head_count_kv = self.head_count;
        }
        if (self.arch == .gemma or self.arch == .gemma2) {
            self.use_gemma_rms_unit_offset = true;
        } else if (self.arch == .gemma4) {
            self.use_gemma_rms_unit_offset = false;
        }
    }
};

pub const RoPEType = enum {
    normal, // standard LLaMA (half split / complex rotate)
    neox, // GPT-NeoX style (interleaved / even-odd rotate)
};

// ============================================================================
// Tensor Descriptor & Slicing
// ============================================================================

pub const Tensor = struct {
    name: []const u8,
    type: GGMLType,
    n_dims: usize,
    shape: [4]usize, // [d0, d1, d2, d3] in GGUF order (d0 is innermost/contiguous)
    offset: u64,
    data: []const u8, // raw bytes pointer into mmap or buffer

    pub fn elements(self: Tensor) usize {
        var count: usize = 1;
        var i: usize = 0;
        while (i < self.n_dims) : (i += 1) {
            count *= self.shape[i];
        }
        return count;
    }

    pub fn sizeBytes(self: Tensor) usize {
        const blk_size = self.type.blockSize();
        const type_size = self.type.typeSize();
        const n_elem = self.elements();
        return (n_elem / blk_size) * type_size;
    }

    pub fn getRowBytes(self: Tensor) usize {
        const blk_size = self.type.blockSize();
        const type_size = self.type.typeSize();
        return (self.shape[0] / blk_size) * type_size;
    }

    pub fn getRow(self: Tensor, row_idx: usize) []const u8 {
        const row_bytes = self.getRowBytes();
        const start = row_idx * row_bytes;
        return self.data[start .. start + row_bytes];
    }
};

// ============================================================================
// Generation & Sampler Configuration
// ============================================================================

pub const SamplerParams = struct {
    temperature: f32 = 0.7,
    top_k: usize = 40,
    top_p: f32 = 0.9,
    min_p: f32 = 0.05,
    repetition_penalty: f32 = 1.1,
    presence_penalty: f32 = 0.0,
    frequency_penalty: f32 = 0.0,
    repetition_penalty_window: usize = 64,
    seed: u64 = 42,
    greedy: bool = false,
};

pub const GenerationOptions = struct {
    max_tokens: usize = 512,
    sampler: SamplerParams = .{},
    stop_tokens: []const u32 = &.{},
    echo_prompt: bool = false,
};

pub const GenerationStats = struct {
    prompt_tokens: usize = 0,
    completion_tokens: usize = 0,
    prefill_time_ms: f64 = 0.0,
    generation_time_ms: f64 = 0.0,
    total_time_ms: f64 = 0.0,

    pub fn prefillTokensPerSec(self: GenerationStats) f64 {
        if (self.prefill_time_ms <= 0.0) return 0.0;
        return (@as(f64, @floatFromInt(self.prompt_tokens)) / self.prefill_time_ms) * 1000.0;
    }

    pub fn generationTokensPerSec(self: GenerationStats) f64 {
        if (self.generation_time_ms <= 0.0) return 0.0;
        return (@as(f64, @floatFromInt(self.completion_tokens)) / self.generation_time_ms) * 1000.0;
    }
};

test "GGMLType sizes and block counts" {
    try std.testing.expectEqual(@as(usize, 18), @sizeOf(BlockQ4_0));
    try std.testing.expectEqual(@as(usize, 20), @sizeOf(BlockQ4_1));
    try std.testing.expectEqual(@as(usize, 22), @sizeOf(BlockQ5_0));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(BlockQ5_1));
    try std.testing.expectEqual(@as(usize, 34), @sizeOf(BlockQ8_0));
    try std.testing.expectEqual(@as(usize, 36), @sizeOf(BlockQ8_1));
    try std.testing.expectEqual(@as(usize, 84), @sizeOf(BlockQ2_K));
    try std.testing.expectEqual(@as(usize, 110), @sizeOf(BlockQ3_K));
    try std.testing.expectEqual(@as(usize, 144), @sizeOf(BlockQ4_K));
    try std.testing.expectEqual(@as(usize, 176), @sizeOf(BlockQ5_K));
    try std.testing.expectEqual(@as(usize, 210), @sizeOf(BlockQ6_K));
    try std.testing.expectEqual(@as(usize, 292), @sizeOf(BlockQ8_K));
    try std.testing.expectEqual(@as(usize, 18), @sizeOf(BlockIQ4_NL));

    try std.testing.expectEqual(@as(usize, 32), GGMLType.Q4_0.blockSize());
    try std.testing.expectEqual(@as(usize, 32), GGMLType.Q5_0.blockSize());
    try std.testing.expectEqual(@as(usize, 32), GGMLType.Q8_0.blockSize());
    try std.testing.expectEqual(@as(usize, 256), GGMLType.Q2_K.blockSize());
    try std.testing.expectEqual(@as(usize, 256), GGMLType.Q3_K.blockSize());
    try std.testing.expectEqual(@as(usize, 256), GGMLType.Q4_K.blockSize());
    try std.testing.expectEqual(@as(usize, 256), GGMLType.Q5_K.blockSize());
    try std.testing.expectEqual(@as(usize, 256), GGMLType.Q6_K.blockSize());
    try std.testing.expectEqual(@as(usize, 256), GGMLType.Q8_K.blockSize());
}
