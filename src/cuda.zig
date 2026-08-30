const std = @import("std");
const types = @import("types.zig");
const GGMLType = types.GGMLType;

pub const CudaStream_t = ?*anyopaque;

// ============================================================================
// Direct CUDA C ABI Declarations
// ============================================================================

pub extern "c" fn cuda_device_get_info(device_id: c_int, name: [*c]u8, name_len: usize, total_vram_bytes: *usize) c_int;
pub extern "c" fn cuda_malloc(ptr: *?*anyopaque, bytes: usize) c_int;
pub extern "c" fn cuda_free(ptr: ?*anyopaque) c_int;
pub extern "c" fn cuda_memcpy_h2d(dst: ?*anyopaque, src: ?*const anyopaque, bytes: usize, stream: CudaStream_t) c_int;
pub extern "c" fn cuda_memcpy_d2h(dst: ?*anyopaque, src: ?*const anyopaque, bytes: usize, stream: CudaStream_t) c_int;
pub extern "c" fn cuda_memcpy_d2d(dst: ?*anyopaque, src: ?*const anyopaque, bytes: usize, stream: CudaStream_t) c_int;
pub extern "c" fn cuda_stream_create(stream: *CudaStream_t) c_int;
pub extern "c" fn cuda_stream_destroy(stream: CudaStream_t) c_int;
pub extern "c" fn cuda_stream_sync(stream: CudaStream_t) c_int;

pub extern "c" fn cuda_gemv_q4_0(weights: ?*const anyopaque, x: [*]const f32, y: [*]f32, rows: c_int, cols: c_int, stream: CudaStream_t) void;
pub extern "c" fn cuda_gemv_q8_0(weights: ?*const anyopaque, x: [*]const f32, y: [*]f32, rows: c_int, cols: c_int, stream: CudaStream_t) void;
pub extern "c" fn cuda_gemv_q4_k(weights: ?*const anyopaque, x: [*]const f32, y: [*]f32, rows: c_int, cols: c_int, stream: CudaStream_t) void;
pub extern "c" fn cuda_gemv_q6_k(weights: ?*const anyopaque, x: [*]const f32, y: [*]f32, rows: c_int, cols: c_int, stream: CudaStream_t) void;
pub extern "c" fn cuda_gemv_f16(weights: ?*const anyopaque, x: [*]const f32, y: [*]f32, rows: c_int, cols: c_int, stream: CudaStream_t) void;
pub extern "c" fn cuda_gemv_bf16(weights: ?*const anyopaque, x: [*]const f32, y: [*]f32, rows: c_int, cols: c_int, stream: CudaStream_t) void;
pub extern "c" fn cuda_gemv_f32(weights: ?*const anyopaque, x: [*]const f32, y: [*]f32, rows: c_int, cols: c_int, stream: CudaStream_t) void;

pub extern "c" fn cuda_rmsnorm(x: [*]const f32, weight: ?[*]const f32, out: [*]f32, n: c_int, eps: f32, use_unit_offset: c_int, stream: CudaStream_t) void;
pub extern "c" fn cuda_rope(q: ?[*]f32, k: ?[*]f32, pos: c_int, num_heads: c_int, num_kv_heads: c_int, head_dim: c_int, freq_base: f32, stream: CudaStream_t) void;

pub extern "c" fn cuda_geglu(gate: [*]const f32, up: [*]const f32, out: [*]f32, n: c_int, stream: CudaStream_t) void;
pub extern "c" fn cuda_swiglu(gate: [*]const f32, up: [*]const f32, out: [*]f32, n: c_int, stream: CudaStream_t) void;
pub extern "c" fn cuda_add(x: [*]f32, residual: [*]const f32, n: c_int, stream: CudaStream_t) void;
pub extern "c" fn cuda_scale(x: [*]f32, scale: f32, n: c_int, stream: CudaStream_t) void;
pub extern "c" fn cuda_tanh_softcap(x: [*]f32, cap: f32, n: c_int, stream: CudaStream_t) void;

pub extern "c" fn cuda_kv_cache_put(
    k_cache: [*]f32,
    v_cache: [*]f32,
    k: [*]const f32,
    v: [*]const f32,
    layer_idx: c_int,
    pos: c_int,
    max_seq: c_int,
    n_kv_heads: c_int,
    head_dim: c_int,
    stream: CudaStream_t,
) void;

pub extern "c" fn cuda_attention_forward(
    q: [*]const f32,
    k_cache: [*]const f32,
    v_cache: [*]const f32,
    out: [*]f32,
    donor_layer: c_int,
    pos: c_int,
    max_seq: c_int,
    n_heads: c_int,
    n_kv_heads: c_int,
    head_dim: c_int,
    attn_scale: f32,
    softcap: f32,
    sliding_window: c_int,
    stream: CudaStream_t,
) void;

pub extern "c" fn cuda_ple_gate_gelu(
    ple_gate_in: [*]const f32,
    ple_slice: [*]const f32,
    ple_buf_out: [*]f32,
    ple_dim: c_int,
    stream: CudaStream_t,
) void;

pub extern "c" fn cuda_ple_ctx_fuse(
    ctx_ple_buf: [*]f32,
    ctx_scratch: [*]const f32,
    n: c_int,
    add_token_embd: c_int,
    stream: CudaStream_t,
) void;

// ============================================================================
// High-Level Zig Wrappers
// ============================================================================

pub const CudaBuffer = struct {
    ptr: ?*anyopaque,
    size_bytes: usize,

    pub fn upload(self: *const CudaBuffer, host_src: []const u8, stream: CudaStream_t) !void {
        const copy_len = @min(self.size_bytes, host_src.len);
        if (copy_len == 0) return;
        const err = cuda_memcpy_h2d(self.ptr, host_src.ptr, copy_len, stream);
        if (err != 0) return error.CudaMemcpyFailed;
    }

    pub fn download(self: *const CudaBuffer, host_dst: []u8, stream: CudaStream_t) !void {
        const copy_len = @min(self.size_bytes, host_dst.len);
        if (copy_len == 0) return;
        const err = cuda_memcpy_d2h(host_dst.ptr, self.ptr, copy_len, stream);
        if (err != 0) return error.CudaMemcpyFailed;
    }

    pub fn deinit(self: *CudaBuffer) void {
        if (self.ptr) |p| {
            _ = cuda_free(p);
            self.ptr = null;
        }
    }
};

pub const CudaDevice = struct {
    allocator: std.mem.Allocator,
    stream: CudaStream_t,
    name: [256]u8,
    name_len: usize,
    total_vram_bytes: usize,

    pub fn init(allocator: std.mem.Allocator) !*CudaDevice {
        var dev_name: [256]u8 = undefined;
        var total_vram: usize = 0;
        const info_res = cuda_device_get_info(0, &dev_name, dev_name.len, &total_vram);
        if (info_res != 0) return error.CudaDeviceNotFound;

        var stream: CudaStream_t = null;
        const stream_res = cuda_stream_create(&stream);
        if (stream_res != 0) return error.CudaStreamCreateFailed;

        const name_len = std.mem.indexOfScalar(u8, &dev_name, 0) orelse dev_name.len;

        const self = try allocator.create(CudaDevice);
        self.* = .{
            .allocator = allocator,
            .stream = stream,
            .name = dev_name,
            .name_len = name_len,
            .total_vram_bytes = total_vram,
        };
        return self;
    }

    pub fn deinit(self: *CudaDevice) void {
        _ = cuda_stream_destroy(self.stream);
        self.allocator.destroy(self);
    }

    pub fn getName(self: *const CudaDevice) []const u8 {
        return self.name[0..self.name_len];
    }

    pub fn alloc(self: *const CudaDevice, size_bytes: usize) !CudaBuffer {
        _ = self;
        var ptr: ?*anyopaque = null;
        const res = cuda_malloc(&ptr, size_bytes);
        if (res != 0 or ptr == null) return error.CudaOutOfMemory;
        return CudaBuffer{
            .ptr = ptr,
            .size_bytes = size_bytes,
        };
    }

    pub fn sync(self: *const CudaDevice) void {
        _ = cuda_stream_sync(self.stream);
    }

    pub fn gemv(
        self: *const CudaDevice,
        qtype: GGMLType,
        d_weights: ?*const anyopaque,
        d_x: [*]const f32,
        d_y: [*]f32,
        rows: usize,
        cols: usize,
    ) void {
        const r: c_int = @intCast(rows);
        const c: c_int = @intCast(cols);
        switch (qtype) {
            .Q4_0 => cuda_gemv_q4_0(d_weights, d_x, d_y, r, c, self.stream),
            .Q8_0 => cuda_gemv_q8_0(d_weights, d_x, d_y, r, c, self.stream),
            .Q4_K => cuda_gemv_q4_k(d_weights, d_x, d_y, r, c, self.stream),
            .Q6_K => cuda_gemv_q6_k(d_weights, d_x, d_y, r, c, self.stream),
            .F16 => cuda_gemv_f16(d_weights, d_x, d_y, r, c, self.stream),
            .BF16 => cuda_gemv_bf16(d_weights, d_x, d_y, r, c, self.stream),
            .F32 => cuda_gemv_f32(d_weights, d_x, d_y, r, c, self.stream),
            else => cuda_gemv_f32(d_weights, d_x, d_y, r, c, self.stream),
        }
    }

    pub fn rmsNorm(
        self: *const CudaDevice,
        d_x: [*]const f32,
        d_weight: ?[*]const f32,
        d_out: [*]f32,
        n: usize,
        eps: f32,
        use_unit_offset: bool,
    ) void {
        cuda_rmsnorm(d_x, d_weight, d_out, @intCast(n), eps, if (use_unit_offset) 1 else 0, self.stream);
    }

    pub fn rope(
        self: *const CudaDevice,
        d_q: ?[*]f32,
        d_k: ?[*]f32,
        pos: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        freq_base: f32,
    ) void {
        cuda_rope(d_q, d_k, @intCast(pos), @intCast(num_heads), @intCast(num_kv_heads), @intCast(head_dim), freq_base, self.stream);
    }

    pub fn kvCachePut(
        self: *const CudaDevice,
        d_k_cache: [*]f32,
        d_v_cache: [*]f32,
        d_k: [*]const f32,
        d_v: [*]const f32,
        layer_idx: usize,
        pos: usize,
        max_seq: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) void {
        cuda_kv_cache_put(
            d_k_cache,
            d_v_cache,
            d_k,
            d_v,
            @intCast(layer_idx),
            @intCast(pos),
            @intCast(max_seq),
            @intCast(n_kv_heads),
            @intCast(head_dim),
            self.stream,
        );
    }

    pub fn attentionForward(
        self: *const CudaDevice,
        d_q: [*]const f32,
        d_k_cache: [*]const f32,
        d_v_cache: [*]const f32,
        d_out: [*]f32,
        donor_layer: usize,
        pos: usize,
        max_seq: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        attn_scale: f32,
        softcap: f32,
        sliding_window: usize,
    ) void {
        cuda_attention_forward(
            d_q,
            d_k_cache,
            d_v_cache,
            d_out,
            @intCast(donor_layer),
            @intCast(pos),
            @intCast(max_seq),
            @intCast(n_heads),
            @intCast(n_kv_heads),
            @intCast(head_dim),
            attn_scale,
            softcap,
            @intCast(sliding_window),
            self.stream,
        );
    }

    pub fn geglu(
        self: *const CudaDevice,
        d_gate: [*]const f32,
        d_up: [*]const f32,
        d_out: [*]f32,
        n: usize,
    ) void {
        cuda_geglu(d_gate, d_up, d_out, @intCast(n), self.stream);
    }

    pub fn pleGateGelu(
        self: *const CudaDevice,
        d_ple_gate: [*]const f32,
        d_ple_slice: [*]const f32,
        d_ple_buf: [*]f32,
        ple_dim: usize,
    ) void {
        cuda_ple_gate_gelu(d_ple_gate, d_ple_slice, d_ple_buf, @intCast(ple_dim), self.stream);
    }

    pub fn pleCtxFuse(
        self: *const CudaDevice,
        d_ctx_ple_buf: [*]f32,
        d_ctx_scratch: [*]const f32,
        n: usize,
        add_token_embd: bool,
    ) void {
        cuda_ple_ctx_fuse(d_ctx_ple_buf, d_ctx_scratch, @intCast(n), if (add_token_embd) 1 else 0, self.stream);
    }

    pub fn add(
        self: *const CudaDevice,
        d_x: [*]f32,
        d_res: [*]const f32,
        n: usize,
    ) void {
        cuda_add(d_x, d_res, @intCast(n), self.stream);
    }

    pub fn scale(
        self: *const CudaDevice,
        d_x: [*]f32,
        s: f32,
        n: usize,
    ) void {
        cuda_scale(d_x, s, @intCast(n), self.stream);
    }

    pub fn tanhSoftcap(
        self: *const CudaDevice,
        d_x: [*]f32,
        cap: f32,
        n: usize,
    ) void {
        cuda_tanh_softcap(d_x, cap, @intCast(n), self.stream);
    }
};
