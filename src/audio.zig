const std = @import("std");

pub const AudioFormat = enum {
    pcm16,
    pcm32f,
    unknown,
};

pub const AudioData = struct {
    sample_rate: u32,
    channels: u16,
    samples: []f32, // Normalized to [-1.0, 1.0]

    pub fn deinit(self: *AudioData, allocator: std.mem.Allocator) void {
        allocator.free(self.samples);
    }
};

/// Pure Zig WAV (RIFF/WAVE) audio decoder
pub fn loadWav(allocator: std.mem.Allocator, file_path: []const u8) !AudioData {
    const fd = try std.posix.openat(std.posix.AT.FDCWD, file_path, .{ .ACCMODE = .RDONLY }, 0);
    defer _ = std.posix.system.close(fd);

    const end_pos = std.posix.system.lseek(fd, 0, 2); // SEEK_END
    if (end_pos <= 0) return error.EmptyFile;
    _ = std.posix.system.lseek(fd, 0, 0); // SEEK_SET
    const size = @as(usize, @intCast(end_pos));

    const bytes = try allocator.alloc(u8, size);
    defer allocator.free(bytes);

    var total_read: usize = 0;
    while (total_read < size) {
        const n = try std.posix.read(fd, bytes[total_read..]);
        if (n == 0) break;
        total_read += n;
    }

    return parseWavBytes(allocator, bytes[0..total_read]);
}

pub fn parseWavBytes(allocator: std.mem.Allocator, bytes: []const u8) !AudioData {
    if (bytes.len < 44) return error.InvalidWavHeader;

    if (!std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE")) {
        return error.InvalidWavHeader;
    }

    var offset: usize = 12;
    var audio_format: u16 = 1;
    var channels: u16 = 1;
    var sample_rate: u32 = 16000;
    var bits_per_sample: u16 = 16;
    var data_slice: ?[]const u8 = null;

    while (offset + 8 <= bytes.len) {
        const chunk_id = bytes[offset .. offset + 4];
        const chunk_size = std.mem.readInt(u32, bytes[offset + 4 .. offset + 8][0..4], .little);
        offset += 8;

        if (std.mem.eql(u8, chunk_id, "fmt ")) {
            if (offset + 16 > bytes.len) return error.InvalidWavHeader;
            audio_format = std.mem.readInt(u16, bytes[offset .. offset + 2][0..2], .little);
            channels = std.mem.readInt(u16, bytes[offset + 2 .. offset + 4][0..2], .little);
            sample_rate = std.mem.readInt(u32, bytes[offset + 4 .. offset + 8][0..4], .little);
            bits_per_sample = std.mem.readInt(u16, bytes[offset + 14 .. offset + 16][0..2], .little);
        } else if (std.mem.eql(u8, chunk_id, "data")) {
            const end = @min(offset + chunk_size, bytes.len);
            data_slice = bytes[offset..end];
        }

        offset += (chunk_size + 1) & ~@as(usize, 1);
    }

    if (data_slice == null or channels == 0) return error.MissingWavData;

    const data = data_slice.?;
    var final_samples: []f32 = undefined;

    if (bits_per_sample == 16 and audio_format == 1) {
        const num_samples = data.len / 2;
        const total_frames = num_samples / channels;
        const samples = try allocator.alloc(f32, total_frames);
        errdefer allocator.free(samples);

        for (0..total_frames) |frame_idx| {
            var sum: f32 = 0.0;
            for (0..channels) |ch| {
                const s_idx = (frame_idx * channels + ch) * 2;
                if (s_idx + 2 <= data.len) {
                    const raw = std.mem.readInt(i16, data[s_idx .. s_idx + 2][0..2], .little);
                    sum += @as(f32, @floatFromInt(raw)) / 32768.0;
                }
            }
            samples[frame_idx] = sum / @as(f32, @floatFromInt(channels));
        }
        final_samples = samples;
    } else if (bits_per_sample == 32 and audio_format == 3) {
        // IEEE Float 32-bit
        const num_samples = data.len / 4;
        const total_frames = num_samples / channels;
        const samples = try allocator.alloc(f32, total_frames);
        errdefer allocator.free(samples);

        for (0..total_frames) |frame_idx| {
            var sum: f32 = 0.0;
            for (0..channels) |ch| {
                const s_idx = (frame_idx * channels + ch) * 4;
                if (s_idx + 4 <= data.len) {
                    const raw = @as(f32, @bitCast(std.mem.readInt(u32, data[s_idx .. s_idx + 4][0..4], .little)));
                    sum += raw;
                }
            }
            samples[frame_idx] = sum / @as(f32, @floatFromInt(channels));
        }
        final_samples = samples;
    } else {
        return error.UnsupportedWavFormat;
    }
    if (sample_rate != 16000 and sample_rate > 0) {
        const target_sr: u32 = 16000;
        const target_len = @as(usize, @intFromFloat(@as(f32, @floatFromInt(final_samples.len)) * (16000.0 / @as(f32, @floatFromInt(sample_rate)))));
        var resampled = try allocator.alloc(f32, target_len);
        errdefer allocator.free(resampled);

        const ratio = @as(f32, @floatFromInt(sample_rate)) / 16000.0;
        for (0..target_len) |i| {
            const src_pos = @as(f32, @floatFromInt(i)) * ratio;
            const idx0 = @as(usize, @intFromFloat(src_pos));
            const idx1 = @min(idx0 + 1, final_samples.len - 1);
            const frac = src_pos - @as(f32, @floatFromInt(idx0));
            resampled[i] = final_samples[idx0] * (1.0 - frac) + final_samples[idx1] * frac;
        }

        allocator.free(final_samples);
        final_samples = resampled;
        sample_rate = target_sr;
    }

    return AudioData{
        .sample_rate = sample_rate,
        .channels = 1,
        .samples = final_samples,
    };
}

/// Mel Filterbank & Log-Mel Spectrogram Generator
pub const LogMelSpectrogram = struct {
    allocator: std.mem.Allocator,
    n_mels: usize = 80,
    n_fft: usize = 512,
    hop_length: usize = 160,
    sample_rate: u32 = 16000,
    mel_filters: []f32, // [n_mels * (n_fft / 2 + 1)]

    pub fn init(allocator: std.mem.Allocator, n_mels: usize, n_fft: usize, hop_length: usize, sample_rate: u32) !LogMelSpectrogram {
        const num_freq_bins = n_fft / 2 + 1;
        const filter_size = n_mels * num_freq_bins;
        const mel_filters = try allocator.alloc(f32, filter_size);
        @memset(mel_filters, 0.0);

        // Precompute Triangular Mel filterbanks
        const f_min: f32 = 0.0;
        const f_max: f32 = @as(f32, @floatFromInt(sample_rate)) / 2.0;

        const mel_min = hzToMel(f_min);
        const mel_max = hzToMel(f_max);
        const mel_step = (mel_max - mel_min) / @as(f32, @floatFromInt(n_mels + 1));

        for (0..n_mels) |m| {
            const m_center = mel_min + @as(f32, @floatFromInt(m + 1)) * mel_step;
            const m_left = m_center - mel_step;
            const m_right = m_center + mel_step;

            const f_left = melToHz(m_left);
            const f_center = melToHz(m_center);
            const f_right = melToHz(m_right);

            for (0..num_freq_bins) |k| {
                const freq = @as(f32, @floatFromInt(k)) * @as(f32, @floatFromInt(sample_rate)) / @as(f32, @floatFromInt(n_fft));
                var weight: f32 = 0.0;
                if (freq >= f_left and freq <= f_center and f_center > f_left) {
                    weight = (freq - f_left) / (f_center - f_left);
                } else if (freq > f_center and freq <= f_right and f_right > f_center) {
                    weight = (f_right - freq) / (f_right - f_center);
                }
                mel_filters[m * num_freq_bins + k] = weight;
            }
        }

        return LogMelSpectrogram{
            .allocator = allocator,
            .n_mels = n_mels,
            .n_fft = n_fft,
            .hop_length = hop_length,
            .sample_rate = sample_rate,
            .mel_filters = mel_filters,
        };
    }

    pub fn deinit(self: *LogMelSpectrogram) void {
        self.allocator.free(self.mel_filters);
    }

    pub fn compute(self: *const LogMelSpectrogram, samples: []const f32) ![]f32 {
        if (samples.len < self.n_fft) return error.AudioTooShort;

        const num_freq_bins = self.n_fft / 2 + 1;
        const n_frames = (samples.len - self.n_fft) / self.hop_length + 1;
        const result = try self.allocator.alloc(f32, n_frames * self.n_mels);
        errdefer self.allocator.free(result);

        // Precompute Hanning window
        var window: [512]f32 = undefined;
        const n_fft_f = @as(f32, @floatFromInt(self.n_fft));
        for (0..self.n_fft) |i| {
            const angle = 2.0 * std.math.pi * @as(f32, @floatFromInt(i)) / (n_fft_f - 1.0);
            window[i] = 0.5 * (1.0 - @cos(angle));
        }

        var power_spectrum: [257]f32 = undefined;

        for (0..n_frames) |frame_idx| {
            const frame_start = frame_idx * self.hop_length;
            const frame = samples[frame_start .. frame_start + self.n_fft];

            // Compute DFT power spectrum for real windowed signal
            for (0..num_freq_bins) |k| {
                var real: f32 = 0.0;
                var imag: f32 = 0.0;
                const k_f = @as(f32, @floatFromInt(k));

                for (0..self.n_fft) |n| {
                    const val = frame[n] * window[n];
                    const theta = 2.0 * std.math.pi * k_f * @as(f32, @floatFromInt(n)) / n_fft_f;
                    real += val * @cos(theta);
                    imag -= val * @sin(theta);
                }
                power_spectrum[k] = (real * real + imag * imag) / n_fft_f;
            }

            // Apply Mel Filterbanks & Log scale
            for (0..self.n_mels) |m| {
                var mel_energy: f32 = 0.0;
                const filter_offset = m * num_freq_bins;
                for (0..num_freq_bins) |k| {
                    mel_energy += power_spectrum[k] * self.mel_filters[filter_offset + k];
                }
                const log_val = @log10(@max(mel_energy, 1e-5));
                result[frame_idx * self.n_mels + m] = (log_val + 4.0) / 4.0; // Normalized log mel
            }
        }

        return result;
    }

    fn hzToMel(hz: f32) f32 {
        return 2595.0 * @log10(1.0 + hz / 700.0);
    }

    fn melToHz(mel: f32) f32 {
        return 700.0 * (std.math.pow(f32, 10.0, mel / 2595.0) - 1.0);
    }
};

const types = @import("types.zig");
const math = @import("math.zig");
const quant = @import("quant.zig");
const ThreadPool = @import("thread_pool.zig").ThreadPool;

pub const AudioLayerWeights = struct {
    // FFN 0
    ffn_norm: ?[]const f32 = null,
    ffn_up: ?types.Tensor = null,
    ffn_down: ?types.Tensor = null,
    ffn_post_norm: ?[]const f32 = null,
    
    // Attention
    attn_pre_norm: ?[]const f32 = null,
    attn_q: ?types.Tensor = null,
    attn_k: ?types.Tensor = null,
    attn_v: ?types.Tensor = null,
    attn_k_rel: ?types.Tensor = null,
    per_dim_scale: ?[]const f32 = null,
    attn_out: ?types.Tensor = null,
    attn_post_norm: ?[]const f32 = null,
    
    // Conv
    norm_conv: ?[]const f32 = null,
    conv_pw1: ?types.Tensor = null,
    conv_dw: ?types.Tensor = null,
    conv_norm: ?[]const f32 = null,
    conv_pw2: ?types.Tensor = null,
    
    // FFN 1
    ffn_norm_1: ?[]const f32 = null,
    ffn_up_1: ?types.Tensor = null,
    ffn_down_1: ?types.Tensor = null,
    ffn_post_norm_1: ?[]const f32 = null,
    
    // Final LN
    ln2: ?[]const f32 = null,
};

pub const AudioEncoder = struct {
    allocator: std.mem.Allocator,
    conv1d_0_weight: ?types.Tensor = null,
    conv1d_0_norm: ?[]const f32 = null,
    conv1d_1_weight: ?types.Tensor = null,
    conv1d_1_norm: ?[]const f32 = null,
    input_proj: ?types.Tensor = null,
    pre_encode_out: ?types.Tensor = null,
    pre_encode_bias: ?[]const f32 = null,
    output_proj: ?types.Tensor = null,
    layers: []AudioLayerWeights,
    hidden_size: usize = 1024,
    intermediate_size: usize = 4096,
    num_heads: usize = 8,
    head_dim: usize = 128,
    llm_dim: usize = 1536,

    pub fn init(
        allocator: std.mem.Allocator,
        conv0_w: ?types.Tensor,
        conv0_norm: ?[]const f32,
        conv1_w: ?types.Tensor,
        conv1_norm: ?[]const f32,
        inp_proj: ?types.Tensor,
        pre_out: ?types.Tensor,
        pre_bias: ?[]const f32,
        out_proj: ?types.Tensor,
        layers: []AudioLayerWeights,
    ) AudioEncoder {
        return .{
            .allocator = allocator,
            .conv1d_0_weight = conv0_w,
            .conv1d_0_norm = conv0_norm,
            .conv1d_1_weight = conv1_w,
            .conv1d_1_norm = conv1_norm,
            .input_proj = inp_proj,
            .pre_encode_out = pre_out,
            .pre_encode_bias = pre_bias,
            .output_proj = out_proj,
            .layers = layers,
        };
    }

    pub fn deinit(self: *AudioEncoder) void {
        if (self.conv1d_0_norm) |n| self.allocator.free(n);
        if (self.conv1d_1_norm) |n| self.allocator.free(n);
        if (self.pre_encode_bias) |b| self.allocator.free(b);
        for (self.layers) |l| {
            if (l.ffn_norm) |n| self.allocator.free(n);
            if (l.ffn_post_norm) |n| self.allocator.free(n);
            if (l.attn_pre_norm) |n| self.allocator.free(n);
            if (l.per_dim_scale) |n| self.allocator.free(n);
            if (l.attn_post_norm) |n| self.allocator.free(n);
            if (l.norm_conv) |n| self.allocator.free(n);
            if (l.conv_norm) |n| self.allocator.free(n);
            if (l.ffn_norm_1) |n| self.allocator.free(n);
            if (l.ffn_post_norm_1) |n| self.allocator.free(n);
            if (l.ln2) |n| self.allocator.free(n);
        }
        self.allocator.free(self.layers);
    }

    pub fn encode(self: *const AudioEncoder, allocator: std.mem.Allocator, mel_spec: []const f32, pool: ?*ThreadPool) ![]f32 {
        const num_mel_frames = mel_spec.len / 128;
        if (num_mel_frames == 0) return try allocator.alloc(f32, 0);

        // --- Subsampling ---
        const t2 = num_mel_frames / 2;
        const conv0_out = try allocator.alloc(f32, t2 * 64 * 128);
        defer allocator.free(conv0_out);
        @memset(conv0_out, 0.0);
        
        if (self.conv1d_0_weight) |w0| {
            const w_slice = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(w0.data)));
            for (0..t2) |t_out| {
                for (0..64) |f_out| {
                    for (0..128) |oc| {
                        var sum: f32 = 0.0;
                        for (0..3) |ky| {
                            for (0..3) |kx| {
                                const t_in: isize = @as(isize, @intCast(t_out * 2)) + @as(isize, @intCast(ky)) - 1;
                                const f_in: isize = @as(isize, @intCast(f_out * 2)) + @as(isize, @intCast(kx)) - 1;
                                if (t_in >= 0 and t_in < num_mel_frames and f_in >= 0 and f_in < 128) {
                                    const mel_val = mel_spec[@as(usize, @intCast(t_in)) * 128 + @as(usize, @intCast(f_in))];
                                    sum += mel_val * w_slice[oc * 9 + ky * 3 + kx];
                                }
                            }
                        }
                        conv0_out[(t_out * 64 + f_out) * 128 + oc] = sum;
                    }
                }
            }
            if (self.conv1d_0_norm) |norm| {
                for (0..t2 * 64) |i| {
                    const slice = conv0_out[i * 128 .. (i + 1) * 128];
                    // LayerNorm with elementwise_affine=True (wait, bias=False!)
                    // PyTorch LayerNorm has weight but bias=False in Gemma4
                    // Let's implement standard LayerNorm: mean, var over the last dim
                    var mean: f32 = 0;
                    for (slice) |v| mean += v;
                    mean /= 128.0;
                    var var_: f32 = 0;
                    for (slice) |v| var_ += (v - mean) * (v - mean);
                    var_ /= 128.0;
                    const inv_std = 1.0 / @sqrt(var_ + 1e-6);
                    for (slice, 0..) |*v, d| {
                        v.* = (v.* - mean) * inv_std * norm[d];
                        if (v.* < 0) v.* = 0; // ReLU
                    }
                }
            }
        }

        const t4 = t2 / 2;
        const conv1_out = try allocator.alloc(f32, t4 * 32 * 32); // output: [T/4, 32_freq, 32_channels] -> 32*32=1024
        defer allocator.free(conv1_out);
        @memset(conv1_out, 0.0);
        
        if (self.conv1d_1_weight) |w1| {
            const w_slice = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(w1.data)));
            for (0..t4) |t_out| {
                for (0..32) |f_out| {
                    for (0..32) |oc| {
                        var sum: f32 = 0.0;
                        for (0..3) |ky| {
                            for (0..3) |kx| {
                                const t_in: isize = @as(isize, @intCast(t_out * 2)) + @as(isize, @intCast(ky)) - 1;
                                const f_in: isize = @as(isize, @intCast(f_out * 2)) + @as(isize, @intCast(kx)) - 1;
                                if (t_in >= 0 and t_in < t2 and f_in >= 0 and f_in < 64) {
                                    for (0..128) |ic| {
                                        const val = conv0_out[(@as(usize, @intCast(t_in)) * 64 + @as(usize, @intCast(f_in))) * 128 + ic];
                                        sum += val * w_slice[(oc * 128 + ic) * 9 + ky * 3 + kx];
                                    }
                                }
                            }
                        }
                        conv1_out[(t_out * 32 + f_out) * 32 + oc] = sum;
                    }
                }
            }
            if (self.conv1d_1_norm) |norm| {
                for (0..t4 * 32) |i| {
                    const slice = conv1_out[i * 32 .. (i + 1) * 32];
                    var mean: f32 = 0;
                    for (slice) |v| mean += v;
                    mean /= 32.0;
                    var var_: f32 = 0;
                    for (slice) |v| var_ += (v - mean) * (v - mean);
                    var_ /= 32.0;
                    const inv_std = 1.0 / @sqrt(var_ + 1e-6);
                    for (slice, 0..) |*v, d| {
                        v.* = (v.* - mean) * inv_std * norm[d];
                        if (v.* < 0) v.* = 0; // ReLU
                    }
                }
            }
        }

        const target_frames = t4;
        const states = try allocator.alloc(f32, target_frames * self.hidden_size);
        defer allocator.free(states);
        @memset(states, 0.0);
        
        // Reshape [t4, 32_freq, 32_ch] -> [t4, 1024] -> linear project to states
        if (self.input_proj) |proj| {
            // conv1_out is [t4, 32, 32] which is flat [t4, 1024].
            math.gemm(pool, proj.type, proj.data, conv1_out, states, target_frames, self.hidden_size, 1024);
        } else {
            @memcpy(states, conv1_out);
        }

                // Now apply conformer layers
        for (self.layers) |layer| {
            const num_frames = target_frames;
            
            // --- FFN 1 ---
            if (layer.ffn_norm) |norm| {
                const p_norm = try allocator.alloc(f32, num_frames * self.hidden_size);
                defer allocator.free(p_norm);
                
                for (0..num_frames) |t| {
                    const src = states[t * self.hidden_size .. (t + 1) * self.hidden_size];
                    const dst = p_norm[t * self.hidden_size .. (t + 1) * self.hidden_size];
                    var mean: f32 = 0; for (src) |v| mean += v; mean /= @as(f32, @floatFromInt(self.hidden_size));
                    var var_: f32 = 0; for (src) |v| var_ += (v - mean) * (v - mean); var_ /= @as(f32, @floatFromInt(self.hidden_size));
                    const inv_std = 1.0 / @sqrt(var_ + 1e-6);
                    for (src, 0..) |v, d| dst[d] = (v - mean) * inv_std * norm[d];
                }
                
                const p_ffn_up = try allocator.alloc(f32, num_frames * self.intermediate_size);
                defer allocator.free(p_ffn_up);
                if (layer.ffn_up) |up| {
                    math.gemm(pool, up.type, up.data, p_norm, p_ffn_up, num_frames, self.intermediate_size, self.hidden_size);
                }
                
                // SiLU
                for (p_ffn_up) |*v| v.* = v.* / (1.0 + @exp(-v.*));
                
                const p_ffn_down = try allocator.alloc(f32, num_frames * self.hidden_size);
                defer allocator.free(p_ffn_down);
                if (layer.ffn_down) |down| {
                    math.gemm(pool, down.type, down.data, p_ffn_up, p_ffn_down, num_frames, self.hidden_size, self.intermediate_size);
                }
                
                if (layer.ffn_post_norm) |pnorm| {
                    for (0..num_frames) |t| {
                        const src = p_ffn_down[t * self.hidden_size .. (t + 1) * self.hidden_size];
                        var mean: f32 = 0; for (src) |v| mean += v; mean /= @as(f32, @floatFromInt(self.hidden_size));
                        var var_: f32 = 0; for (src) |v| var_ += (v - mean) * (v - mean); var_ /= @as(f32, @floatFromInt(self.hidden_size));
                        const inv_std = 1.0 / @sqrt(var_ + 1e-6);
                        for (src, 0..) |*v, d| states[t * self.hidden_size + d] += 0.5 * ((v.* - mean) * inv_std * pnorm[d]);
                    }
                }
            }
            
            // --- Attention ---
            if (layer.attn_pre_norm) |norm| {
                const p_norm = try allocator.alloc(f32, num_frames * self.hidden_size);
                defer allocator.free(p_norm);
                
                for (0..num_frames) |t| {
                    const src = states[t * self.hidden_size .. (t + 1) * self.hidden_size];
                    const dst = p_norm[t * self.hidden_size .. (t + 1) * self.hidden_size];
                    var mean: f32 = 0; for (src) |v| mean += v; mean /= @as(f32, @floatFromInt(self.hidden_size));
                    var var_: f32 = 0; for (src) |v| var_ += (v - mean) * (v - mean); var_ /= @as(f32, @floatFromInt(self.hidden_size));
                    const inv_std = 1.0 / @sqrt(var_ + 1e-6);
                    for (src, 0..) |v, d| dst[d] = (v - mean) * inv_std * norm[d];
                }
                
                const q_buf = try allocator.alloc(f32, num_frames * self.hidden_size); defer allocator.free(q_buf);
                const k_buf = try allocator.alloc(f32, num_frames * self.hidden_size); defer allocator.free(k_buf);
                const v_buf = try allocator.alloc(f32, num_frames * self.hidden_size); defer allocator.free(v_buf);
                
                if (layer.attn_q) |t| math.gemm(pool, t.type, t.data, p_norm, q_buf, num_frames, self.hidden_size, self.hidden_size);
                if (layer.attn_k) |t| math.gemm(pool, t.type, t.data, p_norm, k_buf, num_frames, self.hidden_size, self.hidden_size);
                if (layer.attn_v) |t| math.gemm(pool, t.type, t.data, p_norm, v_buf, num_frames, self.hidden_size, self.hidden_size);
                
                const attn_out = try allocator.alloc(f32, num_frames * self.hidden_size); defer allocator.free(attn_out);
                
                // Very naive full attention (ignoring chunking/rel pos for simplicity)
                const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
                for (0..self.num_heads) |h| {
                    for (0..num_frames) |t_q| {
                        var max_score: f32 = -std.math.inf(f32);
                        const scores = try allocator.alloc(f32, num_frames);
                        defer allocator.free(scores);
                        
                        for (0..num_frames) |t_k| {
                            var score: f32 = 0;
                            for (0..self.head_dim) |d| {
                                score += q_buf[t_q * self.hidden_size + h * self.head_dim + d] * k_buf[t_k * self.hidden_size + h * self.head_dim + d];
                            }
                            score *= scale;
                            scores[t_k] = score;
                            if (score > max_score) max_score = score;
                        }
                        
                        var sum_exp: f32 = 0;
                        for (0..num_frames) |t_k| {
                            scores[t_k] = @exp(scores[t_k] - max_score);
                            sum_exp += scores[t_k];
                        }
                        
                        for (0..self.head_dim) |d| {
                            var val: f32 = 0;
                            for (0..num_frames) |t_k| {
                                val += (scores[t_k] / sum_exp) * v_buf[t_k * self.hidden_size + h * self.head_dim + d];
                            }
                            attn_out[t_q * self.hidden_size + h * self.head_dim + d] = val;
                        }
                    }
                }
                
                const out_proj = try allocator.alloc(f32, num_frames * self.hidden_size); defer allocator.free(out_proj);
                if (layer.attn_out) |t| math.gemm(pool, t.type, t.data, attn_out, out_proj, num_frames, self.hidden_size, self.hidden_size);
                
                if (layer.attn_post_norm) |pnorm| {
                    for (0..num_frames) |t| {
                        const src = out_proj[t * self.hidden_size .. (t + 1) * self.hidden_size];
                        var mean: f32 = 0; for (src) |v| mean += v; mean /= @as(f32, @floatFromInt(self.hidden_size));
                        var var_: f32 = 0; for (src) |v| var_ += (v - mean) * (v - mean); var_ /= @as(f32, @floatFromInt(self.hidden_size));
                        const inv_std = 1.0 / @sqrt(var_ + 1e-6);
                        for (src, 0..) |*v, d| states[t * self.hidden_size + d] += ((v.* - mean) * inv_std * pnorm[d]);
                    }
                }
            }
            
            // --- Conv Module ---
            if (layer.norm_conv) |norm| {
                const p_norm = try allocator.alloc(f32, num_frames * self.hidden_size); defer allocator.free(p_norm);
                for (0..num_frames) |t| {
                    const src = states[t * self.hidden_size .. (t + 1) * self.hidden_size];
                    const dst = p_norm[t * self.hidden_size .. (t + 1) * self.hidden_size];
                    var mean: f32 = 0; for (src) |v| mean += v; mean /= @as(f32, @floatFromInt(self.hidden_size));
                    var var_: f32 = 0; for (src) |v| var_ += (v - mean) * (v - mean); var_ /= @as(f32, @floatFromInt(self.hidden_size));
                    const inv_std = 1.0 / @sqrt(var_ + 1e-6);
                    for (src, 0..) |v, d| dst[d] = (v - mean) * inv_std * norm[d];
                }
                
                const pw1_out = try allocator.alloc(f32, num_frames * self.hidden_size * 2); defer allocator.free(pw1_out);
                if (layer.conv_pw1) |pw1| math.gemm(pool, pw1.type, pw1.data, p_norm, pw1_out, num_frames, self.hidden_size * 2, self.hidden_size);
                
                // GLU
                const glu_out = try allocator.alloc(f32, num_frames * self.hidden_size); defer allocator.free(glu_out);
                for (0..num_frames) |t| {
                    for (0..self.hidden_size) |d| {
                        const v1 = pw1_out[t * self.hidden_size * 2 + d];
                        const v2 = pw1_out[t * self.hidden_size * 2 + self.hidden_size + d];
                        glu_out[t * self.hidden_size + d] = v1 * (1.0 / (1.0 + @exp(-v2)));
                    }
                }
                
                // 1D Depthwise Conv (kernel=5, causal: left_pad=4)
                const dw_out = try allocator.alloc(f32, num_frames * self.hidden_size); defer allocator.free(dw_out);
                if (layer.conv_dw) |dw| {
                    const w_slice = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(dw.data)));
                    for (0..num_frames) |t| {
                        for (0..self.hidden_size) |d| {
                            var sum: f32 = 0;
                            // kernel=5, causal padding: left_pad=4, right_pad=0
                            for (0..5) |k| {
                                const t_in: isize = @as(isize, @intCast(t)) + @as(isize, @intCast(k)) - 4;
                                if (t_in >= 0 and t_in < num_frames) {
                                    // GGUF stores [5, 1024] = [kernel, channels]
                                    sum += glu_out[@as(usize, @intCast(t_in)) * self.hidden_size + d] * w_slice[k * self.hidden_size + d];
                                }
                            }
                            dw_out[t * self.hidden_size + d] = sum;
                        }
                    }
                }
                
                // Conv Norm (LayerNorm)
                if (layer.conv_norm) |cnorm| {
                    for (0..num_frames) |t| {
                        const src = dw_out[t * self.hidden_size .. (t + 1) * self.hidden_size];
                        var mean: f32 = 0; for (src) |v| mean += v; mean /= @as(f32, @floatFromInt(self.hidden_size));
                        var var_: f32 = 0; for (src) |v| var_ += (v - mean) * (v - mean); var_ /= @as(f32, @floatFromInt(self.hidden_size));
                        const inv_std = 1.0 / @sqrt(var_ + 1e-6);
                        for (src, 0..) |*v, d| v.* = (v.* - mean) * inv_std * cnorm[d];
                    }
                }
                
                // SiLU
                for (dw_out) |*v| v.* = v.* / (1.0 + @exp(-v.*));
                
                const pw2_out = try allocator.alloc(f32, num_frames * self.hidden_size); defer allocator.free(pw2_out);
                if (layer.conv_pw2) |pw2| math.gemm(pool, pw2.type, pw2.data, dw_out, pw2_out, num_frames, self.hidden_size, self.hidden_size);
                
                for (0..num_frames) |t| {
                    for (0..self.hidden_size) |d| {
                        states[t * self.hidden_size + d] += pw2_out[t * self.hidden_size + d];
                    }
                }
            }
            
            // --- FFN 2 ---
            if (layer.ffn_norm_1) |norm| {
                const p_norm = try allocator.alloc(f32, num_frames * self.hidden_size); defer allocator.free(p_norm);
                for (0..num_frames) |t| {
                    const src = states[t * self.hidden_size .. (t + 1) * self.hidden_size];
                    const dst = p_norm[t * self.hidden_size .. (t + 1) * self.hidden_size];
                    var mean: f32 = 0; for (src) |v| mean += v; mean /= @as(f32, @floatFromInt(self.hidden_size));
                    var var_: f32 = 0; for (src) |v| var_ += (v - mean) * (v - mean); var_ /= @as(f32, @floatFromInt(self.hidden_size));
                    const inv_std = 1.0 / @sqrt(var_ + 1e-6);
                    for (src, 0..) |v, d| dst[d] = (v - mean) * inv_std * norm[d];
                }
                
                const p_ffn_up = try allocator.alloc(f32, num_frames * self.intermediate_size); defer allocator.free(p_ffn_up);
                if (layer.ffn_up_1) |up| math.gemm(pool, up.type, up.data, p_norm, p_ffn_up, num_frames, self.intermediate_size, self.hidden_size);
                
                for (p_ffn_up) |*v| v.* = v.* / (1.0 + @exp(-v.*));
                
                const p_ffn_down = try allocator.alloc(f32, num_frames * self.hidden_size); defer allocator.free(p_ffn_down);
                if (layer.ffn_down_1) |down| math.gemm(pool, down.type, down.data, p_ffn_up, p_ffn_down, num_frames, self.hidden_size, self.intermediate_size);
                
                if (layer.ffn_post_norm_1) |pnorm| {
                    for (0..num_frames) |t| {
                        const src = p_ffn_down[t * self.hidden_size .. (t + 1) * self.hidden_size];
                        var mean: f32 = 0; for (src) |v| mean += v; mean /= @as(f32, @floatFromInt(self.hidden_size));
                        var var_: f32 = 0; for (src) |v| var_ += (v - mean) * (v - mean); var_ /= @as(f32, @floatFromInt(self.hidden_size));
                        const inv_std = 1.0 / @sqrt(var_ + 1e-6);
                        for (src, 0..) |*v, d| states[t * self.hidden_size + d] += 0.5 * ((v.* - mean) * inv_std * pnorm[d]);
                    }
                }
            }
            
            // --- LN2 ---
            if (layer.ln2) |ln2| {
                for (0..num_frames) |t| {
                    const src = states[t * self.hidden_size .. (t + 1) * self.hidden_size];
                    var mean: f32 = 0; for (src) |v| mean += v; mean /= @as(f32, @floatFromInt(self.hidden_size));
                    var var_: f32 = 0; for (src) |v| var_ += (v - mean) * (v - mean); var_ /= @as(f32, @floatFromInt(self.hidden_size));
                    const inv_std = 1.0 / @sqrt(var_ + 1e-6);
                    for (src, 0..) |*v, d| v.* = (v.* - mean) * inv_std * ln2[d];
                }
            }
        }
const out_embeddings = try allocator.alloc(f32, target_frames * self.llm_dim);
        
        // Output projection: conformer_states -> pre_encode_out (1024->1536) -> RMSNorm -> mm.a.input_projection (1536->1536)
        for (0..target_frames) |t| {
            const p_state = states[t * self.hidden_size .. (t + 1) * self.hidden_size];
            const p_out = out_embeddings[t * self.llm_dim .. (t + 1) * self.llm_dim];

            if (self.pre_encode_out) |pe_out| {
                math.gemv(pool, pe_out.type, pe_out.data, p_state, p_out, self.llm_dim, self.hidden_size);
                if (self.pre_encode_bias) |b| {
                    for (0..self.llm_dim) |d| p_out[d] += b[d];
                }
            } else {
                @memset(p_out, 0.0);
                @memcpy(p_out[0..@min(self.hidden_size, self.llm_dim)], p_state[0..@min(self.hidden_size, self.llm_dim)]);
            }

            // Apply embed_audio: RMSNorm(no scale) + mm.a.input_projection
            if (self.output_proj) |mm_proj| {
                // RMSNorm without scale weights
                var sum_sq: f32 = 0.0;
                for (p_out[0..self.llm_dim]) |v| sum_sq += v * v;
                const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(self.llm_dim)) + 1e-6);
                const inv_rms = 1.0 / rms;
                var normed = allocator.alloc(f32, self.llm_dim) catch unreachable;
                defer allocator.free(normed);
                for (0..self.llm_dim) |d| normed[d] = p_out[d] * inv_rms;

                // mm.a.input_projection: Linear(1536, 1536, bias=False)
                math.gemv(pool, mm_proj.type, mm_proj.data, normed, p_out, self.llm_dim, self.llm_dim);
            }
        }

        return out_embeddings;
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "Audio WAV parsing and Log-Mel Spectrogram computation" {
    const allocator = std.testing.allocator;

    // Generate synthetic 16kHz sine wave WAV buffer
    const sample_rate: u32 = 16000;
    const num_samples: usize = 16000; // 1 second of 440Hz sine wave
    var wav_bytes = try allocator.alloc(u8, 44 + num_samples * 2);
    defer allocator.free(wav_bytes);

    // RIFF Header
    @memcpy(wav_bytes[0..4], "RIFF");
    std.mem.writeInt(u32, wav_bytes[4..8], @intCast(36 + num_samples * 2), .little);
    @memcpy(wav_bytes[8..12], "WAVE");

    // fmt chunk
    @memcpy(wav_bytes[12..16], "fmt ");
    std.mem.writeInt(u32, wav_bytes[16..20], 16, .little);
    std.mem.writeInt(u16, wav_bytes[20..22], 1, .little); // PCM
    std.mem.writeInt(u16, wav_bytes[22..24], 1, .little); // 1 channel
    std.mem.writeInt(u32, wav_bytes[24..28], sample_rate, .little);
    std.mem.writeInt(u32, wav_bytes[28..32], sample_rate * 2, .little); // Byte rate
    std.mem.writeInt(u16, wav_bytes[32..34], 2, .little); // Block align
    std.mem.writeInt(u16, wav_bytes[34..36], 16, .little); // 16-bit
    @memcpy(wav_bytes[36..40], "data");
    std.mem.writeInt(u32, wav_bytes[40..44], @intCast(num_samples * 2), .little);

    for (0..num_samples) |i| {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(sample_rate));
        const val = @sin(2.0 * std.math.pi * 440.0 * t) * 0.8;
        const val_i16 = @as(i16, @intFromFloat(val * 32767.0));
        std.mem.writeInt(i16, wav_bytes[44 + i * 2 .. 44 + (i + 1) * 2][0..2], val_i16, .little);
    }

    var audio = try parseWavBytes(allocator, wav_bytes);
    defer audio.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 16000), audio.sample_rate);
    try std.testing.expectEqual(num_samples, audio.samples.len);

    var mel_gen = try LogMelSpectrogram.init(allocator, 80, 512, 160, 16000);
    defer mel_gen.deinit();

    const spec = try mel_gen.compute(audio.samples);
    defer allocator.free(spec);

    try std.testing.expect(spec.len > 0);
}
