const std = @import("std");
const types = @import("types.zig");
const SamplerParams = types.SamplerParams;

pub const TokenProb = struct {
    id: u32,
    prob: f32,
    logit: f32,
};

pub const Sampler = struct {
    allocator: std.mem.Allocator,
    candidates: []TokenProb,
    prng: std.Random.DefaultPrng,

    pub fn init(allocator: std.mem.Allocator, vocab_size: usize, seed: u64) !*Sampler {
        const self = try allocator.create(Sampler);
        self.* = .{
            .allocator = allocator,
            .candidates = try allocator.alloc(TokenProb, vocab_size),
            .prng = std.Random.DefaultPrng.init(seed),
        };
        return self;
    }

    pub fn deinit(self: *Sampler) void {
        self.allocator.free(self.candidates);
        self.allocator.destroy(self);
    }

    pub fn sample(
        self: *Sampler,
        logits: []const f32,
        history: []const u32,
        params: SamplerParams,
    ) u32 {
        const vocab_size = logits.len;
        std.debug.assert(self.candidates.len >= vocab_size);

        // 1. Greedy (argmax) short-circuit if temp <= 0 or greedy requested
        if (params.greedy or params.temperature <= 0.0) {
            return sampleGreedy(logits);
        }

        // Initialize candidates
        for (0..vocab_size) |i| {
            self.candidates[i] = .{
                .id = @intCast(i),
                .prob = 0.0,
                .logit = logits[i],
            };
        }
        var active_count: usize = vocab_size;

        // 2. Apply Repetition, Frequency, and Presence penalties
        if (params.repetition_penalty != 1.0 or params.presence_penalty != 0.0 or params.frequency_penalty != 0.0) {
            const window_start = if (history.len > params.repetition_penalty_window)
                history.len - params.repetition_penalty_window
            else
                0;
            const window = history[window_start..];

            // Count frequencies in window
            for (window) |tok| {
                if (tok < vocab_size) {
                    var cand = &self.candidates[tok];
                    if (cand.logit > 0.0) {
                        cand.logit /= params.repetition_penalty;
                    } else {
                        cand.logit *= params.repetition_penalty;
                    }
                    cand.logit -= params.presence_penalty;
                }
            }
        }

        // 3. Apply Temperature
        const inv_temp = 1.0 / params.temperature;
        for (self.candidates[0..active_count]) |*c| {
            c.logit *= inv_temp;
        }

        // 4. Softmax over logits to get probabilities
        var max_logit: f32 = self.candidates[0].logit;
        for (self.candidates[1..active_count]) |c| {
            if (c.logit > max_logit) max_logit = c.logit;
        }

        var sum_exp: f32 = 0.0;
        for (self.candidates[0..active_count]) |*c| {
            const exp_val = @exp(c.logit - max_logit);
            c.prob = exp_val;
            sum_exp += exp_val;
        }

        const inv_sum = 1.0 / sum_exp;
        for (self.candidates[0..active_count]) |*c| {
            c.prob *= inv_sum;
        }

        // 5. Sort candidates descending by probability
        std.mem.sort(TokenProb, self.candidates[0..active_count], {}, struct {
            fn greaterThan(_: void, a: TokenProb, b: TokenProb) bool {
                return a.prob > b.prob;
            }
        }.greaterThan);

        // 6. Apply Top-K
        if (params.top_k > 0 and params.top_k < active_count) {
            active_count = params.top_k;
        }

        // 7. Apply Min-P
        if (params.min_p > 0.0 and active_count > 0) {
            const top_prob = self.candidates[0].prob;
            const min_thresh = top_prob * params.min_p;
            var i: usize = 0;
            while (i < active_count) : (i += 1) {
                if (self.candidates[i].prob < min_thresh) {
                    active_count = @max(1, i);
                    break;
                }
            }
        }

        // 8. Apply Top-P (Nucleus)
        if (params.top_p < 1.0 and active_count > 0) {
            var cum_sum: f32 = 0.0;
            var cutoff: usize = active_count;
            for (0..active_count) |i| {
                cum_sum += self.candidates[i].prob;
                if (cum_sum >= params.top_p) {
                    cutoff = i + 1;
                    break;
                }
            }
            active_count = cutoff;
        }

        // 9. Re-normalize remaining probabilities
        var new_sum: f32 = 0.0;
        for (self.candidates[0..active_count]) |c| {
            new_sum += c.prob;
        }

        // 10. Sample from categorical distribution
        const random_val = self.prng.random().float(f32) * new_sum;
        var running_sum: f32 = 0.0;
        for (self.candidates[0..active_count]) |c| {
            running_sum += c.prob;
            if (random_val <= running_sum) {
                return c.id;
            }
        }

        return self.candidates[0].id;
    }

    pub fn sampleGreedy(logits: []const f32) u32 {
        var max_idx: usize = 0;
        var max_val: f32 = logits[0];
        for (logits[1..], 1..) |val, i| {
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        return @intCast(max_idx);
    }
};

test "Sampler greedy selection" {
    const logits = [_]f32{ 0.1, 2.5, 0.3, -1.0 };
    const sampled = Sampler.sampleGreedy(&logits);
    try std.testing.expectEqual(@as(u32, 1), sampled);
}

test "Sampler stochastic selection" {
    const allocator = std.testing.allocator;
    var sampler = try Sampler.init(allocator, 4, 1234);
    defer sampler.deinit();

    const logits = [_]f32{ 10.0, 0.0, 0.0, 0.0 };
    const history = [_]u32{};
    const sampled = sampler.sample(&logits, &history, .{ .temperature = 0.7 });
    try std.testing.expectEqual(@as(u32, 0), sampled);
}
