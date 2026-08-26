const std = @import("std");
const Atomic = std.atomic.Value;

pub const TaskFn = *const fn (ctx: ?*anyopaque, item_idx: usize, thread_idx: usize) void;
pub const RangeTaskFn = *const fn (ctx: ?*anyopaque, start: usize, end: usize, thread_idx: usize) void;

const WorkerState = struct {
    start: usize = 0,
    end: usize = 0,
    task_fn: ?TaskFn = null,
    range_task_fn: ?RangeTaskFn = null,
    task_ctx: ?*anyopaque = null,
    epoch: Atomic(u64) = Atomic(u64).init(0),
    done_epoch: Atomic(u64) = Atomic(u64).init(0),
    _pad: [16]u8 = undefined,
};

pub const ThreadPool = struct {
    threads: []std.Thread,
    workers: []WorkerState,
    allocator: std.mem.Allocator,
    num_threads: usize,
    current_epoch: u64 = 0,
    shutdown: Atomic(bool) = Atomic(bool).init(false),

    pub fn init(allocator: std.mem.Allocator, num_threads_opt: ?usize) !*ThreadPool {
        const num_threads = num_threads_opt orelse blk: {
            const cpus = std.Thread.getCpuCount() catch 4;
            if (cpus >= 16) {
                break :blk cpus / 2;
            } else if (cpus > 8) {
                break :blk 8;
            } else {
                break :blk @max(1, cpus);
            }
        };

        const self = try allocator.create(ThreadPool);
        self.* = .{
            .threads = &[_]std.Thread{},
            .workers = &[_]WorkerState{},
            .allocator = allocator,
            .num_threads = num_threads,
            .current_epoch = 0,
        };

        if (num_threads > 1) {
            const n_workers = num_threads - 1;
            self.threads = try allocator.alloc(std.Thread, n_workers);
            errdefer allocator.free(self.threads);

            self.workers = try allocator.alloc(WorkerState, n_workers);
            errdefer allocator.free(self.workers);

            for (0..n_workers) |i| {
                self.workers[i] = .{};
                self.threads[i] = try std.Thread.spawn(.{}, workerLoop, .{ self, i });
            }
        }

        return self;
    }

    pub fn deinit(self: *ThreadPool) void {
        if (self.num_threads > 1) {
            self.shutdown.store(true, .release);

            for (self.threads) |t| {
                t.join();
            }
            self.allocator.free(self.threads);
            self.allocator.free(self.workers);
        }
        self.allocator.destroy(self);
    }

    fn workerLoop(self: *ThreadPool, worker_idx: usize) void {
        var w = &self.workers[worker_idx];
        var last_epoch: u64 = 0;

        while (!self.shutdown.load(.acquire)) {
            const job_epoch = w.epoch.load(.acquire);
            if (job_epoch != last_epoch) {
                if (w.range_task_fn) |rf| {
                    rf(w.task_ctx, w.start, w.end, worker_idx + 1);
                } else if (w.task_fn) |f| {
                    var idx = w.start;
                    while (idx < w.end) : (idx += 1) {
                        f(w.task_ctx, idx, worker_idx + 1);
                    }
                }
                last_epoch = job_epoch;
                w.done_epoch.store(job_epoch, .release);
            } else {
                std.atomic.spinLoopHint();
            }
        }
    }

    pub fn parallelFor(self: *ThreadPool, total_items: usize, ctx: ?*anyopaque, task_fn: TaskFn) void {
        if (total_items == 0) return;

        if (self.num_threads <= 1 or total_items == 1) {
            for (0..total_items) |i| {
                task_fn(ctx, i, 0);
            }
            return;
        }

        self.current_epoch += 1;
        const epoch = self.current_epoch;

        const n_threads = self.num_threads;
        const chunk_size = (total_items + n_threads - 1) / n_threads;

        // Dispatch chunks to workers (1 .. n_threads - 1)
        const n_workers = self.workers.len;
        for (0..n_workers) |i| {
            const thread_id = i + 1;
            const start = @min(total_items, thread_id * chunk_size);
            const end = @min(total_items, start + chunk_size);

            self.workers[i].start = start;
            self.workers[i].end = end;
            self.workers[i].task_fn = task_fn;
            self.workers[i].range_task_fn = null;
            self.workers[i].task_ctx = ctx;
            self.workers[i].epoch.store(epoch, .release);
        }

        // Main thread executes chunk 0
        const main_start: usize = 0;
        const main_end = @min(total_items, chunk_size);
        for (main_start..main_end) |idx| {
            task_fn(ctx, idx, 0);
        }

        // Wait for all workers to complete this epoch
        for (0..n_workers) |i| {
            var spins: usize = 0;
            while (self.workers[i].done_epoch.load(.acquire) != epoch) {
                spins += 1;
                if (spins < 10000) {
                    std.atomic.spinLoopHint();
                } else {
                    std.Thread.yield() catch {};
                }
            }
        }
    }

    pub fn parallelForRange(self: *ThreadPool, total_items: usize, ctx: ?*anyopaque, range_fn: RangeTaskFn) void {
        if (total_items == 0) return;

        if (self.num_threads <= 1 or total_items == 1) {
            range_fn(ctx, 0, total_items, 0);
            return;
        }

        self.current_epoch += 1;
        const epoch = self.current_epoch;

        const n_threads = self.num_threads;
        const chunk_size = (total_items + n_threads - 1) / n_threads;

        // Dispatch chunks to workers (1 .. n_threads - 1)
        const n_workers = self.workers.len;
        for (0..n_workers) |i| {
            const thread_id = i + 1;
            const start = @min(total_items, thread_id * chunk_size);
            const end = @min(total_items, start + chunk_size);

            self.workers[i].start = start;
            self.workers[i].end = end;
            self.workers[i].task_fn = null;
            self.workers[i].range_task_fn = range_fn;
            self.workers[i].task_ctx = ctx;
            self.workers[i].epoch.store(epoch, .release);
        }

        // Main thread executes chunk 0
        const main_start: usize = 0;
        const main_end = @min(total_items, chunk_size);
        if (main_end > main_start) {
            range_fn(ctx, main_start, main_end, 0);
        }

        // Wait for all workers to complete this epoch
        for (0..n_workers) |i| {
            var spins: usize = 0;
            while (self.workers[i].done_epoch.load(.acquire) != epoch) {
                spins += 1;
                if (spins < 10000) {
                    std.atomic.spinLoopHint();
                } else {
                    std.Thread.yield() catch {};
                }
            }
        }
    }
};

test "ThreadPool parallelFor correctness" {
    const allocator = std.testing.allocator;
    var pool = try ThreadPool.init(allocator, 4);
    defer pool.deinit();

    const N = 100;
    var data: [N]usize = undefined;
    for (0..N) |i| data[i] = 0;

    const Context = struct {
        arr: []usize,
    };
    var ctx = Context{ .arr = &data };

    const worker = struct {
        fn run(c: ?*anyopaque, idx: usize, _: usize) void {
            const self_ctx: *Context = @ptrCast(@alignCast(c.?));
            self_ctx.arr[idx] = idx * 2 + 1;
        }
    }.run;

    pool.parallelFor(N, &ctx, worker);

    for (0..N) |i| {
        try std.testing.expectEqual(i * 2 + 1, data[i]);
    }
}
