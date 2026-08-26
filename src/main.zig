const std = @import("std");
const ziglm = @import("ziglm");

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    var arg_it = std.process.Args.Iterator.init(init.minimal.args);
    const args = ziglm.cli.parseArgsFromIterator(&arg_it);
    try ziglm.cli.runCli(allocator, args);
}
