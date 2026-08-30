const std = @import("std");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    // Standard target options allow the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});
    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});
    // It's also possible to define more custom flags to toggle optional features
    // of this build script using `b.option()`. All defined flags (including
    // target and optimize options) will be listed when running `zig build --help`
    // in this directory.

    // This creates a module, which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // Zig modules are the preferred way of making Zig code available to consumers.
    // addModule defines a module that we intend to make available for importing
    // to our consumers. We must give it a name because a Zig package can expose
    // multiple modules and consumers will need to be able to specify which
    // Compile CUDA C Bridge using nvcc
    const nvcc_cmd = b.addSystemCommand(&.{
        "/opt/cuda/bin/nvcc",
        "-c",
    });
    nvcc_cmd.addFileArg(b.path("src/cuda_bridge.cu"));
    nvcc_cmd.addArg("-o");
    const cuda_obj = nvcc_cmd.addOutputFileArg("cuda_bridge.o");
    nvcc_cmd.addArgs(&.{
        "-O3",
        "-Xcompiler",
        "-fPIC",
        "-arch=compute_75",
        "-Isrc",
        "-I/opt/cuda/include",
    });

    const mod = b.addModule("ziglm", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    mod.addObjectFile(cuda_obj);
    mod.addLibraryPath(.{ .cwd_relative = "/opt/cuda/lib64" });
    mod.linkSystemLibrary("cudart", .{});
    mod.linkSystemLibrary("stdc++", .{});
    mod.linkSystemLibrary("c", .{});

    const exe = b.addExecutable(.{
        .name = "ziglm",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ziglm", .module = mod },
            },
        }),
    });
    exe.root_module.addLibraryPath(.{ .cwd_relative = "/opt/cuda/lib64" });
    exe.root_module.linkSystemLibrary("cudart", .{});
    exe.root_module.linkSystemLibrary("stdc++", .{});
    exe.root_module.linkSystemLibrary("c", .{});

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });
    mod_tests.root_module.linkSystemLibrary("c", .{});

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    exe_tests.root_module.linkSystemLibrary("c", .{});

    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);

    // Just like flags, top level steps are also listed in the `--help` menu.
    //
    // The Zig build system is entirely implemented in userland, which means
    // that it cannot hook into private compiler APIs. All compilation work
    // orchestrated by the build system will result in other Zig compiler
    // subcommands being invoked with the right flags defined. You can observe
    // these invocations when one fails (or you pass a flag to increase
    // verbosity) to validate assumptions and diagnose problems.
    //
    // Lastly, the Zig build system is relatively simple and self-contained,
    // and reading its source code will allow you to master it.
}
