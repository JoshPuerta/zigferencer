//! Op micro-benchmarks. `zig build bench` builds them at `-O ReleaseFast` with the CPU kernels,
//! and `zig build bench -Druntime=vulkan` adds the GPU kernels. One table makes a speed
//! regression visible next to a correctness regression, because every row carries an output hash
//! and a time. Each GPU variant also carries a relative error and a pass/fail mark against its
//! CPU reference.
//!
//! Comptime. `build_options.use_vulkan` gates the GPU code, so a CPU build never reads it.
//!
//! Layout. `main` owns the lifetime of the GPU runner and drives the section files. `bench_util`
//! holds the shared timing and printing helpers. The per-family sections live in `bench_matmul`,
//! `bench_layer`, `bench_gdn`, `bench_parallel`, and `bench_transfer`.
//!
//! GPU runner. One runner gives every GPU variant the same Context and the same persistent
//! buffers inside each kernel. A Vulkan init failure prints a banner and leaves every CPU row to
//! run. Each section skips its own GPU rows when no runner is present.
//!
//! Order. The crossing latency section runs before the staged-copy sweep, because the sweep
//! returns early when its 512 MiB pair does not fit.
//!
//! Attention grid alone. `zig build bench -- attn-grid` runs the attention kernel set grid alone,
//! so a kernel experiment skips the full table.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const zigferencer = @import("zigferencer");

const bench_util = @import("bench_util.zig");
const matmul = @import("bench_matmul.zig");
const layer = @import("bench_layer.zig");
const gdn = @import("bench_gdn.zig");
const par = @import("bench_parallel.zig");
const transfer = @import("bench_transfer.zig");

// -- api --------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    // Start the GPU runner.
    const RunnerOpt = if (build_options.use_vulkan)
        ?*zigferencer.vulkan_runner.Runner
    else
        void;

    const runner: RunnerOpt = if (comptime build_options.use_vulkan)
        zigferencer.vulkan_runner.Runner.init(gpa) catch |e| brk: {
            std.debug.print("vulkan: skipped ({t})\n\n", .{e});
            break :brk null;
        }
    else {};

    defer if (comptime build_options.use_vulkan) {
        if (runner) |r| {
            r.deinit();
        }
    };

    // Turn color on for a terminal alone, so a piped diff stays free of ANSI escapes.
    bench_util.setColor(std.Io.File.stderr().isTty(io) catch false);

    // Print the banner.
    std.debug.print("zigferencer bench\n", .{});
    std.debug.print(
        "build target: {s} ({s})\n",
        .{ @tagName(builtin.cpu.arch), builtin.cpu.model.name },
    );

    // Run the attention kernel set grid alone when asked.
    if (comptime build_options.use_vulkan) {
        if (runner) |r| {
            const args = try init.minimal.args.toSlice(init.arena.allocator());
            if (args.len >= 2 and std.mem.eql(u8, args[1], "attn-grid")) {
                try layer.benchAttnGrid(gpa, io, r);
                return;
            }
        }
    }

    // Run the CPU sections, which every build includes.
    try matmul.runQ80(gpa, io, runner);
    try matmul.benchQ40(gpa, io, runner);
    try layer.benchElementwise(gpa, io, runner);
    try gdn.benchGdn(gpa, io, runner);
    try gdn.benchGdnChunked(gpa, io, runner);
    try par.benchFanoutCost(gpa, io);
    try par.benchDispatchCost(io);
    try par.benchBandwidth(gpa, io);

    // Run the GPU-only sections.
    if (comptime build_options.use_vulkan) {
        if (runner) |r| {
            try layer.benchSigmoidGate(gpa, io, r);
            try matmul.benchBatchedMatmul(gpa, io, r);
            try transfer.benchCrossingLatency(io, r);
            try transfer.benchStagedCopy(io, r);
        }
    }
}
