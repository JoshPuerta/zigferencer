//! Op micro-benchmarks. `zig build bench` builds them at `-O ReleaseFast` with
//! the CPU kernels, and `zig build bench -Druntime=vulkan` adds the GPU kernels.
//! One table makes a speed regression visible next to a correctness regression,
//! because every row carries an output hash and a time. Each GPU variant also
//! carries a relative error and a pass/fail mark against its CPU reference.
//!
//! `main` is the control unit. It owns the lifetime of the GPU runner and drives
//! the section files. `bench_util` holds the shared timing and printing helpers.
//! The per-family sections live in `bench_matmul`, `bench_layer`, `bench_gdn`,
//! `bench_parallel`, and `bench_transfer`.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const z = @import("zigferencer");

const bu = @import("bench_util.zig");
const matmul = @import("bench_matmul.zig");
const layer = @import("bench_layer.zig");
const gdn = @import("bench_gdn.zig");
const par = @import("bench_parallel.zig");
const transfer = @import("bench_transfer.zig");

// -- api --------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    // One runner gives every GPU variant the same Context, and the same
    // persistent buffers inside each kernel. A Vulkan init failure prints a
    // banner and leaves every CPU row to run.
    const RunnerOpt = if (build_options.use_vulkan)
        ?*z.vulkan_runner.Runner
    else
        void;
    const runner: RunnerOpt = if (comptime build_options.use_vulkan)
        z.vulkan_runner.Runner.init(gpa) catch |e| brk: {
            std.debug.print("vulkan: skipped ({t})\n\n", .{e});
            break :brk null;
        }
    else {};
    defer if (comptime build_options.use_vulkan) {
        if (runner) |r| r.deinit();
    };

    // Color goes on for a terminal alone, so piped output stays free of ANSI
    // escapes for a diff or a snapshot.
    bu.setColor(std.Io.File.stderr().isTty(io) catch false);

    std.debug.print("zigferencer bench\n", .{});
    std.debug.print("build target: {s} ({s})\n", .{ @tagName(builtin.cpu.arch), builtin.cpu.model.name });

    // `zig build bench -- attn-kit` runs the attention kit grid alone, so a
    // kernel experiment skips the full table.
    if (comptime build_options.use_vulkan) {
        if (runner) |r| {
            const args = try init.minimal.args.toSlice(init.arena.allocator());
            if (args.len >= 2 and std.mem.eql(u8, args[1], "attn-kit")) {
                try layer.benchAttnKitGrid(gpa, io, r);
                return;
            }
        }
    }

    // The CPU reference rows run on every build. Each section skips its own GPU
    // rows when no runner is present.
    try matmul.runQ80(gpa, io, runner);
    try matmul.benchQ40(gpa, io, runner);
    try layer.benchElementwise(gpa, io, runner);
    try gdn.benchGdn(gpa, io, runner);
    try gdn.benchGdnChunked(gpa, io, runner);
    try par.benchFanoutCost(gpa, io);
    try par.benchDispatchCost(io);
    try par.benchBandwidth(gpa, io);
    if (comptime build_options.use_vulkan) {
        if (runner) |r| {
            try layer.benchSigmoidGate(gpa, io, r);
            try matmul.benchBatchedMatmul(gpa, io, r);
            // Crossing latency runs first. The staged-copy sweep returns early
            // when its 512 MiB pair does not fit.
            try transfer.benchCrossingLatency(io, r);
            try transfer.benchStagedCopy(io, r);
        }
    }
}
