//! The developer commands. `zigferencer-dev` dispatches the release table plus the table here.
//! The release module cannot read this file, so `command.dev_names` repeats the names.

const std = @import("std");
const zigferencer = @import("zigferencer");

const bench_decode_host_cmd = @import("bench_decode_host_cmd.zig");
const bench_model_cmd = @import("bench_model/bench_model_cmd.zig");
const bench_serve_cmd = @import("bench_serve_cmd.zig");
const bench_swap_cmd = @import("bench_swap_cmd.zig");
const gpu_selftest_cmd = @import("gpu_selftest/gpu_selftest_cmd.zig");
const serve_demo_cmd = @import("serve_demo_cmd.zig");
const tokens_cmd = @import("tokens_cmd.zig");

const command = zigferencer.cli.command;

// -- constants --------------------------------------------------------------

/// The commands that only `zigferencer-dev` holds.
const dev = [_]command.Subcommand{
    .{ .name = "bench-model", .summary = "Benchmark a model: TTFT, ITL, hash", .handler = bench_model_cmd.run },
    .{ .name = "bench-serve", .summary = "Benchmark serve throughput vs batch size", .handler = bench_serve_cmd.run },
    .{ .name = "bench-swap", .summary = "Benchmark tiered-KV swap cost (go/no-go)", .handler = bench_swap_cmd.run },
    .{ .name = "bench-decode-host", .summary = "Time the host steps of one decode step", .handler = bench_decode_host_cmd.run },
    .{ .name = "gpu-selftest", .summary = "Bit-check a GPU kernel vs its CPU reference", .handler = gpu_selftest_cmd.run },
    .{ .name = "tokens", .summary = "Print a prompt's token ids", .handler = tokens_cmd.run },
    .{ .name = "serve-demo", .summary = "Demo the serve connector end-to-end", .handler = serve_demo_cmd.run },
};

comptime {
    std.debug.assert(dev.len == command.dev_names.len);

    for (dev, command.dev_names) |d, name| {
        if (!std.mem.eql(u8, d.name, name)) {
            @compileError("dev_names diverges from dev at '" ++ d.name ++ "'");
        }
    }
}

// -- api --------------------------------------------------------------------

/// Dispatch argv over the release and the dev commands.
pub fn dispatch(init: std.process.Init, stdout: *std.Io.File.Writer) !void {
    return command.dispatchOver(init, stdout, &(command.release ++ dev), "zigferencer-dev");
}
