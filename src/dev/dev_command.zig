//! The developer commands. `zigferencer-dev` dispatches the release table plus the table here.
//! The release module cannot read this file, so `command.dev_names` repeats the names.
//!
//! Comptime. A block fails the build when `command.dev_names` differs from the table here.

const std = @import("std");
const zigferencer = @import("zigferencer");

const bench_decode_host_cmd = @import("bench_decode_host_cmd.zig");
const bench_model_cmd = @import("bench_model/bench_model_cmd.zig");
const bench_serve_cmd = @import("bench_serve_cmd.zig");
const bench_swap_cmd = @import("bench_swap_cmd.zig");
const compare_logits_cmd = @import("compare_logits_cmd.zig");
const dispatch_list_cmd = @import("dispatch_list_cmd.zig");
const gpu_selftest_cmd = @import("gpu_selftest/gpu_selftest_cmd.zig");
const serve_demo_cmd = @import("serve_demo_cmd.zig");
const tokens_cmd = @import("tokens_cmd.zig");

const command = zigferencer.cli.command;

// -- constants --------------------------------------------------------------

/// The commands that only `zigferencer-dev` holds.
const dev = [_]command.Subcommand{
    row("bench-model", "Benchmark a model: TTFT, ITL, hash", bench_model_cmd.run),
    row("bench-serve", "Benchmark serve throughput vs batch size", bench_serve_cmd.run),
    row("bench-swap", "Benchmark tiered-KV swap cost (go/no-go)", bench_swap_cmd.run),
    row("bench-decode-host", "Time the host steps of one decode step", bench_decode_host_cmd.run),
    row("gpu-selftest", "Bit-check a GPU kernel vs its CPU reference", gpu_selftest_cmd.run),
    row("tokens", "Print a prompt's token ids", tokens_cmd.run),
    row("serve-demo", "Demo the serve connector end-to-end", serve_demo_cmd.run),
    row("dispatch-list", "Write the Vulkan commands of six sections", dispatch_list_cmd.run),
    row("compare-logits", "Compare two logits files of bench-model", compare_logits_cmd.run),
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

// -- helpers ----------------------------------------------------------------

/// One table row. A function keeps the rows inside 100 columns.
fn row(name: []const u8, summary: []const u8, handler: command.Handler) command.Subcommand {
    return .{ .name = name, .summary = summary, .handler = handler };
}
