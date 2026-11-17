//! Entry point of `zigferencer-dev`: the release commands plus the developer
//! commands. This file roots a module of its own, so every file under `src/dev/`
//! reads the engine through the `zigferencer` module. A relative import out of
//! this directory puts a file into two modules, and Zig rejects that. The
//! `zigferencer` module has no path back here, so the release binary cannot hold
//! a developer command.

const std = @import("std");
const zigferencer = @import("zigferencer");
const dev_command = @import("dev_command.zig");

pub const std_options = zigferencer.std_options;

pub fn main(init: std.process.Init) !void {
    return zigferencer.entry(init, dev_command.dispatch);
}

// The block pulls the developer commands into the `dev_unit` test binary. A test of the
// `zigferencer` module never runs here, so `unit` and `dev_unit` share no test.
test {
    // A command body is analyzed only where it is named, so this line makes every prong of every
    // developer command compile under `test-build`.
    _ = &dev_command.dispatch;
    _ = @import("bench_decode_host_cmd.zig");
    _ = @import("bench_model/bench_model_common.zig");
    _ = @import("bench_model/bench_model_logits.zig");
    _ = @import("bench_model/bench_model_report.zig");
    _ = @import("bench_serve_cmd.zig");
    _ = @import("bench_swap_cmd.zig");
    _ = @import("compare_logits_cmd.zig");
}
