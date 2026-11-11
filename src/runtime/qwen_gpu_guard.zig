//! The load gate of the Qwen3.5 GPU kernels. A GGUF file is untrusted input, and a ReleaseFast
//! build compiles out the asserts of `GdnDecode.recordRun`, `GdnChunkParallel.recordConv`,
//! `GdnChunkParallel.recordScan`, and `QwenQkNorm.recordRow`. A shape past those asserts overruns
//! a shader array or writes garbage, so the fused runner refuses it here, before the first
//! allocation. `qwen_layer.fusedGuard` is the one caller.
//!
//! Comptime. `refuse` takes its format at comptime, as the log functions do.
//!
//! This file touches no device, so a CPU test proves each refusal with an in-memory card.

const std = @import("std");
const builtin = @import("builtin");
const limits = @import("../core/limits.zig");
const loader = @import("../io/gguf/loader.zig");
const dispatch_guard = @import("dispatch_guard.zig");

const log = std.log.scoped(.runner);

// -- constants --------------------------------------------------------------

const heads_fmt = "The model has {d} GDN key heads and {d} GDN value heads. " ++
    "GPU kernels take at most {d} value heads per key head, as a whole multiple. " ++
    "Run this model on a CPU build";

const head_dim_fmt = "The model head_dim is {d}. GPU QK norm takes a head_dim of {d} only. " ++
    "Run this model on a CPU build";

const state_size_fmt = "The model GDN state size is {d}. " ++
    "GPU prefill takes a state size of {d} only. Run this model on a CPU build";

/// The test runner fails a test that logs an error, and a refusal test expects these lines.
const refusal_level: std.log.Level = if (builtin.is_test) .debug else .err;

// -- types ------------------------------------------------------------------

pub const Error = error{
    GdnHeadsUnsupported,
    QwenHeadDimUnsupported,
    GdnHeadDimUnsupported,
} || dispatch_guard.Error;

// -- api --------------------------------------------------------------------

/// Refuse a shape that the Qwen GPU kernels cannot take, with one log line of both values.
/// The card must hold GDN parameters.
pub fn requireShape(card: *const loader.ModelCard) Error!void {
    try dispatch_guard.requireHeads(card);
    const g = card.gdn.?;
    if (!gdnHeadsFit(g.group_count, g.time_step_rank)) {
        refuse(heads_fmt, .{ g.group_count, g.time_step_rank, limits.gdn_max_v_per_k });
        return error.GdnHeadsUnsupported;
    }

    if (card.head_dim != limits.qwen_qk_head_dim) {
        refuse(head_dim_fmt, .{ card.head_dim, limits.qwen_qk_head_dim });
        return error.QwenHeadDimUnsupported;
    }

    if (g.state_size != limits.gdn_head_dim) {
        refuse(state_size_fmt, .{ g.state_size, limits.gdn_head_dim });
        return error.GdnHeadDimUnsupported;
    }
}

// -- helpers ----------------------------------------------------------------

fn gdnHeadsFit(n_k_heads: u32, n_v_heads: u32) bool {
    if (n_k_heads == 0) return false;
    if (n_v_heads % n_k_heads != 0) return false;
    return n_v_heads / n_k_heads <= limits.gdn_max_v_per_k;
}

fn refuse(comptime fmt: []const u8, args: anytype) void {
    switch (refusal_level) {
        .err => log.err(fmt, args),
        else => log.debug(fmt, args),
    }
}

// -- tests ------------------------------------------------------------------

/// The 2B anchor card at another head shape.
fn shapeCard(head_dim: u32, n_k_heads: u32, n_v_heads: u32) loader.ModelCard {
    var card = @import("test_fakes.zig").testCard(.qwen35);
    card.head_dim = head_dim;
    card.gdn.?.group_count = n_k_heads;
    card.gdn.?.time_step_rank = n_v_heads;
    return card;
}

test "requireShape: the file values of the 2B and the 9B pass" {
    try requireShape(&shapeCard(256, 16, 16));
    try requireShape(&shapeCard(256, 16, 32));
    try requireShape(&shapeCard(256, 2, 2 * limits.gdn_max_v_per_k));
}

test "requireShape: a shape past a shader array is refused" {
    const heads = error.GdnHeadsUnsupported;
    try std.testing.expectError(heads, requireShape(&shapeCard(256, 2, 18)));
    try std.testing.expectError(heads, requireShape(&shapeCard(256, 16, 24)));
    try std.testing.expectError(heads, requireShape(&shapeCard(256, 0, 16)));

    const head_dim = error.QwenHeadDimUnsupported;
    try std.testing.expectError(head_dim, requireShape(&shapeCard(128, 16, 16)));
    try std.testing.expectError(head_dim, requireShape(&shapeCard(512, 16, 32)));

    var wide_state = shapeCard(256, 16, 16);
    wide_state.gdn.?.state_size = 2 * limits.gdn_head_dim;
    try std.testing.expectError(error.GdnHeadDimUnsupported, requireShape(&wide_state));
}
