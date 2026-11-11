//! Role. The dispatch gate of the fused kernels. A fused kernel dispatches one workgroup per head
//! on one grid axis, so a head count of the card is a workgroup count. A count past the device
//! limit fails the dispatch, so the fused runner refuses it before the first allocation.
//!
//! Limit. The runner holds no queried `maxComputeWorkGroupCount`, so the gate reads
//! `limits.max_workgroups`, the count that Vulkan guarantees on every axis. The attention kernels
//! assert against the same constant.
//!
//! This file touches no device, so a CPU test proves each refusal with an in-memory card.

const std = @import("std");
const builtin = @import("builtin");
const limits = @import("../core/limits.zig");
const loader = @import("../io/gguf/loader.zig");

const log = std.log.scoped(.runner);

// -- constants --------------------------------------------------------------

/// The test runner fails a test that logs an error, and a refusal test expects these lines.
const refusal_level: std.log.Level = if (builtin.is_test) .debug else .err;

// -- types ------------------------------------------------------------------

pub const Error = error{HeadsPastDispatchLimit};

// -- api --------------------------------------------------------------------

/// Refuse a card whose attention or GDN head count passes the workgroup count of one grid axis.
pub fn requireHeads(card: *const loader.ModelCard) Error!void {
    const g = card.gdn;
    const heads = [_]struct { name: []const u8, count: u32 }{
        .{ .name = "attention heads", .count = card.n_heads },
        .{ .name = "KV heads", .count = card.n_kv_heads },
        .{ .name = "GDN key heads", .count = if (g) |p| p.group_count else 1 },
        .{ .name = "GDN value heads", .count = if (g) |p| p.time_step_rank else 1 },
    };
    for (heads) |h| {
        if (h.count <= limits.max_workgroups) continue;

        const format = "The model has {d} {s}. A fused kernel dispatches one workgroup per " ++
            "head, and the device takes {d} on one axis. Run this model on a CPU build";
        const args = .{ h.count, h.name, limits.max_workgroups };
        switch (refusal_level) {
            .err => log.err(format, args),
            else => log.debug(format, args),
        }

        return error.HeadsPastDispatchLimit;
    }
}

// -- tests ------------------------------------------------------------------

const test_fakes = @import("test_fakes.zig");

test "requireHeads: the anchor cards pass, and a head count past one axis refuses" {
    try requireHeads(&test_fakes.testCard(.llama));
    try requireHeads(&test_fakes.testCard(.qwen35));

    const past = limits.max_workgroups + 1;
    var heads = test_fakes.testCard(.llama);
    heads.n_heads = past;
    heads.n_kv_heads = 1;
    try std.testing.expectError(error.HeadsPastDispatchLimit, requireHeads(&heads));

    var kv = test_fakes.testCard(.llama);
    kv.n_kv_heads = past;
    try std.testing.expectError(error.HeadsPastDispatchLimit, requireHeads(&kv));

    var gdn = test_fakes.testCard(.qwen35);
    gdn.gdn.?.time_step_rank = past;
    try std.testing.expectError(error.HeadsPastDispatchLimit, requireHeads(&gdn));

    var at_limit = test_fakes.testCard(.qwen35);
    at_limit.gdn.?.group_count = limits.max_workgroups;
    try requireHeads(&at_limit);
}
