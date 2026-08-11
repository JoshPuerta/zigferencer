//! Shared test support: the two fixed test windows, and the pieces of the
//! canned-logits runner fakes. `runtime/session.zig`, `serve/residency.zig`,
//! `serve/scheduler.zig`, and `serve/server.zig` drive their tests against a
//! fake runner, and every one of those fakes writes the same logits. Test code
//! imports this file. The root module re-exports it only for the tests under
//! `tests/`.
//!
//! The fake model is a pure function of the token of each row: argmax of the
//! logits of row i is `(rows[i].token + 1) % vocab`. Greedy parameters then give
//! fully predictable streams, so a reader derives every expected stream by hand.
//! Each fake keeps its own hooks, such as a row log, a per-slot marker, or the
//! swap surface, and calls `writeCannedLogits` for the shared body.

const std = @import("std");
const limits = @import("../core/limits.zig");
const StepRow = @import("step_row.zig").StepRow;

// -- api --------------------------------------------------------------------

/// The KV window of every test that sizes a window from a real model. A test
/// never takes the default rule, which reads the model cost and the free VRAM.
/// A test therefore runs the same shape on any machine.
pub const test_window: u32 = 8192;

/// The KV window of a test that plans several slots at once. It keeps the KV of
/// every slot small, and the token streams do not depend on the window.
pub const test_serve_window: u32 = 512;

comptime {
    std.debug.assert(test_window <= limits.max_model_ctx);
    std.debug.assert(test_serve_window <= test_window);
}

/// Vocabulary size of every canned-logits fake.
pub const vocab: usize = 8;

/// The compact want-logits write the shipped runners perform: only the suffix
/// of `want_logits` rows lands, at `logits[0..]`, where `Sessions.advance`
/// reads it. Row j of that suffix gets a one-hot row at `(token + 1) % vocab`.
pub fn writeCannedLogits(logits: []f32, rows: []const StepRow) void {
    var first: usize = rows.len;
    while (first > 0 and rows[first - 1].want_logits) first -= 1;
    std.debug.assert(logits.len >= (rows.len - first) * vocab);
    for (rows[first..], 0..) |r, j| {
        const row = logits[j * vocab ..][0..vocab];
        @memset(row, 0);
        row[(r.token + 1) % vocab] = 1;
    }
}
