//! Shared test support: the two fixed test windows, the anchor card of each family, and the
//! pieces of the canned-logits runner fakes. `runtime/session.zig`, `serve/residency.zig`,
//! `serve/scheduler.zig`, and `serve/server.zig` drive their tests against a fake runner, and
//! every fake writes the same logits. Test code imports this file. The root module re-exports it
//! only for the tests under `tests/`.
//!
//! Comptime. Asserts pin the test windows to the model context limit.
//!
//! Anchor cards. `testCard` holds one device-free card of each family, a Llama-3.1-8B shape and
//! the Qwen3.5-2B shape. The layer, runner, cost, and guard tests all read these two cards.
//!
//! Fake model. The model is a pure function of the token of each row: argmax of the logits of row
//! i is `(rows[i].token + 1) % vocab`. Greedy parameters then give fully predictable streams, so a
//! reader derives every expected stream by hand. Each fake keeps its own hooks, such as a row log,
//! a per-slot marker, or the swap surface, and calls `writeCannedLogits` for the shared body.

const std = @import("std");
const limits = @import("../core/limits.zig");
const loader = @import("../io/gguf/loader.zig");
const StepRow = @import("step_row.zig").StepRow;

// -- constants --------------------------------------------------------------

/// The KV window of every test that sizes a window from a real model. It replaces the default
/// rule, which reads the free VRAM, so a test runs the same shape on any machine.
pub const test_window: u32 = 8192;

/// The KV window of a test that plans several slots at once. Token streams do not depend on it.
pub const test_serve_window: u32 = 512;

comptime {
    std.debug.assert(test_window <= limits.max_model_ctx);
    std.debug.assert(test_serve_window <= test_window);
}

/// Vocabulary size of every canned-logits fake.
pub const vocab: usize = 8;

// -- api --------------------------------------------------------------------

/// The compact want-logits write of the shipped runners. Only the `want_logits` suffix lands, at
/// `logits[0..]`, where `Sessions.advance` reads it. Row j gets a one-hot at `(token + 1) % vocab`.
pub fn writeCannedLogits(logits: []f32, rows: []const StepRow) void {
    var first: usize = rows.len;
    while (first > 0 and rows[first - 1].want_logits) {
        first -= 1;
    }

    std.debug.assert(logits.len >= (rows.len - first) * vocab);
    for (rows[first..], 0..) |r, j| {
        const row = logits[j * vocab ..][0..vocab];
        @memset(row, 0);
        row[(r.token + 1) % vocab] = 1;
    }
}

/// The anchor card of a family: a Llama-3.1-8B shape, or the Qwen3.5-2B shape, 24 layers at
/// interval 4, so layers 3, 7, 11, 15, 19, and 23 are full attention.
pub fn testCard(comptime arch: loader.Arch) loader.ModelCard {
    return switch (arch) {
        .llama => .{
            .arch = .llama,
            .architecture = "llama",
            .n_layers = 32,
            .n_heads = 32,
            .n_kv_heads = 8,
            .head_dim = 128,
            .embedding_length = 4096,
            .feed_forward_length = 14336,
            .context_length = 131072,
            .vocab_size = 128256,
            .rope_freq_base = 500000.0,
            .rope_dim_count = 128,
            .rms_norm_eps = 1e-5,
            .bos_token_id = 128000,
            .eos_token_id = 128009,
            .chat_template = "",
        },
        .qwen35 => .{
            .arch = .qwen35,
            .architecture = "qwen3.5",
            .n_layers = 24,
            .n_heads = 16,
            .n_kv_heads = 2,
            .head_dim = 256,
            .embedding_length = 2048,
            .feed_forward_length = 8192,
            .context_length = 262144,
            .vocab_size = 248320,
            .rope_freq_base = 5000000.0,
            .rope_dim_count = 64,
            .rms_norm_eps = 1e-6,
            .bos_token_id = 0,
            .eos_token_id = 0,
            .chat_template = "",
            .gdn = .{
                .conv_kernel = 4,
                .state_size = 128,
                .group_count = 16,
                .inner_size = 2048,
                .time_step_rank = 16,
                .full_attention_interval = 4,
            },
        },
    };
}
