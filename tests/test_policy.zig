//! The thread policy of the runner tests, and the start of the pool that
//! serves it. Every test that builds a runner starts its own pool here, so no
//! pool outlives the test that owns it.
//!
//! Comptime. An assert pins the test thread count at one or more.

const std = @import("std");
const zigferencer = @import("zigferencer");
const test_options = @import("test_options");

pub const Policy = zigferencer.parallel.Policy;
pub const Pool = zigferencer.pool.Pool;

/// The pool size of `-Dtest-threads`, 4 by default.
const test_threads: u32 = test_options.test_threads;

comptime {
    std.debug.assert(test_threads >= 1);
}

/// The policy of a runner test whose thread count is incidental. The floor
/// of zero fans out every op on more than one thread, so the suite runs the
/// pool path at short rows. A serial reference path uses `.{ .n_threads = 1 }`.
pub fn policy() Policy {
    return .{
        .n_threads = test_threads,
        .min_parallel_flops = 0,
        .spin_ns = zigferencer.parallel.default_spin_ns,
    };
}

/// Start `pool` for `pool_policy`. The caller owns the pool and must call
/// `pool.deinit()` before the test returns.
pub fn startPool(pool: *Pool, io: std.Io, pool_policy: Policy) !void {
    try zigferencer.parallel.initPool(pool, io, pool_policy);
    std.debug.assert(pool.threads() == @max(pool_policy.n_threads, 1));
}
