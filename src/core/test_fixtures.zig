//! Shared fixture lookup for the model tests. Test code imports this file. The
//! root module re-exports it only for the tests under `tests/`.
//!
//! A test names a registry model, and `path` finds its file through
//! `registry.locate`: the repo-local `models/` first, then the store that
//! `$ZIGFERENCER_MODELS_DIR` names. A missing file skips the test only when the
//! variable is unset. A machine that says where its models are must run every
//! model test, so there a missing file fails the test.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const registry = @import("registry.zig");

// -- types ------------------------------------------------------------------

pub const Error = error{ SkipZigTest, FixtureMissing };

// -- api --------------------------------------------------------------------

/// The on-disk path of registry model `pub_name` for a test.
pub fn path(arena: Allocator, pub_name: []const u8) (Error || Allocator.Error)![]const u8 {
    const entry = registry.lookup(pub_name);
    assert(entry != null);
    const model = entry.?.*;
    const env = std.testing.environ.getPosix(registry.models_dir_env);
    const env_set = if (env) |v| v.len > 0 else false;
    const root = registry.resolveRoot(env);
    const found = try registry.locate(arena, std.testing.io, root, model);
    return decide(found, env_set) catch |err| {
        if (err == error.FixtureMissing) {
            const local = try registry.modelFilePath(arena, registry.default_models_root, model);
            const store = try registry.modelFilePath(arena, root, model);
            std.debug.print("test_fixtures: {s} is missing: not at {s}, not at {s}\n", .{ pub_name, local, store });
        }
        return err;
    };
}

/// The pure decision, separate so a test can enumerate it.
pub fn decide(found: ?[]const u8, env_set: bool) Error![]const u8 {
    if (found) |p| {
        assert(p.len > 0);
        return p;
    }
    return if (env_set) error.FixtureMissing else error.SkipZigTest;
}

// -- tests ------------------------------------------------------------------

test "decide: a found file runs, and a missing file skips only without the store" {
    const p = "models/a/b.gguf";
    for ([_]bool{ false, true }) |env_set| {
        try std.testing.expectEqualStrings(p, try decide(p, env_set));
    }
    try std.testing.expectError(error.SkipZigTest, decide(null, false));
    try std.testing.expectError(error.FixtureMissing, decide(null, true));
}
