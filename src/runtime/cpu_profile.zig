//! The per-op CPU decode profile, shared by the llama and the qwen CPU runners.
//!
//! Categories. The `op_*` names of `span.Name` are the categories of both families. A family uses a
//! fixed subset, and that subset partitions one decode step: the timed brackets do not overlap, and
//! together they cover the step. `CpuProfile` records its family, so the report prints only the
//! rows of the family.
//!
//! Source. The runners open a `span.begin` scope per bracket. A caller installs `CpuProfile.sums`
//! with `span.installSums` on the decode thread, and each closed op scope adds its time there.
//! `bench-model --profile-cpu-decode` is the one caller. The profile is plain data in every build,
//! and a build without `-Dtrace` leaves it at zero.

const std = @import("std");
const span = @import("../core/span.zig");

const Name = span.Name;

// -- types ------------------------------------------------------------------

/// The runner family a profile belongs to. Each family charges its own subset of the op names.
pub const Family = enum { llama, qwen };

/// Per-op CPU-time accumulator. The caller installs `sums` and sets `steps`.
pub const CpuProfile = struct {
    family: Family,
    sums: span.Sums = .initFill(0),
    steps: u64 = 0,

    /// Nanoseconds charged to one op name.
    pub fn nsOf(self: *const CpuProfile, name: Name) u64 {
        std.debug.assert(name.isOp());

        return self.sums.get(name);
    }

    /// Sum across the op names of the family. The per-step CPU time is `total() / steps`.
    pub fn total(self: *const CpuProfile) u64 {
        var sum: u64 = 0;
        for (std.enums.values(Name)) |name| {
            if (name.isOp() and usedBy(name, self.family)) {
                sum += self.sums.get(name);
            }
        }

        return sum;
    }

    /// Time spent in the op names that fan out on the pool.
    pub fn nsParallel(self: *const CpuProfile) u64 {
        var sum: u64 = 0;
        for (std.enums.values(Name)) |name| {
            if (name.isOp() and usedBy(name, self.family) and isParallel(name)) {
                sum += self.sums.get(name);
            }
        }

        return sum;
    }
};

// -- api --------------------------------------------------------------------

/// True for the row-parallel matmuls. Each runner charges them at its one matmul boundary.
pub fn isMatmul(name: Name) bool {
    std.debug.assert(name.isOp());

    return switch (name) {
        .op_gdn_proj,
        .op_gdn_out,
        .op_qkv_proj,
        .op_attn_proj,
        .op_o_proj,
        .op_gate_up_proj,
        .op_down_proj,
        .op_lm_head,
        => true,
        else => false,
    };
}

/// True for the op names that fan out on the pool. The report labels the other rows serial.
pub fn isParallel(name: Name) bool {
    return isMatmul(name) or name == .op_attention or name == .op_gdn_step;
}

/// True when `family` charges this op name.
pub fn usedBy(name: Name, family: Family) bool {
    std.debug.assert(name.isOp());

    return switch (name) {
        .op_embed,
        .op_rmsnorm,
        .op_attention,
        .op_o_proj,
        .op_gate_up_proj,
        .op_swiglu,
        .op_down_proj,
        .op_residual,
        .op_lm_head,
        => true,
        .op_qkv_proj, .op_rope => family == .llama,
        .op_gdn_proj,
        .op_gdn_step,
        .op_gdn_out,
        .op_attn_proj,
        .op_qk_norm_rope,
        .op_kv_commit,
        .op_attn_gate,
        => family == .qwen,
        else => unreachable,
    };
}

// -- tests ------------------------------------------------------------------

test "cpu profile: each family uses a fixed op set" {
    var llama: u32 = 0;
    var qwen: u32 = 0;
    for (std.enums.values(Name)) |name| {
        if (!name.isOp()) continue;

        // No op is dead: at least one family charges it.
        try std.testing.expect(usedBy(name, .llama) or usedBy(name, .qwen));
        if (usedBy(name, .llama)) {
            llama += 1;
        }

        if (usedBy(name, .qwen)) {
            qwen += 1;
        }
    }

    try std.testing.expectEqual(@as(u32, 11), llama);
    try std.testing.expectEqual(@as(u32, 16), qwen);
    // The lm_head and the FFN matmuls are common to both families.
    for ([_]Name{ .op_gate_up_proj, .op_down_proj, .op_lm_head }) |name| {
        try std.testing.expect(isMatmul(name));
        try std.testing.expect(usedBy(name, .llama));
        try std.testing.expect(usedBy(name, .qwen));
    }

    try std.testing.expect(!isMatmul(.op_attention));
    try std.testing.expect(!isMatmul(.op_gdn_step));
    // Every matmul fans out. So do attention and the GDN step.
    for (std.enums.values(Name)) |name| {
        if (name.isOp() and isMatmul(name)) {
            try std.testing.expect(isParallel(name));
        }
    }

    try std.testing.expect(isParallel(.op_attention));
    try std.testing.expect(isParallel(.op_gdn_step));
    try std.testing.expect(!isParallel(.op_rmsnorm));
}

test "cpu profile: the op names are exactly the names that isOp accepts" {
    var ops: u32 = 0;
    for (std.enums.values(Name)) |name| {
        const prefixed = std.mem.startsWith(u8, @tagName(name), "op_");
        try std.testing.expectEqual(prefixed, name.isOp());
        if (name.isOp()) {
            ops += 1;
        }
    }

    try std.testing.expectEqual(@as(u32, 18), ops);
}
