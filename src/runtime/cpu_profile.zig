//! The per-op CPU decode profile, shared by the CPU runners of both families.
//!
//! Rows. The ops of `layer_op.LayerOp` are the rows of both families, and a report groups them by
//! `OpClass`. A family times a fixed set of ops, and that set partitions one decode step: the timed
//! brackets do not overlap, and together they cover the step. `CpuProfile` records its family, so
//! the report prints only the rows of the family. Every Llama op is also a Qwen op.
//!
//! Source. The runners open one `span.begin` scope per op. A caller installs `CpuProfile.sums`
//! with `span.installSums` on the decode thread, and each closed op scope adds its time there.
//! `bench-model --profile-cpu-decode` is the one caller. The profile is plain data in every build,
//! and a build without `-Dtrace` leaves it at zero.

const std = @import("std");
const span = @import("../core/span.zig");
const layer_op = @import("../core/layer_op.zig");

const LayerOp = layer_op.LayerOp;
const OpKey = layer_op.OpKey;

// -- types ------------------------------------------------------------------

/// The runner family a profile belongs to. Each family times its own subset of the ops.
pub const Family = enum { llama, qwen };

/// Per-op CPU-time accumulator. The caller installs `sums` and sets `steps`.
pub const CpuProfile = struct {
    family: Family,
    sums: span.Sums = .{},
    steps: u64 = 0,

    /// Nanoseconds charged to one op.
    pub fn nsOf(self: *const CpuProfile, op: LayerOp) u64 {
        return self.sums.get(.ofOp(op));
    }

    /// Sum across the ops of the family. The per-step CPU time is `total() / steps`.
    pub fn total(self: *const CpuProfile) u64 {
        var sum: u64 = 0;
        for (std.enums.values(LayerOp)) |op| {
            if (usedBy(op, self.family)) {
                sum += self.nsOf(op);
            }
        }

        return sum;
    }

    /// Time spent in the ops that fan out on the pool.
    pub fn nsParallel(self: *const CpuProfile) u64 {
        var sum: u64 = 0;
        for (std.enums.values(LayerOp)) |op| {
            if (usedBy(op, self.family) and isParallel(op)) {
                sum += self.nsOf(op);
            }
        }

        return sum;
    }
};

// -- api --------------------------------------------------------------------

/// True for the ops that fan out on the pool: the matmuls, the attention, and the GDN step. The
/// report labels the other rows serial.
pub fn isParallel(op: LayerOp) bool {
    return switch (op.class()) {
        .matmul, .attention, .recurrent => true,
        .elementwise, .norm, .transfer => false,
    };
}

/// True when `family` times this op. The two transfers of a GPU step have no CPU form.
pub fn usedBy(op: LayerOp, family: Family) bool {
    return switch (op) {
        .embed,
        .attn_norm,
        .attn_proj,
        .rope,
        .kv_commit,
        .attention,
        .o_proj,
        .mixer_residual,
        .ffn_norm,
        .gate_up_proj,
        .swiglu,
        .down_proj,
        .ffn_residual,
        .final_norm,
        .lm_head,
        => true,
        .gdn_proj,
        .gdn,
        .gdn_out,
        .qk_norm,
        .attn_gate,
        => family == .qwen,
        .stage_in, .readback => false,
    };
}

// -- tests ------------------------------------------------------------------

test "cpu profile: each family times a fixed op set" {
    var llama: u32 = 0;
    var qwen: u32 = 0;
    for (std.enums.values(LayerOp)) |op| {
        // Only the GPU transfers are dead on the CPU.
        const gpu_only = op == .stage_in or op == .readback;
        try std.testing.expectEqual(!gpu_only, usedBy(op, .llama) or usedBy(op, .qwen));
        if (usedBy(op, .llama)) {
            // One op has one name, so a Llama row is also a Qwen row.
            try std.testing.expect(usedBy(op, .qwen));
            llama += 1;
        }

        if (usedBy(op, .qwen)) {
            qwen += 1;
        }
    }

    try std.testing.expectEqual(@as(u32, 15), llama);
    try std.testing.expectEqual(@as(u32, 20), qwen);
}

test "cpu profile: the parallel rows are the matmul, attention, and recurrent classes" {
    for (std.enums.values(LayerOp)) |op| {
        const pooled = op.class() == .matmul or op == .attention or op == .gdn;
        try std.testing.expectEqual(pooled, isParallel(op));
    }

    try std.testing.expect(isParallel(.lm_head));
    try std.testing.expect(!isParallel(.attn_norm));
    try std.testing.expect(!isParallel(.kv_commit));
}
