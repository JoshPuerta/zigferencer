//! Per-op CPU decode timer, shared by the llama and the qwen CPU runners.
//!
//! One `ProfileCat` enum names every category of both families. A family uses
//! a fixed subset, and that subset partitions one decode step: the timed
//! brackets do not overlap, and together they cover the step. `CpuProfile`
//! records its family, so a bracket that charges a category of the other
//! family fails an assert, and the report prints only the rows of the family.
//!
//! `-Dtrace` decides whether the instrument exists. Without it, `CpuProfile`,
//! `Slot`, and `Span` are `void`, and `begin` and `end` compile to nothing, so
//! a release runner reads no clock in the hot path. With it, the brackets stay
//! dormant until a caller installs an accumulator or the span recorder runs.
//! `bench-model --profile-cpu-decode` is the one caller that installs one, and
//! each closed bracket also feeds the span recorder as an `op_*` span.

const std = @import("std");
const core_log = @import("../core/log.zig");
const span = @import("../core/span.zig");

// -- constants --------------------------------------------------------------

/// The empty slot value of both build modes.
pub const slot_off: Slot = if (core_log.hot) null else {};

// -- types ------------------------------------------------------------------

/// The runner family a profile belongs to. Each family charges its own subset
/// of `ProfileCat`.
pub const Family = enum { llama, qwen };

/// The op categories of one decode step. The declaration order is the row
/// order of the report, and the rows of each family keep the order of its
/// forward pass.
pub const ProfileCat = enum {
    embed,
    rmsnorm,
    gdn_proj,
    gdn_step,
    gdn_out,
    qkv_proj,
    attn_proj,
    rope,
    qk_norm_rope,
    kv_commit,
    attention,
    attn_gate,
    o_proj,
    gate_up_proj,
    swiglu,
    down_proj,
    residual,
    lm_head,

    /// True for the row-parallel matmuls. Each runner charges them at its one
    /// matmul boundary, so the layer body carries no matmul bracket.
    pub fn isMatmul(self: ProfileCat) bool {
        return switch (self) {
            .gdn_proj,
            .gdn_out,
            .qkv_proj,
            .attn_proj,
            .o_proj,
            .gate_up_proj,
            .down_proj,
            .lm_head,
            => true,
            else => false,
        };
    }

    /// True for the categories that fan out on the pool: the matmuls,
    /// attention, and the GDN step. The report labels the other rows serial.
    pub fn isParallel(self: ProfileCat) bool {
        return self.isMatmul() or self == .attention or self == .gdn_step;
    }

    /// True when `family` charges this category.
    pub fn usedBy(self: ProfileCat, family: Family) bool {
        return switch (self) {
            .embed,
            .rmsnorm,
            .attention,
            .o_proj,
            .gate_up_proj,
            .swiglu,
            .down_proj,
            .residual,
            .lm_head,
            => true,
            .qkv_proj, .rope => family == .llama,
            .gdn_proj,
            .gdn_step,
            .gdn_out,
            .attn_proj,
            .qk_norm_rope,
            .kv_commit,
            .attn_gate,
            => family == .qwen,
        };
    }
};

/// Per-category CPU-time accumulator. It is `void` without `-Dtrace`.
pub const CpuProfile = if (core_log.hot) struct {
    family: Family,
    ns: std.EnumArray(ProfileCat, u64) = .initFill(0),
    steps: u64 = 0,

    /// Nanoseconds charged to one category.
    pub fn nsOf(self: *const CpuProfile, cat: ProfileCat) u64 {
        return self.ns.get(cat);
    }

    /// Sum across categories. The per-step CPU time is `total() / steps`.
    pub fn total(self: *const CpuProfile) u64 {
        var sum: u64 = 0;
        for (self.ns.values) |v| {
            sum += v;
        }

        return sum;
    }

    /// Time spent in the categories that fan out on the pool.
    pub fn nsParallel(self: *const CpuProfile) u64 {
        var sum: u64 = 0;
        for (std.enums.values(ProfileCat)) |cat| {
            if (cat.isParallel()) {
                sum += self.ns.get(cat);
            }
        }

        return sum;
    }
} else void;

/// The runner-side timer slot. It is `void` without `-Dtrace`, so the release
/// runner carries neither the pointer nor the branch that reads it.
pub const Slot = if (core_log.hot) ?*CpuProfile else void;

/// One open timing span. It is `void` without `-Dtrace`.
pub const Span = if (core_log.hot) std.Io.Clock.Timestamp else void;

// -- api --------------------------------------------------------------------

/// Open a timing span. With `-Dtrace`, the span reads the clock only when a
/// profile is installed or the span recorder runs. Otherwise it returns
/// `undefined`, which `end` never reads because its guard is false. The off
/// path is then two predictable branches and zero clock reads.
pub inline fn begin(slot: Slot, io: std.Io) Span {
    if (comptime core_log.hot) {
        return if (slot != null or span.enabled()) std.Io.Clock.Timestamp.now(
            io,
            .awake,
        ) else undefined;
    } else {
        return {};
    }
}

/// Close a timing span opened by `begin`, charge the elapsed time to `cat`,
/// and hand the span to the recorder. The recorder only turns off after
/// startup, so a span it sees here always opened with a clock read.
pub inline fn end(slot: Slot, io: std.Io, t0: Span, cat: ProfileCat) void {
    if (comptime core_log.hot) {
        const tracing = span.enabled();
        if (slot == null and !tracing) return;

        const t1 = std.Io.Clock.Timestamp.now(io, .awake);
        if (slot) |p| {
            std.debug.assert(cat.usedBy(p.family));

            p.ns.getPtr(cat).* += @intCast(t0.durationTo(t1).raw.toNanoseconds());
        }

        if (tracing) {
            span.record(spanName(cat), t0.raw, t1.raw);
        }
    }
}

/// The span name of a category.
pub fn spanName(cat: ProfileCat) span.Name {
    return switch (cat) {
        inline else => |c| @field(span.Name, "op_" ++ @tagName(c)),
    };
}

// -- tests ------------------------------------------------------------------

test "cpu profile: each family uses a fixed category set" {
    var llama: u32 = 0;
    var qwen: u32 = 0;
    for (std.enums.values(ProfileCat)) |cat| {
        // No category is dead: at least one family charges it.
        try std.testing.expect(cat.usedBy(.llama) or cat.usedBy(.qwen));
        if (cat.usedBy(.llama)) {
            llama += 1;
        }

        if (cat.usedBy(.qwen)) {
            qwen += 1;
        }
    }

    try std.testing.expectEqual(@as(u32, 11), llama);
    try std.testing.expectEqual(@as(u32, 16), qwen);
    // The lm_head and the FFN matmuls are common to both families.
    for ([_]ProfileCat{ .gate_up_proj, .down_proj, .lm_head }) |cat| {
        try std.testing.expect(cat.isMatmul());
        try std.testing.expect(cat.usedBy(.llama));
        try std.testing.expect(cat.usedBy(.qwen));
    }

    try std.testing.expect(!ProfileCat.attention.isMatmul());
    try std.testing.expect(!ProfileCat.gdn_step.isMatmul());
    // Every matmul fans out. Attention and the GDN step fan out without a
    // matmul boundary.
    for (std.enums.values(ProfileCat)) |cat| {
        if (cat.isMatmul()) {
            try std.testing.expect(cat.isParallel());
        }
    }

    try std.testing.expect(ProfileCat.attention.isParallel());
    try std.testing.expect(ProfileCat.gdn_step.isParallel());
    try std.testing.expect(!ProfileCat.rmsnorm.isParallel());
}

test "cpu profile: every category has a span name in its order" {
    for (std.enums.values(ProfileCat), 0..) |cat, i| {
        const name = spanName(cat);
        try std.testing.expect(name.isHot());
        try std.testing.expectEqual(@intFromEnum(span.Name.op_embed) + i, @intFromEnum(name));
    }
}
