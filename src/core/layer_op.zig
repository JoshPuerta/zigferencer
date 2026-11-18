//! The one list of op names: the ops of a forward pass, and the keys that a CPU span and a GPU
//! mark time. Pure values, std only.
//!
//! Ops. `LayerOp` holds the ops of a decoder layer, then the ops of the runner: the embed, the
//! final norm, the lm_head, and the two transfers of a GPU step. A decoder layer calls each layer
//! op at most once, so a layer op keys the binding of one call. Each op has one `OpClass`, so a
//! profile table can group its lines.
//!
//! Keys. A CPU span or a GPU mark times one `OpKey`: one op, one pass of a split op, or the op set
//! of one fused kernel. A pass carries its `OpPart`, and each part belongs to one op. A profile
//! sums the passes into the line of their op. A fused kernel records in the method of its first op
//! and marks the whole set. The set must stand in `fused_sets`.
//! `fused_sets` is empty when no kernel fuses two ops.
//!
//! Names. A key reads as its op, as `op.part` for a pass, or as `a+b` for a fused set. The span
//! names, the profile lines, and the trace args read this text.
//!
//! Comptime. The key count and the name table come from the enums and from `fused_sets`.

const std = @import("std");

const assert = std.debug.assert;

// -- constants --------------------------------------------------------------

pub const op_count: u32 = std.meta.fields(LayerOp).len;
pub const part_count: u32 = std.meta.fields(OpPart).len;

/// The op sets of the fused kernels. A kernel that fuses two or more ops adds its set here.
pub const fused_sets: []const OpSet = &.{};

comptime {
    assert(OpKey.count <= std.math.maxInt(u8) + 1);
    for (fused_sets) |set| {
        assert(set.count() >= 2);
    }
}

/// The text of each key, at its value.
const key_names: [OpKey.count][]const u8 = blk: {
    var out: [OpKey.count][]const u8 = undefined;
    for (std.meta.fields(LayerOp), 0..) |f, i| {
        out[i] = f.name;
    }

    for (std.meta.fields(OpPart), 0..) |f, i| {
        const p: OpPart = @enumFromInt(f.value);
        out[op_count + i] = @tagName(p.op()) ++ "." ++ f.name;
    }

    for (fused_sets, 0..) |set, i| {
        var text: []const u8 = "";
        var it = set.iterator();
        while (it.next()) |member| {
            text = if (text.len == 0) @tagName(member) else text ++ "+" ++ @tagName(member);
        }

        out[op_count + part_count + i] = text;
    }

    break :blk out;
};

// -- types ------------------------------------------------------------------

/// One op of a forward pass: the ops of a decoder layer, then the ops of the runner.
pub const LayerOp = enum(u8) {
    attn_norm,
    attn_proj,
    qk_norm,
    rope,
    kv_commit,
    attention,
    attn_gate,
    o_proj,
    gdn_proj,
    gdn,
    gdn_out,
    mixer_residual,
    ffn_norm,
    gate_up_proj,
    swiglu,
    down_proj,
    ffn_residual,
    /// The lookup of the embedding of each token.
    embed,
    final_norm,
    lm_head,
    /// The copy of the input rows into the residual of a GPU step.
    stage_in,
    /// The copy of the logits rows out of a GPU step.
    readback,

    pub fn class(self: LayerOp) OpClass {
        return switch (self) {
            .attn_proj, .o_proj, .gdn_proj, .gdn_out => .matmul,
            .gate_up_proj, .down_proj, .lm_head => .matmul,
            .attention => .attention,
            .gdn => .recurrent,
            .rope, .attn_gate, .mixer_residual, .swiglu, .ffn_residual => .elementwise,
            .attn_norm, .qk_norm, .ffn_norm, .final_norm => .norm,
            .kv_commit, .embed, .stage_in, .readback => .transfer,
        };
    }
};

/// The ops that one span or one mark times.
pub const OpSet = std.EnumSet(LayerOp);

/// One pass of an op that a kernel set splits.
pub const OpPart = enum(u8) {
    conv,
    scan,
    split,
    merge,
    narrow,
    copy,

    /// The op that splits into this pass.
    pub fn op(self: OpPart) LayerOp {
        return switch (self) {
            .conv, .scan => .gdn,
            .split, .merge => .attention,
            .narrow, .copy => .kv_commit,
        };
    }
};

/// The kind of work of an op: the group of its profile line.
pub const OpClass = enum(u8) { matmul, attention, recurrent, elementwise, norm, transfer };

/// What one span or one mark times: one op, one pass of an op, or one fused op set. The values
/// hold one key per op, then one per part, then one per fused set.
pub const OpKey = enum(u8) {
    _,

    pub const count: u32 = op_count + part_count + fused_sets.len;

    /// The key of `set`, or of the pass `part` of the one op of `set`.
    pub fn of(set: OpSet, part_of: ?OpPart) OpKey {
        assert(set.count() >= 1);

        if (part_of) |p| {
            assert(set.eql(.initOne(p.op())));
            return @enumFromInt(op_count + @intFromEnum(p));
        }

        if (set.count() == 1) return ofOp(firstOp(set));

        // A kernel set marks only a fused set that `fused_sets` holds.
        const fused_index = for (fused_sets, 0..) |fused, k| {
            if (fused.eql(set)) break k;
        } else unreachable;
        return @enumFromInt(op_count + part_count + fused_index);
    }

    pub fn ofOp(one: LayerOp) OpKey {
        return @enumFromInt(@intFromEnum(one));
    }

    /// The ops that the key times.
    pub fn ops(self: OpKey) OpSet {
        const i = self.index();
        if (i < op_count) return .initOne(@enumFromInt(i));
        if (i < op_count + part_count) return .initOne(self.part().?.op());

        // A fused key exists only while `fused_sets` holds a set, so the build skips the index.
        const fused = i - op_count - part_count;
        return if (comptime fused_sets.len == 0) unreachable else fused_sets[fused];
    }

    /// The pass of a pass key. Null for every other key.
    pub fn part(self: OpKey) ?OpPart {
        const i = self.index();
        if (i < op_count or i >= op_count + part_count) return null;
        return @enumFromInt(i - op_count);
    }

    /// The op of the line that holds the key: the op of a pass, or the first op of a fused set.
    pub fn op(self: OpKey) LayerOp {
        return firstOp(self.ops());
    }

    pub fn class(self: OpKey) OpClass {
        return self.op().class();
    }

    /// The text of the key: `op`, `op.part`, or `a+b`.
    pub fn name(self: OpKey) []const u8 {
        return key_names[self.index()];
    }

    fn index(self: OpKey) u32 {
        const i: u32 = @intFromEnum(self);
        assert(i < count);
        return i;
    }
};

// -- helpers ----------------------------------------------------------------

/// The op of lowest value in `set`.
fn firstOp(set: OpSet) LayerOp {
    var it = set.iterator();
    return it.next().?;
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;

test "OpKey: every key maps back to its ops and its part, and reads as its name" {
    var i: u32 = 0;
    while (i < OpKey.count) : (i += 1) {
        const key: OpKey = @enumFromInt(i);
        try testing.expectEqual(key, OpKey.of(key.ops(), key.part()));
        try testing.expect(key.ops().contains(key.op()));
        if (key.part()) |p| {
            try testing.expectEqual(p.op(), key.op());
        }
    }

    try testing.expectEqualStrings("attn_proj", OpKey.ofOp(.attn_proj).name());
    const split = OpKey.of(.initOne(.attention), .split);
    try testing.expectEqualStrings("attention.split", split.name());
    try testing.expectEqualStrings("gdn.conv", OpKey.of(.initOne(.gdn), .conv).name());
    try testing.expectEqual(op_count + part_count, OpKey.count);
}

test "OpPart: each split op holds two passes, and only the three split ops hold any" {
    var passes: std.EnumArray(LayerOp, u32) = .initFill(0);
    for (std.enums.values(OpPart)) |p| {
        passes.getPtr(p.op()).* += 1;
    }

    for (std.enums.values(LayerOp)) |op| {
        const split = op == .gdn or op == .attention or op == .kv_commit;
        try testing.expectEqual(@as(u32, if (split) 2 else 0), passes.get(op));
    }
}
