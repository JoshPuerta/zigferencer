//! The attention recorder: the one control unit that turns an `attn_split.AttnPlan` into
//! dispatches. The fused and hybrid runners call it for every full-attention layer of a step and
//! of a prefill chunk, so the record order of the resident rows, the split passes, and the merges
//! has one home.
//!
//! Kernel set. The recorder knows no backend. It records through an attention kernel set, a type
//! with a fixed method set: `recordResident`, `recordResidentRun`, `recordSplit`, `recordMerge`,
//! and `recordBarrier`. A kernel set also declares two values: `order`, the class of its summation
//! order, and `tolerance`, its declared error against the CPU reference. A kernel set names its
//! command handle `Cmd` and its per-layer bindings `Binding`. The build selects the Vulkan kernel
//! set with `-Dattn-kernel`.
//!
//! Comptime. `assertKernels` checks the kernel set when the recorder is built, so a backend that
//! lacks a method fails the build.
//!
//! Record order. A plan records in this order:
//!
//!   1. The resident rows: one dispatch per row for a step, or one run of the resident prefix for
//!      a prefill chunk.
//!   2. Per record group: the split passes, a barrier, then the merges. Entry `k` of a group
//!      writes scratch row `k` at the stride of the group.
//!   3. A barrier before each group after the first, because its split passes write the scratch
//!      rows that the merges before it read.
//! The caller records the barrier after the last merge.

const std = @import("std");
const attn_split = @import("../core/attn_split.zig");
const limits = @import("../core/limits.zig");
const prop = @import("../core/prop.zig");
const step_row = @import("step_row.zig");

const assert = std.debug.assert;

// -- api --------------------------------------------------------------------

/// The plan of the batched rows of one step against a scratch of `max_splits` splits.
/// Row `i` reads its own slot over `pos + 1` positions.
pub fn planStep(rows: []const step_row.StepRow, max_splits: u32) attn_split.AttnPlan {
    assert(rows.len >= 1);
    assert(rows.len <= attn_split.max_plan_rows);

    var sites: [attn_split.max_plan_rows]attn_split.RowSite = undefined;
    for (rows, sites[0..rows.len]) |r, *site| {
        site.* = .{ .slot = r.slot, .t_len = r.pos + 1 };
    }

    const plan = attn_split.planRows(sites[0..rows.len], max_splits);
    assert(plan.resident_count + plan.split_count == rows.len);
    return plan;
}

/// Record the attention of one layer of a step: every dispatch of `plan` through `kernels` into
/// `cmd`.
pub fn record(
    comptime Kernels: type,
    kernels: Kernels,
    cmd: Kernels.Cmd,
    plan: *const attn_split.AttnPlan,
    binding: Kernels.Binding,
) void {
    comptime assertKernels(Kernels);
    plan.assertValid();
    for (plan.residentRows()) |r| {
        kernels.recordResident(cmd, binding, r);
    }

    recordGroups(Kernels, kernels, cmd, plan, binding);
}

/// Record the attention of one layer of a prefill chunk from `start_pos`. The resident rows are
/// a prefix, so one run of the resident kernel records them.
pub fn recordPrefill(
    comptime Kernels: type,
    kernels: Kernels,
    cmd: Kernels.Cmd,
    plan: *const attn_split.AttnPlan,
    binding: Kernels.Binding,
    start_pos: u32,
) void {
    comptime assertKernels(Kernels);
    plan.assertValid();

    // Check that the plan matches the chunk: resident rows first, positions ascending.
    const resident = plan.residentRows();
    for (resident, 0..) |r, i| {
        assert(r.row == i);
        assert(r.t_len == start_pos + r.row + 1);
    }

    for (plan.splitRows()) |r| {
        assert(r.row >= resident.len);
        assert(r.t_len == start_pos + r.row + 1);
    }

    // Record the resident prefix as one run.
    if (resident.len > 0) {
        const count: u32 = @intCast(resident.len);
        // The run ends at or before S, the bound of the resident kernel.
        assert(start_pos + count <= limits.attn_split_len);
        kernels.recordResidentRun(cmd, binding, start_pos, count);
    }

    recordGroups(Kernels, kernels, cmd, plan, binding);
}

// -- helpers ----------------------------------------------------------------

fn recordGroups(
    comptime Kernels: type,
    kernels: Kernels,
    cmd: Kernels.Cmd,
    plan: *const attn_split.AttnPlan,
    binding: Kernels.Binding,
) void {
    for (plan.recordGroups(), 0..) |g, gi| {
        if (gi > 0) {
            kernels.recordBarrier(cmd);
        }

        const rows = plan.rowsOf(g);
        for (rows, 0..) |r, k| {
            kernels.recordSplit(cmd, binding, r, @intCast(k), g.stride);
        }

        kernels.recordBarrier(cmd);
        for (rows, 0..) |r, k| {
            kernels.recordMerge(cmd, binding, r, @intCast(k), g.stride);
        }
    }
}

/// A kernel set must declare the method set and both values. One that lacks any fails the build.
fn assertKernels(comptime Kernels: type) void {
    for ([_][]const u8{
        "recordResident",
        "recordResidentRun",
        "recordSplit",
        "recordMerge",
        "recordBarrier",
        "order",
        "tolerance",
        "Cmd",
        "Binding",
    }) |name| {
        if (!@hasDecl(Kernels, name)) {
            @compileError("attention kernel set " ++ @typeName(Kernels) ++ " lacks " ++ name);
        }
    }
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;
const S = limits.attn_split_len;

/// A kernel set that logs each call, so a test reads the record order back.
const LogKernels = struct {
    log: *Log,

    pub const order = .split_ascending;
    pub const tolerance = 0;
    pub const Cmd = u32;
    pub const Binding = void;

    const Event = union(enum) {
        resident: u32,
        resident_run: struct { start_pos: u32, count: u32 },
        split: struct { row: u32, scratch: u32, stride: u32 },
        merge: struct { row: u32, scratch: u32, stride: u32 },
        barrier,
    };

    const Log = struct {
        events: [4 * attn_split.max_plan_rows]Event = undefined,
        len: u32 = 0,

        fn add(self: *Log, e: Event) void {
            assert(self.len < self.events.len);

            self.events[self.len] = e;
            self.len += 1;
        }
    };

    pub fn recordResident(self: LogKernels, cmd: Cmd, _: Binding, r: attn_split.PlanRow) void {
        assert(cmd == 7);

        self.log.add(.{ .resident = r.row });
    }

    pub fn recordResidentRun(
        self: LogKernels,
        cmd: Cmd,
        _: Binding,
        start_pos: u32,
        count: u32,
    ) void {
        assert(cmd == 7);

        self.log.add(.{ .resident_run = .{ .start_pos = start_pos, .count = count } });
    }

    pub fn recordSplit(
        self: LogKernels,
        cmd: Cmd,
        _: Binding,
        r: attn_split.PlanRow,
        scratch_row: u32,
        stride: u32,
    ) void {
        assert(cmd == 7);

        self.log.add(.{ .split = .{ .row = r.row, .scratch = scratch_row, .stride = stride } });
    }

    pub fn recordMerge(
        self: LogKernels,
        cmd: Cmd,
        _: Binding,
        r: attn_split.PlanRow,
        scratch_row: u32,
        stride: u32,
    ) void {
        assert(cmd == 7);

        self.log.add(.{ .merge = .{ .row = r.row, .scratch = scratch_row, .stride = stride } });
    }

    pub fn recordBarrier(self: LogKernels, cmd: Cmd) void {
        assert(cmd == 7);

        self.log.add(.barrier);
    }
};

/// Replay a log against a model of the scratch: the row each scratch row holds, whether a barrier
/// made it visible, and whether its merge read it. Each merge reads at the stride of its split.
fn checkLog(
    log: *const LogKernels.Log,
    t_lens: []const u32,
    budget: u32,
    resident_seen: []bool,
    merged_seen: []bool,
) !void {
    const Cell = struct {
        row: ?u32 = null,
        stride: u32 = 0,
        visible: bool = false,
        merged: bool = true,
    };
    var cells = [_]Cell{.{}} ** attn_split.max_plan_rows;
    // A merge since the last barrier marks a read that a later split must wait for.
    var read_since_barrier = false;
    for (log.events[0..log.len]) |e| {
        switch (e) {
            .resident => |row| {
                try testing.expect(t_lens[row] <= S);
                try testing.expect(!resident_seen[row]);
                resident_seen[row] = true;
            },
            .resident_run => |run| {
                var i: u32 = 0;
                while (i < run.count) : (i += 1) {
                    try testing.expectEqual(run.start_pos + i + 1, t_lens[i]);
                    try testing.expect(t_lens[i] <= S);
                    try testing.expect(!resident_seen[i]);
                    resident_seen[i] = true;
                }
            },
            .split => |s| {
                try testing.expect(t_lens[s.row] > S);
                try testing.expect(attn_split.splitCount(t_lens[s.row]) <= s.stride);
                try testing.expect((@as(u64, s.scratch) + 1) * s.stride <= budget);
                // The row before it was merged, and that read finished.
                try testing.expect(cells[s.scratch].merged);
                try testing.expect(!read_since_barrier);
                cells[s.scratch] = .{
                    .row = s.row,
                    .stride = s.stride,
                    .visible = false,
                    .merged = false,
                };
            },
            .merge => |m| {
                const cell = &cells[m.scratch];
                try testing.expectEqual(@as(?u32, m.row), cell.row);
                try testing.expectEqual(cell.stride, m.stride);
                try testing.expect(cell.visible);
                try testing.expect(!cell.merged);
                try testing.expect(!merged_seen[m.row]);
                cell.merged = true;
                merged_seen[m.row] = true;
                read_since_barrier = true;
            },
            .barrier => {
                for (&cells) |*c| {
                    c.visible = c.row != null;
                }

                read_since_barrier = false;
            },
        }
    }

    for (cells) |c| {
        try testing.expect(c.merged);
    }
}

fn checkRecordMerge(_: void, c: prop.Trial) anyerror!void {
    var prng: std.Random.DefaultPrng = .init(c.seed);
    const random = prng.random();
    var rows: [attn_split.max_plan_rows]step_row.StepRow = undefined;
    var t_lens: [attn_split.max_plan_rows]u32 = undefined;
    const n = random.intRangeAtMost(u32, 1, attn_split.max_plan_rows);
    for (rows[0..n], t_lens[0..n]) |*r, *t| {
        r.* = .{
            .slot = random.intRangeLessThan(u32, 0, 8),
            .token = 0,
            .pos = random.intRangeLessThan(u32, 0, 3 * S),
            .want_logits = false,
        };
        t.* = r.pos + 1;
    }

    const max_splits = random.intRangeAtMost(u32, 3, 12);
    const plan = planStep(rows[0..n], max_splits);
    var log: LogKernels.Log = .{};
    record(LogKernels, .{ .log = &log }, 7, &plan, {});
    var resident_seen = [_]bool{false} ** attn_split.max_plan_rows;
    var merged_seen = [_]bool{false} ** attn_split.max_plan_rows;
    try checkLog(&log, t_lens[0..n], plan.budget, &resident_seen, &merged_seen);
    // Every row ran exactly one of the two paths.
    for (t_lens[0..n], 0..) |t, i| {
        try testing.expectEqual(t <= S, resident_seen[i]);
        try testing.expectEqual(t > S, merged_seen[i]);
    }
}

test "record: each merge reads its split row behind a barrier, no split clobbers an unread row" {
    try prop.forAll(
        .{ .site = 0x0031_7ec0, .cases = 300, .max_size = 299 },
        {},
        prop.Trial,
        prop.genTrial(void),
        checkRecordMerge,
    );
}

fn checkRecordPrefill(_: void, c: prop.Trial) anyerror!void {
    var prng: std.Random.DefaultPrng = .init(c.seed);
    const random = prng.random();
    var t_lens: [attn_split.max_plan_rows]u32 = undefined;
    const trial = c.index;
    const b = random.intRangeAtMost(u32, 1, attn_split.max_plan_rows);
    // Half the chunks start inside the first split, so the chunk can straddle S.
    const start_pos = if (trial % 2 == 0) random.intRangeLessThan(
        u32,
        0,
        S,
    ) else random.intRangeAtMost(
        u32,
        S,
        4 * S,
    );

    for (t_lens[0..b], 0..) |*t, i| {
        t.* = start_pos + @as(u32, @intCast(i)) + 1;
    }

    const plan = attn_split.planPrefill(0, start_pos, b, attn_split.splitCount(8 * S));
    var log: LogKernels.Log = .{};
    recordPrefill(LogKernels, .{ .log = &log }, 7, &plan, {}, start_pos);
    var runs: u32 = 0;
    for (log.events[0..log.len]) |e| {
        if (e == .resident_run) {
            runs += 1;
        }

        try testing.expect(e != .resident);
    }

    try testing.expectEqual(
        @as(u32, @intFromBool(attn_split.prefillResident(start_pos, b) > 0)),
        runs,
    );
    var resident_seen = [_]bool{false} ** attn_split.max_plan_rows;
    var merged_seen = [_]bool{false} ** attn_split.max_plan_rows;
    try checkLog(&log, t_lens[0..b], plan.budget, &resident_seen, &merged_seen);
    for (t_lens[0..b], 0..) |t, i| {
        try testing.expectEqual(t <= S, resident_seen[i]);
        try testing.expectEqual(t > S, merged_seen[i]);
    }
}

test "recordPrefill: one resident run for the prefix, then the split groups, every row once" {
    try prop.forAll(
        .{ .site = 0x0031_7ec1, .cases = 300, .max_size = 299 },
        {},
        prop.Trial,
        prop.genTrial(void),
        checkRecordPrefill,
    );
}

test "record: a decode row past one split records its split, one barrier, and its merge" {
    var log: LogKernels.Log = .{};
    const plan = attn_split.planDecode(S + 1);
    record(LogKernels, .{ .log = &log }, 7, &plan, {});
    try testing.expectEqual(@as(u32, 3), log.len);
    try testing.expectEqual(
        LogKernels.Event{ .split = .{ .row = 0, .scratch = 0, .stride = 2 } },
        log.events[0],
    );
    try testing.expectEqual(LogKernels.Event.barrier, log.events[1]);
    try testing.expectEqual(
        LogKernels.Event{ .merge = .{ .row = 0, .scratch = 0, .stride = 2 } },
        log.events[2],
    );
}
