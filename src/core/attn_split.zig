//! The geometry of split-depth GPU attention.
//!
//! Comptime. An assert keeps `splitCount` inside 32 bits.
//!
//! Splits. A decode or prefill row of `t_len` positions splits its walk into runs of
//! S = `limits.attn_split_len` positions. Split `j` holds the positions
//! `[j * S, min((j + 1) * S, t_len))`, so every position lands in one split and no split is empty.
//! Each split writes one partial record per query head: the value sums, the maximum score, and the
//! sum of the shifted exponentials. A merge pass combines the records in split order. The split
//! count is a function of `t_len` alone, so a row gets the same geometry at every window, batch,
//! and slot.
//!
//! Scratch. The records live in a scratch of `limits.attn_split_rows` rows. A record element
//! offset is a mixed-radix numeral with the radices `n_heads`, `max_splits`, and the record width
//! (Knuth, TAOCP vol 1, section 2.2.6). So each coordinate of the record box gets its own element,
//! and every element stays below the scratch size. A record group of the planner lays out its rows
//! at its own stride: row `k` of the group, head `h`, and split `j` sit at record
//! `(k * n_heads + h) * stride + j`. The group holds at most `rows * max_splits` records per head.
//!
//! Planner. It turns the rows of one step into an `AttnPlan`: the resident rows, the split rows,
//! and the record groups that share the scratch. A group takes its first row, then the next while
//! (rows in group) * (largest splitCount in group) <= budget. The plan names no backend.
//! `runtime/attn_record.zig` walks it through the attention kernel set of the build.

const std = @import("std");
const prop = @import("utils/prop.zig");
const limits = @import("limits.zig");

const assert = std.debug.assert;

// -- constants --------------------------------------------------------------

/// Bytes one scratch element occupies. The records are f32 at every KV dtype.
const elem_bytes: u64 = @sizeOf(f32);

/// Splits of a row at the engine context bound: the deepest record box.
const max_splits_bound: u32 = splitCount(limits.max_model_ctx);

comptime {
    // `splitCount` adds `S - 1` to a row length inside 32 bits.
    assert(@as(u64, limits.max_model_ctx) + limits.attn_split_len - 1 <= std.math.maxInt(u32));
}

/// Rows one plan holds: the widest step a runner records, one serve prefill
/// chunk.
pub const max_plan_rows: u32 = limits.max_prefill_chunk;

// -- types ------------------------------------------------------------------

/// The partial records of the split attention: `rows * max_splits` records per query head.
/// A record holds `head_dim` value sums, then the maximum score, then the exponential sum.
pub const Scratch = struct {
    rows: u32,
    n_heads: u32,
    head_dim: u32,
    max_splits: u32,

    /// The scratch of a model with `n_heads` query heads of `head_dim`
    /// elements, for rows of up to `ceiling` positions.
    pub fn init(n_heads: u32, head_dim: u32, ceiling: u32) Scratch {
        assert(n_heads >= 1);
        assert(head_dim >= 1);
        assert(ceiling >= 1);
        assert(ceiling <= limits.max_model_ctx);

        const scratch: Scratch = .{
            .rows = limits.attn_split_rows,
            .n_heads = n_heads,
            .head_dim = head_dim,
            .max_splits = splitCount(ceiling),
        };
        scratch.assertValid();
        return scratch;
    }

    /// Elements one record holds: the value sums, then the maximum, then the
    /// exponential sum.
    pub fn recordElems(self: Scratch) u32 {
        self.assertValid();
        const elems = self.head_dim + 2;
        assert(elems > self.head_dim);
        return elems;
    }

    /// The element index of element `e` of the record of `(row, head, split)`.
    pub fn offset(self: Scratch, row: u32, head: u32, split: u32, e: u32) u64 {
        assert(row < self.rows);
        assert(head < self.n_heads);
        assert(split < self.max_splits);
        assert(e < self.recordElems());

        const index = ((@as(u64, row) * self.n_heads + head) * self.max_splits + split) *
            self.recordElems() + e;
        assert(index < self.totalElems());
        return index;
    }

    /// Elements the scratch holds: every record of every row.
    pub fn totalElems(self: Scratch) u64 {
        const elems = @as(u64, self.rows) * self.n_heads * self.max_splits * self.recordElems();
        assert(elems >= 3);
        return elems;
    }

    /// Bytes the scratch holds.
    pub fn totalBytes(self: Scratch) u64 {
        const record_bytes = @as(u64, self.recordElems()) * elem_bytes;
        const bytes = @as(u64, self.rows) * self.n_heads * self.max_splits * record_bytes;
        assert(bytes == self.totalElems() * elem_bytes);
        return bytes;
    }

    fn assertValid(self: Scratch) void {
        assert(self.rows >= 1);
        assert(self.rows <= limits.attn_split_rows);
        assert(self.n_heads >= 1);
        assert(self.head_dim >= 1);
        // `recordElems` adds the maximum and the sum inside 32 bits.
        assert(self.head_dim <= std.math.maxInt(u32) - 2);
        assert(self.max_splits >= 1);
        assert(self.max_splits <= max_splits_bound);

        // The engine bounds neither the head count nor the head width, so check in 128 bits.
        const widest = @as(u128, self.rows) * self.n_heads * self.max_splits *
            (@as(u128, self.head_dim) + 2) * elem_bytes;
        assert(widest <= std.math.maxInt(u64));
    }
};

/// The input of the planner for one row of a step: the KV stream the row
/// reads and the positions it attends to.
pub const RowSite = struct {
    slot: u32,
    t_len: u32,
};

/// One row of a plan. Query row `row` reads stream `slot` over `t_len`
/// positions and writes output row `row`.
pub const PlanRow = struct {
    row: u32,
    slot: u32,
    t_len: u32,
};

/// A record group: the split rows `[first, first + len)` of a plan. Entry `k`
/// of a group holds scratch row `k` from its split pass to its merge. The
/// records of a group row sit `stride` splits apart per head, and `stride` is
/// the largest split count of the rows of the group.
pub const PlanGroup = struct {
    first: u32,
    len: u32,
    stride: u32,
};

/// A greedy record group of the rows `[lo, hi)` of a count slice.
const BudgetGroup = struct {
    lo: u32,
    hi: u32,
};

/// The attention dispatches of one step, as a value that names no backend. A resident row runs
/// one dispatch. A split row runs its split pass in a record group, and the merges of a group
/// follow its split passes. Only the first `count` entries of each array hold data. `budget` is
/// the record count, in units of one head, that one group can hold.
pub const AttnPlan = struct {
    budget: u32,
    resident_count: u32,
    split_count: u32,
    group_count: u32,
    resident: [max_plan_rows]PlanRow,
    split: [max_plan_rows]PlanRow,
    groups: [max_plan_rows]PlanGroup,

    pub fn residentRows(self: *const AttnPlan) []const PlanRow {
        assert(self.resident_count <= max_plan_rows);

        return self.resident[0..self.resident_count];
    }

    pub fn splitRows(self: *const AttnPlan) []const PlanRow {
        assert(self.split_count <= max_plan_rows);

        return self.split[0..self.split_count];
    }

    pub fn recordGroups(self: *const AttnPlan) []const PlanGroup {
        assert(self.group_count <= max_plan_rows);

        return self.groups[0..self.group_count];
    }

    /// The split rows of group `g`. Entry `k` writes scratch row `k`.
    pub fn rowsOf(self: *const AttnPlan, g: PlanGroup) []const PlanRow {
        assert(g.len >= 1);
        assert(g.first + g.len <= self.split_count);
        assert(@as(u64, g.len) * g.stride <= self.budget);

        return self.split[g.first..][0..g.len];
    }

    /// Each resident row stays inside one split, and each split row goes past one split. The
    /// groups cover the split rows in order. A row of a group has a split count within its stride.
    pub fn assertValid(self: *const AttnPlan) void {
        assert(self.budget >= 1);
        assert(self.resident_count + self.split_count <= max_plan_rows);
        assert(self.group_count <= self.split_count);

        for (self.residentRows()) |r| {
            assert(!usesSplit(r.t_len));
        }

        for (self.splitRows()) |r| {
            assert(usesSplit(r.t_len));
        }

        var next: u32 = 0;
        for (self.recordGroups()) |g| {
            assert(g.first == next);
            assert(g.len >= 1);
            assert(@as(u64, g.len) * g.stride <= self.budget);

            var widest: u32 = 0;
            for (self.split[g.first..][0..g.len]) |r| {
                const count = splitCount(r.t_len);
                assert(count <= g.stride);
                widest = @max(widest, count);
            }

            // The stride is the largest count of the group, not more.
            assert(widest == g.stride);
            next += g.len;
        }

        assert(next == self.split_count);
    }
};

// -- api --------------------------------------------------------------------

/// The splits a row of `t_len` positions walks: `ceil(t_len / S)`.
pub fn splitCount(t_len: u32) u32 {
    assert(t_len >= 1);
    assert(t_len <= limits.max_model_ctx);

    const count = (t_len + limits.attn_split_len - 1) / limits.attn_split_len;
    assert(count >= 1);
    // The last split starts inside the row, so no split is empty.
    assert(@as(u64, count - 1) * limits.attn_split_len < t_len);
    // The splits cover the row.
    assert(@as(u64, count) * limits.attn_split_len >= t_len);
    return count;
}

/// Whether the GPU split pass takes heads of `head_dim` elements. The pass
/// stages the q row of a head in shared memory of
/// `limits.attn_max_head_dim` floats, and it reads K in whole 16-byte loads.
pub fn headDimFits(head_dim: u32) bool {
    const fits_lds = head_dim <= limits.attn_max_head_dim;
    const whole_loads = head_dim % limits.attn_head_dim_granule == 0;
    return head_dim >= 1 and fits_lds and whole_loads;
}

/// The workgroups of the GPU split pass that serve one GQA group of `group` query heads:
/// `ceil(group / attn_split_queries)`. Workgroup c serves the heads
/// `[c * attn_split_queries, min((c + 1) * attn_split_queries, group))`.
pub fn queryGroups(group: u32) u32 {
    assert(group >= 1);

    const count = (group + limits.attn_split_queries - 1) / limits.attn_split_queries;
    assert(count >= 1);
    // The last workgroup holds at least one head, and the workgroups cover the group.
    assert((count - 1) * limits.attn_split_queries < group);
    assert(@as(u64, count) * limits.attn_split_queries >= group);
    return count;
}

/// Whether a row of `t_len` positions takes the split path. A row of one
/// split runs the resident kernel.
pub fn usesSplit(t_len: u32) bool {
    const uses = t_len > limits.attn_split_len;
    assert(uses == (splitCount(t_len) >= 2));
    return uses;
}

/// The first device limit the scratch breaks (`limits.bufferFits`). The scratch is bound whole at
/// offset 0, so the verdict is never `.misaligned`.
pub fn fits(s: Scratch, lim: limits.BufferLimits) limits.BufferVerdict {
    const elems = s.totalElems();
    const bytes = s.totalBytes();
    var unaligned = lim;
    unaligned.offset_alignment = 1;
    const verdict = limits.bufferFits(elems, bytes, unaligned);
    assert(verdict != .misaligned);
    if (verdict == .fits) {
        assert(elems <= limits.max_kv_index_elems);
        assert(bytes <= lim.allocation_size);
        assert(bytes <= lim.storage_range);
    }

    return verdict;
}

/// Plan the attention of one step against the scratch of `max_splits` splits per row and head.
/// Row `i` of `sites` is query row `i` and output row `i`. A group stays within the
/// `limits.attn_split_rows * max_splits` records of the scratch.
pub fn planRows(sites: []const RowSite, max_splits: u32) AttnPlan {
    assert(max_splits >= 1);
    assert(max_splits <= max_splits_bound);

    return planRowsBudget(sites, limits.attn_split_rows * max_splits);
}

/// Plan the attention of one step with groups of at most `budget` records per
/// head. `planRows` passes the record count of the scratch. A smaller budget
/// gives smaller groups and the same record content.
pub fn planRowsBudget(sites: []const RowSite, budget: u32) AttnPlan {
    assert(sites.len >= 1);
    assert(sites.len <= max_plan_rows);
    assert(budget >= 1);

    var plan: AttnPlan = .{
        .budget = budget,
        .resident_count = 0,
        .split_count = 0,
        .group_count = 0,
        // The counts bound every read, and the loops write each entry below its count.
        .resident = undefined,
        .split = undefined,
        .groups = undefined,
    };
    // Sort the rows into resident rows and split rows.
    var counts: [max_plan_rows]u32 = undefined;
    for (sites, 0..) |site, i| {
        const row: PlanRow = .{ .row = @intCast(i), .slot = site.slot, .t_len = site.t_len };
        if (!usesSplit(site.t_len)) {
            plan.resident[plan.resident_count] = row;
            plan.resident_count += 1;
            continue;
        }

        counts[plan.split_count] = splitCount(site.t_len);
        // A row that the scratch cannot hold alone has no group.
        assert(counts[plan.split_count] <= budget);
        plan.split[plan.split_count] = row;
        plan.split_count += 1;
    }

    // Group the split rows greedily, left to right.
    if (plan.split_count > 0) {
        var spans: [max_plan_rows]BudgetGroup = undefined;
        for (budgetGroups(budget, counts[0..plan.split_count], &spans)) |span| {
            var widest: u32 = 0;
            for (counts[span.lo..span.hi]) |c| {
                widest = @max(widest, c);
            }

            plan.groups[plan.group_count] = .{
                .first = span.lo,
                .len = span.hi - span.lo,
                .stride = widest,
            };
            plan.group_count += 1;
        }
    }

    assert(plan.resident_count + plan.split_count == sites.len);
    plan.assertValid();
    return plan;
}

/// Plan one decode row of `t_len` positions on stream 0. A split row takes the
/// stride of its own split count, the smallest budget that holds it.
pub fn planDecode(t_len: u32) AttnPlan {
    const sites = [_]RowSite{.{ .slot = 0, .t_len = t_len }};
    const plan = planRowsBudget(&sites, splitCount(t_len));
    assert(plan.resident_count + plan.split_count == 1);
    return plan;
}

/// Plan the `b` rows of one prefill chunk on stream `slot`: row `i` sits at
/// position `start_pos + i`, so it attends to `start_pos + i + 1` positions.
/// The resident rows are the prefix that `prefillResident` counts.
pub fn planPrefill(slot: u32, start_pos: u32, b: u32, max_splits: u32) AttnPlan {
    assert(b >= 1);
    assert(b <= max_plan_rows);
    assert(@as(u64, start_pos) + b <= limits.max_model_ctx);

    var sites: [max_plan_rows]RowSite = undefined;
    for (sites[0..b], 0..) |*site, i| {
        site.* = .{ .slot = slot, .t_len = start_pos + @as(u32, @intCast(i)) + 1 };
    }

    const plan = planRows(sites[0..b], max_splits);
    assert(plan.resident_count == prefillResident(start_pos, b));
    for (plan.residentRows(), 0..) |r, i| {
        assert(r.row == i);
    }

    return plan;
}

/// The resident prefix of a prefill chunk of `b` rows from `start_pos`: the
/// rows whose depth `start_pos + i + 1` stays at or below S.
pub fn prefillResident(start_pos: u32, b: u32) u32 {
    const split_len = limits.attn_split_len;
    const count = if (start_pos < split_len) @min(b, split_len - start_pos) else 0;
    assert(count <= b);
    if (count > 0) assert(@as(u64, start_pos) + count <= split_len);

    return count;
}

// -- helpers ----------------------------------------------------------------

/// The row count of the greedy record group that starts at `lo`. The group
/// takes each next row while its row count times its largest count stays
/// within `budget`.
fn groupLen(budget: u32, counts: []const u32, lo: u32) u32 {
    assert(counts.len <= std.math.maxInt(u32));
    assert(lo < counts.len);
    assert(counts[lo] <= budget);

    var rows: u32 = 1;
    var widest: u32 = counts[lo];
    while (lo + rows < counts.len) {
        const next_widest = @max(widest, counts[lo + rows]);
        const cost = (@as(u64, rows) + 1) * @as(u64, next_widest);
        if (cost > budget) break;

        widest = next_widest;
        rows += 1;
    }

    assert(rows >= 1);
    assert(lo + rows <= counts.len);
    assert(@as(u64, rows) * widest <= budget);
    return rows;
}

/// Fill `out` with the greedy record groups of `counts`, left to right, and
/// return the filled prefix. The groups cover every row, and each group fits
/// `budget`.
fn budgetGroups(budget: u32, counts: []const u32, out: []BudgetGroup) []BudgetGroup {
    assert(counts.len >= 1);
    assert(counts.len <= std.math.maxInt(u32));
    assert(out.len >= counts.len);

    var filled: usize = 0;
    var lo: u32 = 0;
    // Each group holds at least one row, so the loop runs at most `counts.len` times.
    while (lo < counts.len) {
        const rows = groupLen(budget, counts, lo);
        assert(filled < counts.len);
        out[filled] = .{ .lo = lo, .hi = lo + rows };
        filled += 1;
        lo += rows;
    }

    assert(filled >= 1);
    assert(filled <= counts.len);
    assert(out[filled - 1].hi == counts.len);
    return out[0..filled];
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;
const S = limits.attn_split_len;

/// Limits with every bound far above any test scratch, so a test moves one.
const wide_limits: limits.BufferLimits = .{
    .storage_range = 1 << 40,
    .allocation_size = 1 << 40,
    .offset_alignment = 16,
};

/// The narrowest limits a device can report: the Vulkan minimum storage range, the 2 GiB
/// allocation size of a translation-layer driver, and the largest offset alignment Vulkan allows.
const spec_min_limits: limits.BufferLimits = .{
    .storage_range = 134_217_728,
    .allocation_size = 2_147_483_648,
    .offset_alignment = 256,
};

test "splitCount: one split up to S positions, two from S + 1, and the engine bound" {
    try testing.expectEqual(@as(u32, 1), splitCount(1));
    try testing.expectEqual(@as(u32, 1), splitCount(S));
    try testing.expectEqual(@as(u32, 2), splitCount(S + 1));
    try testing.expectEqual(@as(u32, 2), splitCount(2 * S));
    try testing.expectEqual(@as(u32, 3), splitCount(2 * S + 1));
    try testing.expectEqual(limits.max_model_ctx / S, splitCount(limits.max_model_ctx));
    try testing.expectEqual(max_splits_bound, splitCount(limits.max_model_ctx));
}

test "usesSplit: the resident kernel runs up to S positions and the split path past S" {
    try testing.expect(!usesSplit(1));
    try testing.expect(!usesSplit(S - 1));
    try testing.expect(!usesSplit(S));
    try testing.expect(usesSplit(S + 1));
    try testing.expect(usesSplit(limits.max_model_ctx));
}

test "headDimFits: common head widths fit, a head past the bound or off the granule does not" {
    for ([_]u32{ 64, 80, 96, 112, 128, 256 }) |hd| {
        try testing.expect(headDimFits(hd));
    }

    try testing.expect(headDimFits(limits.attn_head_dim_granule));
    try testing.expect(headDimFits(limits.attn_max_head_dim));
    try testing.expect(!headDimFits(0));
    try testing.expect(!headDimFits(limits.attn_max_head_dim + limits.attn_head_dim_granule));
    for ([_]u32{ 1, 4, 60, 100, 250 }) |hd| {
        try testing.expect(!headDimFits(hd));
    }
}

test "queryGroups: every head of a group lands in one workgroup slot, every slot holds a head" {
    const Q = limits.attn_split_queries;
    try testing.expectEqual(@as(u32, 1), queryGroups(1));
    try testing.expectEqual(@as(u32, 1), queryGroups(Q));
    try testing.expectEqual(@as(u32, 2), queryGroups(Q + 1));
    var group: u32 = 1;
    while (group <= 64) : (group += 1) {
        const n_kv: u32 = 3;
        const qgroups = queryGroups(group);
        var seen = [_]bool{false} ** (64 * 3);
        var x: u32 = 0;
        while (x < n_kv * qgroups) : (x += 1) {
            // The shader maps workgroup x to the heads g * G + c * Q + qi.
            const g = x / qgroups;
            const c = x % qgroups;
            const count = @min(Q, group - c * Q);
            try testing.expect(count >= 1);
            var qi: u32 = 0;
            while (qi < count) : (qi += 1) {
                const h = g * group + c * Q + qi;
                try testing.expect(h < n_kv * group);
                try testing.expect(!seen[h]);
                seen[h] = true;
            }
        }

        for (seen[0 .. n_kv * group]) |s| {
            try testing.expect(s);
        }
    }
}

test "splitCount: the splits of every row up to four splits cover it, and none is empty" {
    var t_len: u32 = 1;
    while (t_len <= 4 * S + 1) : (t_len += 1) {
        const count = splitCount(t_len);
        // The split of each position is monotone in the position, and the
        // last position lands in the last split, so every position lands in a
        // split below the count.
        try testing.expectEqual(count - 1, (t_len - 1) / S);
        // Every split below the count starts inside the row.
        var j: u32 = 0;
        while (j < count) : (j += 1) {
            try testing.expect(j * S < t_len);
        }

        // A longer row never needs fewer splits.
        if (t_len >= 2) {
            try testing.expect(splitCount(t_len - 1) <= count);
        }

        try testing.expectEqual(count >= 2, usesSplit(t_len));
    }
}

test "offset: every coordinate of a small box gets its own element below the scratch" {
    var rows: u32 = 1;
    while (rows <= 2) : (rows += 1) {
        var n_heads: u32 = 1;
        while (n_heads <= 3) : (n_heads += 1) {
            var max_splits: u32 = 1;
            while (max_splits <= 3) : (max_splits += 1) {
                var head_dim: u32 = 1;
                while (head_dim <= 2) : (head_dim += 1) {
                    const scratch: Scratch = .{
                        .rows = rows,
                        .n_heads = n_heads,
                        .head_dim = head_dim,
                        .max_splits = max_splits,
                    };
                    const record = scratch.recordElems();
                    // The largest box holds 2 * 3 * 3 * 4 elements.
                    var seen = [_]bool{false} ** 72;
                    var count: u64 = 0;
                    for (0..rows) |r| {
                        for (0..n_heads) |h| {
                            for (0..max_splits) |j| {
                                for (0..record) |e| {
                                    const at = scratch.offset(
                                        @intCast(r),
                                        @intCast(h),
                                        @intCast(j),
                                        @intCast(e),
                                    );
                                    try testing.expect(at < scratch.totalElems());
                                    try testing.expect(!seen[@intCast(at)]);
                                    seen[@intCast(at)] = true;
                                    count += 1;
                                }
                            }
                        }
                    }

                    // The box and the scratch have one size, so the offset
                    // covers every element, and the last coordinate lands on
                    // the last element.
                    try testing.expectEqual(scratch.totalElems(), count);
                    try testing.expectEqual(
                        scratch.totalElems() - 1,
                        scratch.offset(rows - 1, n_heads - 1, max_splits - 1, record - 1),
                    );
                    try testing.expectEqual(scratch.totalElems() * 4, scratch.totalBytes());
                }
            }
        }
    }
}

test "anchors: the scratch of the four registry shapes at S = 256 and 4 rows" {
    try testing.expectEqual(@as(u32, 256), S);
    try testing.expectEqual(@as(u32, 4), limits.attn_split_rows);
    // Qwen3.5-2B: 8 query heads of 256 at the 262144 ceiling.
    const qwen_2b = Scratch.init(8, 256, 262_144);
    try testing.expectEqual(@as(u32, 1024), qwen_2b.max_splits);
    try testing.expectEqual(@as(u64, 8_454_144), qwen_2b.totalElems());
    try testing.expectEqual(@as(u64, 33_816_576), qwen_2b.totalBytes());
    // Qwen3.5-9B: 16 query heads of 256 at the 262144 ceiling.
    const qwen_9b = Scratch.init(16, 256, 262_144);
    try testing.expectEqual(@as(u32, 1024), qwen_9b.max_splits);
    try testing.expectEqual(@as(u64, 16_908_288), qwen_9b.totalElems());
    try testing.expectEqual(@as(u64, 67_633_152), qwen_9b.totalBytes());
    // Llama-3.1-8B: 32 query heads of 128 at the 131072 ceiling.
    const llama_8b = Scratch.init(32, 128, 131_072);
    try testing.expectEqual(@as(u32, 512), llama_8b.max_splits);
    try testing.expectEqual(@as(u64, 8_519_680), llama_8b.totalElems());
    try testing.expectEqual(@as(u64, 34_078_720), llama_8b.totalBytes());
    // SmolLM2-135M: 9 query heads of 64 at the 8192 ceiling.
    const smollm2 = Scratch.init(9, 64, 8192);
    try testing.expectEqual(@as(u32, 32), smollm2.max_splits);
    try testing.expectEqual(@as(u64, 76_032), smollm2.totalElems());
    try testing.expectEqual(@as(u64, 304_128), smollm2.totalBytes());
    // The 128 MiB storage range and the 2 GiB allocation accept all four.
    for ([_]Scratch{ qwen_2b, qwen_9b, llama_8b, smollm2 }) |scratch| {
        try testing.expectEqual(limits.BufferVerdict.fits, fits(scratch, spec_min_limits));
    }
}

test "fits: the 31-bit index and the storage range pass at their bounds, fail one step past" {
    // 131072 heads of 254 elements at 64 splits hold 64 * 256 * 131072 = 2^31
    // elements in one row.
    const at_bound: Scratch = .{ .rows = 1, .n_heads = 131_072, .head_dim = 254, .max_splits = 64 };
    try testing.expectEqual(limits.max_kv_index_elems, at_bound.totalElems());
    try testing.expectEqual(limits.BufferVerdict.fits, fits(at_bound, wide_limits));
    var past = at_bound;
    past.n_heads += 1;
    try testing.expectEqual(limits.BufferVerdict.index_past_31_bits, fits(past, wide_limits));
    // The storage range, at the bytes of the 9B scratch and one element short.
    const qwen_9b = Scratch.init(16, 256, 262_144);
    var lim = wide_limits;
    lim.storage_range = qwen_9b.totalBytes();
    try testing.expectEqual(limits.BufferVerdict.fits, fits(qwen_9b, lim));
    lim.storage_range -= elem_bytes;
    try testing.expectEqual(limits.BufferVerdict.past_storage_range, fits(qwen_9b, lim));
}

test "fits: every registry model at its declared context fits the spec-minimum limits" {
    // The shapes come from a pinned table and not from the model files. So the
    // test runs where no file is present, and it opens no model file. Each
    // registry entry must have exactly one row, so a new entry fails the test
    // until it gets a row.
    const registry = @import("registry.zig");
    const Row = struct { name: []const u8, n_heads: u32, head_dim: u32, ctx: u32 };
    const pinned = [_]Row{
        .{ .name = "SmolLM2-135M-Instruct-Q8_0", .n_heads = 9, .head_dim = 64, .ctx = 8192 },
        .{ .name = "SmolLM2-135M-Instruct-Q4_0", .n_heads = 9, .head_dim = 64, .ctx = 8192 },
        .{
            .name = "Meta-Llama-3.1-8B-Instruct-Q8_0",
            .n_heads = 32,
            .head_dim = 128,
            .ctx = 131_072,
        },
        .{
            .name = "Meta-Llama-3.1-8B-Instruct-Q4_0",
            .n_heads = 32,
            .head_dim = 128,
            .ctx = 131_072,
        },
        .{ .name = "Qwen3.5-2B-Q8_0", .n_heads = 8, .head_dim = 256, .ctx = 262_144 },
        .{ .name = "Qwen3.5-9B-Q8_0", .n_heads = 16, .head_dim = 256, .ctx = 262_144 },
    };
    try testing.expectEqual(registry.all().len, pinned.len);
    for (registry.all()) |m| {
        var matches: u32 = 0;
        for (pinned) |row| {
            if (!std.mem.eql(u8, row.name, m.pub_name)) continue;

            matches += 1;
            const scratch = Scratch.init(row.n_heads, row.head_dim, row.ctx);
            try testing.expectEqual(limits.BufferVerdict.fits, fits(scratch, spec_min_limits));
        }

        try testing.expectEqual(@as(u32, 1), matches);
    }
}

test "planDecode: a row of one split runs resident, and a longer row runs one group of one row" {
    for ([_]u32{ 1, S - 1, S }) |t_len| {
        const plan = planDecode(t_len);
        try testing.expectEqual(@as(u32, 1), plan.resident_count);
        try testing.expectEqual(@as(u32, 0), plan.split_count);
        try testing.expectEqual(@as(u32, 0), plan.group_count);
        try testing.expectEqual(
            PlanRow{ .row = 0, .slot = 0, .t_len = t_len },
            plan.residentRows()[0],
        );
    }

    for ([_]u32{ S + 1, 32_704, limits.max_model_ctx }) |t_len| {
        const plan = planDecode(t_len);
        try testing.expectEqual(@as(u32, 0), plan.resident_count);
        try testing.expectEqual(@as(u32, 1), plan.group_count);
        try testing.expectEqual(
            PlanGroup{ .first = 0, .len = 1, .stride = splitCount(t_len) },
            plan.recordGroups()[0],
        );
        try testing.expectEqual(
            PlanRow{ .row = 0, .slot = 0, .t_len = t_len },
            plan.rowsOf(plan.recordGroups()[0])[0],
        );
    }
}

fn checkPlanRows(_: void, c: prop.Trial) anyerror!void {
    var prng: std.Random.DefaultPrng = .init(c.seed);
    const random = prng.random();
    var sites: [max_plan_rows]RowSite = undefined;
    const trial = c.index;
    const n = random.intRangeAtMost(u32, 1, max_plan_rows);
    // Half the trials mix depths around S, and the rest put every row
    // past it, so groups of one row and groups of many rows both occur.
    const low: u32 = if (trial % 2 == 0) 1 else S + 1;
    for (sites[0..n]) |*site| {
        site.* = .{
            .slot = random.intRangeLessThan(u32, 0, 8),
            .t_len = random.intRangeAtMost(u32, low, 3 * S),
        };
    }

    const max_splits = random.intRangeAtMost(u32, splitCount(3 * S), 12);
    const plan = planRows(sites[0..n], max_splits);
    try testing.expectEqual(limits.attn_split_rows * max_splits, plan.budget);
    try testing.expectEqual(n, plan.resident_count + plan.split_count);
    // Each list keeps the step order, and each entry copies its site.
    var seen = [_]bool{false} ** max_plan_rows;
    for ([_][]const PlanRow{ plan.residentRows(), plan.splitRows() }) |list| {
        for (list, 0..) |r, k| {
            if (k > 0) {
                try testing.expect(list[k - 1].row < r.row);
            }

            try testing.expect(!seen[r.row]);
            seen[r.row] = true;
            try testing.expectEqual(sites[r.row].slot, r.slot);
            try testing.expectEqual(sites[r.row].t_len, r.t_len);
        }
    }

    for (seen[0..n]) |landed| {
        try testing.expect(landed);
    }

    // The groups cover the split rows in order. Each group fits the
    // budget at its stride, and the next row breaks the budget.
    var next: u32 = 0;
    for (plan.recordGroups()) |g| {
        try testing.expectEqual(next, g.first);
        var widest: u32 = 0;
        for (plan.rowsOf(g)) |r| {
            widest = @max(widest, splitCount(r.t_len));
        }

        try testing.expectEqual(widest, g.stride);
        try testing.expect(g.len * g.stride <= plan.budget);
        if (g.first + g.len < plan.split_count) {
            const after = splitCount(plan.splitRows()[g.first + g.len].t_len);
            try testing.expect((g.len + 1) * @max(widest, after) > plan.budget);
        }

        next += g.len;
    }

    try testing.expectEqual(plan.split_count, next);
}

test "planRows: every row lands once, in step order, and split rows fill maximal groups" {
    try prop.forAll(
        .{ .site = 0x0031_a7a5, .cases = 400, .max_size = 399 },
        {},
        prop.Trial,
        prop.genTrial(void),
        checkPlanRows,
    );
}

test "planRows: the four registry anchors group a prefill chunk of 128 rows" {
    const Anchor = struct { start_pos: u32, ctx: u32, groups: u32, len: u32, stride: u32 };
    const anchors = [_]Anchor{
        // Qwen3.5-2B at depth 8064: one group of 128 rows.
        .{ .start_pos = 7936, .ctx = 262_144, .groups = 1, .len = 128, .stride = 32 },
        // Qwen3.5-2B at depth 32704: four groups of 32 rows.
        .{ .start_pos = 32_576, .ctx = 262_144, .groups = 4, .len = 32, .stride = 128 },
        // Llama-3.1-8B at depth 131072: 32 groups of 4 rows.
        .{ .start_pos = 130_944, .ctx = 131_072, .groups = 32, .len = 4, .stride = 512 },
        // SmolLM2-135M at depth 8192: 32 groups of 4 rows.
        .{ .start_pos = 8064, .ctx = 8192, .groups = 32, .len = 4, .stride = 32 },
    };
    for (anchors) |a| {
        const plan = planPrefill(0, a.start_pos, 128, splitCount(a.ctx));
        try testing.expectEqual(@as(u32, 0), plan.resident_count);
        try testing.expectEqual(a.groups, plan.group_count);
        for (plan.recordGroups()) |g| {
            try testing.expectEqual(a.len, g.len);
            try testing.expectEqual(a.stride, g.stride);
        }
    }
}

test "planPrefill: the resident rows of a chunk are the prefix that stays within S" {
    // A chunk that ends at S, one that straddles S, one that starts at S, and
    // one past it. Row i is resident exactly when i < S - start_pos.
    const Case = struct { start_pos: u32, b: u32, resident: u32 };
    const cases = [_]Case{
        .{ .start_pos = 0, .b = S, .resident = S },
        .{ .start_pos = S - 3, .b = 7, .resident = 3 },
        .{ .start_pos = S - 1, .b = 1, .resident = 1 },
        .{ .start_pos = S, .b = 5, .resident = 0 },
        .{ .start_pos = 4 * S + 1, .b = 3, .resident = 0 },
    };
    for (cases) |c| {
        try testing.expectEqual(c.resident, prefillResident(c.start_pos, c.b));
        const plan = planPrefill(2, c.start_pos, c.b, splitCount(8 * S));
        try testing.expectEqual(c.resident, plan.resident_count);
        try testing.expectEqual(c.b - c.resident, plan.split_count);
        for (plan.residentRows(), 0..) |r, i| {
            try testing.expectEqual(@as(u32, @intCast(i)), r.row);
            try testing.expectEqual(@as(u32, 2), r.slot);
            try testing.expect(c.start_pos + r.row + 1 <= S);
        }

        for (plan.splitRows()) |r| {
            try testing.expect(r.row >= c.resident);
        }

        // The resident run of the chunk ends at S or before it.
        try testing.expect(c.start_pos + c.resident <= S or c.resident == 0);
    }
}

/// The largest count over the rows `[lo, hi)` times the row count: the cost
/// of one group.
fn spanCost(counts: []const u32, lo: u32, hi: u32) u64 {
    var widest: u64 = 0;
    for (counts[lo..hi]) |c| {
        widest = @max(widest, c);
    }

    return @as(u64, hi - lo) * widest;
}

/// Check the groups of a chunk of rows at depth `s` against the named sizes
/// and stride.
fn expectAnchor(s: u32, ctx: u32, sizes: []const u32, stride: u32) !void {
    var counts: [128]u32 = undefined;
    for (&counts, 0..) |*c, i| {
        c.* = splitCount(s + @as(u32, @intCast(i)) + 1);
    }

    const budget = 4 * splitCount(ctx);
    var out: [128]BudgetGroup = undefined;
    const groups = budgetGroups(budget, &counts, &out);
    try testing.expectEqual(sizes.len, groups.len);
    for (groups, sizes) |g, want| {
        try testing.expectEqual(want, g.hi - g.lo);
    }

    // The stride of the chunk is its largest count, and every group has it.
    var max: u32 = 0;
    for (counts) |c| {
        max = @max(max, c);
    }

    try testing.expectEqual(stride, max);
    for (groups) |g| {
        var group_max: u32 = 0;
        for (counts[g.lo..g.hi]) |c| {
            group_max = @max(group_max, c);
        }

        try testing.expectEqual(stride, group_max);
    }
}

test "budgetGroups: the anchors of the four registry shapes" {
    try testing.expectEqual(@as(u32, 4096), 4 * splitCount(262_144));
    try testing.expectEqual(@as(u32, 2048), 4 * splitCount(131_072));
    try testing.expectEqual(@as(u32, 128), 4 * splitCount(8192));
    // Qwen3.5-2B at depth 8064: one group of 128 rows.
    try expectAnchor(7936, 262_144, &.{128}, 32);
    // Qwen3.5-2B at depth 32704: four groups of 32 rows.
    try expectAnchor(32_576, 262_144, &.{ 32, 32, 32, 32 }, 128);
    // Llama-3.1-8B at depth 131072: 32 groups of 4 rows.
    try expectAnchor(130_944, 131_072, &([_]u32{4} ** 32), 512);
    // SmolLM2-135M at depth 8192: 32 groups of 4 rows.
    try expectAnchor(8064, 8192, &([_]u32{4} ** 32), 32);
}

fn checkBudgetGroups(_: void, seed: u64) anyerror!void {
    var prng: std.Random.DefaultPrng = .init(seed);
    const random = prng.random();
    var counts: [128]u32 = undefined;
    var out: [128]BudgetGroup = undefined;
    const n = random.intRangeAtMost(u32, 1, 128);
    var max: u32 = 0;
    for (counts[0..n]) |*c| {
        c.* = random.intRangeAtMost(u32, 1, 64);
        max = @max(max, c.*);
    }

    // Every single row fits the budget, so the precondition holds.
    const budget = random.intRangeAtMost(u32, max, 4 * max);
    const groups = budgetGroups(budget, counts[0..n], &out);
    var next: u32 = 0;
    for (groups) |g| {
        try testing.expectEqual(next, g.lo);
        try testing.expect(g.lo < g.hi);
        try testing.expect(spanCost(counts[0..n], g.lo, g.hi) <= budget);
        // The group is maximal, so the next row exceeds the budget.
        if (g.hi < n) {
            try testing.expect(spanCost(counts[0..n], g.lo, g.hi + 1) > budget);
        }

        next = g.hi;
    }

    try testing.expectEqual(n, next);
}

test "budgetGroups: random counts partition the rows in order within the budget" {
    try prop.forAll(
        .{ .site = 0x0031_a7a6, .cases = 500 },
        {},
        u64,
        prop.genSeed(void),
        checkBudgetGroups,
    );
}

test "budgetGroups: the minimum budget gives one row per group when counts are equal" {
    var out: [64]BudgetGroup = undefined;
    var counts: [64]u32 = undefined;
    for ([_]u32{ 1, 5, 37 }) |c| {
        for ([_]u32{ 1, 13, 64 }) |len| {
            @memset(counts[0..len], c);
            const groups = budgetGroups(c, counts[0..len], &out);
            try testing.expectEqual(@as(usize, len), groups.len);
            for (groups, 0..) |g, k| {
                try testing.expectEqual(@as(u32, @intCast(k)), g.lo);
                try testing.expectEqual(@as(u32, 1), g.hi - g.lo);
            }
        }
    }
}
