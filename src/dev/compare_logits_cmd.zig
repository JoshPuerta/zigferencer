//! `zigferencer-dev compare-logits REF CAND`: compare two logits files of `bench-model
//! --logits-out` entry by entry. REF comes from the reference run, and CAND from the candidate run.
//! It prints how often the greedy pick agrees, the largest logit distance, and the KL divergence of
//! the softmax of each entry.
//!
//! Histories. An entry is comparable only when both runs fed the same ids before it. So the command
//! refuses files whose prompt hashes or fed ids differ. A run with other fed ids reruns with
//! `--force-tokens REF`.
//!
//! Bound. With every |dlogit| at most T, the top-1 and the top-2 of an entry move by at most T
//! each. So a flip of the greedy pick needs a reference margin of at most 2T. A flip at a larger
//! margin breaks that rule and fails on its own. A bound of 0 demands equal logits: every |dlogit|
//! is 0, so no greedy pick can flip. `tools/gate.py` reads the `--json` line.
//!
//! Memory. The command holds the logits of two entries at a time, so a file of any entry count
//! streams through.

const std = @import("std");
const zigferencer = @import("zigferencer");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const context = zigferencer.cli.context;
const args = zigferencer.cli.args;
const sampler = zigferencer.sampler;
const logits_file = @import("bench_model/bench_model_logits.zig");

// -- constants --------------------------------------------------------------

/// The buffer of each file reader.
const io_buffer_bytes = 64 * 1024;

// -- types ------------------------------------------------------------------

/// The flags of `compare-logits`. The two files come first and take no flag name.
const Flags = struct {
    max_dlogit: ?f32 = null,
    json: bool = false,

    pub const help: args.Help = .{
        .usage = "zigferencer-dev compare-logits REF CAND [flags]",
        .about =
        \\Compare two logits files of bench-model --logits-out entry by entry: REF of the
        \\reference run and CAND of the candidate run. Print the entry count, top1_agree
        \\(entries where the greedy picks agree), first_top1_diff (the first entry where
        \\they differ, or -1) with the top-1 minus top-2 margin of both files, max_dlogit,
        \\kl_max and kl_mean of KL(REF || CAND), and the smallest margin of REF. The
        \\files must share the vocab, the prompt length and hash, the first position,
        \\the entry count, and the fed ids.
        \\
        ,
        .footer =
        \\examples:
        \\  zigferencer-dev compare-logits ref.bin cand.bin
        \\  zigferencer-dev compare-logits ref.bin cand.bin --max-dlogit 0.05 --json
        \\
        ,
    };
    pub const docs = .{
        .max_dlogit = args.Doc{
            .text = "exit 1 when any |dlogit| is more than T, or a greedy pick\n" ++
                "flips at a REF margin over 2T. T = 0 demands equal logits",
            .value = "T",
        },
        .json = args.Doc{ .text = "print one JSON line (tools/gate.py)" },
    };
};

/// The greedy picks and the top-1 minus top-2 margins of one entry pair, and their distance.
pub const EntryStats = struct {
    top1_ref: u32,
    top1_cand: u32,
    margin_ref: f64,
    margin_cand: f64,
    max_dlogit: f64,
    kl: f64,
};

/// The sums of a comparison over every entry so far.
pub const Summary = struct {
    entries: u64 = 0,
    top1_agree: u64 = 0,
    first_top1_diff: ?u64 = null,
    ref_margin_at_diff: f64 = 0,
    cand_margin_at_diff: f64 = 0,
    max_dlogit: f64 = 0,
    max_dlogit_entry: u64 = 0,
    kl_max: f64 = 0,
    kl_sum: f64 = 0,
    min_ref_margin: f64 = std.math.inf(f64),
    min_ref_margin_entry: u64 = 0,
    /// The largest reference margin of an entry whose greedy pick flips, and its entry.
    flip_margin_max: f64 = 0,
    flip_margin_entry: ?u64 = null,

    pub fn add(self: *Summary, s: EntryStats) void {
        assert(s.max_dlogit >= 0);
        assert(s.margin_ref >= 0);

        const entry = self.entries;
        self.entries += 1;
        if (s.top1_ref == s.top1_cand) {
            self.top1_agree += 1;
        } else {
            if (self.first_top1_diff == null) {
                self.first_top1_diff = entry;
                self.ref_margin_at_diff = s.margin_ref;
                self.cand_margin_at_diff = s.margin_cand;
            }
            if (self.flip_margin_entry == null or s.margin_ref > self.flip_margin_max) {
                self.flip_margin_max = s.margin_ref;
                self.flip_margin_entry = entry;
            }
        }

        if (s.max_dlogit > self.max_dlogit) {
            self.max_dlogit = s.max_dlogit;
            self.max_dlogit_entry = entry;
        }
        if (s.margin_ref < self.min_ref_margin) {
            self.min_ref_margin = s.margin_ref;
            self.min_ref_margin_entry = entry;
        }

        self.kl_max = @max(self.kl_max, s.kl);
        self.kl_sum += s.kl;
        assert(self.top1_agree <= self.entries);
    }

    pub fn klMean(self: *const Summary) f64 {
        assert(self.entries > 0);

        return self.kl_sum / @as(f64, @floatFromInt(self.entries));
    }
};

/// The result of a comparison against the bound `--max-dlogit T`.
pub const Verdict = enum { within, over_bound, flip_over_margin };

// -- api --------------------------------------------------------------------

pub fn run(ctx: context.Context) !void {
    const io = ctx.io;

    // The two files lead, so the flag parser sees only the flags after them.
    var lead: usize = 0;
    while (lead < 2 and lead < ctx.args.len and !std.mem.startsWith(u8, ctx.args[lead], "-")) {
        lead += 1;
    }

    var rest = ctx;
    rest.args = ctx.args[lead..];
    const flags = try args.parseInto(Flags, rest) orelse return;
    if (lead != 2) return args.fail(io, "compare-logits needs two files: REF CAND", .{});

    if (flags.max_dlogit) |t| {
        if (!std.math.isFinite(t) or t < 0) {
            return args.fail(io, "--max-dlogit must be a finite value of 0 or more", .{});
        }
    }

    const summary = try compareFiles(ctx, ctx.args[0], ctx.args[1]);
    try printSummary(ctx, summary, flags.json);

    const t = flags.max_dlogit orelse return;
    switch (verdict(summary, t)) {
        .within => {},
        .over_bound => return args.fail(
            io,
            "max_dlogit {d} at entry {d} is more than the bound {d}",
            .{ summary.max_dlogit, summary.max_dlogit_entry, t },
        ),
        .flip_over_margin => return args.fail(
            io,
            "the greedy pick flips at entry {d} with a REF margin {d} over 2 x {d}",
            .{ summary.flip_margin_entry.?, summary.flip_margin_max, t },
        ),
    }
}

/// The stats of one entry pair. Both entries hold only finite values.
pub fn entryStats(ref: []const f32, cand: []const f32) EntryStats {
    assert(ref.len == cand.len);
    assert(ref.len >= 2);

    // The distances and the margins take f64, so no f32 rounding moves them against the 2T rule.
    var max_d: f64 = 0;
    for (ref, cand) |r, o| {
        max_d = @max(max_d, @abs(@as(f64, r) - @as(f64, o)));
    }

    const top_ref = sampler.greedy(ref);
    const top_cand = sampler.greedy(cand);
    const out: EntryStats = .{
        .top1_ref = top_ref,
        .top1_cand = top_cand,
        .margin_ref = margin(ref, top_ref),
        .margin_cand = margin(cand, top_cand),
        .max_dlogit = max_d,
        .kl = klDivergence(ref, cand),
    };
    assert(out.margin_ref >= 0);
    assert(out.margin_cand >= 0);
    return out;
}

/// The result against the bound `t`. A flip is legal only at a reference margin of at most 2t.
/// With `t` = 0, only equal logits pass.
pub fn verdict(s: Summary, t: f32) Verdict {
    assert(t >= 0);
    assert(s.entries > 0);

    const bound: f64 = t;
    if (s.max_dlogit > bound) return .over_bound;
    if (s.flip_margin_entry != null and s.flip_margin_max > 2 * bound) return .flip_over_margin;
    return .within;
}

// -- helpers ----------------------------------------------------------------

/// Stream both files entry by entry into one summary, after the checks that make the entries
/// comparable.
fn compareFiles(ctx: context.Context, ref_path: []const u8, cand_path: []const u8) !Summary {
    const io = ctx.io;

    // Open both files and check that their entries line up.
    var ref = try openFile(ctx, ref_path);
    defer ref.close(io);
    var cand = try openFile(ctx, cand_path);
    defer cand.close(io);

    const a = ref.header;
    const b = cand.header;
    if (a.vocab != b.vocab) return args.fail(io, "vocab {d} and {d} differ", .{ a.vocab, b.vocab });
    if (a.prompt_len != b.prompt_len) {
        return args.fail(io, "prompt length {d} and {d} differ", .{ a.prompt_len, b.prompt_len });
    }
    if (a.first_pos != b.first_pos) {
        return args.fail(io, "first position {d} and {d} differ", .{ a.first_pos, b.first_pos });
    }
    if (a.prompt_hash != b.prompt_hash) {
        return args.fail(io, "prompt hash 0x{x:0>16} and 0x{x:0>16} differ", .{
            a.prompt_hash, b.prompt_hash,
        });
    }
    if (ref.entries != cand.entries) {
        return args.fail(io, "entry count {d} and {d} differ", .{ ref.entries, cand.entries });
    }
    if (a.vocab < 2) return args.fail(io, "a vocab of {d} has no top-2 margin", .{a.vocab});

    // Compare each entry pair. The entry count bounds the loop.
    const logits_ref = try ctx.arena.alloc(f32, a.vocab);
    const logits_cand = try ctx.arena.alloc(f32, a.vocab);
    var summary: Summary = .{};
    var i: u64 = 0;
    while (i < ref.entries) : (i += 1) {
        const id_ref = try readEntry(ctx, &ref, ref_path, i, logits_ref);
        const id_cand = try readEntry(ctx, &cand, cand_path, i, logits_cand);
        summary.add(entryStats(logits_ref, logits_cand));

        // The entries after a different fed id see another history.
        if (id_ref != id_cand) {
            return args.fail(
                io,
                "entry {d}: REF fed {d} and CAND fed {d}. Rerun CAND with --force-tokens REF",
                .{ i, id_ref, id_cand },
            );
        }
    }

    assert(summary.entries == ref.entries);
    return summary;
}

fn openFile(ctx: context.Context, path: []const u8) !logits_file.Reader {
    const buffer = try ctx.arena.alloc(u8, io_buffer_bytes);
    return logits_file.Reader.open(ctx.io, path, buffer) catch |err| {
        return args.fail(ctx.io, "{s}: {t}", .{ path, err });
    };
}

fn readEntry(
    ctx: context.Context,
    reader: *logits_file.Reader,
    path: []const u8,
    index: u64,
    logits: []f32,
) error{UsageError}!u32 {
    const id = reader.readEntry(index, logits) catch |err| {
        return args.fail(ctx.io, "{s}: entry {d}: {t}", .{ path, index, err });
    };

    for (logits, 0..) |v, k| {
        if (!std.math.isFinite(v)) {
            return args.fail(ctx.io, "{s}: entry {d} holds {d} at id {d}", .{ path, index, v, k });
        }
    }

    return id;
}

/// The top-1 value minus the largest other value of `logits`.
fn margin(logits: []const f32, top: u32) f64 {
    assert(top < logits.len);

    var second = -std.math.inf(f32);
    for (logits, 0..) |v, k| {
        if (k != top) {
            second = @max(second, v);
        }
    }

    const out = @as(f64, logits[top]) - @as(f64, second);
    assert(out >= 0);
    return out;
}

/// KL(p || q) of the softmax of two entries, in f64. Each log-softmax subtracts the maximum of the
/// logits first.
fn klDivergence(p_logits: []const f32, q_logits: []const f32) f64 {
    assert(p_logits.len == q_logits.len);

    const p_shift = logSumExp(p_logits);
    const q_shift = logSumExp(q_logits);
    var kl: f64 = 0;
    for (p_logits, q_logits) |p, q| {
        const lp = @as(f64, p) - p_shift;
        const lq = @as(f64, q) - q_shift;
        kl += @exp(lp) * (lp - lq);
    }

    return kl;
}

fn logSumExp(logits: []const f32) f64 {
    var m: f64 = -std.math.inf(f64);
    for (logits) |v| {
        m = @max(m, @as(f64, v));
    }

    var sum: f64 = 0;
    for (logits) |v| {
        sum += @exp(@as(f64, v) - m);
    }

    assert(sum >= 1);
    return m + @log(sum);
}

fn printSummary(ctx: context.Context, s: Summary, json: bool) !void {
    const w = &ctx.stdout.interface;
    const first: i64 = if (s.first_top1_diff) |r| @intCast(r) else -1;

    if (json) {
        try w.print(
            "{{\"entries\":{d},\"top1_agree\":{d},\"first_top1_diff\":{d}," ++
                "\"ref_margin_at_diff\":{d},\"cand_margin_at_diff\":{d}," ++
                "\"max_dlogit\":{d},\"max_dlogit_entry\":{d},\"kl_max\":{e},\"kl_mean\":{e}," ++
                "\"min_ref_margin\":{d},\"min_ref_margin_entry\":{d}}}\n",
            .{
                s.entries,            s.top1_agree,           first,
                s.ref_margin_at_diff, s.cand_margin_at_diff,  s.max_dlogit,
                s.max_dlogit_entry,   s.kl_max,               s.klMean(),
                s.min_ref_margin,     s.min_ref_margin_entry,
            },
        );
    } else {
        try w.print("entries          {d}\n", .{s.entries});
        try w.print("top1_agree       {d}\n", .{s.top1_agree});
        if (s.first_top1_diff) |r| {
            try w.print("first_top1_diff  {d} (margin REF {d}, CAND {d})\n", .{
                r, s.ref_margin_at_diff, s.cand_margin_at_diff,
            });
        } else {
            try w.print("first_top1_diff  -1\n", .{});
        }
        try w.print("max_dlogit       {d} (entry {d})\n", .{ s.max_dlogit, s.max_dlogit_entry });
        try w.print("kl_max           {e}\n", .{s.kl_max});
        try w.print("kl_mean          {e}\n", .{s.klMean()});
        try w.print("min_ref_margin   {d} (entry {d})\n", .{
            s.min_ref_margin, s.min_ref_margin_entry,
        });
    }

    try w.flush();
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;

test "entryStats: equal logits give zero distance, zero KL, and the margin of the logits" {
    const logits = [_]f32{ 1, 4, 2, -3 };
    const s = entryStats(&logits, &logits);
    try testing.expectEqual(@as(u32, 1), s.top1_ref);
    try testing.expectEqual(@as(u32, 1), s.top1_cand);
    try testing.expectEqual(@as(f64, 2), s.margin_ref);
    try testing.expectEqual(@as(f64, 0), s.max_dlogit);
    try testing.expectEqual(@as(f64, 0), s.kl);
}

test "entryStats: KL of two entries matches the closed form" {
    // Two ids: p = softmax(0, 0) = (1/2, 1/2), q = softmax(0, ln 3) = (1/4, 3/4).
    // KL(p || q) = 1/2 ln 2 + 1/2 ln(2/3) = 1/2 ln(4/3).
    const p = [_]f32{ 0, 0 };
    const q = [_]f32{ 0, @floatCast(@log(@as(f64, 3))) };
    const s = entryStats(&p, &q);
    try testing.expectApproxEqAbs(0.5 * @log(@as(f64, 4.0 / 3.0)), s.kl, 1e-7);

    // A shift of every logit leaves the softmax, so the KL is zero up to rounding.
    const r = [_]f32{ 1, 2, 3 };
    const r_shift = [_]f32{ 6, 7, 8 };
    const s_shift = entryStats(&r, &r_shift);
    try testing.expectApproxEqAbs(@as(f64, 0), s_shift.kl, 1e-12);
    try testing.expectEqual(@as(f64, 5), s_shift.max_dlogit);
}

test "Summary: counts the flips, the first flip with both margins, and the largest distance" {
    var sum: Summary = .{};
    const pairs = [_][2][3]f32{
        .{ .{ 0, 1, 0 }, .{ 0, 1, 0 } }, // agree, margin 1
        .{ .{ 0, 1, 0.9 }, .{ 0, 0.9, 1 } }, // flip: REF margin 0.1, CAND margin 0.1
        .{ .{ 3, 0, 0 }, .{ 0, 0, 3 } }, // flip: REF margin 3, distance 3
        .{ .{ 0, 0, 2 }, .{ 0, 0, 2.5 } }, // agree, distance 0.5
    };
    for (pairs) |pair| {
        sum.add(entryStats(&pair[0], &pair[1]));
    }

    try testing.expectEqual(@as(u64, 4), sum.entries);
    try testing.expectEqual(@as(u64, 2), sum.top1_agree);
    try testing.expectEqual(@as(?u64, 1), sum.first_top1_diff);
    try testing.expectApproxEqAbs(@as(f64, 0.1), sum.ref_margin_at_diff, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.1), sum.cand_margin_at_diff, 1e-6);
    try testing.expectEqual(@as(f64, 3), sum.max_dlogit);
    try testing.expectEqual(@as(u64, 2), sum.max_dlogit_entry);
    try testing.expectEqual(@as(?u64, 2), sum.flip_margin_entry);
    try testing.expectEqual(@as(f64, 3), sum.flip_margin_max);
    try testing.expectApproxEqAbs(@as(f64, 0.1), sum.min_ref_margin, 1e-6);
    try testing.expectEqual(@as(u64, 1), sum.min_ref_margin_entry);
}

test "verdict: the bound on the distance and the flip margin of 2T" {
    // A flip at REF margin 0.1 with a distance of 0.1: legal under T = 0.11 (0.1 <= 2T).
    var near: Summary = .{};
    near.add(entryStats(&.{ 0, 1, 0.9 }, &.{ 0, 0.9, 1 }));
    try testing.expectEqual(Verdict.within, verdict(near, 0.11));
    try testing.expectEqual(Verdict.over_bound, verdict(near, 0.05));

    // A flip at REF margin 1 under T = 0.4 with every distance at most T breaks the rule. A
    // summary cannot reach it from real entries, so the test sets the fields.
    var broken: Summary = .{ .entries = 1, .max_dlogit = 0.4 };
    broken.flip_margin_entry = 0;
    broken.flip_margin_max = 1;
    try testing.expectEqual(Verdict.flip_over_margin, verdict(broken, 0.4));
    try testing.expectEqual(Verdict.within, verdict(broken, 0.5));
}

test "verdict: a bound of 0 passes only equal logits" {
    const logits = [_]f32{ 1, 4, 2, -3 };
    var equal: Summary = .{};
    equal.add(entryStats(&logits, &logits));
    equal.add(entryStats(&logits, &logits));
    try testing.expectEqual(Verdict.within, verdict(equal, 0));

    // One logit one f32 step away, with no flip of the greedy pick, fails the bound of 0.
    const moved = [_]f32{ 1, 4, std.math.nextAfter(f32, 2, 3), -3 };
    var near: Summary = .{};
    near.add(entryStats(&logits, &logits));
    near.add(entryStats(&logits, &moved));
    try testing.expectEqual(@as(?u64, null), near.first_top1_diff);
    try testing.expectEqual(Verdict.over_bound, verdict(near, 0));
    try testing.expectEqual(Verdict.within, verdict(near, 1e-6));
}
