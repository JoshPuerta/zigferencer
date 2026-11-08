//! Output formatting for the bench-model command: the result values of `bench_model_common` as the
//! human table and the single-line JSON. The module is pure and keeps no state.
//!
//! Comptime. `statsOfField` takes the result field at compile time, so one body covers every
//! statistic.

const std = @import("std");
const zigferencer = @import("zigferencer");
const Io = std.Io;
const build_options = @import("build_options");

const parallel = zigferencer.parallel;
const step_stats = zigferencer.step_stats;
const context = zigferencer.cli.context;
const common = @import("bench_model_common.zig");
const RunResult = common.RunResult;
const PlanInfo = common.PlanInfo;
const DepthBucket = common.DepthBucket;

// -- types ------------------------------------------------------------------

const Stats = struct { mean: f64, stddev: f64, min: f64, max: f64 };

// -- api --------------------------------------------------------------------

pub fn printTable(
    w: *Io.Writer,
    model_name: []const u8,
    prompt: []const u8,
    prompt_tokens: usize,
    decode_len: u32,
    results: []const RunResult,
    policy: parallel.Policy,
) !void {
    // Print the run header.
    try w.print("model:    {s}\n", .{model_name});
    try w.print("backend:  {s}\n", .{build_options.ops_backend});
    try w.print(
        "threads:  {d}  (min_parallel_flops={d})\n",
        .{ policy.n_threads, policy.min_parallel_flops },
    );
    try w.print("prompt:   {s}  ({d} tokens)\n", .{ prompt, prompt_tokens });
    try printPlan(w, results[0].plan);
    try w.print("decode:   {d} tokens, {d} run(s)\n", .{ decode_len, results.len });

    // Print one row per run.
    try w.writeAll("\n");
    try w.print(
        "{s:>4}  {s:>10}  {s:>10}  {s:>10}  {s}\n",
        .{ "run", "ttft_ms", "itl_ms", "tokens", "tokens_hash" },
    );
    for (results, 0..) |r, i| {
        try w.print(
            "{d:>4}  {d:>10.3}  {d:>10.3}  {d:>10}  0x{x:0>16}\n",
            .{ i + 1, r.ttft_ms, r.itl_ms, r.tokens_generated, r.tokens_hash },
        );
    }

    // Print the spread across the runs.
    const ts = statsOfField(results, .ttft_ms);
    const is = statsOfField(results, .itl_ms);
    const rs = statsOfField(results, .tok_s);
    try w.writeAll("\n");
    try w.print(
        "ttft_ms:  mean={d:.3}  stddev={d:.3}  min={d:.3}  max={d:.3}\n",
        .{ ts.mean, ts.stddev, ts.min, ts.max },
    );
    try w.print(
        "itl_ms:   mean={d:.3}  stddev={d:.3}  min={d:.3}  max={d:.3}\n",
        .{ is.mean, is.stddev, is.min, is.max },
    );
    try w.print(
        "tok_s:    mean={d:.3}  stddev={d:.3}  min={d:.3}  max={d:.3}  " ++
            "(whole turn, with prefill)\n",
        .{ rs.mean, rs.stddev, rs.min, rs.max },
    );
    try printHostSteps(w, results);

    // Check that every run of one binary produced the same hash.
    var all_match: bool = true;
    for (results[1..]) |r| {
        if (r.tokens_hash != results[0].tokens_hash) {
            all_match = false;
        }
    }

    try w.print(
        "determinism: {s}\n",
        .{if (all_match) "ok (all runs identical)" else "DIVERGED (runs disagree, which is a bug)"},
    );

    try w.flush();
}

pub fn printJson(
    w: *Io.Writer,
    model_name: []const u8,
    prompt_tokens: usize,
    decode_len: u32,
    results: []const RunResult,
    policy: parallel.Policy,
) !void {
    std.debug.assert(results.len > 0);

    // Open the object with the run parameters.
    try w.print(
        \\{{"model":"{s}","backend":"{s}","threads":{d},"min_parallel_flops":{d},"prompt_tokens":{d},"decode_len":{d},"runs":[
    ,
        .{
            model_name,
            build_options.ops_backend,
            policy.n_threads,
            policy.min_parallel_flops,
            prompt_tokens,
            decode_len,
        },
    );
    // Write one member per run.
    for (results, 0..) |r, i| {
        if (i > 0) {
            try w.writeAll(",");
        }

        try w.print(
            "{{\"ttft_ms\":{d:.3},\"itl_ms\":{d:.3},\"tok_s\":{d:.3}," ++
                "\"tokens_generated\":{d},\"tokens_hash\":\"0x{x:0>16}\",",
            .{
                r.ttft_ms, r.itl_ms, r.tok_s, r.tokens_generated, r.tokens_hash,
            },
        );
        try printDepthJson(w, r.itl_by_depth);
        try w.writeAll("}");
    }

    // Close with the KV plan, which all runs share. The key is always present, and zeros mean none.
    const plan = results[0].plan;
    try w.print(
        "],\"kv_plan\":{{\"n_slots\":{d},\"window\":{d},\"kv_mib\":{d},\"kv_dtype\":\"{t}\"}}}}\n",
        .{
            plan.n_slots, plan.window, plan.kv_mib, plan.kv_dtype,
        },
    );
    try w.flush();
}

// -- helpers ----------------------------------------------------------------

/// The host steps of the decode step, one line each, merged over the runs, then the ring counts.
/// A build with no fused runner prints nothing.
fn printHostSteps(w: *Io.Writer, results: []const RunResult) !void {
    var host: step_stats.DecodeHost = .{};
    for (results) |r| {
        host.merge(r.host);
    }

    if (!host.hasDevice()) return;

    inline for (std.meta.fields(step_stats.DecodeHost)) |f| {
        if (f.type != step_stats.Stat) continue;
        const s = @field(host, f.name);
        try w.print(
            "host_" ++ f.name ++ "_ms: avg={d:.3}  max={d:.3}  n={d}\n",
            .{ s.avgMs(), s.maxMs(), s.count },
        );
    }

    const r = host.ring;
    try w.print(
        "host_ring: hits={d}  misses={d}  speculated={d}  wasted={d}\n",
        .{ r.hits, r.misses, r.speculated, r.wasted },
    );
}

/// The ITL depth curve as one JSON member. The keys avoid `itl_ms` and `ttft_ms`, because the
/// sweep scripts grep every match of those.
fn printDepthJson(w: *Io.Writer, buckets: []const DepthBucket) !void {
    try w.writeAll("\"itl_by_depth\":[");
    for (buckets, 0..) |b, i| {
        if (i > 0) {
            try w.writeAll(",");
        }

        try w.print(
            "{{\"pos\":{d},\"steps\":{d},\"mean_ms\":{d:.3}}}",
            .{ b.pos, b.steps, b.meanMs() },
        );
    }

    try w.writeAll("]");
}

/// The KV shape of the run. Zero slots mean no device KV plan, and the header stays unchanged.
fn printPlan(w: *Io.Writer, plan: PlanInfo) !void {
    if (plan.n_slots == 0) return;
    try w.print(
        "kv plan:  {d} slots x {d} window{s} = {d} MiB\n",
        .{ plan.n_slots, plan.window, plan.kv_dtype.planToken(), plan.kv_mib },
    );
}

/// Mean, stddev, min and max of one f64 field across the runs. A non-f64 `field` fails to compile.
fn statsOfField(results: []const RunResult, comptime field: std.meta.FieldEnum(RunResult)) Stats {
    std.debug.assert(results.len > 0);

    const name = @tagName(field);
    var sum: f64 = 0;
    var lo: f64 = @field(results[0], name);
    var hi: f64 = lo;
    for (results) |r| {
        const v: f64 = @field(r, name);
        sum += v;
        if (v < lo) {
            lo = v;
        }

        if (v > hi) {
            hi = v;
        }
    }

    const mean = sum / @as(f64, @floatFromInt(results.len));
    var sq: f64 = 0;
    for (results) |r| {
        const v: f64 = @field(r, name);
        sq += (v - mean) * (v - mean);
    }

    const variance = sq / @as(f64, @floatFromInt(results.len));
    return .{ .mean = mean, .stddev = @sqrt(variance), .min = lo, .max = hi };
}

// -- tests ------------------------------------------------------------------

const json_test_bytes = 256;
const empty_test_bytes = 32;
const steps_test_bytes = 512;

test "printDepthJson: one member on one line, no key the sweep scripts grep" {
    var buf: [json_test_bytes]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    const buckets = [_]DepthBucket{
        .{ .pos = 0, .steps = 4, .ns = 10_000_000 },
        .{ .pos = 512, .steps = 2, .ns = 7_000_000 },
    };
    try printDepthJson(&w, &buckets);
    const out = w.buffered();
    try std.testing.expectEqualStrings(
        \\"itl_by_depth":[{"pos":0,"steps":4,"mean_ms":2.500},{"pos":512,"steps":2,"mean_ms":3.500}]
    , out);
    try std.testing.expect(std.mem.find(u8, out, "\"itl_ms\":") == null);
    try std.testing.expect(std.mem.find(u8, out, "\"ttft_ms\":") == null);
    try std.testing.expect(std.mem.findScalar(u8, out, '\n') == null);

    var empty_buf: [empty_test_bytes]u8 = undefined;
    var empty: Io.Writer = .fixed(&empty_buf);
    try printDepthJson(&empty, &.{});
    try std.testing.expectEqualStrings("\"itl_by_depth\":[]", empty.buffered());
}

test "printHostSteps: merged host steps print one line each, an untimed run prints none" {
    var rr: RunResult = .{
        .ttft_ms = 0,
        .itl_ms = 0,
        .tokens_hash = 0,
        .tokens_generated = 0,
        .itl_by_depth = &.{},
    };
    var buf: [steps_test_bytes]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try printHostSteps(&w, &.{rr});
    try std.testing.expectEqual(@as(usize, 0), w.buffered().len);

    rr.host.record.add(1_000_000);
    rr.host.record.add(3_000_000);
    var rr2 = rr;
    rr2.host.argmax.add(500_000);
    try printHostSteps(&w, &.{ rr, rr2 });
    const out = w.buffered();
    try std.testing.expect(
        std.mem.find(u8, out, "host_record_ms: avg=2.000  max=3.000  n=4\n") != null,
    );
    try std.testing.expect(
        std.mem.find(u8, out, "host_argmax_ms: avg=0.500  max=0.500  n=1\n") != null,
    );
    try std.testing.expect(
        std.mem.find(u8, out, "host_wait_ms: avg=0.000  max=0.000  n=0\n") != null,
    );
    try std.testing.expect(
        std.mem.find(u8, out, "host_ring: hits=0  misses=0  speculated=0  wasted=0\n") != null,
    );

    w = .fixed(&buf);
    rr.host.ring = .{ .hits = 31, .misses = 1, .speculated = 32, .wasted = 1 };
    try printHostSteps(&w, &.{rr});
    try std.testing.expect(
        std.mem.find(
            u8,
            w.buffered(),
            "host_ring: hits=31  misses=1  speculated=32  wasted=1\n",
        ) != null,
    );
}
