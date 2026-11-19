//! `bench-model --profile-cpu-decode`: the per-op decode timer of the CPU runners.
//! It prefills the prompt unprofiled, so its numbers are steady-state decode, and
//! prints the breakdown to stderr, because it is a diagnostic and not a result.
//! The sums fill in a trace build only. The ops of `layer_op.LayerOp` are the one list of lines,
//! grouped by `OpClass`, so a new op reaches the table with no edit here.
//!
//! Comptime. Generic over the model family and the runner, so every CPU runner shares one timer
//! body.

const std = @import("std");
const zigferencer = @import("zigferencer");
const Io = std.Io;
const build_options = @import("build_options");

const common = @import("bench_model_common.zig");
const kv_budget = zigferencer.kv_budget;
const limits = zigferencer.limits;
const cpu_profile = zigferencer.cpu_profile;
const layer_op = zigferencer.layer_op;
const parallel = zigferencer.parallel;
const Pool = zigferencer.pool.Pool;
const sampler = zigferencer.sampler;
const span = zigferencer.span;

const CpuProfile = cpu_profile.CpuProfile;
const LayerOp = layer_op.LayerOp;
const OpClass = layer_op.OpClass;

// -- constants --------------------------------------------------------------

const stderr_buffer_bytes = 4096;

/// The width of the line label: an op name under the indent of its class.
const label_width = 16;

// -- types ------------------------------------------------------------------

/// The decode steps and the total nanoseconds that turn a sum into a table line.
const Scale = struct { steps: f64, total: f64 };

// -- api --------------------------------------------------------------------

pub fn bench(
    comptime Family: type,
    b: *const common.Bench,
    pool: *Pool,
    prompt_ids: []const u32,
    max_seq: u32,
) !void {
    const ctx = b.ctx;
    const decode_len = b.opt.max_tokens;

    var runner = try Family.Cpu.init(ctx.arena, b.loaded, .{
        .plan = kv_budget.pinned(1, max_seq),
        .max_rows = common.cpuPrefillRows(b.opt),
        .pool = pool,
        .policy = b.policy,
        .kv_dtype = b.kvDtype(),
    });
    const prof = try profileDecode(
        Family.Cpu,
        &runner,
        Family.profile_family,
        prompt_ids,
        decode_len,
        max_seq,
    );

    const pos_end: u32 = @intCast(@min(prompt_ids.len, max_seq) + prof.steps);
    try reportCpuProfile(ctx.io, &prof, b, prompt_ids.len, pos_end);
}

// -- helpers ----------------------------------------------------------------

/// Prefill `prompt_ids` unprofiled, then decode up to `decode_len` greedy steps
/// with a `family` profile installed.
fn profileDecode(
    comptime Runner: type,
    runner: *Runner,
    family: cpu_profile.Family,
    prompt_ids: []const u32,
    decode_len: u32,
    max_seq: u32,
) !CpuProfile {
    std.debug.assert(prompt_ids.len > 0);

    const n_prompt: u32 = @intCast(@min(prompt_ids.len, max_seq));
    var pos: u32 = 0;
    var last: []f32 = undefined;
    while (pos < n_prompt) {
        const chunk: u32 = @min(n_prompt - pos, runner.prefill_batch_max);
        last = try runner.prefillBatch(prompt_ids[pos..][0..chunk], pos);
        pos += chunk;
    }

    var tok = sampler.greedy(last);

    var prof: CpuProfile = .{ .family = family };
    span.installSums(&prof.sums);
    defer span.installSums(null);

    var k: u32 = 0;
    while (k < decode_len and pos < max_seq) : (k += 1) {
        const lg = try runner.step(tok, pos);
        pos += 1;
        tok = sampler.greedy(lg);
    }

    prof.steps = k;
    std.debug.assert(prof.steps == k);
    std.debug.assert(prof.total() > 0);
    return prof;
}

fn reportCpuProfile(
    io: Io,
    p: *const CpuProfile,
    b: *const common.Bench,
    prompt_len: usize,
    pos_end: u32,
) !void {
    // Derive the serial share and the Amdahl ceiling.
    const model_name = b.model.pub_name;
    const backend = build_options.ops_backend;
    const policy = b.policy;
    const n_layers = b.loaded.card.n_layers;

    const total: f64 = @floatFromInt(p.total());
    const steps: f64 = @floatFromInt(p.steps);
    const parallel_ns: f64 = @floatFromInt(p.nsParallel());
    const serial_ns = total - parallel_ns;
    const serial_frac = serial_ns / total;
    const n: f64 = @floatFromInt(@max(policy.n_threads, 1));
    // The ceiling treats every parallel line as perfectly parallel.
    // The matmuls are bandwidth-bound.
    const amdahl = 1.0 / (serial_frac + (1.0 - serial_frac) / n);

    // Print the header lines.
    var stderr_buffer: [stderr_buffer_bytes]u8 = undefined;
    var stderr_writer: Io.File.Writer = .initStreaming(.stderr(), io, &stderr_buffer);
    const w = &stderr_writer.interface;

    try w.print(
        "[cpu-profile decode] model={s} backend={s} threads={d} flops={d}\n",
        .{ model_name, backend, policy.n_threads, policy.min_parallel_flops },
    );
    try w.print(
        "  layers={d} prompt_tokens={d} decode_steps={d} pos_end={d}\n\n",
        .{ n_layers, prompt_len, p.steps, pos_end },
    );
    try w.print(
        "  {s:<16}{s:>12}{s:>12}{s:>8}  kind\n",
        .{ "op", "ns/step", "ms/step", "%step" },
    );
    // Print each class of the family, then its ops.
    try w.writeAll("  " ++ "-" ** 56 ++ "\n");
    const scale: Scale = .{ .steps = steps, .total = total };
    try printClasses(w, p, scale);

    // Print the totals and the ceiling.
    try w.writeAll("  " ++ "-" ** 56 ++ "\n");
    try printLine(w, scale, "TOTAL", 0, total, null);
    try w.print(
        "\n  parallel: {d:.1}%   serial: {d:.1}%\n",
        .{ 100.0 * (1.0 - serial_frac), 100.0 * serial_frac },
    );
    try w.print(
        "  ideal-Amdahl ceiling @ {d} threads: {d:.2}x  (real lower: " ++
            "the matmuls are bandwidth-bound)\n",
        .{ policy.n_threads, amdahl },
    );
    try w.flush();
}

/// One line for each class that the family times, then one line for each of its ops.
fn printClasses(w: *Io.Writer, p: *const CpuProfile, scale: Scale) !void {
    for (std.enums.values(OpClass)) |class| {
        var class_ns: u64 = 0;
        var parallel_class = false;
        for (std.enums.values(LayerOp)) |op| {
            if (op.class() != class or !cpu_profile.usedBy(op, p.family)) continue;

            class_ns += p.nsOf(op);
            parallel_class = cpu_profile.isParallel(op);
        }

        if (class_ns == 0) continue;

        const kind: []const u8 = if (parallel_class) "parallel" else "serial";
        try printLine(w, scale, @tagName(class), 0, @floatFromInt(class_ns), kind);
        for (std.enums.values(LayerOp)) |op| {
            if (op.class() != class or !cpu_profile.usedBy(op, p.family)) continue;

            try printLine(w, scale, @tagName(op), 2, @floatFromInt(p.nsOf(op)), null);
        }
    }
}

/// One table line: the label at `indent`, the time per step, the share of the step, and the kind of
/// a class line.
fn printLine(
    w: *Io.Writer,
    scale: Scale,
    label: []const u8,
    indent: usize,
    ns: f64,
    kind: ?[]const u8,
) !void {
    std.debug.assert(indent + label.len <= label_width);

    var text: [label_width]u8 = @splat(' ');
    @memcpy(text[indent..][0..label.len], label);
    const ns_step = ns / scale.steps;
    try w.print("  {s}{d:>12.0}{d:>12.4}{d:>7.1}%", .{
        &text,
        ns_step,
        ns_step / 1e6,
        100.0 * ns / scale.total,
    });
    if (kind) |k| {
        try w.print("  {s}", .{k});
    }

    try w.writeAll("\n");
}
