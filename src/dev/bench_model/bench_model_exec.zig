//! The run body every placement of `bench-model` shares: the prompt ids, one
//! run, and the series of runs on one runner. The code is generic over the
//! runner type, so each family and each placement reach the same body.
//!
//! Comptime. Build options and runner declarations pick the chunked prefill and the recurrent
//! state prongs.
//!
//! Logits. Under `--logits-out` or `--force-tokens`, `runOnce` decodes through a
//! `LogitsRecorder`. Without the flags, the decode steps the runner directly.

const std = @import("std");
const zigferencer = @import("zigferencer");
const Io = std.Io;
const build_options = @import("build_options");

const context = zigferencer.cli.context;
const common = @import("bench_model_common.zig");
const report = @import("bench_model_report.zig");
const logits = @import("bench_model_logits.zig");
const bpe = zigferencer.bpe;
const chat_mod = zigferencer.chat;
const vocab_mod = zigferencer.vocab;
const generate = zigferencer.generate;
const step_row = zigferencer.step_row;

const Bench = common.Bench;
const RunResult = common.RunResult;

// -- types ------------------------------------------------------------------

/// What every run of one bench feeds and decodes.
pub const Turn = struct {
    vocab: *const vocab_mod.Vocab,
    prompt_ids: []const u32,
    decode_len: u32,
    ignore_eos: bool,
    /// The logit count of one entry.
    vocab_size: u32,
    logits: logits.Paths = .{},
};

const HashSink = struct {
    hasher: *std.hash.Fnv1a_64,
    depth: *common.DepthCurve,

    pub fn onToken(self: HashSink, id: u32) !void {
        self.hasher.update(std.mem.asBytes(&id));
    }

    pub fn onStep(self: HashSink, pos: u32, ns: u64) void {
        self.depth.add(pos, ns);
    }
};

// -- api --------------------------------------------------------------------

/// The turn of the flags of `b` over `prompt_ids`.
pub fn turnOf(b: *const Bench, v: *const vocab_mod.Vocab, prompt_ids: []const u32) Turn {
    std.debug.assert(prompt_ids.len >= 1);

    return .{
        .vocab = v,
        .prompt_ids = prompt_ids,
        .decode_len = b.opt.max_tokens,
        .ignore_eos = b.opt.ignore_eos,
        .vocab_size = b.loaded.card.vocab_size,
        .logits = .{ .out = b.opt.logits_out, .force = b.opt.force_tokens },
    };
}

/// The token ids of the bench: synthetic under `--prompt-tokens`, otherwise the
/// encoded prompt. The ids and the decode budget must fit `max_seq`.
pub fn promptIds(b: *const Bench, v: *const vocab_mod.Vocab, max_seq: u32) ![]const u32 {
    const ctx = b.ctx;
    const opt = b.opt;

    const ids = if (opt.prompt_tokens) |n| blk: {
        // The fit check runs first, because the synthesis allocates n ids.
        try common.requireSequenceFits(ctx.io, n, opt.max_tokens, max_seq);
        break :blk try common.synthPromptIds(ctx.arena, n, b.loaded.card.vocab_size);
    } else blk: {
        if (!opt.chat) break :blk try bpe.encodeAlloc(v, ctx.arena, opt.prompt, true);

        const p = try chat_mod.promptAlloc(ctx.arena, b.model.chat_template_kind, opt.prompt);
        break :blk try bpe.encodePromptAlloc(v, ctx.arena, p.text, p.plain(), true);
    };

    if (ids.len == 0) {
        common.writeStderr(ctx.io, "error: the prompt encodes to zero tokens\n");
        return error.EmptyPrompt;
    }

    try common.requireSequenceFits(ctx.io, ids.len, opt.max_tokens, max_seq);
    return ids;
}

/// One prefill chunk and one decode step, so that run 1 starts warm.
pub fn warmup(runner: anytype, prompt_ids: []const u32) !void {
    const chunk = @min(prompt_ids.len, @as(usize, runner.prefill_batch_max));
    _ = try runner.prefillBatch(prompt_ids[0..chunk], 0);
    _ = try runner.step(prompt_ids[prompt_ids.len - 1], @intCast(chunk));
}

/// Prefill the prompt from position zero, then decode up to `turn.decode_len`
/// tokens. The hash takes every generated token, so it is the determinism and
/// cross-backend signature of the run.
pub fn runOnce(runner: anytype, turn: Turn, ctx: context.Context) !RunResult {
    std.debug.assert(turn.prompt_ids.len >= 1);

    // The logits files open before the clock starts, so the open adds no TTFT.
    const prompt_len: u32 = @intCast(turn.prompt_ids.len);
    const header: logits.Header = .{
        .vocab = turn.vocab_size,
        .prompt_len = prompt_len,
        .first_pos = prompt_len,
        .prompt_hash = if (turn.logits.active()) logits.promptHash(turn.prompt_ids) else 0,
    };
    var files: logits.RunFiles = if (turn.logits.active())
        try .open(ctx.io, ctx.arena, turn.logits, header, turn.decode_len)
    else
        .{};
    defer files.close(ctx.io);

    // TTFT spans the prefill and the first token.
    const t_start = Io.Clock.Timestamp.now(ctx.io, .awake);
    const prefilled = try prefill(runner, turn.prompt_ids);

    var hasher = std.hash.Fnv1a_64.init();
    var depth: common.DepthCurve = .{};
    var sink: HashSink = .{ .hasher = &hasher, .depth = &depth };
    const max_seq: u32 = runner.max_seq;
    const generated = if (turn.logits.active()) blk: {
        std.debug.assert(prefilled.pos == header.first_pos);
        std.debug.assert(prefilled.last_logits.len == header.vocab);

        const out = if (files.out) |*w| w else null;
        var recorder = logits.LogitsRecorder(@TypeOf(runner))
            .init(runner, prefilled.last_logits, out, files.forced);
        const g = try decodeTurn(&recorder, max_seq, prefilled, turn, ctx.io, t_start, &sink);
        try recorder.finish();
        break :blk g;
    } else try decodeTurn(runner, max_seq, prefilled, turn, ctx.io, t_start, &sink);
    const t_end = Io.Clock.Timestamp.now(ctx.io, .awake);
    const turn_s = zigferencer.time.msBetween(t_start, t_end) / 1000.0;

    return .{
        .ttft_ms = generated.ttft_ms,
        .itl_ms = generated.itl_ms,
        .tok_s = @as(f64, @floatFromInt(generated.tokens_generated)) / turn_s,
        .tokens_hash = hasher.final(),
        .tokens_generated = generated.tokens_generated,
        .itl_by_depth = try depth.collect(ctx.arena),
        .host = generated.host,
    };
}

/// Run the bench `--runs` times on one runner, then print the report. A warmup
/// left state in the runner when `warm` is true.
pub fn runSeries(
    runner: anytype,
    b: *const Bench,
    turn: Turn,
    plan: common.PlanInfo,
    warm: bool,
) !void {
    std.debug.assert(b.opt.runs >= 1);

    const Runner = std.meta.Child(@TypeOf(runner));
    const results = try b.ctx.arena.alloc(RunResult, b.opt.runs);
    for (results, 0..) |*result, run_index| {
        // A run folds the prompt into the recurrent state, and no prefill rewrites it.
        // Positional KV needs no scrub, because a run writes a row before attention reads it.
        if (comptime @hasDecl(Runner, "has_recurrent_state") and Runner.has_recurrent_state) {
            if (warm or run_index > 0) {
                runner.resetSlotState(0);
            }
        }

        result.* = try runOnce(runner, turn, b.ctx);
        result.plan = plan;
    }

    return printReport(b, turn.prompt_ids.len, results);
}

/// Print `results` in the format of the flags.
pub fn printReport(b: *const Bench, prompt_len: usize, results: []const RunResult) !void {
    const w = &b.ctx.stdout.interface;
    const name = b.model.pub_name;
    const decode_len = b.opt.max_tokens;

    if (b.opt.json) {
        return report.printJson(w, name, prompt_len, decode_len, results, b.policy);
    }

    return report.printTable(w, name, b.opt.prompt, prompt_len, decode_len, results, b.policy);
}

// -- helpers ----------------------------------------------------------------

fn decodeTurn(
    runner: anytype,
    max_seq: u32,
    prefilled: generate.Prefilled,
    turn: Turn,
    io: Io,
    t_start: Io.Clock.Timestamp,
    sink: *HashSink,
) !generate.Generated {
    return generate.decode(
        runner,
        prefilled.last_logits,
        prefilled.pos,
        turn.decode_len,
        max_seq,
        turn.vocab.eos_id,
        turn.ignore_eos,
        io,
        t_start,
        sink,
    );
}

fn prefill(runner: anytype, prompt_ids: []const u32) !generate.Prefilled {
    const chunked = comptime build_options.use_vulkan_fused_ops or
        step_row.chunkedPrefill(@TypeOf(runner));

    if (chunked) {
        return generate.prefillChunks(runner, prompt_ids);
    }

    return generate.prefillTokens(runner, prompt_ids);
}
