//! The run body every placement of `bench-model` shares: the prompt ids, one
//! run, and the series of runs on one runner. The code is generic over the
//! runner type, so each family and each placement reach the same body.
//!
//! Comptime. Build options and runner declarations pick the chunked prefill and the recurrent
//! state prongs.

const std = @import("std");
const zigferencer = @import("zigferencer");
const Io = std.Io;
const build_options = @import("build_options");

const context = zigferencer.cli.context;
const common = @import("bench_model_common.zig");
const report = @import("bench_model_report.zig");
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

    // TTFT spans the prefill and the first token.
    const t_start = Io.Clock.Timestamp.now(ctx.io, .awake);
    const prefilled = try prefill(runner, turn.prompt_ids);

    var hasher = std.hash.Fnv1a_64.init();
    var depth: common.DepthCurve = .{};
    var sink: HashSink = .{ .hasher = &hasher, .depth = &depth };
    const generated = try generate.decode(
        runner,
        prefilled.last_logits,
        prefilled.pos,
        turn.decode_len,
        runner.max_seq,
        turn.vocab.eos_id,
        turn.ignore_eos,
        ctx.io,
        t_start,
        &sink,
    );
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

fn prefill(runner: anytype, prompt_ids: []const u32) !generate.Prefilled {
    const chunked = comptime build_options.use_vulkan_fused_ops or
        step_row.chunkedPrefill(@TypeOf(runner));

    if (chunked) {
        return generate.prefillChunks(runner, prompt_ids);
    }

    return generate.prefillTokens(runner, prompt_ids);
}
