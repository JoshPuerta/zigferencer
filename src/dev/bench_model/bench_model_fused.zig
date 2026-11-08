//! The whole-model GPU path of `bench-model`, on a fused build. Every run shares
//! one runner, so the weights upload once.
//!
//! Comptime. Generic over the model family. A block rejects a non-fused build.

const std = @import("std");
const zigferencer = @import("zigferencer");
const Io = std.Io;
const build_options = @import("build_options");

const common = @import("bench_model_common.zig");
const exec = @import("bench_model_exec.zig");
const boot_resources = zigferencer.cli.boot_resources;
const boot_lines = zigferencer.cli.boot_lines;
const vocab_mod = zigferencer.vocab;
const sampler = zigferencer.sampler;
const vulkan_ctx_mod = zigferencer.vulkan_ctx;

comptime {
    if (!build_options.use_vulkan_fused_ops) {
        @compileError("cli/bench_model_fused.zig was included in a non-fused build.");
    }
}

const Bench = common.Bench;

// -- constants --------------------------------------------------------------

const profile_tag_bytes = 40;

// -- api --------------------------------------------------------------------

pub fn bench(comptime Family: type, b: *const Bench) !void {
    const ctx = b.ctx;
    const opt = b.opt;
    const kv_dtype = b.kvDtype();

    const ctxp = try ctx.arena.create(vulkan_ctx_mod.Context);
    ctxp.* = try vulkan_ctx_mod.Context.init(ctx.arena, ctx.vk_options);
    defer ctxp.deinit();

    const f = try boot_resources.fusedOn(ctx.io, b.boot, ctxp);
    _ = try boot_lines.logFused(ctx.io, b.model.pub_name, f, true, false, kv_dtype);

    const runner = try Family.initFused(ctx.gpa, ctxp, b.loaded, f.plan, kv_dtype);
    defer runner.deinit();

    const v = try vocab_mod.buildFromParsed(ctx.arena, b.loaded.parsed);
    const prompt_ids = try exec.promptIds(b, &v, runner.max_seq);

    if (comptime Family.fused_warmup) {
        try warmup(runner, ctx.io, opt, prompt_ids);
    } else {
        if (opt.profile_prefill) return profileLastChunk(runner, ctx.io, prompt_ids);
        if (opt.profile_decode) return profileOneStep(runner, ctx.io, prompt_ids);
    }

    const turn: exec.Turn = .{
        .vocab = &v,
        .prompt_ids = prompt_ids,
        .decode_len = opt.max_tokens,
        .ignore_eos = opt.ignore_eos,
    };
    const plan = common.PlanInfo.fromPlan(f.plan, kv_dtype);
    return exec.runSeries(runner, b, turn, plan, false);
}

// -- helpers ----------------------------------------------------------------

/// One prefill chunk and one decode step before the timed runs. The two profile
/// flags print the stage breakdown of these dispatches, and the runs follow.
fn warmup(runner: anytype, io: Io, opt: common.Options, prompt_ids: []const u32) !void {
    if (!opt.profile_prefill and !opt.profile_decode) {
        return exec.warmup(runner, prompt_ids);
    }

    const chunk = @min(prompt_ids.len, @as(usize, runner.prefill_batch_max));
    const last_id = prompt_ids[prompt_ids.len - 1];

    if (opt.profile_prefill) {
        try runner.profilePrefill(io, prompt_ids[0..chunk], 0, "warmup");
    } else {
        _ = try runner.prefillBatch(prompt_ids[0..chunk], 0);
    }

    if (opt.profile_decode) {
        try runner.profileStep(io, last_id, @intCast(chunk), "warmup");
    } else {
        _ = try runner.step(last_id, @intCast(chunk));
    }
}

/// Prefill every chunk but the last, then profile the last one, so a long prompt
/// profiles the chunk at its full depth. Profiling mutates the runner state, so
/// no decode follows.
fn profileLastChunk(runner: anytype, io: Io, prompt_ids: []const u32) !void {
    const batch_max: usize = runner.prefill_batch_max;
    const last_len = prompt_ids.len - (prompt_ids.len - 1) / batch_max * batch_max;
    const last_start = prompt_ids.len - last_len;
    std.debug.assert(last_len >= 1);
    std.debug.assert(last_len <= batch_max);

    var pos: usize = 0;
    while (pos < last_start) : (pos += batch_max) {
        _ = try runner.prefillBatch(prompt_ids[pos..][0..batch_max], @intCast(pos));
    }

    std.debug.assert(pos == last_start);

    var tag_buffer: [profile_tag_bytes]u8 = undefined;
    const tag = std.fmt.bufPrint(
        &tag_buffer,
        "profile from {d}",
        .{last_start},
    ) catch unreachable;
    try runner.profilePrefill(io, prompt_ids[last_start..], @intCast(last_start), tag);
}

/// Prefill the prompt, which gives a realistic position and a populated KV and
/// state, then profile one decode step.
fn profileOneStep(runner: anytype, io: Io, prompt_ids: []const u32) !void {
    const window: u32 = runner.max_seq;
    var pos: u32 = 0;
    var last_logits: []f32 = undefined;

    while (pos < prompt_ids.len and pos < window) {
        const room = @min(runner.prefill_batch_max, window - pos);
        const chunk: u32 = @intCast(@min(prompt_ids.len - pos, @as(usize, room)));
        last_logits = try runner.prefillBatch(prompt_ids[pos..][0..chunk], pos);
        pos += chunk;
    }

    try runner.profileStep(io, sampler.greedy(last_logits), pos, "profile");
}
