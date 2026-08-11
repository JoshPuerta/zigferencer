//! The bench-model execution datapath: drive a loaded model through one prompt
//! and its decode, then return the timing and hash values. The code is generic
//! over the runner type (CPU LlamaRunner, QwenRunner, or a GPU runner through
//! `anytype`), so both families and every placement reach one run body. A build
//! links exactly one of them, and the comptime branches resolve to it.

const std = @import("std");
const Io = std.Io;
const build_options = @import("build_options");

const cli = @import("command.zig");
const span = @import("../core/span.zig");
const loader = @import("../io/gguf/loader.zig");
const vocab_mod = @import("../core/tokenizer/vocab.zig");
const limits = @import("../core/limits.zig");
const kv_budget = @import("../core/kv_budget.zig");
const parallel = @import("../runtime/parallel.zig");
const Pool = @import("../runtime/pool.zig").Pool;
const step_row = @import("../runtime/step_row.zig");
const dtype = @import("../core/dtype.zig");
const LlamaRunner = @import("../runtime/llama_runner.zig").LlamaRunner;
const QwenRunner = @import("../runtime/qwen_runner.zig").QwenRunner;
const generate = @import("../runtime/generate.zig");
const common = @import("bench_model_common.zig");
const RunResult = common.RunResult;

// -- types ------------------------------------------------------------------

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

/// Run one bench pass at `max_seq`, the window the caller resolved. A fused
/// build drives the caller's runner, which already holds that window, and a
/// host build builds a fresh runner at it.
pub fn doOneRun(
    fused: anytype,
    loaded: *const loader.Loaded,
    max_seq: u32,
    v: *const vocab_mod.Vocab,
    prompt_ids: []const u32,
    decode_len: u32,
    ignore_eos: bool,
    ctx: cli.Context,
    pool: *Pool,
    policy: parallel.Policy,
    kv_dtype: dtype.KvDtype,
) !RunResult {
    std.debug.assert(max_seq >= 1);
    std.debug.assert(max_seq <= limits.max_model_ctx);

    if (comptime build_options.use_vulkan_fused_ops) {
        std.debug.assert(maxSeqOf(fused) == max_seq);
        // The CLI owns the LlamaFusedRunner, and it is itself the runner `runOnce`
        // drives, because it exposes the same `step`, `prefillBatch` and
        // `max_seq` surface. `run` creates it once and shares it across runs.
        // Nothing zeroes the GPU KV cache between runs, because attention reads
        // only positions 0..t_len, and every one is written before it is read.
        return runOnce(fused, v, prompt_ids, decode_len, ignore_eos, ctx);
    } else {
        // Qwen3.5 has its own CPU runner for the hybrid GDN stack. `run()`
        // rejects GPU qwen, so this branch only compiles for CPU backends, which
        // keeps the GPU ops module free of QwenRunner's full CPU op surface.
        // Both CPU runners share the `step` and `kv` surface `runOnce` needs.
        if (comptime !build_options.use_vulkan_ops) {
            if (loaded.card.arch == .qwen35) {
                var qrunner = try QwenRunner.init(ctx.arena, loaded, kv_budget.pinned(1, max_seq), limits.cpu_prefill_chunk, ctx.io, pool, policy, kv_dtype);
                return runOnce(&qrunner, v, prompt_ids, decode_len, ignore_eos, ctx);
            }
        }

        // A fresh runner per run starts the KV cache at zero.
        var runner = try LlamaRunner.init(ctx.arena, loaded, kv_budget.pinned(1, max_seq), limits.cpu_prefill_chunk, ctx.io, pool, policy, kv_dtype);
        return runOnce(&runner, v, prompt_ids, decode_len, ignore_eos, ctx);
    }
}

/// Prefill `prompt_ids` against the runner from position zero, then decode up
/// to `decode_len` tokens. The decode stops on EOS or on a full KV cache. The
/// hash takes every generated token, so it is the determinism and
/// cross-backend correctness signature of the run.
pub fn runOnce(
    runner: anytype,
    v: *const vocab_mod.Vocab,
    prompt_ids: []const u32,
    decode_len: u32,
    ignore_eos: bool,
    ctx: cli.Context,
) !RunResult {
    std.debug.assert(prompt_ids.len >= 1);
    const io = ctx.io;
    var pos: u32 = 0;

    // t_start starts before prefill, so TTFT spans prefill and the first token.
    const t_start = Io.Clock.Timestamp.now(io, .awake);
    var last_logits: []f32 = undefined;
    if (comptime build_options.use_vulkan_fused_ops or step_row.chunkedPrefill(@TypeOf(runner))) {
        // Batched prefill runs in chunks of up to `prefill_batch_max`. Each
        // chunk writes its KV rows before its rows attend, so the chunks
        // compose, and each one advances `pos` by its own size.
        const batch_max: u32 = runner.prefill_batch_max;
        var off: usize = 0;
        while (off < prompt_ids.len) {
            const remaining = prompt_ids.len - off;
            const cap = maxSeqOf(runner);
            const room = if (pos < cap) cap - pos else 0;
            if (room == 0) break;
            const chunk: u32 = @intCast(@min(remaining, @as(usize, @min(batch_max, room))));
            const sp = span.begin(.prefill_chunk);
            last_logits = try runner.prefillBatch(prompt_ids[off..][0..chunk], pos);
            sp.end();
            pos += chunk;
            off += chunk;
        }
    } else {
        for (prompt_ids) |tid| {
            if (pos >= maxSeqOf(runner)) break;
            const sp = span.begin(.prefill_chunk);
            last_logits = try runner.step(tid, pos);
            sp.end();
            pos += 1;
        }
    }

    var hasher = std.hash.Fnv1a_64.init();
    var depth: common.DepthCurve = .{};
    var sink: HashSink = .{ .hasher = &hasher, .depth = &depth };
    const gen = try generate.decode(runner, last_logits, pos, decode_len, maxSeqOf(runner), v.eos_id, ignore_eos, io, t_start, &sink);
    return .{
        .ttft_ms = gen.ttft_ms,
        .itl_ms = gen.itl_ms,
        .tokens_hash = hasher.final(),
        .tokens_generated = gen.tokens_generated,
        .itl_by_depth = try depth.collect(ctx.arena),
        .host = gen.host,
    };
}

// -- helpers ----------------------------------------------------------------

/// Sequence cap of whichever runner drives the turn. Every runner (LlamaRunner,
/// QwenRunner, and the fused GPU runners) exposes it flat as `max_seq`.
inline fn maxSeqOf(runner: anytype) u32 {
    return runner.max_seq;
}
