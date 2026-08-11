//! `zigferencer run <pub_name> --prompt "..." --max-tokens N`: the full pipeline.
//! Find the model on disk, mmap and parse it, then build the Vocab. Init the
//! runner of the model arch, prefill the prompt, decode N tokens, and stream the text to stdout.
//! `--chat` wraps the prompt with the model's chat template (`core/chat.zig`),
//! and `--thinking` sets the thinking mode of that template.
//! This is the interactive surface, and it complements `bench-model`, which
//! measures instead of streams. Decoding is greedy, so the output is
//! deterministic. See `runtime/sampler.zig`.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const cli = @import("command.zig");
const trace_span = @import("../core/span.zig");
const args = @import("args.zig");
const resolve_mod = @import("../engine/resolve.zig");
const flags_mod = @import("flags.zig");
const style = @import("style.zig");
const select = @import("select.zig");
const serve_common = @import("serve_common.zig");
const op_requires = @import("../core/op_requires.zig");
const generate = @import("../runtime/generate.zig");
const registry = @import("../core/registry.zig");
const loader = @import("../io/gguf/loader.zig");
const vocab_mod = @import("../core/tokenizer/vocab.zig");
const bpe = @import("../core/tokenizer/bpe.zig");
const chat_mod = @import("../core/chat.zig");
const dtype = @import("../core/dtype.zig");
const LlamaRunner = @import("../runtime/llama_runner.zig").LlamaRunner;
const QwenRunner = @import("../runtime/qwen_runner.zig").QwenRunner;
const limits = @import("../core/limits.zig");
const kv_budget = @import("../core/kv_budget.zig");
const time = @import("../core/time.zig");
const parallel = @import("../runtime/parallel.zig");
const Pool = @import("../runtime/pool.zig").Pool;
const step_row = @import("../runtime/step_row.zig");
const prompt_cache_mod = @import("../runtime/prompt_cache.zig");
const build_options = @import("build_options");
const vulkan_ops_wrap = if (build_options.use_vulkan_ops)
    @import("../ops_vulkan/wrap.zig")
else
    void;
const llama_fused = if (build_options.use_vulkan_fused_ops)
    @import("../runtime/llama_fused.zig")
else
    void;
const qwen_fused = if (build_options.use_vulkan_fused_ops)
    @import("../runtime/qwen_fused.zig")
else
    void;
const vulkan_ctx_mod = if (build_options.use_vulkan_fused_ops)
    @import("../runtime/vulkan_ctx.zig")
else
    void;
const llama_hybrid_mod = if (build_options.use_vulkan_fused_ops)
    @import("../runtime/llama_hybrid.zig")
else
    void;
const qwen_hybrid_mod = if (build_options.use_vulkan_fused_ops)
    @import("../runtime/qwen_hybrid.zig")
else
    void;

const log = std.log.scoped(.run);

// -- types ------------------------------------------------------------------

/// The per-token effect of `run`: detokenize the id and stream the piece to
/// stdout as it arrives. EOS is generated, and the bench sink folds it into its
/// hash. This sink never prints EOS, because EOS is a boundary marker, not text.
const StreamSink = struct {
    out: *Io.Writer,
    v: *vocab_mod.Vocab,
    arena: Allocator,
    pub fn onToken(self: StreamSink, id: u32) !void {
        if (id == self.v.eos_id) return;
        const piece = try bpe.decodeAlloc(self.v, self.arena, &[_]u32{id});
        try self.out.writeAll(piece);
        try self.out.flush();
    }
};

/// The outcome of a placed dispatch. `.all_host` hands the run and its host arm
/// back to the caller, which owns the one CPU path, so the placed path never
/// duplicates it.
const Placed = union(enum) { done, all_host: resolve_mod.Result(resolve_mod.CpuArm) };

/// The flags of `run`.
const Flags = struct {
    pub_name: ?[]const u8 = null,
    prompt: []const u8 = "Hello",
    max_tokens: u32 = 32,
    chat: bool = false,
    /// A hit restores the snapshot of the prompt and skips prefill. A miss
    /// prefills, then keeps the end-of-prompt boundary, so a later run is
    /// instant even in a fresh process through the disk tier.
    prompt_cache: bool = false,
    engine: flags_mod.EngineFlags = .{},

    pub const docs_from = flags_mod.EngineDocs;
    pub const positional = "pub_name";
    pub const omit: []const []const u8 = &.{"thinking_budget"};
    pub const help: args.Help = .{
        .usage = "zigferencer run <pub_name> [flags]   (<pub_name> omitted -> pick a model)",
        .about = "Generate text from a prompt, greedy, streamed to stdout.\n",
        .footer =
        \\examples:
        \\  zigferencer run SmolLM2-135M-Instruct-Q8_0 --prompt "Hello" --max-tokens 64
        \\  zigferencer run --chat
        \\  zigferencer run Qwen3.5-2B-Q8_0 --chat --thinking on --max-tokens 256
        \\
        ,
    };
    pub const docs = .{
        .prompt = args.Doc{ .text = "prompt to continue" },
        .max_tokens = args.Doc{ .text = "the most tokens the turn generates" },
        .chat = args.Doc{ .text = "apply the chat template of the model" },
        .prompt_cache = args.Doc{
            .group = "gpu (fused-GPU build only)",
            .text = "reuse a cached prompt to skip prefill. Persist on miss\n(qwen35 fused. Cache dir $ZIGFERENCER_CACHE_DIR)",
        },
        .thinking = args.Doc{
            .group = "thinking (qwen35)",
            .text = "thinking mode under --chat",
            .default = "auto, the model's own",
        },
    };
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: cli.Context) !void {
    const flags = try args.parseInto(Flags, ctx) orelse return;
    const prompt = flags.prompt;
    const max_tokens = flags.max_tokens;
    const chat = flags.chat;
    const prompt_cache_on = flags.prompt_cache;
    const gpu_layers = flags.engine.gpu_layers;
    if (gpu_layers != null and !build_options.use_vulkan_fused_ops) {
        return args.fail(ctx.io, "--gpu-layers requires -Dops-backend=vulkan_fused", .{});
    }
    if (flags.engine.thinking != .auto and !chat) {
        return args.fail(ctx.io, "--thinking requires --chat", .{});
    }
    // The whole-model fused runner runs no CPU op, so its pool starts no
    // workers. A pin of any layer count can run CPU layers.
    const policy = try serve_common.resolvePolicy(
        ctx.io,
        flags.engine.threads.policyFlags(),
        !(build_options.use_vulkan_fused_ops and gpu_layers == null),
    );
    // A name resolves directly. With no name, the command prompts over the
    // on-disk models on a TTY, or returns a usage error otherwise. Flag-shape
    // errors above already returned.
    const model = try select.model(ctx, flags.pub_name);
    const input: resolve_mod.Input = .{ .engine = flags.engine, .prompt_cache = prompt_cache_on };
    const entry = serve_common.entryOf(model);
    // Every refusal that needs no model file lands here, before the load.
    switch (resolve_mod.early(.run, input, entry)) {
        .ok => {},
        .refused => |r| return serve_common.refuse(ctx.io, model.pub_name, r),
    }
    const path = try registry.resolveLoadPath(ctx.arena, ctx.io, ctx.models_root, model.*);

    // loader.load emits the "loading.../loaded ... in N ms" trace (scope .gguf),
    // so every command that loads a model reports it identically.
    var loaded = try loader.load(ctx.io, ctx.arena, path);
    defer loaded.close();

    // One resolve per boot: the dtype, the window under the model ceiling, and
    // the all-host arm with its RAM quote. A device arm resolves again once its
    // context states the device.
    const model_facts = try serve_common.modelFacts(ctx.gpa, &loaded);
    const ram_free = serve_common.hostRamFree(ctx.io);
    const resolved = switch (resolve_mod.resolve(.run, input, entry, loaded.card, .{ .ram_free = ram_free, .model = model_facts })) {
        .ok => |r| r,
        .refused => |r| return serve_common.refuse(ctx.io, model.pub_name, r),
    };
    const chat_kind = resolved.early.chat_kind;
    const kv_dtype = resolved.early.kv_dtype;
    const window = resolved.window;
    const boot: serve_common.DeviceBoot = .{ .kind = .run, .input = input, .entry = entry, .card = loaded.card, .model = model_facts, .ram = .{ .read = ram_free } };

    logBackendLegend();

    // Only the qwen35 fused-GPU path (runQwenFused) honors --prompt-cache. Every
    // other arch and backend ignores it, and so does the all-host pin. The warning
    // is loud on purpose, because a scripted `--prompt-cache` on a CPU build must
    // not look like it worked. A pin of one layer or more, or auto, picks its
    // own arm, so it warns once the split resolves (`runQwenPlaced`).
    const qwen_gpu = loaded.card.arch == .qwen35 and build_options.use_vulkan_fused_ops;
    const zero_pin = if (gpu_layers) |g| g.isZeroCount() else false;
    if (prompt_cache_on and (!qwen_gpu or zero_pin)) {
        log.warn("--prompt-cache applies only to the qwen35 fused-GPU build. Ignored here", .{});
    }

    // One pool serves every CPU op of the command. Its workers live until
    // the command returns.
    var pool: Pool = undefined;
    try parallel.initPool(&pool, ctx.io, policy);
    defer pool.deinit();

    // The shared preamble of every runner path below: the vocab, the text that
    // goes to the tokenizer, the token ids, and the stdout stream. With --chat the
    // text wraps with the model's registered template, in the resolved thinking
    // mode. Without it the prompt is raw text to continue. Each path echoes
    // `prompt`, the user-visible text without the chat scaffolding, right before
    // prefill.
    var v = try vocab_mod.buildFromParsed(ctx.arena, loaded.parsed);
    const text = if (chat)
        try chat_mod.wrapAlloc(ctx.arena, chat_kind, prompt)
    else
        prompt;
    const prompt_ids = try bpe.encodeAlloc(&v, ctx.arena, text, true);
    if (prompt_ids.len == 0) {
        log.err("empty prompt", .{});
        return error.EmptyPrompt;
    }

    var stdout_buf: [4096]u8 = undefined;
    var stdout_fw: Io.File.Writer = .initStreaming(.stdout(), ctx.io, &stdout_buf);
    const out = &stdout_fw.interface;

    // One precision witness per boot, ahead of the arch dispatch, so every arm
    // reports the same map right above its own plan or placement line.
    serve_common.logPrecision(&loaded, kv_dtype);

    // The three-way selection, the same for both families. No flag runs the
    // whole-model fused path. A zero pin runs the all-host CPU runner with no
    // device context. A pin of one layer or more, or auto, prices a split
    // against the device, and an auto split that funds no GPU layer comes back
    // here for the all-host CPU runner.
    var host = resolved.cpu;
    if (loaded.card.arch == .qwen35) {
        if (comptime build_options.use_vulkan_fused_ops) {
            const g = gpu_layers orelse
                return runQwenFused(ctx, &boot, &loaded, model.*, &v, out, prompt, prompt_ids, max_tokens, prompt_cache_on, kv_dtype, null);
            if (!g.isZeroCount()) {
                switch (try runQwenPlaced(ctx, &boot, &loaded, model.*, &v, out, prompt, prompt_ids, max_tokens, g.pin(), prompt_cache_on, &pool, policy, kv_dtype)) {
                    .done => return,
                    .all_host => |c| host = c,
                }
            } else {
                log.info("--gpu-layers 0: the qwen CPU runner decodes every layer", .{});
            }
        }
        return runQwen(ctx, &boot, host, &loaded, &v, out, prompt, prompt_ids, max_tokens, &pool, policy, kv_dtype, window);
    }

    if (comptime build_options.use_vulkan_fused_ops) {
        const g = gpu_layers orelse
            return runLlamaFused(ctx, &boot, &loaded, &v, out, prompt, prompt_ids, max_tokens, kv_dtype, null);
        if (!g.isZeroCount()) {
            switch (try runLlamaPlaced(ctx, &boot, &loaded, &v, out, prompt, prompt_ids, max_tokens, g.pin(), &pool, policy, kv_dtype)) {
                .done => return,
                .all_host => |c| host = c,
            }
        } else {
            log.info("--gpu-layers 0: the llama CPU runner decodes every layer", .{});
        }
    }
    // The host arm states the placement line of the plan the CPU runner takes.
    const run_host = resolve_mod.runHost(loaded.card, model_facts, kv_dtype, window);
    logPlacement(run_host.report.?);
    serve_common.logSessionWindow(window);
    return runLlama(ctx, &boot, host, &loaded, &v, out, prompt, prompt_ids, max_tokens, run_host.plan, &pool, policy, kv_dtype, window);
}

// -- helpers ----------------------------------------------------------------

/// A device context and the fused arm a placement resolved on it. A placement
/// that funds the whole model hands both to the fused path, so a process opens
/// one device context and reads the VRAM figure once.
const Handed = if (build_options.use_vulkan_fused_ops) struct {
    ctxp: *vulkan_ctx_mod.Context,
    fused: resolve_mod.device_arm.Fused,
} else void;

/// Llama-family CPU forward. It drives a `LlamaRunner` over the mmap weights and
/// the host KV slab, the twin of `runQwen`. Every GPU arm returned before this,
/// so the per-op backend still needs its process-wide bundle up here.
fn runLlama(ctx: cli.Context, boot: *const serve_common.DeviceBoot, arm: ?resolve_mod.Result(resolve_mod.CpuArm), loaded: *const loader.Loaded, v: *vocab_mod.Vocab, out: *Io.Writer, prompt: []const u8, prompt_ids: []const u32, max_tokens: u32, plan: kv_budget.KvPlan, pool: *Pool, policy: parallel.Policy, kv_dtype: dtype.KvDtype, window: serve_common.SessionWindow) !void {
    // The per-op GPU backend needs a process-wide Runner and weight cache. The
    // init happens here, so the weight uploads land once and stay up for the whole
    // decode.
    if (comptime build_options.use_vulkan_ops) {
        try vulkan_ops_wrap.initBackend(ctx.arena, .{ .storage_range_override = ctx.vk_storage_range });
    }
    defer if (comptime build_options.use_vulkan_ops) {
        vulkan_ops_wrap.deinitBackend();
    };

    std.debug.assert(plan.window == window.base);
    try hostQuote(ctx, boot, arm, kv_dtype, window);
    const t_init = Io.Clock.Timestamp.now(ctx.io, .awake);
    var runner = try LlamaRunner.init(
        ctx.arena,
        loaded,
        plan,
        limits.cpu_prefill_chunk, // max_rows: one prompt chunk of slot 0
        ctx.io,
        pool,
        policy,
        kv_dtype,
    );
    log.info("llama cpu runner ready in {d:.0} ms", .{time.msBetween(t_init, Io.Clock.Timestamp.now(ctx.io, .awake))});

    try prefillAndStream(ctx.io, out, ctx.arena, v, &runner, runner.max_seq, prompt, prompt_ids, max_tokens);
}

/// Fused GPU decode for the Llama family (-Dops-backend=vulkan_fused): one command
/// buffer and one submit per token, with all state in VRAM. This is the Llama
/// analogue of `runQwenFused`. The CLI owns an explicit Context and LlamaFusedRunner
/// here, so the forward pass needs no process-wide singleton. The `LlamaRunner`
/// therefore stays a pure CPU path.
fn runLlamaFused(ctx: cli.Context, boot: *const serve_common.DeviceBoot, loaded: *const loader.Loaded, v: *vocab_mod.Vocab, out: *Io.Writer, prompt: []const u8, prompt_ids: []const u32, max_tokens: u32, kv_dtype: dtype.KvDtype, handed: ?Handed) !void {
    // The arena holds the Context, so its address stays stable for the lifetime of
    // the LlamaFusedRunner. The kernels capture `&ctx` at init. Teardown is LIFO: the
    // runner and its kernels go down before the device.
    log.info("initializing GPU: context + weight upload to VRAM...", .{});
    const t_init = Io.Clock.Timestamp.now(ctx.io, .awake);
    const ctxp = if (handed) |h| h.ctxp else blk: {
        const c = try ctx.arena.create(vulkan_ctx_mod.Context);
        c.* = try vulkan_ctx_mod.Context.init(ctx.arena, .{ .storage_range_override = ctx.vk_storage_range });
        break :blk c;
    };
    defer ctxp.deinit();
    const f = if (handed) |h| h.fused else try serve_common.fusedOn(ctx.io, boot, ctxp);
    _ = try serve_common.logFused(ctx.io, boot.entry.pub_name, f, true, false, kv_dtype);
    // The placement line prices the device at the window the fit planned.
    logPlacement(f.report.?);
    const fused = try llama_fused.LlamaFusedRunner.initFromLoaded(ctx.gpa, ctxp, loaded, f.plan, kv_dtype);
    defer fused.deinit();
    log.info("llama gpu runner ready in {d:.0} ms", .{time.msBetween(t_init, Io.Clock.Timestamp.now(ctx.io, .awake))});

    try prefillAndStream(ctx.io, out, ctx.arena, v, fused, fused.max_seq, prompt, prompt_ids, max_tokens);
}

/// Placed dispatch (--gpu-layers N > 0, or auto at a null `pin`). Create the
/// device context and resolve the split on it, with a loud refusal. Then run
/// the runner the split selects. Split 0 selects the fused runner, and every
/// other split selects the hybrid, except that an auto split with no GPU layer
/// releases the device and returns `.all_host` with the host arm. The caller
/// then runs the CPU arm of `--gpu-layers 0`.
fn runLlamaPlaced(ctx: cli.Context, boot: *const serve_common.DeviceBoot, loaded: *const loader.Loaded, v: *vocab_mod.Vocab, out: *Io.Writer, prompt: []const u8, prompt_ids: []const u32, max_tokens: u32, pin: ?u32, pool: *Pool, policy: parallel.Policy, kv_dtype: dtype.KvDtype) !Placed {
    const n_layers = loaded.card.n_layers;
    try refusePastModel(ctx, pin, n_layers);
    const ctxp = try ctx.arena.create(vulkan_ctx_mod.Context);
    ctxp.* = try vulkan_ctx_mod.Context.init(ctx.arena, .{ .storage_range_override = ctx.vk_storage_range });
    const dev = boot.resolveOn(ctx.io, ctxp) catch |err| {
        ctxp.deinit();
        return err;
    };
    const p = dev.placed.?;
    if (p.clamp) |c| serve_common.logClamp(c);
    if (p.split) |s| if (pin == null) serve_common.logAutoSplit(s.split, n_layers);
    const arm = switch (dev.arm) {
        .ok => |a| a,
        .refused => |r| {
            ctxp.deinit();
            return serve_common.refuse(ctx.io, boot.entry.pub_name, r);
        },
    };
    const h = switch (arm) {
        .cpu => |c| {
            log.info("--gpu-layers auto: the llama CPU runner decodes every layer", .{});
            ctxp.deinit();
            return .{ .all_host = c };
        },
        .fused => |f| {
            // The device clamp already ran, so it does not log a second time.
            try runLlamaFused(ctx, boot, loaded, v, out, prompt, prompt_ids, max_tokens, kv_dtype, .{ .ctxp = ctxp, .fused = f });
            return .done;
        },
        .hybrid => |h| h,
    };
    switch (h.quote) {
        .ok => |q| serve_common.logRamQuote(q, false),
        .refused => |r| {
            ctxp.deinit();
            return serve_common.refuse(ctx.io, boot.entry.pub_name, r);
        },
    }
    const placement = h.plan.?;
    logPlacement(placement);

    // A placement prices its KV at the window the device limits leave, so the
    // hybrid runs it unchanged.
    serve_common.logSessionWindow(h.window);
    log.info("initializing GPU: context + suffix weight upload to VRAM...", .{});
    const t_init = Io.Clock.Timestamp.now(ctx.io, .awake);
    var ledger_buf: [kv_budget.placement_ledger_text_max]u8 = undefined;
    log.info("{s}", .{kv_budget.formatLedger(placement, &ledger_buf) catch unreachable});
    const hybrid = llama_hybrid_mod.LlamaHybridRunner.init(ctx.gpa, ctxp, loaded, placement, ctx.io, pool, policy, kv_dtype) catch |e| {
        ctxp.deinit();
        return e;
    };
    defer {
        hybrid.deinit();
        ctxp.deinit();
    }
    log.info("llama hybrid runner ready in {d:.0} ms", .{time.msBetween(t_init, Io.Clock.Timestamp.now(ctx.io, .awake))});

    try prefillAndStream(ctx.io, out, ctx.arena, v, hybrid, hybrid.max_seq, prompt, prompt_ids, max_tokens);
    return .done;
}

/// A pin past the layer count of the model needs no device, so it refuses
/// before the context opens.
fn refusePastModel(ctx: cli.Context, pin: ?u32, n_layers: u32) error{UsageError}!void {
    const n_gpu = pin orelse return;
    // The zero pin took the CPU arm, so a pin here names a real GPU suffix.
    std.debug.assert(n_gpu >= 1);
    if (n_gpu > n_layers) {
        return args.fail(ctx.io, "--gpu-layers {d} exceeds the model's {d} layers", .{ n_gpu, n_layers });
    }
}

/// Log one placement in span form. The line names which layers a device
/// budget can hold.
fn logPlacement(placement: kv_budget.PlacementPlan) void {
    var buf: [kv_budget.placement_text_max]u8 = undefined;
    // The buffer holds the widest span the layer limit allows.
    log.info("{s}", .{kv_budget.formatPlacement(placement, &buf) catch unreachable});
}

/// Qwen3.5 hybrid (Gated-DeltaNet) CPU forward. It mirrors `run` and drives a
/// `QwenRunner`, the CPU coherence reference. The fused GPU path is separate.
fn runQwen(ctx: cli.Context, boot: *const serve_common.DeviceBoot, arm: ?resolve_mod.Result(resolve_mod.CpuArm), loaded: *const loader.Loaded, v: *vocab_mod.Vocab, out: *Io.Writer, prompt: []const u8, prompt_ids: []const u32, max_tokens: u32, pool: *Pool, policy: parallel.Policy, kv_dtype: dtype.KvDtype, window: serve_common.SessionWindow) !void {
    // Under the per-op GPU backend (-Dops-backend=vulkan) the QwenRunner ops
    // resolve to ops_vulkan/wrap, which needs its process-wide Context and kernels
    // up. The Llama path inits them after the qwen dispatch, so qwen does it here.
    if (comptime build_options.use_vulkan_ops) {
        try vulkan_ops_wrap.initBackend(ctx.arena, .{ .storage_range_override = ctx.vk_storage_range });
    }
    defer if (comptime build_options.use_vulkan_ops) {
        vulkan_ops_wrap.deinitBackend();
    };

    // The host holds no device budget, so the base window stands.
    serve_common.logSessionWindow(window);
    const plan = kv_budget.pinned(1, window.base);
    try hostQuote(ctx, boot, arm, kv_dtype, window);
    const t_init = Io.Clock.Timestamp.now(ctx.io, .awake);
    var qr = try QwenRunner.init(ctx.arena, loaded, plan, limits.cpu_prefill_chunk, ctx.io, pool, policy, kv_dtype);
    log.info("qwen cpu runner ready in {d:.0} ms", .{time.msBetween(t_init, Io.Clock.Timestamp.now(ctx.io, .awake))});

    try prefillAndStream(ctx.io, out, ctx.arena, v, &qr, qr.max_seq, prompt, prompt_ids, max_tokens);
}

/// Fused GPU decode for Qwen3.5 (-Dops-backend=vulkan_fused): one command buffer
/// and one submit per token, with all state in VRAM. See `runtime/qwen_fused.zig`.
/// This path takes `model`, unlike its siblings, because the prompt cache keys on
/// the pub_name.
fn runQwenFused(ctx: cli.Context, boot: *const serve_common.DeviceBoot, loaded: *const loader.Loaded, model: registry.Model, v: *vocab_mod.Vocab, out: *Io.Writer, prompt: []const u8, prompt_ids: []const u32, max_tokens: u32, prompt_cache_on: bool, kv_dtype: dtype.KvDtype, handed: ?Handed) !void {
    log.info("initializing GPU: context + weight upload to VRAM...", .{});
    const t_init = Io.Clock.Timestamp.now(ctx.io, .awake);
    const ctxp = if (handed) |h| h.ctxp else blk: {
        const c = try ctx.arena.create(vulkan_ctx_mod.Context);
        c.* = try vulkan_ctx_mod.Context.init(ctx.arena, .{ .storage_range_override = ctx.vk_storage_range });
        break :blk c;
    };
    defer ctxp.deinit();
    const f = if (handed) |h| h.fused else try serve_common.fusedOn(ctx.io, boot, ctxp);
    _ = try serve_common.logFused(ctx.io, boot.entry.pub_name, f, true, false, kv_dtype);
    const plan = f.plan;
    const qr = try qwen_fused.QwenFusedRunner.init(ctx.gpa, ctxp, loaded, plan, kv_dtype);
    defer qr.deinit();
    log.info("qwen gpu runner ready in {d:.0} ms", .{time.msBetween(t_init, Io.Clock.Timestamp.now(ctx.io, .awake))});

    if (prompt_ids.len > qr.max_seq) {
        log.err("prompt has {d} tokens, the context limit is {d}", .{ prompt_ids.len, qr.max_seq });
        return error.PromptTooLong;
    }
    try out.writeAll(prompt);
    try out.flush();

    // A prompt-cache hit restores the end-of-prompt snapshot and skips prefill. A
    // miss prefills, then keeps the snapshot in RAM, and on disk too when the
    // per-model `worthwhile` gate clears, for a later cross-process re-run.
    var cache: ?prompt_cache_mod.PromptCache = null;
    var cache_key: u64 = 0;
    if (prompt_cache_on) {
        // The fused runner writes these blobs, so they live in the namespace that
        // the fused arm of serve reads. A fingerprint qualified by the runner
        // family prevents a blob restore across families.
        const fp = prompt_cache_mod.backendFingerprint(
            prompt_cache_mod.modelFingerprint(model.pub_name, loaded.map.bytes().len),
            "vulkan_fused",
        );
        cache_key = prompt_cache_mod.prefixKey(fp, prompt_ids);
        cache = prompt_cache_mod.PromptCache.init(ctx.gpa, limits.prompt_cache_ram_default);
        try cache.?.enableDisk(ctx.io, ctx.cache_dir, limits.prompt_cache_disk_default, fp);
        log.info("prompt-cache dir: {s} (set $ZIGFERENCER_CACHE_DIR to relocate)", .{ctx.cache_dir});
    }
    defer if (cache) |*c| c.deinit();

    const prompt_len: u32 = @intCast(prompt_ids.len);
    var pos: u32 = 0;
    var last_logits: []f32 = undefined;
    const t_start = Io.Clock.Timestamp.now(ctx.io, .awake);

    if (cache) |*c| {
        if (c.lookup(cache_key, prompt_ids)) |hit| {
            pos = try qr.restoreFromBytes(hit.blob, hit.n);
            last_logits = hit.last_logits;
            log.info("prompt-cache HIT ({s}): skipped prefill of {d} tokens", .{ if (c.last_hit_from_disk) "disk" else "ram", prompt_len });
        }
    }

    if (pos == 0) {
        // Miss (or cache off): batched prefill, chunked by prefill_batch_max.
        const t0 = Io.Clock.Timestamp.now(ctx.io, .awake);
        var off: usize = 0;
        while (off < prompt_ids.len) {
            const chunk: u32 = @intCast(@min(prompt_ids.len - off, qr.prefill_batch_max));
            const sp = trace_span.begin(.prefill_chunk);
            last_logits = try qr.prefillBatch(prompt_ids[off..][0..chunk], pos);
            sp.end();
            pos += chunk;
            off += chunk;
        }
        if (cache) |*c| {
            const prefill_ns = time.elapsedNs(t0, Io.Clock.Timestamp.now(ctx.io, .awake));
            const s0 = Io.Clock.Timestamp.now(ctx.io, .awake);
            // The cache copies the snapshot on admit, so the buffer lives for
            // this block only.
            const blob = try ctx.gpa.alloc(u8, qr.snapshotBytes(prompt_len));
            defer ctx.gpa.free(blob);
            try qr.snapshotInto(prompt_len, blob);
            const save_ns = time.elapsedNs(s0, Io.Clock.Timestamp.now(ctx.io, .awake));
            const last_dup = try ctx.arena.dupe(f32, last_logits);
            _ = try c.admit(cache_key, prompt_ids, blob, last_dup, prefill_ns, save_ns);
            log.info("prompt-cache MISS: prefilled + cached (ram={} disk={})", .{ c.last_admit_ram, c.last_admit_disk });
        }
    }

    try streamDecode(ctx.io, out, ctx.arena, v, qr, qr.max_seq, last_logits, pos, max_tokens, t_start);
}

/// Placed dispatch for Qwen3.5 (--gpu-layers N > 0, or auto), the twin of
/// `runLlamaPlaced`. Create the device context, then resolve the split on it,
/// with a loud refusal. Then run the runner the split selects. Split 0 selects
/// the fused runner, an auto split with no GPU layer returns `.all_host`, and
/// every other split selects the hybrid. The two layer kinds of the qwen stack
/// cost different amounts, so the split prices the per-layer cost vector.
fn runQwenPlaced(
    ctx: cli.Context,
    boot: *const serve_common.DeviceBoot,
    loaded: *const loader.Loaded,
    model: registry.Model,
    v: *vocab_mod.Vocab,
    out: *Io.Writer,
    prompt: []const u8,
    prompt_ids: []const u32,
    max_tokens: u32,
    pin: ?u32,
    prompt_cache_on: bool,
    pool: *Pool,
    policy: parallel.Policy,
    kv_dtype: dtype.KvDtype,
) !Placed {
    const n_layers = loaded.card.n_layers;
    try refusePastModel(ctx, pin, n_layers);
    const ctxp = try ctx.arena.create(vulkan_ctx_mod.Context);
    ctxp.* = try vulkan_ctx_mod.Context.init(ctx.arena, .{ .storage_range_override = ctx.vk_storage_range });
    const dev = boot.resolveOn(ctx.io, ctxp) catch |err| {
        ctxp.deinit();
        return err;
    };
    const p = dev.placed.?;
    if (p.clamp) |c| serve_common.logClamp(c);
    if (p.split) |s| {
        logPlacement(s);
        if (pin == null) serve_common.logAutoSplit(s.split, n_layers);
    }
    const arm = switch (dev.arm) {
        .ok => |a| a,
        .refused => |r| {
            ctxp.deinit();
            return serve_common.refuse(ctx.io, boot.entry.pub_name, r);
        },
    };
    if (arm == .fused) {
        // The split funds the whole model, so the fused runner takes over this
        // context. The device clamp already ran, so it does not log twice.
        try runQwenFused(ctx, boot, loaded, model, v, out, prompt, prompt_ids, max_tokens, prompt_cache_on, kv_dtype, .{ .ctxp = ctxp, .fused = arm.fused });
        return .done;
    }
    // The prompt cache keys on a whole-model fused snapshot, so a split blob
    // restores the wrong bytes. The warning is loud rather than silent.
    if (prompt_cache_on) {
        log.warn("--prompt-cache applies to the whole-model qwen35 fused path. This placement ignores it", .{});
    }
    const h = switch (arm) {
        .cpu => |c| {
            log.info("--gpu-layers auto: the qwen CPU runner decodes every layer", .{});
            ctxp.deinit();
            return .{ .all_host = c };
        },
        .fused => unreachable,
        .hybrid => |h| h,
    };
    switch (h.quote) {
        .ok => |q| serve_common.logRamQuote(q, false),
        .refused => |r| {
            ctxp.deinit();
            return serve_common.refuse(ctx.io, boot.entry.pub_name, r);
        },
    }
    const placement = h.plan.?;
    serve_common.logSessionWindow(h.window);

    log.info("initializing GPU: context + suffix weight upload to VRAM...", .{});
    const t_init = Io.Clock.Timestamp.now(ctx.io, .awake);
    var ledger_buf: [kv_budget.placement_ledger_text_max]u8 = undefined;
    log.info("{s}", .{kv_budget.formatLedger(placement, &ledger_buf) catch unreachable});
    const hybrid = qwen_hybrid_mod.QwenHybridRunner.init(ctx.gpa, ctxp, loaded, placement, ctx.io, pool, policy, kv_dtype) catch |e| {
        ctxp.deinit();
        return e;
    };
    defer {
        hybrid.deinit();
        ctxp.deinit();
    }
    log.info("qwen hybrid runner ready in {d:.0} ms", .{time.msBetween(t_init, Io.Clock.Timestamp.now(ctx.io, .awake))});

    try prefillAndStream(ctx.io, out, ctx.arena, v, hybrid, hybrid.max_seq, prompt, prompt_ids, max_tokens);
    return .done;
}

/// The RAM quote line of the all-host arm, or its refusal. A per-op vulkan
/// boot resolved a device arm, so it resolves the host arm here.
fn hostQuote(ctx: cli.Context, boot: *const serve_common.DeviceBoot, arm: ?resolve_mod.Result(resolve_mod.CpuArm), kv_dtype: dtype.KvDtype, window: serve_common.SessionWindow) !void {
    const result = arm orelse
        resolve_mod.cpuArm(.run, boot.input, boot.card, kv_dtype, window, .{ .ram_free = boot.ramFree(ctx.io), .model = boot.model });
    switch (result) {
        .ok => |a| serve_common.logRamQuote(a.quote, false),
        .refused => |r| return serve_common.refuse(ctx.io, boot.entry.pub_name, r),
    }
}

/// Echo the prompt, prefill it, then decode through `streamDecode`. This is the
/// shared tail of the uncached run paths. A runner that declares
/// `chunked_prefill` prefills in chunks of `prefill_batch_max`. The per-op GPU
/// runners prefill token by token, which keeps their GPU token hashes. `runQwenFused`
/// interleaves its prompt cache with a batched prefill, so it keeps its own
/// loop. The function is generic over the runner, like `streamDecode`.
fn prefillAndStream(
    io: Io,
    out: *Io.Writer,
    arena: Allocator,
    v: *vocab_mod.Vocab,
    runner: anytype,
    max_seq: u32,
    prompt: []const u8,
    prompt_ids: []const u32,
    max_tokens: u32,
) !void {
    if (prompt_ids.len > max_seq) {
        log.err("prompt has {d} tokens, the context limit is {d}", .{ prompt_ids.len, max_seq });
        return error.PromptTooLong;
    }
    try out.writeAll(prompt);
    try out.flush();
    const t_start = Io.Clock.Timestamp.now(io, .awake);
    var pos: u32 = 0;
    var last_logits: []f32 = undefined;
    if (comptime step_row.chunkedPrefill(@TypeOf(runner))) {
        var off: usize = 0;
        while (off < prompt_ids.len) {
            const chunk: u32 = @intCast(@min(prompt_ids.len - off, runner.prefill_batch_max));
            const sp = trace_span.begin(.prefill_chunk);
            last_logits = try runner.prefillBatch(prompt_ids[off..][0..chunk], pos);
            sp.end();
            pos += chunk;
            off += chunk;
        }
    } else {
        for (prompt_ids) |tid| {
            const sp = trace_span.begin(.prefill_chunk);
            last_logits = try runner.step(tid, pos);
            sp.end();
            pos += 1;
        }
    }
    std.debug.assert(pos == prompt_ids.len);
    try streamDecode(io, out, arena, v, runner, max_seq, last_logits, pos, max_tokens, t_start);
}

/// Decode up to `max_tokens` greedily from `last_logits`, stream each detokenized
/// piece to `out` on stdout, then log a one-line TTFT and ITL summary. The
/// function is generic over the runner. The CPU `LlamaRunner`, the `QwenRunner`, and a
/// GPU fused runner all expose `step(id, pos)`. The caller captures `t_start`
/// before prefill, so TTFT spans prefill and the first token, the way a reader
/// feels the latency.
fn streamDecode(
    io: Io,
    out: *Io.Writer,
    arena: Allocator,
    v: *vocab_mod.Vocab,
    runner: anytype,
    max_seq: u32,
    last_logits_in: []f32,
    pos_in: u32,
    max_tokens: u32,
    t_start: Io.Clock.Timestamp,
) !void {
    log.info("generating up to {d} tokens (prompt {d} tokens)", .{ max_tokens, pos_in });
    var sink: StreamSink = .{ .out = out, .v = v, .arena = arena };
    const g = try generate.decode(runner, last_logits_in, pos_in, max_tokens, max_seq, v.eos_id, false, io, t_start, &sink);
    try out.writeAll("\n");
    try out.flush();

    if (g.tokens_generated > 1) {
        log.info("{d} tokens | TTFT {d:.1} ms | ITL {d:.1} ms = {d:.1} tok/s", .{ g.tokens_generated, g.ttft_ms, g.itl_ms, 1000.0 / g.itl_ms });
    } else {
        log.info("{d} tokens | TTFT {d:.1} ms", .{ g.tokens_generated, g.ttft_ms });
    }
}

/// Log a one-line legend of the op backends: every variant the engine can build,
/// with the one linked into this binary marked `[+]`. The backend is a build-time
/// choice, so a switch needs a rebuild. `make info` adds per-host eligibility.
fn logBackendLegend() void {
    var buf: [128]u8 = undefined;
    var n: usize = 0;
    inline for (op_requires.ops_backend_names) |name| {
        const seg = if (std.mem.eql(u8, name, build_options.ops_backend))
            (std.fmt.bufPrint(buf[n..], " [+]{s}", .{name}) catch return)
        else
            (std.fmt.bufPrint(buf[n..], " {s}", .{name}) catch return);
        n += seg.len;
    }
    log.info("backend:{s}  ([+] active. Rebuild: -Dops-backend=<name> [+ -Druntime=vulkan for GPU])", .{buf[0..n]});
}
