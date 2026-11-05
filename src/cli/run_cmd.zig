//! `zigferencer run <pub_name> --prompt "..." --max-tokens N`: the full pipeline.
//! Load the model, build the vocab and the runner of its family, prefill the prompt,
//! and stream the decoded text to stdout. Decoding is greedy, so the output is
//! deterministic. `--chat` wraps the prompt with the chat template of the model.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const build_options = @import("build_options");

const cli = @import("command.zig");
const args = @import("args.zig");
const flags_mod = @import("flags.zig");
const select = @import("select.zig");
const serve_common = @import("serve_common.zig");
const family = @import("../engine/family.zig");
const resolve_mod = @import("../engine/resolve.zig");
const bpe = @import("../core/tokenizer/bpe.zig");
const chat_mod = @import("../core/chat.zig");
const dtype = @import("../core/dtype.zig");
const kv_budget = @import("../core/kv_budget.zig");
const limits = @import("../core/limits.zig");
const op_requires = @import("../core/op_requires.zig");
const registry = @import("../core/registry.zig");
const trace_span = @import("../core/span.zig");
const time = @import("../core/time.zig");
const vocab_mod = @import("../core/tokenizer/vocab.zig");
const loader = @import("../io/gguf/loader.zig");
const generate = @import("../runtime/generate.zig");
const parallel = @import("../runtime/parallel.zig");
const Pool = @import("../runtime/pool.zig").Pool;
const prompt_cache_mod = @import("../runtime/prompt_cache.zig");
const step_row = @import("../runtime/step_row.zig");

const log = std.log.scoped(.run);

const fused_build = build_options.use_vulkan_fused_ops;
const per_op_build = build_options.use_vulkan_ops;

// The device paths resolve to `void` on a build without them.
const vulkan_ops_wrap = if (per_op_build) @import("../ops_vulkan/wrap.zig") else void;
const vulkan_ctx_mod = if (fused_build) @import("../runtime/vulkan_ctx.zig") else void;

const device_arm = resolve_mod.device_arm;
const PromptCache = prompt_cache_mod.PromptCache;

// -- constants --------------------------------------------------------------

/// Bytes of the backend legend: every backend name with its marker.
const legend_text_max = 128;

// -- types ------------------------------------------------------------------

/// The per-token effect of `run`: detokenize the id and stream the piece. This
/// sink never prints EOS, because EOS is a boundary marker, not text.
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

/// The flags of `run`.
const Flags = struct {
    pub_name: ?[]const u8 = null,
    prompt: []const u8 = "Hello",
    max_tokens: u32 = 32,
    chat: bool = false,
    /// A hit restores the snapshot of the prompt and skips prefill. A miss keeps
    /// the end-of-prompt boundary, on disk too, for a later process.
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
            .text = "reuse a cached prompt to skip prefill. Persist on miss\n" ++
                "(qwen35 fused. Cache dir $ZIGFERENCER_CACHE_DIR)",
        },
        .thinking = args.Doc{
            .group = "thinking (qwen35)",
            .text = "thinking mode under --chat",
            .default = "auto, the model's own",
        },
    };
};

/// What one boot of `run` holds once the model loaded.
const Boot = struct {
    ctx: cli.Context,
    flags: *const Flags,
    policy: parallel.Policy,
    model: *const registry.Model,
    loaded: *const loader.Loaded,
    resolved: *const resolve_mod.Resolved,
    device: serve_common.DeviceBoot,

    fn kvDtype(self: *const Boot) dtype.KvDtype {
        return self.resolved.early.kv_dtype;
    }
};

/// One turn: the boot, the prompt ids, and where the text goes. The ids carry the
/// chat scaffolding, and `boot.flags.prompt` is the text the turn echoes.
const Turn = struct {
    boot: *const Boot,
    pool: *Pool,
    v: *vocab_mod.Vocab,
    out: *Io.Writer,
    prompt_ids: []const u32,
};

/// The end of a prefill: the next position and the logits of the last row.
const Prefilled = struct {
    pos: u32,
    last_logits: []f32,
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: cli.Context) !void {
    const flags = try args.parseInto(Flags, ctx) orelse return;
    if (flags.engine.gpu_layers != null and !fused_build) {
        return args.fail(ctx.io, "--gpu-layers requires -Dops-backend=vulkan_fused", .{});
    }
    if (flags.engine.thinking != .auto and !flags.chat) {
        return args.fail(ctx.io, "--thinking requires --chat", .{});
    }

    // The whole-model fused runner runs no CPU op, so its pool starts no
    // workers. A pin of any layer count can run CPU layers.
    const runs_cpu_ops = !fused_build or flags.engine.gpu_layers != null;
    const thread_flags = flags.engine.threads.policyFlags();
    const policy = try serve_common.resolvePolicy(ctx.io, thread_flags, runs_cpu_ops);

    const model = try select.model(ctx, flags.pub_name);
    const input: resolve_mod.Input = .{
        .engine = flags.engine,
        .prompt_cache = flags.prompt_cache,
    };
    const entry = serve_common.entryOf(model);

    // Every refusal that needs no model file lands here, before the load.
    switch (resolve_mod.early(.run, input, entry)) {
        .ok => {},
        .refused => |r| return serve_common.refuse(ctx.io, model.pub_name, r),
    }

    const path = try registry.resolveLoadPath(ctx.arena, ctx.io, ctx.models_root, model.*);
    var loaded = try loader.load(ctx.io, ctx.arena, path);
    defer loaded.close();

    // A device arm resolves again once its context states the device.
    const resolved = try serve_common.resolveLoaded(ctx, .run, input, entry, &loaded);

    const boot: Boot = .{
        .ctx = ctx,
        .flags = &flags,
        .policy = policy,
        .model = model,
        .loaded = &loaded,
        .resolved = &resolved.values,
        .device = resolved.device,
    };

    switch (loaded.card.arch) {
        inline else => |arch| return runFamily(family.Of(arch), &boot),
    }
}

// -- helpers: selection -----------------------------------------------------

/// Tokenize the prompt, then pick the placement. No flag runs the whole-model fused
/// path. A zero pin runs the CPU runner with no device context. A pin of one layer
/// or more, or auto, prices a split, and a split with no GPU layer comes back here.
fn runFamily(comptime Family: type, b: *const Boot) !void {
    const ctx = b.ctx;
    const gpu_layers = b.flags.engine.gpu_layers;

    logBackendLegend();
    warnPromptCache(Family, b.flags);

    // One pool serves every CPU op of the command.
    var pool: Pool = undefined;
    try parallel.initPool(&pool, ctx.io, b.policy);
    defer pool.deinit();

    var v = try vocab_mod.buildFromParsed(ctx.arena, b.loaded.parsed);
    const prompt_ids = try promptIds(b, &v);

    var stdout_buffer: [cli.stdout_buffer_bytes]u8 = undefined;
    var stdout_writer: Io.File.Writer = .initStreaming(.stdout(), ctx.io, &stdout_buffer);

    serve_common.logPrecision(b.loaded, b.kvDtype());

    const turn: Turn = .{
        .boot = b,
        .pool = &pool,
        .v = &v,
        .out = &stdout_writer.interface,
        .prompt_ids = prompt_ids,
    };

    var host = b.resolved.cpu;
    if (comptime fused_build) {
        const layers = gpu_layers orelse return runFused(Family, &turn);
        if (layers.isZeroCount()) {
            log.info("--gpu-layers 0: the {s} CPU runner decodes every layer", .{Family.name});
        } else {
            switch (try runPlaced(Family, &turn, layers.pin())) {
                .done => return,
                .all_host => |arm| host = arm,
            }
        }
    }

    return runHost(Family, &turn, host);
}

/// A scripted `--prompt-cache` on a build that ignores it must not look like it
/// worked. A pin of one layer or more, or auto, warns once the split resolves.
fn warnPromptCache(comptime Family: type, flags: *const Flags) void {
    if (!flags.prompt_cache) return;

    const honored = comptime Family.prompt_cache and fused_build;
    const zero_pin = if (flags.engine.gpu_layers) |g| g.isZeroCount() else false;
    if (!honored or zero_pin) {
        log.warn("--prompt-cache applies only to the qwen35 fused-GPU build. Ignored here", .{});
    }
}

/// The token ids of the turn. With `--chat` the text wraps with the template of
/// the model in the resolved thinking mode.
fn promptIds(b: *const Boot, v: *vocab_mod.Vocab) ![]const u32 {
    const arena = b.ctx.arena;
    const prompt = b.flags.prompt;

    const text = if (b.flags.chat)
        try chat_mod.wrapAlloc(arena, b.resolved.early.chat_kind, prompt)
    else
        prompt;

    const ids = try bpe.encodeAlloc(v, arena, text, true);
    if (ids.len == 0) {
        log.err("empty prompt", .{});
        return error.EmptyPrompt;
    }

    return ids;
}

// -- helpers: placements ----------------------------------------------------

/// The CPU runner, over the mmap weights and the host KV slab. Every GPU arm
/// returned before this, so the per-op backend starts its process-wide runner
/// here, and the weight uploads stay up for the whole decode.
fn runHost(
    comptime Family: type,
    t: *const Turn,
    arm: ?resolve_mod.Result(resolve_mod.CpuArm),
) !void {
    const b = t.boot;
    const ctx = b.ctx;
    const window = b.resolved.window;

    const plan = if (comptime Family.run_logs_every_placement)
        logHostPlan(b)
    else
        kv_budget.pinned(1, window.base);

    std.debug.assert(plan.window == window.base);

    if (comptime per_op_build) {
        try vulkan_ops_wrap.initBackend(ctx.arena, .{
            .storage_range_override = ctx.vk_storage_range,
        });
    }

    defer if (comptime per_op_build) {
        vulkan_ops_wrap.deinitBackend();
    };

    if (comptime !Family.run_logs_every_placement) {
        serve_common.logSessionWindow(window);
    }

    try hostQuote(b, arm);

    // One prompt chunk of slot 0 is the most rows a step holds.
    const t_init = Io.Clock.Timestamp.now(ctx.io, .awake);
    var runner = try Family.Cpu.init(
        ctx.arena,
        b.loaded,
        plan,
        limits.cpu_prefill_chunk,
        ctx.io,
        t.pool,
        b.policy,
        b.kvDtype(),
    );
    log.info("{s} cpu runner ready in {d:.0} ms", .{ Family.name, msSince(ctx.io, t_init) });

    return prefillAndStream(&runner, t);
}

/// The whole model on the device: one command buffer and one submit per token.
fn runFused(comptime Family: type, t: *const Turn) !void {
    const b = t.boot;
    const io = b.ctx.io;

    const t_init = logFusedInit(io);
    const ctxp = try serve_common.openDevice(b.ctx);
    defer ctxp.deinit();

    const f = try serve_common.fusedOn(io, &b.device, ctxp);
    return runFusedOn(Family, t, ctxp, f, t_init);
}

/// The fused runner on an open context. The runner goes down before the context.
fn runFusedOn(
    comptime Family: type,
    t: *const Turn,
    ctxp: *vulkan_ctx_mod.Context,
    f: device_arm.Fused,
    t_init: Io.Clock.Timestamp,
) !void {
    const b = t.boot;
    const ctx = b.ctx;
    const kv_dtype = b.kvDtype();

    _ = try serve_common.logFused(ctx.io, b.model.pub_name, f, true, false, kv_dtype);
    if (f.report) |report| {
        logPlacement(report);
    }

    const runner = try Family.initFused(ctx.gpa, ctxp, b.loaded, f.plan, kv_dtype);
    defer runner.deinit();

    log.info("{s} gpu runner ready in {d:.0} ms", .{ Family.name, msSince(ctx.io, t_init) });

    if (comptime Family.prompt_cache) return prefillCachedAndStream(runner, t);

    return prefillAndStream(runner, t);
}

/// `--gpu-layers N > 0`, or auto at a null `pin`. Open the device and resolve the
/// split on it, then run the runner the split selects. An auto split with no GPU
/// layer releases the device and returns `.all_host` with the host arm.
fn runPlaced(comptime Family: type, t: *const Turn, pin: ?u32) !serve_common.Placed {
    const b = t.boot;
    const ctx = b.ctx;
    const name = b.model.pub_name;
    const n_layers = b.loaded.card.n_layers;

    try refusePastModel(ctx, pin, n_layers);

    const ctxp = try serve_common.openDevice(ctx);
    defer ctxp.deinit();

    const dev = try b.device.resolveOn(ctx.io, ctxp);
    const placed = dev.placed.?;
    if (placed.clamp) |c| {
        serve_common.logClamp(c);
    }

    if (placed.split) |split| {
        if (comptime !Family.run_logs_every_placement) {
            logPlacement(split);
        }

        if (pin == null) {
            serve_common.logAutoSplit(split.split, n_layers);
        }
    }

    const arm = switch (dev.arm) {
        .ok => |a| a,
        .refused => |r| return serve_common.refuse(ctx.io, name, r),
    };

    if (arm == .fused) {
        // The split funds the whole model. The clamp logged, so it does not repeat.
        const t_init = logFusedInit(ctx.io);
        try runFusedOn(Family, t, ctxp, arm.fused, t_init);
        return .done;
    }

    // The prompt cache keys on a whole-model fused snapshot, so a split blob
    // restores the wrong bytes.
    if (comptime Family.prompt_cache) {
        if (b.flags.prompt_cache) {
            log.warn("--prompt-cache applies to the whole-model qwen35 fused path. " ++
                "This placement ignores it", .{});
        }
    }

    switch (arm) {
        .fused => unreachable,
        .cpu => |c| {
            log.info("--gpu-layers auto: the {s} CPU runner decodes every layer", .{Family.name});
            return .{ .all_host = c };
        },
        .hybrid => |h| {
            try runHybrid(Family, t, ctxp, h);
            return .done;
        },
    }
}

/// The hybrid runner at a resolved split. A placement prices its KV at the window
/// the device limits leave, so the hybrid runs it unchanged.
fn runHybrid(
    comptime Family: type,
    t: *const Turn,
    ctxp: *vulkan_ctx_mod.Context,
    h: device_arm.Hybrid,
) !void {
    const b = t.boot;
    const ctx = b.ctx;

    switch (h.quote) {
        .ok => |q| serve_common.logRamQuote(q, false),
        .refused => |r| return serve_common.refuse(ctx.io, b.model.pub_name, r),
    }

    const placement = h.plan.?;
    if (comptime Family.run_logs_every_placement) {
        logPlacement(placement);
    }

    serve_common.logSessionWindow(h.window);

    log.info("initializing GPU: context + suffix weight upload to VRAM...", .{});
    const t_init = Io.Clock.Timestamp.now(ctx.io, .awake);
    var ledger_buffer: [kv_budget.placement_ledger_text_max]u8 = undefined;
    log.info("{s}", .{kv_budget.formatLedger(placement, &ledger_buffer)});

    const hybrid = try Family.Hybrid.init(
        ctx.gpa,
        ctxp,
        b.loaded,
        placement,
        ctx.io,
        t.pool,
        b.policy,
        b.kvDtype(),
    );
    defer hybrid.deinit();

    log.info("{s} hybrid runner ready in {d:.0} ms", .{ Family.name, msSince(ctx.io, t_init) });

    return prefillAndStream(hybrid, t);
}

/// The placement line and the window line of the host plan, and the plan. The host
/// holds no device budget, so the base window stands.
fn logHostPlan(b: *const Boot) kv_budget.KvPlan {
    const window = b.resolved.window;
    const host = resolve_mod.runHost(b.loaded.card, b.device.model, b.kvDtype(), window);
    logPlacement(host.report.?);
    serve_common.logSessionWindow(window);
    return host.plan;
}

/// The RAM quote line of the all-host arm, or its refusal. A per-op vulkan boot
/// resolved a device arm, so it resolves the host arm here.
fn hostQuote(b: *const Boot, arm: ?resolve_mod.Result(resolve_mod.CpuArm)) !void {
    const io = b.ctx.io;
    const boot = &b.device;
    const facts: resolve_mod.Facts = .{ .ram_free = boot.ramFree(io), .model = boot.model };

    const result = arm orelse
        resolve_mod.cpuArm(.run, boot.input, boot.card, b.kvDtype(), b.resolved.window, facts);
    switch (result) {
        .ok => |a| serve_common.logRamQuote(a.quote, false),
        .refused => |r| return serve_common.refuse(io, boot.entry.pub_name, r),
    }
}

/// A pin past the layer count of the model needs no device, so it refuses
/// before the context opens.
fn refusePastModel(ctx: cli.Context, pin: ?u32, n_layers: u32) error{UsageError}!void {
    const n_gpu = pin orelse return;

    // The zero pin took the CPU arm, so a pin here names a real GPU suffix.
    std.debug.assert(n_gpu >= 1);
    if (n_gpu > n_layers) {
        const fmt = "--gpu-layers {d} exceeds the model's {d} layers";
        return args.fail(ctx.io, fmt, .{ n_gpu, n_layers });
    }
}

/// Log the first line of a fused boot, and return the start of its timing.
fn logFusedInit(io: Io) Io.Clock.Timestamp {
    log.info("initializing GPU: context + weight upload to VRAM...", .{});
    return Io.Clock.Timestamp.now(io, .awake);
}

/// Log one placement in span form. The buffer holds the widest span the layer
/// limit allows.
fn logPlacement(placement: kv_budget.PlacementPlan) void {
    var buffer: [kv_budget.placement_text_max]u8 = undefined;
    log.info("{s}", .{kv_budget.formatPlacement(placement, &buffer)});
}

/// Log the op backends the engine can build, and mark the one in this binary.
fn logBackendLegend() void {
    var buffer: [legend_text_max]u8 = undefined;
    var n: usize = 0;
    inline for (op_requires.ops_backend_names) |name| {
        const marker = if (std.mem.eql(u8, name, build_options.ops_backend)) "[+]" else "";
        const segment = std.fmt.bufPrint(buffer[n..], " {s}{s}", .{ marker, name }) catch return;
        n += segment.len;
    }

    log.info("backend:{s}  ([+] active. Rebuild: -Dops-backend=<name> " ++
        "[+ -Druntime=vulkan for GPU])", .{buffer[0..n]});
}

// -- helpers: the turn ------------------------------------------------------

/// Echo the prompt, prefill it, then decode. A runner without `chunked_prefill`
/// prefills token by token, which keeps the token hashes of the per-op GPU runners.
fn prefillAndStream(runner: anytype, t: *const Turn) !void {
    const ids = t.prompt_ids;
    try echoPrompt(t, runner.max_seq);

    // TTFT spans the prefill and the first token.
    const t_start = Io.Clock.Timestamp.now(t.boot.ctx.io, .awake);
    const prefilled = if (comptime step_row.chunkedPrefill(@TypeOf(runner)))
        try prefillChunks(runner, ids)
    else
        try prefillTokens(runner, ids);

    std.debug.assert(prefilled.pos == ids.len);

    return streamDecode(runner, t, prefilled, t_start);
}

/// `prefillAndStream` with the prompt cache interleaved. A hit restores the
/// end-of-prompt snapshot and skips prefill. A miss prefills and keeps the snapshot,
/// on disk too when the `worthwhile` gate of the cache clears.
fn prefillCachedAndStream(runner: anytype, t: *const Turn) !void {
    const b = t.boot;
    const ctx = b.ctx;
    const ids = t.prompt_ids;
    try echoPrompt(t, runner.max_seq);

    var cache: ?PromptCache = null;
    var cache_key: u64 = 0;
    if (b.flags.prompt_cache) {
        // The fused runner writes these blobs, so they live in the namespace that
        // the fused arm of serve reads.
        const fp = prompt_cache_mod.backendFingerprint(
            prompt_cache_mod.modelFingerprint(b.model.pub_name, b.loaded.map.bytes().len),
            "vulkan_fused",
        );
        cache_key = prompt_cache_mod.prefixKey(fp, ids);
        cache = PromptCache.init(ctx.gpa, limits.prompt_cache_ram_default);
        try cache.?.enableDisk(ctx.io, ctx.cache_dir, limits.prompt_cache_disk_default, fp);
        log.info("prompt-cache dir: {s} (set $ZIGFERENCER_CACHE_DIR to relocate)", .{
            ctx.cache_dir,
        });
    }

    defer if (cache) |*c| c.deinit();

    const t_start = Io.Clock.Timestamp.now(ctx.io, .awake);
    var prefilled: Prefilled = .{ .pos = 0, .last_logits = undefined };
    if (cache) |*c| {
        if (c.lookup(cache_key, ids)) |hit| {
            prefilled.pos = try runner.restoreFromBytes(hit.blob, hit.n);
            prefilled.last_logits = hit.last_logits;
            log.info("prompt-cache HIT ({s}): skipped prefill of {d} tokens", .{
                if (c.last_hit_from_disk) "disk" else "ram", ids.len,
            });
        }
    }

    if (prefilled.pos == 0) {
        const t0 = Io.Clock.Timestamp.now(ctx.io, .awake);
        prefilled = try prefillChunks(runner, ids);
        if (cache) |*c| {
            try admitPrompt(runner, t, c, cache_key, prefilled.last_logits, t0);
        }
    }

    return streamDecode(runner, t, prefilled, t_start);
}

/// Snapshot the prefilled prompt into the cache. `t0` starts the prefill the cache
/// weighs the snapshot against.
fn admitPrompt(
    runner: anytype,
    t: *const Turn,
    cache: *PromptCache,
    cache_key: u64,
    last_logits: []const f32,
    t0: Io.Clock.Timestamp,
) !void {
    const ctx = t.boot.ctx;
    const ids = t.prompt_ids;
    const prompt_len: u32 = @intCast(ids.len);

    const prefill_ns = time.elapsedNs(t0, Io.Clock.Timestamp.now(ctx.io, .awake));
    const s0 = Io.Clock.Timestamp.now(ctx.io, .awake);

    // The cache copies the snapshot on admit, so the buffer lives for this block only.
    const blob = try ctx.gpa.alloc(u8, runner.snapshotBytes(prompt_len));
    defer ctx.gpa.free(blob);

    try runner.snapshotInto(prompt_len, blob);
    const save_ns = time.elapsedNs(s0, Io.Clock.Timestamp.now(ctx.io, .awake));

    const last_dup = try ctx.arena.dupe(f32, last_logits);
    _ = cache.admit(cache_key, ids, blob, last_dup, prefill_ns, save_ns);
    log.info("prompt-cache MISS: prefilled + cached (ram={} disk={})", .{
        cache.last_admit_ram, cache.last_admit_disk,
    });
}

/// Refuse a prompt past the window, and echo the text without the chat scaffolding.
fn echoPrompt(t: *const Turn, max_seq: u32) !void {
    if (t.prompt_ids.len > max_seq) {
        log.err("prompt has {d} tokens, the context limit is {d}", .{ t.prompt_ids.len, max_seq });
        return error.PromptTooLong;
    }

    try t.out.writeAll(t.boot.flags.prompt);
    try t.out.flush();
}

fn prefillChunks(runner: anytype, ids: []const u32) !Prefilled {
    var pos: u32 = 0;
    var last_logits: []f32 = undefined;

    // Each pass feeds at least one token, so the loop ends in `ids.len` passes.
    while (pos < ids.len) {
        const chunk: u32 = @intCast(@min(ids.len - pos, runner.prefill_batch_max));

        const sp = trace_span.begin(.prefill_chunk);
        last_logits = try runner.prefillBatch(ids[pos..][0..chunk], pos);
        sp.end();

        pos += chunk;
    }

    return .{ .pos = pos, .last_logits = last_logits };
}

fn prefillTokens(runner: anytype, ids: []const u32) !Prefilled {
    var pos: u32 = 0;
    var last_logits: []f32 = undefined;

    for (ids) |id| {
        const sp = trace_span.begin(.prefill_chunk);
        last_logits = try runner.step(id, pos);
        sp.end();

        pos += 1;
    }

    return .{ .pos = pos, .last_logits = last_logits };
}

/// Decode greedily from the prefill, stream each piece, and log TTFT and ITL. The
/// caller captures `t_start` before prefill, so TTFT holds the latency a reader feels.
fn streamDecode(
    runner: anytype,
    t: *const Turn,
    prefilled: Prefilled,
    t_start: Io.Clock.Timestamp,
) !void {
    const ctx = t.boot.ctx;
    const max_tokens = t.boot.flags.max_tokens;

    log.info("generating up to {d} tokens (prompt {d} tokens)", .{ max_tokens, prefilled.pos });

    var sink: StreamSink = .{ .out = t.out, .v = t.v, .arena = ctx.arena };
    const g = try generate.decode(
        runner,
        prefilled.last_logits,
        prefilled.pos,
        max_tokens,
        runner.max_seq,
        t.v.eos_id,
        false,
        ctx.io,
        t_start,
        &sink,
    );
    try t.out.writeAll("\n");
    try t.out.flush();

    if (g.tokens_generated > 1) {
        log.info("{d} tokens | TTFT {d:.1} ms | ITL {d:.1} ms = {d:.1} tok/s", .{
            g.tokens_generated, g.ttft_ms, g.itl_ms, 1000.0 / g.itl_ms,
        });
    } else {
        log.info("{d} tokens | TTFT {d:.1} ms", .{ g.tokens_generated, g.ttft_ms });
    }
}

fn msSince(io: Io, t0: Io.Clock.Timestamp) f64 {
    return time.msBetween(t0, Io.Clock.Timestamp.now(io, .awake));
}
