//! Shared serve setup: the parse-independent model load and engine construction
//! that `serve` and `serve-openai` have in common. A subcommand parses its
//! flags into `flags.ServeFlags`, then hands `buildAndServe` that value and a
//! `launcher` value, which is
//! any value with a generic `serve(EngineT, ctx, *EngineT, CodecParts, Config)`
//! method. buildAndServe builds the runner for the model's arch on the active
//! backend and wraps it in the tiered-KV residency. It then hands the engine to
//! the launcher, which builds its own codec and transport adapter over the
//! shared scheduler.
//!
//! The engine flavors live here once, so every connector gets every backend with
//! no duplication. Four of them are the arch-by-backend grid: Llama and Qwen3.5,
//! CPU and fused-GPU. The last two are the hybrid runners, one per arch, which
//! `--gpu-layers N` or `--gpu-layers auto` selects when the placement planner
//! splits the model across both tiers. The launcher is the only boundary that
//! differs between connectors.

const std = @import("std");
const Io = std.Io;
const build_options = @import("build_options");

const cli = @import("command.zig");
const args = @import("args.zig");
const flags_mod = @import("flags.zig");
const select = @import("select.zig");
const registry = @import("../core/registry.zig");
const dtype = @import("../core/dtype.zig");
const loader = @import("../io/gguf/loader.zig");
const vocab_mod = @import("../core/tokenizer/vocab.zig");
const bpe = @import("../core/tokenizer/bpe.zig");
const chat = @import("../core/chat.zig");
const limits = @import("../core/limits.zig");
const hwinfo = @import("../core/hwinfo.zig");
const kv_budget = @import("../core/kv_budget.zig");
const resolve_mod = @import("../engine/resolve.zig");
const GpuLayers = kv_budget.GpuLayers;
const BlobTier = kv_budget.BlobTier;
const kv_index = @import("../core/kv_index.zig");
const prompt_cache_mod = @import("../runtime/prompt_cache.zig");
const tensor_lookup = @import("../runtime/tensor_lookup.zig");
const kv_cost = @import("../runtime/kv_cost.zig");
const time = @import("../core/time.zig");
const parallel = @import("../runtime/parallel.zig");
const Pool = @import("../runtime/pool.zig").Pool;
const LlamaRunner = @import("../runtime/llama_runner.zig").LlamaRunner;
const QwenRunner = @import("../runtime/qwen_runner.zig").QwenRunner;
const Sessions = @import("../runtime/session.zig").Sessions;
const Residency = @import("../serve/residency.zig").Residency;
const protocol = @import("../serve/protocol.zig");
const scheduler_mod = @import("../serve/scheduler.zig");
const sampler = @import("../runtime/sampler.zig");
const prefill_chunk = @import("../serve/prefill_chunk.zig");
const log = std.log.scoped(.serve);

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

// -- types ------------------------------------------------------------------

/// The knobs of the binary wire that `serve-openai` does not take. Every other
/// serve knob is a field of `flags.ServeFlags`.
pub const Wire = struct {
    /// `--allow-shutdown`: a client can stop the server through the wire.
    allow_shutdown: bool = false,
    /// `--sessions-per-connection`: the sessions one connection can hold, in
    /// 1..max_sessions. Absent takes `max_sessions`.
    sessions_per_connection: ?u16 = null,
    /// `--resume-ttl`, in milliseconds: the longest lease a client can keep,
    /// in 0..limits.max_resume_ttl_ms. 0 turns resumption off.
    resume_ttl_ms: u32 = 0,
};

/// The ingredients a connector needs to build its own codec, either the
/// chat-aware `event.Codec` or the OpenAI `RawCodec`. Plain values cross the
/// boundary.
pub const CodecParts = struct {
    vocab: *const vocab_mod.Vocab,
    pieces: []const []const u8,
    /// The registry kind with the `--thinking` flag applied.
    kind: chat.Kind,
    eos_id: u32,
    /// The think tags as ids, for a kind that has a thinking mode.
    think_ids: ?chat.ThinkIds = null,
};

/// The think tags as the scheduler reads them: the two tags and the ids a
/// spent budget forces.
pub const ThinkTags = struct {
    ids: chat.ThinkIds,
    force: [limits.think_force_ids_max]u32,
    force_len: u8,
};

/// The window of a boot before any device fit. `resolve` states it.
pub const SessionWindow = resolve_mod.Window;

/// Where a window that no device fit cut came from. A named window is the
/// flag. A default at the model ceiling is the ceiling, and a default below it
/// is the session byte budget.
const WindowSource = enum { flag, ceiling, budget };

/// Bytes the widest single-tier family tag occupies: the longest family name and
/// the widest KV dtype component.
const family_tag_buf_len: usize = "vulkan_fused".len + dtype.kv_tag_suffix_max;

/// The refusal of a thinking budget that no turn can spend. It takes the budget
/// and the window.
const thinking_budget_past_window_fmt = "--thinking-budget {d} is at or past the window of {d} positions, so it can never close the block. Use a smaller budget or a larger --max-seq";

/// The refusal of a `--max-seq` value that `baseWindow` puts out of range. It
/// takes the model ceiling. Serve, `run`, and both benches print it.
pub const window_out_of_range_fmt = "--max-seq must be in 1..{d} for this model";

// -- api --------------------------------------------------------------------

/// A card row as wire sampling. The card has no frequency penalty, so the
/// wire field keeps its identity 0.
fn wireSampling(r: registry.Recommended) protocol.Sampling {
    return .{
        .temperature = r.temperature,
        .top_k = r.top_k,
        .top_p = r.top_p,
        .seed = 0,
        .presence_penalty = r.presence_penalty,
        .repetition_penalty = r.repetition_penalty,
        .min_p = r.min_p,
    };
}

/// Load the model, build the engine for its arch on the active backend, and hand
/// it to `launcher.serve`. The launcher owns the transport.
pub fn buildAndServe(ctx: cli.Context, pub_name: ?[]const u8, serve: *const flags_mod.ServeFlags, wire: Wire, launcher: anytype) !void {
    // --max-sessions is the LOGICAL conversation count. --vram-slots is the
    // PHYSICAL VRAM KV-slot count. Only the fused-GPU build can evict idle
    // sessions to RAM, so only it can over-subscribe (logical > physical).
    const can_tier = comptime build_options.use_vulkan_fused_ops;
    const session_cap: u16 = if (can_tier) @intCast(limits.max_logical_sessions) else @intCast(limits.max_batch);
    if (serve.max_sessions == 0 or serve.max_sessions > session_cap) {
        return args.fail(ctx.io, "--max-sessions must be in 1..{d}", .{session_cap});
    }
    if (serve.max_connections == 0 or serve.max_connections > limits.max_connections) {
        return args.fail(ctx.io, "--max-connections must be in 1..{d}", .{limits.max_connections});
    }
    if (wire.resume_ttl_ms > limits.max_resume_ttl_ms) {
        return args.fail(ctx.io, "--resume-ttl must be in 0..{d} ms, the lease ceiling", .{limits.max_resume_ttl_ms});
    }
    const sessions_per_connection: u16 = wire.sessions_per_connection orelse serve.max_sessions;
    if (sessions_per_connection == 0 or sessions_per_connection > serve.max_sessions) {
        return args.fail(ctx.io, "--sessions-per-connection must be in 1..{d}, the --max-sessions value", .{serve.max_sessions});
    }
    // One flag per behavior: --kv-vram states a budget, and --vram-slots states
    // the count that budget buys. Two flags for one decision can disagree, so
    // the pair is a mistake, not a merge.
    if (serve.kv_vram != null and serve.vram_slots != null) {
        return args.fail(ctx.io, "--kv-vram and --vram-slots are mutually exclusive (the budget buys the slots)", .{});
    }
    if (serve.engine.gpu_layers != null and !can_tier) {
        return args.fail(ctx.io, "--gpu-layers requires -Dops-backend=vulkan_fused", .{});
    }
    const kv_vram_bytes: ?u64 = blk: {
        const mib = serve.kv_vram orelse break :blk null;
        if (!can_tier) {
            return args.fail(ctx.io, "--kv-vram applies only to the fused-GPU build (it spends a device KV budget)", .{});
        }
        if (mib == 0) return args.fail(ctx.io, "--kv-vram must be >= 1 (MiB)", .{});
        break :blk @as(u64, mib) << 20;
    };
    // Two flags stating one device budget can only disagree. The placement
    // planner already prices the pinned suffix and its KV against that budget.
    if (serve.engine.gpu_layers != null and kv_vram_bytes != null) {
        return args.fail(ctx.io, "--gpu-layers and --kv-vram are mutually exclusive (a pinned placement prices its own KV, it does not buy slots)", .{});
    }
    // The RAM tier holds evicted KV in host memory, which every backend has, so
    // this budget carries on a CPU build too.
    const kv_ram_bytes: ?u64 = blk: {
        const mib = serve.kv_ram orelse break :blk null;
        if (mib == 0) return args.fail(ctx.io, "--kv-ram must be >= 1 (MiB)", .{});
        break :blk @as(u64, mib) << 20;
    };
    const prefix_disk_bytes: u64 = blk: {
        const mib = serve.prefix_disk orelse break :blk 0;
        if (mib == 0) return args.fail(ctx.io, "--prefix-disk must be >= 1 (MiB)", .{});
        break :blk @as(u64, mib) << 20;
    };
    // Preload files are read before the model loads: a bad path is a config
    // error, and a config error fails in milliseconds, never after a GGUF map. An
    // oversize or empty file fails too, because the operator asked for warmth and
    // a silent skip is a correctness bug.
    const preload_paths = serve.preload.slice();
    std.debug.assert(preload_paths.len <= limits.serve_preload_max);
    const preload_texts: [][]const u8 = try ctx.arena.alloc([]const u8, preload_paths.len);
    for (preload_paths, preload_texts) |p, *out| {
        const text = Io.Dir.cwd().readFileAlloc(ctx.io, p, ctx.arena, .limited(limits.serve_preload_text_max)) catch |err| {
            return args.fail(ctx.io, "--preload {s}: {t} (cap {d} KiB)", .{ p, err, limits.serve_preload_text_max / 1024 });
        };
        if (text.len == 0) return args.fail(ctx.io, "--preload {s}: the file is empty", .{p});
        out.* = text;
    }
    const slots_req: ?u16 = blk: {
        const val = serve.vram_slots orelse break :blk null;
        if (!can_tier) {
            return args.fail(ctx.io, "--vram-slots applies only to the fused-GPU build (VRAM<->RAM KV swapping)", .{});
        }
        if (val == 0 or val > limits.max_vram_slots) {
            return args.fail(ctx.io, "--vram-slots must be in 1..{d}", .{limits.max_vram_slots});
        }
        // A slot no session can occupy is VRAM bought and left dead, so the
        // request caps at the logical session count.
        break :blk @min(val, serve.max_sessions);
    };
    // The structural slot count when no flag pins one: every session resident,
    // which is the shipped shape. A --kv-vram budget lets the planner buy the
    // count, so it needs only a width source here.
    const structural_slots: u16 = slots_req orelse blk: {
        if (kv_vram_bytes != null) break :blk @intCast(@min(@as(u32, serve.max_sessions), limits.max_batch));
        if (serve.max_sessions > limits.max_batch) {
            return args.fail(ctx.io, "--max-sessions > {d} needs --vram-slots or --kv-vram to swap KV between VRAM and RAM", .{limits.max_batch});
        }
        break :blk serve.max_sessions;
    };
    // The CPU runner holds one slot per session, so a zero pin with fewer slots
    // than sessions is a user error, refused before the model loads.
    if (serve.engine.gpu_layers) |g| {
        if (g.isZeroCount()) try refuseCpuSlots(ctx.io, structural_slots, serve.max_sessions);
    }
    // The decode width: slots bound breadth, `limits.max_batch` bounds the rows
    // one tick advances.
    const width_cap: u16 = @intCast(@min(@as(u32, structural_slots), limits.max_batch));
    if (serve.max_batch) |req| {
        if (req == 0 or req > limits.max_batch) return args.fail(ctx.io, "--max-batch must be in 1..{d}", .{limits.max_batch});
    }
    const batch: u16 = @min(serve.max_batch orelse width_cap, width_cap);
    if (serve.max_batch) |req| {
        if (req > batch) log.info("--max-batch {d} is above the {d} slots, so the decode width is {d}", .{ req, structural_slots, batch });
    }
    if (batch == 0) return args.fail(ctx.io, "--max-batch must be >= 1", .{});

    // One forward pass advances at most max_step_rows: the decode rows, at most
    // batch of them, plus a prefill chunk. The CPU runners size their row-scratch
    // from this figure. The fused-GPU runners hold a fixed width, and each branch
    // below clamps to it.
    if ((serve.prefill_chunk orelse 0) > limits.max_prefill_chunk)
        return args.fail(ctx.io, "--prefill-chunk must be in 0..{d}", .{limits.max_prefill_chunk});
    if (serve.engine.thinking_budget > limits.max_model_ctx)
        return args.fail(ctx.io, "--thinking-budget must be in 0..{d}", .{limits.max_model_ctx});
    // Never below the slot count: the scheduler's floor-1 prefill can put up to
    // n_slots rows in a pass even at prefill_chunk == 0. So the runner's row
    // capacity stays at or above the structural slots, and grows to fit an
    // explicit chunk on top. The `batch + 1` floor keeps the one-row prefill
    // legal when a budget buys more slots than the decode width and decode
    // saturates.
    const cpu_prefill = prefill_chunk.fromFlag(serve.prefill_chunk, limits.cpu_prefill_chunk);
    const max_step_rows: u32 = prefill_chunk.stepRows(cpu_prefill, structural_slots, batch);

    // The whole-model fused arm runs no CPU op, so its pool starts no workers.
    // A pin of any layer count can run CPU layers.
    const policy = try resolvePolicy(
        ctx.io,
        serve.engine.threads.policyFlags(),
        !(build_options.use_vulkan_fused_ops and serve.engine.gpu_layers == null),
    );

    const model = try select.model(ctx, pub_name);
    const input: resolve_mod.Input = .{
        .engine = serve.engine,
        .serve = .{
            .max_sessions = serve.max_sessions,
            .structural_slots = structural_slots,
            .kv_ram_bytes = kv_ram_bytes,
            .batch = batch,
            .slots_req = slots_req,
            .kv_vram_bytes = kv_vram_bytes,
        },
    };
    const entry = entryOf(model);
    // Every refusal that needs no model file lands here, before the load.
    switch (resolve_mod.early(.serve, input, entry)) {
        .ok => {},
        .refused => |r| return refuse(ctx.io, model.pub_name, r),
    }
    const path = try registry.resolveLoadPath(ctx.arena, ctx.io, ctx.models_root, model.*);
    var loaded = try loader.load(ctx.io, ctx.arena, path);
    defer loaded.close();

    const model_facts = try modelFacts(ctx.gpa, &loaded);
    const ram_free = hostRamFree(ctx.io);
    const resolved = switch (resolve_mod.resolve(.serve, input, entry, loaded.card, .{ .ram_free = ram_free, .model = model_facts })) {
        .ok => |r| r,
        .refused => |r| return refuse(ctx.io, model.pub_name, r),
    };
    const kind = resolved.early.chat_kind;
    const kv_dtype = resolved.early.kv_dtype;
    const ceiling = resolved.window.ceiling;
    const serve_default = kv_budget.defaultWindows(
        kv_budget.costModel(loaded.card.kvShape(), kv_dtype),
        ceiling,
    ).serve;

    // One pool serves every CPU op of the command. Its workers live until
    // the command returns.
    var pool: Pool = undefined;
    try parallel.initPool(&pool, ctx.io, policy);
    defer pool.deinit();

    var v = try vocab_mod.buildFromParsed(ctx.arena, loaded.parsed);
    const pieces = try buildPieces(ctx.arena, &v);
    var boot_tag_buf: [family_tag_buf_len]u8 = undefined;
    const model_fp = familyFp(model, &loaded, familyTag(
        &boot_tag_buf,
        if (build_options.use_vulkan_fused_ops) "vulkan_fused" else "cpu",
        kv_dtype,
    ));
    const think_tags = try thinkTagsOf(ctx.arena, &v, kind);
    log.info("thinking: {s}, budget {d}", .{ if (chat.thinkOpen(kind)) "on" else "off", serve.engine.thinking_budget });
    const parts: CodecParts = .{
        .vocab = &v,
        .pieces = pieces,
        .kind = kind,
        .eos_id = v.eos_id,
        .think_ids = if (think_tags) |t| t.ids else null,
    };

    // One scheduler Config, built where the branch-independent facts live. Each
    // backend branch patches the three engine-derived fields (max_seq,
    // max_step_rows, cache.ram_bytes) before launch, and the scheduler's init
    // asserts that the branch set them.
    std.debug.assert(sessions_per_connection >= 1);
    var sched_cfg: scheduler_mod.Config = .{
        .max_sessions = serve.max_sessions,
        .sessions_per_connection = sessions_per_connection,
        .max_batch = batch,
        .max_seq = 0, // per branch: the runner's window
        .max_step_rows = 0, // per branch: clamped to the runner's row scratch
        .prefill = cpu_prefill,
        .max_connections = serve.max_connections,
        .allow_shutdown = wire.allow_shutdown,
        .resume_ttl_max_ms = wire.resume_ttl_ms,
        .model_name = model.pub_name,
        .vocab_size = loaded.card.vocab_size,
        .eos_id = v.eos_id,
        .cache = .{
            .disk_bytes = prefix_disk_bytes,
            .dir = ctx.cache_dir,
            .model_fp = model_fp,
            .preload = preload_texts,
        },
        .think = schedulerThink(think_tags, kind, serve.engine.thinking_budget),
    };
    // `hello_ack` carries the card row of each think mode, so a client picks
    // the row of the mode of each turn.
    if (registry.recommendedSampling(model.pub_name, true)) |r| sched_cfg.recommended_think = wireSampling(r);
    if (registry.recommendedSampling(model.pub_name, false)) |r| sched_cfg.recommended_plain = wireSampling(r);
    logSamplerRam(samplerRam(loaded.card.vocab_size, serve.max_sessions));

    // `early` refused the per-op vulkan backend before the load, so a per-op
    // build never reaches the arms below.

    // One precision witness per boot, ahead of the arch branch, so every arm
    // reports the same map right above its own plan line.
    logPrecision(&loaded, kv_dtype);

    // The KV economy is planned, not derived at the runner: `resolve` states
    // the shape every arm runs. A fused-GPU arm reads the one VRAM figure of
    // its context to spend a device budget (--vram-slots or --kv-vram) or to
    // fit a default window. The CPU arms state no device bytes.
    const kv_ram_named = kv_ram_bytes != null;
    // `--gpu-layers 0` is the all-host arm, so it takes the CPU tail and this
    // build creates no device context. An auto split that funds no GPU layer
    // takes the same tail with the host arm of its device resolve.
    var host_arm = resolved.cpu;
    const qwen = loaded.card.arch == .qwen35;
    const family = if (qwen) "qwen" else "llama";
    const cpu_pinned = if (serve.engine.gpu_layers) |g| g.isZeroCount() else false;
    if (cpu_pinned) log.info("--gpu-layers 0: the {s} CPU runner decodes every layer", .{family});

    if (comptime build_options.use_vulkan_fused_ops) device: {
        if (cpu_pinned) break :device;
        const t_init = Io.Clock.Timestamp.now(ctx.io, .awake);
        if (serve.engine.gpu_layers == null)
            log.info("initializing GPU: context + weight upload to VRAM...", .{})
        else
            log.info("initializing GPU: context + suffix weight upload to VRAM...", .{});
        const ctxp = try ctx.arena.create(vulkan_ctx_mod.Context);
        ctxp.* = try vulkan_ctx_mod.Context.init(ctx.arena, .{ .storage_range_override = ctx.vk_storage_range });
        defer ctxp.deinit();
        const boot: DeviceBoot = .{ .kind = .serve, .input = input, .entry = entry, .card = loaded.card, .model = model_facts, .ram = .{ .read = ram_free } };
        const dev = try boot.resolveOn(ctx.io, ctxp);
        const n_layers: u32 = loaded.card.n_layers;
        if (dev.placed) |p| {
            if (p.slots_from) |from| log.info("hybrid: {d} vram kv slots, clamped from {d}", .{ limits.max_batch, from });
            if (p.clamp) |c| logClamp(c);
            if (p.prefix_default) logPrefixDefault();
            if (p.split) |s| {
                var span_buf: [kv_budget.placement_text_max]u8 = undefined;
                log.info("{s}", .{kv_budget.formatPlacement(s, &span_buf) catch unreachable});
                if (p.pin == null) logAutoSplit(s.split, n_layers);
            }
            if (p.selects == .fused) logFusedArm(p.pin);
        }
        const arm = switch (dev.arm) {
            .ok => |a| a,
            .refused => |r| return refuse(ctx.io, model.pub_name, r),
        };
        switch (arm) {
            .cpu => |c| {
                log.info("--gpu-layers auto: the {s} CPU runner decodes every layer", .{family});
                host_arm = c;
                // The context defer releases the device on the way out.
                break :device;
            },
            .hybrid => |h| {
                const quote = switch (h.quote) {
                    .ok => |q| q,
                    .refused => |r| return refuse(ctx.io, model.pub_name, r),
                };
                logRamQuote(quote, kv_ram_named);
                const hybrid_plan = h.plan.?;
                std.debug.assert(hybrid_plan.split > 0);
                std.debug.assert(hybrid_plan.split < n_layers);
                var ledger_buf: [kv_budget.placement_ledger_text_max]u8 = undefined;
                log.info("{s}", .{kv_budget.formatLedger(hybrid_plan, &ledger_buf) catch unreachable});
                const mib = 1024 * 1024;
                const ready_fmt = "{s} hybrid engine ready in {d:.0} ms: logical={d} vram={d} batch={d} max_seq={d} gpu={d}/{d} layers kv={d} MiB vram + {d} MiB host";
                if (qwen) {
                    const hybrid = try qwen_hybrid_mod.QwenHybridRunner.init(ctx.gpa, ctxp, &loaded, hybrid_plan, ctx.io, &pool, policy, kv_dtype);
                    defer hybrid.deinit();
                    log.info(ready_fmt, .{ family, msBetween(ctx.io, t_init), serve.max_sessions, hybrid_plan.n_slots, batch, hybrid_plan.window, n_layers - hybrid_plan.split, n_layers, hybrid_plan.kv_vram / mib, hybrid_plan.ram.host_kv / mib });
                    var tag_buf: [qwen_hybrid_mod.QwenHybridRunner.backend_tag_buf_len]u8 = undefined;
                    // The tag names the family and the placement, so a blob
                    // can never restore across a split or across a family.
                    sched_cfg.cache.model_fp = familyFp(model, &loaded, qwen_hybrid_mod.QwenHybridRunner.backendTag(&tag_buf, hybrid_plan.split, kv_dtype));
                    std.debug.assert(hybrid.max_seq == hybrid_plan.window);
                    sched_cfg.max_seq = hybrid.max_seq;
                    setStepShape(&sched_cfg, serve.prefill_chunk, max_step_rows, hybrid.maxStepRows(), hybrid.maxStepRows());
                    sched_cfg.cache.ram_bytes = hybrid_plan.ram.prefix;
                    return serveWithResidency(qwen_hybrid_mod.QwenHybridRunner, ctx, hybrid, serve.max_sessions, quote.tier, parts, sched_cfg, launcher);
                }
                const hybrid = try llama_hybrid_mod.LlamaHybridRunner.init(ctx.gpa, ctxp, &loaded, hybrid_plan, ctx.io, &pool, policy, kv_dtype);
                defer hybrid.deinit();
                log.info(ready_fmt, .{ family, msBetween(ctx.io, t_init), serve.max_sessions, hybrid_plan.n_slots, batch, hybrid_plan.window, n_layers - hybrid_plan.split, n_layers, hybrid_plan.kv_vram / mib, hybrid_plan.ram.host_kv / mib });
                var tag_buf: [llama_hybrid_mod.LlamaHybridRunner.backend_tag_buf_len]u8 = undefined;
                // The tag names the placement, so a blob can never restore
                // across a different split.
                sched_cfg.cache.model_fp = familyFp(model, &loaded, llama_hybrid_mod.LlamaHybridRunner.backendTag(&tag_buf, hybrid_plan.split, hybrid.quant, kv_dtype));
                std.debug.assert(hybrid.max_seq == hybrid_plan.window);
                sched_cfg.max_seq = hybrid.max_seq;
                setStepShape(&sched_cfg, serve.prefill_chunk, max_step_rows, hybrid.maxStepRows(), hybrid.maxStepRows());
                sched_cfg.cache.ram_bytes = hybrid_plan.ram.prefix;
                return serveWithResidency(llama_hybrid_mod.LlamaHybridRunner, ctx, hybrid, serve.max_sessions, quote.tier, parts, sched_cfg, launcher);
            },
            .fused => |f| {
                const quote = try logFused(ctx.io, model.pub_name, f, false, kv_ram_named, kv_dtype);
                const plan = f.plan;
                std.debug.assert(f.caps.serve_default == serve_default);
                if (qwen) {
                    const fused = try qwen_fused.QwenFusedRunner.init(ctx.gpa, ctxp, &loaded, plan, kv_dtype);
                    defer fused.deinit();
                    log.info("qwen gpu engine ready in {d:.0} ms: logical={d} vram={d} batch={d} max_seq={d}", .{ msBetween(ctx.io, t_init), serve.max_sessions, fused.n_slots, batch, fused.max_seq });
                    std.debug.assert(fused.max_seq == plan.window);
                    sched_cfg.max_seq = fused.max_seq;
                    setStepShape(&sched_cfg, serve.prefill_chunk, max_step_rows, fused.maxStepRows(), fused.maxStepRows());
                    sched_cfg.cache.ram_bytes = plan.ram.prefix;
                    return serveWithResidency(qwen_fused.QwenFusedRunner, ctx, fused, serve.max_sessions, quote.tier, parts, sched_cfg, launcher);
                }
                const fused = try llama_fused.LlamaFusedRunner.initFromLoaded(ctx.gpa, ctxp, &loaded, plan, kv_dtype);
                defer fused.deinit();
                log.info("llama gpu engine ready in {d:.0} ms: logical={d} vram={d} batch={d} max_seq={d}", .{ msBetween(ctx.io, t_init), serve.max_sessions, fused.n_slots, batch, fused.max_seq });
                std.debug.assert(fused.max_seq == plan.window);
                sched_cfg.max_seq = fused.max_seq;
                setStepShape(&sched_cfg, serve.prefill_chunk, max_step_rows, fused.maxStepRows(), fused.maxStepRows());
                sched_cfg.cache.ram_bytes = plan.ram.prefix;
                return serveWithResidency(llama_fused.LlamaFusedRunner, ctx, fused, serve.max_sessions, quote.tier, parts, sched_cfg, launcher);
            },
        }
    }

    const arm = try cpuArmOf(ctx.io, model.pub_name, host_arm.?, kv_ram_named, kv_dtype);
    const t_init = Io.Clock.Timestamp.now(ctx.io, .awake);
    const plan = arm.plan;
    const quote = arm.quote;
    // A fused build reaches this tail only through `--gpu-layers 0`, or an
    // auto split that funds no GPU layer. The CPU runner writes these blobs,
    // so they belong in the cpu namespace and not in the fused one the boot
    // fingerprint named.
    if (comptime build_options.use_vulkan_fused_ops) {
        var cpu_tag_buf: [family_tag_buf_len]u8 = undefined;
        sched_cfg.cache.model_fp = familyFp(model, &loaded, familyTag(&cpu_tag_buf, "cpu", kv_dtype));
    }
    if (qwen) {
        var runner = try QwenRunner.init(ctx.arena, &loaded, plan, max_step_rows, ctx.io, &pool, policy, kv_dtype);
        log.info("qwen engine ready in {d:.0} ms: sessions={d} batch={d} max_seq={d} kv={d} MiB", .{ msBetween(ctx.io, t_init), serve.max_sessions, batch, runner.max_seq, kvMiB(&runner) });
        std.debug.assert(runner.max_seq == plan.window);
        sched_cfg.max_seq = runner.max_seq;
        setStepShape(&sched_cfg, serve.prefill_chunk, max_step_rows, runner.maxStepRows(), limits.cpu_prefill_chunk);
        sched_cfg.cache.ram_bytes = plan.ram.prefix;
        return serveWithResidency(QwenRunner, ctx, &runner, serve.max_sessions, quote.tier, parts, sched_cfg, launcher);
    }
    var runner = try LlamaRunner.init(ctx.arena, &loaded, plan, max_step_rows, ctx.io, &pool, policy, kv_dtype);
    log.info("llama engine ready in {d:.0} ms: sessions={d} batch={d} max_seq={d} kv={d} MiB", .{ msBetween(ctx.io, t_init), serve.max_sessions, batch, runner.max_seq, kvMiB(&runner) });
    std.debug.assert(runner.max_seq == plan.window);
    sched_cfg.max_seq = runner.max_seq;
    setStepShape(&sched_cfg, serve.prefill_chunk, max_step_rows, runner.maxStepRows(), limits.cpu_prefill_chunk);
    sched_cfg.cache.ram_bytes = plan.ram.prefix;
    return serveWithResidency(LlamaRunner, ctx, &runner, serve.max_sessions, quote.tier, parts, sched_cfg, launcher);
}

/// The registry facts `resolve` reads.
pub fn entryOf(model: *const registry.Model) resolve_mod.Entry {
    return .{
        .pub_name = model.pub_name,
        .chat_template_kind = model.chat_template_kind,
        .kv_dtype_default = model.kv_dtype_default,
        .context_limit = model.context_limit,
    };
}

/// Print one refusal of `resolve` with its message, and return the error the
/// exit path knows. `name` is the public name of the model.
pub fn refuse(io: Io, name: []const u8, r: resolve_mod.Refusal) anyerror {
    switch (r) {
        .thinking_unsupported => return args.fail(io, "--thinking on does not apply to {s}. Its chat template has no thinking mode", .{name}),
        .thinking_budget_unsupported => return args.fail(io, "--thinking-budget does not apply to {s}. Its chat template has no thinking mode", .{name}),
        .serve_per_op_vulkan => {
            log.err("serve does not run on the per-op vulkan backend. Build -Dops-backend=vulkan_fused or a CPU backend", .{});
            return error.UnsupportedServeBackend;
        },
        .kv_dtype_per_op_vulkan => |d| return args.fail(io, "--kv-dtype {t} does not run on -Dops-backend=vulkan (that arm uploads the KV slab to the device as f32 on every attention call). Build a CPU backend", .{d}),
        .window_out_of_range => |ceiling| return args.fail(io, window_out_of_range_fmt, .{ceiling}),
        .cpu_slots => |c| return args.fail(io, "the CPU runner holds one slot per session, so --vram-slots {d} cannot serve --max-sessions {d}. Drop --vram-slots, or set it to the --max-sessions value", .{ c.slots, c.sessions }),
        .host_ram => |q| {
            var buf: [kv_budget.ram_refusal_text_max]u8 = undefined;
            log.err("{s}", .{kv_budget.formatRamRefusal(&buf, q)});
            return error.HostRamBudgetTooSmall;
        },
        .blob_ram => |q| {
            var buf: [kv_budget.blob_refusal_text_max]u8 = undefined;
            log.err("{s}", .{kv_budget.formatBlobRefusal(&buf, q)});
            return error.BlobRamBudgetTooSmall;
        },
        .gpu_layers_past_model => |g| return args.fail(io, "--gpu-layers {d} exceeds the model's {d} layers", .{ g.n, g.n_layers }),
        .hybrid_slots => |n| return args.fail(io, "--gpu-layers holds at most {d} vram kv slots, and this run asks for {d}. Lower --vram-slots, or drop --gpu-layers", .{ limits.max_batch, n }),
        .placement_vram => |p| {
            const mib = 1024 * 1024;
            const advice = switch (p.kind) {
                .run => "Lower --gpu-layers, or use a smaller / Q4 model",
                .serve => "Lower --gpu-layers or --max-seq, or use a smaller / Q4 model",
                .bench => "Lower --gpu-layers or use a smaller / Q4 model",
            };
            return args.fail(io, "--gpu-layers {d} needs ~{d} MiB of device budget, usable ~{d} MiB. {s}", .{ p.n_gpu, p.spend / mib, p.usable / mib, advice });
        },
        .whole_model_split => |pin| {
            if (pin) |n_gpu| return args.fail(io, "--gpu-layers {d} resolves to a whole-model placement, which the fused (or CPU) runner already serves. Drop the flag, or pass --force-hybrid to construct the hybrid at this split for the cross-runner gate", .{n_gpu});
            return args.fail(io, "--gpu-layers auto resolves to a whole-model placement, which the fused (or CPU) runner already serves. Drop the flag, or pass a count with --force-hybrid to construct the hybrid at that split for the cross-runner gate", .{});
        },
        .kv_vram_ceiling => |k| {
            log.err("kv budget: --max-seq {d} is past the model ceiling of {d} positions", .{ k.window, k.ceiling });
            return error.WindowExceedsCeiling;
        },
        .kv_vram_small => |k| {
            if (k.clamp) |c| logClamp(c);
            const mib = 1024 * 1024;
            log.err("kv budget: {d} MiB of KV cannot fund {d} slot(s). One slot costs {d:.1} MiB at window {d}, so raise --kv-vram to {d} MiB or more, or lower --max-seq / --vram-slots", .{
                k.x_bytes / mib,
                k.rows,
                @as(f64, @floatFromInt(k.per_slot)) / mib,
                k.window,
                (k.per_slot * k.rows + mib - 1) / mib,
            });
            return error.KvVramBudgetTooSmall;
        },
    }
}

/// The all-host serve arm with its boot lines: the prefix default, the plan,
/// and the RAM quote. `result` is the host arm of the boot, or the host arm of
/// a device resolve whose split keeps no layer on the device.
fn cpuArmOf(io: Io, name: []const u8, result: resolve_mod.Result(resolve_mod.CpuArm), kv_ram_named: bool, kv_dtype: dtype.KvDtype) !resolve_mod.CpuArm {
    const arm = switch (result) {
        .ok => |a| a,
        .refused => |r| return refuse(io, name, r),
    };
    if (arm.prefix_default) logPrefixDefault();
    logPlan(arm.plan, kv_dtype);
    logRamQuote(arm.quote, kv_ram_named);
    return arm;
}

/// What a device boot resolves on: the flags, the registry entry, the card,
/// and the weight census the boot measured after the load.
pub const DeviceBoot = struct {
    kind: resolve_mod.Kind,
    input: resolve_mod.Input,
    entry: resolve_mod.Entry,
    card: loader.ModelCard,
    model: resolve_mod.ModelFacts,
    /// The RAM figure of the boot. A boot that read it before the device passes
    /// it on, so the "RAM check is off" line prints once.
    ram: RamFigure = .unread,

    pub const RamFigure = union(enum) { unread, read: ?u64 };

    /// The free host RAM of the boot: the figure read before the device, or a
    /// fresh read.
    pub fn ramFree(self: *const DeviceBoot, io: Io) ?u64 {
        return switch (self.ram) {
            .unread => hostRamFree(io),
            .read => |free| free,
        };
    }

    /// Measure the device of `ctxp` once and resolve the device arm. The
    /// decisions before the device already passed, so only a device refusal
    /// can follow.
    pub fn resolveOn(self: *const DeviceBoot, io: Io, ctxp: anytype) !resolve_mod.DeviceArm {
        const facts: resolve_mod.Facts = .{
            .ram_free = self.ramFree(io),
            .model = self.model,
            .device = .{ .vram = vramFigure(ctxp), .buffer_limits = ctxp.buffer_limits },
        };
        return switch (resolve_mod.resolve(self.kind, self.input, self.entry, self.card, facts)) {
            .ok => |r| r.device.?,
            .refused => |r| refuse(io, self.entry.pub_name, r),
        };
    }
};

/// The fused arm of a one-stream boot that names no split, resolved on the
/// context.
pub fn fusedOn(io: Io, boot: *const DeviceBoot, ctxp: anytype) !resolve_mod.device_arm.Fused {
    const dev = try boot.resolveOn(io, ctxp);
    std.debug.assert(dev.placed == null);
    return switch (dev.arm) {
        .ok => |a| a.fused,
        .refused => |r| refuse(io, boot.entry.pub_name, r),
    };
}

/// The weight census of one model and the reservations its fused runner
/// makes, each from the function the runner calls. A build without the fused
/// ops measures the census only.
pub fn modelFacts(gpa: std.mem.Allocator, loaded: *const loader.Loaded) !resolve_mod.ModelFacts {
    const card = loaded.card;
    var m: resolve_mod.ModelFacts = .{
        .n_layers = card.n_layers,
        .layer_weights = @splat(0),
        .head_weights = tensor_lookup.headWeightBytes(loaded.tensors),
    };
    _ = tensor_lookup.layerWeightBytes(loaded.tensors, card.n_layers, &m.layer_weights);
    if (comptime build_options.use_vulkan_fused_ops) {
        const qwen = card.gdn != null;
        m.weights = if (qwen)
            qwen_fused.QwenFusedRunner.weightsVramBytes(loaded)
        else
            try llama_fused.LlamaFusedRunner.weightsVramBytesFromLoaded(gpa, loaded);
        m.logits_row = if (qwen)
            qwen_fused.QwenFusedRunner.logitsVramBytes(card.vocab_size, 1)
        else
            llama_fused.LlamaFusedRunner.logitsVramBytes(card.vocab_size, 1);
        // The runner caps and the resolve caps read one set of figures, so
        // the plan and the preflight cannot drift apart.
        const ceiling = @min(card.context_length, limits.max_model_ctx);
        for (std.enums.values(dtype.KvDtype)) |d| {
            if (qwen) {
                m.scratch.set(d, qwen_fused.QwenFusedRunner.scratchVramBytes(card, d));
                std.debug.assert(std.meta.eql(resolve_mod.caps(card, m, 0, ceiling, 1, d), qwen_fused.QwenFusedRunner.kvCaps(loaded, ceiling, 0, 1, d)));
            } else {
                m.scratch.set(d, llama_fused.LlamaFusedRunner.scratchVramBytes(card, d));
                std.debug.assert(std.meta.eql(resolve_mod.caps(card, m, 0, ceiling, 1, d), try llama_fused.LlamaFusedRunner.kvCaps(gpa, loaded, ceiling, 0, 1, d)));
            }
        }
    }
    return m;
}

/// The boot lines of a whole-model fused arm, in boot order, and its RAM
/// quote. `session` names `run` and `bench-model`, which log the window line
/// when no cut replaces it.
pub fn logFused(io: Io, name: []const u8, f: resolve_mod.device_arm.Fused, session: bool, kv_ram_named: bool, kv_dtype: dtype.KvDtype) !kv_budget.RamQuote {
    if (f.clamp) |c| logClamp(c);
    if (f.cut) |c| {
        log.info("kv window: {d} slots at the default window of {d} positions do not fit the free VRAM, so the window is {d}", .{ c.n_slots, c.base, c.window });
    } else if (session) {
        logSessionWindow(f.window);
    }
    if (f.stranded) |st| {
        const mib = 1024 * 1024;
        log.warn("kv budget: the ceiling of {d} slots cut the count this budget bought. {d} MiB of the {d} MiB stays unspendable. Raise --max-seq to give each slot a wider window", .{
            st.max_slots, st.stranded / mib, st.x_bytes / mib,
        });
    }
    if (f.device_slots) |ds| {
        log.info("kv budget: the {s} of {d} {t} holds one KV region of {d} slots at {d} positions, so the budget buys at most {d} slots", .{
            ds.limit.name, ds.limit.value, ds.limit.unit, ds.slots, ds.window, ds.slots,
        });
    }
    if (f.prefix_default) logPrefixDefault();
    logPlan(f.plan, kv_dtype);
    return switch (f.quote) {
        .ok => |q| blk: {
            logRamQuote(q, kv_ram_named);
            break :blk q;
        },
        .refused => |r| refuse(io, name, r),
    };
}

/// The line of a default window that the device buffer limits cut.
pub fn logClamp(c: resolve_mod.device_arm.Clamp) void {
    log.info("kv window: {d} slots at the default window of {d} positions need a KV region of {d} {t}. The {s} is {d} {t}, so the window is {d}", .{
        c.n_slots, c.base, c.need, c.limit.unit, c.limit.name, c.limit.value, c.limit.unit, c.window,
    });
}

/// Zero-config serve keeps the prefix cache at a named default when no
/// `--kv-ram` names a Y.
fn logPrefixDefault() void {
    log.info("prefix cache: {d} MiB default budget (set --kv-ram to size both ram tiers)", .{limits.serve_prefix_ram_default / (1024 * 1024)});
}

/// The free RAM every arm checks its host spend against in its RAM quote:
/// `MemAvailable` at boot. A platform without the figure turns
/// the check off, and one info line says so, because a silent skip hides a
/// missing guard.
pub fn hostRamFree(io: Io) ?u64 {
    const free = hwinfo.readMemAvailable(io);
    if (free == null) log.info("the RAM check is off: /proc/meminfo states no MemAvailable figure on this platform", .{});
    return free;
}

/// The resolved count of `--gpu-layers auto`, one line, so a log names the
/// split the live budget chose.
pub fn logAutoSplit(split: u32, n_layers: u32) void {
    std.debug.assert(n_layers >= 1);
    std.debug.assert(split <= n_layers);
    log.info("--gpu-layers auto: gpu {d}/{d} layers", .{ n_layers - split, n_layers });
}

/// The line of a placement that funds the whole model on the device. Null is
/// the auto pin.
fn logFusedArm(pin: ?u32) void {
    if (pin) |n| {
        log.info("--gpu-layers {d}: the budget funds the whole model, so the fused runner serves it", .{n});
    } else {
        log.info("--gpu-layers auto: the budget funds the whole model, so the fused runner serves it", .{});
    }
}

/// The startup witness of the KV economy: one line, every backend, every mode.
/// Public so the run and bench paths report the same shape serve reports.
/// `kv_dtype` is the dtype the prices came from.
pub fn logPlan(plan: kv_budget.KvPlan, kv_dtype: dtype.KvDtype) void {
    const mib = 1024 * 1024;
    log.info("kv plan: {d} slots x {d} window{s} = {d} MiB vram kv, {d} MiB ram, {d} MiB prefix", .{
        plan.n_slots,
        plan.window,
        kv_dtype.planToken(),
        plan.vram.kv / mib,
        plan.ram.blobs / mib,
        plan.ram.prefix / mib,
    });
}

/// The whole precision map of a boot, in one line. The engine holds no graph IR
/// a reader can inspect, so this line is where the per-site precisions become
/// visible. Every fact on it is known at init, so it costs nothing.
pub fn logPrecision(loaded: *const loader.Loaded, kv_dtype: dtype.KvDtype) void {
    log.info("precision: weights {s}, activations f32, accum f32, kv {t}, logits f32", .{
        weightQuantName(loaded),
        kv_dtype,
    });
}

/// id to displayable bytes for the whole vocab, decoded once at startup. A vocab
/// whose pieces exceed the serve cap is a config error at boot, never a silent
/// truncation later.
pub fn buildPieces(arena: std.mem.Allocator, v: *const vocab_mod.Vocab) ![]const []const u8 {
    const pieces = try arena.alloc([]const u8, v.tokens.len);
    for (pieces, 0..) |*piece, id| {
        piece.* = try bpe.decodeAlloc(v, arena, &[_]u32{@intCast(id)});
        if (piece.len > limits.max_token_text_bytes) {
            log.err("token {d} decodes to {d} bytes (serve cap {d})", .{ id, piece.len, limits.max_token_text_bytes });
            return error.TokenPieceTooLong;
        }
    }
    log.debug("pieces: {d} tokens materialized", .{pieces.len});
    return pieces;
}

/// The one stdout line a connector prints before its engine loop. The spawning
/// process parses the transport and the kernel-resolved address from it.
pub fn printReadyTcp(io: Io, addr: Io.net.IpAddress) !void {
    try printReady(io, "READY tcp {f}\n", .{addr});
}

pub fn printReadyUnix(io: Io, path: []const u8) !void {
    try printReady(io, "READY unix {s}\n", .{path});
}

pub const baseWindow = resolve_mod.baseWindow;

/// Log the base window of a one-session command and its source: the flag, the
/// model ceiling, or the session byte budget. A fused arm calls
/// `fitSessionWindow` instead, so a boot logs one window line. A window that
/// the device limits cut logs nothing here, because the clamp logged it with
/// both numbers.
pub fn logSessionWindow(w: SessionWindow) void {
    if (w.device_clamped) return;
    const mib = 1024 * 1024;
    switch (windowSource(w.base, w.ceiling, w.flag)) {
        .flag => log.info("kv window: {d} positions, set by --max-seq (model ceiling {d})", .{ w.base, w.ceiling }),
        .ceiling => log.info("kv window: {d} positions, the model ceiling", .{w.base}),
        .budget => log.info("kv window: {d} positions, the {d} MiB session budget (model ceiling {d})", .{
            w.base, w.session_bytes / mib, w.ceiling,
        }),
    }
}

/// The KV dtype one process frames every row at. The flag wins. An absent flag
/// takes the model's verified registry default. Every arch frames its attention
/// KV at f32 and at f16, so the choice needs no capability gate. Every command
/// calls this before it plans, prices, or touches a device, so no preflight
/// equality assert can meet a half-applied dtype.
pub fn kvDtypeFor(flag: ?dtype.KvDtype, model: *const registry.Model) dtype.KvDtype {
    return flag orelse model.kv_dtype_default;
}

/// The threading policy of one command: the host defaults, then the flags.
/// A command whose placement runs no CPU op gets a pool of one thread. A
/// `--threads` value over the pool cap or a `--spin-us` value over its cap
/// fails here, before any load or plan.
/// The resolved policy goes to the `parallel` log at info.
pub fn resolvePolicy(io: Io, flags: parallel.Flags, runs_cpu_ops: bool) error{UsageError}!parallel.Policy {
    const resolved = parallel.resolve(parallel.Policy.forHost(io), flags, runs_cpu_ops) catch |err| switch (err) {
        error.TooManyThreads => return args.fail(io, "--threads is {d}. The limit is {d}.", .{ flags.threads.?, limits.max_threads }),
        error.SpinTooLong => return args.fail(io, "--spin-us is {d}. The limit is {d}.", .{ flags.spin_us.?, parallel.max_spin_us }),
    };
    parallel.logResolved(resolved);
    return resolved.policy;
}

// -- helpers ----------------------------------------------------------------

/// The decimal digits of `v` in groups of three, joined by `_`, the way the
/// Zig source writes the constant.
pub fn groupDigits(comptime v: u64) []const u8 {
    comptime {
        const plain = std.fmt.comptimePrint("{d}", .{v});
        const len = plain.len + (plain.len - 1) / 3;
        var buf: [len]u8 = undefined;
        var j: usize = len;
        var i: usize = plain.len;
        var in_group: usize = 0;
        while (i > 0) {
            i -= 1;
            if (in_group == 3) {
                j -= 1;
                buf[j] = '_';
                in_group = 0;
            }
            j -= 1;
            buf[j] = plain[i];
            in_group += 1;
        }
        std.debug.assert(j == 0);
        const final = buf;
        return &final;
    }
}

/// Whether a thinking budget can ever force the close inside a window. Zero is
/// unbounded, so it always fits. The n-th think token lands at `p0 + n - 1`,
/// and the forced close needs one more step. The prompt holds at least one
/// position, so a budget at or past the window can never close the block.
fn thinkingBudgetFits(budget: u32, window: u32) bool {
    std.debug.assert(window >= 1);
    std.debug.assert(window <= limits.max_model_ctx);
    return budget == 0 or budget < window;
}

/// Encode the think tags of a model with a thinking mode. A vocab that breaks a
/// rule of `thinkTagsFrom` refuses the boot. The scheduler asserts the same
/// rules at init, so this check keeps model data away from an assert.
fn thinkTags(arena: std.mem.Allocator, v: *const vocab_mod.Vocab, text: chat.ThinkText) !ThinkTags {
    const open = try bpe.encodeAlloc(v, arena, text.open, false);
    const close = try bpe.encodeAlloc(v, arena, text.close, false);
    // The forced ids are the close and the separator the template writes after
    // a closed block.
    const force_text = try std.mem.concat(arena, u8, &.{ text.close, text.separator });
    const force = try bpe.encodeAlloc(v, arena, force_text, false);
    const tags = thinkTagsFrom(open, close, force, v.eos_id) catch |err| {
        log.err("think tags: {s} encodes to {any}, {s} to {any}, the forced close to {any}", .{ text.open, open, text.close, close, force });
        log.err("think tags: each tag must be one id. The forced close must start with {s} and hold at most {d} ids, none of them end-of-sequence {d}", .{
            text.close, limits.think_force_ids_max, v.eos_id,
        });
        return err;
    };
    log.debug("think tags: open {d}, close {d}, forced {any}", .{ tags.ids.open, tags.ids.close, tags.force[0..tags.force_len] });
    return tags;
}

/// The think tags of `kind`, or null for a kind with no thinking mode.
pub fn thinkTagsOf(arena: std.mem.Allocator, v: *const vocab_mod.Vocab, kind: chat.Kind) !?ThinkTags {
    const text = chat.thinkText(kind) orelse return null;
    return try thinkTags(arena, v, text);
}

/// The scheduler think config of `kind`. A kind with no tags gets the empty
/// config, so the scheduler refuses every open turn and budget.
pub fn schedulerThink(tags: ?ThinkTags, kind: chat.Kind, budget: u32) scheduler_mod.Think {
    const t = tags orelse return .{};
    return .{
        .open_default = chat.thinkOpen(kind),
        .open_id = t.ids.open,
        .close_id = t.ids.close,
        .budget = budget,
        .force = t.force,
        .force_len = t.force_len,
    };
}

/// The rules the scheduler asserts at init, checked on the encoded ids. Each
/// tag is one id. The forced ids start with the close, number at most
/// `think_force_ids_max`, and hold no end-of-sequence id.
fn thinkTagsFrom(open: []const u32, close: []const u32, force: []const u32, eos_id: u32) error{ThinkTagsUnusable}!ThinkTags {
    if (open.len != 1 or close.len != 1) return error.ThinkTagsUnusable;
    if (force.len == 0 or force.len > limits.think_force_ids_max) return error.ThinkTagsUnusable;
    if (force[0] != close[0]) return error.ThinkTagsUnusable;
    if (close[0] == eos_id) return error.ThinkTagsUnusable;
    for (force) |id| if (id == eos_id) return error.ThinkTagsUnusable;
    var tags: ThinkTags = .{
        .ids = .{ .open = open[0], .close = close[0] },
        .force = @splat(scheduler_mod.no_token),
        .force_len = @intCast(force.len),
    };
    @memcpy(tags.force[0..force.len], force);
    std.debug.assert(tags.force[0] == tags.ids.close);
    return tags;
}

/// The weights identity of one runner family: the model fingerprint qualified by
/// the family tag. A blob's numeric provenance is per backend, so an arm that
/// constructs a family the boot fingerprint did not name must patch
/// `cache.model_fp` with this.
fn familyFp(model: *const registry.Model, loaded: *const loader.Loaded, tag: []const u8) u64 {
    return prompt_cache_mod.backendFingerprint(
        prompt_cache_mod.modelFingerprint(model.pub_name, loaded.map.bytes().len),
        tag,
    );
}

/// One family tag qualified by the KV dtype. The f32 tag is the family name byte
/// for byte, so every namespace a shipped blob lives in survives, and a narrower
/// dtype gets its own namespace, so the disk tier never crosses dtypes.
/// `family_tag_buf_len` is the bound that makes the format here infallible.
fn familyTag(buf: []u8, family: []const u8, kv_dtype: dtype.KvDtype) []const u8 {
    std.debug.assert(buf.len >= family.len + dtype.kv_tag_suffix_max);
    return std.fmt.bufPrint(buf, "{s}{s}", .{ family, kv_dtype.tagSuffix() }) catch unreachable;
}

fn windowSource(window: u32, ceiling: u32, flag: ?u32) WindowSource {
    std.debug.assert(window >= 1);
    std.debug.assert(window <= ceiling);
    if (flag) |w| {
        std.debug.assert(w == window);
        return .flag;
    }
    return if (window == ceiling) .ceiling else .budget;
}

/// The one VRAM figure every device arm plans against: the live budget of
/// `VK_EXT_memory_budget` (budget minus usage). A driver without the extension
/// gives the heap size instead, and one line per process says so, because the
/// heap size overstates what a shared device has free.
pub fn vramFigure(ctxp: anytype) u64 {
    if (ctxp.deviceLocalBudget() == null and !heap_fallback_logged) {
        heap_fallback_logged = true;
        log.info("vram: the driver states no live budget (VK_EXT_memory_budget), so the plan reads the heap size", .{});
    }
    return ctxp.deviceLocalAvailableBytes();
}

/// The heap-size line of `vramFigure` prints once per process.
var heap_fallback_logged = false;

/// The CPU arm holds every session resident, and its plan funds no blob. Fewer
/// slots than sessions then park a turn with no eviction to free a slot, so
/// the boot refuses the pair.
fn refuseCpuSlots(io: Io, n_slots: u16, max_sessions: u16) error{UsageError}!void {
    if (n_slots >= max_sessions) return;
    return args.fail(io, "the CPU runner holds one slot per session, so --vram-slots {d} cannot serve --max-sessions {d}. Drop --vram-slots, or set it to the --max-sessions value", .{ n_slots, max_sessions });
}

/// The weight quant the precision line names: the type of the first block's down
/// projection, which every Llama and every Qwen3.5 block holds. It is the same
/// per-layer fact the fused runners select their matmul kernels on, read from one
/// tensor rather than from a census.
fn weightQuantName(loaded: *const loader.Loaded) []const u8 {
    const t = tensor_lookup.find(loaded.tensors, "blk.0.ffn_down.weight") orelse return "unknown";
    return std.enums.tagName(dtype.GgmlType, t.dtype) orelse "unknown";
}

/// Set the two scheduler fields that depend on the runner: the rule and the
/// row cap of one pass. `width` is the rows of one prompt chunk that the runner
/// holds. The derived rule takes every row the runner holds, without a note. An
/// explicit chunk that overshoots the runner gets the note from `clampStepRows`.
/// The spare-rows rule keeps the original size.
fn setStepShape(cfg: *scheduler_mod.Config, flag: ?u32, requested: u32, runner_cap: u32, width: u32) void {
    cfg.prefill = prefill_chunk.fromFlag(flag, width);
    cfg.max_step_rows = switch (cfg.prefill) {
        .derived => runner_cap,
        .spare_rows, .fixed => clampStepRows(requested, runner_cap),
    };
}

/// The scheduler must never compose more rows than the runner's row-scratch
/// holds. CPU runners size that to `max_step_rows` exactly, where the clamp does
/// nothing. The fused-GPU runners cap it at their fixed `prefill_batch_max`, so
/// this function reduces an explicit `--prefill-chunk` that overshoots, with a
/// note.
fn clampStepRows(requested: u32, runner_cap: u32) u32 {
    if (runner_cap < requested)
        log.info("prefill chunk capped at {d} rows/pass by the GPU runner's buffers (requested {d})", .{ runner_cap, requested });
    return @min(requested, runner_cap);
}

/// Wrap a runner's session pool in the tiered-KV residency, then launch. Until
/// `--vram-slots` makes physical slots scarcer than `max_sessions` the residency
/// runs identity, with every session resident, so the behavior stays unchanged.
/// `tier` is the blob tier the RAM quote of the arm resolved.
fn serveWithResidency(
    comptime RunnerT: type,
    ctx: cli.Context,
    runner: *RunnerT,
    max_sessions: u16,
    tier: BlobTier,
    parts: CodecParts,
    sched_cfg: scheduler_mod.Config,
    launcher: anytype,
) !void {
    std.debug.assert(sched_cfg.max_seq >= 1);
    // Every arm reaches this point with its final window, so one check covers
    // them all. The cost is that a refused boot has uploaded the weights.
    if (!thinkingBudgetFits(sched_cfg.think.budget, sched_cfg.max_seq))
        return args.fail(ctx.io, thinking_budget_past_window_fmt, .{ sched_cfg.think.budget, sched_cfg.max_seq });
    const SessionsT = Sessions(RunnerT);
    var sessions = try SessionsT.init(ctx.arena, runner);
    var residency = try Residency(SessionsT).init(ctx.arena, ctx.gpa, &sessions, max_sessions, tier.budget());
    defer residency.deinit();
    return launcher.serve(Residency(SessionsT), ctx, &residency, parts, sched_cfg);
}

/// The boot lines of one RAM quote: the spend, then the blob tier. An absent
/// tier prints no tier line, because no blob can exist.
pub fn logRamQuote(q: kv_budget.RamQuote, kv_ram_named: bool) void {
    const mib = 1024 * 1024;
    const r = q.ram;
    if (q.usable) |usable| {
        log.info("ram: {d} of {d} MiB usable = {d} MiB host kv + {d} MiB blobs + {d} MiB prefix + {d} MiB stages", .{
            r.spend() / mib, usable / mib, r.host_kv / mib, r.blobs / mib, r.prefix / mib, r.swap_stages / mib,
        });
    } else {
        log.info("ram: {d} MiB, not checked = {d} MiB host kv + {d} MiB blobs + {d} MiB prefix + {d} MiB stages", .{
            r.spend() / mib, r.host_kv / mib, r.blobs / mib, r.prefix / mib, r.swap_stages / mib,
        });
    }
    switch (q.tier) {
        .absent => {},
        .unenforced => log.info("kv ram tier: unenforced, {d} MiB need ({d} evictable sessions + 1 in a swap, {d} MiB each). Set --kv-ram to bound it", .{
            q.need / mib, q.evictable, q.blob / mib,
        }),
        .bounded => |bytes| {
            log.info("kv ram tier: bounded, {d} MiB for {d} evictable sessions (need {d} MiB, {d} MiB a blob)", .{
                bytes / mib, q.evictable, q.need / mib, q.blob / mib,
            });
            // A named budget below the need is a choice of backpressure, so
            // the boot accepts it and says what it costs.
            if (kv_ram_named and bytes < q.need) log.warn("kv ram tier: --kv-ram funds {d} MiB of blobs, but the need is {d} MiB. A turn can wait until another session closes", .{
                bytes / mib, q.need / mib,
            });
        },
    }
}

/// The host bytes of the sampler state of one serve boot: two id sets per
/// scheduler slot, and one sample scratch for the session pool.
const SamplerRam = struct {
    sessions: u32,
    per_slot: u64,
    scratch: u64,

    fn total(self: SamplerRam) u64 {
        return @as(u64, self.sessions) * self.per_slot + self.scratch;
    }
};

fn samplerRam(vocab_size: u32, sessions: u32) SamplerRam {
    std.debug.assert(vocab_size >= 1);
    std.debug.assert(sessions >= 1);
    return .{
        .sessions = sessions,
        .per_slot = scheduler_mod.slotSetBytes(vocab_size),
        .scratch = sampler.scratchLen(vocab_size) * @sizeOf(f32),
    };
}

/// The sampler state sits outside the KV quote, so it gets a RAM line of its
/// own. Its size follows the vocab, not the window.
fn logSamplerRam(s: SamplerRam) void {
    const kib = 1024;
    log.info("ram: sampler {d} KiB = {d} sessions x {d} KiB id sets + {d} KiB scratch", .{
        s.total() / kib, s.sessions, s.per_slot / kib, s.scratch / kib,
    });
}

fn kvMiB(runner: anytype) u64 {
    var total: usize = 0;
    for (runner.kv_slots) |*slot| total += slot.sizeBytes();
    return total >> 20;
}

fn msBetween(io: Io, t0: Io.Clock.Timestamp) f64 {
    return time.msBetween(t0, Io.Clock.Timestamp.now(io, .awake));
}

fn printReady(io: Io, comptime fmt: []const u8, fmt_args: anytype) !void {
    var buf: [256]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stdout(), io, &buf);
    const w = &fw.interface;
    try w.print(fmt, fmt_args);
    try w.flush();
}

// -- tests ------------------------------------------------------------------

test "groupDigits: groups of three from the right, as the Zig source writes them" {
    try std.testing.expectEqualStrings("0", comptime groupDigits(0));
    try std.testing.expectEqualStrings("8", comptime groupDigits(8));
    try std.testing.expectEqualStrings("999", comptime groupDigits(999));
    try std.testing.expectEqualStrings("1_000", comptime groupDigits(1_000));
    try std.testing.expectEqualStrings("250_000", comptime groupDigits(250_000));
    try std.testing.expectEqualStrings("12_345_678", comptime groupDigits(12_345_678));
}

test "serve capacity: the resolved dtype prices the plan and the ledger" {
    // Llama-3.1-8B: 32 layers stepping 8 KV heads x 128 head_dim. Every cost
    // model in the tree is this one product, so the registry anchors 262144 and
    // 131072 derive from the dtype geometry rather than from a literal.
    const n_layers: u32 = 32;
    const per_step: u32 = 8 * 128;
    const wide: kv_budget.KvCostModel = .{
        .kv_bytes_per_token = @as(u64, 2) * n_layers * dtype.KvDtype.f32.rowBytes(per_step),
    };
    const half: kv_budget.KvCostModel = .{
        .kv_bytes_per_token = @as(u64, 2) * n_layers * dtype.KvDtype.f16.rowBytes(per_step),
    };
    try std.testing.expectEqual(@as(u64, 262_144), wide.kv_bytes_per_token);
    try std.testing.expectEqual(@as(u64, 131_072), half.kv_bytes_per_token);

    // `--kv-vram 1G` at the serve default window. The flag states bytes, so
    // the dtype spends them on breadth: half the price, twice the slots.
    // The byte rule of Llama-8B at f32 gives the two default windows: 512 at the
    // serve budget and 8192 at the run budget. The request names its window.
    const budget_caps: kv_budget.Caps = .{
        .vram_free = 0,
        .weights = 0,
        .logits = 0,
        .model_ctx = 131_072,
        .serve_default = 512,
        .default_ceil = 8192,
    };
    const req: kv_budget.Request = .{ .x_bytes = 1 << 30, .window = 512, .batch = 8 };
    const wide_plan = try kv_budget.kvPlan(budget_caps, wide, req);
    const half_plan = try kv_budget.kvPlan(budget_caps, half, req);
    try std.testing.expectEqual(@as(u32, 8), wide_plan.n_slots);
    try std.testing.expectEqual(2 * wide_plan.n_slots, half_plan.n_slots);
    try std.testing.expectEqual(wide_plan.window, half_plan.window);

    // `--gpu-layers 16` over the same shape, priced the way the hybrid arm above
    // prices it: one uniform cost vector out of the same geometry. Half the
    // element size halves every KV figure the ledger prints, and moves no other
    // one. The split and the resident weights ride through equal.
    const mib = 1024 * 1024;
    const weights = [_]u64{250 * mib} ** n_layers;
    const head_bytes: u64 = 500 * mib;
    // The placement planner reads neither default window.
    const caps: kv_budget.Caps = .{
        .vram_free = 16 << 30,
        .weights = 0,
        .logits = 0,
        .model_ctx = 131_072,
        .serve_default = limits.min_kv_window,
        .default_ceil = limits.min_kv_window,
    };
    // A serve arm prices both stages, and the placement planner checks no RAM.
    const serve_host: kv_budget.HostRequest = .{ .ram_split = .{ .blobs = 0, .prefix = 0 }, .residency = true };
    const wide_costs = kv_budget.KvCostVector.uniform(n_layers, @as(u64, 2) * dtype.KvDtype.f32.rowBytes(per_step));
    const half_costs = kv_budget.KvCostVector.uniform(n_layers, @as(u64, 2) * dtype.KvDtype.f16.rowBytes(per_step));
    var ledger: [2]kv_budget.PlacementPlan = undefined;
    for (&ledger, [_]kv_budget.KvCostVector{ wide_costs, half_costs }) |*p, costs| {
        p.* = try kv_budget.placementPlan(caps, &weights, head_bytes, .{
            .n_slots = 4,
            .window = 512,
            .costs = costs,
        }, serve_host, 16);
    }
    // 4 slots x 512 rows x 16 layers x 8192 bytes is 256 MiB on each tier at
    // f32, and one slot's suffix pair stages 128 MiB of that.
    try std.testing.expectEqual(@as(u64, 256 * mib), ledger[0].kv_vram);
    try std.testing.expectEqual(@as(u64, 256 * mib), ledger[0].ram.host_kv);
    try std.testing.expectEqual(@as(u64, 128 * mib), ledger[0].ram.swap_stages);
    try std.testing.expectEqual(ledger[0].kv_vram / 2, ledger[1].kv_vram);
    try std.testing.expectEqual(ledger[0].ram.host_kv / 2, ledger[1].ram.host_kv);
    try std.testing.expectEqual(ledger[0].ram.swap_stages / 2, ledger[1].ram.swap_stages);
    try std.testing.expectEqual(ledger[0].split, ledger[1].split);
    try std.testing.expectEqual(ledger[0].weight_resident, ledger[1].weight_resident);
}

test "windowSource: a flag names its window, and a default names the ceiling or the budget that bound it" {
    const ceiling: u32 = 8192;
    try std.testing.expectEqual(WindowSource.ceiling, windowSource(ceiling, ceiling, null));
    try std.testing.expectEqual(WindowSource.budget, windowSource(2816, ceiling, null));
    // A named window is the flag, even at the ceiling or at the default value.
    for ([_]u32{ 1, 2816, ceiling }) |w| {
        try std.testing.expectEqual(WindowSource.flag, windowSource(w, ceiling, w));
    }
    // The smallest ceiling holds one position, and a default there is the ceiling.
    try std.testing.expectEqual(WindowSource.ceiling, windowSource(1, 1, null));
}

test "baseWindow: a named window must lie in 1..ceiling, and no flag takes the serve default" {
    const ceiling: u32 = 8192;
    const serve_default: u32 = 2816;
    try std.testing.expectError(error.WindowOutOfRange, baseWindow(0, ceiling, serve_default));
    try std.testing.expectEqual(@as(u32, 1), try baseWindow(1, ceiling, serve_default));
    try std.testing.expectEqual(ceiling, try baseWindow(ceiling, ceiling, serve_default));
    try std.testing.expectError(error.WindowOutOfRange, baseWindow(ceiling + 1, ceiling, serve_default));
    try std.testing.expectEqual(serve_default, try baseWindow(null, ceiling, serve_default));
}

test "baseWindow: a ceiling below the session budget window is the default window" {
    // Qwen3.5-2B at f16 prices 12288 bytes per position, so the serve budget
    // alone buys 10752 positions. A ceiling of 4096 binds first.
    const cost: kv_budget.KvCostModel = .{ .kv_bytes_per_token = 12_288 };
    try std.testing.expectEqual(@as(u32, 10_752), kv_budget.defaultWindow(cost, limits.max_model_ctx, limits.serve_session_kv_bytes));
    const ceiling: u32 = 4096;
    const serve_default = kv_budget.defaultWindow(cost, ceiling, limits.serve_session_kv_bytes);
    try std.testing.expectEqual(ceiling, try baseWindow(null, ceiling, serve_default));
    try std.testing.expectEqual(ceiling, try baseWindow(ceiling, ceiling, serve_default));
    try std.testing.expectError(error.WindowOutOfRange, baseWindow(ceiling + 1, ceiling, serve_default));
}

test "thinkingBudgetFits: a budget must sit below the window, and zero is unbounded" {
    const window: u32 = 512;
    try std.testing.expect(thinkingBudgetFits(0, window));
    try std.testing.expect(thinkingBudgetFits(window - 1, window));
    try std.testing.expect(!thinkingBudgetFits(window, window));
    try std.testing.expect(!thinkingBudgetFits(window + 1, window));
    // The smallest window fits only the unbounded budget.
    try std.testing.expect(thinkingBudgetFits(0, 1));
    try std.testing.expect(!thinkingBudgetFits(1, 1));
}

test "thinkTagsFrom: one id per tag, the close first, and no end-of-sequence id" {
    // The Qwen3.5 ids: the two tags, and `\n\n`, the separator the forced
    // close carries on that vocab.
    const open = [_]u32{248068};
    const close = [_]u32{248069};
    const sep: u32 = 271;
    const eos: u32 = 248046;
    const tags = try thinkTagsFrom(&open, &close, &.{ 248069, sep }, eos);
    try std.testing.expectEqual(@as(u32, 248068), tags.ids.open);
    try std.testing.expectEqual(@as(u32, 248069), tags.ids.close);
    try std.testing.expectEqualSlices(u32, &.{ 248069, sep }, tags.force[0..tags.force_len]);

    // Each refusal below breaks one rule.
    const bad = error.ThinkTagsUnusable;
    try std.testing.expectError(bad, thinkTagsFrom(&.{ 27, 28 }, &close, &.{ 248069, sep }, eos));
    try std.testing.expectError(bad, thinkTagsFrom(&open, &.{ 27, 28 }, &.{ 27, 28 }, eos));
    try std.testing.expectError(bad, thinkTagsFrom(&open, &close, &.{ sep, 248069 }, eos));
    try std.testing.expectError(bad, thinkTagsFrom(&open, &close, &.{}, eos));
    try std.testing.expectError(bad, thinkTagsFrom(&open, &close, &.{ 248069, sep, sep, sep, sep }, eos));
    try std.testing.expectError(bad, thinkTagsFrom(&open, &.{eos}, &.{eos}, eos));
    try std.testing.expectError(bad, thinkTagsFrom(&open, &close, &.{ 248069, eos }, eos));
    // The bound itself is legal.
    const widest = try thinkTagsFrom(&open, &close, &.{ 248069, sep, sep, sep }, eos);
    try std.testing.expectEqual(limits.think_force_ids_max, widest.force_len);
}

test "serve capacity: a qwen split prices each tier by the layers it holds" {
    // The 2B geometry: 24 layers, every fourth one full attention. The serve arm
    // builds this structure through `kv_cost.costVector` and hands it to the
    // same planner, so the arithmetic below is the arm's.
    const n_layers: u32 = 24;
    var costs = kv_budget.KvCostVector.uniform(n_layers, 0);
    var i: u32 = 0;
    while (i < n_layers) : (i += 1) {
        if ((i + 1) % 4 == 0) costs.set(i, 4_096, 0) else costs.set(i, 0, 1_122_304);
    }
    const mib = 1024 * 1024;
    const weights = [_]u64{62 * mib} ** n_layers;
    const head_bytes: u64 = 540 * mib;
    // The placement planner reads neither default window.
    const caps: kv_budget.Caps = .{
        .vram_free = 16 << 30,
        .weights = 0,
        .logits = 0,
        .model_ctx = 262_144,
        .serve_default = limits.min_kv_window,
        .default_ceil = limits.min_kv_window,
    };
    // A serve arm prices both stages, and the placement planner checks no RAM.
    const serve_host: kv_budget.HostRequest = .{ .ram_split = .{ .blobs = 0, .prefix = 0 }, .residency = true };
    const kv: kv_budget.KvStructure = .{ .n_slots = 4, .window = 512, .costs = costs };

    // Twelve GPU layers leave the split at 12. Each half then holds three
    // attention layers and nine recurrent ones, so both ledgers read equal. A
    // uniform price cannot produce that.
    const plan = try kv_budget.placementPlan(caps, &weights, head_bytes, kv, serve_host, 12);
    const half: u64 = @as(u64, kv.n_slots) * (@as(u64, kv.window) * 3 * 4_096 + 9 * 1_122_304);
    try std.testing.expectEqual(@as(u32, 12), plan.split);
    try std.testing.expectEqual(half, plan.kv_vram);
    try std.testing.expectEqual(half, plan.ram.host_kv);
    // The tied head rides any non-zero suffix, so a split of one layer already
    // pays it whole. That is the cliff the refusal message names.
    const thin = try kv_budget.placementPlan(caps, &weights, head_bytes, kv, serve_host, 1);
    try std.testing.expectEqual(@as(u64, 62 * mib) + head_bytes, thin.weight_resident);
    // A prefix that holds no attention layer charges fixed slabs and no window
    // price: layers 0, 1, and 2 are all recurrent on this stack. It is the
    // boundary value where the host tier's KV height is zero.
    const gdn_only = try kv_budget.placementPlan(caps, &weights, head_bytes, kv, serve_host, n_layers - 3);
    try std.testing.expectEqual(@as(u32, 3), gdn_only.split);
    try std.testing.expectEqual(@as(u64, kv.n_slots) * 3 * 1_122_304, gdn_only.ram.host_kv);
}

test "samplerRam: the Qwen vocab prices two id sets per session and one scratch" {
    // 248320 ids fill 3880 words, so a set takes 31040 bytes. The scratch
    // holds one row and 3880 block sums of f32.
    const s = samplerRam(248320, 4);
    try std.testing.expectEqual(@as(u64, 2 * 31040), s.per_slot);
    try std.testing.expectEqual(@as(u64, (248320 + 3880) * 4), s.scratch);
    try std.testing.expectEqual(@as(u64, 4 * 62080 + 1008800), s.total());
}
