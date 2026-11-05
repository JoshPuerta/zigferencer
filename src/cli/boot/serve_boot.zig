//! The serve boot that `serve` and `serve-openai` share. `buildAndServe` checks the flags,
//! loads the model, builds the runner of its family on the selected placement, wraps it in
//! the tiered-KV residency, and hands it to a launcher: a value with a `serve(EngineT, ...)`
//! method.

const std = @import("std");
const Io = std.Io;
const build_options = @import("build_options");

const context = @import("../context.zig");
const args = @import("../args.zig");
const flags_mod = @import("../flag_groups.zig");
const select = @import("../select.zig");
const family = @import("../../engine/family.zig");
const resolve_mod = @import("../../engine/resolve.zig");
const chat = @import("../../core/chat.zig");
const dtype = @import("../../core/dtype.zig");
const kv_budget = @import("../../core/kv_budget.zig");
const limits = @import("../../core/limits.zig");
const registry = @import("../../core/registry.zig");
const time = @import("../../core/time.zig");
const vocab_mod = @import("../../core/tokenizer/vocab.zig");
const loader = @import("../../io/gguf/loader.zig");
const parallel = @import("../../runtime/parallel.zig");
const Pool = @import("../../runtime/pool.zig").Pool;
const prompt_cache_mod = @import("../../runtime/prompt_cache.zig");
const sampler = @import("../../runtime/sampler.zig");
const Sessions = @import("../../runtime/session.zig").Sessions;
const prefill_chunk = @import("../../serve/prefill_chunk.zig");
const protocol = @import("../../serve/protocol.zig");
const Residency = @import("../../serve/residency.zig").Residency;
const scheduler_mod = @import("../../serve/scheduler.zig");
// The device context resolves to `void` on a CPU build, which never analyzes it.
const vulkan_ctx_mod = if (fused_build) @import("../../runtime/vulkan_ctx.zig") else void;

const log = std.log.scoped(.serve);
const fused_build = build_options.use_vulkan_fused_ops;
const BlobTier = kv_budget.BlobTier;
const device_arm = resolve_mod.device_arm;

const boot_facts = @import("boot_facts.zig");
const boot_lines = @import("boot_lines.zig");
const codec_parts = @import("codec_parts.zig");
const serve_checks = @import("serve_checks.zig");

// -- constants --------------------------------------------------------------

const kib = 1024;

const mib = 1024 * 1024;

/// Bytes the widest single-tier family tag occupies: the longest family name and
/// the widest KV dtype component.
const family_tag_buf_len: usize = "vulkan_fused".len + dtype.kv_tag_suffix_max;

/// Bytes of the one READY line.
/// The refusal of a thinking budget that no turn can spend. It takes the budget
/// and the window.
const thinking_budget_past_window_fmt = "--thinking-budget {d} is at or past the window of " ++
    "{d} positions, so it can never close the block. Use a smaller budget or a larger --max-seq";

// -- types ------------------------------------------------------------------

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

/// What every placement of one serve boot reads.
const ServeBoot = struct {
    ctx: context.Context,
    flags: *const flags_mod.ServeFlags,
    checked: *const serve_checks.CheckedFlags,
    model: *const registry.Model,
    loaded: *const loader.Loaded,
    resolved: *const resolve_mod.Resolved,
    device: boot_facts.DeviceBoot,
    pool: *Pool,
    policy: parallel.Policy,
    parts: codec_parts.CodecParts,

    fn kvDtype(self: *const ServeBoot) dtype.KvDtype {
        return self.resolved.early.kv_dtype;
    }

    fn kvRamNamed(self: *const ServeBoot) bool {
        return self.checked.shape.kv_ram_bytes != null;
    }

    /// The fingerprint of the runner `tag` names, so a blob never restores across
    /// a family, a split, or a dtype.
    fn fingerprint(self: *const ServeBoot, tag: []const u8) u64 {
        return familyFp(self.model, self.loaded, tag);
    }
};

/// An open device context and the time of the start of the device boot.
const OpenDevice = if (fused_build) struct {
    ctxp: *vulkan_ctx_mod.Context,
    t_init: Io.Clock.Timestamp,
} else void;

/// What a built runner fixes for the scheduler.
const RunnerFit = struct {
    /// The window of the plan, which the runner must hold.
    window: u32,
    /// The rows of one prompt chunk that the runner holds.
    prefill_width: u32,
    prefix_ram_bytes: u64,
    tier: BlobTier,
};

// -- api --------------------------------------------------------------------

/// Load the model, build the engine of its family on the placement that the flags
/// and the device select, and hand it to `launcher.serve`.
pub fn buildAndServe(
    ctx: context.Context,
    pub_name: ?[]const u8,
    serve: *const flags_mod.ServeFlags,
    wire: serve_checks.Wire,
    launcher: anytype,
) !void {
    const checked = try serve_checks.checkFlags(ctx, serve, wire);

    // The whole-model fused runner runs no CPU op, so its pool starts no workers.
    // A pin of any layer count can run CPU layers.
    const runs_cpu_ops = !fused_build or serve.engine.gpu_layers != null;
    const policy = try boot_facts.resolvePolicy(
        ctx.io,
        serve.engine.threads.policyFlags(),
        runs_cpu_ops,
    );

    const model = try select.model(ctx, pub_name);
    const input: resolve_mod.Input = .{ .engine = serve.engine, .serve = checked.shape };
    const entry = boot_facts.entryOf(model);

    // Every refusal that needs no model file lands here, before the load.
    switch (resolve_mod.early(.serve, input, entry)) {
        .ok => {},
        .refused => |r| return boot_lines.refuse(ctx.io, model.pub_name, r),
    }

    const path = try registry.resolveLoadPath(ctx.arena, ctx.io, ctx.models_root, model.*);
    var loaded = try loader.load(ctx.io, ctx.arena, path);
    defer loaded.close();

    const resolved = try boot_facts.resolveLoaded(ctx, .serve, input, entry, &loaded);

    // One pool serves every CPU op of the command.
    var pool: Pool = undefined;
    try parallel.initPool(&pool, ctx.io, policy);
    defer pool.deinit();

    var v = try vocab_mod.buildFromParsed(ctx.arena, loaded.parsed);
    const pieces = try codec_parts.buildPieces(ctx.arena, &v);
    const kind = resolved.values.early.chat_kind;
    const thinking_budget = serve.engine.thinking_budget;
    const think_tags = try codec_parts.thinkTagsOf(ctx.arena, &v, kind);
    log.info("thinking: {s}, budget {d}", .{
        if (chat.thinkOpen(kind)) "on" else "off",
        thinking_budget,
    });

    const boot: ServeBoot = .{
        .ctx = ctx,
        .flags = serve,
        .checked = &checked,
        .model = model,
        .loaded = &loaded,
        .resolved = &resolved.values,
        .device = resolved.device,
        .pool = &pool,
        .policy = policy,
        .parts = codec_parts.codecParts(&v, pieces, kind, think_tags),
    };

    const think = codec_parts.schedulerThink(think_tags, kind, thinking_budget);
    const sched_cfg = schedulerConfig(&boot, wire, think);
    logSamplerRam(samplerRam(loaded.card.vocab_size, serve.max_sessions));
    boot_lines.logPrecision(&loaded, boot.kvDtype());

    switch (loaded.card.arch) {
        inline else => |arch| return serveFamily(family.Of(arch), &boot, sched_cfg, launcher),
    }
}

/// The one stdout line a frontend prints before its engine loop. The spawning
/// process parses the transport and the address from it.
pub fn printReadyTcp(w: *Io.Writer, addr: Io.net.IpAddress) !void {
    try printReady(w, "READY tcp {f}\n", .{addr});
}

pub fn printReadyUnix(w: *Io.Writer, path: []const u8) !void {
    try printReady(w, "READY unix {s}\n", .{path});
}

// -- helpers ----------------------------------------------------------------

/// One scheduler config, built from the facts every placement shares. A placement
/// sets the fields that depend on its runner, and `launch` asserts them.
fn schedulerConfig(
    b: *const ServeBoot,
    wire: serve_checks.Wire,
    think: scheduler_mod.Think,
) scheduler_mod.Config {
    const checked = b.checked;
    const model = b.model;

    // A tag without a device names the CPU runner. A fused boot starts in the
    // fused namespace, and each placement names its own.
    var tag_buffer: [family_tag_buf_len]u8 = undefined;
    const boot_family = if (fused_build) "vulkan_fused" else "cpu";
    const boot_tag = familyTag(&tag_buffer, boot_family, b.kvDtype());

    var cfg: scheduler_mod.Config = .{
        .max_sessions = b.flags.max_sessions,
        .sessions_per_connection = checked.sessions_per_connection,
        .max_batch = checked.shape.batch,
        .max_seq = 0, // set by `launch`
        .max_step_rows = 0, // set by `launch`
        .prefill = checked.cpu_prefill,
        .max_connections = b.flags.max_connections,
        .allow_shutdown = wire.allow_shutdown,
        .resume_ttl_max_ms = wire.resume_ttl_ms,
        .model_name = model.pub_name,
        .vocab_size = b.loaded.card.vocab_size,
        .eos_id = b.parts.eos_id,
        .cache = .{
            .disk_bytes = checked.prefix_disk_bytes,
            .dir = b.ctx.cache_dir,
            .model_fp = b.fingerprint(boot_tag),
            .preload = checked.preload,
        },
        .think = think,
    };

    // `hello_ack` carries the card row of each think mode, so a client picks
    // the row of the mode of each turn.
    if (registry.recommendedSampling(model.pub_name, true)) |r| {
        cfg.recommended_think = wireSampling(r);
    }

    if (registry.recommendedSampling(model.pub_name, false)) |r| {
        cfg.recommended_plain = wireSampling(r);
    }

    return cfg;
}

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

/// Pick the placement of `Family`. `--gpu-layers 0` and an auto split that funds no
/// GPU layer take the CPU runner, and only the first of them skips the device.
fn serveFamily(
    comptime Family: type,
    b: *const ServeBoot,
    cfg: scheduler_mod.Config,
    launcher: anytype,
) !void {
    var host_arm = b.resolved.cpu;
    const cpu_pinned = if (b.flags.engine.gpu_layers) |g| g.isZeroCount() else false;
    if (cpu_pinned) {
        log.info("--gpu-layers 0: the {s} CPU runner decodes every layer", .{Family.name});
    }

    if (comptime fused_build) {
        if (!cpu_pinned) {
            switch (try serveOnDevice(Family, b, cfg, launcher)) {
                .done => return,
                .all_host => |arm| host_arm = arm,
            }
        }
    }

    return serveCpu(Family, b, host_arm.?, cfg, launcher);
}

/// Open the device, resolve the arm on it, and serve a fused or hybrid arm. The
/// context stays open until the launcher returns, and `.all_host` releases it.
fn serveOnDevice(
    comptime Family: type,
    b: *const ServeBoot,
    cfg: scheduler_mod.Config,
    launcher: anytype,
) !boot_facts.Placed {
    const ctx = b.ctx;
    const name = b.model.pub_name;

    const t_init = Io.Clock.Timestamp.now(ctx.io, .awake);
    if (b.flags.engine.gpu_layers == null) {
        log.info("initializing GPU: context + weight upload to VRAM...", .{});
    } else {
        log.info("initializing GPU: context + suffix weight upload to VRAM...", .{});
    }

    const ctxp = try boot_facts.openDevice(ctx);
    defer ctxp.deinit();

    const dev = try b.device.resolveOn(ctx.io, ctxp);
    if (dev.placed) |placed| {
        boot_lines.logPlaced(placed, b.loaded.card.n_layers);
    }

    const arm = switch (dev.arm) {
        .ok => |a| a,
        .refused => |r| return boot_lines.refuse(ctx.io, name, r),
    };

    const open: OpenDevice = .{ .ctxp = ctxp, .t_init = t_init };
    switch (arm) {
        .cpu => |c| {
            log.info("--gpu-layers auto: the {s} CPU runner decodes every layer", .{Family.name});
            return .{ .all_host = c };
        },
        .hybrid => |h| try serveHybrid(Family, b, open, h, cfg, launcher),
        .fused => |f| try serveFused(Family, b, open, f, cfg, launcher),
    }

    return .done;
}

fn serveHybrid(
    comptime Family: type,
    b: *const ServeBoot,
    open: OpenDevice,
    hybrid_arm: device_arm.Hybrid,
    cfg_in: scheduler_mod.Config,
    launcher: anytype,
) !void {
    const ctx = b.ctx;
    const kv_dtype = b.kvDtype();
    const n_layers = b.loaded.card.n_layers;

    const quote = switch (hybrid_arm.quote) {
        .ok => |q| q,
        .refused => |r| return boot_lines.refuse(ctx.io, b.model.pub_name, r),
    };

    boot_lines.logRamQuote(quote, b.kvRamNamed());

    const plan = hybrid_arm.plan.?;
    std.debug.assert(plan.split > 0);
    std.debug.assert(plan.split < n_layers);

    var ledger_buffer: [kv_budget.placement_ledger_text_max]u8 = undefined;
    log.info("{s}", .{kv_budget.formatLedger(plan, &ledger_buffer)});

    const hybrid = try Family.Hybrid.init(
        ctx.gpa,
        open.ctxp,
        b.loaded,
        plan,
        ctx.io,
        b.pool,
        b.policy,
        kv_dtype,
    );
    defer hybrid.deinit();

    log.info("{s} hybrid engine ready in {d:.0} ms: logical={d} vram={d} batch={d} max_seq={d} " ++
        "gpu={d}/{d} layers kv={d} MiB vram + {d} MiB host", .{
        Family.name,
        msBetween(ctx.io, open.t_init),
        b.flags.max_sessions,
        plan.n_slots,
        b.checked.shape.batch,
        plan.window,
        n_layers - plan.split,
        n_layers,
        plan.kv_vram / mib,
        plan.ram.host_kv / mib,
    });

    // The tag names the family, the split, and the dtype.
    var cfg = cfg_in;
    var tag_buffer: [Family.Hybrid.backend_tag_buf_len]u8 = undefined;
    cfg.cache.model_fp = b.fingerprint(
        Family.hybridTag(&tag_buffer, hybrid, plan.split, kv_dtype),
    );

    const fit: RunnerFit = .{
        .window = plan.window,
        .prefill_width = hybrid.maxStepRows(),
        .prefix_ram_bytes = plan.ram.prefix,
        .tier = quote.tier,
    };
    return launch(Family.Hybrid, b, hybrid, fit, cfg, launcher);
}

fn serveFused(
    comptime Family: type,
    b: *const ServeBoot,
    open: OpenDevice,
    f: device_arm.Fused,
    cfg: scheduler_mod.Config,
    launcher: anytype,
) !void {
    const ctx = b.ctx;
    const kv_dtype = b.kvDtype();
    const name = b.model.pub_name;

    const quote = try boot_lines.logFused(ctx.io, name, f, false, b.kvRamNamed(), kv_dtype);

    const plan = f.plan;
    const ceiling = b.resolved.window.ceiling;
    const cost = kv_budget.costModel(b.loaded.card.kvShape(), kv_dtype);
    std.debug.assert(f.caps.serve_default == kv_budget.defaultWindows(cost, ceiling).serve);

    const fused = try Family.initFused(ctx.gpa, open.ctxp, b.loaded, plan, kv_dtype);
    defer fused.deinit();

    log.info("{s} gpu engine ready in {d:.0} ms: logical={d} vram={d} batch={d} max_seq={d}", .{
        Family.name,
        msBetween(ctx.io, open.t_init),
        b.flags.max_sessions,
        fused.n_slots,
        b.checked.shape.batch,
        fused.max_seq,
    });

    const fit: RunnerFit = .{
        .window = plan.window,
        .prefill_width = fused.maxStepRows(),
        .prefix_ram_bytes = plan.ram.prefix,
        .tier = quote.tier,
    };
    return launch(Family.Fused, b, fused, fit, cfg, launcher);
}

fn serveCpu(
    comptime Family: type,
    b: *const ServeBoot,
    arm_result: resolve_mod.Result(resolve_mod.CpuArm),
    cfg_in: scheduler_mod.Config,
    launcher: anytype,
) !void {
    const ctx = b.ctx;
    const kv_dtype = b.kvDtype();
    const n_sessions = b.flags.max_sessions;

    const arm = try cpuArmOf(ctx.io, b.model.pub_name, arm_result, b.kvRamNamed(), kv_dtype);
    const t_init = Io.Clock.Timestamp.now(ctx.io, .awake);
    const plan = arm.plan;

    // The CPU runner writes these blobs, so a fused build moves them out of the
    // fused namespace that the boot fingerprint named.
    var cfg = cfg_in;
    if (comptime fused_build) {
        var tag_buffer: [family_tag_buf_len]u8 = undefined;
        cfg.cache.model_fp = b.fingerprint(familyTag(&tag_buffer, "cpu", kv_dtype));
    }

    var runner = try Family.Cpu.init(
        ctx.arena,
        b.loaded,
        plan,
        b.checked.max_step_rows,
        ctx.io,
        b.pool,
        b.policy,
        kv_dtype,
    );
    log.info("{s} engine ready in {d:.0} ms: sessions={d} batch={d} max_seq={d} kv={d} MiB", .{
        Family.name,
        msBetween(ctx.io, t_init),
        n_sessions,
        b.checked.shape.batch,
        runner.max_seq,
        kvMiB(&runner),
    });

    const fit: RunnerFit = .{
        .window = plan.window,
        .prefill_width = limits.cpu_prefill_chunk,
        .prefix_ram_bytes = plan.ram.prefix,
        .tier = arm.quote.tier,
    };
    return launch(Family.Cpu, b, &runner, fit, cfg, launcher);
}

/// The all-host serve arm with its boot lines: the prefix default, the plan,
/// and the RAM quote.
fn cpuArmOf(
    io: Io,
    name: []const u8,
    result: resolve_mod.Result(resolve_mod.CpuArm),
    kv_ram_named: bool,
    kv_dtype: dtype.KvDtype,
) !resolve_mod.CpuArm {
    const arm = switch (result) {
        .ok => |a| a,
        .refused => |r| return boot_lines.refuse(io, name, r),
    };

    if (arm.prefix_default) {
        boot_lines.logPrefixDefault();
    }

    boot_lines.logPlan(arm.plan, kv_dtype);
    boot_lines.logRamQuote(arm.quote, kv_ram_named);
    return arm;
}

/// The common tail of every placement: fit the scheduler to the runner, wrap the
/// session pool in the tiered-KV residency, and launch. Until `--vram-slots` makes
/// slots scarcer than sessions, the residency holds every session resident.
fn launch(
    comptime RunnerT: type,
    b: *const ServeBoot,
    runner: *RunnerT,
    fit: RunnerFit,
    cfg_in: scheduler_mod.Config,
    launcher: anytype,
) !void {
    const ctx = b.ctx;
    const cap = runner.maxStepRows();
    std.debug.assert(runner.max_seq == fit.window);

    var cfg = cfg_in;
    cfg.max_seq = runner.max_seq;
    setStepShape(&cfg, b.flags.prefill_chunk, b.checked.max_step_rows, cap, fit.prefill_width);
    cfg.cache.ram_bytes = fit.prefix_ram_bytes;
    std.debug.assert(cfg.max_seq >= 1);

    // Every placement reaches this point with its final window, so one check covers
    // them all. The cost is that a refused boot has uploaded the weights.
    if (!thinkingBudgetFits(cfg.think.budget, cfg.max_seq)) {
        return args.fail(ctx.io, thinking_budget_past_window_fmt, .{
            cfg.think.budget, cfg.max_seq,
        });
    }

    const SessionsT = Sessions(RunnerT);
    const max_sessions = b.flags.max_sessions;
    var sessions = try SessionsT.init(ctx.arena, runner);
    var residency = try Residency(SessionsT).init(
        ctx.arena,
        ctx.gpa,
        &sessions,
        max_sessions,
        fit.tier.budget(),
    );
    defer residency.deinit();

    return launcher.serve(Residency(SessionsT), ctx, &residency, b.parts, cfg);
}

/// Set the two scheduler fields that depend on the runner: the rule and the row
/// cap of one step. The derived rule takes every row the runner holds, and an
/// explicit chunk that overshoots the runner gets the note of `clampStepRows`.
fn setStepShape(
    cfg: *scheduler_mod.Config,
    flag: ?u32,
    requested: u32,
    runner_cap: u32,
    width: u32,
) void {
    cfg.prefill = prefill_chunk.fromFlag(flag, width);
    cfg.max_step_rows = switch (cfg.prefill) {
        .derived => runner_cap,
        .spare_rows, .fixed => clampStepRows(requested, runner_cap),
    };
}

/// The scheduler must never compose more rows than the row scratch of the runner
/// holds. CPU runners size it to the request, and the fused-GPU runners fix it at
/// `prefill_batch_max`.
fn clampStepRows(requested: u32, runner_cap: u32) u32 {
    if (runner_cap < requested) {
        log.info("prefill chunk capped at {d} rows/pass by the GPU runner's buffers " ++
            "(requested {d})", .{ runner_cap, requested });
    }

    return @min(requested, runner_cap);
}

/// Whether a thinking budget can ever force the close inside a window. Zero is
/// unbounded. The n-th think token lands at `p0 + n - 1`, the forced close needs
/// one more step, and the prompt holds at least one position.
fn thinkingBudgetFits(budget: u32, window: u32) bool {
    std.debug.assert(window >= 1);
    std.debug.assert(window <= limits.max_model_ctx);

    return budget == 0 or budget < window;
}

/// The weights identity of one runner family: the model fingerprint qualified by
/// the family tag. The numeric provenance of a blob is per backend.
fn familyFp(model: *const registry.Model, loaded: *const loader.Loaded, tag: []const u8) u64 {
    return prompt_cache_mod.backendFingerprint(
        prompt_cache_mod.modelFingerprint(model.pub_name, loaded.map.bytes().len),
        tag,
    );
}

/// One family tag qualified by the KV dtype. The f32 tag is the family name byte
/// for byte, so every namespace of an existing blob survives, and the disk tier
/// never crosses dtypes.
fn familyTag(buf: []u8, family_name: []const u8, kv_dtype: dtype.KvDtype) []const u8 {
    std.debug.assert(buf.len >= family_name.len + dtype.kv_tag_suffix_max);

    const suffix = kv_dtype.tagSuffix();
    return std.fmt.bufPrint(buf, "{s}{s}", .{ family_name, suffix }) catch unreachable;
}

fn samplerRam(vocab_size: u32, sessions: u32) SamplerRam {
    std.debug.assert(vocab_size >= 1);
    std.debug.assert(sessions >= 1);

    return .{
        .sessions = sessions,
        .per_slot = scheduler_mod.slotSetBytes(vocab_size),
        .scratch = sampler.scratchLen(vocab_size) * @sizeOf(f32),
    };
}

/// The sampler state sits outside the KV quote, and its size follows the vocab,
/// not the window.
fn logSamplerRam(s: SamplerRam) void {
    log.info("ram: sampler {d} KiB = {d} sessions x {d} KiB id sets + {d} KiB scratch", .{
        s.total() / kib, s.sessions, s.per_slot / kib, s.scratch / kib,
    });
}

fn kvMiB(runner: anytype) u64 {
    var total: usize = 0;
    for (runner.kv_slots) |*slot| {
        total += slot.sizeBytes();
    }

    return total >> 20;
}

fn msBetween(io: Io, t0: Io.Clock.Timestamp) f64 {
    return time.msBetween(t0, Io.Clock.Timestamp.now(io, .awake));
}

fn printReady(w: *Io.Writer, comptime fmt: []const u8, fmt_args: anytype) !void {
    try w.print(fmt, fmt_args);
    try w.flush();
}

// -- tests ------------------------------------------------------------------

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
    // one. The split and the resident weights stay equal.
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
    const serve_host: kv_budget.HostRequest = .{
        .ram_split = .{ .blobs = 0, .prefix = 0 },
        .residency = true,
    };
    const wide_costs = kv_budget.KvCostVector.uniform(
        n_layers,
        @as(u64, 2) * dtype.KvDtype.f32.rowBytes(per_step),
    );
    const half_costs = kv_budget.KvCostVector.uniform(
        n_layers,
        @as(u64, 2) * dtype.KvDtype.f16.rowBytes(per_step),
    );
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

test "serve capacity: a qwen split prices each tier by the layers it holds" {
    // The 2B geometry: 24 layers, every fourth one full attention. The serve arm
    // builds this structure through `kv_cost.costVector` and hands it to the
    // same planner, so the arithmetic below is the arm's.
    const n_layers: u32 = 24;
    var costs = kv_budget.KvCostVector.uniform(n_layers, 0);
    var i: u32 = 0;
    while (i < n_layers) : (i += 1) {
        if ((i + 1) % 4 == 0) {
            costs.set(i, 4_096, 0);
        } else {
            costs.set(i, 0, 1_122_304);
        }
    }

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
    const serve_host: kv_budget.HostRequest = .{
        .ram_split = .{ .blobs = 0, .prefix = 0 },
        .residency = true,
    };
    const kv: kv_budget.KvStructure = .{ .n_slots = 4, .window = 512, .costs = costs };

    // Twelve GPU layers leave the split at 12. Each half then holds three
    // attention layers and nine recurrent ones, so both ledgers read equal. A
    // uniform price cannot produce that.
    const plan = try kv_budget.placementPlan(caps, &weights, head_bytes, kv, serve_host, 12);
    const half: u64 = @as(u64, kv.n_slots) * (@as(u64, kv.window) * 3 * 4_096 + 9 * 1_122_304);
    try std.testing.expectEqual(@as(u32, 12), plan.split);
    try std.testing.expectEqual(half, plan.kv_vram);
    try std.testing.expectEqual(half, plan.ram.host_kv);
    // The tied head goes with any non-zero suffix, so a split of one layer already
    // pays it whole. That is the cliff the refusal message names.
    const thin = try kv_budget.placementPlan(caps, &weights, head_bytes, kv, serve_host, 1);
    try std.testing.expectEqual(@as(u64, 62 * mib) + head_bytes, thin.weight_resident);
    // A prefix that holds no attention layer charges fixed slabs and no window
    // price: layers 0, 1, and 2 are all recurrent on this stack. It is the
    // boundary value where the host tier's KV height is zero.
    const gdn_only = try kv_budget.placementPlan(
        caps,
        &weights,
        head_bytes,
        kv,
        serve_host,
        n_layers - 3,
    );
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
