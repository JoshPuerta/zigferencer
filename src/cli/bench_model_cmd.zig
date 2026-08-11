//! `zigferencer bench-model <pub_name>` runs the full forward pass on a prompt
//! and repeats it `--runs` times. It reports two latency metrics with mean,
//! stddev, min and max, and a hash of the generated token ids. The output is a
//! human-readable table by default, or machine-readable JSON under `--json`.
//!
//! The two metrics split the generation cost the way a user feels it:
//!
//!   TTFT (time to first token) runs from the start of a bench iteration to the
//!     first sampled token. It covers BPE tokenization, prefill over the whole
//!     prompt, and the first decode step.
//!   ITL (inter-token latency) is the mean time per decode step after the first
//!     token, forward and sample. The divisor is the count of steps the decode
//!     ran. A budget stop steps every token, and an EOS stop does not step the
//!     last one. Prefill stays in TTFT, so it does not skew the steady-state
//!     number.
//!
//! `--json` also reports the ITL by depth: the mean step time per bucket of 512
//! KV positions, in position order.
//!
//! The hash is FNV-1a 64 over the generated token ids. A stable hash across runs
//! of one binary proves the forward pass is deterministic. A stable hash across
//! binaries built with different `-Dops-backend` values is the cross-variant
//! correctness gate, because a backend that changes the tokens changed the math.
//! The build links only one backend per binary, so a comparison of backends needs
//! a rebuild. `tools/bench_model_all.sh` drives that loop.

const std = @import("std");
const Io = std.Io;
const build_options = @import("build_options");

const cli = @import("command.zig");
const args = @import("args.zig");
const style = @import("style.zig");
const select = @import("select.zig");
const serve_common = @import("serve_common.zig");
const resolve_mod = @import("../engine/resolve.zig");
const registry = @import("../core/registry.zig");
const loader = @import("../io/gguf/loader.zig");
const vocab_mod = @import("../core/tokenizer/vocab.zig");
const bpe = @import("../core/tokenizer/bpe.zig");
const chat_mod = @import("../core/chat.zig");
const limits = @import("../core/limits.zig");
const core_log = @import("../core/log.zig");
const parallel = @import("../runtime/parallel.zig");
const Pool = @import("../runtime/pool.zig").Pool;

// The sibling modules of the bench-model command. `bench_model_common.zig`
// carries the values they exchange.
const common = @import("bench_model_common.zig");
const report = @import("bench_model_report.zig");
const exec = @import("bench_model_exec.zig");
const llama_bench = @import("bench_model_llama.zig");
const RunResult = common.RunResult;
const writeStderr = common.writeStderr;
const synthPromptIds = common.synthPromptIds;
const doOneRun = exec.doOneRun;
const printTable = report.printTable;
const printJson = report.printJson;

// The GPU-backed modules resolve to `void` on a CPU build, which never analyzes
// the vulkan code, because every call site sits behind a comptime guard.
const vulkan_ops_wrap = if (build_options.use_vulkan_ops)
    @import("../ops_vulkan/wrap.zig")
else
    void;
const llama_fused = if (build_options.use_vulkan_fused_ops)
    @import("../runtime/llama_fused.zig")
else
    void;
const vulkan_ctx_mod = if (build_options.use_vulkan_fused_ops)
    @import("../runtime/vulkan_ctx.zig")
else
    void;
const qwen = if (build_options.use_vulkan_fused_ops)
    @import("bench_model_qwen.zig")
else
    void;

// -- api --------------------------------------------------------------------

pub fn run(ctx: cli.Context) !void {
    // Flag meanings and defaults live on `common.Options`.
    const opt = try args.parseInto(common.Options, ctx) orelse return;
    if (opt.runs == 0 or opt.runs > limits.bench_runs_max) return args.fail(ctx.io, "--runs must be in 1..{d}", .{limits.bench_runs_max});
    if (opt.max_tokens < 2) return args.fail(ctx.io, "--max-tokens must be >= 2 (need at least 1 step for ITL)", .{});
    if (opt.engine.gpu_layers != null and !build_options.use_vulkan_fused_ops) {
        return args.fail(ctx.io, "--gpu-layers requires -Dops-backend=vulkan_fused", .{});
    }
    // The device profilers read device timestamps, so a build without the
    // fused GPU ops cannot honor them. Refuse at use.
    if (!build_options.use_vulkan_fused_ops) {
        const device_only: ?[]const u8 = if (opt.profile_prefill)
            "--profile-prefill"
        else if (opt.profile_decode)
            "--profile-decode"
        else
            null;
        if (device_only) |flag| {
            return args.fail(ctx.io, "{s} requires -Dops-backend=vulkan_fused", .{flag});
        }
    }
    // The per-op CPU timer compiles into a hot-trace build only, so a release
    // binary holds no clock read in the forward pass. Refuse at use and name
    // the builds that carry the instrument.
    if (opt.profile_cpu_decode and !core_log.hot) {
        return args.fail(ctx.io, "--profile-cpu-decode requires a hot-trace build (OPT=Debug or TRACE=1)", .{});
    }
    if (opt.force_hybrid and opt.engine.gpu_layers == null) {
        return args.fail(ctx.io, "--force-hybrid requires --gpu-layers", .{});
    }
    // A forced degenerate split needs a pin. Auto picks its own split, so the
    // pair states no split to force.
    if (opt.force_hybrid and opt.engine.gpu_layers.? == .auto) {
        return args.fail(ctx.io, "--force-hybrid requires a --gpu-layers count", .{});
    }
    if (opt.engine.gpu_layers != null and (opt.profile_prefill or opt.profile_decode)) {
        return args.fail(ctx.io, "--gpu-layers benches the hybrid runner and does not combine with profile flags", .{});
    }

    // The whole-model fused bench runs no CPU op, so its pool starts no
    // workers. A pin of any layer count benches a hybrid with CPU layers.
    const policy = try serve_common.resolvePolicy(
        ctx.io,
        opt.engine.threads.policyFlags(),
        !(build_options.use_vulkan_fused_ops and opt.engine.gpu_layers == null),
    );

    // A name resolves directly. With no name, the command prompts over the
    // on-disk models on a TTY, and returns a usage error otherwise. The flag
    // validations above already returned.
    const model = try select.model(ctx, opt.pub_name);
    const input: resolve_mod.Input = .{ .engine = opt.engine, .force_hybrid = opt.force_hybrid };
    const entry = serve_common.entryOf(model);
    // Every refusal that needs no model file lands here, before the load.
    switch (resolve_mod.early(.bench, input, entry)) {
        .ok => {},
        .refused => |r| return serve_common.refuse(ctx.io, model.pub_name, r),
    }
    const path = try registry.resolveLoadPath(ctx.arena, ctx.io, ctx.models_root, model.*);

    // The model loads once for the whole bench. The KV-cache scratch resets per
    // run, because a runner init re-zeros its slab.
    var loaded = try loader.load(ctx.io, ctx.arena, path);
    defer loaded.close();

    // One resolve per boot, the way `run` resolves: the dtype, the window
    // under the model ceiling at the run budget, and the all-host arm.
    const model_facts = try serve_common.modelFacts(ctx.gpa, &loaded);
    const ram_free = serve_common.hostRamFree(ctx.io);
    const resolved = switch (resolve_mod.resolve(.bench, input, entry, loaded.card, .{ .ram_free = ram_free, .model = model_facts })) {
        .ok => |r| r,
        .refused => |r| return serve_common.refuse(ctx.io, model.pub_name, r),
    };
    const kv_dtype = resolved.early.kv_dtype;
    const window = resolved.window;
    // A device bench resolves its arm again once its context states the
    // device.
    const boot: serve_common.DeviceBoot = .{ .kind = .bench, .input = input, .entry = entry, .card = loaded.card, .model = model_facts, .ram = .{ .read = ram_free } };

    // Qwen3.5 (hybrid GDN) GPU dispatch. The whole-model fused path has its own
    // runner, `qwen_fused.QwenFusedRunner`, which differs from the Llama
    // LlamaFusedRunner, so it bypasses the rest of this Llama-shaped function. A
    // pinned placement takes the shared hybrid bench below, and CPU backends take
    // the generic doOneRun path over QwenRunner.
    if (loaded.card.arch == .qwen35) {
        if (comptime build_options.use_vulkan_fused_ops) {
            // A pinned placement benches the qwen hybrid, which needs the
            // pool its CPU prefix runs on. That pool starts below, so only the
            // whole-model bench dispatches from here.
            if (opt.engine.gpu_layers == null) {
                return qwen.qwenFusedBench(ctx, &boot, &loaded, model.*, opt, kv_dtype);
            }
        }
        if (comptime build_options.use_vulkan_ops) {
            writeStderr(ctx.io, "error: qwen35 bench does not run on per-op vulkan. Build -Dops-backend=vulkan_fused (fused GPU) or a CPU backend\n");
            return error.UnsupportedArchForBackend;
        }
    }

    // One pool serves every CPU op of the command. Its workers live until
    // the command returns.
    var pool: Pool = undefined;
    try parallel.initPool(&pool, ctx.io, policy);
    defer pool.deinit();

    // The hybrid path owns its own Context, planner call, and runner, so it
    // branches before the fused construction below, in the shape the qwen
    // dispatch uses. Each arch reaches its own family's hybrid.
    if (comptime build_options.use_vulkan_fused_ops) {
        if (opt.engine.gpu_layers != null) {
            if (loaded.card.arch == .qwen35) {
                return qwen.qwenHybridBench(ctx, &boot, &loaded, model.*, opt, &pool, policy, kv_dtype);
            }
            return llama_bench.llamaHybridBench(ctx, &boot, &loaded, model.*, opt, &pool, policy, kv_dtype);
        }
    }

    // The GPU backend needs a process-wide Runner and weight cache. The init here
    // lands the weight uploads once across every run of the bench, and the deinit
    // at the end reclaims the VRAM. A CPU build skips both at comptime.
    if (comptime build_options.use_vulkan_ops) {
        try vulkan_ops_wrap.initBackend(ctx.arena, .{ .storage_range_override = ctx.vk_storage_range });
    }
    defer if (comptime build_options.use_vulkan_ops) {
        vulkan_ops_wrap.deinitBackend();
    };
    // The KV shape the runs execute under. It stays zero on every path that plans
    // no device KV, and the report then prints no plan line.
    var plan_info: common.PlanInfo = .{};
    // The fused GPU backend stages the same model and weights, and it records the
    // whole forward pass for one token as a single command buffer. The bench owns
    // an explicit Context and LlamaFusedRunner for their whole lifetime, with no
    // process-wide singleton, and shares them across every run below.
    // The Context is arena-allocated, so its address stays stable for the kernels
    // that capture it. Teardown is LIFO: the runner goes before the device. On a
    // non-fused build `fused` is `{}`, which only passes through and is never
    // dereferenced.
    const fused = if (comptime build_options.use_vulkan_fused_ops) f: {
        const ctxp = try ctx.arena.create(vulkan_ctx_mod.Context);
        ctxp.* = try vulkan_ctx_mod.Context.init(ctx.arena, .{ .storage_range_override = ctx.vk_storage_range });
        errdefer ctxp.deinit();
        // One slot at the base window, which a default shrinks to the device
        // limits and then to the VRAM figure.
        const f = try serve_common.fusedOn(ctx.io, &boot, ctxp);
        _ = try serve_common.logFused(ctx.io, model.pub_name, f, true, false, kv_dtype);
        const plan = f.plan;
        plan_info = common.PlanInfo.fromPlan(plan, kv_dtype);
        break :f try llama_fused.LlamaFusedRunner.initFromLoaded(ctx.gpa, ctxp, &loaded, plan, kv_dtype);
    } else {};
    defer if (comptime build_options.use_vulkan_fused_ops) {
        fused.deinit();
        // LlamaFusedRunner borrows the Context as `*const`. The bench owns it, so it
        // takes mutable access back to release the device once the runner is gone.
        @constCast(fused.ctx).deinit();
    };
    // The window every run executes under: the fused plan, or the base window of
    // a host runner, which has no device budget to fit.
    const max_seq: u32 = if (comptime build_options.use_vulkan_fused_ops) fused.max_seq else window.base;
    if (comptime !build_options.use_vulkan_fused_ops) {
        serve_common.logSessionWindow(window);
        // A host runner holds the KV on the host, so its quote is the CPU end
        // of the vector. The per-op vulkan build planned a device arm, so it
        // resolves the host arm here.
        std.debug.assert(max_seq == window.base);
        const result = resolved.cpu orelse resolve_mod.cpuArm(.bench, input, loaded.card, kv_dtype, window, .{ .ram_free = ram_free });
        switch (result) {
            .ok => |arm| serve_common.logRamQuote(arm.quote, false),
            .refused => |r| return serve_common.refuse(ctx.io, model.pub_name, r),
        }
    }

    var v = try vocab_mod.buildFromParsed(ctx.arena, loaded.parsed);

    // The text that goes to the tokenizer. --chat applies the model's registered
    // template, named by `registry.Model.chat_template_kind`. Without --chat the
    // prompt passes through as raw text.
    const text = if (opt.chat)
        try chat_mod.wrapAlloc(ctx.arena, model.chat_template_kind, opt.prompt)
    else
        opt.prompt;
    // The fit check runs first, because the synthesis allocates n ids.
    if (opt.prompt_tokens) |n| try common.requireSequenceFits(ctx.io, n, opt.max_tokens, max_seq);
    const prompt_ids = if (opt.prompt_tokens) |n|
        try synthPromptIds(ctx.arena, n, loaded.card.vocab_size)
    else
        try bpe.encodeAlloc(&v, ctx.arena, text, true);
    if (prompt_ids.len == 0) {
        writeStderr(ctx.io, "error: the prompt encodes to zero tokens\n");
        return error.EmptyPrompt;
    }
    try common.requireSequenceFits(ctx.io, prompt_ids.len, opt.max_tokens, max_seq);

    // The CPU per-op decode breakdown runs on CPU builds only. It installs a
    // cpu_profile.CpuProfile, prints the per-category split, and exits. The flag
    // validation above refuses every build without the timer, so this arm
    // compiles only where the instrument exists.
    if (comptime core_log.hot) {
        if (opt.profile_cpu_decode) {
            if (comptime build_options.use_vulkan_ops or build_options.use_vulkan_fused_ops) {
                writeStderr(ctx.io, "error: --profile-cpu-decode is for CPU backends (rebuild without -Druntime=vulkan)\n");
                return error.UnsupportedArchForBackend;
            }
            return llama_bench.cpuProfileBench(ctx, &loaded, max_seq, model.pub_name, build_options.ops_backend, prompt_ids, opt.max_tokens, &pool, policy, kv_dtype);
        }
    }

    // The warmup pass on a fused build runs one prefill chunk and one decode step
    // before the timed runs, so run 1 starts warm. Under --profile-prefill or
    // --profile-decode the pass also prints the per-stage breakdown.
    if (comptime build_options.use_vulkan_fused_ops) {
        const batch_max: u32 = fused.prefill_batch_max;
        if (batch_max > 0 and prompt_ids.len > 0) {
            const chunk = @min(prompt_ids.len, @as(usize, batch_max));
            if (opt.profile_prefill)
                try fused.profilePrefill(ctx.io, prompt_ids[0..chunk], 0, "warmup")
            else
                _ = try fused.prefillBatch(prompt_ids[0..chunk], 0);
            if (opt.profile_decode)
                try fused.profileStep(ctx.io, prompt_ids[prompt_ids.len - 1], @intCast(chunk), "warmup")
            else
                _ = try fused.step(prompt_ids[prompt_ids.len - 1], @intCast(chunk));
        }
    }

    const results = try ctx.arena.alloc(RunResult, opt.runs);
    var r: u32 = 0;
    while (r < opt.runs) : (r += 1) {
        results[r] = try doOneRun(fused, &loaded, max_seq, &v, prompt_ids, opt.max_tokens, opt.ignore_eos, ctx, &pool, policy, kv_dtype);
        results[r].plan = plan_info;
    }

    if (opt.json) {
        try printJson(ctx.io, model.pub_name, prompt_ids.len, opt.max_tokens, results, policy);
    } else {
        try printTable(ctx.io, model.pub_name, opt.prompt, prompt_ids.len, opt.max_tokens, results, policy);
    }
}

// -- helpers ----------------------------------------------------------------
