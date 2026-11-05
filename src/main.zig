//! Executable entry point: `main` passes control to `entry` with the release
//! command table. `src/dev_main.zig` calls `entry` with the dev table. The file
//! also re-exports the sub-modules of the engine, so an integration test under
//! `tests/` can `@import("zigferencer")` rather than cross a module root with a
//! relative path, which Zig forbids. The `test {}` block pulls in every module
//! that `main` has no production caller for, such as the non-default ops
//! variants, so their tests run under `zig build test`.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const cli = @import("cli/command.zig");
const log = @import("core/log.zig");
const span = @import("core/span.zig");

// -- constants --------------------------------------------------------------

/// Route `std.log` through the stderr handler that the environment gates. The
/// comptime ceiling is `debug` in every build, so the flow traces always compile
/// in and the runtime ceiling is the only gate ($ZIGFERENCER_LOG, default
/// `info`). A release binary is verbose on demand, with no rebuild. The gate
/// costs almost nothing, because a flow trace sits at a boundary and never
/// inside a per-element loop. A hot-path trace sits behind the comptime
/// `log.hot` gate (`-Dtrace`). See `core/log.zig`.
pub const std_options: std.Options = .{
    .logFn = log.logFn,
    .log_level = .debug,
};

// The re-exports let an integration test under `tests/` reach into the engine
// without a `../src/...` path import, which Zig forbids across module roots.
pub const loader = @import("io/gguf/loader.zig");
pub const parser = @import("io/gguf/parser.zig");
const registry = @import("core/registry.zig");
pub const limits = @import("core/limits.zig");
pub const kv_budget = @import("core/kv_budget.zig");
pub const kv_cost = @import("runtime/kv_cost.zig");
pub const kv_index = @import("core/kv_index.zig");
pub const attn_split = @import("core/attn_split.zig");
pub const attn_cpu = @import("core/attn_cpu.zig");
pub const attn_record = @import("runtime/attn_record.zig");
pub const ring_fsm = @import("core/ring_fsm.zig");
pub const tensor = @import("core/tensor.zig");
pub const dtype = @import("core/dtype.zig");
pub const ops = @import("ops/naive.zig");
pub const gdn = @import("ops/gdn.zig");
pub const vocab = @import("core/tokenizer/vocab.zig");
pub const bpe = @import("core/tokenizer/bpe.zig");
const chat = @import("core/chat.zig");
pub const step_row = @import("runtime/step_row.zig");
pub const llama_runner = @import("runtime/llama_runner.zig");
pub const qwen_runner = @import("runtime/qwen_runner.zig");
const kv_cache = @import("runtime/kv_cache.zig");
pub const session = @import("runtime/session.zig");
pub const sampler = @import("runtime/sampler.zig");
pub const parallel = @import("runtime/parallel.zig");
pub const pool = @import("runtime/pool.zig");
pub const prompt_cache = @import("runtime/prompt_cache.zig");
pub const tensor_lookup = @import("runtime/tensor_lookup.zig");
pub const hwinfo = @import("core/hwinfo.zig");
pub const op_requires = @import("core/op_requires.zig");
pub const serve_protocol = @import("serve/protocol.zig");
pub const serve_zmtp = @import("serve/zmtp.zig");
pub const serve_event = @import("serve/event.zig");
pub const serve_residency = @import("serve/residency.zig");
const serve_scheduler = @import("serve/scheduler.zig");
pub const serve_prefill_chunk = @import("serve/prefill_chunk.zig");
pub const serve_server = @import("serve/server.zig");
pub const serve_client = @import("serve/client.zig");
pub const serve_openai_http = @import("serve/openai/http_server.zig");
pub const serve_openai_map = @import("serve/openai/map.zig");
pub const serve_openai_raw_codec = @import("serve/openai/raw_codec.zig");
pub const client_transcript = @import("cli/client_transcript.zig");
pub const serve_common = @import("cli/serve_common.zig");
pub const ops_simd = @import("ops/simd.zig");
pub const ops_simd_pf = @import("ops/simd_pf.zig");

// The compiler builds the Vulkan modules only under `-Druntime=vulkan`. In a CPU
// build each alias below is an empty namespace, and the compiler never reads the
// source file. A CPU build therefore needs no vulkan-zig dependency and no
// libvulkan link. The aliases sit at two levels. The context, the kernels, and
// the per-op runner follow `use_vulkan`, because a per-op GPU build drives them.
// The fused and the hybrid runners follow `use_vulkan_fused_ops`, the flag every
// construction site gates on. The `test {}` block below and the integration-test
// lists of `build.zig` hold the same two levels.
pub const vulkan_ctx = if (build_options.use_vulkan)
    @import("runtime/vulkan_ctx.zig")
else
    struct {};

pub const ops_vulkan = if (build_options.use_vulkan)
    @import("ops_vulkan/kernels.zig")
else
    struct {};

pub const vulkan_runner = if (build_options.use_vulkan)
    @import("runtime/vulkan_runner.zig")
else
    struct {};

pub const llama_fused = if (build_options.use_vulkan_fused_ops)
    @import("runtime/llama_fused.zig")
else
    struct {};

pub const qwen_fused = if (build_options.use_vulkan_fused_ops)
    @import("runtime/qwen_fused.zig")
else
    struct {};

pub const llama_hybrid = if (build_options.use_vulkan_fused_ops)
    @import("runtime/llama_hybrid.zig")
else
    struct {};

const qwen_hybrid = if (build_options.use_vulkan_fused_ops)
    @import("runtime/qwen_hybrid.zig")
else
    struct {};

// -- api --------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    return entry(init, .release);
}

/// Set the log gate and the trace recorder from the environment, then dispatch
/// argv over the command table of `which`. Both executables share this body.
pub fn entry(init: std.process.Init, comptime which: enum { release, dev }) !void {
    const dispatch = switch (which) {
        .release => cli.dispatchRelease,
        .dev => cli.dispatchDev,
    };

    // The top-level help runs before the log and trace setup, so a bad trace variable cannot hide
    // it.
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len > 1 and cli.isHelpWord(argv[1])) return dispatch(init);

    log.setFromEnv(init.environ_map.get("ZIGFERENCER_LOG"));
    const trace_spans = init.environ_map.get(limits.trace_spans_env);
    span.setFromEnv(init.io, .{
        .out = init.environ_map.get("ZIGFERENCER_TRACE_OUT"),
        .seconds = init.environ_map.get("ZIGFERENCER_TRACE_SECONDS"),
        .gpu = init.environ_map.get("ZIGFERENCER_TRACE_GPU"),
        .spans = trace_spans,
    }) catch {
        printBadTraceSpans(init.io, trace_spans.?);
        std.process.exit(1);
    };
    defer span.finish();

    dispatch(init) catch |err| {
        // A handled user error is one of bad arguments, an unknown model, or no
        // model picked. Its handler already printed a readable message to
        // stderr, so `main` exits non-zero and drops the developer return-trace,
        // which is noise for a usage mistake. An unexpected error (a bug, OOM, a
        // parse failure) falls through with its trace intact, and that is where
        // the trace earns its keep.
        if (isUserError(err)) {
            span.finish();
            std.process.exit(1);
        }

        return err;
    };
}

// -- helpers ----------------------------------------------------------------

// A failed write has nowhere useful to go, as in the dispatcher refusals.
fn printBadTraceSpans(io: std.Io, text: []const u8) void {
    var buf: [256]u8 = undefined;
    var fw: std.Io.File.Writer = .initStreaming(.stderr(), io, &buf);
    const w = &fw.interface;
    w.print(
        "error: " ++ limits.trace_spans_env ++ " must be a ring capacity in 1..{d}. The " ++
            "value is '{s}'\n",
        .{ limits.spans_per_thread_max, text },
    ) catch return;
    w.flush() catch {};
}

/// The errors whose handler already explained itself to the user, so `main` can
/// exit cleanly without a stack trace. Every error the list omits counts as
/// unexpected and keeps its trace. A new usage error belongs here. An internal
/// failure does not.
fn isUserError(err: anyerror) bool {
    return switch (err) {
        error.UsageError,
        error.UnknownModel,
        error.UnknownSubcommand,
        error.NoModelsPresent,
        error.BadHfPath,
        // Download failures: io/gguf already logged the HTTP status and what was
        // wrong with the body. The trace adds nothing a user can act on.
        error.HttpStatusNotOk,
        error.UnexpectedEncoding,
        error.DownloadTruncated,
        error.NotAGguf,
        // The loader already logged the file, the tensor, and the dtype tag it
        // cannot frame.
        error.UnsupportedTensorDtype,
        // The loader already logged the file and the tensor whose shape or range it refused.
        error.BadTensorShape,
        error.TensorSizeOverflow,
        error.NotEnoughBytes,
        // The loader already logged the file and the RoPE scaling type.
        error.UnsupportedRopeScaling,
        // backend/arch refusals: the handler names the build that works.
        error.UnsupportedServeBackend,
        error.UnsupportedArchForBackend,
        error.RequiresVulkan,
        // prompt problems: the handler logged the count and the limit.
        error.EmptyPrompt,
        error.PromptTooLong,
        // The GPU runner refused a config that does not fit VRAM. The preflight
        // already logged the required-vs-available bytes and what to reduce.
        error.VramBudgetExceeded,
        // The KV budget cannot host one decode row. The handler already named
        // the budget, the row count, and the per-slot price that refused it.
        error.KvVramBudgetTooSmall,
        // A serve arm whose host spend passes the usable RAM. The handler
        // already named every tier, the RAM figures, and the fit.
        error.HostRamBudgetTooSmall,
        // An unset --kv-ram whose blob need the usable RAM cannot fund. The
        // handler already named the need, the RAM left, and both fits.
        error.BlobRamBudgetTooSmall,
        // A named window past the model ceiling. The handler already named the
        // window and the ceiling.
        error.WindowExceedsCeiling,
        // A KV region passes a device limit. The guard already named the
        // limit, the byte count, and the widest window that fits.
        error.KvRegionPastDeviceLimit,
        // The split attention records pass a device limit. The guard already
        // named the limit and both counts.
        error.AttentionScratchPastDeviceLimit,
        // The model head_dim breaks a rule of the GPU split attention. The
        // guard already named the head_dim and both rules.
        error.AttentionHeadDimUnsupported,
        // The model GDN state size breaks the rule of the GPU prefill scan.
        // The guard already named both sizes.
        error.GdnHeadDimUnsupported,
        // A malformed storage-range override. The dispatcher already named the
        // variable, its range, and the value.
        error.InvalidStorageRange,
        // A GPU profile refusal: the profile logged the timestamp bits and the period.
        error.TimestampsUnsupported,
        // The client already printed a note for each of these.
        error.WindowOutOfRange,
        error.ReserveOutOfRange,
        error.ThinkingUnsupported,
        error.ConnectionClosed,
        error.SessionLost,
        error.ServerChanged,
        error.ConnectionRefused,
        => true,
        else => false,
    };
}

// -- tests ------------------------------------------------------------------

// The tests under `tests/` reach the test support module through the root.
pub const test_fakes = @import("runtime/test_fakes.zig");
pub const test_fixtures = @import("core/test_fixtures.zig");
pub const prop = @import("core/prop.zig");

// The block pulls every module without a production caller into the test
// binary, so its `test "..."` blocks run under `zig build test`.
test {
    _ = dtype;
    _ = tensor;
    _ = log;
    _ = ops;
    _ = parser;
    _ = loader;
    _ = registry;
    _ = limits;
    _ = kv_budget;
    _ = kv_index;
    _ = attn_split;
    _ = attn_cpu;
    _ = attn_record;
    _ = @import("core/ring_fsm.zig");
    _ = @import("core/time.zig");
    _ = @import("core/step_stats.zig");
    _ = @import("core/record_ring.zig");
    _ = span;
    _ = prop;
    _ = test_fixtures;
    _ = vocab;
    _ = chat;
    // A command body is analyzed only where it is named. `main` names
    // `dispatchRelease`, and a test binary has no `main`. This line names the
    // dev dispatch, whose table holds every command, so every arm of every
    // command compiles under `test-build`. That includes the placement arms,
    // whose device code no other reference reaches.
    _ = &cli.dispatchDev;
    // CLI sub-modules whose tests belong under `zig build test`.
    _ = @import("cli/arch_matrix.zig");
    _ = @import("cli/bench_decode_host_cmd.zig");
    _ = @import("cli/bench_swap_cmd.zig");
    _ = @import("cli/args.zig");
    _ = @import("cli/flags.zig");
    _ = @import("engine/resolve.zig");
    _ = @import("cli/client_cmd.zig");
    _ = @import("cli/client_transcript.zig");
    _ = @import("cli/fetch_cmd.zig");
    _ = @import("cli/inspect_cmd.zig");
    _ = @import("cli/prompt.zig");
    _ = @import("cli/serve_common.zig");
    _ = @import("cli/style.zig");
    _ = gdn;
    _ = prompt_cache;
    _ = kv_cache;
    _ = session;
    _ = sampler;
    _ = @import("runtime/generate.zig");
    _ = tensor_lookup;
    _ = parallel;
    _ = pool;
    _ = hwinfo;
    _ = kv_cost;
    _ = op_requires;
    _ = serve_protocol;
    _ = serve_residency;
    _ = serve_scheduler;
    _ = serve_prefill_chunk;
    _ = serve_server;
    _ = @import("serve/openai/map.zig");
    _ = serve_zmtp;
    _ = ops_simd;
    _ = ops_simd_pf;
    _ = @import("ops/attention_simd.zig");
    if (build_options.use_vulkan) {
        // A function body is analyzed where its ADDRESS is taken. A discarded
        // value (`_ = f;`) resolves the signature and leaves the body alone, so
        // every reference below takes `&`. The streaming ring has no production
        // caller, so these references alone get its bodies analyzed.
        _ = &ops_vulkan.StreamRing.init;
        _ = &ops_vulkan.StreamRing.acquire;
        _ = &ops_vulkan.StreamRing.submit;
        _ = &ops_vulkan.StreamRing.wait;
        _ = &ops_vulkan.StreamRing.view;
        _ = &ops_vulkan.StreamRing.release;
        _ = &ops_vulkan.StreamRing.deinit;
    }

    if (build_options.use_vulkan_fused_ops) {
        _ = llama_fused;
        // The qwen fused runner's tests are device-free (cost vectors, range
        // algebra, scratch arithmetic) and run nowhere without this pull.
        _ = qwen_fused;
        // Same for the qwen hybrid's blob-framing and tag anchors.
        _ = qwen_hybrid;
        // Each method below reaches production through a comptime branch or a
        // generic instantiation, so the `&` references hold the bodies analyzed
        // whichever arms a fused build instantiates.
        _ = &qwen_hybrid.QwenHybridRunner.logitRowsMax;
        _ = &llama_fused.LlamaFusedRunner.initFromLoaded;
    }
}
