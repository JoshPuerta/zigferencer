//! Executable entry point. `main` passes control to `entry` with the release command table.
//! `src/dev/dev_main.zig` calls `entry` with the dev table.
//!
//! Comptime. `entry` takes the dispatch function at comptime, so both executables share one body.
//!
//! Re-exports. The file re-exports the sub-modules of the engine. An integration test under
//! `tests/` then imports `zigferencer` and needs no relative path across a module root, which Zig
//! forbids.
//!
//! A CPU build compiles no Vulkan module, so each alias is an empty namespace.
//! The context, the kernels, and the per-op runner follow `use_vulkan`. The fused and the hybrid
//! runners follow `use_vulkan_fused_ops`, the flag every construction site gates on. The
//! `test {}` block and the integration-test lists of `build.zig` hold the same two levels.
//!
//! User errors. A handled user error has a handler that already printed a message. `entry`
//! exits non-zero and drops the return trace. Any other error keeps its trace.
//!
//! Tests. The `test {}` block pulls in every module without a production caller.

const std = @import("std");
const File = std.Io.File;
const builtin = @import("builtin");
const build_options = @import("build_options");
pub const cli = @import("cli.zig");
pub const log = @import("core/log.zig");
pub const span = @import("core/span.zig");

// -- constants --------------------------------------------------------------

/// Route `std.log` through the gated stderr handler.
pub const std_options: std.Options = .{
    .logFn = log.logFn,
    .log_level = .debug,
};

pub const loader = @import("io/gguf/loader.zig");
pub const parser = @import("io/gguf/parser.zig");
pub const registry = @import("core/registry.zig");
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
pub const step_stats = @import("core/step_stats.zig");
pub const time = @import("core/time.zig");
pub const ops = @import("ops/naive.zig");
pub const gdn = @import("ops/gdn.zig");
pub const vocab = @import("core/tokenizer/vocab.zig");
pub const bpe = @import("core/tokenizer/bpe.zig");
pub const chat = @import("core/chat.zig");
pub const family = @import("engine/family.zig");
pub const resolve = @import("engine/resolve.zig");
pub const generate = @import("runtime/generate.zig");
pub const cpu_profile = @import("runtime/cpu_profile.zig");
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
pub const serve_scheduler = @import("serve/scheduler.zig");
pub const serve_prefill_chunk = @import("serve/prefill_chunk.zig");
pub const serve_engine = @import("serve/engine.zig");
pub const serve_server = @import("serve/server.zig");
pub const serve_client = @import("serve/client.zig");
pub const serve_openai_http = @import("serve/openai/http_server.zig");
pub const serve_openai_map = @import("serve/openai/map.zig");
pub const serve_openai_raw_codec = @import("serve/openai/raw_codec.zig");
pub const client_transcript = @import("cli/client/client_transcript.zig");
pub const codec_parts = @import("cli/boot/codec_parts.zig");
pub const ops_simd = @import("ops/simd.zig");
pub const ops_simd_pf = @import("ops/simd_pf.zig");

pub const vulkan_ctx = if (build_options.use_vulkan)
    @import("runtime/vulkan_ctx.zig")
else
    struct {};

pub const ops_vulkan = if (build_options.use_vulkan)
    @import("ops_vulkan/kernels.zig")
else
    struct {};

pub const ops_vulkan_wrap = if (build_options.use_vulkan)
    @import("ops_vulkan/wrap.zig")
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
    return entry(init, cli.command.dispatchRelease);
}

/// Set the log checks and the trace recorder from the environment, then run `dispatch`.
/// Both executables share this body. It owns the stdout writer, so it can read a write failure.
pub fn entry(
    init: std.process.Init,
    comptime dispatch: fn (std.process.Init, *File.Writer) anyerror!void,
) !void {
    var stdout_buffer: [cli.context.stdout_buffer_bytes]u8 = undefined;
    var stdout: File.Writer = .initStreaming(.stdout(), init.io, &stdout_buffer);

    // Run the top-level help before the trace setup, so a bad trace variable cannot hide it.
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len > 1 and cli.command.isHelpWord(argv[1])) return dispatch(init, &stdout);

    // Set the log checks and the trace recorder.
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

    // Run the command, and map each failure to an exit.
    dispatch(init, &stdout) catch |err| {
        if (isUserError(err)) {
            span.finish();
            std.process.exit(1);
        }

        // A reader that closes the pipe ends the output on purpose.
        if (stdoutCause(err, &stdout)) |cause| {
            if (cause != error.BrokenPipe) {
                printStdoutFailure(init.io, cause);
            }

            span.finish();
            std.process.exit(1);
        }

        return err;
    };
}

// -- helpers ----------------------------------------------------------------

fn printBadTraceSpans(io: std.Io, text: []const u8) void {
    var buf: [256]u8 = undefined;
    var fw: std.Io.File.Writer = .initStreaming(.stderr(), io, &buf);
    const w = &fw.interface;
    w.print(
        "error: " ++ limits.trace_spans_env ++ " must be a ring capacity in 1..{d}. " ++
            "The value is '{s}'\n",
        .{ limits.spans_per_thread_max, text },
    ) catch return;
    w.flush() catch {};
}

fn printStdoutFailure(io: std.Io, cause: File.Writer.Error) void {
    var buf: [128]u8 = undefined;
    var fw: File.Writer = .initStreaming(.stderr(), io, &buf);
    fw.interface.print("error: cannot write the output: {t}\n", .{cause}) catch return;
    fw.interface.flush() catch {};
}

/// The cause of a failed write to stdout, or null when `err` came from another writer.
/// `error.WriteFailed` names no device, so the concrete writer holds the cause.
fn stdoutCause(err: anyerror, stdout: *const File.Writer) ?File.Writer.Error {
    if (err != error.WriteFailed) return null;
    return stdout.err;
}

/// The errors whose handler already explained itself to the user. A new usage error belongs here.
/// An internal failure does not.
fn isUserError(err: anyerror) bool {
    return switch (err) {
        error.UsageError,
        error.UnknownModel,
        error.UnknownSubcommand,
        error.NoModelsPresent,
        error.BadHfPath,
        // Download failures: io/gguf logged the HTTP status and the body fault.
        error.HttpStatusNotOk,
        error.UnexpectedEncoding,
        error.DownloadTruncated,
        error.NotAGguf,
        // Load failures: the loader logged the file and the tensor.
        loader.LoadError.UnsupportedTensorDtype,
        // The runner init logged the tensor and the dtype no kernel reads.
        error.UnsupportedWeightDtype,
        loader.LoadError.BadTensorShape,
        loader.LoadError.TensorSizeOverflow,
        loader.LoadError.NotEnoughBytes,
        loader.LoadError.UnsupportedRopeScaling,
        // Backend and architecture refusals.
        error.UnsupportedServeBackend,
        error.UnsupportedArchForBackend,
        error.RequiresVulkan,
        // Prompt problems.
        error.EmptyPrompt,
        error.PromptTooLong,
        // Budget and device-limit refusals.
        error.VramBudgetExceeded,
        error.KvVramBudgetTooSmall,
        error.HostRamBudgetTooSmall,
        error.BlobRamBudgetTooSmall,
        error.WindowExceedsCeiling,
        error.KvRegionPastDeviceLimit,
        error.AttentionScratchPastDeviceLimit,
        error.AttentionHeadDimUnsupported,
        error.GdnHeadDimUnsupported,
        error.GdnHeadsUnsupported,
        error.QwenHeadDimUnsupported,
        // Configuration refusals.
        error.InvalidStorageRange,
        error.InvalidSoftwareDevice,
        // The profile logged the timestamp bits and the period.
        error.TimestampsUnsupported,
        // Client errors.
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

pub const test_fakes = @import("runtime/test_fakes.zig");
pub const test_fixtures = @import("core/test_fixtures.zig");
pub const prop = @import("core/prop.zig");

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
    // A test binary has no `main`, so name `dispatchRelease` here..
    _ = &cli.command.dispatchRelease;
    // CLI sub-modules whose tests belong under `zig build test`.
    _ = @import("cli/info/arch_matrix.zig");
    _ = @import("cli/args.zig");
    _ = @import("cli/flag_groups.zig");
    _ = @import("engine/resolve.zig");
    _ = @import("cli/client/client_blocking.zig");
    _ = @import("cli/client/client_chat.zig");
    _ = @import("cli/client/client_cmd.zig");
    _ = @import("cli/client/client_history.zig");
    _ = @import("cli/client/client_input.zig");
    _ = @import("cli/client/client_output.zig");
    _ = @import("cli/client/client_reconnect.zig");
    _ = @import("cli/client/client_startup.zig");
    _ = @import("cli/client/client_streaming.zig");
    _ = @import("cli/client/client_transcript.zig");
    _ = @import("cli/fetch_cmd.zig");
    _ = @import("cli/inspect_cmd.zig");
    _ = @import("cli/prompt.zig");
    _ = @import("cli/boot/boot_resources.zig");
    _ = @import("cli/boot/boot_lines.zig");
    _ = @import("cli/boot/codec_parts.zig");
    _ = @import("cli/boot/serve_boot.zig");
    _ = @import("cli/boot/serve_checks.zig");
    _ = @import("cli/style.zig");
    _ = gdn;
    _ = prompt_cache;
    _ = kv_cache;
    _ = session;
    _ = sampler;
    _ = @import("runtime/generate.zig");
    _ = tensor_lookup;
    _ = @import("runtime/linear_dispatch.zig");
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
        // Take the address of each method, because `_ = f;` leaves the body unanalyzed.
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
        // The qwen fused tests are device-free and run only through this pull.
        _ = qwen_fused;
        _ = qwen_hybrid;
        // These methods reach production only through comptime branches, so take their addresses.
        _ = &qwen_hybrid.QwenHybridRunner.logitRowsMax;
        _ = &llama_fused.LlamaFusedRunner.initFromLoaded;
    }
}

test "a failed write to stdout names its cause, and no other error does" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;

    const io = std.testing.io;
    const full = try std.Io.Dir.openFileAbsolute(io, "/dev/full", .{ .mode = .write_only });
    defer full.close(io);

    var buffer: [16]u8 = undefined;
    var writer: File.Writer = .initStreaming(full, io, &buffer);
    try std.testing.expectEqual(null, stdoutCause(error.WriteFailed, &writer));

    try writer.interface.writeAll("text");
    try std.testing.expectError(error.WriteFailed, writer.interface.flush());
    try std.testing.expectEqual(error.NoSpaceLeft, stdoutCause(error.WriteFailed, &writer).?);
    try std.testing.expectEqual(null, stdoutCause(error.OutOfMemory, &writer));
}
