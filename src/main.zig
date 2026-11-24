//! Executable entry point. `main` passes control to `entry` with the release command table.
//! `src/dev/dev_main.zig` calls `entry` with the dev table.
//!
//! Comptime. `entry` takes the dispatch function at comptime, so both executables share one body.
//!
//! Re-exports. The file re-exports the sub-modules of the engine. An integration test under
//! `tests/` then imports `zigferencer` and needs no relative path across a module root, which Zig
//! forbids.
//!
//! Aliases. A CPU build compiles no Vulkan module, so each alias is an empty namespace.
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
pub const log = @import("core/utils/log.zig");
pub const span = @import("core/span.zig");
const console = @import("os/console.zig");

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
pub const layer_op = @import("core/layer_op.zig");
pub const attn_record = @import("runtime/attn_record.zig");
pub const ring_fsm = @import("core/ring_fsm.zig");
pub const tensor = @import("core/tensor.zig");
pub const dtype = @import("core/dtype.zig");
pub const step_stats = @import("core/step_stats.zig");
pub const time = @import("core/utils/time.zig");
pub const ops = @import("ops/naive.zig");
pub const gdn = @import("ops/gdn.zig");
pub const gdn_simd = @import("ops/gdn_simd.zig");
pub const exp_simd = @import("ops/exp_simd.zig");
pub const lane_tree_simd = @import("ops/lane_tree_simd.zig");
pub const vocab = @import("core/tokenizer/vocab.zig");
pub const bpe = @import("core/tokenizer/bpe.zig");
pub const chat = @import("core/chat.zig");
pub const family = @import("resolve/family.zig");
pub const resolve = @import("resolve/resolve.zig");
pub const generate = @import("runtime/generate.zig");
pub const cpu_profile = @import("runtime/cpu_profile.zig");
pub const step_row = @import("runtime/step_row.zig");
pub const cpu_runner = @import("runtime/cpu_runner.zig");
const kv_cache = @import("runtime/kv_cache.zig");
pub const session = @import("runtime/session.zig");
pub const sampler = @import("runtime/sampler.zig");
pub const parallel = @import("runtime/parallel.zig");
pub const pool = @import("runtime/pool.zig");
pub const prompt_cache = @import("runtime/prompt_cache.zig");
pub const tensor_lookup = @import("runtime/tensor_lookup.zig");
pub const hwinfo = @import("core/utils/hwinfo.zig");
pub const slab_arena = @import("core/utils/slab_arena.zig");
pub const os_process = @import("os/process.zig");
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
pub const serve_timed_stream = @import("serve/timed_stream.zig");
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

pub const dispatch_list = if (build_options.use_vulkan and log.hot)
    @import("ops_vulkan/dispatch_list.zig")
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

pub const fused_runner = if (build_options.use_vulkan_fused_ops)
    @import("runtime/fused_runner.zig")
else
    struct {};

pub const hybrid_runner = if (build_options.use_vulkan_fused_ops)
    @import("runtime/hybrid_runner.zig")
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
    // A Ctrl-C leaves the console in UTF-8, because no handler runs the restore.
    const old_code_page = console.setUtf8Output(init.io);
    defer console.restoreOutput(init.io, old_code_page);

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
        exitAfterRestore(init.io, old_code_page);
    };
    defer span.finish();

    // Run the command, and map each failure to an exit.
    dispatch(init, &stdout) catch |err| {
        if (isUserError(err)) {
            span.finish();
            exitAfterRestore(init.io, old_code_page);
        }

        // A reader that closes the pipe ends the output on purpose.
        if (stdoutCause(err, &stdout)) |cause| {
            if (cause != error.BrokenPipe) {
                printStdoutFailure(init.io, cause);
            }

            span.finish();
            exitAfterRestore(init.io, old_code_page);
        }

        return err;
    };
}

// -- helpers ----------------------------------------------------------------

/// Exit with status 1. A `defer` does not run before `exit`, so this restores the code page first.
fn exitAfterRestore(io: std.Io, old_code_page: ?u32) noreturn {
    console.restoreOutput(io, old_code_page);
    std.process.exit(1);
}

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
        error.ModelNotOnDisk,
        error.UnknownSubcommand,
        error.NoModelsPresent,
        error.BadHfPath,
        // Download failures: io/gguf logged the HTTP status and the body fault.
        error.HttpStatusNotOk,
        error.UnexpectedEncoding,
        error.DownloadTruncated,
        error.NotAGguf,
        error.TargetInUse,
        // The loader or `inspect` logged the file and the local-disk advice.
        loader.LoadError.ModelFileNotMapped,
        // Load failures: the loader logged the file and the key or the tensor.
        loader.LoadError.MissingKey,
        loader.LoadError.WrongKeyType,
        loader.LoadError.UnsupportedArchitecture,
        loader.LoadError.BadKeyValue,
        loader.LoadError.TooManyLayers,
        loader.LoadError.VocabTooLarge,
        loader.LoadError.UnsupportedTensorDtype,
        loader.LoadError.BadTensorShape,
        loader.LoadError.TensorSizeOverflow,
        loader.LoadError.TensorDataUnaligned,
        loader.LoadError.TensorDataOverlap,
        loader.LoadError.NotEnoughBytes,
        loader.LoadError.UnsupportedRopeScaling,
        // Parse failures: the loader or `inspect` logged the file and the fault.
        parser.Error.BadMagic,
        parser.Error.UnsupportedVersion,
        parser.Error.Truncated,
        parser.Error.BadValueType,
        parser.Error.NestedArray,
        parser.Error.NameTooLong,
        parser.Error.BadAlignment,
        parser.Error.BadTensorDims,
        parser.Error.CountExceedsFile,
        parser.Error.TensorOffsetsUnordered,
        parser.Error.TensorOffsetUnaligned,
        // The runner init logged the tensor and the dtype no kernel reads.
        error.UnsupportedWeightDtype,
        // The weight check logged the name of the absent tensor.
        error.MissingTensor,
        // The decode loop logged the non-finite logits of a step.
        error.NonFiniteLogits,
        // The tokenizer build logged the key, the merge, the count, or the special id.
        error.MissingTokenizerKey,
        error.WrongTokenizerKeyType,
        error.InvalidMerge,
        error.TokenStringTooLong,
        error.BadTokenizerValue,
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
        error.StepBufferPastDeviceLimit,
        error.AttentionScratchPastDeviceLimit,
        error.AttentionHeadDimUnsupported,
        error.GdnHeadDimUnsupported,
        error.GdnHeadsUnsupported,
        error.HeadsPastDispatchLimit,
        error.QwenHeadDimUnsupported,
        error.NoKvLayerInRange,
        // The fused init logged the device failure and the retry.
        error.DeviceFailedAtInit,
        // Configuration refusals.
        error.InvalidStorageRange,
        error.InvalidSoftwareDevice,
        // The context logged the workgroup counts of the device.
        error.WorkgroupCountBelowMinimum,
        // The context logged the subgroup sizes of the device and the pinned size.
        error.SubgroupSizeUnsupported,
        // The loader library logged its file name and the cause.
        error.VulkanLoaderLibraryMissing,
        error.VulkanLoaderEntryPointMissing,
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
pub const prop = @import("core/utils/prop.zig");

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
    _ = layer_op;
    _ = attn_record;
    _ = @import("runtime/layer_kernels.zig");
    _ = @import("runtime/qwen_layer.zig");
    _ = @import("runtime/llama_layer.zig");
    _ = @import("core/ring_fsm.zig");
    _ = @import("core/utils/time.zig");
    _ = @import("core/step_stats.zig");
    _ = @import("core/record_ring.zig");
    _ = span;
    _ = prop;
    _ = test_fixtures;
    _ = vocab;
    _ = chat;
    // A test binary has no `main`, so name `dispatchRelease` here.
    _ = &cli.command.dispatchRelease;
    // CLI sub-modules whose tests belong under `zig build test`.
    _ = @import("cli/info/arch_matrix.zig");
    _ = @import("cli/args.zig");
    _ = @import("cli/flag_groups.zig");
    _ = @import("resolve/resolve.zig");
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
    _ = @import("cli/select.zig");
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
    _ = @import("runtime/weight_check.zig");
    _ = @import("runtime/dispatch_guard.zig");
    _ = parallel;
    _ = pool;
    _ = hwinfo;
    _ = console;
    _ = os_process;
    _ = @import("os/dynamic_library.zig");
    _ = @import("os/file_map.zig");
    _ = @import("os/host_cores.zig");
    _ = @import("os/host_cpu_name.zig");
    _ = @import("os/host_gpu.zig");
    _ = @import("os/host_memory.zig");
    _ = @import("os/socket_wait.zig");
    _ = @import("os/threads.zig");
    _ = kv_cost;
    _ = op_requires;
    _ = serve_protocol;
    _ = serve_residency;
    _ = serve_scheduler;
    _ = serve_prefill_chunk;
    _ = serve_server;
    _ = @import("serve/conn_slots.zig");
    _ = @import("serve/timed_stream.zig");
    _ = @import("serve/openai/map.zig");
    _ = @import("serve/openai/body.zig");
    _ = serve_zmtp;
    _ = ops_simd;
    _ = ops_simd_pf;
    _ = @import("ops/attention_simd.zig");
    _ = lane_tree_simd;
    _ = gdn_simd;
    _ = @import("ops/linear_simd.zig");
    _ = @import("ops/elementwise_simd.zig");
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

    if (build_options.use_vulkan and log.hot) {
        _ = dispatch_list;
    }

    if (build_options.use_vulkan_fused_ops) {
        // The fused runner tests are device-free and run only through this pull.
        _ = fused_runner;
        _ = @import("runtime/fused_weights.zig");
        // The hybrid tests are device-free and run only through this pull.
        _ = hybrid_runner;
        // These methods reach production only through comptime branches, so take their addresses.
        _ = &family.Of(.llama).Hybrid.logitRowsMax;
        _ = &family.Of(.qwen35).Hybrid.logitRowsMax;
        _ = &family.Of(.llama).Fused.init;
        _ = &family.Of(.qwen35).Fused.init;
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
