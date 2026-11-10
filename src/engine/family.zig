//! The model families. One row per architecture holds the runner of each
//! placement and the few values that differ between families. A command turns
//! the architecture into a row once, and stays generic below that point:
//!
//!     switch (loaded.card.arch) {
//!         inline else => |arch| return bench(family.Of(arch), ...),
//!     }
//!
//! No command needs an edit after that.
//!
//! Comptime. `Of` returns the row of an architecture, so a new architecture fails the build until
//! its row exists.

const std = @import("std");
const build_options = @import("build_options");

const dtype = @import("../core/dtype.zig");
const kv_budget = @import("../core/kv_budget.zig");
const loader = @import("../io/gguf/loader.zig");
const runner_surface = @import("../runtime/runner_surface.zig");
const CpuRunner = @import("../runtime/cpu_runner.zig").CpuRunner;
const layer_kernels = @import("../runtime/layer_kernels.zig");
const llama_layer = @import("../runtime/llama_layer.zig");
const qwen_layer = @import("../runtime/qwen_layer.zig");

// A device runner resolves to `void` on a build without it, so a CPU build never
// analyzes the device code.
const fused_build = build_options.use_vulkan_fused_ops;
const fused_runner = if (fused_build) @import("../runtime/fused_runner.zig") else void;
const hybrid_runner = if (fused_build) @import("../runtime/hybrid_runner.zig") else void;

// -- types ------------------------------------------------------------------

/// The row of `arch`. The return value is a namespace, so each row can hold
/// types and functions with signatures of its own.
pub fn Of(comptime arch: loader.Arch) type {
    return switch (arch) {
        .llama => Llama,
        .qwen35 => Qwen,
    };
}

const Llama = struct {
    pub const name = "llama";
    pub const Cpu = CpuRunner(llama_layer);
    pub const Fused = if (fused_build) fused_runner.FusedRunner(llama_layer) else void;
    pub const Hybrid = if (fused_build) hybrid_runner.HybridRunner(@This()) else void;
    pub const profile_family = llama_layer.profile_family;

    // A build that names the row checks its runners. The llama runners hold no recurrent state.
    comptime {
        runner_surface.requireSurface(Cpu, kv_budget.KvPlan);
        runner_surface.requireNoRecurrentState(Cpu);
        if (fused_build) {
            runner_surface.requireSurface(Fused, kv_budget.KvPlan);
            runner_surface.requireSurface(Hybrid, kv_budget.LayerSplit);
            runner_surface.requireNoRecurrentState(Fused);
            runner_surface.requireNoRecurrentState(Hybrid);
        }
    }

    /// `bench-model` runs on the per-op Vulkan backend.
    pub const per_op_bench = true;

    /// The fused bench warms up before its timed runs.
    pub const fused_warmup = true;

    /// The fewest slots of a fused serve plan. The llama runner keeps a spare stream, so its serve
    /// plan holds two slots. A run or a bench plans one slot in every family.
    pub const fused_min_slots: u32 = 2;

    /// The fused path of `run` does not honor `--prompt-cache`.
    pub const prompt_cache = false;

    /// `run` logs a placement line where each placement starts.
    pub const run_logs_every_placement = true;

    /// The literal frame of a hybrid tag. The bound and the format share it.
    const hybrid_tag_prefix = "vulkan_hybrid:k=";

    pub const hybrid_tag_buf_len: usize = hybridTagBufLen(hybrid_tag_prefix);

    /// The provenance tag of a hybrid blob. `buf` must hold `hybrid_tag_buf_len` bytes.
    pub fn hybridBackendTag(
        buf: []u8,
        host_layers: u32,
        quant: layer_kernels.WeightQuant,
        kv_dtype: dtype.KvDtype,
    ) []const u8 {
        return formatHybridTag(hybrid_tag_prefix, buf, host_layers, quant, kv_dtype);
    }

    /// The tag of a blob of `runner` at `host_layers`.
    pub fn hybridTag(
        buf: []u8,
        runner: *const Hybrid,
        host_layers: u32,
        kv_dtype: dtype.KvDtype,
    ) []const u8 {
        return hybridBackendTag(buf, host_layers, runner.suffix.quant, kv_dtype);
    }
};

const Qwen = struct {
    pub const name = "qwen";
    pub const Cpu = CpuRunner(qwen_layer);
    pub const Fused = if (fused_build) fused_runner.FusedRunner(qwen_layer) else void;
    pub const Hybrid = if (fused_build) hybrid_runner.HybridRunner(@This()) else void;
    pub const profile_family = qwen_layer.profile_family;

    // A build that names the row checks its runners. The qwen runners hold the GDN state.
    comptime {
        runner_surface.requireSurface(Cpu, kv_budget.KvPlan);
        runner_surface.requireRecurrentState(Cpu);
        if (fused_build) {
            runner_surface.requireSurface(Fused, kv_budget.KvPlan);
            runner_surface.requireSurface(Hybrid, kv_budget.LayerSplit);
            runner_surface.requireRecurrentState(Fused);
            runner_surface.requireRecurrentState(Hybrid);
        }
    }

    pub const per_op_bench = false;
    pub const fused_warmup = false;

    pub const fused_min_slots: u32 = 1;

    /// The fused path of `run` honors `--prompt-cache`.
    pub const prompt_cache = true;

    /// `run` logs a placement line for a split only, before the split picks a runner.
    pub const run_logs_every_placement = false;

    /// The literal frame of a hybrid tag. The family name in it keeps the blobs of the two
    /// families apart.
    const hybrid_tag_prefix = "vulkan_hybrid_qwen:k=";

    pub const hybrid_tag_buf_len: usize = hybridTagBufLen(hybrid_tag_prefix);

    /// The provenance tag of a hybrid blob. `buf` must hold `hybrid_tag_buf_len` bytes.
    pub fn hybridBackendTag(
        buf: []u8,
        host_layers: u32,
        quant: layer_kernels.WeightQuant,
        kv_dtype: dtype.KvDtype,
    ) []const u8 {
        return formatHybridTag(hybrid_tag_prefix, buf, host_layers, quant, kv_dtype);
    }

    /// The tag of a blob of `runner` at `host_layers`.
    pub fn hybridTag(
        buf: []u8,
        runner: *const Hybrid,
        host_layers: u32,
        kv_dtype: dtype.KvDtype,
    ) []const u8 {
        return hybridBackendTag(buf, host_layers, runner.suffix.quant, kv_dtype);
    }
};

// -- helpers ----------------------------------------------------------------

/// Bytes of the widest hybrid tag of a row: the literal frame, the widest u32 layer count, the
/// separator, the longest weight quant name, and the widest KV dtype component.
fn hybridTagBufLen(comptime prefix: []const u8) usize {
    var widest: usize = 0;
    inline for (@typeInfo(layer_kernels.WeightQuant).@"enum".fields) |f| {
        widest = @max(widest, f.name.len);
    }

    const layers_max = std.fmt.comptimePrint("{d}", .{std.math.maxInt(u32)});
    return prefix.len + layers_max.len + 1 + widest + dtype.kv_tag_suffix_max;
}

/// The provenance tag of a hybrid blob, so a blob never restores across another family, host
/// layer count, weight quant, or KV dtype. Serve folds it into its backend fingerprint.
fn formatHybridTag(
    comptime prefix: []const u8,
    buf: []u8,
    host_layers: u32,
    quant: layer_kernels.WeightQuant,
    kv_dtype: dtype.KvDtype,
) []const u8 {
    std.debug.assert(buf.len >= comptime hybridTagBufLen(prefix));

    // The dtype component names only a non-identity dtype, so f32 keeps the shipped literal. The
    // bound makes the format infallible.
    return std.fmt.bufPrint(
        buf,
        prefix ++ "{d}:{t}{s}",
        .{ host_layers, quant, kv_dtype.tagSuffix() },
    ) catch unreachable;
}

// -- tests ------------------------------------------------------------------

test "runner surface: every family row holds the session surface" {
    // A comptime block of a container runs only when the build analyzes the container.
    inline for (comptime std.enums.values(loader.Arch)) |arch| {
        _ = &Of(arch).Cpu;
    }
}

test "hybrid tag: the texts on disk stay the shipped literals of both families" {
    const LlamaRow = Of(.llama);
    const QwenRow = Of(.qwen35);
    var buf_a: [LlamaRow.hybrid_tag_buf_len]u8 = undefined;
    var buf_b: [LlamaRow.hybrid_tag_buf_len]u8 = undefined;
    var buf_c: [QwenRow.hybrid_tag_buf_len]u8 = undefined;
    var buf_d: [QwenRow.hybrid_tag_buf_len]u8 = undefined;
    var buf_e: [QwenRow.hybrid_tag_buf_len]u8 = undefined;

    // The tags stay the shipped literals byte for byte, so no cached blob moves. A Qwen runner
    // holds Q8_0 weights only.
    const llama_15 = LlamaRow.hybridBackendTag(&buf_a, 15, .q8_0, .f32);
    const llama_15_f16 = LlamaRow.hybridBackendTag(&buf_b, 15, .q8_0, .f16);
    try std.testing.expectEqualStrings("vulkan_hybrid:k=15:q8_0", llama_15);
    try std.testing.expectEqualStrings("vulkan_hybrid:k=15:q8_0:kv=f16", llama_15_f16);
    const qwen_6 = QwenRow.hybridBackendTag(&buf_c, 6, .q8_0, .f32);
    const qwen_12 = QwenRow.hybridBackendTag(&buf_d, 12, .q8_0, .f32);
    const qwen_6_f16 = QwenRow.hybridBackendTag(&buf_e, 6, .q8_0, .f16);
    try std.testing.expectEqualStrings("vulkan_hybrid_qwen:k=6:q8_0", qwen_6);
    try std.testing.expectEqualStrings("vulkan_hybrid_qwen:k=12:q8_0", qwen_12);
    try std.testing.expectEqualStrings("vulkan_hybrid_qwen:k=6:q8_0:kv=f16", qwen_6_f16);
    try std.testing.expectEqualStrings(
        "vulkan_hybrid:k=15:q4_0",
        LlamaRow.hybridBackendTag(&buf_a, 15, .q4_0, .f32),
    );

    // The family name leads, so a blob of one family never restores into the other.
    try std.testing.expect(!std.mem.startsWith(u8, qwen_6, "vulkan_hybrid:"));
    try std.testing.expect(!std.mem.startsWith(u8, llama_15, "vulkan_hybrid_qwen:"));

    // The widest tag of each family fits its bound exactly.
    const llama_widest = LlamaRow.hybridBackendTag(&buf_a, std.math.maxInt(u32), .q8_0, .f16);
    try std.testing.expectEqual(LlamaRow.hybrid_tag_buf_len, llama_widest.len);
    const qwen_widest = QwenRow.hybridBackendTag(&buf_c, std.math.maxInt(u32), .q8_0, .f16);
    try std.testing.expectEqual(QwenRow.hybrid_tag_buf_len, qwen_widest.len);
}

test "hybrid surface: the marker follows the CPU runner, and the slot reset exists" {
    if (comptime !fused_build) return error.SkipZigTest;

    try std.testing.expect(!Of(.llama).Hybrid.has_recurrent_state);
    try std.testing.expect(Of(.qwen35).Hybrid.has_recurrent_state);
    inline for (comptime std.enums.values(loader.Arch)) |arch| {
        try std.testing.expect(@hasDecl(Of(arch).Hybrid, "resetSlotState"));
        try std.testing.expect(@hasDecl(Of(arch).Hybrid, "decodeHost"));
    }
}
