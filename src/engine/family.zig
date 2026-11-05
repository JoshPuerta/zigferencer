//! The model families. One row per architecture holds the runner of each
//! placement and the few facts that differ between families. A command turns
//! the architecture into a row once, and stays generic below that point:
//!
//!     switch (loaded.card.arch) {
//!         inline else => |arch| return bench(family.Of(arch), ...),
//!     }
//!
//! `Of` names every architecture, so a new one fails the build here until its
//! row exists. No command needs an edit after that.

const std = @import("std");
const Allocator = std.mem.Allocator;
const build_options = @import("build_options");

const dtype = @import("../core/dtype.zig");
const kv_budget = @import("../core/kv_budget.zig");
const loader = @import("../io/gguf/loader.zig");
const cpu_profile = @import("../runtime/cpu_profile.zig");
const LlamaRunner = @import("../runtime/llama_runner.zig").LlamaRunner;

// A device runner resolves to `void` on a build without it, so a CPU build never
// analyzes the device code.
const fused_build = build_options.use_vulkan_fused_ops;
const llama_fused = if (fused_build) @import("../runtime/llama_fused.zig") else void;
const llama_hybrid = if (fused_build) @import("../runtime/llama_hybrid.zig") else void;
const qwen_fused = if (fused_build) @import("../runtime/qwen_fused.zig") else void;
const qwen_hybrid = if (fused_build) @import("../runtime/qwen_hybrid.zig") else void;
const vulkan_ctx = if (fused_build) @import("../runtime/vulkan_ctx.zig") else void;

const QwenRunner = @import("../runtime/qwen_runner.zig").QwenRunner;

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
    pub const Cpu = LlamaRunner;
    pub const Fused = if (fused_build) llama_fused.LlamaFusedRunner else void;
    pub const Hybrid = if (fused_build) llama_hybrid.LlamaHybridRunner else void;
    pub const profile_family: cpu_profile.Family = .llama;

    /// `bench-model` runs on the per-op Vulkan backend.
    pub const per_op_bench = true;

    /// The fused bench warms up before its timed runs.
    pub const fused_warmup = true;

    /// The hybrid can hold a degenerate split, and the cross-runner gate uses one.
    pub const forced_split = true;

    /// The fused runner keeps a spare stream, so its plan holds at least two slots.
    pub const fused_min_slots = 2;

    /// The fused path of `run` does not honor `--prompt-cache`.
    pub const prompt_cache = false;

    /// `run` logs a placement line where each placement starts.
    pub const run_logs_every_placement = true;

    pub fn kvCaps(
        gpa: Allocator,
        loaded: *const loader.Loaded,
        ceiling: u32,
        vram_free: u64,
        logit_rows: u32,
        kv_dtype: dtype.KvDtype,
    ) !kv_budget.Caps {
        return Fused.kvCaps(gpa, loaded, ceiling, vram_free, logit_rows, kv_dtype);
    }

    pub fn initFused(
        gpa: Allocator,
        ctx: *const vulkan_ctx.Context,
        loaded: *const loader.Loaded,
        plan: kv_budget.KvPlan,
        kv_dtype: dtype.KvDtype,
    ) !*Fused {
        return Fused.initFromLoaded(gpa, ctx, loaded, plan, kv_dtype);
    }

    /// The whole-model upload of the fused runner.
    pub fn fusedWeightBytes(gpa: Allocator, loaded: *const loader.Loaded) !u64 {
        return Fused.weightsVramBytesFromLoaded(gpa, loaded);
    }

    /// The tag of a blob: it names the split and the weight quant.
    pub fn hybridTag(
        buf: []u8,
        runner: *const Hybrid,
        split: u32,
        kv_dtype: dtype.KvDtype,
    ) []const u8 {
        return Hybrid.backendTag(buf, split, runner.quant, kv_dtype);
    }
};

const Qwen = struct {
    pub const name = "qwen";
    pub const Cpu = QwenRunner;
    pub const Fused = if (fused_build) qwen_fused.QwenFusedRunner else void;
    pub const Hybrid = if (fused_build) qwen_hybrid.QwenHybridRunner else void;
    pub const profile_family: cpu_profile.Family = .qwen;

    pub const per_op_bench = false;
    pub const fused_warmup = false;

    /// The hybrid refuses a degenerate split, because the whole model on one
    /// placement is a whole-model runner.
    pub const forced_split = false;

    pub const fused_min_slots = 1;

    /// The fused path of `run` honors `--prompt-cache`.
    pub const prompt_cache = true;

    /// `run` logs a placement line for a split only, before the split picks a runner.
    pub const run_logs_every_placement = false;

    pub fn kvCaps(
        gpa: Allocator,
        loaded: *const loader.Loaded,
        ceiling: u32,
        vram_free: u64,
        logit_rows: u32,
        kv_dtype: dtype.KvDtype,
    ) !kv_budget.Caps {
        _ = gpa;
        return Fused.kvCaps(loaded, ceiling, vram_free, logit_rows, kv_dtype);
    }

    pub fn initFused(
        gpa: Allocator,
        ctx: *const vulkan_ctx.Context,
        loaded: *const loader.Loaded,
        plan: kv_budget.KvPlan,
        kv_dtype: dtype.KvDtype,
    ) !*Fused {
        return Fused.init(gpa, ctx, loaded, plan, kv_dtype);
    }

    pub fn fusedWeightBytes(gpa: Allocator, loaded: *const loader.Loaded) !u64 {
        _ = gpa;
        return Fused.weightsVramBytes(loaded);
    }

    /// The tag of a blob: it names the family and the split.
    pub fn hybridTag(
        buf: []u8,
        runner: *const Hybrid,
        split: u32,
        kv_dtype: dtype.KvDtype,
    ) []const u8 {
        _ = runner;
        return Hybrid.backendTag(buf, split, kv_dtype);
    }
};

// -- tests ------------------------------------------------------------------

test "Of: every architecture has a complete row" {
    inline for (comptime std.enums.values(loader.Arch)) |arch| {
        const Row = Of(arch);
        const decls = [_][]const u8{
            "name",
            "Cpu",
            "Fused",
            "Hybrid",
            "profile_family",
            "per_op_bench",
            "fused_warmup",
            "forced_split",
            "fused_min_slots",
            "prompt_cache",
            "run_logs_every_placement",
            "kvCaps",
            "initFused",
            "fusedWeightBytes",
            "hybridTag",
        };
        inline for (decls) |decl| {
            try std.testing.expect(@hasDecl(Row, decl));
        }
    }
}
