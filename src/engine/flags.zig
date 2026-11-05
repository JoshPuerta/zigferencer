//! The options of one engine boot as a plain value. `resolve` reads them, so they
//! live outside the CLI. The CLI attaches the help text in `cli/flag_groups.zig`.

const dtype = @import("../core/dtype.zig");
const chat = @import("../core/chat.zig");
const kv_budget = @import("../core/kv_budget.zig");
const parallel = @import("../runtime/parallel.zig");

// -- types ------------------------------------------------------------------

/// The three thread flags. They mirror `parallel.Flags`.
pub const ThreadFlags = struct {
    /// Zero keeps the host default.
    threads: ?u32 = null,
    min_parallel_flops: ?u64 = null,
    spin_us: ?u64 = null,

    pub fn policyFlags(self: ThreadFlags) parallel.Flags {
        return .{
            .threads = self.threads,
            .min_parallel_flops = self.min_parallel_flops,
            .spin_us = self.spin_us,
        };
    }
};

/// The options of one engine boot.
pub const EngineFlags = struct {
    /// Null takes the default of the model (`registry.Model.kv_dtype_default`).
    kv_dtype: ?dtype.KvDtype = null,
    /// Null takes the default window, which a byte budget per session sizes.
    max_seq: ?u32 = null,
    /// Null takes the whole-model selection of the build.
    gpu_layers: ?kv_budget.GpuLayers = null,
    threads: ThreadFlags = .{},
    /// `auto` keeps the registry mode of the model.
    thinking: chat.ThinkingFlag = .auto,
    /// Think tokens before the forced close. Zero is unbounded.
    thinking_budget: u32 = 0,
};
