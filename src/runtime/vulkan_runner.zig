//! A Vulkan `Context` and the compute pipelines that act on it, kept together at a fixed heap
//! address.
//!
//! Stable address. Every GPU kernel captures `*const Context` at init time, so the address of that
//! Context must not change while a kernel holds the pointer. `LinearQ80.init` stores `&ctx`. A
//! `ctx` can move: a factory returns it by value, or a call copies it across a function boundary.
//! That leaves the stored pointer dangling, and the next `cmdBindPipeline` reads through it into
//! freed stack. The `Runner.init` factory allocates the whole bundle on the heap, so the
//! construction fixes the address, not a rule that each caller has to remember:
//!
//!     gpa.create(Runner) -> stable address on the heap
//!       Runner
//!         +-- ctx         : Context       (lives here, never moves)
//!         +-- linear_q8_0 : LinearQ80 ----> captures &runner.ctx
//!         +-- vec_add     : VecAdd    ----> captures &runner.ctx
//!
//! Per-op set. The pipelines of the Runner (`vec_add`, `rope`, `linear_q8_0_subgroup`,
//! `linear_q8_0`, and their siblings) are the per-op set. They serve the per-op reference path in
//! `ops_vulkan/wrap.zig`, the benches, and the GPU tests. Each fused runner builds its own kernels
//! on a `Context` of its own, so none of these pipelines runs during inference.

const std = @import("std");
const build_options = @import("build_options");
const vulkan_ctx = @import("vulkan_ctx.zig");
const kernels = @import("../ops_vulkan/kernels.zig");

comptime {
    if (!build_options.use_vulkan) {
        @compileError("runtime/vulkan_runner.zig was included in a non-vulkan build.");
    }
}

// -- types ------------------------------------------------------------------

pub const Runner = struct {
    gpa: std.mem.Allocator,
    ctx: vulkan_ctx.Context,
    linear_q8_0: kernels.LinearQ80,
    linear_q8_0_reduce: kernels.LinearQ80Reduce,
    linear_q8_0_subgroup: kernels.LinearQ80Subgroup,
    linear_q4_0_reduce: kernels.LinearQ40Reduce,
    residual_add: kernels.ResidualAdd,
    swiglu: kernels.Swiglu,
    rmsnorm: kernels.Rmsnorm,
    rope: kernels.Rope,
    attention: kernels.Attention,
    vec_add: kernels.VecAdd,

    pub const InitError = vulkan_ctx.InitError ||
        kernels.LinearQ80.InitError ||
        kernels.LinearQ80Reduce.InitError ||
        kernels.LinearQ80Subgroup.InitError ||
        kernels.LinearQ40Reduce.InitError ||
        kernels.ResidualAdd.InitError ||
        kernels.Swiglu.InitError ||
        kernels.Rmsnorm.InitError ||
        kernels.Rope.InitError ||
        kernels.Attention.InitError ||
        kernels.VecAdd.InitError ||
        std.mem.Allocator.Error;

    /// The bundle on a context with the reported device limits. A kernel test or a bench states no
    /// override.
    pub fn init(gpa: std.mem.Allocator) InitError!*Runner {
        return initWith(gpa, .{});
    }

    /// The bundle on a context opened with `options`. The per-op backend passes the override that
    /// the CLI resolved.
    pub fn initWith(gpa: std.mem.Allocator, options: vulkan_ctx.Options) InitError!*Runner {
        const self = try gpa.create(Runner);
        errdefer gpa.destroy(self);

        self.gpa = gpa;
        self.ctx = try vulkan_ctx.Context.init(gpa, options);
        errdefer self.ctx.deinit();

        // From here on, `&self.ctx` is stable for the lifetime of the runner. Each kernel below
        // stores it.
        self.vec_add = try kernels.VecAdd.init(&self.ctx);
        errdefer self.vec_add.deinit();

        self.residual_add = try kernels.ResidualAdd.init(&self.ctx);
        errdefer self.residual_add.deinit();

        self.swiglu = try kernels.Swiglu.init(&self.ctx);
        errdefer self.swiglu.deinit();

        self.rmsnorm = try kernels.Rmsnorm.init(&self.ctx);
        errdefer self.rmsnorm.deinit();

        self.rope = try kernels.Rope.init(&self.ctx);
        errdefer self.rope.deinit();

        self.attention = try kernels.Attention.init(&self.ctx);
        errdefer self.attention.deinit();

        self.linear_q8_0 = try kernels.LinearQ80.init(&self.ctx);
        errdefer self.linear_q8_0.deinit();

        self.linear_q8_0_reduce = try kernels.LinearQ80Reduce.init(&self.ctx);
        errdefer self.linear_q8_0_reduce.deinit();

        self.linear_q8_0_subgroup = try kernels.LinearQ80Subgroup.init(&self.ctx);
        errdefer self.linear_q8_0_subgroup.deinit();

        self.linear_q4_0_reduce = try kernels.LinearQ40Reduce.init(&self.ctx);
        return self;
    }

    pub fn deinit(self: *Runner) void {
        self.linear_q4_0_reduce.deinit();
        self.linear_q8_0_subgroup.deinit();
        self.linear_q8_0_reduce.deinit();
        self.linear_q8_0.deinit();
        self.attention.deinit();
        self.rope.deinit();
        self.rmsnorm.deinit();
        self.swiglu.deinit();
        self.residual_add.deinit();
        self.vec_add.deinit();
        self.ctx.deinit();
        self.gpa.destroy(self);
    }
};
