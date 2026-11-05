//! The batched-linear interface: one function picks the kernel by the weight dtype
//! and fans the quantized paths across rows. Both CPU runners call it, so the
//! dtype ladder cannot drift between the two families.

const build_options = @import("build_options");
const Tensor = @import("../core/tensor.zig").Tensor;
const parallel = @import("parallel.zig");
const Pool = @import("pool.zig").Pool;
const Error = @import("step_row.zig").Error;

/// Backend selected at build time by `-Dops-backend=naive|simd|simd_pf|vulkan|auto`.
/// The default is `auto`, which picks the strongest eligible CPU variant. The
/// `vulkan` backend needs `-Druntime=vulkan`. It wraps the GPU kernels behind
/// the same op API. On any other GPU build the CPU runners ride along as CPU
/// capability, so this interface takes the `cpu_ops` variant the build resolved.
const ops = if (build_options.use_vulkan_ops)
    @import("../ops_vulkan/wrap.zig")
else if (build_options.cpu_ops_simd_pf)
    @import("../ops/simd_pf.zig")
else if (build_options.cpu_ops_simd)
    @import("../ops/simd.zig")
else
    @import("../ops/naive.zig");

// -- api --------------------------------------------------------------------

/// Dispatch the right batched linear kernel by weight dtype, with optional
/// row-parallel fan-out for the quantized paths. Shared by the two CPU
/// runners so the dtype ladder cannot drift. The f32 path is for norms
/// (tiny tensors), which are not worth parallelizing. The kernel writes
/// `outs` rows at stride `w.shape[1]`.
pub fn linearBatchDispatch(
    pool: *Pool,
    policy: parallel.Policy,
    inputs: []const f32,
    w: Tensor,
    outs: []f32,
    batch: usize,
) Error!void {
    switch (w.dtype) {
        .f32 => ops.linearF32Batch(inputs, w, outs, batch, @intCast(w.shape[1])),
        .q8_0 => if (comptime build_options.use_vulkan_ops) {
            // GPU dispatches the whole matmul as one workgroup grid. A row
            // split across CPU threads issues N independent submits, which
            // strictly loses against one submit.
            ops.linearF32Q80Batch(inputs, w, outs, batch, @intCast(w.shape[1]));
        } else {
            parallel.linearBatch(pool, policy, ops.linearF32Q80Batch, inputs, w, outs, batch);
        },
        .q4_0 => if (comptime build_options.use_vulkan_ops) {
            return error.UnsupportedWeightDtype; // GPU per-op backend has no q4 kernel
        } else {
            parallel.linearBatch(pool, policy, ops.linearF32Q40Batch, inputs, w, outs, batch);
        },
        .q4_1 => if (comptime build_options.use_vulkan_ops) {
            return error.UnsupportedWeightDtype; // GPU per-op backend has no q4 kernel
        } else {
            parallel.linearBatch(pool, policy, ops.linearF32Q41Batch, inputs, w, outs, batch);
        },
        .q6_k => if (comptime build_options.use_vulkan_ops) {
            return error.UnsupportedWeightDtype; // GPU per-op backend requantizes q6_k at load
        } else {
            parallel.linearBatch(pool, policy, ops.linearF32Q6KBatch, inputs, w, outs, batch);
        },
        else => return error.UnsupportedWeightDtype,
    }
}
