//! The batched-linear interface: one function picks the kernel by the weight dtype and fans the
//! quantized paths across rows. Both CPU runners call it, so the dtype ladder cannot drift between
//! the two families.
//!
//! Comptime. `build_options.use_vulkan_ops` picks the Vulkan path, so a CPU build never reads it.
//!
//! Load gate. A GGUF names each dtype, so a file can name one that no kernel reads.
//! `requireWeights` refuses such a file at runner init and names the tensor. The step functions
//! then carry no dtype error, and the dispatch treats any other dtype as unreachable.
//!
//! Fan-out. The quantized paths split rows across CPU threads. The f32 path serves norms, which
//! are tiny, so it stays serial. The Vulkan backend submits one workgroup grid for the whole
//! matmul, because N independent submits from a row split lose against one. That backend also
//! has no q4 or q6_k kernel.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const dtype = @import("../core/dtype.zig");
const kv_budget = @import("../core/kv_budget.zig");
const Tensor = @import("../core/tensor.zig").Tensor;
const loader = @import("../io/gguf/loader.zig");
const parallel = @import("parallel.zig");
const Pool = @import("pool.zig").Pool;
const tensor_lookup = @import("tensor_lookup.zig");

/// The backend that `-Dops-backend` selected. On a GPU build without the `vulkan` backend, the CPU
/// runners take the `cpu_ops` variant the build resolved.
const ops = if (build_options.use_vulkan_ops)
    @import("../ops_vulkan/wrap.zig")
else if (build_options.cpu_ops_simd_pf)
    @import("../ops/simd_pf.zig")
else if (build_options.cpu_ops_simd)
    @import("../ops/simd.zig")
else
    @import("../ops/naive.zig");

const log = std.log.scoped(.runner);

// -- constants --------------------------------------------------------------

/// The name endings of the block weights that reach `linearBatchDispatch`, in both families.
const block_linear_ends = [_][]const u8{
    ".attn_q.weight",
    ".attn_k.weight",
    ".attn_v.weight",
    ".attn_output.weight",
    ".attn_qkv.weight",
    ".attn_gate.weight",
    ".ssm_alpha.weight",
    ".ssm_beta.weight",
    ".ssm_out.weight",
    ".ffn_gate.weight",
    ".ffn_up.weight",
    ".ffn_down.weight",
};

/// The name endings of the block weights that the runners read as f32, with no kernel between.
const block_f32_ends = [_][]const u8{
    "_norm.weight",
    ".ssm_a",
    ".ssm_dt.bias",
    ".ssm_conv1d.weight",
};

pub const embed_name = "token_embd.weight";
pub const head_name = "output.weight";
const output_norm_name = "output_norm.weight";

/// The refusal line: the tensor name, then the dtype.
pub const refusal_fmt = "tensor {s}: no kernel reads dtype {t} in this build";

/// The test runner fails a test that logs an error, and a refusal test expects this line.
const refusal_level: std.log.Level = if (builtin.is_test) .debug else .err;

// -- api --------------------------------------------------------------------

/// True when `linearBatchDispatch` holds a kernel for `d` in this build.
pub fn linearHasKernel(d: dtype.GgmlType) bool {
    return switch (d) {
        .f32, .q8_0 => true,
        .q4_0, .q4_1, .q6_k => !build_options.use_vulkan_ops,
        else => false,
    };
}

/// True when `embedLookup` reads `d`.
pub fn embedHasKernel(d: dtype.GgmlType) bool {
    return switch (d) {
        .f32, .q8_0, .q4_0, .q4_1 => true,
        else => false,
    };
}

/// The first weight of layers `[0..n_layers)`, the embedding, or the lm_head with a dtype that
/// no kernel reads, or null. A model with no `output.weight` ties the lm_head to the embedding.
/// A norm or a GDN parameter must be f32, because a runner views its bytes as f32 in place.
pub fn refusedWeight(refs: []const loader.TensorRef, n_layers: u32) ?*const loader.TensorRef {
    const tied_head = tensor_lookup.find(refs, head_name) == null;
    for (refs) |*r| {
        const d = r.tensor.dtype;
        if (std.mem.eql(u8, r.name, embed_name)) {
            if (!embedHasKernel(d)) return r;
            if (tied_head and !linearHasKernel(d)) return r;
        } else if (isLinearWeight(r.name, n_layers) and !linearHasKernel(d)) {
            return r;
        } else if (isF32Weight(r.name, n_layers) and d != .f32) {
            return r;
        }
    }

    return null;
}

/// Refuse the weights that `refusedWeight` finds, with one log line that names the tensor.
pub fn requireWeights(
    refs: []const loader.TensorRef,
    n_layers: u32,
) error{UnsupportedWeightDtype}!void {
    const refused = refusedWeight(refs, n_layers) orelse return;
    const args = .{ refused.name, refused.tensor.dtype };
    switch (refusal_level) {
        .err => log.err(refusal_fmt, args),
        else => log.debug(refusal_fmt, args),
    }

    return error.UnsupportedWeightDtype;
}

/// Dispatch the batched linear kernel for the weight dtype. The kernel writes `outs` rows at
/// stride `w.shape[1]`. `requireWeights` proves the dtype at runner init.
pub fn linearBatchDispatch(
    pool: *Pool,
    policy: parallel.Policy,
    inputs: []const f32,
    w: Tensor,
    outs: []f32,
    batch: usize,
) void {
    switch (w.dtype) {
        .f32 => ops.linearF32Batch(inputs, w, outs, batch, @intCast(w.shape[1])),
        .q8_0 => if (comptime build_options.use_vulkan_ops) {
            // One submit for the whole matmul. The header explains the fan-out.
            ops.linearF32Q80Batch(inputs, w, outs, batch, @intCast(w.shape[1]));
        } else {
            parallel.linearBatch(pool, policy, ops.linearF32Q80Batch, inputs, w, outs, batch);
        },
        .q4_0 => if (comptime build_options.use_vulkan_ops) {
            unreachable; // The Vulkan backend has no q4 kernel.
        } else {
            parallel.linearBatch(pool, policy, ops.linearF32Q40Batch, inputs, w, outs, batch);
        },
        .q4_1 => if (comptime build_options.use_vulkan_ops) {
            unreachable; // The Vulkan backend has no q4 kernel.
        } else {
            parallel.linearBatch(pool, policy, ops.linearF32Q41Batch, inputs, w, outs, batch);
        },
        .q6_k => if (comptime build_options.use_vulkan_ops) {
            unreachable; // The Vulkan backend has no q6_k kernel.
        } else {
            parallel.linearBatch(pool, policy, ops.linearF32Q6KBatch, inputs, w, outs, batch);
        },
        else => unreachable,
    }
}

// -- helpers ----------------------------------------------------------------

fn isLinearWeight(name: []const u8, n_layers: u32) bool {
    if (std.mem.eql(u8, name, head_name)) return true;

    const block = tensor_lookup.blockIndex(name) orelse return false;
    if (block >= n_layers) return false;

    for (block_linear_ends) |end| {
        if (std.mem.endsWith(u8, name, end)) return true;
    }

    return false;
}

fn isF32Weight(name: []const u8, n_layers: u32) bool {
    if (std.mem.eql(u8, name, output_norm_name)) return true;

    const block = tensor_lookup.blockIndex(name) orelse return false;
    if (block >= n_layers) return false;

    for (block_f32_ends) |end| {
        if (std.mem.endsWith(u8, name, end)) return true;
    }

    return false;
}

// -- tests ------------------------------------------------------------------

fn refFor(name: []const u8, d: dtype.GgmlType) loader.TensorRef {
    var tensor: Tensor = undefined;
    tensor.dtype = d;
    return .{ .name = name, .tensor = tensor };
}

test "refusedWeight: the embedding, the lm_head, and a block weight in range" {
    const norm = "blk.0.attn_norm.weight";
    const q0 = "blk.0.attn_q.weight";
    const q1 = "blk.1.attn_q.weight";
    const Row = struct { refs: []const loader.TensorRef, want: ?[]const u8 };
    const rows = [_]Row{
        .{ .refs = &.{ refFor(embed_name, .q8_0), refFor(q0, .q8_0) }, .want = null },
        .{ .refs = &.{ refFor(embed_name, .q5_0), refFor(q0, .q8_0) }, .want = embed_name },
        .{ .refs = &.{ refFor(embed_name, .q8_0), refFor(q0, .q5_0) }, .want = q0 },
        .{ .refs = &.{ refFor(embed_name, .q8_0), refFor(head_name, .i8) }, .want = head_name },
        // q6_k has a matmul kernel and no embedding kernel.
        .{ .refs = &.{refFor(embed_name, .q6_k)}, .want = embed_name },
        // Only layer 0 runs here.
        .{ .refs = &.{ refFor(q1, .q5_0), refFor(norm, .f32) }, .want = null },
        // A runner views a norm as f32 in place, so a quantized norm has no reader.
        .{ .refs = &.{refFor(norm, .q8_0)}, .want = norm },
        .{ .refs = &.{refFor(output_norm_name, .f16)}, .want = output_norm_name },
        .{ .refs = &.{refFor("blk.0.ssm_a", .q8_0)}, .want = "blk.0.ssm_a" },
        .{ .refs = &.{refFor("blk.1.ssm_a", .q8_0)}, .want = null },
    };
    for (rows) |row| {
        const got = refusedWeight(row.refs, 1);
        if (row.want) |want| {
            try std.testing.expectEqualStrings(want, got.?.name);
            var buf: [96]u8 = undefined;
            const text = try std.fmt.bufPrint(&buf, refusal_fmt, .{ want, got.?.tensor.dtype });
            try std.testing.expect(std.mem.find(u8, text, want) != null);
        } else {
            try std.testing.expectEqual(null, got);
        }
    }
}

fn expectRunnerRefuses(comptime Runner: type, card: loader.ModelCard) !void {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();

    var pool: Pool = undefined;
    try pool.init(std.testing.io, 1, 0);
    defer pool.deinit();

    const cases = [_][]const loader.TensorRef{
        &.{refFor(embed_name, .q5_0)},
        &.{ refFor(embed_name, .q8_0), refFor("blk.0.attn_q.weight", .q5_0) },
    };
    for (cases) |refs| {
        var loaded: loader.Loaded = undefined;
        loaded.card = card;
        loaded.tensors = refs;
        try std.testing.expectError(error.UnsupportedWeightDtype, Runner.init(
            arena_state.allocator(),
            &loaded,
            kv_budget.pinned(1, 16),
            1,
            &pool,
            .{},
            .f32,
        ));
    }
}

test "runner init refuses a dtype from the file before the first step" {
    var card: loader.ModelCard = undefined;
    card.n_layers = 1;
    card.arch = .llama;
    card.gdn = null;
    try expectRunnerRefuses(@import("llama_runner.zig").LlamaRunner, card);

    var gdn: loader.GdnParams = undefined;
    gdn.full_attention_interval = 1;
    card.arch = .qwen35;
    card.gdn = gdn;
    try expectRunnerRefuses(@import("qwen_runner.zig").QwenRunner, card);
}
