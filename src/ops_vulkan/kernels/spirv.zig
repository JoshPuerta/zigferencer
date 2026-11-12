//! The SPIR-V blob of each compute shader, and the table that names each blob in a trace build.
//!
//! SPIR-V embeds. `glslc` makes each blob at build time, and `build.zig` declares its anonymous
//! import. `createShaderModule` casts the blob back to `[*]const u32`, so every declaration forces
//! `align(@alignOf(u32))` on the embed. A bare `@embedFile` has alignment 1.

const std = @import("std");
const core_log = @import("../../core/utils/log.zig");

// -- constants --------------------------------------------------------------

// One blob per shader, in the form that "SPIR-V embeds" in the header states.
pub const vec_add_spv: [
    @embedFile(
        "vec_add.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("vec_add.spv").*;

pub const residual_add_spv: [
    @embedFile(
        "residual_add.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("residual_add.spv").*;

pub const sigmoid_gate_spv: [
    @embedFile(
        "sigmoid_gate.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("sigmoid_gate.spv").*;

pub const attention_spv: [
    @embedFile(
        "attention.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("attention.spv").*;

pub const attention_f16_spv: [
    @embedFile(
        "attention_f16.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("attention_f16.spv").*;

pub const attention_split_spv: [
    @embedFile(
        "attention_split.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("attention_split.spv").*;

pub const attention_split_f16_spv: [
    @embedFile(
        "attention_split_f16.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("attention_split_f16.spv").*;

pub const attention_merge_spv: [
    @embedFile(
        "attention_merge.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("attention_merge.spv").*;

pub const rope_spv: [@embedFile("rope.spv").len]u8 align(@alignOf(u32)) = @embedFile("rope.spv").*;

pub const rope_multi_spv: [
    @embedFile(
        "rope_multi.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("rope_multi.spv").*;

pub const rmsnorm_spv: [
    @embedFile(
        "rmsnorm.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("rmsnorm.spv").*;

pub const swiglu_spv: [
    @embedFile(
        "swiglu.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("swiglu.spv").*;

pub const linear_q8_0_spv: [
    @embedFile(
        "linear_q8_0.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("linear_q8_0.spv").*;

pub const linear_q8_0_reduce_spv: [
    @embedFile(
        "linear_q8_0_reduce.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("linear_q8_0_reduce.spv").*;

pub const linear_q4_0_reduce_spv: [
    @embedFile(
        "linear_q4_0_reduce.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("linear_q4_0_reduce.spv").*;

pub const linear_q8_0_subgroup_spv: [
    @embedFile(
        "linear_q8_0_subgroup.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("linear_q8_0_subgroup.spv").*;

pub const linear_q8_0_batch_spv: [
    @embedFile(
        "linear_q8_0_batch.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("linear_q8_0_batch.spv").*;

pub const linear_q8_0_gemm_spv: [
    @embedFile(
        "linear_q8_0_gemm.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("linear_q8_0_gemm.spv").*;

pub const linear_q4_0_batch_spv: [
    @embedFile(
        "linear_q4_0_batch.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("linear_q4_0_batch.spv").*;

pub const rmsnorm_batch_spv: [
    @embedFile(
        "rmsnorm_batch.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("rmsnorm_batch.spv").*;

pub const rope_multi_batch_spv: [
    @embedFile(
        "rope_multi_batch.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("rope_multi_batch.spv").*;

pub const attention_batch_spv: [
    @embedFile(
        "attention_batch.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("attention_batch.spv").*;

pub const attention_batch_f16_spv: [
    @embedFile(
        "attention_batch_f16.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("attention_batch_f16.spv").*;

pub const attention_multi_stream_spv: [
    @embedFile(
        "attention_multi_stream.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("attention_multi_stream.spv").*;

pub const attention_multi_stream_f16_spv: [
    @embedFile(
        "attention_multi_stream_f16.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("attention_multi_stream_f16.spv").*;

pub const kv_commit_f16_spv: [
    @embedFile(
        "kv_commit_f16.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("kv_commit_f16.spv").*;

pub const gdn_decode_spv: [
    @embedFile(
        "gdn_decode.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("gdn_decode.spv").*;

pub const gdn_chunk_conv_spv: [
    @embedFile(
        "gdn_chunk_conv.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("gdn_chunk_conv.spv").*;

pub const gdn_chunk_scan_spv: [
    @embedFile(
        "gdn_chunk_scan.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("gdn_chunk_scan.spv").*;

pub const gdn_chunk_norm_spv: [
    @embedFile(
        "gdn_chunk_norm.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("gdn_chunk_norm.spv").*;

pub const qwen_qk_norm_spv: [
    @embedFile(
        "qwen_qk_norm.spv",
    ).len
]u8 align(@alignOf(u32)) = @embedFile("qwen_qk_norm.spv").*;

/// The name of each SPIR-V blob, keyed by its address. A trace build reads it to name the shader
/// of a recorded dispatch, and a build without trace holds an empty table.
pub const spv_names: []const SpvName = if (core_log.hot) &.{
    .{ .code = &vec_add_spv, .name = "vec_add" },
    .{ .code = &residual_add_spv, .name = "residual_add" },
    .{ .code = &sigmoid_gate_spv, .name = "sigmoid_gate" },
    .{ .code = &attention_spv, .name = "attention" },
    .{ .code = &attention_f16_spv, .name = "attention_f16" },
    .{ .code = &attention_split_spv, .name = "attention_split" },
    .{ .code = &attention_split_f16_spv, .name = "attention_split_f16" },
    .{ .code = &attention_merge_spv, .name = "attention_merge" },
    .{ .code = &rope_spv, .name = "rope" },
    .{ .code = &rope_multi_spv, .name = "rope_multi" },
    .{ .code = &rmsnorm_spv, .name = "rmsnorm" },
    .{ .code = &swiglu_spv, .name = "swiglu" },
    .{ .code = &linear_q8_0_spv, .name = "linear_q8_0" },
    .{ .code = &linear_q8_0_reduce_spv, .name = "linear_q8_0_reduce" },
    .{ .code = &linear_q4_0_reduce_spv, .name = "linear_q4_0_reduce" },
    .{ .code = &linear_q8_0_subgroup_spv, .name = "linear_q8_0_subgroup" },
    .{ .code = &linear_q8_0_batch_spv, .name = "linear_q8_0_batch" },
    .{ .code = &linear_q8_0_gemm_spv, .name = "linear_q8_0_gemm" },
    .{ .code = &linear_q4_0_batch_spv, .name = "linear_q4_0_batch" },
    .{ .code = &rmsnorm_batch_spv, .name = "rmsnorm_batch" },
    .{ .code = &rope_multi_batch_spv, .name = "rope_multi_batch" },
    .{ .code = &attention_batch_spv, .name = "attention_batch" },
    .{ .code = &attention_batch_f16_spv, .name = "attention_batch_f16" },
    .{ .code = &attention_multi_stream_spv, .name = "attention_multi_stream" },
    .{ .code = &attention_multi_stream_f16_spv, .name = "attention_multi_stream_f16" },
    .{ .code = &kv_commit_f16_spv, .name = "kv_commit_f16" },
    .{ .code = &gdn_decode_spv, .name = "gdn_decode" },
    .{ .code = &gdn_chunk_conv_spv, .name = "gdn_chunk_conv" },
    .{ .code = &gdn_chunk_scan_spv, .name = "gdn_chunk_scan" },
    .{ .code = &gdn_chunk_norm_spv, .name = "gdn_chunk_norm" },
    .{ .code = &qwen_qk_norm_spv, .name = "qwen_qk_norm" },
} else &.{};

// -- types ------------------------------------------------------------------

/// One row of `spv_names`.
pub const SpvName = struct { code: *const anyopaque, name: []const u8 };

// -- tests ------------------------------------------------------------------

test "spv_names holds a row for every SPIR-V blob of this file" {
    const source = @embedFile("spirv.zig");
    var declared: usize = 0;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "pub const ")) continue;

        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const name = line["pub const ".len..colon];
        if (!std.mem.endsWith(u8, name, "_spv")) continue;

        var needle_buffer: [96]u8 = undefined;
        const needle = try std.fmt.bufPrint(&needle_buffer, ".code = &{s},", .{name});
        std.testing.expect(std.mem.indexOf(u8, source, needle) != null) catch |err| {
            std.debug.print("spv_names lacks a row for {s}\n", .{name});
            return err;
        };
        declared += 1;
    }

    try std.testing.expect(declared > 0);
    if (core_log.hot) {
        try std.testing.expectEqual(declared, spv_names.len);
    }
}
