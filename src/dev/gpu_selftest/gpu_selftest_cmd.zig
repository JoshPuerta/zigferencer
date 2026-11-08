//! `zigferencer gpu-selftest [gdn|attn|matmul]` checks one GPU kernel against its
//! verified CPU reference on random inputs. A divergence shows here as a numeric diff
//! on one kernel, not as wrong tokens many layers deep in an end-to-end run.
//! Each check stays small, because a long submit is fragile under a translation-layer driver.
//!
//! Comptime. The `use_vulkan` build option removes the check path from a build without Vulkan.

const std = @import("std");
const build_options = @import("build_options");

const zigferencer = @import("zigferencer");

const context = zigferencer.cli.context;
const args = zigferencer.cli.args;

const gpu_selftest_attn = @import("gpu_selftest_attn.zig");
const gpu_selftest_gdn = @import("gpu_selftest_gdn.zig");
const gpu_selftest_matmul = @import("gpu_selftest_matmul.zig");

// -- types ------------------------------------------------------------------

/// The flags of `gpu-selftest`.
const Flags = struct {
    kernel: ?[]const u8 = null,

    pub const positional = "kernel";
    pub const help: args.Help = .{
        .usage = "zigferencer-dev gpu-selftest [gdn|attn|matmul]   (vulkan builds only)",
        .about =
        \\Bit-check one GPU kernel against its CPU reference on random inputs.
        \\gdn (default) the Gated-DeltaNet step, attn full attention and the
        \\attention kernel set, matmul the Q8_0 prefill matmuls. Exits non-zero on a
        \\numeric mismatch. Takes no flags.
        \\
        ,
    };
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: context.Context) !void {
    // The parse runs first, so --help works on any build. The guard then returns on
    // a CPU build, so the vulkan-only checks are never analyzed.
    const flags = try args.parseInto(Flags, ctx) orelse return;
    const which = flags.kernel orelse "gdn";

    if (comptime !build_options.use_vulkan) {
        args.writeStderr(
            ctx.io,
            "gpu-selftest requires a Vulkan build (-Druntime=vulkan -Dops-backend=vulkan_fused)\n",
        );
        return error.RequiresVulkan;
    }

    if (std.mem.eql(u8, which, "gdn")) return gpu_selftest_gdn.gdnCheck(ctx);
    if (std.mem.eql(u8, which, "attn")) return gpu_selftest_attn.attnCheck(ctx);
    if (std.mem.eql(u8, which, "matmul")) return gpu_selftest_matmul.matmulCheck(ctx);

    return args.fail(ctx.io, "unknown kernel '{s}' (expected gdn|attn|matmul)", .{which});
}
