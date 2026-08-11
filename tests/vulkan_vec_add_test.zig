//! Proof-of-life gate for the Vulkan op backend.
//!
//! The gate runs `vec_add` on the device, and every element must equal the CPU
//! sum bit for bit. The comparison is exact because the shader does one scalar
//! f32 add, with no reduction and no fused multiply-add.
//!
//! Two element counts cover the workgroup tail. 1024 is an exact multiple of
//! the 256-wide workgroup. 1000 leaves 24 lanes that take the early-return path
//! inside the shader and write nothing.
//!
//! The test returns `error.SkipZigTest` when no Vulkan device is reachable.
//! That covers a host with no driver, and the llvmpipe software device that
//! `pickPhysicalDevice` rejects by design.

const std = @import("std");
const z = @import("zigferencer");

// -- fixtures ---------------------------------------------------------------

fn runVecAddCheck(allocator: std.mem.Allocator, n: usize) !void {
    const runner = z.vulkan_runner.Runner.init(allocator) catch |e| switch (e) {
        error.NoVulkanDevice,
        error.InitializationFailed,
        error.IncompatibleDriver,
        error.FeatureNotPresent,
        => return error.SkipZigTest,
        else => return e,
    };
    defer runner.deinit();
    const kernel = &runner.vec_add;

    const a = try allocator.alloc(f32, n);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, n);
    defer allocator.free(b);
    const out_gpu = try allocator.alloc(f32, n);
    defer allocator.free(out_gpu);

    // The ramp exposes an off-by-one index or a wrong stride without a wide
    // range of magnitudes.
    for (a, 0..) |*v, i| v.* = @floatFromInt(i);
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * -0.5 + 1.0;

    try kernel.run(a, b, out_gpu);

    for (0..n) |i| {
        const expected = a[i] + b[i];
        if (out_gpu[i] != expected) {
            std.debug.print(
                "mismatch at i={d}: gpu={d} expected={d} (a={d}, b={d})\n",
                .{ i, out_gpu[i], expected, a[i], b[i] },
            );
            return error.GpuResultMismatchesCpu;
        }
    }
}

// -- tests ------------------------------------------------------------------

test "vec_add: GPU result matches CPU bit-exact, n=1024" {
    try runVecAddCheck(std.testing.allocator, 1024);
}

test "vec_add: GPU result matches CPU bit-exact, n=1000" {
    try runVecAddCheck(std.testing.allocator, 1000);
}
