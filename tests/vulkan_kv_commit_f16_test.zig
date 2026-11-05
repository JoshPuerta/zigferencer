//! The rounding gate for the f16 KV cache.
//!
//! The quant kernels already read `float16_t` buffers, so f16 loads are proven
//! on this stack. Nothing proved an f16 STORE, and the KV cache needs one: the
//! commit narrows f32 rows into f16 storage. `kv_commit_f16.comp` is the kernel
//! that does it on every f16 token, and this file gates it.
//!
//! The gate is bits, not values. The CPU commit narrows with F16C, which rounds
//! to nearest even. The shader pins SPIR-V `RoundingModeRTE` at width 16 to ask
//! the device for the same rule. Two executors that round the same way are
//! the property under test. The bits are what the cache stores, so comparing
//! them states that property directly and keeps a second conversion out of the
//! path under test.
//!
//! Each fixture below carries the f16 bit pattern that IEEE 754 requires,
//! written out by hand. The first check is that the `@floatCast` of Zig
//! produces it, so a mistyped fixture cannot pass as a device fault. The second
//! check is that the device produces the same. Together they witness the
//! rounding identity that the CPU commit and the device commit must share. One
//! KV cache then reads the same on both.
//!
//! The kernel narrows K and V under one element count and one dispatch, so
//! every test drives both buffers. V carries the fixtures in reverse order, so
//! a kernel that feeds one source into both outputs fails the gate.
//!
//! The three tests cover the finite range, the dispatch tail and overflow. The
//! 18 finite fixtures hold exact values and both directions of ordinary
//! rounding. They also hold ties that resolve up and down to an even
//! significand, and a tie that carries into the exponent. The last of them
//! cover the subnormal range with its tie to even zero, and the
//! subnormal-to-normal boundary. The sweep runs 1000 elements, so the dispatch
//! takes 16 groups of 64 plus a tail whose invocations take the early-return
//! path. The overflow test is an observation and not an equality gate, for the
//! reason its own comment states.
//!
//! The tests return `error.SkipZigTest` when Vulkan cannot start.

const std = @import("std");
const z = @import("zigferencer");

// -- fixtures ---------------------------------------------------------------

const Case = struct {
    /// The f32 input the host uploads.
    v: f32,
    /// The f16 bit pattern IEEE 754 round-to-nearest-even requires.
    bits: u16,
};

/// Finite fixtures. Every one is exactly representable in f32, so the value the
/// device receives is the value written here.
const finite_cases = [_]Case{
    .{ .v = 0.0, .bits = 0x0000 },
    .{ .v = 1.0, .bits = 0x3C00 },
    .{ .v = -1.0, .bits = 0xBC00 },
    // 1 + 2^-12 sits below the halfway point above 1.0, so it rounds down.
    .{ .v = 1.000244140625, .bits = 0x3C00 },
    // 1 + 2^-11 + 2^-12 sits above it, so it rounds up.
    .{ .v = 1.000732421875, .bits = 0x3C01 },
    // 1 + 2^-11 is the halfway point itself. The neighbors are 0x3C00 and
    // 0x3C01, and the even significand is 0x3C00, so the tie rounds down.
    .{ .v = 1.00048828125, .bits = 0x3C00 },
    // 1 + 2^-10 + 2^-11 is the next halfway point up. Its neighbors are 0x3C01
    // and 0x3C02, and now the even significand is the upper one, so the same
    // rule sends this tie up. A truncating device fails here, and a
    // round-half-away device fails the fixture above.
    .{ .v = 1.00146484375, .bits = 0x3C02 },
    .{ .v = -1.00048828125, .bits = 0xBC00 },
    .{ .v = -1.00146484375, .bits = 0xBC02 },
    // 1 - 2^-12 is the halfway point below 1.0. Rounding to the even
    // significand carries into the exponent field.
    .{ .v = 0.999755859375, .bits = 0x3C00 },
    // 2^-24, the smallest f16 subnormal, exact.
    .{ .v = 5.9604644775390625e-8, .bits = 0x0001 },
    // 2^-25 is the halfway point between zero and that subnormal. Zero has the
    // even significand, so the tie rounds to zero.
    .{ .v = 2.98023223876953125e-8, .bits = 0x0000 },
    // 1.5 * 2^-24 is the next halfway point up, and it rounds away from zero
    // for the same even-significand reason.
    .{ .v = 8.940696716308594e-8, .bits = 0x0002 },
    // 1023 * 2^-24, the largest f16 subnormal, exact.
    .{ .v = 6.097555160522461e-5, .bits = 0x03FF },
    // 1023.5 * 2^-24 is the halfway point across the subnormal-to-normal
    // boundary, where the exponent field changes from 0 to 1.
    .{ .v = 6.100535392761230e-5, .bits = 0x0400 },
    // 2^-14, the smallest f16 normal, exact.
    .{ .v = 6.103515625e-5, .bits = 0x0400 },
    // 65504 is the largest finite f16, and 65519 still sits below the halfway
    // point above it.
    .{ .v = 65504.0, .bits = 0x7BFF },
    .{ .v = 65519.0, .bits = 0x7BFF },
};

/// Fixtures whose exact result overflows the f16 exponent range. 65520 is the
/// halfway point above the largest finite value, and round-to-nearest-even
/// resolves it upward, to infinity.
const overflow_cases = [_]Case{
    .{ .v = 65520.0, .bits = 0x7C00 },
    .{ .v = 70000.0, .bits = 0x7C00 },
    .{ .v = -70000.0, .bits = 0xFC00 },
};

fn initCtx() !z.vulkan_ctx.Context {
    return z.vulkan_ctx.Context.init(std.testing.allocator, .{}) catch |e| switch (e) {
        error.NoVulkanDevice,
        error.NoComputeQueue,
        error.InitializationFailed,
        error.IncompatibleDriver,
        error.FeatureNotPresent,
        => return error.SkipZigTest,
        else => return e,
    };
}

/// Narrow `k_src` and `v_src` on the device and write the raw f16 bits into
/// `k_bits` and `v_bits`.
fn deviceBits(
    ctx: *const z.vulkan_ctx.Context,
    k_src: []const f32,
    v_src: []const f32,
    k_bits: []u16,
    v_bits: []u16,
) !void {
    var kernel = try z.ops_vulkan.KvCommitF16.init(ctx);
    defer kernel.deinit();

    try z.ops_vulkan.kvCommitF16OneShot(&kernel, k_src, v_src, k_bits, v_bits);
}

/// Write `src` in reverse order into `dst`, the V-side source of every test.
fn reverseInto(dst: []f32, src: []const f32) void {
    std.debug.assert(dst.len == src.len);

    for (src, 0..) |v, i| {
        dst[src.len - 1 - i] = v;
    }
}

/// Accept the two outcomes Vulkan permits for a narrowing overflow, and print
/// the one the device chose.
fn reportOverflow(tensor: []const u8, c: Case, bits: u16) !void {
    // Same sign, and either the infinity or the largest finite value.
    const saturated: u16 = (c.bits & 0x8000) | 0x7BFF;
    if (bits == c.bits) {
        std.debug.print("overflow of {s} {d}: infinity, 0x{X:0>4}\n", .{ tensor, c.v, bits });
    } else if (bits == saturated) {
        std.debug.print(
            "overflow of {s} {d}: saturated to the largest finite, 0x{X:0>4}\n",
            .{ tensor, c.v, bits },
        );
    } else {
        std.debug.print(
            "overflow of {s} {d}: 0x{X:0>4}, neither infinity 0x{X:0>4} nor saturation 0x{X:0>4}\n",
            .{ tensor, c.v, bits, c.bits, saturated },
        );
        return error.OverflowLeftTheFiniteRangeWrongly;
    }
}

// -- tests ------------------------------------------------------------------

test "kv_commit_f16: finite values round to nearest even" {
    const n = finite_cases.len;

    var k_src: [n]f32 = undefined;
    var want: [n]u16 = undefined;
    for (finite_cases, 0..) |c, i| {
        k_src[i] = c.v;
        want[i] = c.bits;
    }

    var v_src: [n]f32 = undefined;
    reverseInto(&v_src, &k_src);

    // The host runs first. A fixture that the narrowing of Zig disagrees with
    // is a fixture bug, and a catch here keeps it from reading as a device
    // fault.
    var k_host: [n]u16 = undefined;
    for (k_src, 0..) |v, i| {
        k_host[i] = @bitCast(@as(f16, @floatCast(v)));
    }

    try std.testing.expectEqualSlices(u16, &want, &k_host);
    var v_host: [n]u16 = undefined;
    for (v_src, 0..) |v, i| {
        v_host[i] = @bitCast(@as(f16, @floatCast(v)));
    }

    var ctx = try initCtx();
    defer ctx.deinit();

    var k_got: [n]u16 = undefined;
    var v_got: [n]u16 = undefined;
    try deviceBits(&ctx, &k_src, &v_src, &k_got, &v_got);
    try std.testing.expectEqualSlices(u16, &k_host, &k_got);
    try std.testing.expectEqualSlices(u16, &v_host, &v_got);
}

test "kv_commit_f16: a sweep straddles the workgroup boundary" {
    const allocator = std.testing.allocator;
    const n: usize = 1000;

    // A deterministic sweep across the exponent range, dense enough that a
    // sizeable share of the values land on or beside a rounding boundary. The
    // count is not a multiple of the 64-wide workgroup, so the tail
    // invocations take the early-return path and must leave nothing behind.
    var prng = std.Random.DefaultPrng.init(0x16F16);
    const rand = prng.random();
    const k_src = try allocator.alloc(f32, n);
    defer allocator.free(k_src);

    for (k_src, 0..) |*v, i| {
        const scale = std.math.ldexp(@as(f32, 1.0), @as(i32, @intCast(i % 30)) - 20);
        v.* = (rand.float(f32) * 2.0 - 1.0) * scale;
    }

    const v_src = try allocator.alloc(f32, n);
    defer allocator.free(v_src);

    reverseInto(v_src, k_src);

    const k_host = try allocator.alloc(u16, n);
    defer allocator.free(k_host);

    for (k_src, 0..) |v, i| {
        k_host[i] = @bitCast(@as(f16, @floatCast(v)));
    }

    const v_host = try allocator.alloc(u16, n);
    defer allocator.free(v_host);

    for (v_src, 0..) |v, i| {
        v_host[i] = @bitCast(@as(f16, @floatCast(v)));
    }

    var ctx = try initCtx();
    defer ctx.deinit();

    const k_got = try allocator.alloc(u16, n);
    defer allocator.free(k_got);

    const v_got = try allocator.alloc(u16, n);
    defer allocator.free(v_got);

    try deviceBits(&ctx, k_src, v_src, k_got, v_got);
    try std.testing.expectEqualSlices(u16, k_host, k_got);
    try std.testing.expectEqualSlices(u16, v_host, v_got);
}

test "kv_commit_f16: overflow leaves the finite range" {
    const n = overflow_cases.len;

    // This one is an observation, not an equality gate, and the reason is a
    // recorded device property: the float controls report
    // `shaderSignedZeroInfNanPreserveFloat16 = false`. Vulkan then permits an
    // implementation to saturate a narrowing overflow to the largest finite
    // value instead of producing an infinity, and both outcomes are legal. The
    // KV cache is unaffected either way, because its values are finite by
    // construction. The test therefore asserts the two permitted outcomes and
    // prints the one the device chose. A gate pinned to one outcome makes a
    // legal device read as a broken one.
    var k_src: [n]f32 = undefined;
    for (overflow_cases, 0..) |c, i| {
        k_src[i] = c.v;
    }

    var v_src: [n]f32 = undefined;
    reverseInto(&v_src, &k_src);

    var ctx = try initCtx();
    defer ctx.deinit();

    var k_got: [n]u16 = undefined;
    var v_got: [n]u16 = undefined;
    try deviceBits(&ctx, &k_src, &v_src, &k_got, &v_got);

    for (overflow_cases, k_got) |c, bits| {
        try reportOverflow("k", c, bits);
    }

    for (0..n) |i| {
        try reportOverflow("v", overflow_cases[n - 1 - i], v_got[i]);
    }
}
