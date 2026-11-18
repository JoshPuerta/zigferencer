//! Digest gate for the batched-prefill kernels of the Qwen3.5-2B shapes.
//!
//! Comptime. Asserts pin the shapes and batch bounds that the digest tables assume.
//!
//! Digests. A kernel change that must keep the output bytes proves it here. Each test runs the
//! current kernels on fixed seeded inputs and compares a 64-bit FNV-1a digest of the output bytes
//! with a table that a device run minted. A null entry fails the test and prints the table to paste
//! into the source. A digest that moved fails and names the case.
//!
//! GDN. The chunked-parallel prefill runs a conv pass and a scan pass at the 2B shape: 16 key
//! heads, 16 value heads, head dim 128, conv kernel 4, conv dim 6144. The incoming state and conv
//! ring are not zero. The widths 64 and 128 are the prefill chunks, and 37 leaves a tail of five
//! tokens in the last sub-chunk of 16. The conv digest covers the three conv outputs and the
//! advanced conv ring. The scan digest covers the readout rows and the state. The kernels take no
//! position, so a chunk start does not change their input.
//!
//! GEMM. The tiled kernel runs at the batch widths 64 and 128 on the three projection shapes of the
//! 2B model. The single-wave batch kernel runs at width 7 on one shape, because it serves every
//! width below 64.
//!
//! Skips. Each test returns `error.SkipZigTest` when no Vulkan device is reachable.

const std = @import("std");
const zigferencer = @import("zigferencer");

// -- fixtures ---------------------------------------------------------------

fn openRunner(allocator: std.mem.Allocator) !*zigferencer.vulkan_runner.Runner {
    return zigferencer.vulkan_runner.Runner.init(allocator) catch |e| switch (e) {
        error.NoVulkanDevice,
        error.InitializationFailed,
        error.IncompatibleDriver,
        error.FeatureNotPresent,
        => return error.SkipZigTest,
        else => return e,
    };
}

fn fillN(r: std.Random, slice: []f32, scale: f32) void {
    for (slice) |*v| {
        v.* = (r.float(f32) - 0.5) * 2.0 * scale;
    }
}

fn allFinite(slice: []const f32) bool {
    for (slice) |v| {
        if (!std.math.isFinite(v)) return false;
    }

    return true;
}

/// Fill `data` with a synthetic Q8_0 weight: `n` outputs of `k / 32` blocks, each
/// of a small random f16 scale and 32 random i8 quants.
fn buildRandomQ80(r: std.Random, data: []u8, k: u32, n: u32) void {
    const block_bytes: usize = 34;
    const blocks_per_output = k / 32;
    std.debug.assert(data.len == @as(usize, n) * blocks_per_output * block_bytes);
    var off: usize = 0;
    while (off < data.len) : (off += block_bytes) {
        const scale: f16 = @floatCast(0.005 + 0.02 * r.float(f32));
        std.mem.writeInt(u16, data[off..][0..2], @bitCast(scale), .little);
        for (data[off + 2 ..][0..32]) |*q| {
            const v: i8 = @intCast(@as(i32, @intFromFloat(@round((r.float(f32) - 0.5) * 250.0))));
            q.* = @bitCast(v);
        }
    }
}

/// Compare a table of minted digests with the digests of this run. A null
/// entry or a moved entry fails. The failure prints the table of this run in
/// the form that the source holds.
fn settle(
    comptime table: []const u8,
    comptime cases: usize,
    comptime cols: usize,
    want: [cases][cols]?u64,
    got: [cases][cols]u64,
    labels: [cases][]const u8,
) !void {
    var missing: u32 = 0;
    var moved: u32 = 0;
    for (0..cases) |i| {
        for (0..cols) |c| {
            const w = want[i][c] orelse {
                missing += 1;
                continue;
            };
            if (w == got[i][c]) continue;

            moved += 1;
            std.debug.print(
                "{s} {s} column {d}: got 0x{x:0>16}, want 0x{x:0>16}\n",
                .{ table, labels[i], c, got[i][c], w },
            );
        }
    }

    if (missing == 0 and moved == 0) return;

    std.debug.print(
        "mint: {d} digests missing, {d} moved in {s}. The table of this run:\n",
        .{ missing, moved, table },
    );
    std.debug.print("const {s}: [{d}][{d}]?u64 = .{{\n", .{ table, cases, cols });
    for (0..cases) |i| {
        std.debug.print("    .{{", .{});
        for (0..cols) |c| {
            std.debug.print(" 0x{x:0>16}{s}", .{ got[i][c], if (c + 1 < cols) "," else "" });
        }

        std.debug.print(" }}, // {s}\n", .{labels[i]});
    }

    std.debug.print("}};\n", .{});
    return if (moved > 0) error.PrefillDigestMoved else error.PrefillDigestNotMinted;
}

// -- gdn --------------------------------------------------------------------

const n_k: u32 = 16;
const n_v: u32 = 16;
const head_dim: u32 = 128;
const conv_kernel: u32 = 4;
const gdn_eps: f32 = 1e-6;
const key_dim: usize = n_k * head_dim;
const value_dim: usize = n_v * head_dim;
const conv_dim: usize = 2 * key_dim + value_dim;

const gdn_widths = [_]u32{ 64, 128, 37 };
const gdn_labels = [_][]const u8{ "b=64", "b=128", "b=37" };

comptime {
    std.debug.assert(conv_dim == 6144);
    std.debug.assert(gdn_widths[2] % 16 != 0);
}

/// The digests of `[width][conv, scan]`. A device run mints them.
const gdn_digests: [gdn_widths.len][2]?u64 = .{
    .{ 0xa84cc3baa01e1684, 0x7a57f0dbd09b61e2 }, // b=64
    .{ 0x4029ac4a3c3472e8, 0x01880899c5d03f8d }, // b=128
    .{ 0xea976cfbe633f6c9, 0x80551da9e3e32ecd }, // b=37
};

const Fnv = std.hash.Fnv1a_64;

fn digestSlices(slices: []const []const f32) u64 {
    var h = Fnv.init();
    for (slices) |s| {
        h.update(std.mem.sliceAsBytes(s));
    }

    return h.final();
}

/// Run the conv pass and the scan pass on seeded inputs with a non-zero state
/// and conv ring, and return the conv digest and the scan digest.
fn runGdnCase(
    allocator: std.mem.Allocator,
    kp: *zigferencer.ops_vulkan.GdnChunkParallel,
    t: u32,
) ![2]u64 {
    const T: usize = t;
    const qkv = try allocator.alloc(f32, T * conv_dim);
    defer allocator.free(qkv);

    const zg = try allocator.alloc(f32, T * value_dim);
    defer allocator.free(zg);

    const alpha = try allocator.alloc(f32, T * n_v);
    defer allocator.free(alpha);

    const beta = try allocator.alloc(f32, T * n_v);
    defer allocator.free(beta);

    const conv_w = try allocator.alloc(f32, @as(usize, conv_kernel) * conv_dim);
    defer allocator.free(conv_w);

    const ssm_a = try allocator.alloc(f32, n_v);
    defer allocator.free(ssm_a);

    const dt_bias = try allocator.alloc(f32, n_v);
    defer allocator.free(dt_bias);

    const ssm_norm = try allocator.alloc(f32, head_dim);
    defer allocator.free(ssm_norm);

    const state = try allocator.alloc(f32, value_dim * head_dim);
    defer allocator.free(state);

    const ring = try allocator.alloc(f32, @as(usize, conv_kernel - 1) * conv_dim);
    defer allocator.free(ring);

    const o_out = try allocator.alloc(f32, T * value_dim);
    defer allocator.free(o_out);

    var prng = std.Random.DefaultPrng.init(0x9A5EED00 + @as(u64, t));
    const r = prng.random();
    fillN(r, qkv, 1.0);
    fillN(r, zg, 1.0);
    fillN(r, alpha, 3.0);
    fillN(r, beta, 1.0);
    fillN(r, conv_w, 0.5);
    fillN(r, dt_bias, 0.3);
    fillN(r, ssm_norm, 0.2);
    for (ssm_norm) |*v| {
        v.* += 1.0;
    }

    fillN(r, state, 0.1);
    fillN(r, ring, 0.3);
    for (ssm_a) |*v| {
        v.* = -@exp((r.float(f32) - 0.5) * 2.0);
    }

    try kp.runChunkParallel(
        t,
        n_k,
        n_v,
        head_dim,
        conv_kernel,
        qkv,
        zg,
        alpha,
        beta,
        conv_w,
        ssm_a,
        dt_bias,
        ssm_norm,
        gdn_eps,
        state,
        ring,
        o_out,
    );

    // The gate buffers 3, 4, and 5 hold the conv outputs q_n, k_n, and v_conv.
    const q_n = kp.bufs[3].sliceTyped(f32, T * key_dim);
    const k_n = kp.bufs[4].sliceTyped(f32, T * key_dim);
    const v_conv = kp.bufs[5].sliceTyped(f32, T * value_dim);
    try std.testing.expect(
        allFinite(q_n) and allFinite(k_n) and allFinite(v_conv) and allFinite(ring),
    );
    try std.testing.expect(allFinite(o_out) and allFinite(state));
    return .{
        digestSlices(&.{ q_n, k_n, v_conv, ring }),
        digestSlices(&.{ o_out, state }),
    };
}

fn runGdnDigests() !void {
    const allocator = std.testing.allocator;
    const runner = try openRunner(allocator);
    defer runner.deinit();

    var kp = try zigferencer.ops_vulkan.GdnChunkParallel.init(&runner.ctx);
    defer kp.deinit();

    var got: [gdn_widths.len][2]u64 = undefined;
    for (gdn_widths, 0..) |t, i| {
        got[i] = try runGdnCase(allocator, &kp, t);
    }

    try settle("gdn_digests", gdn_widths.len, 2, gdn_digests, got, gdn_labels);
}

// -- gemm -------------------------------------------------------------------

const GemmCase = struct { b: u32, k: u32, n: u32 };

/// The tiled kernel serves widths from 64. The batch kernel serves the rest.
const gemm_min_batch: u32 = 64;

const gemm_cases = [_]GemmCase{
    .{ .b = 64, .k = 2048, .n = 2048 },
    .{ .b = 64, .k = 2048, .n = 6144 },
    .{ .b = 64, .k = 6144, .n = 2048 },
    .{ .b = 128, .k = 2048, .n = 2048 },
    .{ .b = 128, .k = 2048, .n = 6144 },
    .{ .b = 128, .k = 6144, .n = 2048 },
};
const gemm_labels = [_][]const u8{
    "b=64 k=2048 n=2048",
    "b=64 k=2048 n=6144",
    "b=64 k=6144 n=2048",
    "b=128 k=2048 n=2048",
    "b=128 k=2048 n=6144",
    "b=128 k=6144 n=2048",
};

const batch_case: GemmCase = .{ .b = 7, .k = 2048, .n = 2048 };
const batch_label = [_][]const u8{"b=7 k=2048 n=2048"};

comptime {
    for (gemm_cases) |c| {
        std.debug.assert(c.b >= gemm_min_batch);
    }

    std.debug.assert(batch_case.b < gemm_min_batch);
}

/// The digests of the tiled kernel, one per case.
const gemm_digests: [gemm_cases.len][1]?u64 = .{
    .{0x901aaa3cac86f5b1}, // b=64 k=2048 n=2048
    .{0xcee2b5fb8f80ac40}, // b=64 k=2048 n=6144
    .{0x2a35be95cd690f60}, // b=64 k=6144 n=2048
    .{0xacf9d2b579417d20}, // b=128 k=2048 n=2048
    .{0xd42d9d8c8b54e2b6}, // b=128 k=2048 n=6144
    .{0x255ba61d5d6c397c}, // b=128 k=6144 n=2048
};

/// The digest of the single-wave batch kernel at width 7.
const batch_digests: [1][1]?u64 = .{.{0x8fe90b66cf02e893}};

/// Run `kernel` on seeded activations and a seeded Q8_0 weight, and return the
/// digest of the output bytes.
fn runGemmCase(allocator: std.mem.Allocator, kernel: anytype, c: GemmCase) !u64 {
    const output_bytes: usize = (c.k / 32) * 34;
    const weight = try allocator.alloc(u8, @as(usize, c.n) * output_bytes);
    defer allocator.free(weight);

    const input = try allocator.alloc(f32, @as(usize, c.b) * c.k);
    defer allocator.free(input);

    const out = try allocator.alloc(f32, @as(usize, c.b) * c.n);
    defer allocator.free(out);

    var prng = std.Random.DefaultPrng.init(
        0x6EA77000 + @as(u64, c.b) * 100_003 + @as(u64, c.k) * 7 + c.n,
    );
    const r = prng.random();
    buildRandomQ80(r, weight, c.k, c.n);
    fillN(r, input, 1.0);
    try kernel.runOneShot(input, weight, c.b, c.k, c.n, out);
    try std.testing.expect(allFinite(out));
    return Fnv.hash(std.mem.sliceAsBytes(out));
}

fn runGemmDigests() !void {
    const allocator = std.testing.allocator;
    const runner = try openRunner(allocator);
    defer runner.deinit();

    var gemm = try zigferencer.ops_vulkan.LinearQ80Gemm.init(&runner.ctx);
    defer gemm.deinit();

    var batch = try zigferencer.ops_vulkan.LinearQ80Batch.init(&runner.ctx);
    defer batch.deinit();

    var got: [gemm_cases.len][1]u64 = undefined;
    for (gemm_cases, 0..) |c, i| {
        got[i][0] = try runGemmCase(allocator, &gemm, c);
    }

    const got_batch: [1][1]u64 = .{.{try runGemmCase(allocator, &batch, batch_case)}};
    try settle("gemm_digests", gemm_cases.len, 1, gemm_digests, got, gemm_labels);
    try settle("batch_digests", 1, 1, batch_digests, got_batch, batch_label);
}

// -- tests ------------------------------------------------------------------

test "prefill kernels keep their minted digests: GDN chunk conv and scan at three widths" {
    try runGdnDigests();
}

test "prefill kernels keep their minted digests: Q8_0 GEMM at two widths, batch kernel at width 7" {
    try runGemmDigests();
}
