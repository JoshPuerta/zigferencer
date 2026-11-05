//! Fidelity gate for the Vulkan attention kernels.
//!
//! Each kernel must stay inside a 2e-3 relative tolerance against the CPU
//! reference, the same tolerance the bench declares. The gate reads the
//! maximum relative error over the whole output, so a metric artifact cannot
//! hide a regression. The tests return `error.SkipZigTest` when no Vulkan
//! device is reachable.
//!
//! Decode. The SmolLM2 shape `(n_heads=9, n_kv_heads=3, head_dim=64,
//! t_len=128)` stays pinned on the resident kernel, with a Llama-1B and a
//! Llama-8B shape at `t_len=512`. The head geometries of the three shapes, the
//! Qwen3.5 geometry `(8, 2, 256)`, the Phi-3-mini and Phi-2 geometries
//! `(32, 32, 96)` and `(32, 32, 80)`, and a multi-query geometry run at the
//! resident depths up to `limits.attn_split_len` and at five split depths past
//! it, at both KV element types. The Phi-2 head at both KV types and the
//! Phi-3-mini head at f16 KV are not whole 128-byte lines, so they run the
//! tail loads of the split pass. The multi-query geometry `(16, 1, 128)` has
//! a group wider than one split workgroup serves. One test repeats a split row
//! to pin its bytes.
//!
//! Split digests. The needle rows of the Qwen3.5-2B, Llama-3.1-8B,
//! Qwen3.5-9B, and SmolLM2 geometries at three split depths and both KV types
//! must keep the output digests that a device run minted. So a kernel change
//! that must keep the bytes proves it on every family.
//!
//! Prefill. A chunk records through `attn_record.recordPrefill`, as a runner
//! records it: one batch run for the resident prefix, then the record groups
//! of the split rows. Four `(start_pos, b)` cases per geometry put rows inside
//! one split, across S, and past it, against the CPU reference. The byte
//! tests check that a prefill row gets the bytes of the decode row of its
//! depth, that every chunk width and the minimum record budget give the same
//! bytes, and that two runs match.
//!
//! Multi-row decode. Four KV streams of different content share one K region
//! and one V region. The multi-stream kernel and the split passes must read
//! the stream that each row names, with the q row base and the stream base
//! set apart.
//!
//! The three pinned shapes keep the trig data of the bench. Every other case
//! puts a needle in each query row. Each head gives its last causal position,
//! and the position after it, a score near `needle_score`, far above the
//! rest. A row that reads one position too few or too many therefore moves
//! its output past the tolerance. V sits in [0.5, 1.5], so no output cancels
//! toward zero, and the relative error stays well conditioned at depth.

const std = @import("std");
const z = @import("zigferencer");

// -- fixtures ---------------------------------------------------------------

const AtnShape = struct {
    label: []const u8,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    t_len: u32,
};

/// The head geometry of a shape, which the cases run at several depths.
const Geometry = struct {
    label: []const u8,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,

    fn at(self: Geometry, t_len: u32) AtnShape {
        return .{
            .label = self.label,
            .n_heads = self.n_heads,
            .n_kv_heads = self.n_kv_heads,
            .head_dim = self.head_dim,
            .t_len = t_len,
        };
    }

    /// Elements of one query row or one output row.
    fn rowElems(self: Geometry) usize {
        return @as(usize, self.n_heads) * self.head_dim;
    }

    /// Elements of one K or V position.
    fn kvElems(self: Geometry) usize {
        return @as(usize, self.n_kv_heads) * self.head_dim;
    }
};

const smollm2: Geometry = .{ .label = "smollm2", .n_heads = 9, .n_kv_heads = 3, .head_dim = 64 };
const llama1b: Geometry = .{ .label = "llama1b", .n_heads = 32, .n_kv_heads = 8, .head_dim = 64 };
const llama8b: Geometry = .{ .label = "llama8b", .n_heads = 32, .n_kv_heads = 8, .head_dim = 128 };
const qwen35: Geometry = .{ .label = "qwen35", .n_heads = 8, .n_kv_heads = 2, .head_dim = 256 };
/// The Phi-3-mini geometry. At f16 KV its head of 96 elements is not whole
/// 128-byte lines, so the split pass reads the end of each K row in its tail
/// loads.
const phi3mini: Geometry = .{
    .label = "phi3mini",
    .n_heads = 32,
    .n_kv_heads = 32,
    .head_dim = 96,
};
/// The Phi-2 geometry. Its head of 80 elements is not whole 128-byte lines at
/// either KV type, so the tail loads run at f32 KV too.
const phi2: Geometry = .{ .label = "phi2", .n_heads = 32, .n_kv_heads = 32, .head_dim = 80 };
/// A multi-query geometry. Its group of 16 heads is wider than
/// `limits.attn_split_queries`, so the split pass serves it in several
/// workgroups per KV head.
const mq16: Geometry = .{ .label = "mq16", .n_heads = 16, .n_kv_heads = 1, .head_dim = 128 };
/// A group of 9 heads. A sub-group holds 4 or 8 heads, so the last sub-group
/// of each KV head holds 1 head. It runs its tail guards and writes its
/// records past the full sub-groups.
const g9: Geometry = .{ .label = "g9", .n_heads = 18, .n_kv_heads = 2, .head_dim = 64 };

const decode_geometries = [_]Geometry{
    smollm2,
    llama1b,
    llama8b,
    qwen35,
    phi3mini,
    phi2,
    mq16,
    g9,
};

const split_len = z.limits.attn_split_len;
const cpu_tile = z.limits.attn_cpu_tile;

/// The resident depths: one position, and the two deepest rows of one split.
const resident_depths = [_]u32{ 1, split_len - 1, split_len };

/// A split row whose last split holds three positions.
const three_tail_depth = 16 * split_len + 3;

/// The deepest split row. Its CPU reference ends on a tile of three positions
/// after four full tiles.
const deep_depth = 4 * cpu_tile + 3;

/// The split depths: the first split row, the first row of three splits, a
/// row whose CPU reference ends on a tile of one position, and the two rows
/// above.
const split_depths = [_]u32{
    split_len + 1,
    2 * split_len + 1,
    cpu_tile + 1,
    three_tail_depth,
    deep_depth,
};

comptime {
    for (resident_depths) |t| {
        std.debug.assert(t <= split_len);
    }

    for (split_depths) |t| {
        std.debug.assert(t > split_len);
    }
}

/// The score that a needle position gets, near the natural log of the deepest
/// row. The needle then holds a large share of the softmax at every depth.
const needle_score: f32 = 9;

fn openRunner(allocator: std.mem.Allocator) !*z.vulkan_runner.Runner {
    return z.vulkan_runner.Runner.init(allocator) catch |e| switch (e) {
        error.NoVulkanDevice,
        error.InitializationFailed,
        error.IncompatibleDriver,
        error.FeatureNotPresent,
        => return error.SkipZigTest,
        else => return e,
    };
}

/// The q, K, and V of a shape, and the CPU reference output, at KV element
/// type `T`.
fn Case(comptime T: type) type {
    return struct {
        q: []f32,
        k: []T,
        v: []T,
        out_ref: []f32,

        const Self = @This();

        fn init(allocator: std.mem.Allocator, s: AtnShape) !Self {
            const q_len: usize = @as(usize, s.n_heads) * @as(usize, s.head_dim);
            const kv_len: usize = @as(usize, s.t_len) * @as(
                usize,
                s.n_kv_heads,
            ) * @as(usize, s.head_dim);
            const q = try allocator.alloc(f32, q_len);
            errdefer allocator.free(q);

            const k = try allocator.alloc(T, kv_len);
            errdefer allocator.free(k);

            const v = try allocator.alloc(T, kv_len);
            errdefer allocator.free(v);

            const out_ref = try allocator.alloc(f32, q_len);
            errdefer allocator.free(out_ref);

            const parts = try allocator.alloc(f32, z.attn_cpu.partialLen(s.n_heads, s.head_dim));
            defer allocator.free(parts);

            // The generator matches the bench: small-magnitude trig, so the
            // softmax never overflows and the reference path stays well
            // defined. At f16 each value rounds once to the cache type.
            for (q, 0..) |*x, i| {
                x.* = @sin(@as(f32, @floatFromInt(i)) * 0.07);
            }

            for (k, 0..) |*x, i| {
                x.* = @floatCast(@cos(@as(f32, @floatFromInt(i)) * 0.03));
            }

            for (v, 0..) |*x, i| {
                x.* = @floatCast(@sin(@as(f32, @floatFromInt(i)) * 0.05));
            }

            z.ops.attentionRowTyped(
                T,
                q,
                s.n_heads,
                s.head_dim,
                k,
                v,
                s.n_kv_heads,
                s.t_len,
                parts,
                out_ref,
            );
            return .{ .q = q, .k = k, .v = v, .out_ref = out_ref };
        }

        fn deinit(self: Self, allocator: std.mem.Allocator) void {
            allocator.free(self.q);
            allocator.free(self.k);
            allocator.free(self.v);
            allocator.free(self.out_ref);
        }
    };
}

/// The largest relative error of a row and the index where it occurs.
const Worst = struct {
    rel: f32,
    idx: usize,
};

fn worstRelative(ref: []const f32, got: []const f32) Worst {
    std.debug.assert(ref.len == got.len);
    std.debug.assert(ref.len >= 1);

    // The 1e-5 floor on the denominator matches `computeErrors` in
    // tools/bench.zig. The floor stops a near-zero cancellation output from
    // dominating the relative error.
    const abs_floor: f32 = 1e-5;
    var worst: Worst = .{ .rel = 0, .idx = 0 };
    for (ref, got, 0..) |r, g, i| {
        const denom = if (@abs(r) < abs_floor) abs_floor else @abs(r);
        const rel = @abs(r - g) / denom;
        // A NaN fails every comparison, so it must take the worst slot itself.
        if (rel > worst.rel or std.math.isNan(rel)) {
            worst = .{ .rel = rel, .idx = i };
        }

        if (std.math.isNan(rel)) break;
    }

    return worst;
}

fn runAttentionCheck(
    comptime T: type,
    allocator: std.mem.Allocator,
    attn: *z.ops_vulkan.Attention,
    s: AtnShape,
    tol: f32,
) !void {
    const case: Case(T) = try .init(allocator, s);
    defer case.deinit(allocator);

    const out_gpu = try allocator.alloc(f32, case.out_ref.len);
    defer allocator.free(out_gpu);

    try attn.runTyped(
        T,
        case.q,
        case.k,
        case.v,
        s.n_heads,
        s.n_kv_heads,
        s.head_dim,
        s.t_len,
        out_gpu,
    );

    const worst = worstRelative(case.out_ref, out_gpu);
    if (!(worst.rel <= tol)) {
        const h: usize = worst.idx / s.head_dim;
        const d_in_head: usize = worst.idx % s.head_dim;
        std.debug.print(
            "{s} kv={s} t_len={d}: max_rel={e:.3} > tol={e:.3} at out[{d}] (h={d}, d={d}) " ++
                "ref={d} gpu={d}\n",
            .{ s.label, @typeName(T), s.t_len, worst.rel, tol, worst.idx, h, d_in_head, case.out_ref[worst.idx], out_gpu[worst.idx] },
        );
        return error.GpuAttentionExceedsTolerance;
    }
}

/// Run one decode row of `t_len` positions on the needle data at KV element
/// type `T`, and return the digest of its output bytes. The caches hold one
/// position past the row, so a row that reads too far meets the needle partner.
fn runNeedleRow(
    comptime T: type,
    allocator: std.mem.Allocator,
    attn: *z.ops_vulkan.Attention,
    g: Geometry,
    t_len: u32,
) !u64 {
    const kv_len = (@as(usize, t_len) + 1) * g.kvElems();
    const k = try allocator.alloc(f32, kv_len);
    defer allocator.free(k);

    const v = try allocator.alloc(f32, kv_len);
    defer allocator.free(v);

    const k_typed = try allocator.alloc(T, kv_len);
    defer allocator.free(k_typed);

    const v_typed = try allocator.alloc(T, kv_len);
    defer allocator.free(v_typed);

    const q = try allocator.alloc(f32, g.rowElems());
    defer allocator.free(q);

    const ref = try allocator.alloc(f32, g.rowElems());
    defer allocator.free(ref);

    const got = try allocator.alloc(f32, g.rowElems());
    defer allocator.free(got);

    const parts = try allocator.alloc(f32, z.attn_cpu.partialLen(g.n_heads, g.head_dim));
    defer allocator.free(parts);

    fillKv(k, v, 1);
    for (k_typed, k) |*d, x| {
        d.* = @floatCast(x);
    }

    for (v_typed, v) |*d, x| {
        d.* = @floatCast(x);
    }

    fillNeedleRow(q, g, k, t_len - 1, 3);
    z.ops.attentionRowTyped(f32, q, g.n_heads, g.head_dim, k, v, g.n_kv_heads, t_len, parts, ref);
    try attn.runTyped(T, q, k_typed, v_typed, g.n_heads, g.n_kv_heads, g.head_dim, t_len, got);
    try expectRows(
        ref,
        got,
        g,
        0,
        1,
        "decode {s} kv={s} t_len {d}",
        .{ g.label, @typeName(T), t_len },
    );
    return std.hash.Fnv1a_64.hash(std.mem.sliceAsBytes(got));
}

/// Run every decode geometry at every resident and split depth at KV element
/// type `T`. The needle data keeps every output far from zero, so the
/// relative error stays well conditioned at the deepest row.
fn runDecodeChecks(comptime T: type, kv: z.dtype.KvDtype) !void {
    const allocator = std.testing.allocator;
    const runner = try openRunner(allocator);
    defer runner.deinit();

    var attn = try z.ops_vulkan.Attention.initKv(&runner.ctx, kv);
    defer attn.deinit();

    for (decode_geometries) |geometry| {
        for (resident_depths ++ split_depths) |t_len| {
            try std.testing.expectEqual(t_len > split_len, z.attn_split.usesSplit(t_len));
            _ = try runNeedleRow(T, allocator, &attn, geometry, t_len);
        }
    }
}

/// Run one split row twice through one kernel. The merge sums the splits in a
/// fixed order, so the two outputs must match byte for byte.
fn repeatSplitRow(comptime T: type, kv: z.dtype.KvDtype) !void {
    const allocator = std.testing.allocator;
    const runner = try openRunner(allocator);
    defer runner.deinit();

    var attn = try z.ops_vulkan.Attention.initKv(&runner.ctx, kv);
    defer attn.deinit();

    const s = qwen35.at(three_tail_depth);
    const case: Case(T) = try .init(allocator, s);
    defer case.deinit(allocator);

    const first = try allocator.alloc(f32, case.out_ref.len);
    defer allocator.free(first);

    const second = try allocator.alloc(f32, case.out_ref.len);
    defer allocator.free(second);

    try attn.runTyped(
        T,
        case.q,
        case.k,
        case.v,
        s.n_heads,
        s.n_kv_heads,
        s.head_dim,
        s.t_len,
        first,
    );
    try attn.runTyped(
        T,
        case.q,
        case.k,
        case.v,
        s.n_heads,
        s.n_kv_heads,
        s.head_dim,
        s.t_len,
        second,
    );
    try std.testing.expectEqualSlices(
        u8,
        std.mem.sliceAsBytes(first),
        std.mem.sliceAsBytes(second),
    );
}

// -- split digests ----------------------------------------------------------

const qwen9b: Geometry = .{ .label = "qwen9b", .n_heads = 16, .n_kv_heads = 4, .head_dim = 256 };

/// The head geometries of the four registry families: Qwen3.5-2B,
/// Llama-3.1-8B, Qwen3.5-9B, and SmolLM2.
const digest_geometries = [_]Geometry{ qwen35, llama8b, qwen9b, smollm2 };
const digest_kv = [_]z.dtype.KvDtype{ .f32, .f16 };

/// The first split row, a row one past eight splits, and a row three past 32
/// splits.
const digest_depths = [_]u32{ split_len + 1, 8 * split_len + 1, 32 * split_len + 3 };

comptime {
    std.debug.assert(digest_depths[0] == 257);
    std.debug.assert(digest_depths[1] == 2049);
    std.debug.assert(digest_depths[2] == 8195);
}

/// The FNV-1a digest of the output bytes of each needle row, indexed as
/// `[geometry][kv][depth]`. A device run mints them. A null entry fails the
/// test and prints the table to paste here. A kernel change that keeps the
/// summation order keeps every entry.
const split_digests: [digest_geometries.len][digest_kv.len][digest_depths.len]?u64 = .{
    .{
        .{ 0x71255138392cf8bb, 0xc904c48eb772eec5, 0x1cdfc57b25c96a62 },
        .{ 0x71255138392cf8bb, 0xc904c48eb772eec5, 0x1cdfc57b25c96a62 },
    }, // qwen35
    .{
        .{ 0xe3ee5f3160fffb20, 0x28fdbd15847b12a1, 0xa12cf72c4d20d371 },
        .{ 0xe3ee5f3160fffb20, 0x28fdbd15847b12a1, 0xa12cf72c4d20d371 },
    }, // llama8b
    .{
        .{ 0xa6a7ee28a0c051fc, 0x4057b1059c455e85, 0x40727de9ec4c932e },
        .{ 0xa6a7ee28a0c051fc, 0x4057b1059c455e85, 0x40727de9ec4c932e },
    }, // qwen9b
    .{
        .{ 0x734484fc10402e75, 0xc66d58e793564960, 0x263f809aa3e348e2 },
        .{ 0x734484fc10402e75, 0xc66d58e793564960, 0x263f809aa3e348e2 },
    }, // smollm2
};

/// Digest the split output of every geometry, KV type, and depth, and compare
/// each digest with the table. The kit of the build must sum in the
/// `.split_ascending` order, because only that order promises fixed bytes.
fn runSplitDigests() !void {
    if (z.ops_vulkan.AttnKit.order != .split_ascending) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const runner = try openRunner(allocator);
    defer runner.deinit();

    var got: [digest_geometries.len][digest_kv.len][digest_depths.len]u64 = undefined;
    for (digest_kv, 0..) |kv, ki| {
        var attn = try z.ops_vulkan.Attention.initKv(&runner.ctx, kv);
        defer attn.deinit();

        for (digest_geometries, 0..) |g, gi| {
            for (digest_depths, 0..) |t_len, di| {
                std.debug.assert(z.attn_split.usesSplit(t_len));

                got[gi][ki][di] = switch (kv) {
                    .f32 => try runNeedleRow(f32, allocator, &attn, g, t_len),
                    .f16 => try runNeedleRow(f16, allocator, &attn, g, t_len),
                };
            }
        }
    }

    var missing: u32 = 0;
    var moved: u32 = 0;
    for (digest_geometries, 0..) |g, gi| {
        for (digest_kv, 0..) |kv, ki| {
            for (digest_depths, 0..) |t_len, di| {
                const want = split_digests[gi][ki][di] orelse {
                    missing += 1;
                    continue;
                };
                if (want != got[gi][ki][di]) {
                    moved += 1;
                    std.debug.print(
                        "split digest {s} kv={t} t_len {d}: got 0x{x:0>16}, want 0x{x:0>16}\n",
                        .{ g.label, kv, t_len, got[gi][ki][di], want },
                    );
                }
            }
        }
    }

    if (missing == 0 and moved == 0) return;

    std.debug.print(
        "mint: {d} split digests missing, {d} moved. The table of this run:\n",
        .{ missing, moved },
    );
    std.debug.print(
        "const split_digests: [digest_geometries.len][digest_kv.len][digest_depths.len]?u64 " ++
            "= .{{\n",
        .{},
    );
    for (digest_geometries, 0..) |g, gi| {
        std.debug.print("    .{{", .{});
        for (0..digest_kv.len) |ki| {
            const d = got[gi][ki];
            std.debug.print(
                " .{{ 0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16} }}{s}",
                .{ d[0], d[1], d[2], if (ki + 1 < digest_kv.len) "," else "" },
            );
        }

        std.debug.print(" }}, // {s}\n", .{g.label});
    }

    std.debug.print("}};\n", .{});
    return if (moved > 0) error.SplitDigestMoved else error.SplitDigestNotMinted;
}

// -- record-only harness ----------------------------------------------------

const Context = z.vulkan_ctx.Context;
const HostBuffer = z.ops_vulkan.HostBuffer;
const OneShot = z.ops_vulkan.OneShot;
const CommandBuffer = @FieldType(OneShot, "cmd_buffer");
/// The test module imports no Vulkan binding, so it reads the descriptor
/// type from the binding helper.
const DescriptorBufferInfo = @typeInfo(
    @typeInfo(@TypeOf(z.ops_vulkan.updateStorageBindings)).@"fn".params[2].type.?,
).pointer.child;

/// A stateless generator: value `i` of sequence `seed`, uniform in [-1, 1).
/// A value depends on its index alone, so a stream fills the same at any
/// window.
fn unit(seed: u64, i: u64) f32 {
    // The SplitMix64 finalizer over a mix of the two keys.
    var x: u64 = seed *% 0x9E3779B97F4A7C15 +% i *% 0xBF58476D1CE4E5B9;
    x = (x ^ (x >> 30)) *% 0xBF58476D1CE4E5B9;
    x = (x ^ (x >> 27)) *% 0x94D049BB133111EB;
    x ^= x >> 31;
    // 24 bits fit an f32 significand, so the division is exact.
    const r = @as(f32, @floatFromInt(x >> 40)) / (1 << 23) - 1;
    std.debug.assert(r >= -1);
    std.debug.assert(r < 1);
    return r;
}

/// Fill K uniform in [-1, 1) and V uniform in [0.5, 1.5). Each value is exact
/// in f16, so one f32 reference serves both KV element types.
fn fillKv(k: []f32, v: []f32, seed: u64) void {
    std.debug.assert(k.len == v.len);

    for (k, v, 0..) |*kx, *vx, i| {
        kx.* = @as(f16, @floatCast(unit(seed, i)));
        vx.* = @as(f16, @floatCast(1 + 0.5 * unit(seed + 1, i)));
    }
}

/// Fill one query row so that each head scores K positions `p` and `p + 1` of
/// `k` near `needle_score`. `k` holds positions in the layout
/// `[position][kv head][head_dim]`.
fn fillNeedleRow(q_row: []f32, g: Geometry, k: []const f32, p: u32, seed: u64) void {
    std.debug.assert(q_row.len == g.rowElems());
    std.debug.assert((@as(usize, p) + 2) * g.kvElems() <= k.len);

    // A K value uniform in [-1, 1) has a mean square of 1/3. The score of
    // position p is then near c * head_dim / 3 / sqrt(head_dim), and this c
    // sets that score to needle_score.
    const hd: f32 = @floatFromInt(g.head_dim);
    const c = 3 * needle_score / @sqrt(hd);
    const group = g.n_heads / g.n_kv_heads;
    const kv_row = g.kvElems();
    for (0..g.n_heads) |h| {
        const kv_h = h / group;
        const at_p = k[p * kv_row + kv_h * g.head_dim ..][0..g.head_dim];
        const at_next = k[(p + 1) * kv_row + kv_h * g.head_dim ..][0..g.head_dim];
        const q_h = q_row[h * g.head_dim ..][0..g.head_dim];
        for (q_h, at_p, at_next, 0..) |*x, a, b, d| {
            x.* = c * (a + b + 0.25 * unit(seed, h * g.head_dim + d));
        }
    }
}

/// Copy `src` into `buf` as element type `T`, and return the bytes that the
/// binding covers.
fn upload(comptime T: type, buf: *HostBuffer, src: []const f32) !u64 {
    std.debug.assert(src.len >= 1);

    // The f16 shaders read the caches as 32-bit words, so each binding covers
    // whole words.
    const bytes = std.mem.alignForward(u64, src.len * @sizeOf(T), 4);
    try buf.ensure(bytes);
    const dst = buf.sliceTyped(T, src.len);
    if (T == f32) {
        @memcpy(dst, src);
    } else {
        for (dst, src) |*d, s| {
            d.* = @floatCast(s);
        }
    }

    std.debug.assert(bytes <= buf.capacity);
    return bytes;
}

fn begin(ctx: *const Context, cb: CommandBuffer) !void {
    try ctx.vkd.resetCommandBuffer(cb, .{});
    try ctx.vkd.beginCommandBuffer(cb, &.{ .flags = .{ .one_time_submit_bit = true } });
}

fn submitAndWait(ctx: *const Context, cb: CommandBuffer) !void {
    try ctx.vkd.endCommandBuffer(cb);
    const cmds = [_]CommandBuffer{cb};
    try ctx.vkd.queueSubmit(
        ctx.queue,
        &.{.{ .command_buffer_count = 1, .p_command_buffers = &cmds }},
        .null_handle,
    );
    try ctx.vkd.queueWaitIdle(ctx.queue);
}

/// A shader read waits for the shader writes before it.
fn barrierShader(ctx: *const Context, cb: CommandBuffer) void {
    ctx.vkd.cmdPipelineBarrier(
        cb,
        .{ .compute_shader_bit = true },
        .{ .compute_shader_bit = true },
        .{},
        &.{
            .{ .src_access_mask = .{ .shader_write_bit = true }, .dst_access_mask = .{ .shader_read_bit = true } },
        },
        null,
        null,
    );
}

/// Compare each row of `got` against `ref` and name the first row past 2e-3.
/// The message counts rows from `first_row`.
fn expectRows(
    ref: []const f32,
    got: []const f32,
    g: Geometry,
    first_row: u32,
    rows: u32,
    comptime fmt: []const u8,
    args: anytype,
) !void {
    const row_elems = g.rowElems();
    std.debug.assert(ref.len == rows * row_elems);
    std.debug.assert(got.len == ref.len);
    for (0..rows) |r| {
        const lo = r * row_elems;
        const worst = worstRelative(ref[lo..][0..row_elems], got[lo..][0..row_elems]);
        if (!(worst.rel <= 2e-3)) {
            std.debug.print(fmt, args);
            std.debug.print(
                " row {d}: max_rel={e:.3} > 2e-3 at h={d} d={d}, ref={d} gpu={d}\n",
                .{ first_row + r, worst.rel, worst.idx / g.head_dim, worst.idx % g.head_dim, ref[lo + worst.idx], got[lo + worst.idx] },
            );
            return error.GpuAttentionExceedsTolerance;
        }
    }
}

// -- prefill fixtures -------------------------------------------------------

/// One prefill chunk: `b` rows from position `start_pos`. Row i attends to
/// `start_pos + i + 1` positions.
const ChunkCase = struct {
    label: []const u8,
    start_pos: u32,
    b: u32,
};

/// The oracle cases: a chunk inside one split, a chunk that straddles S, and
/// two chunks past it at an odd row count.
const chunk_cases = [_]ChunkCase{
    .{ .label = "resident", .start_pos = 0, .b = 64 },
    .{ .label = "straddle", .start_pos = split_len - 3, .b = 7 },
    .{ .label = "deep", .start_pos = 2046, .b = 7 },
    .{ .label = "deeper", .start_pos = 4 * split_len + 1, .b = 5 },
};

const chunk_geometries = [_]Geometry{ smollm2, llama8b, qwen35 };

/// The q rows, the f16-exact K and V, and the CPU reference rows of `b` rows
/// from `start_pos`. K and V hold one position past the last row, so a row
/// that reads too far meets a needle.
const ChunkData = struct {
    q: []f32,
    k: []f32,
    v: []f32,
    ref: []f32,
    positions: u32,

    fn init(allocator: std.mem.Allocator, g: Geometry, start_pos: u32, b: u32) !ChunkData {
        const positions: u32 = start_pos + b + 1;
        const q = try allocator.alloc(f32, b * g.rowElems());
        errdefer allocator.free(q);

        const k = try allocator.alloc(f32, positions * g.kvElems());
        errdefer allocator.free(k);

        const v = try allocator.alloc(f32, positions * g.kvElems());
        errdefer allocator.free(v);

        const ref = try allocator.alloc(f32, b * g.rowElems());
        errdefer allocator.free(ref);

        const parts = try allocator.alloc(f32, z.attn_cpu.partialLen(g.n_heads, g.head_dim));
        defer allocator.free(parts);

        fillKv(k, v, 1);
        for (0..b) |bi| {
            const row: u32 = @intCast(bi);
            const q_row = q[bi * g.rowElems() ..][0..g.rowElems()];
            fillNeedleRow(q_row, g, k, start_pos + row, 3 + bi);
            const t_len: usize = @as(usize, start_pos) + bi + 1;
            z.ops.attentionRowTyped(
                f32,
                q_row,
                g.n_heads,
                g.head_dim,
                k,
                v,
                g.n_kv_heads,
                t_len,
                parts,
                ref[bi * g.rowElems() ..][0..g.rowElems()],
            );
        }

        return .{ .q = q, .k = k, .v = v, .ref = ref, .positions = positions };
    }

    fn deinit(self: ChunkData, allocator: std.mem.Allocator) void {
        allocator.free(self.q);
        allocator.free(self.k);
        allocator.free(self.v);
        allocator.free(self.ref);
    }
};

/// Every attention pipeline of one KV element type behind the attention kit,
/// as a runner holds them, with the sets that a prefill chunk binds and the
/// host buffers they bind. K and V hold one stream, the slot 0 of a runner.
const PrefillRig = struct {
    ctx: *const Context,
    single: z.ops_vulkan.Attention,
    multi: z.ops_vulkan.AttentionMultiStream,
    split: z.ops_vulkan.AttentionSplit,
    batch: z.ops_vulkan.AttentionBatch,
    batch_set: OneShot,
    split_set: OneShot,
    merge_set: OneShot,
    q: HostBuffer,
    k: HostBuffer,
    v: HostBuffer,
    scratch: HostBuffer,
    out: HostBuffer,

    fn init(ctx: *const Context, kv: z.dtype.KvDtype) !PrefillRig {
        var single = try z.ops_vulkan.Attention.initKv(ctx, kv);
        errdefer single.deinit();

        var multi = try z.ops_vulkan.AttentionMultiStream.initKv(ctx, kv);
        errdefer multi.deinit();

        var split = try z.ops_vulkan.AttentionSplit.initKv(ctx, kv);
        errdefer split.deinit();

        var batch = try z.ops_vulkan.AttentionBatch.initKv(ctx, kv);
        errdefer batch.deinit();

        const batch_set = try OneShot.init(ctx, batch.pipeline.set_layout, 4);
        errdefer batch_set.deinit(ctx);

        const split_set = try OneShot.init(
            ctx,
            split.split.set_layout,
            z.ops_vulkan.AttentionSplit.split_bindings,
        );
        errdefer split_set.deinit(ctx);

        const merge_set = try OneShot.init(
            ctx,
            split.merge.set_layout,
            z.ops_vulkan.AttentionSplit.merge_bindings,
        );
        return .{
            .ctx = ctx,
            .single = single,
            .multi = multi,
            .split = split,
            .batch = batch,
            .batch_set = batch_set,
            .split_set = split_set,
            .merge_set = merge_set,
            .q = .{ .ctx = ctx },
            .k = .{ .ctx = ctx },
            .v = .{ .ctx = ctx },
            .scratch = .{ .ctx = ctx },
            .out = .{ .ctx = ctx },
        };
    }

    fn deinit(self: *PrefillRig) void {
        self.out.deinit();
        self.scratch.deinit();
        self.v.deinit();
        self.k.deinit();
        self.q.deinit();
        self.merge_set.deinit(self.ctx);
        self.split_set.deinit(self.ctx);
        self.batch_set.deinit(self.ctx);
        self.batch.deinit();
        self.split.deinit();
        self.multi.deinit();
        self.single.deinit();
    }

    /// Upload K and V of `data` at KV element type `T`.
    fn uploadKv(self: *PrefillRig, comptime T: type, data: ChunkData) !u64 {
        const kv_bytes = try upload(T, &self.k, data.k);
        _ = try upload(T, &self.v, data.v);
        return kv_bytes;
    }

    /// Record the rows `[first, first + b)` of `data` as one prefill chunk
    /// through the recorder, with `plan` over those rows, and copy the output
    /// rows into `out`. The q rows of the chunk are rows `first..` of `data`.
    fn runChunk(
        self: *PrefillRig,
        g: Geometry,
        data: ChunkData,
        kv_bytes: u64,
        start_pos: u32,
        first: u32,
        plan: *const z.attn_split.AttnPlan,
        out: []f32,
    ) !void {
        const b = plan.resident_count + plan.split_count;
        std.debug.assert(out.len == b * g.rowElems());
        const q_rows = data.q[first * g.rowElems() ..][0 .. b * g.rowElems()];
        const q_bytes = try upload(f32, &self.q, q_rows);
        const out_bytes: u64 = out.len * @sizeOf(f32);
        try self.out.ensure(out_bytes);
        // A row that no dispatch writes keeps the NaN sentinel and fails the
        // check, where a stale row of the run before can pass it.
        @memset(self.out.sliceTyped(u32, out.len), sentinel_bits);
        const scratch = z.attn_split.Scratch.init(g.n_heads, g.head_dim, data.positions);
        try self.scratch.ensure(scratch.totalBytes());
        const q_info: DescriptorBufferInfo = .{
            .buffer = self.q.buffer,
            .offset = 0,
            .range = q_bytes,
        };
        const k_info: DescriptorBufferInfo = .{
            .buffer = self.k.buffer,
            .offset = 0,
            .range = kv_bytes,
        };
        const v_info: DescriptorBufferInfo = .{
            .buffer = self.v.buffer,
            .offset = 0,
            .range = kv_bytes,
        };
        const scratch_info: DescriptorBufferInfo = .{
            .buffer = self.scratch.buffer,
            .offset = 0,
            .range = scratch.totalBytes(),
        };
        const out_info: DescriptorBufferInfo = .{
            .buffer = self.out.buffer,
            .offset = 0,
            .range = out_bytes,
        };
        z.ops_vulkan.AttentionSplit.assertKBinding(k_info);
        z.ops_vulkan.AttentionSplit.assertVBinding(v_info);
        z.ops_vulkan.updateStorageBindings(
            self.ctx,
            self.batch_set.desc_set,
            &.{ q_info, k_info, v_info, out_info },
        );
        z.ops_vulkan.updateStorageBindings(
            self.ctx,
            self.split_set.desc_set,
            &.{ q_info, k_info, v_info, scratch_info },
        );
        z.ops_vulkan.updateStorageBindings(
            self.ctx,
            self.merge_set.desc_set,
            &.{ scratch_info, out_info },
        );
        const kit: z.ops_vulkan.AttnKit = .{
            .single = &self.single,
            .multi = &self.multi,
            .passes = &self.split,
            .batch = &self.batch,
        };
        const binding: z.ops_vulkan.AttnKit.Binding = .{
            .shape = .{
                .n_heads = g.n_heads,
                .n_kv_heads = g.n_kv_heads,
                .head_dim = g.head_dim,
                .stream_stride_elems = @intCast(data.positions * g.kvElems()),
                .max_splits = scratch.max_splits,
            },
            .resident = .{ .batch = self.batch_set.desc_set },
            .split = self.split_set.desc_set,
            .merge = self.merge_set.desc_set,
        };
        const cb = self.batch_set.cmd_buffer;
        try begin(self.ctx, cb);
        z.attn_record.recordPrefill(z.ops_vulkan.AttnKit, kit, cb, plan, binding, start_pos);
        try submitAndWait(self.ctx, cb);
        @memcpy(out, self.out.sliceTyped(f32, out.len));
    }

    /// Run the rows of `data` from `start_pos` in chunks of `width` rows at
    /// the full record budget, and copy every row into `out`.
    fn runChunked(
        self: *PrefillRig,
        comptime T: type,
        g: Geometry,
        data: ChunkData,
        start_pos: u32,
        b: u32,
        width: u32,
        out: []f32,
    ) !void {
        std.debug.assert(width >= 1);

        const kv_bytes = try self.uploadKv(T, data);
        const max_splits = z.attn_split.splitCount(data.positions);
        var first: u32 = 0;
        // Each chunk holds at least one row, so the loop runs at most b times.
        while (first < b) {
            const n = @min(width, b - first);
            const plan = z.attn_split.planPrefill(0, start_pos + first, n, max_splits);
            try self.runChunk(
                g,
                data,
                kv_bytes,
                start_pos + first,
                first,
                &plan,
                out[first * g.rowElems() ..][0 .. n * g.rowElems()],
            );
            first += n;
        }
    }
};

/// Run the decode row of each row of `data`: the row at position
/// `start_pos + i` through `Attention.runTyped`, the path of a decode step.
fn decodeRows(
    comptime T: type,
    allocator: std.mem.Allocator,
    attn: *z.ops_vulkan.Attention,
    g: Geometry,
    data: ChunkData,
    start_pos: u32,
    b: u32,
    out: []f32,
) !void {
    const k_typed = try allocator.alloc(T, data.k.len);
    defer allocator.free(k_typed);

    const v_typed = try allocator.alloc(T, data.v.len);
    defer allocator.free(v_typed);

    for (k_typed, data.k) |*d, x| {
        d.* = @floatCast(x);
    }

    for (v_typed, data.v) |*d, x| {
        d.* = @floatCast(x);
    }

    for (0..b) |i| {
        const q_row = data.q[i * g.rowElems() ..][0..g.rowElems()];
        const t_len = start_pos + @as(u32, @intCast(i)) + 1;
        try attn.runTyped(
            T,
            q_row,
            k_typed,
            v_typed,
            g.n_heads,
            g.n_kv_heads,
            g.head_dim,
            t_len,
            out[i * g.rowElems() ..][0..g.rowElems()],
        );
    }
}

fn expectSameBytes(
    want: []const f32,
    got: []const f32,
    comptime fmt: []const u8,
    args: anytype,
) !void {
    if (std.mem.eql(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got))) return;

    var first: usize = 0;
    while (@as(u32, @bitCast(want[first])) == @as(u32, @bitCast(got[first]))) {
        first += 1;
    }

    std.debug.print(
        fmt ++ ": first differing element {d}: {d} vs {d}\n",
        args ++ .{ first, want[first], got[first] },
    );
    return error.GpuAttentionBytesDiffer;
}

fn PrefillHarness(comptime T: type) type {
    return struct {
        const kv: z.dtype.KvDtype = if (T == f32) .f32 else .f16;

        /// Prefill chunks across S against the CPU reference, at the
        /// declared tolerance.
        fn oracle() !void {
            const allocator = std.testing.allocator;
            const runner = try openRunner(allocator);
            defer runner.deinit();

            var rig: PrefillRig = try .init(&runner.ctx, kv);
            defer rig.deinit();

            for (chunk_geometries) |g| {
                for (chunk_cases) |c| {
                    const data: ChunkData = try .init(allocator, g, c.start_pos, c.b);
                    defer data.deinit(allocator);

                    const out = try allocator.alloc(f32, data.ref.len);
                    defer allocator.free(out);

                    try rig.runChunked(T, g, data, c.start_pos, c.b, c.b, out);
                    try expectRows(
                        data.ref,
                        out,
                        g,
                        0,
                        c.b,
                        "prefill {s} {s} (start_pos {d}) kv={s}",
                        .{ g.label, c.label, c.start_pos, @typeName(T) },
                    );
                }
            }
        }

        /// (a) and (d): a prefill row gets the bytes of the decode row of the
        /// same depth. A resident row compares the batch kernel with
        /// `attention.comp`, and a split row compares the record groups of a
        /// chunk with the one-row plan of a decode step.
        fn equalsDecode(starts: []const u32, b: u32) !void {
            const allocator = std.testing.allocator;
            const runner = try openRunner(allocator);
            defer runner.deinit();

            var rig: PrefillRig = try .init(&runner.ctx, kv);
            defer rig.deinit();

            var attn = try z.ops_vulkan.Attention.initKv(&runner.ctx, kv);
            defer attn.deinit();

            for ([_]Geometry{ smollm2, qwen35 }) |g| {
                for (starts) |start_pos| {
                    const data: ChunkData = try .init(allocator, g, start_pos, b);
                    defer data.deinit(allocator);

                    const prefill = try allocator.alloc(f32, data.ref.len);
                    defer allocator.free(prefill);

                    const decode = try allocator.alloc(f32, data.ref.len);
                    defer allocator.free(decode);

                    try rig.runChunked(T, g, data, start_pos, b, b, prefill);
                    try decodeRows(T, allocator, &attn, g, data, start_pos, b, decode);
                    try expectSameBytes(
                        decode,
                        prefill,
                        "prefill {s} kv={s} start_pos {d} b {d} vs decode",
                        .{ g.label, @typeName(T), start_pos, b },
                    );
                }
            }
        }

        /// (b): chunk widths 1, 2, 3, 7, and b give the same bytes for the
        /// same rows.
        fn widths(starts: []const u32, b: u32) !void {
            const allocator = std.testing.allocator;
            const runner = try openRunner(allocator);
            defer runner.deinit();

            var rig: PrefillRig = try .init(&runner.ctx, kv);
            defer rig.deinit();

            for ([_]Geometry{ smollm2, qwen35 }) |g| {
                for (starts) |start_pos| {
                    const data: ChunkData = try .init(allocator, g, start_pos, b);
                    defer data.deinit(allocator);

                    const whole = try allocator.alloc(f32, data.ref.len);
                    defer allocator.free(whole);

                    const part = try allocator.alloc(f32, data.ref.len);
                    defer allocator.free(part);

                    try rig.runChunked(T, g, data, start_pos, b, b, whole);
                    for ([_]u32{ 1, 2, 3, 7 }) |width| {
                        try rig.runChunked(T, g, data, start_pos, b, width, part);
                        try expectSameBytes(
                            whole,
                            part,
                            "prefill {s} kv={s} start_pos {d} width {d} vs width {d}",
                            .{ g.label, @typeName(T), start_pos, width, b },
                        );
                    }
                }
            }
        }

        /// (c): the minimum record budget, one row per group, gives the bytes
        /// of the full budget.
        fn minimumBudget(start_pos: u32, b: u32) !void {
            const allocator = std.testing.allocator;
            const runner = try openRunner(allocator);
            defer runner.deinit();

            var rig: PrefillRig = try .init(&runner.ctx, kv);
            defer rig.deinit();

            for ([_]Geometry{ smollm2, qwen35 }) |g| {
                const data: ChunkData = try .init(allocator, g, start_pos, b);
                defer data.deinit(allocator);

                const full = try allocator.alloc(f32, data.ref.len);
                defer allocator.free(full);

                const single = try allocator.alloc(f32, data.ref.len);
                defer allocator.free(single);

                try rig.runChunked(T, g, data, start_pos, b, b, full);
                // The budget of the deepest row alone holds one row per group.
                var sites: [z.attn_split.max_plan_rows]z.attn_split.RowSite = undefined;
                for (sites[0..b], 0..) |*site, i| {
                    site.* = .{ .slot = 0, .t_len = start_pos + @as(u32, @intCast(i)) + 1 };
                }

                const plan = z.attn_split.planRowsBudget(
                    sites[0..b],
                    z.attn_split.splitCount(start_pos + b),
                );
                try std.testing.expectEqual(plan.split_count, plan.group_count);
                const kv_bytes = try rig.uploadKv(T, data);
                try rig.runChunk(g, data, kv_bytes, start_pos, 0, &plan, single);
                try expectSameBytes(
                    full,
                    single,
                    "prefill {s} kv={s} start_pos {d}: one row per group vs the full budget",
                    .{ g.label, @typeName(T), start_pos },
                );
            }
        }

        /// The deepest chunk twice through one rig. Each row sums in a fixed
        /// order, so the two outputs must match byte for byte.
        fn repeat() !void {
            const allocator = std.testing.allocator;
            const runner = try openRunner(allocator);
            defer runner.deinit();

            var rig: PrefillRig = try .init(&runner.ctx, kv);
            defer rig.deinit();

            const c = chunk_cases[chunk_cases.len - 1];
            const data: ChunkData = try .init(allocator, qwen35, c.start_pos, c.b);
            defer data.deinit(allocator);

            const first = try allocator.alloc(f32, data.ref.len);
            defer allocator.free(first);

            const second = try allocator.alloc(f32, data.ref.len);
            defer allocator.free(second);

            try rig.runChunked(T, qwen35, data, c.start_pos, c.b, c.b, first);
            try rig.runChunked(T, qwen35, data, c.start_pos, c.b, c.b, second);
            try expectSameBytes(first, second, "prefill qwen35 kv={s}: two runs", .{@typeName(T)});
        }
    };
}

// -- multi-stream fixtures --------------------------------------------------

const n_streams = 4;

/// The positions of one stream. One spare position past the deepest row holds
/// the needle partner, and the window rounds up to whole splits.
const stream_window: u32 = std.mem.alignForward(u32, deep_depth + 1, split_len);

// One group of split rows, one row per stream, fills every scratch row.
comptime {
    std.debug.assert(n_streams == z.limits.attn_split_rows);
}

/// One row of a multi-row dispatch: query row `q_row` reads stream `stream`
/// and writes output row `out_row`.
const Pair = struct {
    q_row: u32,
    stream: u32,
    out_row: u32,
};

/// Output rows of a multi-row dispatch: four rows that read their own stream,
/// plus one row of the split test.
const out_rows = n_streams + 1;

/// The NaN bits that an output row holds before a dispatch. A multi-stream row
/// that no pair names must still hold them after the dispatch.
const sentinel_bits: u32 = 0x7FC0_0BAD;

/// The K and V regions of `n_streams` streams of different content, at the
/// layout `[stream][position][kv head][head_dim]`.
const Streams = struct {
    k: []f32,
    v: []f32,
    stride: u32,

    fn init(allocator: std.mem.Allocator, g: Geometry) !Streams {
        const stride_elems = @as(usize, stream_window) * g.kvElems();
        // The shaders index a region in 32 bits.
        std.debug.assert(n_streams * stride_elems <= z.limits.max_kv_index_elems);
        const k = try allocator.alloc(f32, n_streams * stride_elems);
        errdefer allocator.free(k);

        const v = try allocator.alloc(f32, n_streams * stride_elems);
        errdefer allocator.free(v);

        for (0..n_streams) |s| {
            const lo = s * stride_elems;
            fillKv(k[lo..][0..stride_elems], v[lo..][0..stride_elems], 100 + 2 * s);
        }

        return .{ .k = k, .v = v, .stride = @intCast(stride_elems) };
    }

    fn deinit(self: Streams, allocator: std.mem.Allocator) void {
        allocator.free(self.k);
        allocator.free(self.v);
    }

    fn kOf(self: Streams, s: u32) []const f32 {
        return self.k[@as(usize, s) * self.stride ..][0..self.stride];
    }

    fn vOf(self: Streams, s: u32) []const f32 {
        return self.v[@as(usize, s) * self.stride ..][0..self.stride];
    }
};

/// The multi-stream kernel and the split passes of one KV element type, with
/// their sets, one command buffer, and the host buffers the sets bind.
const StreamRig = struct {
    ctx: *const Context,
    multi: z.ops_vulkan.AttentionMultiStream,
    split: z.ops_vulkan.AttentionSplit,
    multi_set: OneShot,
    split_set: OneShot,
    merge_set: OneShot,
    q: HostBuffer,
    k: HostBuffer,
    v: HostBuffer,
    scratch: HostBuffer,
    out: HostBuffer,
    kv_bytes: u64 = 0,

    fn init(ctx: *const Context, kv: z.dtype.KvDtype) !StreamRig {
        var multi = try z.ops_vulkan.AttentionMultiStream.initKv(ctx, kv);
        errdefer multi.deinit();

        var split = try z.ops_vulkan.AttentionSplit.initKv(ctx, kv);
        errdefer split.deinit();

        const multi_set = try OneShot.init(ctx, multi.pipeline.set_layout, 4);
        errdefer multi_set.deinit(ctx);

        const split_set = try OneShot.init(
            ctx,
            split.split.set_layout,
            z.ops_vulkan.AttentionSplit.split_bindings,
        );
        errdefer split_set.deinit(ctx);

        const merge_set = try OneShot.init(
            ctx,
            split.merge.set_layout,
            z.ops_vulkan.AttentionSplit.merge_bindings,
        );
        return .{
            .ctx = ctx,
            .multi = multi,
            .split = split,
            .multi_set = multi_set,
            .split_set = split_set,
            .merge_set = merge_set,
            .q = .{ .ctx = ctx },
            .k = .{ .ctx = ctx },
            .v = .{ .ctx = ctx },
            .scratch = .{ .ctx = ctx },
            .out = .{ .ctx = ctx },
        };
    }

    fn deinit(self: *StreamRig) void {
        self.out.deinit();
        self.scratch.deinit();
        self.v.deinit();
        self.k.deinit();
        self.q.deinit();
        self.merge_set.deinit(self.ctx);
        self.split_set.deinit(self.ctx);
        self.multi_set.deinit(self.ctx);
        self.split.deinit();
        self.multi.deinit();
    }

    /// Upload both regions at KV element type `T`. Each region must fit the
    /// storage range of the device, because one binding covers it.
    fn uploadStreams(self: *StreamRig, comptime T: type, streams: Streams) !void {
        const range = self.ctx.buffer_limits.storage_range;
        const bytes = std.mem.alignForward(u64, streams.k.len * @sizeOf(T), 4);
        if (bytes > range) {
            std.debug.print(
                "multi-stream kv={s}: region of {d} bytes past the storage range of {d}\n",
                .{ @typeName(T), bytes, range },
            );
            return error.RegionPastStorageRange;
        }

        self.kv_bytes = try upload(T, &self.k, streams.k);
        _ = try upload(T, &self.v, streams.v);
        std.debug.assert(self.kv_bytes == bytes);
    }

    /// Upload the query rows, fill the output with the sentinel, and bind the
    /// three sets.
    fn stage(self: *StreamRig, q: []const f32, g: Geometry) !void {
        std.debug.assert(q.len == n_streams * g.rowElems());
        std.debug.assert(self.kv_bytes >= 4);

        const q_bytes = try upload(f32, &self.q, q);
        const out_elems = out_rows * g.rowElems();
        const out_bytes: u64 = out_elems * @sizeOf(f32);
        try self.out.ensure(out_bytes);
        @memset(self.out.sliceTyped(u32, out_elems), sentinel_bits);
        const scratch = z.attn_split.Scratch.init(g.n_heads, g.head_dim, deep_depth);
        const scratch_bytes = scratch.totalBytes();
        try self.scratch.ensure(scratch_bytes);
        const q_info: DescriptorBufferInfo = .{
            .buffer = self.q.buffer,
            .offset = 0,
            .range = q_bytes,
        };
        const k_info: DescriptorBufferInfo = .{
            .buffer = self.k.buffer,
            .offset = 0,
            .range = self.kv_bytes,
        };
        const v_info: DescriptorBufferInfo = .{
            .buffer = self.v.buffer,
            .offset = 0,
            .range = self.kv_bytes,
        };
        const scratch_info: DescriptorBufferInfo = .{
            .buffer = self.scratch.buffer,
            .offset = 0,
            .range = scratch_bytes,
        };
        const out_info: DescriptorBufferInfo = .{
            .buffer = self.out.buffer,
            .offset = 0,
            .range = out_bytes,
        };
        z.ops_vulkan.updateStorageBindings(
            self.ctx,
            self.multi_set.desc_set,
            &.{ q_info, k_info, v_info, out_info },
        );
        z.ops_vulkan.AttentionSplit.assertKBinding(k_info);
        z.ops_vulkan.AttentionSplit.assertVBinding(v_info);
        z.ops_vulkan.updateStorageBindings(
            self.ctx,
            self.split_set.desc_set,
            &.{ q_info, k_info, v_info, scratch_info },
        );
        z.ops_vulkan.updateStorageBindings(
            self.ctx,
            self.merge_set.desc_set,
            &.{ scratch_info, out_info },
        );
    }
};

/// Build the query rows of one dispatch. Row j puts its needle on position
/// `t_len - 1` of stream `needles[j]`.
fn fillStreamQueries(
    q: []f32,
    g: Geometry,
    streams: Streams,
    t_len: u32,
    needles: [n_streams]u32,
) void {
    std.debug.assert(q.len == n_streams * g.rowElems());
    std.debug.assert(t_len < stream_window);

    for (needles, 0..) |s, j| {
        fillNeedleRow(
            q[j * g.rowElems() ..][0..g.rowElems()],
            g,
            streams.kOf(s),
            t_len - 1,
            300 + j,
        );
    }
}

/// The CPU reference of each pair, at the output row it names.
fn streamReference(
    allocator: std.mem.Allocator,
    g: Geometry,
    streams: Streams,
    q: []const f32,
    t_len: u32,
    pairs: []const Pair,
    ref: []f32,
) !void {
    std.debug.assert(ref.len == out_rows * g.rowElems());

    const parts = try allocator.alloc(f32, z.attn_cpu.partialLen(g.n_heads, g.head_dim));
    defer allocator.free(parts);

    const kv_len = @as(usize, t_len) * g.kvElems();
    for (pairs) |p| {
        const q_row = q[p.q_row * g.rowElems() ..][0..g.rowElems()];
        const out = ref[p.out_row * g.rowElems() ..][0..g.rowElems()];
        z.ops.attentionRowTyped(
            f32,
            q_row,
            g.n_heads,
            g.head_dim,
            streams.kOf(p.stream)[0..kv_len],
            streams.vOf(p.stream)[0..kv_len],
            g.n_kv_heads,
            t_len,
            parts,
            out,
        );
    }
}

/// Compare the rows that the pairs name, and check that every other output row
/// still holds the sentinel.
fn expectStreamRows(
    g: Geometry,
    label: []const u8,
    kv: []const u8,
    t_len: u32,
    pairs: []const Pair,
    ref: []const f32,
    got: []const f32,
) !void {
    const row_elems = g.rowElems();
    var named = [_]bool{false} ** out_rows;
    for (pairs) |p| {
        named[p.out_row] = true;
        const lo = p.out_row * row_elems;
        try expectRows(
            ref[lo..][0..row_elems],
            got[lo..][0..row_elems],
            g,
            p.out_row,
            1,
            "multi-stream {s} {s} kv={s} t_len {d} q row {d} stream {d} out",
            .{ g.label, label, kv, t_len, p.q_row, p.stream },
        );
    }

    for (named, 0..) |is_named, r| {
        if (is_named) continue;
        const row: []const u32 = @ptrCast(got[r * row_elems ..][0..row_elems]);
        for (row) |bits| {
            if (bits != sentinel_bits) {
                std.debug.print(
                    "multi-stream {s} {s} kv={s} t_len {d}: out row {d} was written\n",
                    .{ g.label, label, kv, t_len, r },
                );
                return error.GpuAttentionWroteForeignRow;
            }
        }
    }
}

fn runStreamChecks(
    comptime T: type,
    rig: *StreamRig,
    allocator: std.mem.Allocator,
    g: Geometry,
    streams: Streams,
) !void {
    const ctx = rig.ctx;
    const kv = @typeName(T);
    try rig.uploadStreams(T, streams);
    const q = try allocator.alloc(f32, n_streams * g.rowElems());
    defer allocator.free(q);

    const ref = try allocator.alloc(f32, out_rows * g.rowElems());
    defer allocator.free(ref);

    const got = try allocator.alloc(f32, out_rows * g.rowElems());
    defer allocator.free(got);

    const cb = rig.multi_set.cmd_buffer;

    // Resident rows. The second dispatch sets the q row base and the stream
    // base apart, so a kernel that reads one base for both fails.
    const own: [n_streams]Pair = .{
        .{ .q_row = 0, .stream = 0, .out_row = 0 },
        .{ .q_row = 1, .stream = 1, .out_row = 1 },
        .{ .q_row = 2, .stream = 2, .out_row = 2 },
        .{ .q_row = 3, .stream = 3, .out_row = 3 },
    };
    const shifted = [_]Pair{
        .{ .q_row = 1, .stream = 2, .out_row = 1 },
        .{ .q_row = 2, .stream = 3, .out_row = 2 },
    };
    for ([_]u32{ 1, split_len }) |t_len| {
        std.debug.assert(!z.attn_split.usesSplit(t_len));

        fillStreamQueries(q, g, streams, t_len, .{ 0, 1, 2, 3 });
        try streamReference(allocator, g, streams, q, t_len, &own, ref);
        try rig.stage(q, g);
        try begin(ctx, cb);
        rig.multi.recordInto(
            cb,
            rig.multi_set.desc_set,
            g.n_heads,
            g.n_kv_heads,
            g.head_dim,
            t_len,
            streams.stride,
            0,
            0,
            n_streams,
        );
        try submitAndWait(ctx, cb);
        @memcpy(got, rig.out.sliceTyped(f32, got.len));
        try expectStreamRows(g, "resident", kv, t_len, &own, ref, got);

        fillStreamQueries(q, g, streams, t_len, .{ 0, 2, 3, 1 });
        try streamReference(allocator, g, streams, q, t_len, &shifted, ref);
        try rig.stage(q, g);
        try begin(ctx, cb);
        rig.multi.recordInto(
            cb,
            rig.multi_set.desc_set,
            g.n_heads,
            g.n_kv_heads,
            g.head_dim,
            t_len,
            streams.stride,
            1,
            2,
            shifted.len,
        );
        try submitAndWait(ctx, cb);
        @memcpy(got, rig.out.sliceTyped(f32, got.len));
        try expectStreamRows(g, "resident shifted", kv, t_len, &shifted, ref, got);
    }

    // Split rows, recorded as a runner records them: one group of four rows
    // that read their own streams, then a second group that reuses scratch
    // row 0 for a row whose q row and stream differ.
    const extra: Pair = .{ .q_row = 3, .stream = 1, .out_row = n_streams };
    const split_pairs = own ++ [_]Pair{extra};
    const max_splits = z.attn_split.Scratch.init(g.n_heads, g.head_dim, deep_depth).max_splits;
    for ([_]u32{ split_len + 1, deep_depth }) |t_len| {
        std.debug.assert(z.attn_split.usesSplit(t_len));

        fillStreamQueries(q, g, streams, t_len, .{ 0, 1, 2, 3 });
        try streamReference(allocator, g, streams, q, t_len, &split_pairs, ref);
        try rig.stage(q, g);
        try begin(ctx, cb);
        for (own, 0..) |p, slot| {
            rig.split.recordSplit(
                cb,
                rig.split_set.desc_set,
                g.n_heads,
                g.n_kv_heads,
                g.head_dim,
                t_len,
                streams.stride,
                p.q_row,
                p.stream,
                @intCast(slot),
                max_splits,
            );
        }

        barrierShader(ctx, cb);
        for (own, 0..) |p, slot| {
            rig.split.recordMerge(
                cb,
                rig.merge_set.desc_set,
                g.n_heads,
                g.head_dim,
                t_len,
                @intCast(slot),
                max_splits,
                p.out_row,
            );
        }

        // The second group writes scratch row 0, which the merges above read.
        barrierShader(ctx, cb);
        rig.split.recordSplit(
            cb,
            rig.split_set.desc_set,
            g.n_heads,
            g.n_kv_heads,
            g.head_dim,
            t_len,
            streams.stride,
            extra.q_row,
            extra.stream,
            0,
            max_splits,
        );
        barrierShader(ctx, cb);
        rig.split.recordMerge(
            cb,
            rig.merge_set.desc_set,
            g.n_heads,
            g.head_dim,
            t_len,
            0,
            max_splits,
            extra.out_row,
        );
        try submitAndWait(ctx, cb);
        @memcpy(got, rig.out.sliceTyped(f32, got.len));
        try expectStreamRows(g, "split", kv, t_len, &split_pairs, ref, got);
    }
}

fn runMultiStreamChecks() !void {
    const allocator = std.testing.allocator;
    const runner = try openRunner(allocator);
    defer runner.deinit();

    var rig32: StreamRig = try .init(&runner.ctx, .f32);
    defer rig32.deinit();

    var rig16: StreamRig = try .init(&runner.ctx, .f16);
    defer rig16.deinit();

    for ([_]Geometry{ smollm2, qwen35 }) |g| {
        const streams: Streams = try .init(allocator, g);
        defer streams.deinit(allocator);

        try runStreamChecks(f32, &rig32, allocator, g, streams);
        try runStreamChecks(f16, &rig16, allocator, g, streams);
    }
}

// -- tests ------------------------------------------------------------------

test "attention: SmolLM2 shape (9h,3kv,d64,t128) within 2e-3" {
    const s: AtnShape = smollm2.at(128);
    const runner = try openRunner(std.testing.allocator);
    defer runner.deinit();

    try runAttentionCheck(f32, std.testing.allocator, &runner.attention, s, 2e-3);
}

test "attention: Llama-1B shape (32h,8kv,d64,t512) within 2e-3" {
    const s: AtnShape = llama1b.at(512);
    const runner = try openRunner(std.testing.allocator);
    defer runner.deinit();

    try runAttentionCheck(f32, std.testing.allocator, &runner.attention, s, 2e-3);
}

test "attention: Llama-8B shape (32h,8kv,d128,t512) within 2e-3" {
    const s: AtnShape = llama8b.at(512);
    const runner = try openRunner(std.testing.allocator);
    defer runner.deinit();

    try runAttentionCheck(f32, std.testing.allocator, &runner.attention, s, 2e-3);
}

test "attention: decode rows of eight shapes at resident and split depths within 2e-3 at f32 KV" {
    try runDecodeChecks(f32, .f32);
}

test "attention: decode rows of eight shapes at resident and split depths within 2e-3 at f16 KV" {
    try runDecodeChecks(f16, .f16);
}

test "attention: a split row gives the same bytes on two runs at both KV types" {
    try repeatSplitRow(f32, .f32);
    try repeatSplitRow(f16, .f16);
}

test "attention: split rows of four families keep their minted digests at both KV types" {
    try runSplitDigests();
}

test "attention prefill: chunks across S on three shapes within 2e-3 at both KV types" {
    try PrefillHarness(f32).oracle();
    try PrefillHarness(f16).oracle();
}

test "attention prefill: a chunk past four splits gives the same bytes on two runs at both KV types" {
    try PrefillHarness(f32).repeat();
    try PrefillHarness(f16).repeat();
}

test "attention prefill: a row past S gets the bytes of the decode row of its depth at both KV types" {
    const starts = [_]u32{ split_len - 3, 4 * split_len + 1, 2046, 8192 };
    try PrefillHarness(f32).equalsDecode(&starts, 7);
    try PrefillHarness(f16).equalsDecode(&starts, 7);
}

test "attention prefill: chunk widths 1, 2, 3, 7, and 13 give the same bytes at both KV types" {
    const starts = [_]u32{ split_len - 3, 4 * split_len + 1 };
    try PrefillHarness(f32).widths(&starts, 13);
    try PrefillHarness(f16).widths(&starts, 13);
}

test "attention prefill: one row per record group gives the bytes of the full budget at both KV types" {
    try PrefillHarness(f32).minimumBudget(2046, 13);
    try PrefillHarness(f16).minimumBudget(2046, 13);
}

test "attention prefill: a resident batch row gets the bytes of the attention.comp row at both KV types" {
    const starts = [_]u32{ 0, split_len - 9 };
    try PrefillHarness(f32).equalsDecode(&starts, 9);
    try PrefillHarness(f16).equalsDecode(&starts, 9);
}

test "attention: multi-stream rows read their own streams within 2e-3 at both KV types" {
    try runMultiStreamChecks();
}
