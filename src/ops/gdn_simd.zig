//! The lane arithmetic of the Gated DeltaNet step for the AVX2 and FMA backends. `kernels` runs
//! the control flow of `gdn.Kernels` over `@Vector(8, f32)` lanes, so the head mapping, the q
//! scale, and the per-head scalars (`softplus`, the decay `@exp`, `sigmoid(b)`) are those of
//! `gdn.reference`. The bits differ from the reference inside a measured tolerance, and
//! `gdn.reference` stays the reference.
//!
//! Comptime. `recurPanel` takes its panel width and its padding at compile time, so the wide, the
//! narrow, and the padded panels share one body.
//!
//! Bit rule. The bits of a value are a function of its inputs alone. Five rules hold that:
//! 1. Every element after the conv takes the vector arithmetic. A last vector under 8 lanes is
//!    padded with zeros, so a channel or a state column gets the same bits at any lane position.
//! 2. Every multiply-accumulate after the conv is one `@mulAdd`.
//! 3. A state column adds its products in ascending order of the key index, one lane per column,
//!    so the panel width changes no bit.
//! 4. Every sum over the elements of a head is `lane_tree_simd.sumSquares` or `lane_tree_simd.dot`,
//!    in the one order of that file.
//! 5. Every SiLU is `x * exp_simd.sigmoid(x)`.
//!
//! So a key-head split, a thread split, and a split of the channels into ranges give the bits of
//! the serial step of this file.
//!
//! The conv. `convStepSilu` runs `gdn.convStep`, the scalar conv of the reference, and only the
//! SiLU after it takes the lanes. Each channel adds its taps in one fixed order, so the conv keeps
//! the bits of a channel at any channel range too.
//!
//! Readout form. The scalar step reads the state twice: once with k for the delta, and once with q
//! after the rank-1 write. With `S' = decay * S + k delta^T`, the readout is
//! `o = S'^T q = (decay * S)^T q + (k . q) delta`. So pass 1 reads the state once and accumulates
//! both `(decay * S)^T k` and `(decay * S)^T q`. Pass 2 reads each state value again and writes
//! `decay * S + k delta^T`, with no readout.
//!
//! Panels. A wide panel holds 4 vectors of columns, so pass 1 keeps 8 independent multiply-add
//! chains in flight. Zen 3 has two FMA pipes at a latency of 4 cycles (the AMD optimization guide
//! for family 19h), so 8 chains keep both busy.

const std = @import("std");
const exp_simd = @import("exp_simd.zig");
const gdn = @import("gdn.zig");
const lane_tree_simd = @import("lane_tree_simd.zig");
const limits = @import("../core/limits.zig");

const gdn_simd = @This();
const assert = std.debug.assert;

// -- constants --------------------------------------------------------------

const lanes = exp_simd.lanes;

/// The largest per-head state dimension. The loader refuses a larger state size.
const max_head_dim: usize = limits.gdn_head_dim;

/// The vectors of one wide panel of `recurStep`.
const wide_panel = 4;

/// The GDN step of the SIMD backends. `simd` and `simd_pf` export its `stepKeyHead`.
pub const kernels = gdn.Kernels(LaneMath);

// -- types ------------------------------------------------------------------

const Vec = exp_simd.Vec;

/// The lane arithmetic that `gdn.Kernels` calls.
const LaneMath = struct {
    pub const convStepSilu = gdn_simd.convStepSilu;
    pub const l2normInplace = gdn_simd.l2normInplace;
    pub const recurStep = gdn_simd.recurStep;
    pub const gatedRmsNormInplace = gdn_simd.gatedRmsNormInplace;
};

// -- helpers ----------------------------------------------------------------

/// The conv of `gdn.convStep` over the channels `[c_lo, c_hi)`, then the lane SiLU, in place.
fn convStepSilu(
    d: gdn.Dims,
    conv_w: []const f32,
    conv_state: []f32,
    qkv: []f32,
    c_lo: usize,
    c_hi: usize,
) void {
    assert(c_lo < c_hi);
    assert(c_hi <= qkv.len);

    gdn.convStep(d, conv_w, conv_state, qkv, c_lo, c_hi);
    const x = qkv[c_lo..c_hi];
    var c: usize = 0;
    while (c + lanes <= x.len) : (c += lanes) {
        const v: Vec = x[c..][0..lanes].*;
        x[c..][0..lanes].* = silu(v);
    }

    if (c < x.len) {
        storePart(x[c..], silu(loadPad(x[c..])));
    }
}

/// L2-normalize `x` in place: `x * (1 / sqrt(sum(x^2) + eps))`.
fn l2normInplace(x: []f32, eps: f32) void {
    assert(x.len > 0);
    assert(eps >= 0);

    const inv: Vec = @splat(1.0 / @sqrt(lane_tree_simd.sumSquares(x) + eps));
    var j: usize = 0;
    while (j + lanes <= x.len) : (j += lanes) {
        const v: Vec = x[j..][0..lanes].*;
        x[j..][0..lanes].* = v * inv;
    }

    if (j < x.len) {
        storePart(x[j..], loadPad(x[j..]) * inv);
    }
}

/// Gated RMSNorm over the value dim of one head, then the product with `silu(gate)`, in place.
/// `o[j] = o[j] * rsqrt(mean(o^2) + eps) * weight[j] * silu(gate[j])`, in that order.
fn gatedRmsNormInplace(o: []f32, weight: []const f32, gate: []const f32, eps: f32) void {
    const dim = o.len;
    assert(dim > 0);
    assert(weight.len == dim);
    assert(gate.len == dim);

    const ss = lane_tree_simd.sumSquares(o);
    const inv: Vec = @splat(1.0 / @sqrt(ss / @as(f32, @floatFromInt(dim)) + eps));
    var j: usize = 0;
    while (j + lanes <= dim) : (j += lanes) {
        const ov: Vec = o[j..][0..lanes].*;
        const wv: Vec = weight[j..][0..lanes].*;
        const gv: Vec = gate[j..][0..lanes].*;
        o[j..][0..lanes].* = ov * inv * wv * silu(gv);
    }

    if (j < dim) {
        const ov = loadPad(o[j..]);
        storePart(o[j..], ov * inv * loadPad(weight[j..]) * silu(loadPad(gate[j..])));
    }
}

/// One head's gated delta-rule step in the readout form: decay, read with k and q, delta,
/// readout, write. `s` is the head's d x d state (`s[i*d + j]`), mutated. `q` and `k` arrive
/// L2-normed.
fn recurStep(
    s: []f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    decay: f32,
    beta: f32,
    o: []f32,
) void {
    const dim = q.len;
    assert(dim > 0);
    assert(dim <= max_head_dim);
    assert(s.len == dim * dim);
    assert(k.len == dim);
    assert(v.len == dim);
    assert(o.len == dim);

    // One dot per head step gives every panel the same `k . q`.
    const qk_v: Vec = @splat(lane_tree_simd.dot(q, k));
    const decay_v: Vec = @splat(decay);
    const beta_v: Vec = @splat(beta);
    var j0: usize = 0;
    while (j0 + wide_panel * lanes <= dim) : (j0 += wide_panel * lanes) {
        recurPanel(wide_panel, false, s, q, k, v, qk_v, decay_v, beta_v, o, j0);
    }

    while (j0 + lanes <= dim) : (j0 += lanes) {
        recurPanel(1, false, s, q, k, v, qk_v, decay_v, beta_v, o, j0);
    }

    if (j0 < dim) {
        recurPanel(1, true, s, q, k, v, qk_v, decay_v, beta_v, o, j0);
    }
}

/// One panel of `n` vectors of columns from column `j0`. A padded panel holds the last
/// `dim - j0` columns, fewer than 8, in one vector with zero pad lanes. A pad lane reads a zero
/// state and a zero `v`, so its delta and readout are zero, and no store writes it.
inline fn recurPanel(
    comptime n: usize,
    comptime padded: bool,
    s: []f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    qk_v: Vec,
    decay_v: Vec,
    beta_v: Vec,
    o: []f32,
    j0: usize,
) void {
    const dim = q.len;
    comptime assert(!padded or n == 1);
    const width = if (padded) dim - j0 else n * lanes;
    assert(width > 0);
    assert(j0 + width <= dim);
    if (padded) assert(width < lanes);

    // Pass 1: read the decayed state once, with k and with q.
    var mem: [n]Vec = @splat(@as(Vec, @splat(0)));
    var acc: [n]Vec = @splat(@as(Vec, @splat(0)));
    var i: usize = 0;
    while (i < dim) : (i += 1) {
        const kv: Vec = @splat(k[i]);
        const qv: Vec = @splat(q[i]);
        const cols = s[i * dim + j0 ..][0..width];
        inline for (0..n) |c| {
            const sd = loadCols(padded, cols, c) * decay_v;
            mem[c] = @mulAdd(Vec, sd, kv, mem[c]);
            acc[c] = @mulAdd(Vec, sd, qv, acc[c]);
        }
    }

    // The delta and the readout `(decay * S)^T q + (k . q) delta`.
    var delta: [n]Vec = undefined;
    inline for (0..n) |c| {
        delta[c] = (loadCols(padded, v[j0..][0..width], c) - mem[c]) * beta_v;
        storeCols(padded, o[j0..][0..width], c, @mulAdd(Vec, qk_v, delta[c], acc[c]));
    }

    // Pass 2: write the state once, `k[i] * delta + decay * S`.
    i = 0;
    while (i < dim) : (i += 1) {
        const kv: Vec = @splat(k[i]);
        const cols = s[i * dim + j0 ..][0..width];
        inline for (0..n) |c| {
            const sd = loadCols(padded, cols, c) * decay_v;
            storeCols(padded, cols, c, @mulAdd(Vec, kv, delta[c], sd));
        }
    }
}

/// Vector `c` of the panel columns at one key index. A padded panel holds fewer than 8 columns and
/// gets zero pad lanes.
inline fn loadCols(comptime padded: bool, cols: []const f32, c: usize) Vec {
    if (padded) return loadPad(cols);
    return cols[c * lanes ..][0..lanes].*;
}

/// Store vector `c` of the panel columns at one key index. A padded panel takes only its own lanes.
inline fn storeCols(comptime padded: bool, cols: []f32, c: usize, x: Vec) void {
    if (padded) return storePart(cols, x);
    cols[c * lanes ..][0..lanes].* = x;
}

/// `x * sigmoid(x)` in each lane.
inline fn silu(x: Vec) Vec {
    return x * exp_simd.sigmoid(x);
}

/// The values of `x`, fewer than 9, in the low lanes of a vector with zero pad lanes.
inline fn loadPad(x: []const f32) Vec {
    assert(x.len > 0);
    assert(x.len <= lanes);

    var pad: [lanes]f32 = @splat(0);
    @memcpy(pad[0..x.len], x);
    return pad;
}

/// Store the low `x.len` lanes of `v` into `x`.
inline fn storePart(x: []f32, v: Vec) void {
    assert(x.len > 0);
    assert(x.len <= lanes);

    const a: [lanes]f32 = v;
    @memcpy(x, a[0..x.len]);
}

// -- tests ------------------------------------------------------------------

/// The constant inputs of a test step, shared by the f32 kernels and the f64 step.
const TestWeights = struct {
    conv_w: []const f32,
    ssm_a: []const f32,
    dt_bias: []const f32,
    ssm_norm: []const f32,
    eps: f32,
};

/// The per-token inputs of a test step.
const TestToken = struct {
    qkv: []const f32,
    z: []const f32,
    a: []const f32,
    b: []const f32,
};

/// One full GDN step in f64, written from the recurrence of the `gdn.zig` header: the conv and
/// the SiLU, the q and k norms and the q scale, then per value head the decay, the read with k,
/// the delta, the rank-1 write, the read with q, and the gated norm. `x` is `convDim` scratch.
fn stepF64(
    d: gdn.Dims,
    w: TestWeights,
    tok: TestToken,
    state: []f64,
    conv: []f64,
    x: []f64,
    o: []f64,
) void {
    const dim = d.head_dim;
    const cd = d.convDim();
    const key_dim = d.keyDim();
    const kk = d.conv_kernel;
    assert(state.len == d.stateLen());
    assert(conv.len == d.convStateLen());
    assert(x.len == cd);
    assert(o.len == d.valueDim());

    for (0..cd) |c| {
        var acc: f64 = 0;
        for (0..kk - 1) |t| {
            acc += @as(f64, w.conv_w[c * kk + t]) * conv[t * cd + c];
        }

        acc += @as(f64, w.conv_w[c * kk + kk - 1]) * tok.qkv[c];
        for (0..kk - 2) |t| {
            conv[t * cd + c] = conv[(t + 1) * cd + c];
        }

        conv[(kk - 2) * cd + c] = tok.qkv[c];
        x[c] = siluF64(acc);
    }

    for (0..d.n_k_heads) |kh| {
        l2normF64(x[kh * dim ..][0..dim], w.eps);
        l2normF64(x[key_dim + kh * dim ..][0..dim], w.eps);
        for (x[kh * dim ..][0..dim]) |*qi| {
            qi.* /= @sqrt(@as(f64, @floatFromInt(dim)));
        }
    }

    for (0..d.n_v_heads) |vh| {
        const ki = vh % d.n_k_heads;
        const q = x[ki * dim ..][0..dim];
        const k = x[key_dim + ki * dim ..][0..dim];
        const v = x[2 * key_dim + vh * dim ..][0..dim];
        const s = state[vh * dim * dim ..][0 .. dim * dim];
        const oh = o[vh * dim ..][0..dim];
        const pre = @as(f64, tok.a[vh]) + w.dt_bias[vh];
        const softplus = if (pre > 20) pre else @log(1 + @exp(pre));
        const decay = @exp(@as(f64, w.ssm_a[vh]) * softplus);
        const beta = 1 / (1 + @exp(-@as(f64, tok.b[vh])));
        for (s) |*sij| {
            sij.* *= decay;
        }

        for (0..dim) |j| {
            var mem: f64 = 0;
            for (0..dim) |i| {
                mem += s[i * dim + j] * k[i];
            }

            const delta = (v[j] - mem) * beta;
            var acc: f64 = 0;
            for (0..dim) |i| {
                s[i * dim + j] += k[i] * delta;
                acc += s[i * dim + j] * q[i];
            }

            oh[j] = acc;
        }

        var ss: f64 = 0;
        for (oh) |oj| {
            ss += oj * oj;
        }

        const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(dim)) + w.eps);
        for (oh, 0..) |*oj, j| {
            oj.* = oj.* * inv * w.ssm_norm[j] * siluF64(tok.z[vh * dim + j]);
        }
    }
}

fn siluF64(x: f64) f64 {
    return x / (1 + @exp(-x));
}

fn l2normF64(x: []f64, eps: f32) void {
    var ss: f64 = 0;
    for (x) |xi| {
        ss += xi * xi;
    }

    const inv = 1 / @sqrt(ss + eps);
    for (x) |*xi| {
        xi.* *= inv;
    }
}

/// `||got - want|| / ||want||` in the 2-norm, in f64.
fn normwise(comptime W: type, got: []const f32, want: []const W) f64 {
    assert(got.len == want.len);
    var num: f64 = 0;
    var den: f64 = 0;
    for (got, want) |g, w_raw| {
        const w: f64 = w_raw;
        num += (g - w) * (g - w);
        den += w * w;
    }

    assert(den > 0);
    return @sqrt(num / den);
}

/// The f32 state and ring of one `gdn.Kernels` type, and the f64 state and ring, over the tokens.
const TestRun = struct {
    state: []f32,
    conv: []f32,
    qkv: []f32,
    o: []f32,

    fn init(
        arena: std.mem.Allocator,
        d: gdn.Dims,
        state0: []const f32,
        conv0: []const f32,
    ) !TestRun {
        return .{
            .state = try arena.dupe(f32, state0),
            .conv = try arena.dupe(f32, conv0),
            .qkv = try arena.alloc(f32, d.convDim()),
            .o = try arena.alloc(f32, d.valueDim()),
        };
    }

    fn step(self: TestRun, comptime K: type, d: gdn.Dims, w: TestWeights, tok: TestToken) void {
        @memcpy(self.qkv, tok.qkv);
        K.step(
            d,
            self.qkv,
            tok.z,
            tok.a,
            tok.b,
            w.conv_w,
            w.ssm_a,
            w.dt_bias,
            w.ssm_norm,
            w.eps,
            self.state,
            self.conv,
            self.o,
        );
    }
};

/// Uniform values in `[lo, hi)`.
fn fillUniform(r: std.Random, x: []f32, lo: f32, hi: f32) void {
    for (x) |*xi| {
        xi.* = lo + (hi - lo) * r.float(f32);
    }
}

/// The largest errors of one comparison: the readout of any token, and the final state.
const Gap = struct { o: f64 = 0, state: f64 = 0 };

/// Run `tokens` steps of `kernels`, `gdn.reference`, and `stepF64` from one start. Return the gaps
/// of both f32 steps to f64 and between the two f32 steps. `a_lo` is the low end of the decay
/// input, so a caller can push the decay near 1.
fn trackF64(
    gpa: std.mem.Allocator,
    d: gdn.Dims,
    seed: u64,
    tokens: usize,
    a_lo: f32,
) !struct { lane: Gap, reference: Gap, pair: Gap } {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const cd = d.convDim();
    const vd = d.valueDim();
    const nv = d.n_v_heads;

    // Fix the weights and the start state.
    const conv_w = try arena.alloc(f32, d.conv_kernel * cd);
    const ssm_a = try arena.alloc(f32, nv);
    const dt_bias = try arena.alloc(f32, nv);
    const ssm_norm = try arena.alloc(f32, d.head_dim);
    fillUniform(r, conv_w, -0.5, 0.5);
    // GGUF ssm_a is -exp(A_log), so it is negative.
    fillUniform(r, ssm_a, -0.2, -0.05);
    fillUniform(r, dt_bias, -0.5, 0.5);
    fillUniform(r, ssm_norm, 0.5, 1.5);
    const w: TestWeights = .{
        .conv_w = conv_w,
        .ssm_a = ssm_a,
        .dt_bias = dt_bias,
        .ssm_norm = ssm_norm,
        .eps = 1e-6,
    };

    const state0 = try arena.alloc(f32, d.stateLen());
    const conv0 = try arena.alloc(f32, d.convStateLen());
    fillUniform(r, state0, -0.1, 0.1);
    fillUniform(r, conv0, -1, 1);
    const lane_run = try TestRun.init(arena, d, state0, conv0);
    const ref_run = try TestRun.init(arena, d, state0, conv0);
    const state64 = try arena.alloc(f64, d.stateLen());
    const conv64 = try arena.alloc(f64, d.convStateLen());
    for (state64, state0) |*s, s0| {
        s.* = s0;
    }

    for (conv64, conv0) |*c, c0| {
        c.* = c0;
    }

    const x64 = try arena.alloc(f64, cd);
    const o64 = try arena.alloc(f64, vd);

    // Step the three forms over fresh tokens.
    const qkv = try arena.alloc(f32, cd);
    const z = try arena.alloc(f32, vd);
    const a = try arena.alloc(f32, nv);
    const b = try arena.alloc(f32, nv);
    var lane: Gap = .{};
    var pair: Gap = .{};
    var reference: Gap = .{};
    for (0..tokens) |_| {
        fillUniform(r, qkv, -1, 1);
        fillUniform(r, z, -2, 2);
        fillUniform(r, a, a_lo, 2);
        fillUniform(r, b, -3, 3);
        const tok: TestToken = .{ .qkv = qkv, .z = z, .a = a, .b = b };
        lane_run.step(kernels, d, w, tok);
        ref_run.step(gdn.reference, d, w, tok);
        stepF64(d, w, tok, state64, conv64, x64, o64);
        lane.o = @max(lane.o, normwise(f64, lane_run.o, o64));
        reference.o = @max(reference.o, normwise(f64, ref_run.o, o64));
        pair.o = @max(pair.o, normwise(f32, lane_run.o, ref_run.o));
    }

    lane.state = normwise(f64, lane_run.state, state64);
    reference.state = normwise(f64, ref_run.state, state64);
    pair.state = normwise(f32, lane_run.state, ref_run.state);
    return .{ .lane = lane, .reference = reference, .pair = pair };
}

test "gdn lanes: six steps track the f64 step at every panel shape" {
    // Head dims 1 and 7 are one padded panel alone, 8 one narrow panel, 24 three, 40 a wide and
    // a narrow panel, and 128 the registry width. Head dim 41 adds a padded panel, which the
    // loader accepts below 128. Each head dim also runs with three value heads per key head.
    const head_dims = [_]usize{ 1, 7, 8, 24, 40, 41, 128 };
    for (head_dims) |dim| {
        for ([_][2]usize{ .{ 1, 1 }, .{ 2, 6 } }) |heads| {
            const d: gdn.Dims = .{
                .n_k_heads = heads[0],
                .n_v_heads = heads[1],
                .head_dim = dim,
                .conv_kernel = 4,
            };
            const gap = try trackF64(std.testing.allocator, d, 0x6D17_5100 + dim, 6, -1);
            // Measured over these 14 shapes on x86_64 with FMA and glibc: the lane gap at most
            // 3.04e-7 on the readout and 3.08e-7 on the state, the reference gap at most 6.85e-7
            // and 2.31e-7. The per-head scalars call the expf and logf of libm, so each bound
            // keeps a margin over its maximum for another libm.
            try std.testing.expect(gap.lane.o <= 5e-7);
            try std.testing.expect(gap.lane.state <= 5e-7);
            try std.testing.expect(gap.reference.o <= 1e-6);
            try std.testing.expect(gap.reference.state <= 4e-7);
        }
    }
}

test "gdn lanes: 8192 steps at the 2B head dims keep a bounded gap to the f64 step" {
    // One head of the 2B shape. This seed draws ssm_a = -0.1756 and dt_bias = 0.3547, so the
    // decay input from -6 gives a decay from 0.651 to 1 - 6.2e-4. The state keeps a long memory,
    // and a drift of the recurrence can grow.
    const d: gdn.Dims = .{ .n_k_heads = 1, .n_v_heads = 1, .head_dim = 128, .conv_kernel = 4 };
    const gap = try trackF64(std.testing.allocator, d, 0x6D17_5200, 8192, -6);
    // Measured on x86_64 with FMA and glibc: the largest readout gap of any step is 7.47e-7 for
    // the lanes, 1.075e-6 for the reference, and 1.24e-6 between the two. The final state gap is
    // 1.79e-7, 1.84e-7, and 2.07e-7. The state gap stays from 1.4e-7 to 2.1e-7 at every power of
    // two from step 64 to step 8192, so the recurrence does not drift. Each bound keeps a margin
    // over its maximum for another libm.
    try std.testing.expect(gap.lane.o <= 1e-6);
    try std.testing.expect(gap.lane.state <= 3e-7);
    try std.testing.expect(gap.reference.o <= 2e-6);
    try std.testing.expect(gap.reference.state <= 3e-7);
    try std.testing.expect(gap.pair.o <= 2e-6);
    try std.testing.expect(gap.pair.state <= 3e-7);
}

test "gdn lanes: stepKeyHead in reverse order gives the bits of step at any channel offset" {
    const gpa = std.testing.allocator;
    // The 2B, the 9B, and two small shapes. At head dim 41 and 12 a head starts at a channel off
    // the lane grid, so a value meets another lane position in `step` than in `stepKeyHead`.
    const shapes = [_]gdn.Dims{
        .{ .n_k_heads = 16, .n_v_heads = 16, .head_dim = 128, .conv_kernel = 4 },
        .{ .n_k_heads = 16, .n_v_heads = 32, .head_dim = 128, .conv_kernel = 4 },
        .{ .n_k_heads = 2, .n_v_heads = 6, .head_dim = 41, .conv_kernel = 4 },
        .{ .n_k_heads = 3, .n_v_heads = 3, .head_dim = 12, .conv_kernel = 4 },
    };
    for (shapes, 0..) |d, si| {
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        var prng = std.Random.DefaultPrng.init(0x6D17_5300 + si);
        const r = prng.random();
        const cd = d.convDim();
        const vd = d.valueDim();
        const nv = d.n_v_heads;
        const conv_w = try arena.alloc(f32, d.conv_kernel * cd);
        const ssm_a = try arena.alloc(f32, nv);
        const dt_bias = try arena.alloc(f32, nv);
        const ssm_norm = try arena.alloc(f32, d.head_dim);
        fillUniform(r, conv_w, -0.5, 0.5);
        fillUniform(r, ssm_a, -1, -0.1);
        fillUniform(r, dt_bias, -0.5, 0.5);
        fillUniform(r, ssm_norm, 0.5, 1.5);
        const state0 = try arena.alloc(f32, d.stateLen());
        const conv0 = try arena.alloc(f32, d.convStateLen());
        fillUniform(r, state0, -0.1, 0.1);
        fillUniform(r, conv0, -1, 1);
        const want = try TestRun.init(arena, d, state0, conv0);
        const got = try TestRun.init(arena, d, state0, conv0);
        const w: TestWeights = .{
            .conv_w = conv_w,
            .ssm_a = ssm_a,
            .dt_bias = dt_bias,
            .ssm_norm = ssm_norm,
            .eps = 1e-6,
        };
        const qkv = try arena.alloc(f32, cd);
        const z = try arena.alloc(f32, vd);
        const a = try arena.alloc(f32, nv);
        const b = try arena.alloc(f32, nv);

        // Several tokens carry the state, so a head that reads a stale key or writes a
        // neighbor moves the later tokens.
        for (0..3) |_| {
            fillUniform(r, qkv, -1, 1);
            fillUniform(r, z, -2, 2);
            fillUniform(r, a, -1, 1);
            fillUniform(r, b, -3, 3);
            const tok: TestToken = .{ .qkv = qkv, .z = z, .a = a, .b = b };
            want.step(kernels, d, w, tok);

            @memcpy(got.qkv, qkv);
            @memset(got.o, std.math.nan(f32));
            const heads: gdn.Heads = .{
                .qkv = got.qkv,
                .z = z,
                .a = a,
                .b = b,
                .ssm_a = ssm_a,
                .dt_bias = dt_bias,
                .ssm_norm = ssm_norm,
                .rms_eps = w.eps,
                .state = got.state,
                .o_out = got.o,
            };
            var kh: usize = d.n_k_heads;
            while (kh > 0) : (kh -= 1) {
                kernels.stepKeyHead(d, got.qkv, conv_w, got.conv, heads, kh - 1);
            }

            const pairs = [_][2][]const f32{
                .{ want.qkv, got.qkv },
                .{ want.o, got.o },
                .{ want.state, got.state },
                .{ want.conv, got.conv },
            };
            for (pairs) |p| {
                try std.testing.expectEqualSlices(
                    u8,
                    std.mem.sliceAsBytes(p[0]),
                    std.mem.sliceAsBytes(p[1]),
                );
            }
        }
    }
}
