//! Gated DeltaNet (GDN) linear-attention CPU reference. The file holds the
//! recurrent decode form of one Qwen3.5 or Qwen3-Next GDN layer per-token step.
//! It is a pure datapath over caller-owned state. The runner owns the recurrent
//! state `S` and the conv ring buffer, and this module only transforms them.
//! GDN replaces the growing KV cache with a fixed-size d x d state matrix per
//! head, so its cost is O(1) in sequence length. This scalar form is the
//! correctness reference and the specification that the GPU kernel mirrors.
//!
//!   Hybrid layer schedule (full_attention_interval = 4). Layer i is full
//!   softmax attention when (i+1) % 4 == 0, and GDN linear attention otherwise.
//!   On the 2B model 18 of the 24 layers are GDN:
//!
//!     layer:  0    1    2    3    4    5    6    7   ...  23
//!     type:  GDN  GDN  GDN  ATT  GDN  GDN  GDN  ATT  ...  ATT
//!
//! The math comes from the recurrent path of HF `Qwen3NextGatedDeltaNet`. The
//! tensor names are the GGUF ones. Per token and per head h the state is S[i,j],
//! with i over the key dim and j over the value dim, and
//! key_dim == value_dim == head_dim = 128:
//!
//!   1. project (caller): qkv = attn_qkv.x_n  (q|k|v, each n_heads*head_dim)
//!      z = attn_gate.x_n
//!      a = ssm_alpha.x_n
//!      b = ssm_beta.x_n  (a,b: [n_heads])
//!   2. causal depthwise conv1d (kernel K=4) over the 3*inner qkv channels,
//!      then SiLU. Decode keeps the last K-1 inputs per channel (conv_state).
//!   3. split q,k,v. Then L2-normalize q and k per head (eps 1e-6).
//!   4. decay  g = ssm_a[h] * softplus(a[h] + dt_bias[h])   (GGUF ssm_a is
//!      already -exp(A_log), so multiply directly and do NOT re-exponentiate)
//!      beta   beta = sigmoid(b[h])
//!   5. gated delta rule, decay-then-read:
//!        S        *= exp(g)                          (scalar decay)
//!        kv_mem[j] = sum_i S[i,j] * k[i]             (read memory with key)
//!        delta[j]  = (v[j] - kv_mem[j]) * beta       (delta-rule correction)
//!        S[i,j]   += k[i] * delta[j]                  (rank-1 write)
//!        o[j]      = sum_i S[i,j] * q[i]             (read memory with query)
//!   6. gated RMSNorm (per head over value dim, weight ssm_norm[head_dim]),
//!      then x silu(z) POST-norm:
//!        o[j] = o[j]*rsqrt(mean_j(o^2)+eps)*ssm_norm[j]*silu(z[j])
//!   7. project (caller): out = ssm_out.o  -> hidden
//!
//! Layout, following the llama.cpp qwen3next optimized path. qkv is contiguous
//! as `[q(inner) | k(inner) | v(inner)]`, a flat wqkv projection split at offset
//! key_dim and 2*key_dim, and each block `[head_dim, n_heads]` is contiguous.
//! The readout `o` and the gate `z` are likewise contiguous per head. The conv
//! weight `w(tap,chan)` sits at `conv_w[chan*K + tap]`, with tap 0 the oldest.
//! The entry conv_state[t*conv_dim + c] holds the input t steps in the past,
//! with t=0 the oldest. The model full-attention q and gate use a per-head
//! interleaved layout instead, and the runner handles that one.

const std = @import("std");

// -- types ------------------------------------------------------------------

/// Largest per-head state dimension that the stack-allocated scratch accepts.
/// Qwen3.5 uses ssm.state_size = 128 on both the 2B and the 9B. The bound keeps
/// the recurrence scratch, the delta vector and a state row, on the stack.
const max_head_dim: usize = 128;

pub const Dims = struct {
    /// GDN key/query heads (= ssm.group_count: 16 on both 2B and 9B).
    n_k_heads: usize,
    /// GDN value heads (= ssm.time_step_rank: 16 on the 2B, 32 on the 9B).
    /// The heads decouple when n_v_heads > n_k_heads. Each key head then feeds
    /// n_v/n_k value heads in the GQA style, and each VALUE head carries its own
    /// dxd recurrent state. The 2B is the n_v == n_k special case.
    n_v_heads: usize,
    /// Per-head state dim d (= ssm.state_size, 128). Key and value head dims are equal.
    head_dim: usize,
    /// Causal conv kernel width (= ssm.conv_kernel, 4).
    conv_kernel: usize,

    /// q and k width: n_k_heads x d (= ssm.group_count x state_size).
    pub fn keyDim(self: Dims) usize {
        return self.n_k_heads * self.head_dim;
    }
    /// v / z / readout width: n_v_heads x d (= ssm.inner_size).
    pub fn valueDim(self: Dims) usize {
        return self.n_v_heads * self.head_dim;
    }
    /// Conv channels: [q(key_dim) | k(key_dim) | v(value_dim)].
    pub fn convDim(self: Dims) usize {
        return 2 * self.keyDim() + self.valueDim();
    }
    /// Recurrent state: one dxd matrix per VALUE head.
    pub fn stateLen(self: Dims) usize {
        return self.n_v_heads * self.head_dim * self.head_dim;
    }
    /// Floats of conv ring buffer: (K-1) x convDim.
    pub fn convStateLen(self: Dims) usize {
        return (self.conv_kernel - 1) * self.convDim();
    }
};

// -- api --------------------------------------------------------------------

/// Chunk-parallel gated delta-rule scan for ONE value head. The result matches
/// `recurStep` run over the `T` tokens in sequence, but the form is matrix ops
/// plus one TxT strictly-lower-triangular solve. The GPU prefill kernel mirrors
/// that form. The serial t-loop collapses into the solve, and the rest is
/// parallel matmuls.
///
/// Derivation. State S[i,j] has i over the key dim and j over the value dim.
/// The per-token scalar decay d_t is in (0,1], and beta_t is in (0,1). S_in is
/// the incoming state:
///   P[r] = prod_{u=0..r} d_u                       (cumulative decay, <= 1)
///   delta_r = beta_r (v_r - P[r]*Sin^T k_r) - sum_{s<r} beta_r (P[r]/P[s])(k_s.k_r) delta_s
///   o_r  = P[r]*Sin^T q_r + sum_{s<=r} (P[r]/P[s])(k_s.q_r) delta_s
///   S_out = P[T-1]*S_in + sum_s (P[T-1]/P[s]) k_s (x) delta_s
/// Every decay factor is a product of d_u values in (0,1], so every ratio stays
/// at or below 1. The scan is therefore stable across a full 128-token chunk,
/// with no 1/A blow-up. A sub-chunked GPU port is a free choice under bounded
/// shared memory, not a numerical need.
///
/// `s` is [d*d] in and out. `q`, `k` and `v` are [T*d], the value head's
/// per-token rows, with q already normed and scaled and k normed. `decay` and
/// `beta` are [T]. `o` is [T*d]. The call allocates O(T^2 + T*d) scratch, so it
/// belongs on the reference and test path, not in the hot loop.
pub fn chunkScan(
    gpa: std.mem.Allocator,
    d: usize,
    n_tokens: usize,
    s: []f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    decay: []const f32,
    beta: []const f32,
    o: []f32,
) std.mem.Allocator.Error!void {
    const T = n_tokens;
    std.debug.assert(s.len == d * d);
    std.debug.assert(q.len == T * d);
    std.debug.assert(k.len == T * d);
    std.debug.assert(v.len == T * d);
    std.debug.assert(decay.len == T);
    std.debug.assert(beta.len == T);
    std.debug.assert(o.len == T * d);
    if (T == 0) return;

    // The cumulative decay P from the derivation.
    const P = try gpa.alloc(f32, T);
    defer gpa.free(P);
    {
        var acc: f32 = 1;
        for (0..T) |r| {
            acc *= decay[r];
            P[r] = acc;
        }
    }

    // `s` still holds S_in here, because the state update comes last.
    const sin_k = try gpa.alloc(f32, T * d);
    defer gpa.free(sin_k);
    const sin_q = try gpa.alloc(f32, T * d);
    defer gpa.free(sin_q);
    for (0..T) |r| {
        const kr = k[r * d ..][0..d];
        const qr = q[r * d ..][0..d];
        for (0..d) |j| {
            var sk: f32 = 0;
            var sq: f32 = 0;
            for (0..d) |i| {
                const sij = s[i * d + j];
                sk += sij * kr[i];
                sq += sij * qr[i];
            }
            sin_k[r * d + j] = sk;
            sin_q[r * d + j] = sq;
        }
    }

    // delta from the derivation, solved by forward substitution.
    const delta = try gpa.alloc(f32, T * d);
    defer gpa.free(delta);
    for (0..T) |r| {
        const kr = k[r * d ..][0..d];
        const vr = v[r * d ..][0..d];
        for (0..d) |j| delta[r * d + j] = beta[r] * (vr[j] - P[r] * sin_k[r * d + j]);
        var sidx: usize = 0;
        while (sidx < r) : (sidx += 1) {
            const ks = k[sidx * d ..][0..d];
            var kk: f32 = 0;
            for (0..d) |i| kk += ks[i] * kr[i];
            const coef = beta[r] * (P[r] / P[sidx]) * kk;
            for (0..d) |j| delta[r * d + j] -= coef * delta[sidx * d + j];
        }
    }

    // The per-token readout o from the derivation.
    for (0..T) |r| {
        const qr = q[r * d ..][0..d];
        for (0..d) |j| o[r * d + j] = P[r] * sin_q[r * d + j];
        var sidx: usize = 0;
        while (sidx <= r) : (sidx += 1) {
            const ks = k[sidx * d ..][0..d];
            var kq: f32 = 0;
            for (0..d) |i| kq += ks[i] * qr[i];
            const coef = (P[r] / P[sidx]) * kq;
            for (0..d) |j| o[r * d + j] += coef * delta[sidx * d + j];
        }
    }

    // The outgoing state S_out from the derivation.
    const plast = P[T - 1];
    for (s) |*x| x.* *= plast;
    var sidx: usize = 0;
    while (sidx < T) : (sidx += 1) {
        const ks = k[sidx * d ..][0..d];
        const ds = delta[sidx * d ..][0..d];
        const coef = plast / P[sidx];
        for (0..d) |i| {
            const ci = coef * ks[i];
            const row = s[i * d ..][0..d];
            for (0..d) |j| row[j] += ci * ds[j];
        }
    }
    if (std.debug.runtime_safety) {
        for (o[0 .. T * d]) |x| std.debug.assert(std.math.isFinite(x));
        for (s) |x| std.debug.assert(std.math.isFinite(x));
    }
}

/// The inputs and outputs of the value-head loop of one GDN step. The
/// prologue writes `qkv` before this loop starts. A value-head range reads
/// `qkv`, the per-head inputs, and the weights, and it writes only the state
/// and the readout of its own heads. So disjoint ranges can run at the same
/// time, and each head gets the bits of the serial step.
pub const Heads = struct {
    /// [convDim] q|k|v after `stepPrologue`: conv, SiLU, and the q and k norms.
    qkv: []const f32,
    /// [inner] gate projection.
    z: []const f32,
    /// [n_v_heads] decay projection (ssm_alpha * x_n).
    a: []const f32,
    /// [n_v_heads] beta projection (ssm_beta * x_n).
    b: []const f32,
    /// [n_v_heads] A_log.
    ssm_a: []const f32,
    /// [n_v_heads] ssm_dt.bias.
    dt_bias: []const f32,
    /// [head_dim] gated-rmsnorm weight.
    ssm_norm: []const f32,
    rms_eps: f32,
    /// [stateLen] recurrent state. Head `vh` owns `[vh*d*d, (vh+1)*d*d)`.
    state: []f32,
    /// [inner] output before ssm_out. Head `vh` owns `[vh*d, (vh+1)*d)`.
    o_out: []f32,
};

/// Full per-token GDN step. It mutates `qkv` with the conv result, plus `state`
/// and `conv_state`. It writes the result before the output projection into
/// `o_out`, which has length `inner`. The caller supplies the projected `qkv`,
/// the gate `z`, the decay and beta projections `a` and `b`, and the GDN
/// weights. The caller applies `ssm_out` to `o_out` afterwards. The step is
/// `stepPrologue`, then `stepHeads` over every value head.
pub fn step(
    d: Dims,
    qkv: []f32, // [convDim] projected q|k|v (mutated: conv + silu in place)
    z: []const f32, // [inner] gate projection
    a: []const f32, // [n_heads] decay projection (ssm_alpha * x_n)
    b: []const f32, // [n_heads] beta projection (ssm_beta * x_n)
    conv_w: []const f32, // [K*convDim] ssm_conv1d weights
    ssm_a: []const f32, // [n_heads] A_log
    dt_bias: []const f32, // [n_heads] ssm_dt.bias
    ssm_norm: []const f32, // [head_dim] gated-rmsnorm weight
    rms_eps: f32,
    state: []f32, // [stateLen] recurrent state (mutated)
    conv_state: []f32, // [convStateLen] conv ring buffer (mutated)
    o_out: []f32, // [inner] output before ssm_out
) void {
    stepPrologue(d, qkv, conv_w, conv_state, rms_eps);
    const heads: Heads = .{
        .qkv = qkv,
        .z = z,
        .a = a,
        .b = b,
        .ssm_a = ssm_a,
        .dt_bias = dt_bias,
        .ssm_norm = ssm_norm,
        .rms_eps = rms_eps,
        .state = state,
        .o_out = o_out,
    };
    stepHeads(d, heads, 0, d.n_v_heads);
    assertFinite(d, o_out);
}

/// The serial prologue of one GDN step: the conv and SiLU over every channel,
/// then the L2 norm of q and k per key head and the scale of q. It mutates
/// `qkv` and `conv_state`. Every value head reads its key head from the
/// result, so the prologue must end before any value head starts.
pub fn stepPrologue(
    d: Dims,
    qkv: []f32, // [convDim] projected q|k|v (mutated: conv + silu in place)
    conv_w: []const f32, // [K*convDim] ssm_conv1d weights
    conv_state: []f32, // [convStateLen] conv ring buffer (mutated)
    rms_eps: f32,
) void {
    const dim = d.head_dim;
    const key_dim = d.keyDim();
    std.debug.assert(dim <= max_head_dim);
    std.debug.assert(qkv.len == d.convDim());
    std.debug.assert(conv_state.len == d.convStateLen());

    // 1-2. conv + silu over the concatenated channels.
    convStepSilu(d, conv_w, conv_state, qkv);
    // The contiguous layout gives q and k key_dim each, over n_k heads, and v
    // value_dim over n_v heads. The 2B has n_v == n_k, so all three match.
    const q = qkv[0..key_dim];
    const k = qkv[key_dim .. 2 * key_dim];

    // 3. L2-normalize q and k per KEY head, then scale q by 1/sqrt(head_dim).
    // The scale runs once per key head, not per value head, so it lands exactly
    // once even when v_per_k > 1. It matches llama.cpp `ggml_scale(q,
    // 1/sqrt(S_k))` and feeds only the readout. The gated RMSNorm is not
    // scale-invariant when the state is small, because eps breaks it. Dropping
    // the scale inflates the output about 1.5x.
    const q_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
    var kh: usize = 0;
    while (kh < d.n_k_heads) : (kh += 1) {
        const qh = q[kh * dim ..][0..dim];
        const kk = k[kh * dim ..][0..dim];
        l2normInplace(qh, rms_eps);
        l2normInplace(kk, rms_eps);
        for (qh) |*x| x.* *= q_scale;
    }
    if (std.debug.runtime_safety) {
        for (qkv[0 .. 2 * key_dim]) |x| std.debug.assert(std.math.isFinite(x));
    }
}

/// The recurrence and the gated norm of the value heads `[h_lo, h_hi)`, after
/// `stepPrologue`. It writes only the state and the readout of those heads.
pub fn stepHeads(d: Dims, heads: Heads, h_lo: usize, h_hi: usize) void {
    const dim = d.head_dim;
    const key_dim = d.keyDim();
    const value_dim = d.valueDim();
    std.debug.assert(h_lo < h_hi);
    std.debug.assert(h_hi <= d.n_v_heads);
    std.debug.assert(dim <= max_head_dim);
    std.debug.assert(heads.qkv.len == d.convDim());
    std.debug.assert(heads.z.len == value_dim);
    std.debug.assert(heads.o_out.len == value_dim);
    std.debug.assert(heads.a.len == d.n_v_heads);
    std.debug.assert(heads.b.len == d.n_v_heads);
    std.debug.assert(heads.ssm_a.len == d.n_v_heads);
    std.debug.assert(heads.dt_bias.len == d.n_v_heads);
    std.debug.assert(heads.ssm_norm.len == dim);
    std.debug.assert(heads.state.len == d.stateLen());

    const q = heads.qkv[0..key_dim];
    const k = heads.qkv[key_dim .. 2 * key_dim];
    const v = heads.qkv[2 * key_dim .. 2 * key_dim + value_dim];

    // 4-6. recurrence + gated-norm per VALUE head. The key head is vh % n_k.
    // llama.cpp repeats q/k from n_k to n_v heads with ggml_repeat, which TILES
    // the key heads. Value head j uses key head j mod n_k, NOT contiguous j/v_per_k.
    var vh: usize = h_lo;
    while (vh < h_hi) : (vh += 1) {
        const ki = vh % d.n_k_heads;
        const qh = q[ki * dim ..][0..dim];
        const kk = k[ki * dim ..][0..dim];
        const vv = v[vh * dim ..][0..dim];
        const oh = heads.o_out[vh * dim ..][0..dim];
        // GGUF ssm_a is already -exp(A_log), so multiply directly and do NOT re-exp.
        const g = heads.ssm_a[vh] * softplus(heads.a[vh] + heads.dt_bias[vh]);
        const beta = sigmoid(heads.b[vh]);
        recurStep(heads.state[vh * dim * dim ..][0 .. dim * dim], qh, kk, vv, @exp(g), beta, oh);
        gatedRmsNormInplace(oh, heads.ssm_norm, heads.z[vh * dim ..][0..dim], heads.rms_eps);
    }
}

/// Assert in safe builds that every readout of a finished step is finite. A
/// caller that splits the value heads runs it after every range ends.
pub fn assertFinite(d: Dims, o_out: []const f32) void {
    std.debug.assert(o_out.len == d.valueDim());
    if (std.debug.runtime_safety) {
        for (o_out) |x| std.debug.assert(std.math.isFinite(x));
    }
}

// -- helpers ----------------------------------------------------------------

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn silu(x: f32) f32 {
    return x * sigmoid(x);
}

fn softplus(x: f32) f32 {
    // log(1+exp(x)). For large x it is close to x, and the branch avoids an
    // exp overflow.
    return if (x > 20.0) x else @log(1.0 + @exp(x));
}

/// L2-normalize `x` in place: x /= sqrt(sum(x^2) + eps).
fn l2normInplace(x: []f32, eps: f32) void {
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss + eps);
    for (x) |*v| v.* *= inv;
}

/// Causal depthwise conv1d over all `convDim` channels of `qkv`, then SiLU, in
/// place. `conv_state` holds the last K-1 inputs per channel with t=0 oldest,
/// and this function advances it by one step. `conv_w` is
/// `w(tap,chan) = conv_w[chan*K+tap]`.
fn convStepSilu(d: Dims, conv_w: []const f32, conv_state: []f32, qkv: []f32) void {
    const k = d.conv_kernel;
    const cdim = d.convDim();
    std.debug.assert(conv_w.len == k * cdim);
    std.debug.assert(conv_state.len == (k - 1) * cdim);
    std.debug.assert(qkv.len == cdim);

    var c: usize = 0;
    while (c < cdim) : (c += 1) {
        var acc: f32 = 0;
        var t: usize = 0;
        while (t < k - 1) : (t += 1) acc += conv_w[c * k + t] * conv_state[t * cdim + c];
        acc += conv_w[c * k + (k - 1)] * qkv[c];

        // Advance the ring: drop the oldest entry, append the current input.
        t = 0;
        while (t < k - 2) : (t += 1) conv_state[t * cdim + c] = conv_state[(t + 1) * cdim + c];
        conv_state[(k - 2) * cdim + c] = qkv[c];

        qkv[c] = silu(acc);
    }
}

/// One head's gated delta-rule step: decay, read, delta, write, readout.
/// `s` is the head's dxd state (`s[i*D + j]`) and the call mutates it. `o`
/// receives the length-D readout. `q`, `k` and `v` are the head's inputs, with
/// q and k already L2-normed.
fn recurStep(s: []f32, q: []const f32, k: []const f32, v: []const f32, decay: f32, beta: f32, o: []f32) void {
    const dim = q.len;
    std.debug.assert(s.len == dim * dim);
    std.debug.assert(k.len == dim);
    std.debug.assert(v.len == dim);
    std.debug.assert(o.len == dim);
    std.debug.assert(dim <= max_head_dim);

    for (s) |*x| x.* *= decay;

    var delta: [max_head_dim]f32 = undefined;
    var j: usize = 0;
    while (j < dim) : (j += 1) {
        var mem: f32 = 0;
        var i: usize = 0;
        while (i < dim) : (i += 1) mem += s[i * dim + j] * k[i];
        delta[j] = (v[j] - mem) * beta;
    }

    @memset(o, 0);
    var i: usize = 0;
    while (i < dim) : (i += 1) {
        const ki = k[i];
        const qi = q[i];
        const row = s[i * dim ..][0..dim];
        j = 0;
        while (j < dim) : (j += 1) {
            row[j] += ki * delta[j];
            o[j] += row[j] * qi;
        }
    }
}

/// Gated RMSNorm over a head's value dim, then x silu(gate), in place.
/// `o[j] = o[j]*rsqrt(mean(o^2)+eps)*weight[j]*silu(gate[j])`.
fn gatedRmsNormInplace(o: []f32, weight: []const f32, gate: []const f32, eps: f32) void {
    const dim = o.len;
    std.debug.assert(weight.len == dim);
    std.debug.assert(gate.len == dim);
    var ss: f32 = 0;
    for (o) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(dim)) + eps);
    var j: usize = 0;
    while (j < dim) : (j += 1) o[j] = o[j] * inv * weight[j] * silu(gate[j]);
}

// -- tests ------------------------------------------------------------------

test "convStepSilu tap order: only-current vs only-oldest" {
    const d: Dims = .{ .n_k_heads = 1, .n_v_heads = 1, .head_dim = 1, .conv_kernel = 4 };
    // convDim = 3 (q,k,v are 1 channel each). Test channel 0 only, others 0.
    // conv_state laid out [t*cdim + c], t=0 oldest. Set channel 0 history 1,2,3.
    var conv_state = [_]f32{0} ** 9; // (K-1=3) * cdim(3)
    conv_state[0 * 3 + 0] = 1; // t=0 (oldest)
    conv_state[1 * 3 + 0] = 2; // t=1
    conv_state[2 * 3 + 0] = 3; // t=2 (newest in state)
    // weight w(tap,chan)=conv_w[chan*K+tap]. Channel 0 taps:
    var conv_w = [_]f32{0} ** 12; // K(4)*cdim(3)
    conv_w[0 * 4 + 3] = 1; // only the current tap (tap K-1)
    var qkv = [_]f32{ 4, 0, 0 }; // current input for channel 0 is 4
    convStepSilu(d, &conv_w, &conv_state, &qkv);
    try std.testing.expectApproxEqAbs(silu(4.0), qkv[0], 1e-5); // picks current

    // Reset. Weight only the oldest tap (tap 0).
    conv_state = [_]f32{0} ** 9;
    conv_state[0 * 3 + 0] = 1;
    conv_state[1 * 3 + 0] = 2;
    conv_state[2 * 3 + 0] = 3;
    conv_w = [_]f32{0} ** 12;
    conv_w[0 * 4 + 0] = 1; // only the oldest tap
    qkv = .{ 4, 0, 0 };
    convStepSilu(d, &conv_w, &conv_state, &qkv);
    try std.testing.expectApproxEqAbs(silu(1.0), qkv[0], 1e-5); // picks oldest (=1)
}

test "recurStep first token: o = (k·q)·beta·v when state starts at zero" {
    // S_0 = 0, decay arbitrary (S stays 0 after decay). Then:
    //   kv_mem = 0
    //   delta = v*beta
    //   S = k(x)delta
    //   o[j] = sum_i k[i]*(v[j]*beta)*q[i] = (k.q)*beta*v[j]
    const dim = 4;
    var s = [_]f32{0} ** (dim * dim);
    const q = [_]f32{ 0.5, -0.5, 1.0, 0.25 };
    const k = [_]f32{ 1.0, 2.0, -1.0, 0.5 };
    const v = [_]f32{ 3.0, -1.0, 0.0, 2.0 };
    const beta: f32 = 0.7;
    var o = [_]f32{0} ** dim;
    recurStep(&s, &q, &k, &v, 1.0, beta, &o);

    var kq: f32 = 0;
    for (q, k) |qi, ki| kq += qi * ki;
    for (v, o) |vj, oj| try std.testing.expectApproxEqAbs(kq * beta * vj, oj, 1e-5);
}

test "chunkScan matches sequential recurStep (state + per-token readout)" {
    const gpa = std.testing.allocator;
    // Math is head-dim agnostic. Small d plus several T sizes (incl. T=1 and a
    // multi-token chunk) exercise the triangular solve + state carry.
    const d: usize = 24;
    for ([_]usize{ 1, 2, 7, 33 }) |T| {
        var prng = std.Random.DefaultPrng.init(0xC0DE + T);
        const r = prng.random();

        const q = try gpa.alloc(f32, T * d);
        defer gpa.free(q);
        const k = try gpa.alloc(f32, T * d);
        defer gpa.free(k);
        const v = try gpa.alloc(f32, T * d);
        defer gpa.free(v);
        const decay = try gpa.alloc(f32, T);
        defer gpa.free(decay);
        const beta = try gpa.alloc(f32, T);
        defer gpa.free(beta);
        for (q) |*x| x.* = (r.float(f32) - 0.5);
        for (k) |*x| x.* = (r.float(f32) - 0.5);
        for (v) |*x| x.* = (r.float(f32) - 0.5) * 2.0;
        // decay in (0,1) via exp(g) with g<0. beta in (0,1) via sigmoid.
        for (decay) |*x| x.* = @exp(-(r.float(f32) * 0.5)); // ~[0.6, 1.0]
        for (beta) |*x| x.* = sigmoid((r.float(f32) - 0.5) * 4.0);

        const s0 = try gpa.alloc(f32, d * d);
        defer gpa.free(s0);
        for (s0) |*x| x.* = (r.float(f32) - 0.5) * 0.2;

        // Sequential reference: recurStep over the T tokens.
        const s_seq = try gpa.dupe(f32, s0);
        defer gpa.free(s_seq);
        const o_seq = try gpa.alloc(f32, T * d);
        defer gpa.free(o_seq);
        for (0..T) |t| {
            recurStep(s_seq, q[t * d ..][0..d], k[t * d ..][0..d], v[t * d ..][0..d], decay[t], beta[t], o_seq[t * d ..][0..d]);
        }

        // Chunked.
        const s_chunk = try gpa.dupe(f32, s0);
        defer gpa.free(s_chunk);
        const o_chunk = try gpa.alloc(f32, T * d);
        defer gpa.free(o_chunk);
        try chunkScan(gpa, d, T, s_chunk, q, k, v, decay, beta, o_chunk);

        for (o_seq, o_chunk) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-4);
        for (s_seq, s_chunk) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-4);
    }
}

test "l2normInplace yields unit norm" {
    var x = [_]f32{ 3, 4, 0, 0 };
    l2normInplace(&x, 0);
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ss, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), x[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), x[1], 1e-5);
}

test "step: the prologue, then every value head alone in reverse order, gives the bits of step" {
    const gpa = std.testing.allocator;
    // The registry shapes: the 2B (16 key and 16 value heads) and the 9B
    // (16 key and 32 value heads), head dim 128, conv kernel 4.
    const shapes = [_]Dims{
        .{ .n_k_heads = 16, .n_v_heads = 16, .head_dim = 128, .conv_kernel = 4 },
        .{ .n_k_heads = 16, .n_v_heads = 32, .head_dim = 128, .conv_kernel = 4 },
    };
    const tokens = 3;
    for (shapes) |d| {
        for ([_]u64{ 0x6D17_0001, 0x6D17_0002, 0x6D17_0003 }) |seed| {
            var prng = std.Random.DefaultPrng.init(seed);
            const r = prng.random();
            const cd = d.convDim();
            const vd = d.valueDim();
            const nv = d.n_v_heads;
            const buf = try gpa.alloc(f32, 2 * cd + vd + 2 * nv + d.conv_kernel * cd + 2 * nv + d.head_dim);
            defer gpa.free(buf);
            for (buf) |*x| x.* = r.float(f32) * 2.0 - 1.0;
            var at: usize = 0;
            const qkv_in = buf[at..][0..cd];
            at += cd;
            const z = buf[at..][0..vd];
            at += vd;
            const a = buf[at..][0..nv];
            at += nv;
            const b = buf[at..][0..nv];
            at += nv;
            const conv_w = buf[at..][0 .. d.conv_kernel * cd];
            at += d.conv_kernel * cd;
            const ssm_a = buf[at..][0..nv];
            at += nv;
            const dt_bias = buf[at..][0..nv];
            at += nv;
            const ssm_norm = buf[at..][0..d.head_dim];
            // GGUF ssm_a is -exp(A_log), so it is negative.
            for (ssm_a) |*x| x.* = -@abs(x.*) - 0.1;

            const state_ref = try gpa.alloc(f32, d.stateLen());
            defer gpa.free(state_ref);
            for (state_ref) |*x| x.* = (r.float(f32) - 0.5) * 0.2;
            const state_split = try gpa.dupe(f32, state_ref);
            defer gpa.free(state_split);
            const conv_ref = try gpa.alloc(f32, d.convStateLen());
            defer gpa.free(conv_ref);
            for (conv_ref) |*x| x.* = r.float(f32) - 0.5;
            const conv_split = try gpa.dupe(f32, conv_ref);
            defer gpa.free(conv_split);
            const qkv_ref = try gpa.alloc(f32, cd);
            defer gpa.free(qkv_ref);
            const qkv_split = try gpa.alloc(f32, cd);
            defer gpa.free(qkv_split);
            const o_ref = try gpa.alloc(f32, vd);
            defer gpa.free(o_ref);
            const o_split = try gpa.alloc(f32, vd);
            defer gpa.free(o_split);

            // Several tokens carry the state, so a head that reads a stale key
            // or writes a neighbor moves the later tokens.
            for (0..tokens) |t| {
                for (qkv_ref, qkv_in) |*x, y| x.* = y + @as(f32, @floatFromInt(t)) * 0.25;
                @memcpy(qkv_split, qkv_ref);
                step(d, qkv_ref, z, a, b, conv_w, ssm_a, dt_bias, ssm_norm, 1e-6, state_ref, conv_ref, o_ref);

                @memset(o_split, std.math.nan(f32));
                stepPrologue(d, qkv_split, conv_w, conv_split, 1e-6);
                const heads: Heads = .{
                    .qkv = qkv_split,
                    .z = z,
                    .a = a,
                    .b = b,
                    .ssm_a = ssm_a,
                    .dt_bias = dt_bias,
                    .ssm_norm = ssm_norm,
                    .rms_eps = 1e-6,
                    .state = state_split,
                    .o_out = o_split,
                };
                var vh: usize = nv;
                while (vh > 0) : (vh -= 1) stepHeads(d, heads, vh - 1, vh);
                assertFinite(d, o_split);

                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(qkv_ref), std.mem.sliceAsBytes(qkv_split));
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(o_ref), std.mem.sliceAsBytes(o_split));
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(state_ref), std.mem.sliceAsBytes(state_split));
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(conv_ref), std.mem.sliceAsBytes(conv_split));
            }
        }
    }
}
