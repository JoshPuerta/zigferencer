//! Token selection from a logits vector.
//!
//! Comptime. Asserts pin the weight block to the lane count and the radix key to 32 bits.
//!
//! Greedy. `greedy` is argmax. It is deterministic by construction, with no RNG, temperature, or
//! top-k/p, and it anchors every cross-backend token-stream hash gate. The sampled path never
//! touches it.
//!
//! Reproducibility. `sample` adds seeded temperature, top-k, min-p, and top-p for sampling variety.
//! The caller owns the RNG, one per session, seeded on purpose. Every sampled draw takes exactly
//! one float from it, and a greedy draw takes none. The same seed, the same logits, and the same
//! build then reproduce the same token. The promise holds for one build only. A build that changes
//! the exponential or a cut can draw a different id for the same seed, from the same distribution.
//! Bit-identity of a sampled stream across platforms is not promised, because the softmax goes
//! through f32 exponentials. The cross-backend gate stays greedy.
//!
//! Chain. The chain is the penalty fix-up, top_k, temperature, min_p, top_p, then the draw. min_p
//! keeps the ids whose weight is at least `min_p` times the largest weight. top_p then keeps the
//! smallest ranked prefix of the min_p survivors whose mass reaches `top_p` of their mass. Ties
//! rank the lower id first, the order `greedy` uses.
//!
//! Pool and full vocab. A positive top_k selects a pool of at most `limits.max_top_k` candidates,
//! and every cut runs inside the pool. With top_k 0 every cut is exact over the full vocab. One
//! vector exponential pass writes the weights and their block sums into the caller scratch. A
//! radix select over the weight bits finds the cut, one pass zeroes the weights below it, and the
//! CDF walk reads the rest. No step sorts the full vocab. The CDF walk skips whole blocks by their
//! sums, so it reads about `V / 64` sums and at most 64 weights.
//!
//! Penalties. A request with a penalty copies the row into the caller scratch and changes the logit
//! of each marked id. An id in the context set takes the repetition rule `l > 0 ? l / r : l * r`,
//! and an id in the presence set then loses `presence_penalty`. A neutral request reads the logits
//! buffer as it is. The sampler never writes the logits buffer, because the prefix cache keeps
//! final logits. Every greedy hash therefore stays unchanged. The penalty walk reads the presence
//! set only when the presence penalty is not 0, and the context set only when the factor is not 1.
//! It walks the set bits, so its cost past the copy is `V / 64` words plus one step per marked id.
//! It reads no bit past `source.len`, so a padding bit of the last word never names an id. A result
//! past the finite f32 range clamps to it, and a marked NaN or -inf becomes `-floatMax`, so the row
//! maximum stays finite. A presence-only id computes `l - presence`, the f32 operation of a
//! per-element view, so a presence stream keeps its bits.
//!
//! Exact top_p cut. A weight in [0, 1] has f32 bits that order as the values do, so a histogram of
//! a bit field ranks whole buckets by mass. Each radix level adds the mass of a field value from
//! the top down, until the mass reaches the target. That value is the boundary bucket, and every
//! higher bucket survives whole. A boundary bucket of at most `cut_candidates_max` ids is sorted,
//! and the walk inside it finds the cut. A larger bucket takes the next 10 bits. After three
//! levels the bucket holds one weight value, and only its tie count is left to find. The masses sum
//! in f64, so the cut matches a sorted f64 walk to f64 rounding. The scans skip every weight at or
//! below `skip = (1 - top_p) floor_mass / (2 V)`, where `floor_mass` is a lower bound on the mass
//! of the floor set, at least 1. Those weights hold at most half of `(1 - top_p)` of the floor set
//! mass. The mass above `skip` therefore passes the target with room, so the cut lies above
//! `skip`, and a skipped weight never survives.
//!
//! Non-finite logits. A NaN or a +inf weight in a model file can reach the logits: in a release
//! build, or through a GPU kernel. The loader cannot refuse it without a scan of every weight.
//! `sample` therefore refuses a row with a NaN, a +inf, or no finite logit, with
//! `error.NonFiniteLogits`, before any other read. A -inf logit is a masked id and stays valid.
//! After the check, the row maximum is finite, so the asserts on the weight sums hold.
//!
//! Pool selection. `samplePool` stays `noinline`. Its selection loop is branchy, and inlined into a
//! caller it measured 107 us or 155 us per Qwen row with the caller layout alone changed. Out of
//! line it measured 107 us in both layouts.

const std = @import("std");
const prop = @import("../core/prop.zig");
const limits = @import("../core/limits.zig");
const span = @import("../core/span.zig");
const exp_simd = @import("../ops/exp_simd.zig");

const assert = std.debug.assert;
const lanes = exp_simd.lanes;
const Vec = exp_simd.Vec;

// -- constants --------------------------------------------------------------

/// Weights per block sum of the full-vocab draw.
const weight_block = 64;

/// The weight bits of one radix level. A weight in [0, 1] has bits below 2^30, so three levels
/// read every bit.
const radix_bits = 10;

const radix_buckets = 1 << radix_bits;
const radix_levels = 3;

/// The histogram copies of one radix level. Four copies break the add chains of a shared bucket.
const radix_ways = 4;

/// The most ids the cut sorts. A larger boundary bucket takes the next radix level.
const cut_candidates_max = 1024;

comptime {
    assert(weight_block % lanes == 0);
    assert(@as(u32, @bitCast(@as(f32, 1))) < 1 << (radix_bits * radix_levels));
}

// -- types ------------------------------------------------------------------

/// The refusal of a logits row that holds a NaN, a +inf, or no finite logit.
pub const Error = error{NonFiniteLogits};

/// Per-session sampling settings. The zero value is greedy decoding, so the
/// default keeps every determinism gate.
pub const Params = struct {
    /// 0 = greedy (argmax, no RNG draw). Higher flattens.
    temperature: f32 = 0,
    /// Candidate pool size. 0 = no cap, and every cut runs over the full vocab.
    /// At most `limits.max_top_k`.
    top_k: u32 = 0,
    /// Nucleus cut: the mass share to keep. 1.0 = off. Must be in (0, 1].
    top_p: f32 = 1.0,
    /// Keep the ids with at least min_p times the largest probability. 0 = off. Must be in [0, 1].
    min_p: f32 = 0,
    /// Subtracted from the logit of every id in `presence`. 0 is off.
    presence_penalty: f32 = 0,
    /// One bit per vocab id for the presence penalty. Empty is none. The caller owns and marks it.
    presence: []const u64 = &.{},
    /// The repetition factor for every id in `context`. 1 is off.
    /// In (0, `limits.max_repetition_penalty`].
    repetition_penalty: f32 = 1,
    /// One bit per vocab id for the repetition penalty. Empty is none. The caller owns it.
    context: []const u64 = &.{},

    fn penalized(p: Params) bool {
        if (p.presence_penalty != 0 and p.presence.len > 0) return true;
        return p.repetition_penalty != 1 and p.context.len > 0;
    }

    fn penaltySets(p: Params) PenaltySets {
        return .{ .seen = p.presence, .context = p.context };
    }

    fn penalties(p: Params) Penalties {
        return .{ .presence = p.presence_penalty, .repetition = p.repetition_penalty };
    }
};

/// The slot state that `penalize` reads, borrowed for one draw. A new penalty adds a field here.
/// Each set is empty or holds `setWords(vocab)` words.
pub const PenaltySets = struct {
    /// The ids accepted since the turn began.
    seen: []const u64 = &.{},
    /// The ids fed to the session since it opened.
    context: []const u64 = &.{},
};

/// The penalty settings of one request. The zero value is neutral.
pub const Penalties = struct {
    /// Subtracted from the logit of every id in `PenaltySets.seen`.
    presence: f32 = 0,
    /// The factor of the repetition rule for every id in `PenaltySets.context`.
    repetition: f32 = 1,
};

const Candidate = struct {
    v: f32,
    i: u32,

    /// Descending by value. Ties resolve to the lower index.
    fn moreLikely(_: void, a: Candidate, b: Candidate) bool {
        if (a.v != b.v) return a.v > b.v;
        return a.i < b.i;
    }
};

/// The full-vocab cut. Every weight above `weight` survives, and so do the
/// first `ties` weights equal to it, in id order.
const Cut = struct {
    weight: f32,
    ties: u32,
};

/// The pool cut: the first `n` candidates survive, with the mass `total`.
const PoolCut = struct {
    n: usize,
    total: f32,
};

const Keys = @Vector(lanes, u32);
const Wide = @Vector(lanes / 2, f64);

// -- api --------------------------------------------------------------------

/// The words of an id set over `vocab` ids.
pub fn setWords(vocab: usize) usize {
    return std.math.divCeil(usize, vocab, 64) catch unreachable;
}

/// Mark `id` in an id set over `vocab` ids.
pub fn markId(set: []u64, vocab: usize, id: u32) void {
    assert(set.len == setWords(vocab));
    assert(id < vocab);

    set[id / 64] |= @as(u64, 1) << @intCast(id % 64);
}

/// Mark every id of `ids` in an id set over `vocab` ids.
pub fn markIds(set: []u64, vocab: usize, ids: []const u32) void {
    for (ids) |id| {
        markId(set, vocab, id);
    }
}

/// Whether an id set holds `id`.
pub fn isMarked(set: []const u64, id: u32) bool {
    assert(id / 64 < set.len);

    return set[id / 64] & (@as(u64, 1) << @intCast(id % 64)) != 0;
}

/// The f32 count of the scratch that `sample` takes for a `vocab`-id row: one row, then one sum
/// per weight block.
pub fn scratchLen(vocab: usize) usize {
    return vocab + blockCount(vocab);
}

/// Index of the highest-scoring logit.
pub fn greedy(logits: []const f32) u32 {
    return argmax(logits);
}

/// Draw one token. The same logits, params, and RNG state give the same token on one build.
/// `scratch` holds `scratchLen(logits.len)` floats that the draw can overwrite.
pub fn sample(logits: []const f32, params: Params, random: std.Random, scratch: []f32) Error!u32 {
    const sp = span.begin(.sample);
    defer sp.end();

    assert(logits.len > 0);
    try checkLogits(logits);
    assert(params.temperature >= 0);
    assert(params.top_p > 0);
    assert(params.top_p <= 1);
    assert(params.min_p >= 0);
    assert(params.min_p <= 1);
    assert(params.top_k <= limits.max_top_k);
    assert(std.math.isFinite(params.presence_penalty));
    assert(params.repetition_penalty > 0);
    assert(params.repetition_penalty <= limits.max_repetition_penalty);
    if (params.presence.len > 0) assert(params.presence.len == setWords(logits.len));
    if (params.context.len > 0) assert(params.context.len == setWords(logits.len));

    assert(scratch.len == scratchLen(logits.len));

    const row: []const f32 = if (params.penalized()) fixed: {
        const target = scratch[0..logits.len];
        penalize(logits, target, params.penaltySets(), params.penalties());
        break :fixed target;
    } else logits;

    const id = pick(row, params, random, scratch);
    assert(id < logits.len);
    return id;
}

/// Refuse a row that holds a NaN or a +inf, or that holds no logit above -inf. The loop runs
/// `row.len / lanes` times, then once per tail element.
pub fn checkLogits(row: []const f32) Error!void {
    assert(row.len > 0);

    const inf: Vec = @splat(std.math.inf(f32));
    const zero: Vec = @splat(0);
    const one: Vec = @splat(1);
    var bad: Vec = zero;
    var m_v: Vec = -inf;
    var i: usize = 0;
    while (i + lanes <= row.len) : (i += lanes) {
        const v: Vec = row[i..][0..lanes].*;
        bad += @select(f32, v != v, one, zero) + @select(f32, v == inf, one, zero);
        m_v = @max(m_v, v);
    }

    var flagged = @reduce(.Add, bad) != 0;
    var m = @reduce(.Max, m_v);
    while (i < row.len) : (i += 1) {
        const x = row[i];
        flagged = flagged or std.math.isNan(x) or x == std.math.inf(f32);
        m = @max(m, x);
    }

    if (flagged or m == -std.math.inf(f32)) return error.NonFiniteLogits;
}

/// Copy `source` into `target` and apply the penalties to every id that the sets mark.
pub fn penalize(source: []const f32, target: []f32, sets: PenaltySets, knobs: Penalties) void {
    assert(source.len == target.len);
    assert(source.len > 0);

    // Check the sets and the settings.
    const words = setWords(source.len);
    assert(sets.seen.len == 0 or sets.seen.len == words);
    assert(sets.context.len == 0 or sets.context.len == words);
    assert(std.math.isFinite(knobs.presence));
    assert(knobs.repetition > 0);
    assert(knobs.repetition <= limits.max_repetition_penalty);

    // Copy the row.
    @memcpy(target, source);
    const seen: []const u64 = if (knobs.presence != 0) sets.seen else &.{};
    const context: []const u64 = if (knobs.repetition != 1) sets.context else &.{};

    // Walk the marked ids word by word.
    for (0..words) |w| {
        const s = if (w < seen.len) seen[w] else 0;
        const c = if (w < context.len) context[w] else 0;
        var bits = (s | c) & validBits(w, source.len);
        while (bits != 0) : (bits &= bits - 1) {
            const bit = @as(u64, 1) << @intCast(@ctz(bits));
            const i = w * 64 + @ctz(bits);
            assert(i < source.len);
            target[i] = penalizeOne(source[i], c & bit != 0, s & bit != 0, knobs);
        }
    }
}

// -- helpers ----------------------------------------------------------------

fn blockCount(vocab: usize) usize {
    return std.math.divCeil(usize, vocab, weight_block) catch unreachable;
}

/// The bits of word `w` that name ids below `vocab`.
fn validBits(w: usize, vocab: usize) u64 {
    const first = w * 64;
    assert(first < vocab);
    if (vocab - first >= 64) return std.math.maxInt(u64);

    return (@as(u64, 1) << @intCast(vocab - first)) - 1;
}

/// One marked logit: the repetition rule, then the presence penalty, then the clamp.
fn penalizeOne(l: f32, in_context: bool, in_seen: bool, knobs: Penalties) f32 {
    var v = l;
    if (in_context) {
        v = if (v > 0) v / knobs.repetition else v * knobs.repetition;
    }

    if (in_seen) {
        v -= knobs.presence;
    }

    const max = std.math.floatMax(f32);
    // The clamp alone sends a NaN to `floatMax`, the top of the row.
    if (std.math.isNan(v)) return -max;

    return std.math.clamp(v, -max, max);
}

fn pick(row: []const f32, params: Params, random: std.Random, scratch: []f32) u32 {
    if (params.temperature == 0) return argmax(row);
    if (params.top_k > 0) return samplePool(row, params, random);
    if (params.top_p == 1 and params.min_p == 0) {
        return sampleFullVocab(row, params.temperature, random, scratch);
    }

    return sampleFullCut(row, params, random, scratch);
}

/// The first index of the highest value. The loop runs `row.len` times.
fn argmax(row: []const f32) u32 {
    assert(row.len > 0);

    var best: usize = 0;
    var best_v: f32 = row[0];
    var i: usize = 1;
    while (i < row.len) : (i += 1) {
        const l = row[i];
        if (l > best_v) {
            best_v = l;
            best = i;
        }
    }

    return @intCast(best);
}

/// Temperature-only draw over the whole vocab: one weight pass, then a CDF walk. `row` can alias
/// the row part of `scratch`, because the weight pass reads each element before it writes it.
fn sampleFullVocab(row: []const f32, temperature: f32, random: std.Random, scratch: []f32) u32 {
    assert(temperature > 0);

    const n = row.len;
    const weights = scratch[0..n];
    const sums = scratch[n..][0..blockCount(n)];
    const total = fullVocabWeights(row, temperature, weights, sums);
    return drawWeights(weights, sums, total, random);
}

/// The draw over the full vocab with an exact min_p or top_p cut.
fn sampleFullCut(row: []const f32, params: Params, random: std.Random, scratch: []f32) u32 {
    const n = row.len;
    const total = fullCutWeights(row, params, scratch);
    return drawWeights(scratch[0..n], scratch[n..][0..blockCount(n)], total, random);
}

/// Write the full-vocab weights into the scratch, zero every weight that the min_p and top_p cuts
/// drop, and return the sum of the block sums. The radix select reads the vocab at most
/// `radix_levels` times.
fn fullCutWeights(row: []const f32, params: Params, scratch: []f32) f32 {
    assert(params.temperature > 0);
    assert(params.top_k == 0);
    assert(params.top_p < 1 or params.min_p > 0);

    const n = row.len;
    const weights = scratch[0..n];
    const sums = scratch[n..][0..blockCount(n)];
    const all = fullVocabWeights(row, params.temperature, weights, sums);
    // The largest weight is 1, so min_p is a floor. With no min_p the floor drops zero weights.
    const floor: f32 = if (params.min_p > 0) params.min_p else std.math.floatTrueMin(f32);
    // A lower bound on the floor set mass. The f32 block sums err by far less than the 1 % margin.
    const floor_mass: f32 = if (params.min_p > 0) 1 else @max(1, all * 0.99);
    const cut: Cut = if (params.top_p < 1)
        selectCut(weights, floor, floor_mass, params.top_p)
    else
        .{ .weight = floor, .ties = std.math.maxInt(u32) };

    const total = applyCut(weights, sums, cut);
    return total;
}

/// One CDF draw over the weights, block sum by block sum, then inside one block.
/// `total` is the sum of the block sums in block order.
fn drawWeights(weights: []const f32, sums: []const f32, total: f32, random: std.Random) u32 {
    // The term at the maximum is exactly 1, and every cut keeps it.
    assert(total >= 1);

    const r = random.float(f32) * total;
    var acc: f32 = 0;
    for (sums, 0..) |sum, b| {
        // No earlier block passed `r`, so `acc <= r`, and this block holds a
        // positive weight.
        if (acc + sum > r) return drawInBlock(weights, b, acc, r);
        acc += sum;
    }

    // f32 rounding can leave the sum of the block sums at or below `r`. The
    // draw then takes the last id with a positive weight.
    var b = sums.len;
    while (b > 0) {
        b -= 1;
        if (sums[b] > 0) return lastPositive(weights, b);
    }

    unreachable;
}

/// Write `exp((l - max) / T)` for every id into `weights`, and the sum of each block of
/// `weight_block` weights into `sums`. Returns the sum of the block sums, in block order.
fn fullVocabWeights(row: []const f32, temperature: f32, weights: []f32, sums: []f32) f32 {
    assert(weights.len == row.len);
    assert(sums.len == blockCount(row.len));

    // Find the row maximum.
    const m = rowMax(row);
    const m_v: Vec = @splat(m);
    const t_v: Vec = @splat(temperature);

    // Write the weights and the sum of each block.
    var total: f32 = 0;
    for (sums, 0..) |*sum, b| {
        const start = b * weight_block;
        const end = @min(start + weight_block, row.len);
        var acc: Vec = @splat(0);
        var i = start;
        while (i + lanes <= end) : (i += lanes) {
            const w = exp_simd.expNeg((@as(Vec, row[i..][0..lanes].*) - m_v) / t_v);
            weights[i..][0..lanes].* = w;
            acc += w;
        }

        if (i < end) {
            // A pad of -inf gives a weight of 0, so the short tail adds nothing.
            var pad: [lanes]f32 = @splat(-std.math.inf(f32));
            @memcpy(pad[0 .. end - i], row[i..end]);
            const w: [lanes]f32 = exp_simd.expNeg((@as(Vec, pad) - m_v) / t_v);
            @memcpy(weights[i..end], w[0 .. end - i]);
            acc += w;
        }

        sum.* = @reduce(.Add, acc);
        total += sum.*;
    }

    assert(std.math.isFinite(total));
    return total;
}

/// The largest value of a finite row.
fn rowMax(row: []const f32) f32 {
    assert(row.len > 0);

    var m_v: Vec = @splat(-std.math.inf(f32));
    var i: usize = 0;
    while (i + lanes <= row.len) : (i += lanes) {
        m_v = @max(m_v, @as(Vec, row[i..][0..lanes].*));
    }

    var m = @reduce(.Max, m_v);
    while (i < row.len) : (i += 1) {
        m = @max(m, row[i]);
    }

    assert(std.math.isFinite(m));
    return m;
}

/// The exact top_p cut among the weights at or above `floor`: the smallest ranked prefix whose
/// mass reaches `top_p` of their mass. `floor_mass` is at least 1.
fn selectCut(weights: []const f32, floor: f32, floor_mass: f32, top_p: f32) Cut {
    assert(floor > 0);
    assert(floor_mass >= 1);
    assert(top_p > 0);
    assert(top_p < 1);

    const n: f64 = @floatFromInt(weights.len);
    const skip: f32 = @floatCast((1 - @as(f64, top_p)) * floor_mass / (2 * n));
    const scan_key = @max(@as(u32, @bitCast(floor)), @as(u32, @bitCast(skip)) + 1);

    // Narrow the boundary bucket one radix level at a time.
    var mass: [radix_buckets]f64 = undefined;
    var count: [radix_buckets]u32 = undefined;
    var prefix: u32 = 0;
    var above: f64 = 0;
    var target: f64 = 0;
    for (0..radix_levels) |level| {
        const shift: u5 = @intCast(radix_bits * (radix_levels - 1 - level));
        const total = radixLevel(
            weights,
            floor,
            scan_key,
            shift,
            prefix,
            &mass,
            &count,
            level == 0,
        );
        if (level == 0) {
            // The largest weight is 1, and the floor is at most 1.
            assert(total >= 1);

            target = top_p * total;
        }

        const bucket = boundaryBucket(&mass, &count, &above, target);
        prefix = prefix << radix_bits | bucket;
        if (level + 1 == radix_levels) {
            return tieCut(@bitCast(prefix), count[bucket], above, target);
        }
        if (count[bucket] <= cut_candidates_max) {
            return sortedCut(weights, scan_key, prefix, shift, above, target);
        }
    }

    unreachable;
}

/// One radix level: the mass and count of each field value at `shift`, over the weights at or
/// above `scan_key` whose higher bits equal `prefix`. With `with_total` it also returns the f64
/// mass of every weight at or above `floor`. A vector compare skips a lane group with no hit.
fn radixLevel(
    weights: []const f32,
    floor: f32,
    scan_key: u32,
    shift: u5,
    prefix: u32,
    mass: *[radix_buckets]f64,
    count: *[radix_buckets]u32,
    with_total: bool,
) f64 {
    // Each lane adds into the copy of its index modulo `radix_ways`, so adds do not chain.
    var mass_ways: [radix_ways][radix_buckets]f64 = @splat(@splat(0));
    var count_ways: [radix_ways][radix_buckets]u32 = @splat(@splat(0));
    const floor_v: Vec = @splat(floor);
    const scan_v: Vec = @splat(@as(f32, @bitCast(scan_key)));
    const up: @Vector(lanes, u5) = @splat(shift + radix_bits);
    const sh: @Vector(lanes, u5) = @splat(shift);
    const prefix_v: Keys = @splat(prefix);
    const field_v: Keys = @splat(radix_buckets - 1);
    const zero: Vec = @splat(0);
    const none: @Vector(lanes, bool) = @splat(false);
    var lo: Wide = @splat(0);
    var hi: Wide = @splat(0);

    // Accumulate the lane groups into the histogram copies.
    var i: usize = 0;
    while (i + lanes <= weights.len) : (i += lanes) {
        const w: Vec = weights[i..][0..lanes].*;
        if (with_total) {
            const e = @select(f32, w >= floor_v, w, zero);
            lo += @as(Wide, @floatCast(@shuffle(f32, e, undefined, [4]i32{ 0, 1, 2, 3 })));
            hi += @as(Wide, @floatCast(@shuffle(f32, e, undefined, [4]i32{ 4, 5, 6, 7 })));
        }

        const keys: Keys = @bitCast(w);
        const hot = @select(bool, w >= scan_v, keys >> up == prefix_v, none);
        if (!@reduce(.Or, hot)) continue;

        // A lane outside the level adds 0 to a valid bucket, so the adds need
        // no branch per lane.
        const add = @select(f32, hot, w, zero);
        const one: Keys = @intFromBool(hot);
        const d = (keys >> sh) & field_v;
        inline for (0..lanes) |l| {
            mass_ways[l % radix_ways][d[l]] += add[l];
            count_ways[l % radix_ways][d[l]] += one[l];
        }
    }

    // Accumulate the scalar tail.
    var total: f64 = @reduce(.Add, lo) + @reduce(.Add, hi);
    while (i < weights.len) : (i += 1) {
        const w = weights[i];
        if (with_total and w >= floor) {
            total += w;
        }

        const key: u32 = @bitCast(w);
        if (key < scan_key or key >> shift >> radix_bits != prefix) continue;

        const d = (key >> shift) & (radix_buckets - 1);
        mass_ways[0][d] += w;
        count_ways[0][d] += 1;
    }

    // Merge the copies.
    for (mass, count, 0..) |*m, *c, d| {
        m.* = 0;
        c.* = 0;
        for (0..radix_ways) |way| {
            m.* += mass_ways[way][d];
            c.* += count_ways[way][d];
        }
    }

    return total;
}

/// The highest bucket whose mass, added to `above.*`, reaches `target`. Higher buckets add their
/// mass to `above.*`. If f64 rounding leaves the walk short, the lowest nonempty bucket wins.
fn boundaryBucket(
    mass: *const [radix_buckets]f64,
    count: *const [radix_buckets]u32,
    above: *f64,
    target: f64,
) u32 {
    var lowest: ?u32 = null;
    var d: u32 = radix_buckets;
    while (d > 0) {
        d -= 1;
        if (count[d] == 0) continue;
        if (above.* + mass[d] >= target) return d;

        above.* += mass[d];
        lowest = d;
    }

    const last = lowest.?;
    above.* -= mass[last];
    return last;
}

/// The cut inside a boundary bucket of at most `cut_candidates_max` ids: sort it, then walk it.
fn sortedCut(
    weights: []const f32,
    scan_key: u32,
    prefix: u32,
    shift: u5,
    above: f64,
    target: f64,
) Cut {
    // Collect the bucket ids.
    var cand: [cut_candidates_max]Candidate = undefined;
    var n: usize = 0;
    const scan_v: Vec = @splat(@as(f32, @bitCast(scan_key)));
    const sh: @Vector(lanes, u5) = @splat(shift);
    const prefix_v: Keys = @splat(prefix);
    const none: @Vector(lanes, bool) = @splat(false);
    var i: usize = 0;
    while (i + lanes <= weights.len) : (i += lanes) {
        const w: Vec = weights[i..][0..lanes].*;
        const keys: Keys = @bitCast(w);
        const hot = @select(bool, w >= scan_v, keys >> sh == prefix_v, none);
        if (!@reduce(.Or, hot)) continue;

        inline for (0..lanes) |l| {
            if (hot[l]) {
                assert(n < cand.len);

                cand[n] = .{ .v = w[l], .i = @intCast(i + l) };
                n += 1;
            }
        }
    }

    while (i < weights.len) : (i += 1) {
        const key: u32 = @bitCast(weights[i]);
        if (key < scan_key or key >> shift != prefix) continue;

        assert(n < cand.len);
        cand[n] = .{ .v = weights[i], .i = @intCast(i) };
        n += 1;
    }

    // Sort the bucket and walk its mass up to the target.
    assert(n >= 1);
    std.mem.sort(Candidate, cand[0..n], {}, Candidate.moreLikely);
    var j: usize = 0;
    var acc = above + cand[0].v;
    while (acc < target and j + 1 < n) {
        j += 1;
        acc += cand[j].v;
    }

    // Count the ties that the cut keeps.
    const weight = cand[j].v;
    var ties: u32 = 0;
    for (cand[0 .. j + 1]) |c| {
        if (c.v == weight) {
            ties += 1;
        }
    }

    assert(ties >= 1);
    return .{ .weight = weight, .ties = ties };
}

/// The cut inside a bucket of `n` equal weights: the ties, in id order, that reach the target.
fn tieCut(weight: f32, n: u32, above: f64, target: f64) Cut {
    assert(n >= 1);
    assert(weight > 0);

    var ties: u32 = 1;
    var acc = above + weight;
    while (acc < target and ties < n) {
        ties += 1;
        acc += weight;
    }

    return .{ .weight = weight, .ties = ties };
}

/// Zero every weight that `cut` drops and rewrite the block sums. Returns the sum of the block
/// sums, in block order. A second pass zeroes the surplus ties in id order.
fn applyCut(weights: []f32, sums: []f32, cut: Cut) f32 {
    assert(sums.len == blockCount(weights.len));
    assert(cut.weight > 0);
    assert(cut.ties >= 1);

    // Keep every weight at or above the cut, and count the ties.
    const t_v: Vec = @splat(cut.weight);
    const zero: Vec = @splat(0);
    var tied: usize = 0;
    for (sums, 0..) |*sum, b| {
        const start = b * weight_block;
        const end = @min(start + weight_block, weights.len);
        var acc: Vec = @splat(0);
        var i = start;
        while (i + lanes <= end) : (i += lanes) {
            const w: Vec = weights[i..][0..lanes].*;
            const kept = @select(f32, w >= t_v, w, zero);
            weights[i..][0..lanes].* = kept;
            acc += kept;
            tied += std.simd.countTrues(w == t_v);
        }

        var s = @reduce(.Add, acc);
        while (i < end) : (i += 1) {
            if (weights[i] == cut.weight) {
                tied += 1;
            }

            if (weights[i] < cut.weight) {
                weights[i] = 0;
            }

            s += weights[i];
        }

        sum.* = s;
    }

    // Zero the surplus ties in id order and sum the blocks again.
    if (tied > cut.ties) {
        var kept_ties: u32 = 0;
        for (weights) |*w| {
            if (w.* != cut.weight) continue;
            if (kept_ties < cut.ties) {
                kept_ties += 1;
            } else {
                w.* = 0;
            }
        }

        assert(kept_ties == cut.ties);
        for (sums, 0..) |*sum, b| {
            const start = b * weight_block;
            var s: f32 = 0;
            for (weights[start..@min(start + weight_block, weights.len)]) |w| {
                s += w;
            }

            sum.* = s;
        }
    }

    // Sum the blocks in block order.
    var total: f32 = 0;
    for (sums) |s| {
        total += s;
    }

    assert(std.math.isFinite(total));
    return total;
}

/// The CDF walk inside block `b`, from `acc` up to `r`. A zero weight never
/// moves `acc` past `r`, so the walk returns a positive weight.
fn drawInBlock(weights: []const f32, b: usize, acc_start: f32, r: f32) u32 {
    const start = b * weight_block;
    const end = @min(start + weight_block, weights.len);
    var acc = acc_start;
    for (weights[start..end], start..) |w, i| {
        acc += w;
        if (acc > r) return @intCast(i);
    }

    // f32 rounding can leave the weights of the block short of its vector sum.
    return lastPositive(weights, b);
}

/// The last id of block `b` with a positive weight. The block must hold one.
fn lastPositive(weights: []const f32, b: usize) u32 {
    const start = b * weight_block;
    var i = @min(start + weight_block, weights.len);
    while (i > start) {
        i -= 1;
        if (weights[i] > 0) return @intCast(i);
    }

    unreachable;
}

/// Pool draw: the top `top_k` candidates as weights, the min_p and nucleus cuts, then one CDF draw
/// within the cut.
noinline fn samplePool(row: []const f32, params: Params, random: std.Random) u32 {
    var pool: [limits.max_top_k]Candidate = undefined;
    var probs: [limits.max_top_k]f32 = undefined;
    const k = poolWeights(row, params.top_k, params.temperature, &pool, &probs);
    const cut = poolCut(probs[0..k], params.min_p, params.top_p);
    const r = random.float(f32) * cut.total;
    var acc: f32 = 0;
    for (probs[0..cut.n], 0..) |p, j| {
        acc += p;
        if (acc > r) return pool[j].i;
    }

    return pool[cut.n - 1].i;
}

/// Select the top `min(top_k, V)` candidates with a bounded min-heap (O(V log k) worst case), sort
/// them descending, and write the weight `exp((l - max) / T)` of each. The first weight is
/// exactly 1. Returns the candidate count.
fn poolWeights(
    row: []const f32,
    top_k: u32,
    temperature: f32,
    pool: *[limits.max_top_k]Candidate,
    probs: *[limits.max_top_k]f32,
) usize {
    assert(top_k >= 1);
    assert(temperature > 0);

    const k = @min(top_k, row.len);
    assert(k <= limits.max_top_k);
    var n: usize = 0;
    for (row, 0..) |v, i| {
        const c: Candidate = .{ .v = v, .i = @intCast(i) };
        if (n < k) {
            pool[n] = c;
            n += 1;
            if (n == k) {
                buildMinHeap(pool[0..k]);
            }
        } else if (c.v > pool[0].v) {
            pool[0] = c;
            siftDown(pool[0..k], 0);
        }
    }

    assert(n == k);
    std.mem.sort(Candidate, pool[0..k], {}, Candidate.moreLikely);
    const m = pool[0].v;
    for (pool[0..k], 0..) |c, j| {
        probs[j] = @exp((c.v - m) / temperature);
    }

    assert(probs[0] == 1);
    return k;
}

/// The min_p cut, then the nucleus cut, over pool weights in rank order. The first weight is 1,
/// so min_p keeps the ranked prefix at or above `min_p`. The first candidate always survives.
fn poolCut(probs: []const f32, min_p: f32, top_p: f32) PoolCut {
    assert(probs.len >= 1);
    assert(probs[0] == 1);

    var n = probs.len;
    if (min_p > 0) {
        n = 1;
        while (n < probs.len and probs[n] >= min_p) {
            n += 1;
        }
    }

    var total: f32 = 0;
    for (probs[0..n]) |p| {
        total += p;
    }

    assert(total > 0);
    if (top_p < 1) {
        var acc: f32 = 0;
        for (probs[0..n], 0..) |p, j| {
            acc += p;
            if (acc >= top_p * total) {
                n = j + 1;
                break;
            }
        }

        total = acc;
    }

    assert(n >= 1);
    return .{ .n = n, .total = total };
}

fn buildMinHeap(items: []Candidate) void {
    assert(items.len >= 1);

    var i = items.len / 2;
    while (i > 0) {
        i -= 1;
        siftDown(items, i);
    }
}

fn siftDown(items: []Candidate, start: usize) void {
    assert(start < items.len);

    var i = start;
    while (true) {
        const l = 2 * i + 1;
        const rgt = 2 * i + 2;
        var smallest = i;
        if (l < items.len and items[l].v < items[smallest].v) {
            smallest = l;
        }

        if (rgt < items.len and items[rgt].v < items[smallest].v) {
            smallest = rgt;
        }

        if (smallest == i) return;

        std.mem.swap(Candidate, &items[i], &items[smallest]);
        i = smallest;
    }
}

// -- tests ------------------------------------------------------------------

/// A scratch for a row of `vocab` ids.
fn TestScratch(comptime vocab: usize) type {
    return [scratchLen(vocab)]f32;
}

test "sample refuses a row with a NaN, a +inf, or no finite logit, on every path" {
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    const rows = [_][4]f32{
        .{ 1, nan, 0, 2 },
        .{ 1, inf, 0, 2 },
        .{ -inf, -inf, -inf, -inf },
    };
    const shapes = [_]Params{
        .{},
        .{ .temperature = 1 },
        .{ .temperature = 1, .top_k = 2 },
        .{ .temperature = 1, .top_p = 0.5 },
    };
    var scratch: TestScratch(4) = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    for (rows) |row| {
        for (shapes) |p| {
            try std.testing.expectError(
                error.NonFiniteLogits,
                sample(&row, p, prng.random(), &scratch),
            );
        }
    }

    // A masked id at -inf stays a valid logit.
    const masked = [_]f32{ 1, -inf, 0, 2 };
    try std.testing.expectEqual(@as(u32, 3), try sample(&masked, .{}, prng.random(), &scratch));
}

test "greedy picks the max index" {
    const x = [_]f32{ -1, 5, 3, 5.1, 0 };
    try std.testing.expectEqual(@as(u32, 3), greedy(&x));
}

test "sample with temperature 0 is greedy and never draws" {
    const x = [_]f32{ 0.5, -2, 7, 3 };
    var scratch: TestScratch(x.len) = undefined;
    var prng = std.Random.DefaultPrng.init(42);
    const before = prng;
    const id = try sample(&x, .{}, prng.random(), &scratch);
    try std.testing.expectEqual(greedy(&x), id);
    // `sample` leaves the RNG state untouched, so a greedy session draws nothing.
    try std.testing.expectEqual(before.s, prng.s);
}

test "sample with top_k 1 is greedy at any temperature" {
    const x = [_]f32{ 1, 9, 2, 8.9, -4 };
    var scratch: TestScratch(x.len) = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    for (0..16) |_| {
        const id = try sample(&x, .{ .temperature = 1.3, .top_k = 1 }, prng.random(), &scratch);
        try std.testing.expectEqual(greedy(&x), id);
    }
}

test "top_k restricts draws to the k most likely tokens" {
    const x = [_]f32{ 10, 9.5, -20, -20, -20, -20 };
    var scratch: TestScratch(x.len) = undefined;
    var prng = std.Random.DefaultPrng.init(99);
    for (0..64) |_| {
        const id = try sample(&x, .{ .temperature = 1.0, .top_k = 2 }, prng.random(), &scratch);
        try std.testing.expect(id == 0 or id == 1);
    }
}

test "tiny top_p collapses to the most likely token" {
    const x = [_]f32{ 3, 2.5, 2.4, 2.3 };
    var scratch: TestScratch(x.len) = undefined;
    var prng = std.Random.DefaultPrng.init(5);
    for (0..32) |_| {
        const id = try sample(&x, .{ .temperature = 1.0, .top_p = 0.01 }, prng.random(), &scratch);
        try std.testing.expectEqual(@as(u32, 0), id);
        const pooled = try sample(
            &x,
            .{ .temperature = 1.0, .top_k = 4, .top_p = 0.01 },
            prng.random(),
            &scratch,
        );
        try std.testing.expectEqual(@as(u32, 0), pooled);
    }
}

test "full-vocab temperature sampling overwhelmingly picks a dominant logit" {
    const x = [_]f32{ 0, 12, 0, 0 };
    var scratch: TestScratch(x.len) = undefined;
    var prng = std.Random.DefaultPrng.init(2026);
    var hits: u32 = 0;
    for (0..100) |_| {
        if (try sample(&x, .{ .temperature = 1.0 }, prng.random(), &scratch) == 1) {
            hits += 1;
        }
    }

    try std.testing.expect(hits >= 95);
}

/// An id set over `vocab` ids with `ids` marked.
fn testSet(comptime vocab: usize, ids: []const u32) [setWords(vocab)]u64 {
    var set: [setWords(vocab)]u64 = @splat(0);
    markIds(&set, vocab, ids);
    return set;
}

test "presence penalty: greedy moves off a present id and leaves the logits" {
    const x = [_]f32{ 1, 5, 4.5, 0 };
    const before = x;
    var scratch: TestScratch(x.len) = undefined;
    const set = testSet(x.len, &.{1});
    var prng = std.Random.DefaultPrng.init(1);
    // 5 - 1 = 4 < 4.5, so the argmax moves to id 2. A smaller penalty keeps 1.
    try std.testing.expectEqual(
        @as(u32, 2),
        try sample(&x, .{ .presence_penalty = 1, .presence = &set }, prng.random(), &scratch),
    );
    try std.testing.expectEqual(
        @as(u32, 1),
        try sample(&x, .{ .presence_penalty = 0.25, .presence = &set }, prng.random(), &scratch),
    );
    // A negative penalty favors a present id: 4.5 + 2 passes 5.
    const low = testSet(x.len, &.{2});
    try std.testing.expectEqual(
        @as(u32, 2),
        try sample(&x, .{ .presence_penalty = -2, .presence = &low }, prng.random(), &scratch),
    );
    try std.testing.expectEqualSlices(f32, &before, &x);
}

test "presence penalty: the pool and the full vocab never draw a heavily penalized id" {
    const x = [_]f32{ 10, 9.9, -5, -5, -5, -5 };
    var scratch: TestScratch(x.len) = undefined;
    const set = testSet(x.len, &.{0});
    var prng = std.Random.DefaultPrng.init(77);
    for (0..64) |_| {
        // 10 - 2 = 8 is still a candidate in a pool of two. At low temperature id 1 holds the mass.
        const pooled = try sample(
            &x,
            .{ .temperature = 0.05, .top_k = 2, .presence_penalty = 2, .presence = &set },
            prng.random(),
            &scratch,
        );
        try std.testing.expectEqual(@as(u32, 1), pooled);
        const full = try sample(
            &x,
            .{ .temperature = 0.05, .presence_penalty = 2, .presence = &set },
            prng.random(),
            &scratch,
        );
        try std.testing.expectEqual(@as(u32, 1), full);
    }
}

test "neutral penalties equal the plain path draw for draw" {
    var x: [100]f32 = undefined;
    for (&x, 0..) |*v, i| {
        v.* = @cos(@as(f32, @floatFromInt(i)) * 0.7) * 3;
    }

    var scratch: TestScratch(x.len) = undefined;
    const set = testSet(x.len, &.{ 0, 5, 17, 99 });
    const shapes = [_]Params{
        .{},
        .{ .temperature = 0.9 },
        .{ .temperature = 0.9, .top_k = 20, .top_p = 0.95 },
        .{ .temperature = 0.9, .top_p = 0.9, .min_p = 0.1 },
    };
    for (shapes) |base| {
        var zero = base;
        zero.presence = &set;
        zero.context = &set;
        var empty = base;
        empty.presence_penalty = 1.5;
        empty.repetition_penalty = 1.5;
        var a = std.Random.DefaultPrng.init(9);
        var b = std.Random.DefaultPrng.init(9);
        var c = std.Random.DefaultPrng.init(9);
        for (0..32) |_| {
            const want = try sample(&x, base, a.random(), &scratch);
            try std.testing.expectEqual(want, try sample(&x, zero, b.random(), &scratch));
            try std.testing.expectEqual(want, try sample(&x, empty, c.random(), &scratch));
        }
    }
}

// The property tests below run over seeded random rows. The row size is off the lane grid and
// off the weight block grid, so every tail path runs.

const prop_vocab = 1000 + 3;
const prop_rounds = 64;

fn fillRow(r: std.Random, row: []f32) void {
    for (row) |*v| {
        v.* = r.floatNorm(f32) * 4;
    }
}

fn fillSet(r: std.Random, set: []u64, vocab: usize) void {
    @memset(set, 0);
    const n = r.uintLessThan(usize, vocab / 4);
    for (0..n) |_| {
        markId(set, vocab, r.uintLessThan(u32, @intCast(vocab)));
    }
}

/// Every path of the chain, with and without each penalty.
fn propShapes(ids: []const u64) [11]Params {
    const everything: Params = .{
        .temperature = 1,
        .top_p = 0.8,
        .min_p = 0.02,
        .presence_penalty = 1,
        .presence = ids,
        .repetition_penalty = 1.5,
        .context = ids,
    };
    const pool_presence: Params = .{
        .temperature = 1,
        .top_k = 20,
        .top_p = 0.95,
        .presence_penalty = -0.5,
        .presence = ids,
    };
    const pool_repetition: Params = .{
        .temperature = 1,
        .top_k = 40,
        .min_p = 0.05,
        .repetition_penalty = 1.2,
        .context = ids,
    };
    return .{
        .{},
        .{ .presence_penalty = 1.5, .presence = ids },
        .{ .repetition_penalty = 1.3, .context = ids },
        .{ .temperature = 0.8 },
        .{ .temperature = 0.8, .presence_penalty = 1.5, .presence = ids },
        .{ .temperature = 1, .top_k = 20, .top_p = 0.95 },
        pool_presence,
        pool_repetition,
        .{ .temperature = 0.6, .top_p = 0.9 },
        .{ .temperature = 0.6, .min_p = 0.1 },
        everything,
    };
}

fn checkGreedyInvariance(_: void, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    var row: [prop_vocab]f32 = undefined;
    var set: [setWords(prop_vocab)]u64 = undefined;
    var scratch: TestScratch(prop_vocab) = undefined;
    fillRow(r, &row);
    fillSet(r, &set, prop_vocab);
    // Neutral in every form: no penalty, a neutral setting over a full set,
    // a penalty over an empty set, and cut settings at temperature zero.
    const neutral = [_]Params{
        .{},
        .{ .presence = &set, .context = &set },
        .{ .presence_penalty = 2, .repetition_penalty = 1.5 },
        .{ .top_p = 0.5, .min_p = 0.5, .top_k = 3 },
    };
    for (neutral) |p| {
        @memset(&scratch, std.math.nan(f32));
        var rng = std.Random.DefaultPrng.init(r.int(u64));
        const before = rng.s;
        try std.testing.expectEqual(greedy(&row), try sample(&row, p, rng.random(), &scratch));
        try std.testing.expectEqual(before, rng.s);
        for (scratch) |v| {
            try std.testing.expect(std.math.isNan(v));
        }
    }
}

test "greedy invariance: a neutral request is greedy, draws nothing, and writes no scratch" {
    try prop.forAll(
        .{ .site = 0x7001, .cases = prop_rounds },
        {},
        u64,
        prop.genSeed(void),
        checkGreedyInvariance,
    );
}

fn checkSeedReproducibility(_: void, case_seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(case_seed);
    const r = prng.random();
    var row: [prop_vocab]f32 = undefined;
    var set: [setWords(prop_vocab)]u64 = undefined;
    var scratch: TestScratch(prop_vocab) = undefined;
    fillRow(r, &row);
    fillSet(r, &set, prop_vocab);
    const seed = r.int(u64);
    for (propShapes(&set)) |p| {
        var a = std.Random.DefaultPrng.init(seed);
        var b = std.Random.DefaultPrng.init(seed);
        // The twin takes one float per sampled token, so the RNG count of
        // each path is fixed: none for greedy, one for a draw.
        var twin = std.Random.DefaultPrng.init(seed);
        for (0..8) |_| {
            const ia = try sample(&row, p, a.random(), &scratch);
            const ib = try sample(&row, p, b.random(), &scratch);
            try std.testing.expectEqual(ia, ib);
            if (p.temperature > 0) {
                _ = twin.random().float(f32);
            }

            try std.testing.expectEqual(twin.s, a.s);
        }
    }
}

test "seed reproducibility: one seed gives one stream and one draw per token on every path" {
    try prop.forAll(
        .{ .site = 0x7002, .cases = prop_rounds },
        {},
        u64,
        prop.genSeed(void),
        checkSeedReproducibility,
    );
}

/// The logit of id `i` as the per-element penalty view read it: the oracle of
/// the fix-up.
fn viewAt(row: []const f32, set: []const u64, penalty: f32, i: usize) f32 {
    return if (isMarked(set, @intCast(i))) row[i] - penalty else row[i];
}

fn checkFixupEquivalence(_: void, c: prop.Trial) anyerror!void {
    var prng = std.Random.DefaultPrng.init(c.seed);
    const r = prng.random();
    const round = c.index;
    var row: [prop_vocab]f32 = undefined;
    var set: [setWords(prop_vocab)]u64 = undefined;
    var target: [prop_vocab]f32 = undefined;
    var viewed: [prop_vocab]f32 = undefined;
    var scratch: TestScratch(prop_vocab) = undefined;
    fillRow(r, &row);
    fillSet(r, &set, prop_vocab);
    // Every fourth round holds -inf logits, marked and unmarked. A marked
    // -inf becomes -floatMax, and an unmarked one stays.
    const with_inf = round % 4 == 3;
    if (with_inf) {
        for (0..16) |_| {
            row[r.uintLessThan(usize, prop_vocab)] = -std.math.inf(f32);
        }
    }

    const penalty = (r.float(f32) - 0.5) * 8;
    penalize(&row, &target, .{ .seen = &set }, .{ .presence = penalty });
    for (&viewed, 0..) |*v, i| {
        v.* = viewAt(&row, &set, penalty, i);
    }

    for (viewed, target, 0..) |v, t, i| {
        if (std.math.isNegativeInf(v) and isMarked(&set, @intCast(i))) {
            try std.testing.expectEqual(-std.math.floatMax(f32), t);
        } else {
            try std.testing.expectEqual(@as(u32, @bitCast(v)), @as(u32, @bitCast(t)));
        }
    }

    // The chain over the penalty equals the plain chain over the viewed
    // row, draw for draw, on every path.
    const seed = r.int(u64);
    for (propShapes(&set)) |p| {
        var plain = p;
        plain.presence_penalty = penalty;
        plain.presence = &set;
        plain.repetition_penalty = 1;
        var bare = p;
        bare.presence_penalty = 0;
        bare.presence = &.{};
        bare.repetition_penalty = 1;
        var a = std.Random.DefaultPrng.init(seed);
        var b = std.Random.DefaultPrng.init(seed);
        for (0..8) |_| {
            const want = try sample(&viewed, bare, a.random(), &scratch);
            try std.testing.expectEqual(want, try sample(&row, plain, b.random(), &scratch));
        }
    }
}

test "fix-up equivalence: the scratch row equals the per-element view bit for bit" {
    try prop.forAll(
        .{ .site = 0x7006, .cases = prop_rounds, .max_size = prop_rounds - 1 },
        {},
        prop.Trial,
        prop.genTrial(void),
        checkFixupEquivalence,
    );
}

test "penalize clamps a result past the f32 range and sends a marked NaN or -inf to the floor" {
    const max = std.math.floatMax(f32);
    const inf = std.math.inf(f32);
    const x = [_]f32{ -max, max, 1, -max, std.math.nan(f32), -inf, inf };
    const set = testSet(x.len, &.{ 0, 1, 4, 5, 6 });
    var y: [x.len]f32 = undefined;
    penalize(&x, &y, .{ .seen = &set }, .{ .presence = max });
    try std.testing.expectEqual(-max, y[0]);
    try std.testing.expectEqual(@as(f32, 0), y[1]);
    try std.testing.expectEqual(@as(f32, 1), y[2]);
    try std.testing.expectEqual(-max, y[3]);
    try std.testing.expectEqual(-max, y[4]);
    try std.testing.expectEqual(-max, y[5]);
    try std.testing.expectEqual(max, y[6]);
    penalize(&x, &y, .{ .seen = &set }, .{ .presence = -max });
    try std.testing.expectEqual(@as(f32, 0), y[0]);
    try std.testing.expectEqual(max, y[1]);
    // The repetition rule meets the same floor.
    penalize(&x, &y, .{ .context = &set }, .{ .repetition = 2 });
    try std.testing.expectEqual(-max, y[0]);
    try std.testing.expectEqual(max / 2, y[1]);
    try std.testing.expectEqual(-max, y[4]);
    try std.testing.expectEqual(-max, y[5]);
}

test "penalize reads no padding bit past the vocab" {
    // A set word with every bit high, padding included. Only the three ids
    // of the row move, and the target past the row stays untouched.
    const x = [_]f32{ 1, 2, 3 };
    const set = [_]u64{std.math.maxInt(u64)};
    var y: [x.len + 1]f32 = @splat(7);
    penalize(
        &x,
        y[0..x.len],
        .{ .seen = &set, .context = &set },
        .{ .presence = 1, .repetition = 2 },
    );
    try std.testing.expectEqualSlices(f32, &.{ -0.5, 0, 0.5, 7 }, &y);
}

test "repetition golden row: the context rule, then presence, in the vLLM order" {
    // Row [2, 1, 0, -1], r = 2, presence 1, context {0, 1, 2}, seen {1, 2}.
    // id 0: 2 / 2 = 1. id 1: 1 / 2 - 1 = -0.5. id 2: 0 * 2 - 1 = -1. id 3 is
    // in neither set and keeps -1.
    const x = [_]f32{ 2, 1, 0, -1 };
    const context = testSet(x.len, &.{ 0, 1, 2 });
    const seen = testSet(x.len, &.{ 1, 2 });
    var y: [x.len]f32 = undefined;
    penalize(&x, &y, .{ .seen = &seen, .context = &context }, .{ .presence = 1, .repetition = 2 });
    try std.testing.expectEqualSlices(f32, &.{ 1, -0.5, -1, -1 }, &y);
}

fn checkPenaltyMarked(_: void, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    var row: [prop_vocab]f32 = undefined;
    var seen: [setWords(prop_vocab)]u64 = undefined;
    var context: [setWords(prop_vocab)]u64 = undefined;
    var low: [prop_vocab]f32 = undefined;
    var high: [prop_vocab]f32 = undefined;
    fillRow(r, &row);
    fillSet(r, &seen, prop_vocab);
    fillSet(r, &context, prop_vocab);
    const sets: PenaltySets = .{ .seen = &seen, .context = &context };
    const ap = r.float(f32) * limits.max_presence_penalty;
    const rep = 1 + r.float(f32) * (limits.max_repetition_penalty - 1);
    // A second set of settings at or past the first in both settings.
    const ap2 = ap + r.float(f32) * (limits.max_presence_penalty - ap);
    const rep2 = rep + r.float(f32) * (limits.max_repetition_penalty - rep);
    penalize(&row, &low, sets, .{ .presence = ap, .repetition = rep });
    penalize(&row, &high, sets, .{ .presence = ap2, .repetition = rep2 });
    for (row, low, high, 0..) |l, a, b, i| {
        try std.testing.expect(std.math.isFinite(a));
        try std.testing.expect(std.math.isFinite(b));
        const marked = (ap != 0 and isMarked(&seen, @intCast(i))) or (rep != 1 and isMarked(
            &context,
            @intCast(i),
        ));
        if (marked) {
            try std.testing.expect(a <= l);
            try std.testing.expect(b <= a);
        } else {
            try std.testing.expectEqual(@as(u32, @bitCast(l)), @as(u32, @bitCast(a)));
        }
    }
}

test "penalty monotonicity: no marked logit rises, no other logit moves, results stay finite" {
    try prop.forAll(
        .{ .site = 0x7003, .cases = prop_rounds },
        {},
        u64,
        prop.genSeed(void),
        checkPenaltyMarked,
    );
}

fn checkPenaltyProbability(_: void, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    var row: [prop_vocab]f32 = undefined;
    var y: [prop_vocab]f32 = undefined;
    fillRow(r, &row);
    const id = r.uintLessThan(u32, prop_vocab);
    const set = testSet(prop_vocab, &.{id});
    var last: f64 = 2;
    for ([_]f32{ 1, 1.1, 1.5, 2 }) |rep| {
        penalize(&row, &y, .{ .context = &set }, .{ .repetition = rep });
        var total: f64 = 0;
        for (y) |v| {
            total += @exp(@as(f64, v));
        }

        const p = @exp(@as(f64, y[id])) / total;
        try std.testing.expect(p <= last);
        last = p;
    }
}

test "penalty monotonicity: the probability of a lone marked id never rises with the factor" {
    try prop.forAll(
        .{ .site = 0x7013, .cases = prop_rounds },
        {},
        u64,
        prop.genSeed(void),
        checkPenaltyProbability,
    );
}

/// A random source that repeats one byte. 0x00 makes `float(f32)` return 0,
/// and 0xff makes it return the largest f32 below 1.
const ConstSource = struct {
    byte: u8,

    fn fill(self: *ConstSource, buf: []u8) void {
        @memset(buf, self.byte);
    }

    fn random(self: *ConstSource) std.Random {
        return std.Random.init(self, fill);
    }
};

test "the draw at the edges of u: 0 takes the first positive weight, near 1 the last" {
    var zero: ConstSource = .{ .byte = 0 };
    var one: ConstSource = .{ .byte = 0xff };
    try std.testing.expectEqual(@as(f32, 0), zero.random().float(f32));
    try std.testing.expectEqual(@as(f32, 1) - std.math.floatEps(f32) / 2, one.random().float(f32));
    // Ids 0..4 and the last 3 ids fall under the exp cut at T 1. The rest tie,
    // so `u * total` rounds to the total on the full-vocab path.
    var row: [prop_vocab]f32 = @splat(0);
    for (row[0..5]) |*v| {
        v.* = -200;
    }

    for (row[prop_vocab - 3 ..]) |*v| {
        v.* = -200;
    }

    var scratch: TestScratch(prop_vocab) = undefined;
    const shapes = [_]Params{
        .{ .temperature = 1 },
        .{ .temperature = 1, .top_p = 0.999 },
        .{ .temperature = 1, .min_p = 0.5 },
    };
    for (shapes) |p| {
        try std.testing.expectEqual(@as(u32, 5), try sample(&row, p, zero.random(), &scratch));
        const last: u32 = if (p.top_p < 1) cutLast(prop_vocab - 8, p.top_p) + 5 else prop_vocab - 4;
        try std.testing.expectEqual(last, try sample(&row, p, one.random(), &scratch));
    }

    // The pool walks its own CDF, and the edges hold there too.
    const pool: Params = .{ .temperature = 1, .top_k = 50 };
    try std.testing.expectEqual(@as(u32, 5), try sample(&row, pool, zero.random(), &scratch));
    try std.testing.expectEqual(@as(u32, 54), try sample(&row, pool, one.random(), &scratch));
}

/// The index of the last id that a top_p cut keeps among `n` equal weights.
fn cutLast(n: u32, top_p: f32) u32 {
    const target = @as(f64, top_p) * @as(f64, @floatFromInt(n));
    return @as(u32, @intFromFloat(@ceil(target))) - 1;
}

fn checkFullVocabWeights(_: void, c: prop.Trial) anyerror!void {
    var prng = std.Random.DefaultPrng.init(c.seed);
    const r = prng.random();
    const round = c.index;
    var row: [prop_vocab]f32 = undefined;
    var scratch: TestScratch(prop_vocab) = undefined;
    const weights = scratch[0..prop_vocab];
    const sums = scratch[prop_vocab..];
    const temperatures = [_]f32{ 0.05, 0.6, 1, 1.7, 30 };
    fillRow(r, &row);
    const t = temperatures[round % temperatures.len];
    const total = fullVocabWeights(&row, t, weights, sums);
    const m = rowMax(&row);
    var max_w: f32 = 0;
    var total_ref: f64 = 0;
    for (row, weights) |l, w| {
        // The same f32 input the weight pass forms in its lanes.
        const x = (l - m) / t;
        if (x < exp_simd.exp_cut) {
            try std.testing.expectEqual(@as(f32, 0), w);
        } else {
            try std.testing.expect(exp_simd.expUlpErr(x, w) <= exp_simd.exp_ulp_tol);
        }

        max_w = @max(max_w, w);
        total_ref += @exp(@as(f64, x));
    }

    try std.testing.expectEqual(@as(f32, 1), max_w);
    // The f32 sum of about a thousand weights stays within a few ulps.
    try std.testing.expectApproxEqRel(total_ref, @as(f64, total), 1e-5);
    for (sums, 0..) |s, b| {
        const start = b * weight_block;
        const end = @min(start + weight_block, prop_vocab);
        var block_ref: f64 = 0;
        for (weights[start..end]) |w| {
            block_ref += w;
        }

        try std.testing.expectApproxEqRel(block_ref, @as(f64, s), 1e-5);
    }
}

test "full-vocab weights match the f64 exponential within the exp bound" {
    try prop.forAll(
        .{ .site = 0x70E0, .cases = prop_rounds, .max_size = prop_rounds - 1 },
        {},
        prop.Trial,
        prop.genTrial(void),
        checkFullVocabWeights,
    );
}

test "full-vocab draw: every id has a positive weight and the frequencies follow the weights" {
    var prng = std.Random.DefaultPrng.init(0x70D0);
    const r = prng.random();
    // Ten ids carry the mass at T 1, and the rest fall under the cut.
    var row: [prop_vocab]f32 = @splat(-200);
    const hot = [_]u32{ 0, 7, 63, 64, 65, 300, 511, 512, 1000, 1002 };
    for (hot, 0..) |id, j| {
        row[id] = @as(f32, @floatFromInt(j)) * 0.25;
    }

    var scratch: TestScratch(prop_vocab) = undefined;
    var counts: [hot.len]u32 = @splat(0);
    const draws = 20_000;
    for (0..draws) |_| {
        const id = try sample(&row, .{ .temperature = 1 }, r, &scratch);
        const j = std.mem.indexOfScalar(u32, &hot, id) orelse return error.TestUnexpectedResult;
        counts[j] += 1;
    }

    var mass: f64 = 0;
    for (0..hot.len) |j| {
        mass += @exp(@as(f64, @floatFromInt(j)) * 0.25);
    }

    for (counts, 0..) |c, j| {
        const want = @exp(@as(f64, @floatFromInt(j)) * 0.25) / mass * draws;
        // Five standard deviations of a binomial count.
        try std.testing.expect(@abs(@as(f64, @floatFromInt(c)) - want) <= 5 * @sqrt(want));
    }
}

/// The pool weights and cut of `row` at `top_k`, the way the pool draw forms
/// them.
const TestPool = struct {
    pool: [limits.max_top_k]Candidate = undefined,
    probs: [limits.max_top_k]f32 = undefined,
    k: usize = 0,
    cut: PoolCut = undefined,

    fn init(row: []const f32, top_k: u32, temperature: f32, min_p: f32, top_p: f32) TestPool {
        var t: TestPool = .{};
        t.k = poolWeights(row, top_k, temperature, &t.pool, &t.probs);
        t.cut = poolCut(t.probs[0..t.k], min_p, top_p);
        return t;
    }
};

test "min_p golden rows: three of four ids survive, and the order of the cuts keeps two" {
    // Weights 0.5, 0.3, 0.15, 0.05 at min_p 0.2: the floor is 0.1, so 0.15
    // stays and 0.05 goes.
    var row = [_]f32{ @log(0.5), @log(0.3), @log(0.15), @log(0.05) };
    const golden = TestPool.init(&row, 4, 1, 0.2, 1);
    try std.testing.expectEqual(@as(usize, 3), golden.cut.n);
    // Weights 0.4, 0.3, 0.2, 0.1 at min_p 0.3 and top_p 0.75: min_p keeps
    // three ids of mass 0.9, and top_p then needs 0.675, which two ids reach.
    row = .{ @log(0.4), @log(0.3), @log(0.2), @log(0.1) };
    const order = TestPool.init(&row, 4, 1, 0.3, 0.75);
    try std.testing.expectEqual(@as(usize, 2), order.cut.n);

    // The full-vocab cut keeps the same ids.
    var scratch: TestScratch(row.len) = undefined;
    const row_a: [4]f32 = .{ @log(0.5), @log(0.3), @log(0.15), @log(0.05) };
    const row_b: [4]f32 = .{ @log(0.4), @log(0.3), @log(0.2), @log(0.1) };
    const cases = [_]struct { row: [4]f32, min_p: f32, top_p: f32, kept: []const u32 }{
        .{ .row = row_a, .min_p = 0.2, .top_p = 1, .kept = &.{ 0, 1, 2 } },
        .{ .row = row_b, .min_p = 0.3, .top_p = 0.75, .kept = &.{ 0, 1 } },
    };
    for (cases) |c| {
        const kept = fullCutKept(&c.row, 1, c.min_p, c.top_p, &scratch);
        for (0..row.len) |i| {
            const want = std.mem.indexOfScalar(u32, c.kept, @intCast(i)) != null;
            try std.testing.expectEqual(want, kept[i] > 0);
        }
    }
}

/// The weights the full-vocab cut leaves in the scratch: a zero weight is a
/// dropped id.
fn fullCutKept(
    row: []const f32,
    temperature: f32,
    min_p: f32,
    top_p: f32,
    scratch: []f32,
) []const f32 {
    const total = fullCutWeights(
        row,
        .{ .temperature = temperature, .min_p = min_p, .top_p = top_p },
        scratch,
    );
    std.debug.assert(total >= 1);
    return scratch[0..row.len];
}

fn checkMinPPool(_: void, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    // The whole row fits a pool, so a pool of the row size is the full vocab.
    var row: [200]f32 = undefined;
    fillRow(r, &row);
    const t = 0.3 + r.float(f32) * 1.5;
    const m = r.float(f32);
    const k = 1 + r.uintLessThan(u32, row.len);
    const full = TestPool.init(&row, row.len, t, m, 1);
    const top = TestPool.init(&row, k, t, m, 1);
    for ([_]*const TestPool{ &full, &top }) |p| {
        try std.testing.expect(p.cut.n >= 1);
        // The pool is in rank order, so the survivors are a ranked prefix.
        for (1..p.k) |j| {
            try std.testing.expect(!Candidate.moreLikely({}, p.pool[j], p.pool[j - 1]));
        }

        for (p.probs[0..p.cut.n]) |w| {
            try std.testing.expect(w >= m * p.probs[0]);
        }

        for (p.probs[p.cut.n..p.k]) |w| {
            try std.testing.expect(w < m * p.probs[0]);
        }
    }

    // min_p then top_k equals top_k then min_p: the same ranked prefix.
    try std.testing.expectEqual(@min(k, full.cut.n), top.cut.n);
    for (top.pool[0..top.cut.n], full.pool[0..top.cut.n]) |a, b| {
        try std.testing.expectEqual(a.i, b.i);
    }
}

test "min_p bound on the pool: survivors reach the floor, drops do not, top_k commutes" {
    try prop.forAll(
        .{ .site = 0x7004, .cases = prop_rounds * 4 },
        {},
        u64,
        prop.genSeed(void),
        checkMinPPool,
    );
}

fn checkMinPFull(_: void, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    var row: [prop_vocab]f32 = undefined;
    var scratch: TestScratch(prop_vocab) = undefined;
    var weights: [prop_vocab]f32 = undefined;
    fillRow(r, &row);
    const t = 0.3 + r.float(f32) * 1.5;
    const m = r.float(f32);
    _ = fullVocabWeights(&row, t, &weights, scratch[prop_vocab..]);
    const kept = fullCutKept(&row, t, m, 1, &scratch);
    var n: usize = 0;
    for (weights, kept) |w, k| {
        if (k > 0) {
            n += 1;
            try std.testing.expectEqual(w, k);
            try std.testing.expect(w >= m);
        } else {
            try std.testing.expect(w < m);
        }
    }

    try std.testing.expect(n >= 1);
    var rng = std.Random.DefaultPrng.init(r.int(u64));
    for (0..16) |_| {
        const id = try sample(&row, .{ .temperature = t, .min_p = m }, rng.random(), &scratch);
        try std.testing.expect(weights[id] >= m);
    }
}

test "min_p bound on the full vocab: survivors reach the floor and dropped ids do not" {
    try prop.forAll(
        .{ .site = 0x7014, .cases = prop_rounds },
        {},
        u64,
        prop.genSeed(void),
        checkMinPFull,
    );
}

/// The oracle of the exact cut: sort the eligible ids by weight, ties by id,
/// and walk the f64 mass to `top_p` of the total. Marks the kept ids.
fn refCut(
    gpa: std.mem.Allocator,
    weights: []const f32,
    min_p: f32,
    top_p: f32,
    kept: []bool,
) !void {
    var ids: std.ArrayList(Candidate) = .empty;
    defer ids.deinit(gpa);

    for (weights, 0..) |w, i| {
        if (w > 0 and w >= min_p) {
            try ids.append(gpa, .{ .v = w, .i = @intCast(i) });
        }
    }

    std.mem.sort(Candidate, ids.items, {}, Candidate.moreLikely);
    var total: f64 = 0;
    for (ids.items) |c| {
        total += c.v;
    }

    @memset(kept, false);
    var acc: f64 = 0;
    for (ids.items) |c| {
        kept[c.i] = true;
        acc += c.v;
        if (top_p < 1 and acc >= top_p * total) break;
    }
}

fn checkExactTopP(_: void, c: prop.Trial) anyerror!void {
    var prng = std.Random.DefaultPrng.init(c.seed);
    const r = prng.random();
    const round = c.index;
    const gpa = std.testing.allocator;
    // Three row shapes: random logits, a flat run of ties past the sort
    // bound, and a narrow spread that crowds one bucket past the sort bound.
    const vocab = 6000 + 5;
    const row = try gpa.alloc(f32, vocab);
    defer gpa.free(row);

    const scratch = try gpa.alloc(f32, scratchLen(vocab));
    defer gpa.free(scratch);

    const weights = try gpa.alloc(f32, vocab);
    defer gpa.free(weights);

    const want = try gpa.alloc(bool, vocab);
    defer gpa.free(want);

    switch (round % 3) {
        0 => fillRow(r, row),
        1 => {
            @memset(row, 0);
            for (0..vocab / 8) |_| {
                row[r.uintLessThan(usize, vocab)] = -r.float(f32) * 4;
            }
        },
        else => for (row) |*v| {
            v.* = r.float(f32) * 0.01;
        },
    }

    const t = 0.5 + r.float(f32);
    const top_p = 0.05 + r.float(f32) * 0.9;
    const min_p: f32 = if (round % 2 == 0) 0 else r.float(f32) * 0.5;
    _ = fullVocabWeights(row, t, weights, scratch[vocab..]);
    try refCut(gpa, weights, min_p, top_p, want);
    const kept = fullCutKept(row, t, min_p, top_p, scratch);
    for (want, kept) |w, k| {
        try std.testing.expectEqual(w, k > 0);
    }
}

test "exact top_p over the full vocab equals a sorted f64 walk" {
    try prop.forAll(
        .{ .site = 0x7015, .cases = prop_rounds, .max_size = prop_rounds - 1 },
        {},
        prop.Trial,
        prop.genTrial(void),
        checkExactTopP,
    );
}
