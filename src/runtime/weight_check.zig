//! Role. Check every weight of a loaded model against its card before a runner allocates. After the
//! check, each index that a runner computes from the card lies inside the tensor it addresses, so
//! the index asserts of the kernels hold for every loaded model.
//!
//! Values. The check reads no weight value. A NaN or a +inf weight passes it. The value asserts
//! of the CPU kernels are checks of a safety build only. A release build, and a GPU runner in any
//! build, carry the value to the logits, and the sampler refuses the row with
//! `error.NonFiniteLogits`.
//!
//! Rules. The layer file of the family names each layer weight (`tensorName`) and the dims the card
//! requires of it (`weightDims`). A weight with a second dim of 0 is a vector. Every layer weight
//! and the four model tensors must exist where a runner reads them, with the required dimension
//! count and dims. A vector and the rope factors must be f32, because the kernels read them in
//! place. A matrix takes any dtype here, because `linear_dispatch.requireWeights` refuses a dtype
//! that no kernel reads.
//!
//! Comptime. `checkWeights` takes the family at comptime, so each weight name format is a comptime
//! string, as `tensor_lookup.roleName` needs.
//!
//! Refusals. A refusal logs one line: the two architectures, or the tensor name with its found
//! and required shape or its dtype. It returns a typed `Error`. A test of a refusal logs at
//! `debug`, because the test runner fails a test that logs an error.

const std = @import("std");
const builtin = @import("builtin");
const dtype = @import("../core/dtype.zig");
const limits = @import("../core/limits.zig");
const loader = @import("../io/gguf/loader.zig");
const layer_kernels = @import("layer_kernels.zig");
const tensor_lookup = @import("tensor_lookup.zig");

const Weight = layer_kernels.Weight;
const assert = std.debug.assert;

const log = std.log.scoped(.runner);

// -- constants --------------------------------------------------------------

/// The test runner fails a test that logs an error, and a refusal test expects these lines.
const refusal_level: std.log.Level = if (builtin.is_test) .debug else .err;

// -- types ------------------------------------------------------------------

pub const Error = error{
    UnsupportedArchitecture,
    MissingTensor,
    BadTensorShape,
    UnsupportedWeightDtype,
};

/// The dtype rule of one tensor.
const DtypeRule = enum { any, f32 };

// -- api --------------------------------------------------------------------

/// Refuse a card of another family, and any layer weight or model tensor that is absent where a
/// runner reads it, or that holds another shape than the card requires.
pub fn checkWeights(
    comptime Family: type,
    card: *const loader.ModelCard,
    tensors: []const loader.TensorRef,
) Error!void {
    try requireFamily(Family, card);
    assert(card.n_layers >= 1);
    assert(card.n_layers <= limits.max_layers);

    // Check every weight of every layer.
    var layer: u32 = 0;
    while (layer < card.n_layers) : (layer += 1) {
        for (Family.layerWeights(card, layer)) |role| {
            try checkLayerWeight(Family, card, tensors, role, layer);
        }
    }

    // Check the model tensors. A tied head reads the embedding table, which the first check covers.
    const emb = card.embedding_length;
    try checkTensor(tensors, loader.embed_tensor, .{ emb, card.vocab_size }, .any);
    if (tensor_lookup.find(tensors, loader.head_tensor) != null) {
        try checkTensor(tensors, loader.head_tensor, .{ emb, card.vocab_size }, .any);
    }

    try checkTensor(tensors, loader.output_norm_tensor, .{ emb, 0 }, .f32);
    if (Family.rope_freqs) |name| {
        try checkRopeFreqs(card, tensors, name);
    }
}

/// Refuse a card of another family than the runner of `Family` runs.
pub fn requireFamily(
    comptime Family: type,
    card: *const loader.ModelCard,
) error{UnsupportedArchitecture}!void {
    comptime layer_kernels.assertFamily(Family);

    if (card.arch == Family.arch) return;

    refuse("the card holds architecture {t}, and this runner runs {t}", .{
        card.arch, Family.arch,
    });
    return error.UnsupportedArchitecture;
}

// -- helpers ----------------------------------------------------------------

fn checkLayerWeight(
    comptime Family: type,
    card: *const loader.ModelCard,
    tensors: []const loader.TensorRef,
    role: Weight,
    layer: u32,
) Error!void {
    const dims = Family.weightDims(card, role);
    assert(dims[0] >= 1);

    var buf: [tensor_lookup.role_name_bytes_max]u8 = undefined;
    const name = tensor_lookup.roleName(Family, &buf, role, layer);
    const rule: DtypeRule = if (role.isMatrix()) .any else .f32;
    return checkTensor(tensors, name, dims, rule);
}

/// Check one tensor against its required dims and dtype rule.
fn checkTensor(
    tensors: []const loader.TensorRef,
    name: []const u8,
    dims: [2]u32,
    rule: DtypeRule,
) Error!void {
    assert(dims[0] >= 1);

    const ref = tensor_lookup.findRef(tensors, name) orelse {
        refuse("tensor {s} is missing", .{name});
        return error.MissingTensor;
    };
    const t = ref.tensor;
    const n_dims: u8 = if (dims[1] == 0) 1 else 2;
    const found = t.shape[0..t.n_dims];
    if (t.n_dims != n_dims or t.shape[0] != dims[0] or (n_dims == 2 and t.shape[1] != dims[1])) {
        refuse("tensor {s} has shape {any}, and the card requires {any}", .{
            name, found, dims[0..n_dims],
        });
        return error.BadTensorShape;
    }

    if (rule == .f32 and t.dtype != .f32) {
        refuse("tensor {s} has dtype {t}, and the engine reads it as f32", .{ name, t.dtype });
        return error.UnsupportedWeightDtype;
    }
}

/// The rope factors divide the frequency of each rotated pair, so the tensor holds one per pair.
fn checkRopeFreqs(
    card: *const loader.ModelCard,
    tensors: []const loader.TensorRef,
    name: []const u8,
) Error!void {
    const ref = tensor_lookup.findRef(tensors, name) orelse return;
    const t = ref.tensor;
    const pairs = card.rope_dim_count / 2;
    if (t.n_dims != 1 or t.shape[0] < pairs) {
        refuse("tensor {s} has shape {any}, and the card requires at least [{d}]", .{
            name, t.shape[0..t.n_dims], pairs,
        });
        return error.BadTensorShape;
    }

    if (t.dtype != .f32) {
        refuse("tensor {s} has dtype {t}, and the engine reads it as f32", .{ name, t.dtype });
        return error.UnsupportedWeightDtype;
    }
}

fn refuse(comptime format: []const u8, args: anytype) void {
    switch (refusal_level) {
        .err => log.err(format, args),
        else => log.debug(format, args),
    }
}

// -- tests ------------------------------------------------------------------

const llama_layer = @import("llama_layer.zig");
const qwen_layer = @import("qwen_layer.zig");
const test_fakes = @import("test_fakes.zig");

/// The two families, each with its anchor card.
const test_families = .{ llama_layer, qwen_layer };

test "weightDims: every projected weight holds the widths that the forward pass reads and writes" {
    inline for (test_families) |Family| {
        const card = test_fakes.testCard(Family.arch);
        var layer: u32 = 0;
        while (layer < card.n_layers) : (layer += 1) {
            var log_state: layer_kernels.LogKernels.Log = .{};
            var kernels: layer_kernels.LogKernels = .{ .card = &card, .log = &log_state };
            Family.forward(layer_kernels.LogKernels, &kernels, layer);

            for (Family.layerWeights(&card, layer)) |role| {
                const pair = log_state.projections.get(role) orelse continue;
                const want = [2]u32{
                    Family.activationWidth(&card, pair[0]),
                    Family.activationWidth(&card, pair[1]),
                };
                errdefer std.debug.print("{s} {d} {t}\n", .{ @typeName(Family), layer, role });
                try std.testing.expectEqual(want, Family.weightDims(&card, role));
            }
        }
    }
}

test "weightDims: every weight of a layer has an input width" {
    inline for (test_families) |Family| {
        const card = test_fakes.testCard(Family.arch);
        var layer: u32 = 0;
        while (layer < card.n_layers) : (layer += 1) {
            for (Family.layerWeights(&card, layer)) |role| {
                const dims = Family.weightDims(&card, role);
                errdefer std.debug.print("{s} {t}\n", .{ @typeName(Family), role });
                try std.testing.expect(dims[0] >= 1);
                if (role.isMatrix()) {
                    try std.testing.expect(dims[1] >= 1);
                }
            }
        }
    }
}

const kv_budget = @import("../core/kv_budget.zig");
const resolve = @import("../resolve/resolve.zig");
const chat = @import("../core/chat.zig");
const sampler = @import("sampler.zig");
const parser = @import("../io/gguf/parser.zig");
const prop = @import("../core/utils/prop.zig");
const test_gguf = @import("test_gguf.zig");
const vocab = @import("../core/tokenizer/vocab.zig");
const CpuRunner = @import("cpu_runner.zig").CpuRunner;
const Pool = @import("pool.zig").Pool;

/// One test file after every load check, with the plan that `resolve` chose for it.
const Staged = struct {
    parsed: parser.Parsed,
    card: loader.ModelCard,
    refs: []const loader.TensorRef,
    vocab: vocab.Vocab,
    plan: kv_budget.KvPlan,
};

/// Load `bytes` the way a `run` boot does: parse, the card, the bind, the weight check, the
/// vocabulary, and the placement of `resolve` at a window of 16. A refusal of `resolve` returns
/// `error.Refused`.
fn stageFile(comptime Family: type, arena: std.mem.Allocator, bytes: []const u8) !Staged {
    const file = "test.gguf";
    const parsed = try parser.parse(arena, bytes);
    const card = try loader.cardFromParsed(parsed, file, .debug);
    const refs = try loader.bindTensors(arena, parsed, bytes, file, .debug);
    try checkWeights(Family, &card, refs);
    const v = try vocab.buildFromParsed(arena, parsed);

    const chat_kind: chat.Kind = switch (Family.arch) {
        .llama => .chatml_smollm2,
        .qwen35 => .{ .chatml_qwen = .off },
    };
    const entry: resolve.Entry = .{
        .pub_name = "test",
        .chat_template_kind = chat_kind,
        .kv_dtype_default = .f32,
        .context_limit = null,
    };
    const input: resolve.Input = .{ .engine = .{ .max_seq = 16 } };
    const resolved = switch (resolve.resolve(.run, input, entry, card, .{ .ram_free = null })) {
        .ok => |r| r,
        .refused => return error.Refused,
    };
    const placement = resolved.cpu orelse return error.Refused;
    const plan = switch (placement) {
        .ok => |h| h.plan,
        .refused => return error.Refused,
    };

    return .{ .parsed = parsed, .card = card, .refs = refs, .vocab = v, .plan = plan };
}

/// Build a CPU runner on a staged file, run one step with the BOS token, and sample one token.
fn stepFile(comptime Family: type, arena: std.mem.Allocator, staged: *const Staged) !u32 {
    var pool: Pool = undefined;
    try pool.init(std.testing.io, 1, 0);
    defer pool.deinit();

    // The runner reads no map, so the loaded value holds none.
    var loaded: loader.Loaded = .{
        .map = undefined,
        .parsed = staged.parsed,
        .card = staged.card,
        .tensors = staged.refs,
    };
    var runner = try CpuRunner(Family).init(arena, &loaded, .{
        .plan = staged.plan,
        .max_rows = 1,
        .pool = &pool,
        .policy = .{ .n_threads = 1 },
        .kv_dtype = .f32,
    });
    const logits = try runner.step(staged.vocab.bos_id, 0);
    const scratch = try arena.alloc(f32, sampler.scratchLen(logits.len));
    var prng = std.Random.DefaultPrng.init(1);
    return sampler.sample(logits, .{ .temperature = 1, .top_p = 0.9 }, prng.random(), scratch);
}

/// Stage and step `bytes`, and return the sampled token.
fn runFile(comptime Family: type, arena: std.mem.Allocator, bytes: []const u8) !u32 {
    const staged = try stageFile(Family, arena, bytes);
    return stepFile(Family, arena, &staged);
}

test "test_gguf: the file of each family loads, resolves, steps, and samples a token" {
    inline for (test_families) |Family| {
        var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena_state.deinit();

        const arena = arena_state.allocator();
        const built = try test_gguf.build(arena, Family);
        const token = try runFile(Family, arena, built.bytes);
        try std.testing.expect(token < test_gguf.tokens.len);
    }
}

test "a NaN weight value ends the step with a typed error or a sampled token" {
    // The kernels check their outputs for finite values in a safety build, so only a release build
    // reaches the sampler with a NaN.
    if (std.debug.runtime_safety) return error.SkipZigTest;

    // Layer 1 is a full attention layer in both files, so it holds `attn_q`.
    const targets = [_][]const u8{ loader.output_norm_tensor, "blk.1.attn_q.weight" };
    inline for (test_families) |Family| {
        for (targets) |name| {
            var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
            defer arena_state.deinit();

            const arena = arena_state.allocator();
            const built = try test_gguf.build(arena, Family);
            const staged = try stageFile(Family, arena, built.bytes);
            poisonFirst(staged.refs, name);
            if (stepFile(Family, arena, &staged)) |token| {
                try std.testing.expect(token < test_gguf.tokens.len);
            } else |err| {
                try std.testing.expectEqual(error.NonFiniteLogits, err);
            }
        }
    }
}

/// Write a NaN into the first value of tensor `name`: the first f32, or the f16 scale of the first
/// Q8_0 block.
fn poisonFirst(refs: []const loader.TensorRef, name: []const u8) void {
    const ref = tensor_lookup.findRef(refs, name).?;
    const data: []u8 = @constCast(ref.tensor.data);
    switch (ref.tensor.dtype) {
        .f32 => std.mem.writeInt(u32, data[0..4], @bitCast(std.math.nan(f32)), .little),
        .q8_0 => std.mem.writeInt(u16, data[0..2], @bitCast(std.math.nan(f16)), .little),
        else => unreachable, // The test file holds f32 vectors and Q8_0 matrices only.
    }
}

/// The two files, built once for every case, and the count of cases that reach the sampled token.
const MutationRun = struct {
    llama: test_gguf.Built,
    qwen: test_gguf.Built,
    stepped: u32 = 0,
};

/// One change of one field: the family, the field index, and the new value by its op.
const Mutation = struct {
    qwen: bool,
    field: usize,
    op: u3,
};

/// Half of the cases pick a card or a tokenizer field, because those fields drive the most rules.
fn genMutation(run: *MutationRun, r: std.Random, size: u32) Mutation {
    _ = size;
    const qwen = r.boolean();
    const fields = if (qwen) run.qwen.fields else run.llama.fields;
    const want_header = r.boolean();
    var field = r.uintLessThan(usize, fields.len);
    // The draws stop after a bounded count, because the header fields are a fixed share.
    var tries: u32 = 0;
    while (want_header and fields[field].group == .tensor and tries < 64) : (tries += 1) {
        field = r.uintLessThan(usize, fields.len);
    }

    return .{ .qwen = qwen, .field = field, .op = r.uintLessThan(u3, 6) };
}

/// The new bits of field `f` under op `op`: an edge of the integer range, or for a float, 0, a
/// negative, a NaN, an infinity, a huge value, or the double.
fn mutatedBits(f: test_gguf.Field, op: u3) u64 {
    const max: u64 = if (f.width == 4) std.math.maxInt(u32) else std.math.maxInt(u64);
    if (f.kind == .float) {
        const old: f32 = @bitCast(@as(u32, @intCast(f.value)));
        const values = [_]f32{ 0, -1, std.math.nan(f32), std.math.inf(f32), 1e30, 2 * old };
        return @as(u32, @bitCast(values[op]));
    }

    return switch (op) {
        0 => 0,
        1 => 1,
        2 => if (f.value == 0) max else f.value - 1,
        3 => @min(f.value +| 1, max),
        4 => @min(f.value *| 2, max),
        else => max,
    };
}

/// Change one field of a copy of the file, then stage, step, and sample it. A typed error at any
/// stage passes, and so does a sampled token. A panic or an assert aborts the test binary.
fn checkMutation(run: *MutationRun, m: Mutation) anyerror!void {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();

    const arena = arena_state.allocator();
    const built = if (m.qwen) run.qwen else run.llama;
    const f = built.fields[m.field];
    const value = mutatedBits(f, m.op);

    const bytes = try arena.alignedAlloc(u8, .fromByteUnits(test_gguf.alignment), built.bytes.len);
    @memcpy(bytes, built.bytes);
    switch (f.width) {
        4 => std.mem.writeInt(u32, bytes[f.offset..][0..4], @intCast(value), .little),
        8 => std.mem.writeInt(u64, bytes[f.offset..][0..8], value, .little),
        else => unreachable, // The writer records 4-byte and 8-byte fields only.
    }

    const result = if (m.qwen)
        runFile(qwen_layer, arena, bytes)
    else
        runFile(llama_layer, arena, bytes);
    _ = result catch |err| {
        try std.testing.expect(err != error.OutOfMemory);
        return;
    };
    run.stepped += 1;
}

test "property: a file with one changed field fails typed at some stage, or samples a token" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();

    var run: MutationRun = .{
        .llama = try test_gguf.build(arena_state.allocator(), llama_layer),
        .qwen = try test_gguf.build(arena_state.allocator(), qwen_layer),
    };
    const cases = 400;
    const opts: prop.Options = .{ .site = 0x77C4_EC01, .cases = cases };
    try prop.forAll(opts, &run, Mutation, genMutation, checkMutation);

    // Most changes break a rule. A floor of one case in forty keeps the sampler on the path under
    // any seed.
    try std.testing.expect(run.stepped >= cases / 40);
}
