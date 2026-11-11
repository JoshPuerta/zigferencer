//! Role. Build a `ModelCard` and a tensor list from a GGUF file. The loader maps the file, calls
//! the parser, and reads the architecture-specific KV keys. The card is the typed view of the
//! metadata the engine needs. Each `TensorRef` holds a `Tensor` whose `data` slices into the map.
//!
//! Comptime. The refusal level is a comptime parameter, and an assert pins the load errors to
//! `LoadError`.
//!
//! Validation in pairs. The parser asserts the byte-level form. This layer asserts semantic
//! completeness: every architecture key resolves, the dtype and shape make sense, and the engine
//! bounds in `limits.zig` hold. Neither side trusts the other.
//!
//! Refusals. A refusal of a file value logs once with the file name and the tensor name, and
//! returns a typed `Error`. A test of a refusal logs at `debug`, because the test runner fails a
//! test that logs an error.
//!
//! Alignment. The kernels view tensor data in place. A scalar dtype needs its element width as the
//! address alignment, and a quant block type needs none. The parser proves only the offset a
//! multiple of `general.alignment`, so `bindTensors` checks the address.

const std = @import("std");
const dtype = @import("../../core/dtype.zig");
const kv_budget = @import("../../core/kv_budget.zig");
const limits = @import("../../core/limits.zig");
const span = @import("../../core/span.zig");
const mmap = @import("../mmap.zig");
const parser = @import("parser.zig");
const prop = @import("../../core/prop.zig");
const Tensor = @import("../../core/tensor.zig").Tensor;
const Allocator = std.mem.Allocator;
const Io = std.Io;

const log = std.log.scoped(.gguf);

// -- constants --------------------------------------------------------------

/// The tensor that holds the llama3 RoPE frequency factors.
pub const rope_freqs_tensor = "rope_freqs.weight";

/// The token embedding table. A model without `head_tensor` reads it as its lm_head too.
pub const embed_tensor = "token_embd.weight";

/// The lm_head of a model that does not tie it to the embedding table.
pub const head_tensor = "output.weight";

/// The norm before the lm_head.
pub const output_norm_tensor = "output_norm.weight";

// -- types ------------------------------------------------------------------

pub const Error = error{
    MissingKey,
    WrongKeyType,
    UnsupportedArchitecture,
    NotEnoughBytes,
    BadKeyValue,
    TooManyLayers,
    VocabTooLarge,
    UnsupportedRopeScaling,
    UnsupportedTensorDtype,
    BadTensorShape,
    TensorSizeOverflow,
    /// A tensor starts at an address that breaks the alignment of its element type.
    TensorDataUnaligned,
    /// A tensor starts inside the data of the tensor before it.
    TensorDataOverlap,
} || parser.Error;

/// What `load` and `loadWith` return: a refused file, or a failed map of the file.
pub const LoadError = Error || Allocator.Error || Io.File.OpenError || Io.File.LengthError ||
    Io.File.MemoryMap.CreateError;

/// Architectures the engine understands. The `general.architecture` string is the KV-key prefix.
pub const Arch = enum { llama, qwen35 };

/// Gated-DeltaNet (linear-attention) parameters, under `qwen35.ssm.*`. Present only for `.qwen35`.
pub const GdnParams = struct {
    /// Causal depthwise conv kernel width over the q|k|v channels (= 4).
    conv_kernel: u32,
    /// Per-head recurrent state dimension d (= 128), the GDN "head_dim".
    state_size: u32,
    /// Number of GDN heads (= 16 on the 2B).
    group_count: u32,
    /// Width of v, z, and the readout: time_step_rank * state_size. The loader refuses another
    /// value, and only `inspect` reads it.
    inner_size: u32,
    /// Rank of the dt (time-step) projection feeding the decay gate (= 16).
    time_step_rank: u32,
    /// Every `full_attention_interval`-th layer is full softmax attention. The
    /// rest are GDN. Layer i is full iff (i+1) % interval == 0 (interval = 4).
    full_attention_interval: u32,

    /// True for the full-attention (softmax) layers, false for GDN layers.
    pub fn isFullAttention(self: GdnParams, layer: u32) bool {
        return (layer + 1) % self.full_attention_interval == 0;
    }
};

pub const ModelCard = struct {
    arch: Arch,
    architecture: []const u8,
    name: ?[]const u8 = null,

    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    embedding_length: u32,
    feed_forward_length: u32,
    /// The context the file declares, capped at `limits.max_model_ctx`.
    context_length: u32,
    vocab_size: u32,
    rope_freq_base: f32,
    rope_dim_count: u32,
    rms_norm_eps: f32,

    bos_token_id: u32,
    eos_token_id: u32,
    chat_template: []const u8,

    /// Gated-DeltaNet params. Null for Llama.
    gdn: ?GdnParams = null,

    /// The dimensions the KV price reads, as a plain value for the planner.
    pub fn kvShape(self: ModelCard) kv_budget.KvShape {
        return .{
            .n_layers = self.n_layers,
            .kv_row_elems = self.n_kv_heads * self.head_dim,
            .recurrent = if (self.gdn) |g| .{
                .full_attention_interval = g.full_attention_interval,
                .conv_kernel = g.conv_kernel,
                .state_size = g.state_size,
                .group_count = g.group_count,
                .time_step_rank = g.time_step_rank,
            } else null,
        };
    }
};

pub const TensorRef = struct {
    name: []const u8,
    tensor: Tensor,
};

pub const Loaded = struct {
    map: mmap.Mapped,
    parsed: parser.Parsed,
    card: ModelCard,
    tensors: []const TensorRef,

    pub fn close(self: *Loaded) void {
        self.map.close();
    }
};

pub const LoadOptions = struct {
    /// Prefault all pages at map time, so the decode loop takes no page faults. Turn it off for
    /// a metadata-only reader such as `inspect`.
    prefault: bool = true,
    /// Suppress the `loading...` and `loaded...` info logs and log each refusal at `debug`, for a
    /// caller that reads many models and lists a bad file without its numbers.
    quiet: bool = false,
};

/// One dimension of the model card, named by its key after the arch prefix.
const Dim = struct { key: []const u8, value: u32 };

/// The KV reads of one card. Each refusal logs the file name and the key once.
const Keys = struct {
    parsed: parser.Parsed,
    file: []const u8,
    level: std.log.Level,

    fn fail(self: Keys, comptime format: []const u8, args: anytype) void {
        switch (self.level) {
            inline else => |level| refuse(level, "{s}: " ++ format, .{self.file} ++ args),
        }
    }

    fn value(self: Keys, key: []const u8) Error!*const parser.Value {
        return lookupKv(self.parsed, key) orelse {
            self.fail("the key {s} is missing", .{key});
            return error.MissingKey;
        };
    }

    fn wrongType(self: Keys, key: []const u8, v: *const parser.Value) Error {
        self.fail("the key {s} holds a value of type {t}", .{ key, v.* });
        return error.WrongKeyType;
    }

    fn string(self: Keys, key: []const u8) Error![]const u8 {
        const v = try self.value(key);
        return switch (v.*) {
            .string => |s| s,
            else => self.wrongType(key, v),
        };
    }

    fn strings(self: Keys, key: []const u8) Error![]const []const u8 {
        const v = try self.value(key);
        return switch (v.*) {
            .array => |a| switch (a) {
                .strings => |s| s,
                else => self.wrongType(key, v),
            },
            else => self.wrongType(key, v),
        };
    }

    /// A value that does not fit a `u32` is a bad value, not a narrowing cast.
    fn int(self: Keys, key: []const u8) Error!u32 {
        const v = try self.value(key);
        const wide: i128 = switch (v.*) {
            .u32_val => |u| u,
            .u64_val => |u| u,
            .i32_val => |i| i,
            .i64_val => |i| i,
            else => return self.wrongType(key, v),
        };

        return std.math.cast(u32, wide) orelse {
            self.fail("the key {s} holds {d}, which is not a 32-bit count", .{ key, wide });
            return error.BadKeyValue;
        };
    }

    /// The value of an optional key. Only an absent key takes `default`. A bad value fails.
    fn intOr(self: Keys, key: []const u8, default: u32) Error!u32 {
        if (lookupKv(self.parsed, key) == null) return default;
        return self.int(key);
    }

    fn float(self: Keys, key: []const u8) Error!f32 {
        const v = try self.value(key);
        return switch (v.*) {
            .f32_val => |x| x,
            .f64_val => |x| @floatCast(x),
            else => self.wrongType(key, v),
        };
    }

    /// The value of an optional key. Only an absent key takes `default`. A wrong type fails.
    fn floatOr(self: Keys, key: []const u8, default: f32) Error!f32 {
        if (lookupKv(self.parsed, key) == null) return default;
        return self.float(key);
    }
};

// -- api --------------------------------------------------------------------

/// Load `path` with the default options: a prefaulted map and the info logs.
pub fn load(io: Io, arena: Allocator, path: []const u8) LoadError!Loaded {
    return loadWith(io, arena, path, .{});
}

/// Map `path`, parse it, and build the card and the tensor list. The arena owns every slice.
pub fn loadWith(
    io: Io,
    arena: Allocator,
    path: []const u8,
    opts: LoadOptions,
) LoadError!Loaded {
    const sp = span.begin(.model_load);
    defer sp.end();

    // Map the file. The prefault is the visible wait, so the announce comes first.
    const file = std.Io.Dir.path.basename(path);
    if (!opts.quiet) {
        log.info("loading {s}...", .{file});
    }

    const t_start = Io.Clock.Timestamp.now(io, .awake);
    var map = if (opts.prefault)
        try mmap.openReadOnly(io, path)
    else
        try mmap.openReadOnlyLazy(io, path);

    errdefer map.close();

    const bytes = map.bytes();
    const t_mapped = Io.Clock.Timestamp.now(io, .awake);

    // Parse the bytes.
    const parsed = parser.parse(arena, bytes) catch |err| {
        if (opts.quiet) {
            log.debug("{s}: the GGUF file is not valid: {t}", .{ file, err });
        } else {
            log.err("{s}: the GGUF file is not valid: {t}", .{ file, err });
        }

        return err;
    };
    const t_parsed = Io.Clock.Timestamp.now(io, .awake);

    // Build the card.
    const card = if (opts.quiet)
        try cardFromParsed(parsed, file, .debug)
    else
        try cardFromParsed(parsed, file, .err);

    // Bind the tensors and log the timings.
    const refs = if (opts.quiet)
        try bindTensors(arena, parsed, bytes, file, .debug)
    else
        try bindTensors(arena, parsed, bytes, file, .err);

    const t_done = Io.Clock.Timestamp.now(io, .awake);
    if (!opts.quiet) {
        log.info("loaded {s}: {d} tensors, ctx {d}, {d:.0} MiB in {d:.0} ms", .{
            card.architecture,
            refs.len,
            card.context_length,
            mibOf(bytes.len),
            elapsedMs(t_start, t_done),
        });
    }

    log.debug("  timings: mmap {d:.1} ms | parse {d:.1} ms | build {d:.1} ms", .{
        elapsedMs(t_start, t_mapped), elapsedMs(t_mapped, t_parsed), elapsedMs(t_parsed, t_done),
    });

    return .{
        .map = map,
        .parsed = parsed,
        .card = card,
        .tensors = refs,
    };
}

/// The card of a parsed file, or the typed refusal of a card value. The function reads no file.
/// A refusal logs the file name once, at `refusal_level`.
pub fn cardFromParsed(
    parsed: parser.Parsed,
    file: []const u8,
    comptime refusal_level: std.log.Level,
) Error!ModelCard {
    const keys: Keys = .{ .parsed = parsed, .file = file, .level = refusal_level };

    // Read the architecture and the shape keys.
    const arch_str = try keys.string("general.architecture");
    const arch: Arch = if (std.mem.eql(u8, arch_str, "llama"))
        .llama
    else if (std.mem.eql(u8, arch_str, "qwen35"))
        .qwen35
    else {
        keys.fail("the architecture {s} is not supported", .{arch_str});
        return error.UnsupportedArchitecture;
    };

    // One key buffer is safe because each lookup consumes its key before the next write.
    var kb: [96]u8 = undefined;
    const n_layers = try keys.int(pk(&kb, arch_str, ".block_count"));
    const n_heads = try keys.int(pk(&kb, arch_str, ".attention.head_count"));
    // The head count divides the embedding width below.
    if (n_heads == 0) {
        keys.fail("the card value attention.head_count is 0", .{});
        return error.BadKeyValue;
    }

    const n_kv_heads = try keys.intOr(pk(&kb, arch_str, ".attention.head_count_kv"), n_heads);
    const embedding_length = try keys.int(pk(&kb, arch_str, ".embedding_length"));
    const ffn_length = try keys.int(pk(&kb, arch_str, ".feed_forward_length"));
    const ctx_declared = try keys.int(pk(&kb, arch_str, ".context_length"));

    // A file without the key derives the head width, so a refusal names the source it read.
    const key_length_held = lookupKv(parsed, pk(&kb, arch_str, ".attention.key_length")) != null;
    const head_dim_source = if (key_length_held)
        "attention.key_length"
    else
        "embedding_length / attention.head_count";
    const head_dim_explicit = try keys.intOr(
        pk(&kb, arch_str, ".attention.key_length"),
        embedding_length / n_heads,
    );
    const rope_dim_count = try keys.intOr(
        pk(&kb, arch_str, ".rope.dimension_count"),
        head_dim_explicit,
    );
    const rope_freq_base = try keys.floatOr(pk(&kb, arch_str, ".rope.freq_base"), 10000.0);
    const rms_norm_eps = try keys.float(pk(&kb, arch_str, ".attention.layer_norm_rms_epsilon"));

    // Read the Gated-DeltaNet keys, which only qwen35 has.
    const gdn: ?GdnParams = if (arch == .qwen35) GdnParams{
        .conv_kernel = try keys.int(pk(&kb, arch_str, ".ssm.conv_kernel")),
        .state_size = try keys.int(pk(&kb, arch_str, ".ssm.state_size")),
        .group_count = try keys.int(pk(&kb, arch_str, ".ssm.group_count")),
        .inner_size = try keys.int(pk(&kb, arch_str, ".ssm.inner_size")),
        .time_step_rank = try keys.int(pk(&kb, arch_str, ".ssm.time_step_rank")),
        .full_attention_interval = try keys.int(pk(&kb, arch_str, ".full_attention_interval")),
    } else null;

    const tokens = try keys.strings("tokenizer.ggml.tokens");

    // Check the bounds. A typed error here keeps a file from sizing past `core/limits.zig`.
    if (n_layers > limits.max_layers) {
        keys.fail("{d} layers pass the limit of {d}", .{ n_layers, limits.max_layers });
        return error.TooManyLayers;
    }
    if (tokens.len > limits.max_vocab) {
        keys.fail("{d} tokens pass the limit of {d}", .{ tokens.len, limits.max_vocab });
        return error.VocabTooLarge;
    }

    const vocab_size: u32 = @intCast(tokens.len);
    // A zero context leaves no window to plan.
    if (ctx_declared == 0) {
        keys.fail("the card value context_length is 0", .{});
        return error.BadKeyValue;
    }

    const dims = [_]Dim{
        .{ .key = "block_count", .value = n_layers },
        .{ .key = "attention.head_count_kv", .value = n_kv_heads },
        .{ .key = "embedding_length", .value = embedding_length },
        .{ .key = "feed_forward_length", .value = ffn_length },
        .{ .key = head_dim_source, .value = head_dim_explicit },
    };
    if (zeroDim(&dims, gdn)) |key| {
        keys.fail("the card value {s} is 0", .{key});
        return error.BadKeyValue;
    }

    // Check the head grouping: each KV head serves a whole group of query heads.
    if (n_heads % n_kv_heads != 0) {
        keys.fail(
            "{d} attention heads do not split into groups over {d} KV heads",
            .{ n_heads, n_kv_heads },
        );
        return error.BadKeyValue;
    }

    // Cap the context. The file loads with the bound, and the log names both numbers. A test
    // that refuses at `debug` also notes the cap at `debug`.
    const ctx_length = limits.contextCeiling(ctx_declared, null);
    if (ctx_length < ctx_declared) {
        const format = "{s}: the file declares a context of {d} positions, " ++
            "and the engine caps it at {d}";
        const args = .{ file, ctx_declared, ctx_length };
        if (refusal_level == .debug) {
            log.debug(format, args);
        } else {
            log.warn(format, args);
        }
    }

    // Refuse RoPE scaling that the engine does not apply, because nothing downstream detects it.
    const scaling_key = pk(&kb, arch_str, ".rope.scaling.type");
    const scaling_type: ?[]const u8 = if (lookupKv(parsed, scaling_key) != null)
        try keys.string(scaling_key)
    else
        null;

    // Only the Llama runners read the llama3 factor tensor.
    const applies_rope_freqs = arch == .llama and hasTensor(parsed, rope_freqs_tensor);
    if (!ropeScalingSupported(scaling_type, applies_rope_freqs)) {
        std.debug.assert(scaling_type != null);

        keys.fail(
            "RoPE scaling type {s} is not supported. The engine applies only the " ++
                "llama3 factors of a " ++ rope_freqs_tensor ++ " tensor",
            .{scaling_type.?},
        );
        return error.UnsupportedRopeScaling;
    }

    // Read the special tokens and build the card. BOS is optional: ChatML models omit it.
    const bos = try keys.intOr("tokenizer.ggml.bos_token_id", 0);
    const eos = try keys.int("tokenizer.ggml.eos_token_id");
    const chat = lookupStringOptional(parsed, "tokenizer.chat_template") orelse "";

    const name = lookupStringOptional(parsed, "general.name");

    const card: ModelCard = .{
        .arch = arch,
        .architecture = arch_str,
        .name = name,
        .n_layers = n_layers,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim_explicit,
        .embedding_length = embedding_length,
        .feed_forward_length = ffn_length,
        .context_length = ctx_length,
        .vocab_size = vocab_size,
        .rope_freq_base = rope_freq_base,
        .rope_dim_count = rope_dim_count,
        .rms_norm_eps = rms_norm_eps,
        .bos_token_id = bos,
        .eos_token_id = eos,
        .chat_template = chat,
        .gdn = gdn,
    };
    try checkCard(card, file, refusal_level);
    return card;
}

/// Turn each tensor descriptor into a `Tensor` that borrows its bytes from `bytes`.
/// The file supplies every dimension, size, range, and alignment, so each gets a check here.
/// A refusal logs the file and the tensor name once, at `refusal_level`.
pub fn bindTensors(
    arena: Allocator,
    parsed: parser.Parsed,
    bytes: []const u8,
    file: []const u8,
    comptime refusal_level: std.log.Level,
) Error![]const TensorRef {
    const refs = try arena.alloc(TensorRef, parsed.tensors.len);
    // The parser proves the starts in table order, so each tensor must start past the previous end.
    var prev_end: u64 = parsed.data_offset;
    for (parsed.tensors, refs) |ti, *ref| {
        std.debug.assert(ti.n_dims >= 1 and ti.n_dims <= 4);

        // Refuse a dtype without block geometry, because only a subset can be sized.
        if (!dtype.hasBlockGeometry(ti.dtype)) {
            refuse(
                refusal_level,
                "{s}: tensor {s} carries dtype {s} (ggml id {d}), which the engine cannot size",
                .{
                    file,
                    ti.name,
                    std.enums.tagName(dtype.GgmlType, ti.dtype) orelse "unknown",
                    @intFromEnum(ti.dtype),
                },
            );
            return error.UnsupportedTensorDtype;
        }

        // Check the dimensions and count the elements.
        var shape32: [4]u32 = .{ 0, 0, 0, 0 };
        var numel: u64 = 1;
        for (0..ti.n_dims) |i| {
            const dim = std.math.cast(u32, ti.shape[i]) orelse {
                refuse(
                    refusal_level,
                    "{s}: tensor {s} has dimension {d} of {d}, which passes the 32-bit limit",
                    .{ file, ti.name, i, ti.shape[i] },
                );
                return error.BadTensorShape;
            };
            if (dim == 0) {
                refuse(
                    refusal_level,
                    "{s}: tensor {s} has an empty dimension {d}",
                    .{ file, ti.name, i },
                );
                return error.BadTensorShape;
            }

            shape32[i] = dim;
            numel = std.math.mul(u64, numel, dim) catch {
                refuse(
                    refusal_level,
                    "{s}: tensor {s} has an element count that passes 64 bits",
                    .{ file, ti.name },
                );
                return error.TensorSizeOverflow;
            };
        }

        // Check that the row holds whole blocks and size the tensor.
        const block = dtype.blockSize(ti.dtype);
        if (shape32[0] % block != 0) {
            refuse(
                refusal_level,
                "{s}: tensor {s} has a row of {d} elements, " ++
                    "which is not a multiple of the {d}-element block of its dtype",
                .{
                    file, ti.name, shape32[0], block,
                },
            );
            return error.BadTensorShape;
        }

        // Every axis past the row counts whole rows, so the tensor is a whole number of blocks.
        const total_bytes = dtype.bytesForNumelChecked(ti.dtype, numel) orelse {
            refuse(
                refusal_level,
                "{s}: tensor {s} has {d} elements, and their byte size passes 64 bits",
                .{ file, ti.name, numel },
            );
            return error.TensorSizeOverflow;
        };
        const row_bytes = dtype.bytesForNumelChecked(ti.dtype, shape32[0]).?;
        std.debug.assert(row_bytes <= total_bytes);

        // Check that the byte range lies inside the file.
        const start = std.math.add(u64, parsed.data_offset, ti.offset_in_data) catch {
            refuse(
                refusal_level,
                "{s}: tensor {s} starts at data offset {d}, which passes 64 bits",
                .{ file, ti.name, ti.offset_in_data },
            );
            return error.TensorSizeOverflow;
        };
        const end = std.math.add(u64, start, total_bytes) catch {
            refuse(
                refusal_level,
                "{s}: tensor {s} ends past 64 bits ({d} bytes from byte {d})",
                .{ file, ti.name, total_bytes, start },
            );
            return error.TensorSizeOverflow;
        };
        if (end > bytes.len) {
            refuse(
                refusal_level,
                "{s}: tensor {s} needs bytes {d} to {d}, and the file holds {d}",
                .{ file, ti.name, start, end, bytes.len },
            );
            return error.NotEnoughBytes;
        }
        if (start < prev_end) {
            refuse(
                refusal_level,
                "{s}: tensor {s} starts at byte {d}, inside the tensor before it, " ++
                    "which ends at byte {d}",
                .{ file, ti.name, start, prev_end },
            );
            return error.TensorDataOverlap;
        }

        prev_end = end;

        // Check the address alignment of the view.
        const data = bytes[@intCast(start)..@intCast(end)];
        const need = elementAlign(ti.dtype);
        if (@intFromPtr(data.ptr) % need != 0) {
            refuse(
                refusal_level,
                "{s}: tensor {s} starts at byte {d}, " ++
                    "which breaks the {d}-byte alignment of its dtype",
                .{ file, ti.name, start, need },
            );
            return error.TensorDataUnaligned;
        }

        // Build the tensor.
        const stride = std.math.cast(u32, row_bytes) orelse {
            refuse(
                refusal_level,
                "{s}: tensor {s} has a row of {d} bytes, which passes the 32-bit stride limit",
                .{ file, ti.name, row_bytes },
            );
            return error.TensorSizeOverflow;
        };

        ref.* = .{
            .name = ti.name,
            .tensor = .{
                .dtype = ti.dtype,
                .n_dims = ti.n_dims,
                .shape = shape32,
                .byte_stride = .{ stride, 0, 0, 0 },
                .data = data,
            },
        };
        ref.tensor.assertValid();
        std.debug.assert(ref.tensor.data.len == ref.tensor.bytesRequired());
    }

    return refs;
}

/// Whether the engine applies the RoPE scaling a file declares. It applies none, or the llama3
/// factors from a `rope_freqs.weight` tensor. It applies no YaRN and no linear scaling.
pub fn ropeScalingSupported(scaling_type: ?[]const u8, has_rope_freqs: bool) bool {
    const kind = scaling_type orelse return true;
    if (std.mem.eql(u8, kind, "none")) return true;
    if (std.mem.eql(u8, kind, "llama3")) return has_rope_freqs;

    return false;
}

// -- helpers ----------------------------------------------------------------

/// Refuse a card value that a runner turns into an index past its buffer. The zero and head
/// grouping checks run first.
fn checkCard(card: ModelCard, file: []const u8, comptime refusal_level: std.log.Level) Error!void {
    std.debug.assert(card.n_heads >= 1);
    std.debug.assert(card.n_kv_heads >= 1);

    // RoPE rotates pairs of the first `rope_dim_count` lanes of each head. The GPU binds a
    // factor buffer of one entry per pair, so the count holds one pair at least.
    const rope = card.rope_dim_count;
    if (rope < 2 or rope % 2 != 0 or rope > card.head_dim) {
        refuse(
            refusal_level,
            "{s}: the RoPE dimension count {d} is not an even count in 2..{d}, the head width",
            .{ file, rope, card.head_dim },
        );
        return error.BadKeyValue;
    }

    // The norm divides by a root that holds the epsilon, and RoPE takes a power of the base.
    const floats = [_]struct { key: []const u8, value: f32 }{
        .{ .key = "attention.layer_norm_rms_epsilon", .value = card.rms_norm_eps },
        .{ .key = "rope.freq_base", .value = card.rope_freq_base },
    };
    for (floats) |f| {
        if (!std.math.isFinite(f.value) or !(f.value > 0)) {
            refuse(
                refusal_level,
                "{s}: the card value {s} is {d}, and it must be finite and above 0",
                .{ file, f.key, f.value },
            );
            return error.BadKeyValue;
        }
    }

    // The logits row holds one value per token, and the special ids index the token list.
    if (card.vocab_size == 0) {
        refuse(refusal_level, "{s}: the token list is empty", .{file});
        return error.BadKeyValue;
    }
    if (card.bos_token_id >= card.vocab_size or card.eos_token_id >= card.vocab_size) {
        refuse(
            refusal_level,
            "{s}: the BOS id {d} or the EOS id {d} is not below the token count {d}",
            .{ file, card.bos_token_id, card.eos_token_id, card.vocab_size },
        );
        return error.BadKeyValue;
    }

    // The layer files compute these widths as u32.
    const q_dim = try cardProduct(file, refusal_level, "head_count * key_length", .{
        card.n_heads, card.head_dim,
    });
    const kv_row = try cardProduct(file, refusal_level, "head_count_kv * key_length", .{
        card.n_kv_heads, card.head_dim,
    });
    if (kv_row > limits.max_kv_row_elems) {
        refuse(
            refusal_level,
            "{s}: the KV row of {d} elements passes the limit of {d}",
            .{ file, kv_row, limits.max_kv_row_elems },
        );
        return error.BadKeyValue;
    }

    const g = card.gdn orelse return;
    try checkGdn(card, g, q_dim, file, refusal_level);
}

/// The Qwen3.5 rules: the GDN state fits the scratch of the kernels, the conv ring holds at least
/// one past step, and every GDN width fits a u32.
fn checkGdn(
    card: ModelCard,
    g: GdnParams,
    q_dim: u32,
    file: []const u8,
    comptime refusal_level: std.log.Level,
) Error!void {
    std.debug.assert(card.arch == .qwen35);
    std.debug.assert(q_dim >= 1);

    if (g.state_size > limits.gdn_head_dim) {
        refuse(
            refusal_level,
            "{s}: the GDN state size {d} passes the limit of {d}",
            .{ file, g.state_size, limits.gdn_head_dim },
        );
        return error.BadKeyValue;
    }
    if (g.conv_kernel < 2) {
        refuse(
            refusal_level,
            "{s}: the GDN conv kernel {d} is less than 2",
            .{ file, g.conv_kernel },
        );
        return error.BadKeyValue;
    }

    // The q projection interleaves q and the attention gate per head.
    _ = try cardProduct(file, refusal_level, "2 * head_count * key_length", .{ 2, q_dim });
    const key_dim = try cardProduct(file, refusal_level, "group_count * state_size", .{
        g.group_count, g.state_size,
    });
    const value_dim = try cardProduct(file, refusal_level, "time_step_rank * state_size", .{
        g.time_step_rank, g.state_size,
    });
    if (g.inner_size != value_dim) {
        refuse(
            refusal_level,
            "{s}: the GDN inner size {d} differs from time_step_rank * state_size, {d}",
            .{ file, g.inner_size, value_dim },
        );
        return error.BadKeyValue;
    }

    const qkv_wide = 2 * @as(u64, key_dim) + value_dim;
    const qkv = std.math.cast(u32, qkv_wide) orelse {
        refuse(
            refusal_level,
            "{s}: the GDN qkv width {d} passes 32 bits",
            .{ file, qkv_wide },
        );
        return error.BadKeyValue;
    };
    const state = try cardProduct(file, refusal_level, "time_step_rank * state_size^2", .{
        value_dim, g.state_size,
    });
    const conv = try cardProduct(file, refusal_level, "(conv_kernel - 1) * qkv width", .{
        g.conv_kernel - 1, qkv,
    });

    // The planner prices the GDN state of one slot under the two ceilings of `limits.zig`.
    const layer_bytes = (@as(u64, state) + conv) * @sizeOf(f32);
    const n_gdn: u64 = card.n_layers - card.n_layers / g.full_attention_interval;
    if (layer_bytes > limits.max_fixed_state_bytes_per_layer or
        n_gdn * layer_bytes > limits.max_fixed_state_bytes)
    {
        refuse(
            refusal_level,
            "{s}: the GDN state of one slot is {d} bytes on each of {d} layers, past the " ++
                "limits of {d} bytes a layer and {d} bytes in all",
            .{
                file,
                layer_bytes,
                n_gdn,
                limits.max_fixed_state_bytes_per_layer,
                limits.max_fixed_state_bytes,
            },
        );
        return error.BadKeyValue;
    }
}

/// The product of two card values, or a refusal when it passes 32 bits.
fn cardProduct(
    file: []const u8,
    comptime refusal_level: std.log.Level,
    comptime what: []const u8,
    factors: [2]u32,
) Error!u32 {
    return std.math.mul(u32, factors[0], factors[1]) catch {
        refuse(
            refusal_level,
            "{s}: the card product " ++ what ++ " of {d} and {d} passes 32 bits",
            .{ file, factors[0], factors[1] },
        );
        return error.BadKeyValue;
    };
}

/// Log one refusal at `level`.
fn refuse(comptime level: std.log.Level, comptime format: []const u8, args: anytype) void {
    switch (level) {
        .err => log.err(format, args),
        .warn => log.warn(format, args),
        .info => log.info(format, args),
        .debug => log.debug(format, args),
    }
}

/// The address alignment that a typed in-place view of `t` needs.
fn elementAlign(t: dtype.GgmlType) usize {
    std.debug.assert(dtype.hasBlockGeometry(t));

    const need: usize = if (dtype.blockSize(t) == 1) dtype.bytesPerBlock(t) else 1;
    std.debug.assert(std.math.isPowerOfTwo(need));
    return need;
}

fn hasTensor(p: parser.Parsed, name: []const u8) bool {
    for (p.tensors) |ti| {
        if (std.mem.eql(u8, ti.name, name)) return true;
    }

    return false;
}

fn mibOf(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}

fn elapsedMs(t0: Io.Clock.Timestamp, t1: Io.Clock.Timestamp) f64 {
    const ns: u64 = @intCast(t0.durationTo(t1).raw.toNanoseconds());
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// Build a prefixed KV key such as `<arch>.block_count` into `buf`. The key always fits.
fn pk(buf: []u8, prefix: []const u8, suffix: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ prefix, suffix }) catch unreachable;
}

fn lookupKv(p: parser.Parsed, key: []const u8) ?*const parser.Value {
    for (p.kvs) |*kv| {
        if (std.mem.eql(u8, kv.key, key)) return &kv.value;
    }

    return null;
}

fn lookupStringOptional(p: parser.Parsed, key: []const u8) ?[]const u8 {
    const v = lookupKv(p, key) orelse return null;
    return switch (v.*) {
        .string => |s| s,
        else => null,
    };
}

/// The first card dimension or GDN parameter that holds 0, or null. A zero breaks a division later.
fn zeroDim(dims: []const Dim, gdn: ?GdnParams) ?[]const u8 {
    for (dims) |d| {
        if (d.value == 0) return d.key;
    }

    const g = gdn orelse return null;
    inline for (@typeInfo(GdnParams).@"struct".fields) |f| {
        if (@field(g, f.name) == 0) return f.name;
    }

    return null;
}

// -- tests ------------------------------------------------------------------

test "zeroDim: the first zero card dimension or GDN parameter, else null" {
    const good = [_]Dim{ .{ .key = "a", .value = 1 }, .{ .key = "b", .value = 7 } };
    const gdn: GdnParams = .{
        .conv_kernel = 4,
        .state_size = 128,
        .group_count = 16,
        .inner_size = 2048,
        .time_step_rank = 16,
        .full_attention_interval = 4,
    };
    const Case = struct { dims: []const Dim, gdn: ?GdnParams, want: ?[]const u8 };
    var no_interval = gdn;
    no_interval.full_attention_interval = 0;
    const zero_b = [_]Dim{ .{ .key = "a", .value = 1 }, .{ .key = "b", .value = 0 } };
    const cases = [_]Case{
        .{ .dims = &good, .gdn = null, .want = null },
        .{ .dims = &good, .gdn = gdn, .want = null },
        .{ .dims = &zero_b, .gdn = gdn, .want = "b" },
        .{ .dims = &good, .gdn = no_interval, .want = "full_attention_interval" },
    };
    for (cases) |c| {
        const got = zeroDim(c.dims, c.gdn);
        if (c.want) |w| {
            try std.testing.expectEqualStrings(w, got.?);
        } else {
            try std.testing.expect(got == null);
        }
    }
}

test "load and loadWith name their error set" {
    const returnedSet = struct {
        fn of(comptime f: type) type {
            return @typeInfo(@typeInfo(f).@"fn".return_type.?).error_union.error_set;
        }
    }.of;
    comptime std.debug.assert(returnedSet(@TypeOf(load)) == LoadError);
    comptime std.debug.assert(returnedSet(@TypeOf(loadWith)) == LoadError);
}

test "Keys.floatOr: an absent key takes the default, and a wrong type refuses" {
    const key = "llama.rope.freq_base";
    const keysOf = struct {
        fn of(kvs: []const parser.Kv) Keys {
            const parsed: parser.Parsed = .{
                .header = undefined,
                .kvs = kvs,
                .tensors = &.{},
                .data_offset = 0,
                .alignment = 1,
            };
            return .{ .parsed = parsed, .file = "case.gguf", .level = .debug };
        }
    }.of;
    const absent = keysOf(&.{});
    const value = keysOf(&.{.{ .key = key, .value = .{ .f32_val = 2.5 } }});
    const wrong = keysOf(&.{.{ .key = key, .value = .{ .string = "2.5" } }});
    try std.testing.expectEqual(7.0, try absent.floatOr(key, 7.0));
    try std.testing.expectEqual(2.5, try value.floatOr(key, 7.0));
    try std.testing.expectError(error.WrongKeyType, wrong.floatOr(key, 7.0));
}

test "ropeScalingSupported: the engine loads only the scaling it applies" {
    // No scaling key, or an explicit `none`, needs nothing from the engine.
    try std.testing.expect(ropeScalingSupported(null, false));
    try std.testing.expect(ropeScalingSupported("none", false));
    // The llama3 factors apply only through the factor tensor.
    try std.testing.expect(ropeScalingSupported("llama3", true));
    try std.testing.expect(!ropeScalingSupported("llama3", false));
    // YaRN and linear scaling have no implementation, with or without the
    // tensor.
    try std.testing.expect(!ropeScalingSupported("yarn", false));
    try std.testing.expect(!ropeScalingSupported("yarn", true));
    try std.testing.expect(!ropeScalingSupported("linear", false));
    try std.testing.expect(!ropeScalingSupported("linear", true));
}

/// A small card of each family that every card rule accepts.
fn smallCard(comptime arch: Arch) ModelCard {
    var card: ModelCard = .{
        .arch = arch,
        .architecture = @tagName(arch),
        .n_layers = 4,
        .n_heads = 4,
        .n_kv_heads = 2,
        .head_dim = 16,
        .embedding_length = 32,
        .feed_forward_length = 64,
        .context_length = 64,
        .vocab_size = card_tokens.len,
        .rope_freq_base = 10000.0,
        .rope_dim_count = 8,
        .rms_norm_eps = 1e-6,
        .bos_token_id = 1,
        .eos_token_id = 2,
        .chat_template = "",
    };
    if (arch == .qwen35) {
        card.gdn = .{
            .conv_kernel = 4,
            .state_size = 16,
            .group_count = 2,
            .inner_size = 32,
            .time_step_rank = 2,
            .full_attention_interval = 4,
        };
    }

    return card;
}

const card_tokens = [_][]const u8{ "a", "b", "c", "d", "e", "f", "g", "h" };

/// The KVs of `card` as a file holds them. The token list sets the vocab size.
fn cardKvs(comptime arch: Arch, card: ModelCard, out: *[24]parser.Kv) []const parser.Kv {
    std.debug.assert(card.arch == arch);
    std.debug.assert(card.vocab_size <= card_tokens.len);

    const p = @tagName(arch);
    const u = struct {
        fn of(x: u32) parser.Value {
            return .{ .u32_val = x };
        }
    }.of;
    const common = [_]parser.Kv{
        .{ .key = "general.architecture", .value = .{ .string = p } },
        .{ .key = p ++ ".block_count", .value = u(card.n_layers) },
        .{ .key = p ++ ".attention.head_count", .value = u(card.n_heads) },
        .{ .key = p ++ ".attention.head_count_kv", .value = u(card.n_kv_heads) },
        .{ .key = p ++ ".embedding_length", .value = u(card.embedding_length) },
        .{ .key = p ++ ".feed_forward_length", .value = u(card.feed_forward_length) },
        .{ .key = p ++ ".context_length", .value = u(card.context_length) },
        .{ .key = p ++ ".attention.key_length", .value = u(card.head_dim) },
        .{ .key = p ++ ".rope.dimension_count", .value = u(card.rope_dim_count) },
        .{
            .key = p ++ ".attention.layer_norm_rms_epsilon",
            .value = .{ .f32_val = card.rms_norm_eps },
        },
        .{ .key = p ++ ".rope.freq_base", .value = .{ .f32_val = card.rope_freq_base } },
        .{
            .key = "tokenizer.ggml.tokens",
            .value = .{ .array = .{ .strings = card_tokens[0..card.vocab_size] } },
        },
        .{ .key = "tokenizer.ggml.bos_token_id", .value = u(card.bos_token_id) },
        .{ .key = "tokenizer.ggml.eos_token_id", .value = u(card.eos_token_id) },
    };
    @memcpy(out[0..common.len], &common);
    const g = card.gdn orelse return out[0..common.len];

    const ssm = [_]parser.Kv{
        .{ .key = p ++ ".ssm.conv_kernel", .value = u(g.conv_kernel) },
        .{ .key = p ++ ".ssm.state_size", .value = u(g.state_size) },
        .{ .key = p ++ ".ssm.group_count", .value = u(g.group_count) },
        .{ .key = p ++ ".ssm.inner_size", .value = u(g.inner_size) },
        .{ .key = p ++ ".ssm.time_step_rank", .value = u(g.time_step_rank) },
        .{ .key = p ++ ".full_attention_interval", .value = u(g.full_attention_interval) },
    };
    @memcpy(out[common.len..][0..ssm.len], &ssm);
    return out[0 .. common.len + ssm.len];
}

/// Keep the GDN inner size equal to the value width after a change of its factors.
fn matchInnerSize(c: *ModelCard) void {
    const g = &c.gdn.?;
    g.inner_size = g.time_step_rank *% g.state_size;
}

/// One card field change and its expected result. Null: the card loads.
const CardCase = struct {
    arch: Arch,
    edit: *const fn (*ModelCard) void,
    want: ?Error,
};

fn expectCardCase(c: CardCase) !void {
    var card = switch (c.arch) {
        inline else => |a| smallCard(a),
    };
    c.edit(&card);
    var buf: [24]parser.Kv = undefined;
    const kvs = switch (c.arch) {
        inline else => |a| cardKvs(a, card, &buf),
    };
    const parsed: parser.Parsed = .{
        .header = undefined,
        .kvs = kvs,
        .tensors = &.{},
        .data_offset = 0,
        .alignment = 1,
    };

    const got = cardFromParsed(parsed, "case.gguf", .debug);
    if (c.want) |err| return std.testing.expectError(err, got);

    const loaded = try got;
    try std.testing.expectEqual(card.head_dim, loaded.head_dim);
    try std.testing.expectEqual(card.vocab_size, loaded.vocab_size);
}

test "cardFromParsed: each card rule refuses a file value that breaks it" {
    const E = struct {
        fn none(_: *ModelCard) void {}
        fn ropeOdd(c: *ModelCard) void {
            c.rope_dim_count = 7;
        }
        fn ropePastHead(c: *ModelCard) void {
            c.rope_dim_count = c.head_dim + 2;
        }
        fn vocabEmpty(c: *ModelCard) void {
            c.vocab_size = 0;
            c.bos_token_id = 0;
            c.eos_token_id = 0;
        }
        fn bosPastVocab(c: *ModelCard) void {
            c.bos_token_id = c.vocab_size;
        }
        fn eosPastVocab(c: *ModelCard) void {
            c.eos_token_id = c.vocab_size;
        }
        fn kvRowPastLimit(c: *ModelCard) void {
            c.head_dim = limits.max_kv_row_elems;
        }
        fn qWidthPast32Bits(c: *ModelCard) void {
            c.n_heads = 1 << 13;
            c.n_kv_heads = 1;
            c.head_dim = 1 << 19;
        }
        fn qGatePast32Bits(c: *ModelCard) void {
            c.n_heads = 1 << 12;
            c.n_kv_heads = 1;
            c.head_dim = 1 << 19;
        }
        fn stateSizePastLimit(c: *ModelCard) void {
            c.gdn.?.state_size = limits.gdn_head_dim * 2;
            matchInnerSize(c);
        }
        fn convKernelOne(c: *ModelCard) void {
            c.gdn.?.conv_kernel = 1;
        }
        fn keyWidthPast32Bits(c: *ModelCard) void {
            c.gdn.?.group_count = 1 << 28;
            matchInnerSize(c);
        }
        fn qkvWidthPast32Bits(c: *ModelCard) void {
            c.gdn.?.state_size = limits.gdn_head_dim;
            c.gdn.?.group_count = 1 << 24;
            matchInnerSize(c);
        }
        fn statePast32Bits(c: *ModelCard) void {
            c.gdn.?.time_step_rank = 1 << 24;
            matchInnerSize(c);
        }
        fn convRingPast32Bits(c: *ModelCard) void {
            c.gdn.?.conv_kernel = 1 << 30;
        }
        fn innerSizeOff(c: *ModelCard) void {
            c.gdn.?.inner_size += 1;
        }
        fn ropeZero(c: *ModelCard) void {
            c.rope_dim_count = 0;
        }
        fn epsZero(c: *ModelCard) void {
            c.rms_norm_eps = 0;
        }
        fn epsNegative(c: *ModelCard) void {
            c.rms_norm_eps = -1e-6;
        }
        fn epsNan(c: *ModelCard) void {
            c.rms_norm_eps = std.math.nan(f32);
        }
        fn epsInf(c: *ModelCard) void {
            c.rms_norm_eps = std.math.inf(f32);
        }
        fn freqBaseZero(c: *ModelCard) void {
            c.rope_freq_base = 0;
        }
        fn freqBaseNegative(c: *ModelCard) void {
            c.rope_freq_base = -10000;
        }
        fn freqBaseNan(c: *ModelCard) void {
            c.rope_freq_base = std.math.nan(f32);
        }
        fn freqBaseInf(c: *ModelCard) void {
            c.rope_freq_base = std.math.inf(f32);
        }
        fn stateBytesPastLayerLimit(c: *ModelCard) void {
            c.gdn.?.time_step_rank = 1 << 21;
            matchInnerSize(c);
        }
        fn stateBytesPastModelLimit(c: *ModelCard) void {
            c.n_layers = 64;
            c.gdn.?.time_step_rank = 1 << 13;
            c.gdn.?.state_size = limits.gdn_head_dim / 2;
            matchInnerSize(c);
        }
    };
    const bad = error.BadKeyValue;
    const cases = [_]CardCase{
        .{ .arch = .llama, .edit = &E.none, .want = null },
        .{ .arch = .qwen35, .edit = &E.none, .want = null },
        .{ .arch = .llama, .edit = &E.ropeOdd, .want = bad },
        .{ .arch = .llama, .edit = &E.ropePastHead, .want = bad },
        .{ .arch = .llama, .edit = &E.vocabEmpty, .want = bad },
        .{ .arch = .llama, .edit = &E.bosPastVocab, .want = bad },
        .{ .arch = .qwen35, .edit = &E.eosPastVocab, .want = bad },
        .{ .arch = .llama, .edit = &E.kvRowPastLimit, .want = bad },
        .{ .arch = .llama, .edit = &E.qWidthPast32Bits, .want = bad },
        .{ .arch = .qwen35, .edit = &E.qGatePast32Bits, .want = bad },
        .{ .arch = .qwen35, .edit = &E.stateSizePastLimit, .want = bad },
        .{ .arch = .qwen35, .edit = &E.convKernelOne, .want = bad },
        .{ .arch = .qwen35, .edit = &E.keyWidthPast32Bits, .want = bad },
        .{ .arch = .qwen35, .edit = &E.qkvWidthPast32Bits, .want = bad },
        .{ .arch = .qwen35, .edit = &E.statePast32Bits, .want = bad },
        .{ .arch = .qwen35, .edit = &E.convRingPast32Bits, .want = bad },
        .{ .arch = .qwen35, .edit = &E.stateBytesPastLayerLimit, .want = bad },
        .{ .arch = .qwen35, .edit = &E.stateBytesPastModelLimit, .want = bad },
        .{ .arch = .llama, .edit = &E.ropeZero, .want = bad },
        .{ .arch = .qwen35, .edit = &E.innerSizeOff, .want = bad },
        .{ .arch = .llama, .edit = &E.epsZero, .want = bad },
        .{ .arch = .qwen35, .edit = &E.epsNegative, .want = bad },
        .{ .arch = .llama, .edit = &E.epsNan, .want = bad },
        .{ .arch = .llama, .edit = &E.epsInf, .want = bad },
        .{ .arch = .llama, .edit = &E.freqBaseZero, .want = bad },
        .{ .arch = .qwen35, .edit = &E.freqBaseNegative, .want = bad },
        .{ .arch = .llama, .edit = &E.freqBaseNan, .want = bad },
        .{ .arch = .llama, .edit = &E.freqBaseInf, .want = bad },
    };
    // Every row runs, so one failure does not hide the next.
    var failed: u32 = 0;
    for (cases, 0..) |c, i| {
        expectCardCase(c) catch {
            std.debug.print("card case {d} failed\n", .{i});
            failed += 1;
        };
    }

    try std.testing.expectEqual(@as(u32, 0), failed);
}

const BindCase = struct {
    dtype: dtype.GgmlType = .f32,
    shape: []const u64,
    data_offset: u64 = 0,
    offset_in_data: u64 = 0,
    file_len: usize = bind_capacity,
    /// Null binds, with the row stride and the byte length below.
    want: ?Error = null,
    stride: u32 = 0,
    len: usize = 0,
};

const bind_capacity = 256;

/// One tensor descriptor binds over a 16-byte aligned file of `file_len`
/// bytes, or it fails with the row's error.
fn checkBind(c: BindCase) !void {
    var file: [bind_capacity]u8 align(16) = @splat(0);
    var shape: [4]u64 = .{ 1, 1, 1, 1 };
    @memcpy(shape[0..c.shape.len], c.shape);
    const infos = [_]parser.TensorInfo{.{
        .name = "t",
        .dtype = c.dtype,
        .n_dims = @intCast(c.shape.len),
        .shape = shape,
        .offset_in_data = c.offset_in_data,
    }};
    const parsed: parser.Parsed = .{
        .header = .{ .version = 3, .tensor_count = 1, .kv_count = 0 },
        .kvs = &.{},
        .tensors = &infos,
        .data_offset = c.data_offset,
        .alignment = 1,
    };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const got = bindTensors(arena.allocator(), parsed, file[0..c.file_len], "case.gguf", .debug);
    if (c.want) |err| return std.testing.expectError(err, got);

    const refs = try got;
    try std.testing.expectEqual(@as(usize, 1), refs.len);
    try std.testing.expectEqual(c.stride, refs[0].tensor.byte_stride[0]);
    try std.testing.expectEqual(c.len, refs[0].tensor.data.len);
}

test "bindTensors: a descriptor binds inside the file at its row stride, or fails typed" {
    const max_u32: u64 = std.math.maxInt(u32);
    const max_u64: u64 = std.math.maxInt(u64);
    const overflow = error.TensorSizeOverflow;
    const cases = [_]BindCase{
        .{ .shape = &.{ 4, 2 }, .stride = 16, .len = 32 },
        .{ .dtype = .q8_0, .shape = &.{ 32, 2 }, .offset_in_data = 2, .stride = 34, .len = 68 },
        .{ .dtype = .f16, .shape = &.{ 8, 3 }, .data_offset = 2, .stride = 16, .len = 48 },
        .{ .shape = &.{64}, .stride = 256, .len = 256 },
        // The element count fits 64 bits, and the byte size wraps to 0.
        .{ .shape = &.{ 1 << 31, 1 << 31, 4 }, .want = overflow },
        .{ .shape = &.{ 1 << 31, 1 << 31, 3 }, .want = overflow },
        .{ .shape = &.{ max_u32, max_u32, max_u32 }, .want = overflow },
        .{ .shape = &.{4}, .data_offset = 16, .offset_in_data = max_u64 - 7, .want = overflow },
        .{ .shape = &.{4}, .offset_in_data = max_u64 - 7, .want = overflow },
        .{ .shape = &.{ 4, 0 }, .want = error.BadTensorShape },
        .{ .shape = &.{ max_u32 + 1, 1 }, .want = error.BadTensorShape },
        .{ .dtype = .q8_0, .shape = &.{ 48, 2 }, .want = error.BadTensorShape },
        .{ .dtype = .q6_k, .shape = &.{ 32, 1 }, .want = error.BadTensorShape },
        .{ .shape = &.{ 4, 2 }, .file_len = 31, .want = error.NotEnoughBytes },
        .{ .shape = &.{64}, .offset_in_data = 4, .want = error.NotEnoughBytes },
        .{ .dtype = .iq2_xxs, .shape = &.{256}, .want = error.UnsupportedTensorDtype },
        .{ .dtype = @enumFromInt(9999), .shape = &.{1}, .want = error.UnsupportedTensorDtype },
        .{ .shape = &.{4}, .data_offset = 2, .want = error.TensorDataUnaligned },
        .{ .dtype = .f16, .shape = &.{4}, .offset_in_data = 1, .want = error.TensorDataUnaligned },
    };
    for (cases) |c| {
        try checkBind(c);
    }
}

test "bindTensors: a tensor that starts inside the tensor before it refuses" {
    var file: [bind_capacity]u8 align(16) = @splat(0);
    // The second tensor starts at byte `at` of the data, and the first ends at byte 32.
    const Case = struct { at: u64, want: ?Error };
    const cases = [_]Case{
        .{ .at = 32, .want = null },
        .{ .at = 16, .want = error.TensorDataOverlap },
        .{ .at = 0, .want = error.TensorDataOverlap },
    };
    const shape: [4]u64 = .{ 8, 1, 1, 1 };
    for (cases) |c| {
        const infos = [_]parser.TensorInfo{
            .{ .name = "a", .dtype = .f32, .n_dims = 1, .shape = shape, .offset_in_data = 0 },
            .{ .name = "b", .dtype = .f32, .n_dims = 1, .shape = shape, .offset_in_data = c.at },
        };
        const parsed: parser.Parsed = .{
            .header = .{ .version = 3, .tensor_count = 2, .kv_count = 0 },
            .kvs = &.{},
            .tensors = &infos,
            .data_offset = 0,
            .alignment = 1,
        };
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();

        const got = bindTensors(arena.allocator(), parsed, &file, "case.gguf", .debug);
        if (c.want) |err| {
            try std.testing.expectError(err, got);
        } else {
            try std.testing.expectEqual(@as(usize, 2), (try got).len);
        }
    }
}

/// A random mutation of the parser sample: byte writes in the header, the kvs, and the tensor
/// table, one 8-byte edge value, and one cut. `parse` then `bindTensors` gives a typed error,
/// or tensors that lie inside the bytes with a row stride that tiles the data and an aligned start.
fn checkParseBind(sample: *const parser.Sample, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();

    // Copy the sample and write random bytes.
    var bytes: [parser.sample_capacity]u8 align(16) = undefined;
    @memcpy(bytes[0..sample.len], sample.bytes[0..sample.len]);
    for (0..r.uintLessThan(usize, 4)) |_| {
        bytes[r.uintLessThan(usize, sample.table_end)] = r.int(u8);
    }

    // Write one 8-byte edge value.
    if (r.boolean()) {
        const edges = [_]u64{
            0,
            1,
            2,
            3,
            4,
            16,
            17,
            1 << 31,
            1 << 32,
            std.math.maxInt(u32),
            std.math.maxInt(u63),
            std.math.maxInt(u64),
            std.math.maxInt(u64) - 15,
        };
        const at = r.uintAtMost(usize, sample.table_end - 8);
        std.mem.writeInt(u64, bytes[at..][0..8], edges[r.uintLessThan(usize, edges.len)], .little);
    }

    // Cut the file at a random length.
    const len = if (r.uintLessThan(u8, 4) == 0) r.uintAtMost(usize, sample.len) else sample.len;
    const file = bytes[0..len];

    // Parse and bind in a bounded heap.
    var heap: [16 * 1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&heap);
    const parsed = parser.parse(fba.allocator(), file) catch |err| {
        try std.testing.expect(err != error.OutOfMemory);
        return;
    };
    const refs = bindTensors(fba.allocator(), parsed, file, "sample.gguf", .debug) catch |err| {
        try std.testing.expect(err != error.OutOfMemory);
        return;
    };

    // Check that each tensor lies inside the file with a tiled stride.
    const lo = @intFromPtr(file.ptr);
    for (refs) |ref| {
        const t = ref.tensor;
        const at = @intFromPtr(t.data.ptr);
        try std.testing.expect(at >= lo and at + t.data.len <= lo + file.len);
        try std.testing.expectEqual(t.bytesRequired(), t.data.len);
        try std.testing.expectEqual(
            t.data.len,
            @as(usize, t.byte_stride[0]) * (t.numel() / t.shape[0]),
        );
        try std.testing.expectEqual(@as(usize, 0), at % elementAlign(t.dtype));
    }
}

test "property: parse then bind over a mutated sample binds inside the bytes or fails typed" {
    const sample = try parser.sampleGguf();
    try prop.forAll(
        .{ .site = 0x6767_B1D0, .cases = 4000 },
        &sample,
        u64,
        prop.genSeed(*const parser.Sample),
        checkParseBind,
    );
}
