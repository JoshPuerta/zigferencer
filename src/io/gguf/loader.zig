//! Role. Build a `ModelCard` and a tensor list from a GGUF file. The loader maps the file, calls
//! the parser, and reads the architecture-specific KV keys. The card is the typed view of the
//! metadata the engine needs. Each `TensorRef` holds a `Tensor` whose `data` slices into the map.
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
    /// Width of each of q, k, v after projection (= group_count * state_size).
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
    /// Prefault all pages at map time. Turn it off for a metadata-only reader such as `inspect`.
    prefault: bool = true,
    /// Suppress the `loading...` and `loaded...` info logs, for a caller that reads many models.
    quiet: bool = false,
};

/// One dimension of the model card, named by its key after the arch prefix.
const Dim = struct { key: []const u8, value: u32 };

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
    if (!opts.quiet) {
        log.info("loading {s}...", .{std.Io.Dir.path.basename(path)});
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
        log.err(
            "{s}: the GGUF file is not valid: {s}",
            .{ std.Io.Dir.path.basename(path), @errorName(err) },
        );
        return err;
    };
    const t_parsed = Io.Clock.Timestamp.now(io, .awake);

    // Read the architecture and the shape keys.
    const arch_str = try lookupString(parsed, "general.architecture");
    const arch: Arch = if (std.mem.eql(u8, arch_str, "llama"))
        .llama
    else if (std.mem.eql(u8, arch_str, "qwen35"))
        .qwen35
    else
        return error.UnsupportedArchitecture;

    // One key buffer is safe because each lookup consumes its key before the next write.
    var kb: [96]u8 = undefined;
    const n_layers = try lookupU32(parsed, pk(&kb, arch_str, ".block_count"));
    const n_heads = try lookupU32(parsed, pk(&kb, arch_str, ".attention.head_count"));
    // The head count divides the embedding width below.
    if (n_heads == 0) return error.BadKeyValue;

    const n_kv_heads = try lookupU32Or(
        parsed,
        pk(&kb, arch_str, ".attention.head_count_kv"),
        n_heads,
    );
    const embedding_length = try lookupU32(parsed, pk(&kb, arch_str, ".embedding_length"));
    const ffn_length = try lookupU32(parsed, pk(&kb, arch_str, ".feed_forward_length"));
    const ctx_declared = try lookupU32(parsed, pk(&kb, arch_str, ".context_length"));

    const head_dim_explicit = try lookupU32Or(
        parsed,
        pk(&kb, arch_str, ".attention.key_length"),
        embedding_length / n_heads,
    );
    const rope_dim_count = try lookupU32Or(
        parsed,
        pk(&kb, arch_str, ".rope.dimension_count"),
        head_dim_explicit,
    );
    const rope_freq_base = try lookupF32Or(parsed, pk(&kb, arch_str, ".rope.freq_base"), 10000.0);
    const rms_norm_eps = try lookupF32(
        parsed,
        pk(&kb, arch_str, ".attention.layer_norm_rms_epsilon"),
    );

    // Read the Gated-DeltaNet keys, which only qwen35 has.
    const gdn: ?GdnParams = if (arch == .qwen35) GdnParams{
        .conv_kernel = try lookupU32(parsed, pk(&kb, arch_str, ".ssm.conv_kernel")),
        .state_size = try lookupU32(parsed, pk(&kb, arch_str, ".ssm.state_size")),
        .group_count = try lookupU32(parsed, pk(&kb, arch_str, ".ssm.group_count")),
        .inner_size = try lookupU32(parsed, pk(&kb, arch_str, ".ssm.inner_size")),
        .time_step_rank = try lookupU32(parsed, pk(&kb, arch_str, ".ssm.time_step_rank")),
        .full_attention_interval = try lookupU32(
            parsed,
            pk(&kb, arch_str, ".full_attention_interval"),
        ),
    } else null;

    const tokens = try lookupStringArray(parsed, "tokenizer.ggml.tokens");

    // Check the bounds. A typed error here keeps a file from sizing past `core/limits.zig`.
    if (n_layers > limits.max_layers) return error.TooManyLayers;
    if (tokens.len > limits.max_vocab) return error.VocabTooLarge;

    const vocab_size: u32 = @intCast(tokens.len);
    // A zero context leaves no window to plan.
    if (ctx_declared == 0) return error.BadKeyValue;

    const dims = [_]Dim{
        .{ .key = "block_count", .value = n_layers },
        .{ .key = "attention.head_count_kv", .value = n_kv_heads },
        .{ .key = "embedding_length", .value = embedding_length },
        .{ .key = "feed_forward_length", .value = ffn_length },
        .{ .key = "attention.key_length", .value = head_dim_explicit },
    };
    if (zeroDim(&dims, gdn)) |key| {
        log.err("{s}: the card value {s} is 0", .{ std.Io.Dir.path.basename(path), key });
        return error.BadKeyValue;
    }

    // Check the head grouping: each KV head serves a whole group of query heads.
    if (n_heads % n_kv_heads != 0) {
        log.err(
            "{s}: {d} attention heads do not split into groups over {d} KV heads",
            .{ std.Io.Dir.path.basename(path), n_heads, n_kv_heads },
        );
        return error.BadKeyValue;
    }

    // Cap the context. The file loads with the bound, and the log names both numbers.
    const ctx_length = limits.contextCeiling(ctx_declared, null);
    if (ctx_length < ctx_declared) {
        log.warn(
            "{s}: the file declares a context of {d} positions, and the engine caps it at {d}",
            .{
                std.Io.Dir.path.basename(path), ctx_declared, ctx_length,
            },
        );
    }

    // Refuse RoPE scaling that the engine does not apply, because nothing downstream detects it.
    const scaling_key = pk(&kb, arch_str, ".rope.scaling.type");
    const scaling_type: ?[]const u8 = if (lookupKv(parsed, scaling_key) != null)
        try lookupString(parsed, scaling_key)
    else
        null;

    // Only the Llama runners read the llama3 factor tensor.
    const applies_rope_freqs = arch == .llama and hasTensor(parsed, rope_freqs_tensor);
    if (!ropeScalingSupported(scaling_type, applies_rope_freqs)) {
        std.debug.assert(scaling_type != null);

        log.err(
            "{s}: RoPE scaling type {s} is not supported. The engine applies only the " ++
                "llama3 factors of a " ++ rope_freqs_tensor ++ " tensor",
            .{
                std.Io.Dir.path.basename(path), scaling_type.?,
            },
        );
        return error.UnsupportedRopeScaling;
    }

    // Read the special tokens and build the card. BOS is optional: ChatML models omit it.
    const bos = try lookupU32Or(parsed, "tokenizer.ggml.bos_token_id", 0);
    const eos = try lookupU32(parsed, "tokenizer.ggml.eos_token_id");
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

    // Bind the tensors and log the timings.
    const refs = try bindTensors(arena, parsed, bytes, std.Io.Dir.path.basename(path), .err);

    const t_done = Io.Clock.Timestamp.now(io, .awake);
    if (!opts.quiet) {
        log.info("loaded {s}: {d} tensors, ctx {d}, {d:.0} MiB in {d:.0} ms", .{
            arch_str, refs.len, ctx_length, mibOf(bytes.len), elapsedMs(t_start, t_done),
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
                "{s}: tensor {s} has a row of {d} elements, which is not a multiple of the " ++
                    "{d}-element block of its dtype",
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

        // Check the address alignment of the view. See "Alignment" in the header.
        const data = bytes[@intCast(start)..@intCast(end)];
        const need = elementAlign(ti.dtype);
        if (@intFromPtr(data.ptr) % need != 0) {
            refuse(
                refusal_level,
                "{s}: tensor {s} starts at byte {d}, which breaks the {d}-byte alignment of " ++
                    "its dtype",
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

/// Log one refusal at `level`. See "Refusals" in the header.
fn refuse(comptime level: std.log.Level, comptime format: []const u8, args: anytype) void {
    switch (level) {
        .err => log.err(format, args),
        .warn => log.warn(format, args),
        .info => log.info(format, args),
        .debug => log.debug(format, args),
    }
}

/// The address alignment that a typed in-place view of `t` needs. See "Alignment" in the header.
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

fn lookupString(p: parser.Parsed, key: []const u8) Error![]const u8 {
    const v = lookupKv(p, key) orelse return error.MissingKey;
    return switch (v.*) {
        .string => |s| s,
        else => error.WrongKeyType,
    };
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

/// A value that does not fit a `u32` is a bad value, not a narrowing cast.
fn lookupU32(p: parser.Parsed, key: []const u8) Error!u32 {
    const v = lookupKv(p, key) orelse return error.MissingKey;
    const value: ?u32 = switch (v.*) {
        .u32_val => |u| u,
        .u64_val => |u| std.math.cast(u32, u),
        .i32_val => |i| std.math.cast(u32, i),
        .i64_val => |i| std.math.cast(u32, i),
        else => return error.WrongKeyType,
    };

    return value orelse error.BadKeyValue;
}

/// The value of an optional key. Only an absent key takes `default`. A bad value fails.
fn lookupU32Or(p: parser.Parsed, key: []const u8, default: u32) Error!u32 {
    return lookupU32(p, key) catch |err| switch (err) {
        error.MissingKey => default,
        else => err,
    };
}

fn lookupF32(p: parser.Parsed, key: []const u8) Error!f32 {
    const v = lookupKv(p, key) orelse return error.MissingKey;
    return switch (v.*) {
        .f32_val => |x| x,
        .f64_val => |x| @floatCast(x),
        else => error.WrongKeyType,
    };
}

/// The value of an optional key. Only an absent key takes `default`. A wrong type fails.
fn lookupF32Or(p: parser.Parsed, key: []const u8, default: f32) Error!f32 {
    return lookupF32(p, key) catch |err| switch (err) {
        error.MissingKey => default,
        else => err,
    };
}

fn lookupStringArray(p: parser.Parsed, key: []const u8) Error![]const []const u8 {
    const v = lookupKv(p, key) orelse return error.MissingKey;
    return switch (v.*) {
        .array => |a| switch (a) {
            .strings => |s| s,
            else => error.WrongKeyType,
        },
        else => error.WrongKeyType,
    };
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
    const cases = [_]Case{
        .{ .dims = &good, .gdn = null, .want = null },
        .{ .dims = &good, .gdn = gdn, .want = null },
        .{ .dims = &.{ .{ .key = "a", .value = 1 }, .{ .key = "b", .value = 0 } }, .gdn = gdn, .want = "b" },
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

test "lookupF32Or: an absent key takes the default, and a wrong type refuses" {
    const key = "llama.rope.freq_base";
    const parsedFrom = struct {
        fn of(kvs: []const parser.Kv) parser.Parsed {
            return .{
                .header = undefined,
                .kvs = kvs,
                .tensors = &.{},
                .data_offset = 0,
                .alignment = 1,
            };
        }
    }.of;
    const absent = parsedFrom(&.{});
    const value = parsedFrom(&.{.{ .key = key, .value = .{ .f32_val = 2.5 } }});
    const wrong = parsedFrom(&.{.{ .key = key, .value = .{ .string = "2.5" } }});
    try std.testing.expectEqual(7.0, try lookupF32Or(absent, key, 7.0));
    try std.testing.expectEqual(2.5, try lookupF32Or(value, key, 7.0));
    try std.testing.expectError(error.WrongKeyType, lookupF32Or(wrong, key, 7.0));
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
    const cases = [_]BindCase{
        .{ .shape = &.{ 4, 2 }, .stride = 16, .len = 32 },
        .{ .dtype = .q8_0, .shape = &.{ 32, 2 }, .offset_in_data = 2, .stride = 34, .len = 68 },
        .{ .dtype = .f16, .shape = &.{ 8, 3 }, .data_offset = 2, .stride = 16, .len = 48 },
        .{ .shape = &.{64}, .stride = 256, .len = 256 },
        // The element count fits 64 bits, and the byte size wraps to 0.
        .{ .shape = &.{ 1 << 31, 1 << 31, 4 }, .want = error.TensorSizeOverflow },
        .{ .shape = &.{ 1 << 31, 1 << 31, 3 }, .want = error.TensorSizeOverflow },
        .{ .shape = &.{ max_u32, max_u32, max_u32 }, .want = error.TensorSizeOverflow },
        .{ .shape = &.{4}, .data_offset = 16, .offset_in_data = max_u64 - 7, .want = error.TensorSizeOverflow },
        .{ .shape = &.{4}, .offset_in_data = max_u64 - 7, .want = error.TensorSizeOverflow },
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

/// A random mutation of the parser sample: byte writes in the header, the kvs,
/// and the tensor table, one 8-byte edge value, and one cut. `parse` then
/// `bindTensors` gives a typed error, or tensors that each lie inside the
/// bytes with a row stride that tiles the data and an aligned start.
fn checkParseBind(sample: *const parser.Sample, seed: u64) anyerror!void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    var bytes: [parser.sample_capacity]u8 align(16) = undefined;
    @memcpy(bytes[0..sample.len], sample.bytes[0..sample.len]);
    for (0..r.uintLessThan(usize, 4)) |_| {
        bytes[r.uintLessThan(usize, sample.table_end)] = r.int(u8);
    }

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

    const len = if (r.uintLessThan(u8, 4) == 0) r.uintAtMost(usize, sample.len) else sample.len;
    const file = bytes[0..len];

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
