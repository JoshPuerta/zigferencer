//! Role. A tiny valid GGUF file of each family, written in memory for the tests. Every width is a
//! small multiple of the Q8_0 block, so the CPU runner builds on the file and runs a step.
//!
//! Fields. The writer records the byte offset of each number that a test can change in place: the
//! integer and float card values, the token ids, every string length, the token counts, and the
//! dtype, dimension count, dims, and data offset of each tensor. A change keeps the tensor data
//! where the original layout put it.
//!
//! Comptime. `build` takes the family at comptime, so each weight name format is a comptime
//! string.
//!
//! Data. The Q8_0 matrices hold small quants under one scale, and the vectors hold values that
//! keep the forward pass finite: unit norms, a negative GDN decay, and a small conv tap.

const std = @import("std");
const dtype = @import("../core/dtype.zig");
const loader = @import("../io/gguf/loader.zig");
const layer_kernels = @import("layer_kernels.zig");
const tensor_lookup = @import("tensor_lookup.zig");
const Allocator = std.mem.Allocator;

const Weight = layer_kernels.Weight;

// -- constants --------------------------------------------------------------

/// The tokens of every file. The step reads token 0, the BOS id.
pub const tokens = [_][]const u8{ "a", "b", "c", "d", "e", "f", "g", "h" };

/// The data alignment of the file, the GGUF default.
pub const alignment = 32;

/// The GGUF value type ids that the writer uses.
const type_u32: u32 = 4;
const type_i32: u32 = 5;
const type_f32: u32 = 6;
const type_string: u32 = 8;
const type_array: u32 = 9;

/// The f16 bits of 1/128, the scale of every Q8_0 block.
const q8_scale_bits: u16 = 0x2000;

/// The most tensors of a file: two layers of the 14 GDN weights, and the three model tensors.
const tensors_max = 2 * 14 + 3;

// -- types ------------------------------------------------------------------

/// One number of the file that a test can change in place.
pub const Field = struct {
    offset: usize,
    /// The width in bytes, 4 or 8. Every field is little endian.
    width: u8,
    /// The bits of the value. A float field holds the bits of an f32.
    value: u64,
    kind: Kind,
    group: Group,

    pub const Kind = enum { int, float, length, offset };
    pub const Group = enum { card, tokenizer, tensor };
};

/// The bytes of one file and its changeable fields. The arena owns both.
pub const Built = struct {
    bytes: []align(alignment) u8,
    fields: []const Field,
};

/// One tensor of the file before the write.
const Spec = struct {
    name: []const u8,
    dims: [2]u32,
    dtype: dtype.GgmlType,
    fill: f32,
};

/// The little-endian writes of the file, and the record of each changeable field.
const Writer = struct {
    arena: Allocator,
    out: *std.ArrayList(u8),
    fields: *std.ArrayList(Field),
    /// The group of the fields that the writes record now.
    group: *Field.Group,

    fn int(self: Writer, comptime T: type, v: T) !void {
        var buf: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &buf, v, .little);
        try self.out.appendSlice(self.arena, &buf);
    }

    /// Write an integer and record it as a field of `kind`.
    fn field(self: Writer, comptime T: type, v: T, kind: Field.Kind) !void {
        try self.fields.append(self.arena, .{
            .offset = self.out.items.len,
            .width = @sizeOf(T),
            .value = v,
            .kind = kind,
            .group = self.group.*,
        });
        try self.int(T, v);
    }

    /// Pad the file to the next aligned offset.
    fn pad(self: Writer) !void {
        const len = self.out.items.len;
        try self.out.appendNTimes(self.arena, 0, std.mem.alignForward(usize, len, alignment) - len);
    }

    fn string(self: Writer, s: []const u8) !void {
        try self.field(u64, s.len, .length);
        try self.out.appendSlice(self.arena, s);
    }

    fn key(self: Writer, name: []const u8, value_type: u32) !void {
        try self.string(name);
        try self.int(u32, value_type);
    }

    fn keyU32(self: Writer, name: []const u8, v: u32) !void {
        try self.key(name, type_u32);
        try self.field(u32, v, .int);
    }

    fn keyF32(self: Writer, name: []const u8, v: f32) !void {
        try self.key(name, type_f32);
        try self.field(u32, @bitCast(v), .float);
    }

    fn keyString(self: Writer, name: []const u8, v: []const u8) !void {
        try self.key(name, type_string);
        try self.string(v);
    }
};

// -- api --------------------------------------------------------------------

/// The card of the file of `arch`. Two layers, and the Qwen interval of 2 makes layer 1 full.
pub fn card(comptime arch: loader.Arch) loader.ModelCard {
    var c: loader.ModelCard = .{
        .arch = arch,
        .architecture = @tagName(arch),
        .n_layers = 2,
        .n_heads = 2,
        .n_kv_heads = 1,
        .head_dim = 16,
        .embedding_length = 32,
        .feed_forward_length = 64,
        .context_length = 64,
        .vocab_size = tokens.len,
        .rope_freq_base = 10000.0,
        .rope_dim_count = 8,
        .rms_norm_eps = 1e-6,
        .bos_token_id = 0,
        .eos_token_id = 1,
        .chat_template = "",
    };
    if (arch == .qwen35) {
        c.gdn = .{
            .conv_kernel = 4,
            .state_size = 16,
            .group_count = 2,
            .inner_size = 32,
            .time_step_rank = 2,
            .full_attention_interval = 2,
        };
    }

    return c;
}

/// Write the file of the layer family `Family`. Llama ties its head and holds rope factors. Qwen
/// holds its own head.
pub fn build(arena: Allocator, comptime Family: type) !Built {
    const c = card(Family.arch);
    var out: std.ArrayList(u8) = .empty;
    var fields: std.ArrayList(Field) = .empty;
    var group: Field.Group = .card;
    const w: Writer = .{ .arena = arena, .out = &out, .fields = &fields, .group = &group };

    // Collect the tensors.
    var specs: [tensors_max]Spec = undefined;
    const n_specs = try tensorSpecs(arena, Family, &c, &specs);

    // Write the header and the card.
    const p = @tagName(Family.arch);
    // Eleven card keys and five tokenizer keys, then the six GDN keys of Qwen.
    const n_kvs: u64 = 16 + @as(u64, if (c.gdn != null) 6 else 0);
    try out.appendSlice(arena, "GGUF");
    try w.int(u32, 3);
    try w.int(u64, n_specs);
    try w.int(u64, n_kvs);
    try w.keyString("general.architecture", p);
    try w.keyU32(p ++ ".block_count", c.n_layers);
    try w.keyU32(p ++ ".attention.head_count", c.n_heads);
    try w.keyU32(p ++ ".attention.head_count_kv", c.n_kv_heads);
    try w.keyU32(p ++ ".embedding_length", c.embedding_length);
    try w.keyU32(p ++ ".feed_forward_length", c.feed_forward_length);
    try w.keyU32(p ++ ".context_length", c.context_length);
    try w.keyU32(p ++ ".attention.key_length", c.head_dim);
    try w.keyU32(p ++ ".rope.dimension_count", c.rope_dim_count);
    try w.keyF32(p ++ ".rope.freq_base", c.rope_freq_base);
    try w.keyF32(p ++ ".attention.layer_norm_rms_epsilon", c.rms_norm_eps);
    if (c.gdn) |g| {
        try w.keyU32(p ++ ".ssm.conv_kernel", g.conv_kernel);
        try w.keyU32(p ++ ".ssm.state_size", g.state_size);
        try w.keyU32(p ++ ".ssm.group_count", g.group_count);
        try w.keyU32(p ++ ".ssm.inner_size", g.inner_size);
        try w.keyU32(p ++ ".ssm.time_step_rank", g.time_step_rank);
        try w.keyU32(p ++ ".full_attention_interval", g.full_attention_interval);
    }

    // Write the tokenizer.
    group = .tokenizer;
    try w.key("tokenizer.ggml.tokens", type_array);
    try w.int(u32, type_string);
    try w.field(u64, tokens.len, .length);
    for (tokens) |t| {
        try w.string(t);
    }

    try w.key("tokenizer.ggml.token_type", type_array);
    try w.int(u32, type_i32);
    try w.field(u64, tokens.len, .length);
    for (tokens) |_| {
        try w.int(i32, 1);
    }

    try w.key("tokenizer.ggml.merges", type_array);
    try w.int(u32, type_string);
    try w.int(u64, 0);
    try w.keyU32("tokenizer.ggml.bos_token_id", c.bos_token_id);
    try w.keyU32("tokenizer.ggml.eos_token_id", c.eos_token_id);

    // Write the tensor table. Each tensor starts at an aligned offset of the data.
    group = .tensor;
    var offset: u64 = 0;
    for (specs[0..n_specs]) |s| {
        const n_dims: u32 = if (s.dims[1] == 0) 1 else 2;
        try w.string(s.name);
        try w.field(u32, n_dims, .int);
        for (s.dims[0..n_dims]) |d| {
            try w.field(u64, d, .int);
        }

        try w.field(u32, @intFromEnum(s.dtype), .int);
        try w.field(u64, offset, .offset);
        offset = std.mem.alignForward(u64, offset + specBytes(s), alignment);
    }

    // Write the data.
    try w.pad();
    for (specs[0..n_specs]) |s| {
        try writeData(w, s);
        try w.pad();
    }

    const bytes = try arena.alignedAlloc(u8, .fromByteUnits(alignment), out.items.len);
    @memcpy(bytes, out.items);
    return .{ .bytes = bytes, .fields = fields.items };
}

// -- helpers ----------------------------------------------------------------

/// The tensors of the file: every layer weight at the dims of the card, then the model tensors.
fn tensorSpecs(
    arena: Allocator,
    comptime Family: type,
    c: *const loader.ModelCard,
    specs: *[tensors_max]Spec,
) !usize {
    var n: usize = 0;
    var layer: u32 = 0;
    while (layer < c.n_layers) : (layer += 1) {
        for (Family.layerWeights(c, layer)) |role| {
            specs[n] = .{
                .name = try layerName(arena, Family, role, layer),
                .dims = Family.weightDims(c, role),
                .dtype = if (role.isMatrix()) .q8_0 else .f32,
                .fill = vectorFill(role),
            };
            n += 1;
        }
    }

    const emb = c.embedding_length;
    const table: [2]u32 = .{ emb, c.vocab_size };
    specs[n] = .{ .name = loader.embed_tensor, .dims = table, .dtype = .q8_0, .fill = 0 };
    const norm: [2]u32 = .{ emb, 0 };
    specs[n + 1] = .{ .name = loader.output_norm_tensor, .dims = norm, .dtype = .f32, .fill = 1 };
    specs[n + 2] = if (Family.rope_freqs) |name|
        .{ .name = name, .dims = .{ c.rope_dim_count / 2, 0 }, .dtype = .f32, .fill = 1 }
    else
        .{ .name = loader.head_tensor, .dims = table, .dtype = .q8_0, .fill = 0 };

    return n + 3;
}

fn layerName(arena: Allocator, comptime Family: type, role: Weight, layer: u32) ![]const u8 {
    var buf: [tensor_lookup.role_name_bytes_max]u8 = undefined;
    return arena.dupe(u8, tensor_lookup.roleName(Family, &buf, role, layer));
}

/// The value of every element of an f32 weight. The GDN decay rate is negative, so a step decays.
fn vectorFill(role: Weight) f32 {
    return switch (role) {
        .ssm_a => -1,
        .ssm_dt => 0,
        .ssm_conv1d => 0.25,
        else => 1,
    };
}

fn specBytes(s: Spec) u64 {
    const numel = @as(u64, s.dims[0]) * @max(s.dims[1], 1);
    return dtype.bytesForNumelChecked(s.dtype, numel).?;
}

fn writeData(w: Writer, s: Spec) !void {
    switch (s.dtype) {
        .f32 => {
            for (0..specBytes(s) / @sizeOf(f32)) |_| {
                try w.int(u32, @bitCast(s.fill));
            }
        },
        .q8_0 => {
            // Each block holds a scale, then 32 quants in -7..7.
            for (0..specBytes(s) / dtype.bytesPerBlock(.q8_0)) |b| {
                try w.int(u16, q8_scale_bits);
                for (0..dtype.blockSize(.q8_0)) |i| {
                    const q: i8 = @intCast(@as(i32, @intCast((b * 7 + i * 5) % 15)) - 7);
                    try w.int(i8, q);
                }
            }
        },
        else => unreachable, // The writer holds Q8_0 matrices and f32 vectors only.
    }
}
