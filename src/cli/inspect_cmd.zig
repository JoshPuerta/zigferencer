//! `zigferencer inspect <pub_name>`: mmap and parse a fetched GGUF, then print its
//! `ModelCard`, the weight inventory, and a chat-template preview. It shows whether
//! the loader accepts the file and which values the runner reads.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const context = @import("context.zig");
const args = @import("args.zig");
const style = @import("style.zig");
const select = @import("select.zig");
const registry = @import("../core/registry.zig");
const dtype = @import("../core/dtype.zig");
const loader = @import("../io/gguf/loader.zig");
const mmap = @import("../io/mmap.zig");
const parser = @import("../io/gguf/parser.zig");
const Tensor = @import("../core/tensor.zig").Tensor;

// -- types ------------------------------------------------------------------

/// A block tensor's layer index and role, split out of `blk.<n>.<role>`.
const BlockRef = struct { layer: u32, role: []const u8 };

/// One role of one layer, with the tensor that carries it.
const RoleRef = struct { role: []const u8, tensor: Tensor };

/// The flags of `inspect`.
const Flags = struct {
    pub_name: ?[]const u8 = null,
    raw: ?[]const u8 = null,

    pub const positional = "pub_name";
    pub const help: args.Help = .{
        .usage = "zigferencer inspect <pub_name> [flags]   (<pub_name> omitted -> pick a model)",
        .about =
        \\Pretty-print a parsed ModelCard (arch, dims, RoPE, vocab, BOS/EOS), the
        \\weight inventory, and the chat-template preview.
        \\
        ,
        .footer =
        \\examples:
        \\  zigferencer inspect SmolLM2-135M-Instruct-Q8_0
        \\  zigferencer inspect --raw ./models/some-model.gguf
        \\
        ,
    };
    pub const docs = .{
        .raw = args.Doc{
            .value = "PATH",
            .text = "dump every raw KV + tensor straight from the parser, bypassing\nthe " ++
                "loader, for an architecture the loader does not accept",
        },
    };
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: context.Context) !void {
    const flags = try args.parseInto(Flags, ctx) orelse return;

    // The raw dump skips the loader, so it can show an architecture the loader refuses.
    if (flags.raw) |path| return dumpRaw(ctx, path);

    const model = try select.model(ctx, flags.pub_name);
    const path = try registry.resolveLoadPath(ctx.arena, ctx.io, ctx.models_root, model.*);

    // Inspect reads the card and the tensor descriptors, never the weights, so the map stays lazy.
    var loaded = try loader.loadWith(ctx.io, ctx.arena, path, .{ .prefault = false });
    defer loaded.close();

    const w = &ctx.stdout.interface;

    try printCard(w, model.pub_name, path, &loaded);
    try w.writeAll("\n");

    try printTensorStructure(w, ctx.arena, &loaded);
    try w.writeAll("\n");

    try printChatTemplate(w, loaded.card.chat_template);

    try w.flush();
}

// -- helpers ----------------------------------------------------------------

fn printCard(
    w: *Io.Writer,
    pub_name: []const u8,
    path: []const u8,
    loaded: *const loader.Loaded,
) !void {
    const c = loaded.card;
    try w.print("model:           {s}\n", .{pub_name});
    if (c.name) |n| {
        try w.print("name:            {s}\n", .{n});
    }

    try w.print("file:            {s}\n", .{path});
    try w.print("file size:       {d} bytes\n", .{loaded.map.bytes().len});
    try w.writeAll("\n");

    try w.print("architecture:    {s}\n", .{c.architecture});
    try w.print("layers:          {d}\n", .{c.n_layers});
    try w.print("heads:           {d}  (kv heads: {d})\n", .{ c.n_heads, c.n_kv_heads });
    try w.print("head dim:        {d}\n", .{c.head_dim});
    try w.print("embedding dim:   {d}\n", .{c.embedding_length});
    try w.print("ffn dim:         {d}\n", .{c.feed_forward_length});
    try w.print("context length:  {d}\n", .{c.context_length});
    try w.print("vocab size:      {d}\n", .{c.vocab_size});
    try w.print("rope freq base:  {d}\n", .{c.rope_freq_base});
    try w.print("rope dim count:  {d}\n", .{c.rope_dim_count});
    try w.print("rms-norm eps:    {e}\n", .{c.rms_norm_eps});
    try w.print("bos id:          {d}\n", .{c.bos_token_id});
    try w.print("eos id:          {d}\n", .{c.eos_token_id});

    if (c.gdn) |g| {
        try w.writeAll("\ngated-deltanet (qwen35):\n");
        try w.print("  conv_kernel:     {d}\n", .{g.conv_kernel});
        try w.print("  state_size:      {d}  (per-head state dim)\n", .{g.state_size});
        try w.print("  group_count:     {d}  (gdn heads)\n", .{g.group_count});
        try w.print("  inner_size:      {d}\n", .{g.inner_size});
        try w.print("  time_step_rank:  {d}\n", .{g.time_step_rank});
        try w.print("  full_attn_every: {d}  (full-attn layers: ", .{g.full_attention_interval});
        var li: u32 = 0;
        while (li < c.n_layers) : (li += 1) {
            if (g.isFullAttention(li)) {
                try w.print("{d} ", .{li});
            }
        }

        try w.writeAll(")\n");
    }
}

fn printChatTemplate(w: *Io.Writer, template: []const u8) !void {
    try w.print("chat template ({d} bytes):\n", .{template.len});
    if (template.len == 0) {
        try w.writeAll("  (none embedded)\n");
        return;
    }

    try w.writeAll(template);
    if (template[template.len - 1] != '\n') {
        try w.writeAll("\n");
    }
}

fn dumpRaw(ctx: context.Context, path: []const u8) !void {
    var map = try mmap.openReadOnlyLazy(ctx.io, path);
    defer map.close();

    const parsed = try parser.parse(ctx.arena, map.bytes());

    const w = &ctx.stdout.interface;

    try w.print("file:            {s}\n", .{path});
    try w.print("file size:       {d} bytes\n", .{map.bytes().len});
    try w.print("gguf version:    {d}\n", .{parsed.header.version});
    try w.print("kv count:        {d}\n", .{parsed.header.kv_count});
    try w.print("tensor count:    {d}\n", .{parsed.header.tensor_count});

    try w.writeAll("\n=== metadata (KV) ===\n");
    for (parsed.kvs) |kv| {
        try w.print("  {s} = ", .{kv.key});
        try printValue(w, kv.value);
        try w.writeAll("\n");
    }

    try w.writeAll("\n=== tensors ===\n");
    for (parsed.tensors) |ti| {
        const type_name = std.enums.tagName(dtype.GgmlType, ti.dtype) orelse "unknown";
        try w.print("  {s:<44}  {s:<6}  [", .{ ti.name, type_name });
        for (ti.shape[0..ti.n_dims], 0..) |d, i| {
            if (i > 0) {
                try w.writeAll(", ");
            }

            try w.print("{d}", .{d});
        }

        try w.writeAll("]\n");
    }

    try w.flush();
}

/// One-line preview of a parsed KV value. Big arrays (tokenizer vocab, merges)
/// print their length only. Small numeric arrays print their elements.
fn printValue(w: *std.Io.Writer, v: parser.Value) !void {
    switch (v) {
        .u8_val => |x| try w.print("{d} (u8)", .{x}),
        .i8_val => |x| try w.print("{d} (i8)", .{x}),
        .u16_val => |x| try w.print("{d} (u16)", .{x}),
        .i16_val => |x| try w.print("{d} (i16)", .{x}),
        .u32_val => |x| try w.print("{d} (u32)", .{x}),
        .i32_val => |x| try w.print("{d} (i32)", .{x}),
        .u64_val => |x| try w.print("{d} (u64)", .{x}),
        .i64_val => |x| try w.print("{d} (i64)", .{x}),
        .f32_val => |x| try w.print("{d} (f32)", .{x}),
        .f64_val => |x| try w.print("{d} (f64)", .{x}),
        .bool_val => |x| try w.print("{} (bool)", .{x}),
        .string => |s| {
            if (s.len <= 96) {
                try w.print("\"{s}\" (string)", .{s});
            } else {
                try w.print("\"{s}...\" (string, {d} bytes)", .{ s[0..96], s.len });
            }
        },
        .array => |a| try printArray(w, a),
    }
}

fn printArray(w: *std.Io.Writer, a: parser.Array) !void {
    switch (a) {
        .strings => |s| {
            try w.print("[{d} strings]", .{s.len});
            if (s.len > 0) {
                const e = s[0];
                try w.print(" e.g. \"{s}\"", .{e[0..@min(e.len, 24)]});
            }
        },
        inline else => |slice| {
            try w.print("[{d} x {s}]", .{ slice.len, @typeName(@TypeOf(slice)) });
            const n = @min(slice.len, 16);
            if (n > 0) {
                try w.writeAll(" {");
                for (slice[0..n], 0..) |e, i| {
                    if (i > 0) {
                        try w.writeAll(", ");
                    }

                    try w.print("{any}", .{e});
                }

                if (slice.len > n) {
                    try w.writeAll(", ...");
                }

                try w.writeAll("}");
            }
        },
    }
}

/// Split a GGUF tensor name `blk.<n>.<role>` into its layer index and role.
/// Returns null for a model-level tensor such as `token_embd.weight`.
fn parseBlock(name: []const u8) ?BlockRef {
    if (!std.mem.startsWith(u8, name, "blk.")) return null;

    const rest = name["blk.".len..];
    const dot = std.mem.findScalar(u8, rest, '.') orelse return null;
    const layer = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
    return .{ .layer = layer, .role = rest[dot + 1 ..] };
}

/// Print the non-layer tensors, then one entry per distinct layer type with the
/// block indices it covers. Two layers share a type when their sorted tensor roles match.
fn printTensorStructure(w: *std.Io.Writer, arena: Allocator, loaded: *const loader.Loaded) !void {
    const tensors = loaded.tensors;
    const n_layers: usize = loaded.card.n_layers;

    try w.print("tensors:         {d}\n", .{tensors.len});

    try w.writeAll("\npre-block weights (embedding):\n");
    try printGlobals(w, tensors, true);
    try w.writeAll("\npost-block weights (final norm + output head):\n");
    try printGlobals(w, tensors, false);

    // Bucket each block's tensors by layer index.
    const per_layer = try arena.alloc(std.ArrayList(RoleRef), n_layers);
    for (per_layer) |*r| {
        r.* = .empty;
    }

    for (tensors) |ref| {
        const b = parseBlock(ref.name) orelse continue;
        if (b.layer < n_layers) {
            try per_layer[b.layer].append(arena, .{ .role = b.role, .tensor = ref.tensor });
        }
    }

    // Sorted roles keep a quantizer that reorders the tensors of a few blocks from splitting a
    // layer type.
    for (per_layer) |*r| {
        std.mem.sort(RoleRef, r.items, {}, roleLess);
    }

    const sigs = try arena.alloc([]const u8, n_layers);
    for (0..n_layers) |i| {
        sigs[i] = try signatureOf(arena, per_layer[i].items);
    }

    try w.writeAll("\nper-layer weights (grouped by layer type):\n");
    const done = try arena.alloc(bool, n_layers);
    @memset(done, false);
    for (0..n_layers) |i| {
        if (done[i]) continue;

        const roles = per_layer[i].items;
        if (roles.len == 0) {
            done[i] = true;
            continue;
        }

        var members: std.ArrayList(u32) = .empty;
        for (i..n_layers) |j| {
            if (!done[j] and std.mem.eql(u8, sigs[j], sigs[i])) {
                try members.append(arena, @intCast(j));
                done[j] = true;
            }
        }

        try w.print("\n  type: {d} weights x {d} layer{s} (blk ", .{
            roles.len,
            members.items.len,
            if (members.items.len == 1) "" else "s",
        });
        try printIndexRanges(w, members.items);
        try w.writeAll("):\n");
        for (roles) |rr| {
            try printTensorRow(w, rr.role, rr.tensor);
        }
    }
}

/// Print the non-block tensors. `embedding` selects the input side
/// (`token_embd*`) over the output side (everything else without a `blk.` prefix).
fn printGlobals(w: *std.Io.Writer, tensors: []const loader.TensorRef, embedding: bool) !void {
    var n: usize = 0;
    for (tensors) |ref| {
        if (parseBlock(ref.name) != null) continue;

        const is_embd = std.mem.startsWith(u8, ref.name, "token_embd");
        if (is_embd != embedding) continue;

        try printTensorRow(w, ref.name, ref.tensor);
        n += 1;
    }

    if (n == 0) {
        try w.writeAll("  (none)\n");
    }
}

fn roleLess(_: void, a: RoleRef, b: RoleRef) bool {
    return std.mem.order(u8, a.role, b.role) == .lt;
}

fn signatureOf(arena: Allocator, roles: []const RoleRef) ![]const u8 {
    var sig: std.ArrayList(u8) = .empty;
    for (roles) |rr| {
        try sig.appendSlice(arena, rr.role);
        try sig.append(arena, '|');
    }

    return sig.items;
}

/// Render a sorted index list as compact ranges: 0,1,2,3 -> "0-3", and 0,1,3 -> "0-1,3".
fn printIndexRanges(w: *std.Io.Writer, idx: []const u32) !void {
    var k: usize = 0;
    var first = true;
    while (k < idx.len) {
        const start = idx[k];
        var end = start;
        while (k + 1 < idx.len and idx[k + 1] == end + 1) {
            k += 1;
            end = idx[k];
        }

        if (!first) {
            try w.writeAll(",");
        }

        first = false;
        if (start == end) {
            try w.print("{d}", .{start});
        } else {
            try w.print("{d}-{d}", .{ start, end });
        }

        k += 1;
    }
}

fn printTensorRow(w: *std.Io.Writer, label: []const u8, t: Tensor) !void {
    const type_name = std.enums.tagName(dtype.GgmlType, t.dtype) orelse "unknown";
    try w.print("  {s:<24} {s:<5} ", .{ label, type_name });
    try printShape(w, t);
    try w.writeAll("\n");
}

fn printShape(w: *std.Io.Writer, t: Tensor) !void {
    try w.writeAll("[");
    for (t.shape[0..t.n_dims], 0..) |d, i| {
        if (i > 0) {
            try w.writeAll(", ");
        }

        try w.print("{d}", .{d});
    }

    try w.writeAll("]");
}

// -- tests ------------------------------------------------------------------

test "parseBlock splits blk.<n>.<role> and rejects non-block names" {
    const b = parseBlock("blk.7.attn_q.weight") orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(u32, 7), b.layer);
    try std.testing.expectEqualStrings("attn_q.weight", b.role);
    try std.testing.expect(parseBlock("token_embd.weight") == null);
    try std.testing.expect(parseBlock("output.weight") == null);
    try std.testing.expect(parseBlock("blk.x.weight") == null); // non-numeric index
}
