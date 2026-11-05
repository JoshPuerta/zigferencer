//! `zigferencer-dev tokens <pub_name> --prompt "..."`: print the token ids of a prompt as
//! `[id, id, ...]`, the format of `llama-tokenize --ids`, so the two tokenizers diff in one
//! command.

const std = @import("std");
const zigferencer = @import("zigferencer");

const context = zigferencer.cli.context;
const args = zigferencer.cli.args;
const select = zigferencer.cli.select;
const registry = zigferencer.registry;
const loader = zigferencer.loader;
const vocab_mod = zigferencer.vocab;
const bpe = zigferencer.bpe;
const chat_mod = zigferencer.chat;

// -- types ------------------------------------------------------------------

/// The flags of `tokens`.
const Flags = struct {
    pub_name: ?[]const u8 = null,
    prompt: []const u8 = "Hello",
    chat: bool = false,

    pub const positional = "pub_name";
    pub const help: args.Help = .{
        .usage = "zigferencer-dev tokens <pub_name> [flags]",
        .about = "Print the token ids of a prompt. The output matches `llama-tokenize " ++
            "--ids`\nfor side-by-side diffing.\n",
    };
    pub const docs = .{
        .prompt = args.Doc{ .text = "text to tokenize" },
        .chat = args.Doc{ .text = "wrap with the chat template of the model first" },
    };
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: context.Context) !void {
    const flags = try args.parseInto(Flags, ctx) orelse return;

    const model = try select.model(ctx, flags.pub_name);
    const path = try registry.resolveLoadPath(ctx.arena, ctx.io, ctx.models_root, model.*);

    // Tokenizing reads only the header KVs, so the load skips the page prefault.
    var loaded = try loader.loadWith(ctx.io, ctx.arena, path, .{ .prefault = false });
    defer loaded.close();

    var v = try vocab_mod.buildFromParsed(ctx.arena, loaded.parsed);
    const text = if (flags.chat)
        try chat_mod.wrapAlloc(ctx.arena, model.chat_template_kind, flags.prompt)
    else
        flags.prompt;

    const ids = try bpe.encodeAlloc(&v, ctx.arena, text, true);

    const w = &ctx.stdout.interface;

    try w.writeAll("[");
    for (ids, 0..) |id, i| {
        if (i > 0) {
            try w.writeAll(", ");
        }

        try w.print("{d}", .{id});
    }

    try w.writeAll("]\n");
    try w.flush();
}
