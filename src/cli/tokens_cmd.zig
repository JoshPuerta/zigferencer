//! `zigferencer tokens <pub_name> --prompt "..."`: print the encoder's token ids
//! for a prompt as `[id, id, ...]`, the format of llama.cpp's `llama-tokenize
//! --ids`. The correctness model treats llama.cpp as the external oracle
//! (docs/architecture.md), and tokenization is the first place the two stacks can
//! diverge. This surface makes the comparison one command per side, with no
//! debugger.

const std = @import("std");
const Io = std.Io;

const cli = @import("command.zig");
const args = @import("args.zig");
const style = @import("style.zig");
const select = @import("select.zig");
const registry = @import("../core/registry.zig");
const loader = @import("../io/gguf/loader.zig");
const vocab_mod = @import("../core/tokenizer/vocab.zig");
const bpe = @import("../core/tokenizer/bpe.zig");
const chat_mod = @import("../core/chat.zig");

// -- types ------------------------------------------------------------------

/// The flags of `tokens`.
const Flags = struct {
    pub_name: ?[]const u8 = null,
    prompt: []const u8 = "Hello",
    chat: bool = false,

    pub const positional = "pub_name";
    pub const help: args.Help = .{
        .usage = "zigferencer-dev tokens <pub_name> [flags]",
        .about = "Print the token ids of a prompt. The output matches `llama-tokenize --ids`\nfor side-by-side diffing.\n",
    };
    pub const docs = .{
        .prompt = args.Doc{ .text = "text to tokenize" },
        .chat = args.Doc{ .text = "wrap with the chat template of the model first" },
    };
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: cli.Context) !void {
    const flags = try args.parseInto(Flags, ctx) orelse return;
    const prompt = flags.prompt;
    const chat = flags.chat;
    const pub_name = flags.pub_name;

    const model = try select.model(ctx, pub_name);
    const path = try registry.resolveLoadPath(ctx.arena, ctx.io, ctx.models_root, model.*);
    // Tokenizing reads only the header KVs, so the load skips the page prefault.
    var loaded = try loader.loadWith(ctx.io, ctx.arena, path, .{ .prefault = false });
    defer loaded.close();

    var v = try vocab_mod.buildFromParsed(ctx.arena, loaded.parsed);
    const text = if (chat)
        try chat_mod.wrapAlloc(ctx.arena, model.chat_template_kind, prompt)
    else
        prompt;
    const ids = try bpe.encodeAlloc(&v, ctx.arena, text, true);

    var buf: [4096]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stdout(), ctx.io, &buf);
    const w = &fw.interface;
    try w.writeAll("[");
    for (ids, 0..) |id, i| {
        if (i > 0) try w.writeAll(", ");
        try w.print("{d}", .{id});
    }
    try w.writeAll("]\n");
    try w.flush();
}

// -- helpers ----------------------------------------------------------------
