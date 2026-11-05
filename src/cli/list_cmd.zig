//! `zigferencer list`: print every registry model, whether its GGUF is on disk, the path it loads
//! from, and its download URL. A missing model shows as "[-] not downloaded", never as an empty
//! list, so a wrong `$ZIGFERENCER_MODELS_DIR` is easy to see.

const std = @import("std");
const Io = std.Io;

const context = @import("context.zig");
const args = @import("args.zig");
const select = @import("select.zig");
const style = @import("style.zig");
const registry = @import("../core/registry.zig");

// -- types ------------------------------------------------------------------

/// The flags of `list`.
const Flags = struct {
    present: bool = false,

    pub const help: args.Help = .{
        .usage = "zigferencer list [flags]",
        .about = "List the registered models and whether each one is on disk.\n",
    };
    pub const docs = .{
        .present = args.Doc{
            .text = "only models on disk, bare names, one per line (for scripts)",
        },
    };
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: context.Context) !void {
    const flags = try args.parseInto(Flags, ctx) orelse return;
    if (flags.present) return printPresent(ctx);

    const p = style.Palette.init(ctx.color);

    const w = &ctx.stdout.interface;

    const models = registry.all();
    const default_root = std.mem.eql(u8, ctx.models_root, registry.default_models_root);
    try w.print(
        "{s}Registered models ({d}){s}   {s}[+]{s} present on disk  {s}[-] not downloaded{s}\n",
        .{
            p.bold, models.len, p.reset, p.green, p.reset, p.dim, p.reset,
        },
    );
    try select.printModelsDir(w, p, ctx.models_root);
    try w.writeAll("\n");

    var present_count: usize = 0;
    for (models) |m| {
        // A non-null path means the GGUF is on disk there.
        const path = try registry.locate(ctx.arena, ctx.io, ctx.models_root, m);
        if (path != null) {
            present_count += 1;
        }

        try printModel(w, p, m, path);
    }

    try w.print("\n{d}/{d} present.\n", .{ present_count, models.len });
    if (present_count < models.len) {
        try w.writeAll("download a missing model with `zigferencer fetch <name>`");
        if (default_root) {
            try w.print(", or set ${s} if your GGUFs live elsewhere", .{registry.models_dir_env});
        }

        try w.writeAll(".\n");
    }

    try w.flush();
}

// -- helpers ----------------------------------------------------------------

/// `--present`: the bare pub_names of the on-disk models, one per line, unstyled.
fn printPresent(ctx: context.Context) !void {
    const w = &ctx.stdout.interface;

    for (registry.all()) |m| {
        if ((try registry.locate(ctx.arena, ctx.io, ctx.models_root, m)) != null) {
            try w.print("{s}\n", .{m.pub_name});
        }
    }

    try w.flush();
}

fn printModel(w: *Io.Writer, p: style.Palette, m: registry.Model, path: ?[]const u8) !void {
    const mark_c = if (path != null) p.green else p.dim;
    const mark: []const u8 = if (path != null) "+" else "-";
    try w.print("  {s}[{s}]{s} {s}\n", .{ mark_c, mark, p.reset, m.pub_name });

    if (path) |loc| {
        try w.print("      {s}path:   {s}{s}\n", .{ p.dim, loc, p.reset });
    } else {
        try w.print("      {s}path:{s}\n", .{ p.dim, p.reset });
    }

    try w.print("      {s}remote: {s}{s}\n", .{ p.dim, m.download_url, p.reset });
}
