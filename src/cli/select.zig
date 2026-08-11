//! Model selection for the CLI: turn an optional `<pub_name>` argument into a
//! registry model, and prompt for one when the argument is absent. Every
//! model-consuming subcommand (inspect, run, bench-model) routes through here.
//! The unknown-model error, the "which model?" picker, and the models-dir banner
//! therefore read the same across the tool.
//!
//! The load picker offers only the models already on disk, the ones a load
//! command can open. `fetchPick` offers every registry entry, because the point of
//! fetch is to download one that is absent. Both name the models dir and
//! $ZIGFERENCER_MODELS_DIR. A store pointed at the wrong disk is therefore
//! obvious at the moment of choosing, not later as a "not found".

const std = @import("std");
const Io = std.Io;

const cli = @import("command.zig");
const args = @import("args.zig");
const style = @import("style.zig");
const registry = @import("../core/registry.zig");
const prompt = @import("prompt.zig");
const loader = @import("../io/gguf/loader.zig");
const limits = @import("../core/limits.zig");
const kv_budget = @import("../core/kv_budget.zig");

// -- types ------------------------------------------------------------------

/// The windows of one picker row, in positions: the model ceiling and the two
/// default windows at the registry KV dtype.
const Windows = struct {
    ceiling: u32,
    run: u32,
    serve: u32,
};

// -- api --------------------------------------------------------------------

/// Resolve a load command's model. A given name goes through the registry lookup,
/// with the one unknown-model error on a miss. With no name, a TTY gets the
/// on-disk models and a choice prompt. A non-TTY has nobody to ask, so the
/// function explains and fails. The returned pointer is into the static registry
/// table.
pub fn model(ctx: cli.Context, maybe_name: ?[]const u8) !*const registry.Model {
    if (maybe_name) |name| {
        return registry.lookup(name) orelse {
            unknownModel(ctx.io, name);
            return error.UnknownModel;
        };
    }
    if (!prompt.stdinIsTty(ctx.io)) {
        args.writeStderr(ctx.io,
            \\error: no model given
            \\       pass a <pub_name> (run 'zigferencer list'), or run interactively to pick one
            \\
        );
        return error.UsageError;
    }
    return pickPresent(ctx);
}

/// The no-argument picker for fetch: list every registry model with a present
/// marker, read a choice, and return the chosen `pub_name`. Null means the user
/// aborted. This picker also offers the absent models, because fetch exists to
/// download one. Enter has no default, so one accidental keystroke cannot start a
/// multi-GB download.
pub fn fetchPick(ctx: cli.Context) !?[]const u8 {
    const p = style.Palette.init(ctx.color_err);
    const models = registry.all();

    var buf: [4096]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stderr(), ctx.io, &buf);
    const w = &fw.interface;
    try printModelsDir(w, p, ctx.models_root);
    try w.print("{s}select a model to fetch{s}  {s}[+]{s}{s} = already on disk{s}\n", .{
        p.bold, p.reset, p.green, p.reset, p.dim, p.reset,
    });
    for (models, 1..) |m, label| {
        const present = (try registry.locate(ctx.arena, ctx.io, ctx.models_root, m)) != null;
        const mark_c = if (present) p.green else p.dim;
        const mark: []const u8 = if (present) "+" else "-";
        try w.print("  {s}[{s}]{s} {s}{d:>2}{s}  {s}\n", .{ mark_c, mark, p.reset, p.bold, label, p.reset, m.pub_name });
    }
    try w.print("{s}[>]{s} choose [1-{d}] {s}(Enter cancels){s}: ", .{ p.bold, p.reset, models.len, p.dim, p.reset });
    try w.flush();

    const idx = prompt.chooseIndex(ctx.io, models.len, null) orelse return null;
    return models[idx].pub_name;
}

/// Print the models-dir search order, one line. The writer belongs to the caller,
/// so `list` prints it to stdout and the pickers print it to stderr.
pub fn printModelsDir(w: *std.Io.Writer, p: style.Palette, models_root: []const u8) !void {
    const is_default = std.mem.eql(u8, models_root, registry.default_models_root);
    if (is_default) {
        try w.print("{s}[*] models dir: {s}  (second store: ${s}){s}\n", .{
            p.dim, models_root, registry.models_dir_env, p.reset,
        });
    } else {
        try w.print("{s}[*] models dirs: {s}, then ${s}{s}\n", .{
            p.dim, registry.default_models_root, registry.models_dir_env, p.reset,
        });
    }
}

/// The one unknown-model error every command prints: the bad name and the pointer
/// to `list`. The write to stderr is best-effort (see `args.writeStderr`). The
/// caller returns `error.UnknownModel` in every case.
fn unknownModel(io: Io, name: []const u8) void {
    var buf: [512]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stderr(), io, &buf);
    const w = &fw.interface;
    w.print("error: model '{s}' is not in the registry\n", .{name}) catch return;
    w.writeAll("       (run 'zigferencer list' for available names)\n") catch return;
    w.flush() catch {};
}

// -- helpers ----------------------------------------------------------------

/// Present the on-disk models on stderr and return the chosen one. `locate` spans
/// the configured root and the repo-local `models/` directory, so the picker
/// offers a model exactly when a load command can open it. With none on disk, the
/// function points at fetch and fails.
fn pickPresent(ctx: cli.Context) !*const registry.Model {
    const p = style.Palette.init(ctx.color_err);
    const models = registry.all();
    const present = try ctx.arena.alloc(*const registry.Model, models.len);
    const windows = try ctx.arena.alloc(?Windows, models.len); // null if unreadable
    var n: usize = 0;
    var name_w: usize = 0;
    for (models) |*m| {
        const path = (try registry.locate(ctx.arena, ctx.io, ctx.models_root, m.*)) orelse continue;
        present[n] = m;
        // The read is metadata-only (lazy mmap, no weight pages), the low-cost
        // path `inspect` uses. A model whose card does not parse still lists,
        // without its window numbers.
        windows[n] = readWindows(ctx, path, m);
        if (m.pub_name.len > name_w) name_w = m.pub_name.len;
        n += 1;
    }

    var buf: [4096]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stderr(), ctx.io, &buf);
    const w = &fw.interface;
    try printModelsDir(w, p, ctx.models_root);
    if (n == 0) {
        try w.print("{s}[-]{s} no models on disk yet. Download one with '{s}zigferencer fetch <name>{s}'.\n", .{
            p.dim, p.reset, p.bold, p.reset,
        });
        try w.flush();
        return error.NoModelsPresent;
    }
    const mib = 1024 * 1024;
    try w.print("{s}select a model{s}  {s}(in positions: ctx = model ceiling, run and serve = default windows from {d} and {d} MiB of KV per session){s}\n", .{
        p.bold, p.reset, p.dim, limits.run_session_kv_bytes / mib, limits.serve_session_kv_bytes / mib, p.reset,
    });
    for (present[0..n], windows[0..n], 1..) |m, win, label| {
        try w.print("  {s}[+]{s} {s}{d:>2}{s}  {s}{s}  ", .{ p.green, p.reset, p.bold, label, p.reset, m.pub_name, padding(name_w - m.pub_name.len) });
        if (win) |x| {
            try w.print("{s}ctx {d:>6}  run {d:>6}  serve {d:>5}{s}\n", .{ p.dim, x.ceiling, x.run, x.serve, p.reset });
        } else {
            try w.print("{s}ctx ?{s}\n", .{ p.dim, p.reset });
        }
    }
    try w.print("{s}[>]{s} choose [1-{d}] {s}(Enter = 1){s}: ", .{ p.bold, p.reset, n, p.dim, p.reset });
    try w.flush();

    const idx = prompt.chooseIndex(ctx.io, n, 0) orelse {
        w.print("{s}[-]{s} no selection. Aborted\n", .{ p.dim, p.reset }) catch {};
        w.flush() catch {};
        return error.UsageError;
    };
    return present[idx];
}

/// A model's ceiling and its two default windows, read through the
/// metadata-only lazy mmap that `inspect` uses. The read faults no weight
/// pages. The windows take the rules `run` and serve apply when no flag names a
/// window. Returns null when the file does not open or its card does not parse,
/// so the picker still lists the model without numbers.
fn readWindows(ctx: cli.Context, path: []const u8, m: *const registry.Model) ?Windows {
    var loaded = loader.loadWith(ctx.io, ctx.arena, path, .{ .prefault = false, .quiet = true }) catch return null;
    defer loaded.close();
    const cost = kv_budget.costModel(loaded.card.kvShape(), m.kv_dtype_default);
    const ceiling = limits.contextCeiling(loaded.card.context_length, m.context_limit);
    const windows = kv_budget.defaultWindows(cost, ceiling);
    return .{ .ceiling = ceiling, .run = windows.run, .serve = windows.serve };
}

/// `n` spaces for column alignment. `pub_name`s are bounded well under 64.
fn padding(n: usize) []const u8 {
    const spaces = " " ** 64;
    return spaces[0..@min(n, spaces.len)];
}
