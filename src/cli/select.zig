//! Model selection for the CLI: turn an optional `<pub_name>` into a registry model, and prompt for
//! one when it is absent. Every model-consuming command routes through here, so the unknown-model
//! error, the picker, and the models-dir banner read the same everywhere.
//!
//! The load picker offers only models on disk. `fetchPick` offers every registry entry, because
//! fetch downloads the absent ones. Both print the models dir, so a store on the wrong disk shows
//! at the moment of choosing.

const std = @import("std");
const Io = std.Io;

const context = @import("context.zig");
const args = @import("args.zig");
const style = @import("style.zig");
const registry = @import("../core/registry.zig");
const prompt = @import("prompt.zig");
const loader = @import("../io/gguf/loader.zig");
const limits = @import("../core/limits.zig");
const kv_budget = @import("../core/kv_budget.zig");

// -- constants --------------------------------------------------------------

/// The stack buffer of one picker writer on stderr.
const picker_buffer_bytes = 4096;

/// The stack buffer of the unknown-model error.
const error_buffer_bytes = 512;

const bytes_per_mib = 1024 * 1024;

/// The longest `pub_name` column the picker can pad.
const name_pad_max = 64;

// -- types ------------------------------------------------------------------

/// The windows of one picker line, in positions: the model ceiling and the two default windows.
const Windows = struct {
    ceiling: u32,
    run: u32,
    serve: u32,
};

/// The outcome of a named selection. `model` prints each refusal, so a test reads the outcome
/// without output.
const Named = union(enum) {
    found: *const registry.Model,
    unknown,
    absent: registry.Candidates,
};

// -- api --------------------------------------------------------------------

/// Resolve a load command's model. A name goes through the registry lookup and must name a file
/// on disk. With no name, a TTY gets the picker and a non-TTY gets an error. The pointer is into
/// the static registry table.
pub fn model(ctx: context.Context, maybe_name: ?[]const u8) !*const registry.Model {
    if (maybe_name) |name| {
        switch (try named(ctx, name)) {
            .found => |m| return m,
            .unknown => {
                unknownModel(ctx.io, name);
                return error.UnknownModel;
            },
            .absent => |c| {
                notOnDisk(ctx.io, name, c);
                return error.ModelNotOnDisk;
            },
        }
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

/// The no-argument picker for fetch: return the chosen `pub_name`, or null when the user aborts.
/// Enter has no default, so one keystroke cannot start a multi-GB download.
pub fn fetchPick(ctx: context.Context) !?[]const u8 {
    // Print the menu of every registry model.
    const p = style.Palette.init(ctx.color_err);
    const models = registry.all();

    var stderr_buffer: [picker_buffer_bytes]u8 = undefined;
    var stderr_writer: Io.File.Writer = .initStreaming(.stderr(), ctx.io, &stderr_buffer);
    const w = &stderr_writer.interface;

    try printModelsDir(w, p, ctx.models_root);
    try w.print("{s}select a model to fetch{s}  {s}[+]{s}{s} = already on disk{s}\n", .{
        p.bold, p.reset, p.green, p.reset, p.dim, p.reset,
    });
    for (models, 1..) |m, label| {
        const present = (try registry.locate(ctx.arena, ctx.io, ctx.models_root, m)) != null;
        const mark_c = if (present) p.green else p.dim;
        const mark: []const u8 = if (present) "+" else "-";
        try w.print(
            "  {s}[{s}]{s} {s}{d:>2}{s}  {s}\n",
            .{ mark_c, mark, p.reset, p.bold, label, p.reset, m.pub_name },
        );
    }

    try w.print(
        "{s}[>]{s} choose [1-{d}] {s}(Enter cancels){s}: ",
        .{ p.bold, p.reset, models.len, p.dim, p.reset },
    );
    try w.flush();

    // Read the choice.
    const idx = prompt.chooseIndex(ctx.io, models.len, null) orelse return null;
    return models[idx].pub_name;
}

/// Print the models-dir search order, one line, to the writer of the caller.
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

// -- helpers ----------------------------------------------------------------

/// Present the on-disk models on stderr and return the chosen one. `locate` spans the same roots as
/// a load command, so the picker offers a model exactly when a load can open it.
fn pickPresent(ctx: context.Context) !*const registry.Model {
    // Collect the models on disk with their windows.
    const p = style.Palette.init(ctx.color_err);
    const models = registry.all();
    const present = try ctx.arena.alloc(*const registry.Model, models.len);
    const windows = try ctx.arena.alloc(?Windows, models.len);
    var n: usize = 0;
    var name_w: usize = 0;
    for (models) |*m| {
        const path = (try registry.locate(ctx.arena, ctx.io, ctx.models_root, m.*)) orelse continue;
        present[n] = m;
        windows[n] = readWindows(ctx, path, m);
        if (m.pub_name.len > name_w) {
            name_w = m.pub_name.len;
        }

        n += 1;
    }

    // Print the menu, or refuse when no model is on disk.
    var stderr_buffer: [picker_buffer_bytes]u8 = undefined;
    var stderr_writer: Io.File.Writer = .initStreaming(.stderr(), ctx.io, &stderr_buffer);
    const w = &stderr_writer.interface;

    try printModelsDir(w, p, ctx.models_root);
    if (n == 0) {
        try w.print(
            "{s}[-]{s} no models on disk yet. " ++
                "Download one with '{s}zigferencer fetch <name>{s}'.\n",
            .{
                p.dim, p.reset, p.bold, p.reset,
            },
        );
        try w.flush();
        return error.NoModelsPresent;
    }

    try printMenu(w, p, present[0..n], windows[0..n], name_w);

    // Read the choice.
    const idx = prompt.chooseIndex(ctx.io, n, 0) orelse {
        w.print("{s}[-]{s} no selection. Aborted\n", .{ p.dim, p.reset }) catch {};
        w.flush() catch {};
        return error.UsageError;
    };
    return present[idx];
}

fn printMenu(
    w: *Io.Writer,
    p: style.Palette,
    present: []const *const registry.Model,
    windows: []const ?Windows,
    name_w: usize,
) !void {
    // Print the heading.
    try w.print(
        "{s}select a model{s}  {s}(in positions: " ++
            "ctx = model ceiling, run and serve = default windows from {d} " ++
            "and {d} MiB of KV per session){s}\n",
        .{
            p.bold,
            p.reset,
            p.dim,
            limits.run_session_kv_bytes / bytes_per_mib,
            limits.serve_session_kv_bytes / bytes_per_mib,
            p.reset,
        },
    );

    // List one line per model.
    for (present, windows, 1..) |m, win, label| {
        const gap = padding(name_w - m.pub_name.len);
        try w.print(
            "  {s}[+]{s} {s}{d:>2}{s}  {s}{s}  ",
            .{ p.green, p.reset, p.bold, label, p.reset, m.pub_name, gap },
        );
        if (win) |x| {
            try w.print(
                "{s}ctx {d:>6}  run {d:>6}  serve {d:>5}{s}\n",
                .{ p.dim, x.ceiling, x.run, x.serve, p.reset },
            );
        } else {
            try w.print("{s}ctx ?{s}\n", .{ p.dim, p.reset });
        }
    }

    // Print the prompt.
    try w.print(
        "{s}[>]{s} choose [1-{d}] {s}(Enter = 1){s}: ",
        .{ p.bold, p.reset, present.len, p.dim, p.reset },
    );
    try w.flush();
}

/// A model's ceiling and its default windows, read without faulting weight pages. Null means the
/// file does not open or its card does not parse, and the picker lists the model without numbers.
fn readWindows(ctx: context.Context, path: []const u8, m: *const registry.Model) ?Windows {
    var loaded = loader.loadWith(
        ctx.io,
        ctx.arena,
        path,
        .{ .prefault = false, .quiet = true },
    ) catch return null;
    defer loaded.close();

    const cost = kv_budget.costModel(loaded.card.kvShape(), m.kv_dtype_default);
    const ceiling = limits.contextCeiling(loaded.card.context_length, m.context_limit);
    const windows = kv_budget.defaultWindows(cost, ceiling);
    return .{ .ceiling = ceiling, .run = windows.run, .serve = windows.serve };
}

/// `n` spaces for column alignment.
fn padding(n: usize) []const u8 {
    const spaces = " " ** name_pad_max;
    return spaces[0..@min(n, spaces.len)];
}

/// The unknown-model error: the bad name and a pointer to `list`. The write is best-effort.
fn unknownModel(io: Io, name: []const u8) void {
    var stderr_buffer: [error_buffer_bytes]u8 = undefined;
    var stderr_writer: Io.File.Writer = .initStreaming(.stderr(), io, &stderr_buffer);
    const w = &stderr_writer.interface;

    w.print("error: model '{s}' is not in the registry\n", .{name}) catch return;
    w.writeAll("       (run 'zigferencer list' for available names)\n") catch return;
    w.flush() catch {};
}

fn named(ctx: context.Context, name: []const u8) std.mem.Allocator.Error!Named {
    const m = registry.lookup(name) orelse return .unknown;
    if ((try registry.locate(ctx.arena, ctx.io, ctx.models_root, m.*)) == null) {
        return .{ .absent = try registry.candidates(ctx.arena, ctx.models_root, m.*) };
    }

    return .{ .found = m };
}

/// The absent-model error: both paths `locate` tried and the fetch command. The write is
/// best-effort.
fn notOnDisk(io: Io, name: []const u8, c: registry.Candidates) void {
    var stderr_buffer: [error_buffer_bytes]u8 = undefined;
    var stderr_writer: Io.File.Writer = .initStreaming(.stderr(), io, &stderr_buffer);
    const w = &stderr_writer.interface;

    writeNotOnDisk(w, name, c) catch return;
    w.flush() catch {};
}

fn writeNotOnDisk(w: *Io.Writer, name: []const u8, c: registry.Candidates) Io.Writer.Error!void {
    try w.print("error: model {s} is not on disk\n", .{name});
    try w.print("       looked in {s}\n", .{c.local});
    if (!c.sameFile()) {
        try w.print("       and in {s}\n", .{c.store});
    }

    try w.print("       run 'zigferencer fetch {s}' to download it\n", .{name});
}

// -- tests ------------------------------------------------------------------

test "a named model passes selection exactly when its file is on disk" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(arena, ".zig-cache/tmp/{s}", .{&tmp.sub_path});
    const ctx: context.Context = .{
        .io = io,
        .arena = arena,
        .gpa = std.testing.allocator,
        .args = &.{},
        .stdout = undefined,
        .models_root = root,
        .cache_dir = "",
        .vk_options = .{},
        .color = false,
        .color_err = false,
    };

    // The repo-local models dir of the test can hold every entry, and then no absent case exists.
    const absent = for (registry.all()) |*m| {
        if ((try registry.locate(arena, io, root, m.*)) == null) break m;
    } else return error.SkipZigTest;

    const refused = try named(ctx, absent.pub_name);
    try std.testing.expectEqualStrings(
        try registry.modelFilePath(arena, root, absent.*),
        refused.absent.store,
    );

    // Put the file in the store, and the same name now resolves to the same entry.
    try tmp.dir.createDirPath(io, absent.local_path);
    const sub = try std.Io.Dir.path.join(arena, &.{ absent.local_path, absent.file_name });
    const file = try tmp.dir.createFile(io, sub, .{});
    file.close(io);
    try std.testing.expectEqual(absent, (try named(ctx, absent.pub_name)).found);
    try std.testing.expectEqual(absent, try model(ctx, absent.pub_name));
    try std.testing.expect(try named(ctx, "no-such-model") == .unknown);
}

test "the absent-model error names the store path only when it differs" {
    var buf: [error_buffer_bytes]u8 = undefined;

    var two: Io.Writer = .fixed(&buf);
    try writeNotOnDisk(&two, "Qwen3.5-2B-Q8_0", .{
        .local = "models/org_repo/q.gguf",
        .store = "/data/store/org_repo/q.gguf",
    });
    try std.testing.expectEqualStrings(
        \\error: model Qwen3.5-2B-Q8_0 is not on disk
        \\       looked in models/org_repo/q.gguf
        \\       and in /data/store/org_repo/q.gguf
        \\       run 'zigferencer fetch Qwen3.5-2B-Q8_0' to download it
        \\
    , two.buffered());

    var one: Io.Writer = .fixed(&buf);
    try writeNotOnDisk(&one, "Qwen3.5-2B-Q8_0", .{
        .local = "models/org_repo/q.gguf",
        .store = "models/org_repo/q.gguf",
    });
    try std.testing.expectEqualStrings(
        \\error: model Qwen3.5-2B-Q8_0 is not on disk
        \\       looked in models/org_repo/q.gguf
        \\       run 'zigferencer fetch Qwen3.5-2B-Q8_0' to download it
        \\
    , one.buffered());
}
