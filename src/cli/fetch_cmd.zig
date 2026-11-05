//! `zigferencer fetch <arg>`: download a GGUF. An `<arg>` without `/` is a registry
//! pub_name. An `<org>/<repo>/<path/to/file.gguf>` is a Hugging Face path, stored
//! under `<org>_<repo>/<basename>` in the models root.
//!
//! A fetch that replaces or duplicates a model already on disk asks first.
//! `--force` skips the question. A stdin that is not a TTY refuses instead.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const context = @import("context.zig");
const args = @import("args.zig");
const style = @import("style.zig");
const prompt = @import("prompt.zig");
const select = @import("select.zig");
const registry = @import("../core/registry.zig");
const download = @import("../io/download.zig");

// -- constants --------------------------------------------------------------

const hf_base_url = "https://huggingface.co";
const gguf_suffix = ".gguf";

/// Size of the stderr buffer for one error message.
const error_buffer_bytes = 512;

/// Size of the buffer for the refetch notice and the dry-run plan.
const notice_buffer_bytes = 1024;

// -- types ------------------------------------------------------------------

const HfSpec = struct {
    org: []const u8,
    repo: []const u8,
    /// Path inside the repo (can contain `/`). Always ends in `.gguf`.
    file_in_repo: []const u8,
};

const HfPathError = error{
    NotAGguf,
    NoRepoSegment,
    NoFileSegment,
    EmptySegment,
};

/// The flags of `fetch`.
const Flags = struct {
    model: ?[]const u8 = null,
    dry_run: bool = false,
    force: bool = false,

    pub const positional = "model";
    pub const help: args.Help = .{
        .usage = "zigferencer fetch <pub_name | org/repo/file.gguf> [flags]",
        .about = "Download a model by its registry name or by its Hugging Face path.\n",
        .footer =
        \\examples:
        \\  zigferencer fetch SmolLM2-135M-Instruct-Q8_0
        \\  zigferencer fetch bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q8_0.gguf
        \\
        \\'zigferencer list' shows registered pub_names.
        \\
        ,
    };
    pub const docs = .{
        .dry_run = args.Doc{ .text = "print the URL, destination, and remote size, then exit" },
        .force = args.Doc{
            .text = "overwrite an existing local copy without prompting",
            .alias = "-f",
        },
    };
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: context.Context) !void {
    const flags = try args.parseInto(Flags, ctx) orelse return;
    var arg = flags.model;
    if (arg == null and prompt.stdinIsTty(ctx.io)) {
        arg = try select.fetchPick(ctx);
    }

    if (arg == null) {
        args.printHelp(Flags, ctx.io, .stderr(), style.Palette.init(ctx.color_err));
        return error.UsageError;
    }

    const a = arg.?;

    const model: registry.Model = if (std.mem.findScalar(u8, a, '/') != null) m: {
        const spec = parseHfPath(a) catch |err| {
            try printHfPathError(ctx.io, a, err);
            return error.BadHfPath;
        };
        break :m try buildSyntheticModel(ctx.arena, spec);
    } else m: {
        const found = registry.lookup(a) orelse {
            var stderr_buffer: [error_buffer_bytes]u8 = undefined;
            var stderr_writer: Io.File.Writer = .initStreaming(.stderr(), ctx.io, &stderr_buffer);
            const w = &stderr_writer.interface;
            try w.print("error: model '{s}' is not in the registry\n", .{a});
            try w.writeAll(
                "       (run 'zigferencer list' for available names, or pass an HF path " ++
                    "like 'org/repo/file.gguf')\n",
            );
            try w.flush();
            return error.UnknownModel;
        };
        break :m found.*;
    };

    if (flags.dry_run) return printPlan(ctx, model);

    // `locate` spans the configured root and the repo-local `models/`, so it finds every copy a
    // load can read.
    if (!flags.force) {
        if (try registry.locate(ctx.arena, ctx.io, ctx.models_root, model)) |found| {
            if (!try confirmRefetch(ctx, model, found)) {
                args.writeStderr(ctx.io, "aborted. Nothing was downloaded or changed\n");
                return;
            }
        }
    }

    try download.fetchModel(ctx.io, ctx.arena, ctx.gpa, ctx.models_root, model);
}

// -- helpers ----------------------------------------------------------------

/// Ask before a second fetch of a model already on disk. A stdin that is not a TTY
/// refuses and points at `--force`.
fn confirmRefetch(ctx: context.Context, model: registry.Model, found: []const u8) !bool {
    const dest = try registry.modelFilePath(ctx.arena, ctx.models_root, model);
    const overwrites = std.mem.eql(u8, found, dest);

    var stderr_buffer: [notice_buffer_bytes]u8 = undefined;
    var stderr_writer: Io.File.Writer = .initStreaming(.stderr(), ctx.io, &stderr_buffer);
    const w = &stderr_writer.interface;
    try w.print("'{s}' is already present:\n  {s}\n", .{ model.pub_name, found });
    if (!overwrites) {
        try w.print("fetching will write a second copy to:\n  {s}\n", .{dest});
    }

    if (!prompt.stdinIsTty(ctx.io)) {
        try w.writeAll("stdin is not a TTY. Re-run with --force to fetch anyway\n");
        try w.flush();
        return false;
    }

    try w.flush();
    return prompt.confirm(
        ctx.io,
        style.Palette.init(ctx.color_err),
        if (overwrites) "re-download and overwrite it?" else "continue?",
    );
}

/// Print the plan of a fetch and download nothing. The size probe is one head-only GET.
fn printPlan(ctx: context.Context, model: registry.Model) !void {
    const dest = try registry.modelFilePath(ctx.arena, ctx.models_root, model);
    const found = try registry.locate(ctx.arena, ctx.io, ctx.models_root, model);

    const w = &ctx.stdout.interface;
    try w.print("plan for fetch: {s}\n", .{model.pub_name});
    try w.print("  from:   {s}\n", .{model.download_url});
    try w.print("  to:     {s}\n", .{dest});

    if (download.remoteSize(ctx.io, ctx.gpa, model.download_url)) |size_opt| {
        if (size_opt) |size| {
            try w.print(
                "  size:   {d:.1} MiB\n",
                .{@as(f64, @floatFromInt(size)) / (1024.0 * 1024.0)},
            );
        } else {
            try w.writeAll("  size:   unknown (no Content-Length from server)\n");
        }
    } else |err| {
        try w.print("  size:   unavailable (probe failed: {t})\n", .{err});
    }

    if (found) |path| {
        if (std.mem.eql(u8, path, dest)) {
            try w.print(
                "  local:  present at {s}, overwritten after the prompt (--force skips it)\n",
                .{path},
            );
        } else {
            try w.print("  local:  present at {s}, a second copy lands at 'to:' above\n", .{path});
        }
    } else {
        try w.writeAll("  local:  not present\n");
    }

    try w.writeAll("(dry run. Re-run without --dry-run to download)\n");
    try w.flush();
}

/// Split `org/repo/file_in_repo` on the first two `/` only. `file_in_repo` can
/// hold further `/` for the subdirectories inside the repo.
fn parseHfPath(s: []const u8) HfPathError!HfSpec {
    if (!std.mem.endsWith(u8, s, gguf_suffix)) return error.NotAGguf;

    const slash1 = std.mem.findScalar(u8, s, '/') orelse return error.NoRepoSegment;
    const slash2 = std.mem.findScalarPos(
        u8,
        s,
        slash1 + 1,
        '/',
    ) orelse return error.NoFileSegment;

    const org = s[0..slash1];
    const repo = s[slash1 + 1 .. slash2];
    const file_in_repo = s[slash2 + 1 ..];

    if (org.len == 0 or repo.len == 0 or file_in_repo.len == 0) return error.EmptySegment;

    return .{ .org = org, .repo = repo, .file_in_repo = file_in_repo };
}

fn buildSyntheticModel(arena: Allocator, spec: HfSpec) Allocator.Error!registry.Model {
    const file_basename = std.Io.Dir.path.basename(spec.file_in_repo);
    const pub_name = file_basename[0 .. file_basename.len - gguf_suffix.len];

    const local_path = try std.fmt.allocPrint(arena, "{s}_{s}", .{ spec.org, spec.repo });
    const download_url = try std.fmt.allocPrint(
        arena,
        hf_base_url ++ "/{s}/{s}/resolve/main/{s}",
        .{ spec.org, spec.repo, spec.file_in_repo },
    );

    return .{
        .pub_name = pub_name,
        .local_path = local_path,
        .file_name = file_basename,
        .download_url = download_url,
    };
}

fn printHfPathError(io: Io, arg: []const u8, err: HfPathError) !void {
    var stderr_buffer: [error_buffer_bytes]u8 = undefined;
    var stderr_writer: Io.File.Writer = .initStreaming(.stderr(), io, &stderr_buffer);
    const w = &stderr_writer.interface;
    try w.print("error: bad HF path '{s}': ", .{arg});
    switch (err) {
        error.NotAGguf => try w.writeAll("must end in .gguf (the engine only loads GGUF files)\n"),
        error.NoRepoSegment => try w.writeAll("missing '/', expected <org>/<repo>/<file.gguf>\n"),
        error.NoFileSegment => try w.writeAll(
            "missing file segment, expected <org>/<repo>/<file.gguf>\n",
        ),
        error.EmptySegment => try w.writeAll("an org/repo/file segment was empty\n"),
    }

    try w.flush();
}

// -- tests ------------------------------------------------------------------

test "parseHfPath: 3-segment path in repo root" {
    const spec = try parseHfPath(
        "bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q8_0.gguf",
    );
    try std.testing.expectEqualStrings("bartowski", spec.org);
    try std.testing.expectEqualStrings("SmolLM2-135M-Instruct-GGUF", spec.repo);
    try std.testing.expectEqualStrings("SmolLM2-135M-Instruct-Q8_0.gguf", spec.file_in_repo);
}

test "parseHfPath: file in a repo subdir is kept in file_in_repo" {
    const spec = try parseHfPath("unsloth/foo/Q4_K_M/foo-q4_k_m.gguf");
    try std.testing.expectEqualStrings("unsloth", spec.org);
    try std.testing.expectEqualStrings("foo", spec.repo);
    try std.testing.expectEqualStrings("Q4_K_M/foo-q4_k_m.gguf", spec.file_in_repo);
}

test "parseHfPath rejects non-gguf" {
    try std.testing.expectError(error.NotAGguf, parseHfPath("org/repo/foo.bin"));
}

test "parseHfPath rejects missing repo segment" {
    try std.testing.expectError(error.NoFileSegment, parseHfPath("org/repo.gguf"));
}

test "parseHfPath rejects empty segments" {
    try std.testing.expectError(error.EmptySegment, parseHfPath("org//file.gguf"));
}

test "buildSyntheticModel constructs the resolve URL and flattens local path" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    const spec: HfSpec = .{
        .org = "bartowski",
        .repo = "SmolLM2-135M-Instruct-GGUF",
        .file_in_repo = "SmolLM2-135M-Instruct-Q8_0.gguf",
    };
    const m = try buildSyntheticModel(arena, spec);

    try std.testing.expectEqualStrings("SmolLM2-135M-Instruct-Q8_0", m.pub_name);
    try std.testing.expectEqualStrings("bartowski_SmolLM2-135M-Instruct-GGUF", m.local_path);
    try std.testing.expectEqualStrings("SmolLM2-135M-Instruct-Q8_0.gguf", m.file_name);
    try std.testing.expectEqualStrings(
        "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf",
        m.download_url,
    );
}

test "buildSyntheticModel flattens subdir paths on disk but keeps them in the URL" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    const spec: HfSpec = .{
        .org = "unsloth",
        .repo = "foo",
        .file_in_repo = "Q4_K_M/foo-q4_k_m.gguf",
    };
    const m = try buildSyntheticModel(arena, spec);

    try std.testing.expectEqualStrings("foo-q4_k_m", m.pub_name);
    try std.testing.expectEqualStrings("unsloth_foo", m.local_path);
    try std.testing.expectEqualStrings("foo-q4_k_m.gguf", m.file_name);
    try std.testing.expectEqualStrings(
        "https://huggingface.co/unsloth/foo/resolve/main/Q4_K_M/foo-q4_k_m.gguf",
        m.download_url,
    );
}
