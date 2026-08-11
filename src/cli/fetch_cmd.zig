//! `zigferencer fetch <arg>`: download a GGUF. A `/` in `<arg>` selects between
//! the two input modes:
//!
//!   1. A registry pub_name, no slash: `fetch SmolLM2-135M-Instruct-Q8_0`.
//!      `registry.lookup` resolves it, and the registry gives the local path.
//!   2. A Hugging Face path, with slashes:
//!      `fetch <org>/<repo>/<path/to/file.gguf>`. The command builds
//!      `https://huggingface.co/<org>/<repo>/resolve/main/<file>` and stores the
//!      file under `models/<org>_<repo>/<basename>`. The name must end in `.gguf`.
//!
//! The registry is the curated path. The HF mode keeps an ad-hoc download cheap
//! and leaves the registry clean. A repo subdirectory stays in the URL and
//! flattens to the basename on disk. That layout is a HF detail, and the engine
//! reads only the file.
//!
//! A fetch that replaces or duplicates a model already on disk prompts for
//! confirmation first. `registry.locate` spans the configured root and the
//! repo-local `models/` directory, so the guard catches an existing copy anywhere
//! a load can read it. `--force` skips the prompt. A non-TTY stdin refuses instead
//! of overwriting unasked. The download always saves under the configured root
//! ($ZIGFERENCER_MODELS_DIR, default `models/`).

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const cli = @import("command.zig");
const args = @import("args.zig");
const style = @import("style.zig");
const prompt = @import("prompt.zig");
const select = @import("select.zig");
const registry = @import("../core/registry.zig");
const download = @import("../io/download.zig");

// -- types ------------------------------------------------------------------

const hf_base_url = "https://huggingface.co";
const gguf_suffix = ".gguf";

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
        .force = args.Doc{ .text = "overwrite an existing local copy without prompting", .alias = "-f" },
    };
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: cli.Context) !void {
    const flags = try args.parseInto(Flags, ctx) orelse return;
    const dry_run = flags.dry_run;
    const force = flags.force;
    var arg = flags.model;
    if (arg == null) {
        // With no model named, a TTY gets a picker over the registry. A pipe gets
        // the usage screen, because nothing there can answer a prompt.
        if (prompt.stdinIsTty(ctx.io)) arg = try select.fetchPick(ctx);
    }
    if (arg == null) {
        args.printHelp(Flags, ctx.io, .stderr(), style.Palette.init(ctx.color_err));
        return error.UsageError;
    }
    const a = arg.?;

    // A slash means an ad-hoc Hugging Face path. Anything else is a registry name.
    const model: registry.Model = if (std.mem.indexOfScalar(u8, a, '/') != null) m: {
        const spec = parseHfPath(a) catch |err| {
            try printHfPathError(ctx.io, a, err);
            return error.BadHfPath;
        };
        break :m try buildSyntheticModel(ctx.arena, spec);
    } else m: {
        const found = registry.lookup(a) orelse {
            var buf: [512]u8 = undefined;
            var fw: Io.File.Writer = .initStreaming(.stderr(), ctx.io, &buf);
            const w = &fw.interface;
            try w.print("error: model '{s}' is not in the registry\n", .{a});
            try w.writeAll("       (run 'zigferencer list' for available names, or pass an HF path like 'org/repo/file.gguf')\n");
            try w.flush();
            return error.UnknownModel;
        };
        break :m found.*;
    };

    if (dry_run) return printPlan(ctx, model);

    // A model already on disk gets a guard. A second fetch of a multi-GB GGUF
    // wastes the download and overwrites a good file. `locate` spans the
    // configured root and the repo-local `models/` directory, so a copy anywhere a
    // load can read it counts as present. `--force` skips the guard.
    if (!force) {
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

/// Ask before a second fetch of a model already on disk, and name where the copy
/// is. When the configured root differs from that copy, the text also names where
/// the download lands. A non-TTY stdin has nobody to answer, so the function
/// refuses and points at `--force`.
fn confirmRefetch(ctx: cli.Context, model: registry.Model, found: []const u8) !bool {
    const dest = try registry.modelFilePath(ctx.arena, ctx.models_root, model);
    const overwrites = std.mem.eql(u8, found, dest);

    var buf: [1024]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stderr(), ctx.io, &buf);
    const w = &fw.interface;
    try w.print("'{s}' is already present:\n  {s}\n", .{ model.pub_name, found });
    if (!overwrites) try w.print("fetching will write a second copy to:\n  {s}\n", .{dest});

    if (!prompt.stdinIsTty(ctx.io)) {
        try w.writeAll("stdin is not a TTY. Re-run with --force to fetch anyway\n");
        try w.flush();
        return false;
    }
    try w.flush();
    return prompt.confirm(ctx.io, style.Palette.init(ctx.color_err), if (overwrites) "re-download and overwrite it?" else "continue?");
}

/// Print the plan of a fetch and download nothing. The plan holds the URL, the
/// destination, the remote size (one head-only GET), and the state of any local
/// copy. It goes to stdout, because it is the output `--dry-run` asks for.
fn printPlan(ctx: cli.Context, model: registry.Model) !void {
    const dest = try registry.modelFilePath(ctx.arena, ctx.models_root, model);
    const found = try registry.locate(ctx.arena, ctx.io, ctx.models_root, model);

    var buf: [1024]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stdout(), ctx.io, &buf);
    const w = &fw.interface;
    try w.print("plan for fetch: {s}\n", .{model.pub_name});
    try w.print("  from:   {s}\n", .{model.download_url});
    try w.print("  to:     {s}\n", .{dest});
    if (download.remoteSize(ctx.io, ctx.gpa, model.download_url)) |size_opt| {
        if (size_opt) |size| {
            try w.print("  size:   {d:.1} MiB\n", .{@as(f64, @floatFromInt(size)) / (1024.0 * 1024.0)});
        } else {
            try w.writeAll("  size:   unknown (no Content-Length from server)\n");
        }
    } else |err| {
        try w.print("  size:   unavailable (probe failed: {t})\n", .{err});
    }
    if (found) |path| {
        if (std.mem.eql(u8, path, dest)) {
            try w.print("  local:  present at {s}, overwritten after the prompt (--force skips it)\n", .{path});
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

    const slash1 = std.mem.indexOfScalar(u8, s, '/') orelse return error.NoRepoSegment;
    const slash2 = std.mem.indexOfScalarPos(u8, s, slash1 + 1, '/') orelse return error.NoFileSegment;

    const org = s[0..slash1];
    const repo = s[slash1 + 1 .. slash2];
    const file_in_repo = s[slash2 + 1 ..];

    if (org.len == 0 or repo.len == 0 or file_in_repo.len == 0) return error.EmptySegment;
    return .{ .org = org, .repo = repo, .file_in_repo = file_in_repo };
}

fn buildSyntheticModel(arena: Allocator, spec: HfSpec) Allocator.Error!registry.Model {
    const file_basename = std.fs.path.basename(spec.file_in_repo);
    // The basename without `.gguf` follows the pub_name convention of the
    // hand-curated registry, such as `SmolLM2-135M-Instruct-Q8_0`.
    const pub_name = file_basename[0 .. file_basename.len - gguf_suffix.len];

    // One subdirectory per model under the models root. `registry.modelsRoot`
    // resolves that root from $ZIGFERENCER_MODELS_DIR, with `models/` as default.
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
    var buf: [512]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stderr(), io, &buf);
    const w = &fw.interface;
    try w.print("error: bad HF path '{s}': ", .{arg});
    switch (err) {
        error.NotAGguf => try w.writeAll("must end in .gguf (the engine only loads GGUF files)\n"),
        error.NoRepoSegment => try w.writeAll("missing '/', expected <org>/<repo>/<file.gguf>\n"),
        error.NoFileSegment => try w.writeAll("missing file segment, expected <org>/<repo>/<file.gguf>\n"),
        error.EmptySegment => try w.writeAll("an org/repo/file segment was empty\n"),
    }
    try w.flush();
}

// -- tests ------------------------------------------------------------------

test "parseHfPath: 3-segment path in repo root" {
    const spec = try parseHfPath("bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q8_0.gguf");
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
