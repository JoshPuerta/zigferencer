//! Fetch a model GGUF over HTTPS with `std.http.Client` and write it to
//! `<root>/<model.local_path>/<model.file_name>`. The write lands through an
//! atomic rename, and the parent directories get created first. The body
//! streams in chunks so a TTY gets a live progress bar. `remoteSize` reads only
//! the response head for the `fetch --dry-run` size preview. The fetch uses
//! only the standard library, including its certificate-verified TLS.
//!
//! Safety rules. The URL host is fixed to huggingface.co, either from a curated
//! registry URL or from the `fetch <org>/<repo>/<file>` form that fills only
//! path segments. Redirects follow to the Hugging Face CDN. The on-disk path
//! resists traversal: org and repo hold no slash, the file name is basenamed,
//! and it must end in .gguf. A non-identity content-encoding fails the fetch
//! instead of writing a compressed blob. The download goes to a
//! `<file>.partial` temp, and only a success renames it. A failed fetch never
//! truncates a good GGUF.
//!
//! The four refusals this module raises are `error.HttpStatusNotOk`,
//! `error.UnexpectedEncoding`, `error.NotAGguf` and `error.DownloadTruncated`.
//! Each one logs what went wrong before it returns, so the CLI exits without a
//! trace.

const std = @import("std");
const registry = @import("../core/registry.zig");
const time = @import("../core/time.zig");
const parser = @import("gguf/parser.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const log = std.log.scoped(.fetch);

// -- constants --------------------------------------------------------------

/// Redirect budget for the one Hugging Face `/resolve/` hop to the CDN, plus
/// headroom. A longer chain signals a problem, not a need.
const max_redirects = 3;

// -- types ------------------------------------------------------------------

/// A throttled download progress bar on stderr. The bar is interactive output,
/// so it bypasses the gated log. On a non-TTY it stays silent, and the log
/// lines plus the final "done" carry the result.
const Progress = struct {
    io: Io,
    total: ?u64,
    tty: bool,
    start: Io.Clock.Timestamp,
    last: Io.Clock.Timestamp,
    drawn: bool = false,

    fn init(io: Io, total: ?u64) Progress {
        const now = Io.Clock.Timestamp.now(io, .awake);
        const tty = (Io.File.stderr().isTty(io) catch false);
        return .{ .io = io, .total = total, .tty = tty, .start = now, .last = now };
    }

    fn update(self: *Progress, downloaded: u64) void {
        if (!self.tty) return;

        const now = Io.Clock.Timestamp.now(self.io, .awake);
        if (self.drawn and time.msBetween(self.last, now) < 100) return;

        self.last = now;
        self.drawn = true;

        const elapsed_s = time.msBetween(self.start, now) / 1000.0;
        const speed = if (elapsed_s > 0) mib(downloaded) / elapsed_s else 0;
        var buf: [160]u8 = undefined;
        var fw: Io.File.Writer = .initStreaming(.stderr(), self.io, &buf);
        const w = &fw.interface;
        if (self.total) |t| {
            const pct = if (t > 0) @as(
                f64,
                @floatFromInt(downloaded),
            ) / @as(f64, @floatFromInt(t)) * 100.0 else 0;

            w.print(
                "\r  {d:>5.1}%   {d:.1} / {d:.1} MiB   {d:.1} MiB/s   ",
                .{ pct, mib(downloaded), mib(t), speed },
            ) catch {};
        } else {
            w.print("\r  {d:.1} MiB   {d:.1} MiB/s   ", .{ mib(downloaded), speed }) catch {};
        }

        w.flush() catch {};
    }

    fn finish(self: *Progress) void {
        if (!self.tty or !self.drawn) return;

        var buf: [2]u8 = undefined;
        var fw: Io.File.Writer = .initStreaming(.stderr(), self.io, &buf);
        fw.interface.writeAll("\n") catch {};
        fw.interface.flush() catch {};
    }
};

// -- api --------------------------------------------------------------------

/// Download `model` from `model.download_url` to
/// `<root>/<local_path>/<file_name>` and replace any existing file through an
/// atomic rename. The call creates the parent directory first. The body streams
/// in chunks and updates a progress bar on a TTY.
pub fn fetchModel(
    io: Io,
    arena: Allocator,
    gpa: Allocator,
    root: []const u8,
    model: registry.Model,
) !void {
    var cwd = Io.Dir.cwd();
    try cwd.createDirPath(io, try registry.modelDir(arena, root, model));
    const file_path = try registry.modelFilePath(arena, root, model);
    // A sibling temp keeps a failed fetch from truncating a good GGUF, and a
    // rename inside one directory is atomic.
    const tmp_path = try std.fmt.allocPrint(arena, "{s}.partial", .{file_path});

    log.info("fetching {s}", .{model.download_url});
    log.info("  -> {s}", .{file_path});

    var client: std.http.Client = .{ .allocator = gpa, .io = io };
    defer client.deinit();

    {
        const file = try cwd.createFile(io, tmp_path, .{});
        errdefer cwd.deleteFile(io, tmp_path) catch {};
        defer file.close(io);

        var file_buf: [64 * 1024]u8 = undefined;
        var file_writer: Io.File.Writer = .init(file, io, &file_buf);

        const uri = try std.Uri.parse(model.download_url);
        var redirect_buf: [8 * 1024]u8 = undefined;
        var req = try client.request(
            .GET,
            uri,
            .{ .redirect_behavior = @enumFromInt(max_redirects) },
        );
        defer req.deinit();

        try req.sendBodiless();
        var response = try req.receiveHead(&redirect_buf);

        if (response.head.status != .ok) {
            log.err("HTTP {d} {t}", .{ @intFromEnum(response.head.status), response.head.status });
            if (response.head.status == .unauthorized or response.head.status == .forbidden) {
                log.err("a gated repo needs a HuggingFace login. Try a public mirror", .{});
            }

            return error.HttpStatusNotOk;
        }

        // A compressed encoding lands a wrong file on disk and breaks the
        // content-length progress.
        if (response.head.content_encoding != .identity) {
            log.err(
                "unexpected content-encoding {t}. Refusing to write",
                .{response.head.content_encoding},
            );
            return error.UnexpectedEncoding;
        }

        var prog = Progress.init(io, response.head.content_length);
        var transfer_buf: [4 * 1024]u8 = undefined;
        const reader = response.reader(&transfer_buf);
        var chunk: [128 * 1024]u8 = undefined;
        var downloaded: u64 = 0;
        // A gated repo or an error page can arrive with status 200, so the
        // leading bytes must still carry the GGUF identifier the parser names.
        var magic: [parser.magic.len]u8 = undefined;
        var magic_len: usize = 0;
        while (true) {
            const n = try reader.readSliceShort(&chunk);
            if (n == 0) break;

            if (magic_len < magic.len) {
                const take = @min(magic.len - magic_len, n);
                @memcpy(magic[magic_len..][0..take], chunk[0..take]);
                magic_len += take;
            }

            try file_writer.interface.writeAll(chunk[0..n]);
            downloaded += n;
            prog.update(downloaded);
        }

        try file_writer.interface.flush();
        prog.finish();

        if (magic_len < magic.len or !std.mem.eql(u8, &magic, &parser.magic)) {
            log.err(
                "response is not a GGUF (wrong magic). A gated repo can serve an HTML page instead",
                .{},
            );
            return error.NotAGguf;
        }

        if (response.head.content_length) |expected| {
            if (downloaded != expected) {
                log.err("connection ended early: {d} of {d} bytes", .{ downloaded, expected });
                return error.DownloadTruncated;
            }
        }

        log.info("done: {d:.1} MiB", .{mib(downloaded)});
    }

    errdefer cwd.deleteFile(io, tmp_path) catch {};

    try cwd.rename(tmp_path, cwd, file_path, io);
}

/// Read the Content-Length of `url` for the `fetch --dry-run` size preview, or
/// null when the server reports no length. The call uses a GET and reads only
/// the response head, because the Hugging Face CDN rejects HEAD on `/resolve/`.
/// The body stays unread, so the call downloads nothing.
pub fn remoteSize(io: Io, gpa: Allocator, url: []const u8) !?u64 {
    var client: std.http.Client = .{ .allocator = gpa, .io = io };
    defer client.deinit();

    const uri = try std.Uri.parse(url);
    var redirect_buf: [8 * 1024]u8 = undefined;
    // Without keep-alive, the close drops the socket instead of draining the
    // whole file to reuse the connection.
    var req = try client.request(
        .GET,
        uri,
        .{ .redirect_behavior = @enumFromInt(max_redirects), .keep_alive = false },
    );
    defer req.deinit();

    try req.sendBodiless();
    const response = try req.receiveHead(&redirect_buf);
    if (response.head.status != .ok) return error.HttpStatusNotOk;

    return response.head.content_length;
}

// -- helpers ----------------------------------------------------------------

fn mib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}
