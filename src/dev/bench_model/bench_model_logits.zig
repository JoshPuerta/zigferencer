//! The logits file of `bench-model --logits-out` and `--force-tokens`, and `LogitsRecorder`, the
//! runner wrapper that writes the file and feeds the forced ids. `compare-logits` reads the file.
//!
//! Format. All integers and floats are little-endian. The file starts with the 8-byte `magic`,
//! then three u32 values and one u64: the vocab size, the prompt length, the first decode
//! position, and the FNV-1a 64 hash of the prompt ids. Each entry follows as one u32 and `vocab`
//! f32 values: the logits. The u32 is the id that the run fed after the entry, and the last entry
//! holds `last_entry_id`. The first entry holds the logits of the prefill.
//!
//! Trust. A file comes from another run or from a person. So the reader checks the magic, the
//! range of each u32 header field, the file size, and each fed id before a caller sees a value.
//!
//! Recorder. `generate.decode` steps a runner with the argmax id of the logits that the previous
//! call returned. The recorder keeps those logits, so each `step` first writes them with the id it
//! feeds. The logits stay valid only until the next runner call, because the runner owns the
//! buffer. The hash sink of the caller still takes the argmax ids that `decode` picks.
//!
//! Comptime. `LogitsRecorder` is generic over the runner pointer type. `@hasDecl` finds the
//! optional `decodeHost` of the runner, which `decode` reads.

const std = @import("std");
const builtin = @import("builtin");
const zigferencer = @import("zigferencer");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const args = zigferencer.cli.args;
const limits = zigferencer.limits;
const step_stats = zigferencer.step_stats;

// -- constants --------------------------------------------------------------

/// The first 8 bytes of every logits file. The digit is the format version.
pub const magic: [8]u8 = "ZFLOGIT2".*;

/// The magic, the three u32 header fields, and the u64 prompt hash.
pub const header_bytes: u64 = magic.len + 3 * @sizeOf(u32) + @sizeOf(u64);

/// The fed id of the last entry: no step followed it.
pub const last_entry_id: u32 = 0xFFFF_FFFF;

/// The buffer of one file writer or one file reader.
const io_buffer_bytes = 64 * 1024;

const native_little = builtin.cpu.arch.endian() == .little;

// -- types ------------------------------------------------------------------

/// The header fields after the magic.
pub const Header = struct {
    vocab: u32,
    prompt_len: u32,
    first_pos: u32,
    /// The FNV-1a 64 hash of the prompt ids, from `promptHash`.
    prompt_hash: u64,

    /// The bytes of one entry with this vocab.
    pub fn entryBytes(self: Header) u64 {
        assert(self.vocab > 0);

        return @sizeOf(u32) + @as(u64, self.vocab) * @sizeOf(f32);
    }
};

/// A file that is not a valid logits file.
pub const FormatError = error{
    ShortHeader,
    NotLogitsFile,
    BadHeader,
    EntryRemainder,
    NoEntries,
    BadFedId,
};

pub const OpenError = Io.File.OpenError || Io.File.Reader.SizeError || Io.Reader.Error ||
    FormatError;

pub const ReadError = Io.File.Reader.SeekError || Io.Reader.Error || FormatError;

pub const CreateError = Io.File.OpenError || Io.Writer.Error;

/// A logits file open for reading, entry by entry.
pub const Reader = struct {
    file: Io.File,
    reader: Io.File.Reader,
    header: Header,
    entries: u64,

    /// Open `path` and validate the header and the size. `buffer` backs the file reader.
    pub fn open(io: Io, path: []const u8, buffer: []u8) OpenError!Reader {
        assert(buffer.len > 0);

        const file = try Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);

        var reader = file.reader(io, buffer);
        const size = try reader.getSize();
        if (size < header_bytes) return error.ShortHeader;

        var head: [header_bytes]u8 = undefined;
        try reader.interface.readSliceAll(&head);
        const header = try parseHeader(&head);

        const body = size - header_bytes;
        const entry_bytes = header.entryBytes();
        if (body % entry_bytes != 0) return error.EntryRemainder;
        if (body == 0) return error.NoEntries;

        const out: Reader = .{
            .file = file,
            .reader = reader,
            .header = header,
            .entries = body / entry_bytes,
        };
        assert(out.entries >= 1);
        return out;
    }

    pub fn close(self: *Reader, io: Io) void {
        self.file.close(io);
    }

    /// Read entry `index` into `logits` and return its fed id. Every entry but the last feeds an
    /// id inside the vocab, and the last entry holds `last_entry_id`.
    pub fn readEntry(self: *Reader, index: u64, logits: []f32) ReadError!u32 {
        assert(index < self.entries);
        assert(logits.len == self.header.vocab);

        const id = try self.fedId(index);
        const bits: []u32 = @ptrCast(logits);
        try self.reader.interface.readSliceAll(std.mem.sliceAsBytes(bits));
        if (!native_little) {
            for (bits) |*b| {
                b.* = @byteSwap(b.*);
            }
        }

        return id;
    }

    /// Return the fed id of entry `index`. The read position ends at the first value of the entry.
    pub fn fedId(self: *Reader, index: u64) ReadError!u32 {
        assert(index < self.entries);

        try self.reader.seekTo(header_bytes + index * self.header.entryBytes());
        const id = try self.reader.interface.takeInt(u32, .little);
        try checkFedId(self.header, self.entries, index, id);
        return id;
    }
};

/// A logits file open for writing. `create` writes the header, `writeEntry` appends one entry.
pub const Writer = struct {
    file: Io.File,
    writer: Io.File.Writer,
    header: Header,
    entries: u64 = 0,

    pub fn create(io: Io, path: []const u8, header: Header, buffer: []u8) CreateError!Writer {
        assert(header.vocab > 0);
        assert(buffer.len > 0);

        const file = try Io.Dir.cwd().createFile(io, path, .{});
        errdefer file.close(io);

        var out: Writer = .{ .file = file, .writer = file.writer(io, buffer), .header = header };
        const w = &out.writer.interface;
        try w.writeAll(&magic);
        try w.writeInt(u32, header.vocab, .little);
        try w.writeInt(u32, header.prompt_len, .little);
        try w.writeInt(u32, header.first_pos, .little);
        try w.writeInt(u64, header.prompt_hash, .little);
        return out;
    }

    pub fn writeEntry(self: *Writer, fed_id: u32, logits: []const f32) Io.Writer.Error!void {
        assert(logits.len == self.header.vocab);
        assert(fed_id < self.header.vocab or fed_id == last_entry_id);

        const w = &self.writer.interface;
        try w.writeInt(u32, fed_id, .little);
        if (native_little) {
            try w.writeAll(std.mem.sliceAsBytes(logits));
        } else {
            for (logits) |v| {
                try w.writeInt(u32, @bitCast(v), .little);
            }
        }

        self.entries += 1;
    }

    pub fn flush(self: *Writer) Io.Writer.Error!void {
        try self.writer.interface.flush();
    }

    pub fn close(self: *Writer, io: Io) void {
        self.file.close(io);
    }
};

/// The two flags of one bench run. Null leaves the flag off.
pub const Paths = struct {
    out: ?[]const u8 = null,
    force: ?[]const u8 = null,

    pub fn active(self: Paths) bool {
        return self.out != null or self.force != null;
    }
};

/// The open files of one bench run: the output file and the forced ids, read in full at open.
pub const RunFiles = struct {
    out: ?Writer = null,
    forced: ?[]const u32 = null,

    /// Open the files of `paths` for a run of `header` that steps `decode_len` times. A forced
    /// file must match the header and hold at least `decode_len` fed ids.
    pub fn open(
        io: Io,
        arena: Allocator,
        paths: Paths,
        header: Header,
        decode_len: u32,
    ) (CreateError || ReadError || Allocator.Error || error{UsageError})!RunFiles {
        assert(paths.active());
        assert(decode_len >= 1);

        var files: RunFiles = .{};
        if (paths.force) |path| {
            files.forced = try readForced(io, arena, path, header, decode_len);
        }

        if (paths.out) |path| {
            const buffer = try arena.alloc(u8, io_buffer_bytes);
            files.out = Writer.create(io, path, header, buffer) catch |err| {
                return args.fail(io, "--logits-out {s}: {t}", .{ path, err });
            };
        }

        if (files.forced) |f| assert(f.len == decode_len);
        return files;
    }

    pub fn close(self: *RunFiles, io: Io) void {
        if (self.out) |*w| {
            w.close(io);
        }
    }
};

// -- api --------------------------------------------------------------------

/// The FNV-1a 64 hash of the prompt ids, in the byte order of the token hash of `bench-model`.
pub fn promptHash(ids: []const u32) u64 {
    assert(ids.len >= 1);

    var hasher = std.hash.Fnv1a_64.init();
    for (ids) |id| {
        hasher.update(std.mem.asBytes(&id));
    }

    return hasher.final();
}

/// A runner wrapper that writes all logits it sees to `out` and feeds the ids of `forced` in place
/// of the ids that `decode` picks. `Runner` is the runner pointer type.
pub fn LogitsRecorder(comptime Runner: type) type {
    const Inner = switch (@typeInfo(Runner)) {
        .pointer => |p| p.child,
        else => @compileError("LogitsRecorder takes a runner pointer type"),
    };

    return struct {
        const Self = @This();

        pub const Error = StepErrorOf(Inner) || Io.Writer.Error;

        inner: Runner,
        out: ?*Writer,
        forced: ?[]const u32,
        /// The logits of the last runner call, the logits of the prefill at the start.
        logits: []f32,
        steps: u32 = 0,

        pub fn init(inner: Runner, first_logits: []f32, out: ?*Writer, forced: ?[]const u32) Self {
            assert(first_logits.len > 0);
            if (out) |w| assert(w.header.vocab == first_logits.len);

            return .{ .inner = inner, .out = out, .forced = forced, .logits = first_logits };
        }

        /// Write the held logits with the id this call feeds, then step the runner with that id.
        pub fn step(self: *Self, id: u32, pos: u32) Error![]f32 {
            const fed = if (self.forced) |f| blk: {
                assert(self.steps < f.len);
                break :blk f[self.steps];
            } else id;
            assert(fed < self.logits.len);

            if (self.out) |w| {
                try w.writeEntry(fed, self.logits);
            }

            self.logits = try self.inner.step(fed, pos);
            self.steps += 1;
            return self.logits;
        }

        pub fn decodeHost(self: *Self) ?*step_stats.DecodeHost {
            if (comptime @hasDecl(Inner, "decodeHost")) return self.inner.decodeHost();
            return null;
        }

        /// Write the last entry and flush. No step follows it.
        pub fn finish(self: *Self) Io.Writer.Error!void {
            const w = self.out orelse return;

            try w.writeEntry(last_entry_id, self.logits);
            try w.flush();
            assert(w.entries == self.steps + 1);
        }
    };
}

// -- helpers ----------------------------------------------------------------

fn StepErrorOf(comptime Inner: type) type {
    const Return = @typeInfo(@TypeOf(Inner.step)).@"fn".return_type.?;
    return @typeInfo(Return).error_union.error_set;
}

fn parseHeader(head: *const [header_bytes]u8) FormatError!Header {
    if (!std.mem.eql(u8, head[0..magic.len], &magic)) return error.NotLogitsFile;

    const header: Header = .{
        .vocab = std.mem.readInt(u32, head[8..12], .little),
        .prompt_len = std.mem.readInt(u32, head[12..16], .little),
        .first_pos = std.mem.readInt(u32, head[16..20], .little),
        .prompt_hash = std.mem.readInt(u64, head[20..28], .little),
    };
    if (header.vocab == 0 or header.vocab > limits.max_vocab) return error.BadHeader;
    if (header.prompt_len == 0 or header.prompt_len > limits.max_model_ctx) return error.BadHeader;
    if (header.first_pos == 0 or header.first_pos > limits.max_model_ctx) return error.BadHeader;

    return header;
}

fn checkFedId(header: Header, entries: u64, index: u64, id: u32) FormatError!void {
    assert(index < entries);

    const last = index + 1 == entries;
    if (last and id != last_entry_id) return error.BadFedId;
    if (!last and id >= header.vocab) return error.BadFedId;
}

/// Read the first `decode_len` fed ids of the file at `path`, after the checks against the run.
fn readForced(
    io: Io,
    arena: Allocator,
    path: []const u8,
    header: Header,
    decode_len: u32,
) (ReadError || Allocator.Error || error{UsageError})![]const u32 {
    const buffer = try arena.alloc(u8, io_buffer_bytes);
    var r = Reader.open(io, path, buffer) catch |err| {
        return args.fail(io, "--force-tokens {s}: {t}", .{ path, err });
    };
    defer r.close(io);

    const h = r.header;
    if (h.vocab != header.vocab) {
        return args.fail(io, "--force-tokens {s}: vocab {d}, this run {d}", .{
            path, h.vocab, header.vocab,
        });
    }
    if (h.prompt_len != header.prompt_len) {
        return args.fail(io, "--force-tokens {s}: prompt length {d}, this run {d}", .{
            path, h.prompt_len, header.prompt_len,
        });
    }
    if (h.first_pos != header.first_pos) {
        return args.fail(io, "--force-tokens {s}: first position {d}, this run {d}", .{
            path, h.first_pos, header.first_pos,
        });
    }
    if (h.prompt_hash != header.prompt_hash) {
        return args.fail(io, "--force-tokens {s}: prompt hash 0x{x:0>16}, this run 0x{x:0>16}", .{
            path, h.prompt_hash, header.prompt_hash,
        });
    }

    const fed_entries = r.entries - 1;
    if (fed_entries < decode_len) {
        return args.fail(io, "--force-tokens {s}: {d} fed ids, this run needs {d}", .{
            path, fed_entries, decode_len,
        });
    }

    const ids = try arena.alloc(u32, decode_len);
    for (ids, 0..) |*id, i| {
        id.* = r.fedId(i) catch |err| {
            return args.fail(io, "--force-tokens {s}: entry {d}: {t}", .{ path, i, err });
        };
    }

    return ids;
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;
const generate = zigferencer.generate;

// A canned-logits runner in the shape of the one of the `generate` tests. Each call returns the
// next fixed logits and records the id it got.
const CannedRunner = struct {
    seq: []const []f32,
    fed: [8]u32 = undefined,
    i: usize = 0,

    pub fn step(self: *CannedRunner, id: u32, pos: u32) error{}![]f32 {
        _ = pos;
        self.fed[self.i] = id;
        defer self.i += 1;

        return self.seq[self.i];
    }
};

const IdSink = struct {
    buf: [8]u32 = undefined,
    n: usize = 0,

    pub fn onToken(self: *IdSink, id: u32) !void {
        self.buf[self.n] = id;
        self.n += 1;
    }
};

fn tmpPath(tmp: *testing.TmpDir, name: []const u8, out: []u8) ![]const u8 {
    return std.fmt.bufPrint(out, ".zig-cache/tmp/{s}/{s}", .{ &tmp.sub_path, name });
}

fn writeFile(tmp: *testing.TmpDir, name: []const u8, bytes: []const u8) !void {
    const file = try tmp.dir.createFile(testing.io, name, .{});
    defer file.close(testing.io);

    var buffer: [256]u8 = undefined;
    var w = file.writer(testing.io, &buffer);
    try w.interface.writeAll(bytes);
    try w.interface.flush();
}

/// Decode `budget` tokens from `l0` over `seq` through a recorder into `path`.
fn recordRun(
    path: []const u8,
    l0: []f32,
    seq: []const []f32,
    forced: ?[]const u32,
    budget: u32,
    runner: *CannedRunner,
    sink: *IdSink,
) !void {
    const io = testing.io;
    var buffer: [256]u8 = undefined;
    const header: Header = .{
        .vocab = @intCast(l0.len),
        .prompt_len = 3,
        .first_pos = 3,
        .prompt_hash = 0x5eed,
    };
    var out = try Writer.create(io, path, header, &buffer);
    defer out.close(io);

    runner.* = .{ .seq = seq };
    var rec = LogitsRecorder(*CannedRunner).init(runner, l0, &out, forced);
    const t0 = Io.Clock.Timestamp.now(io, .awake);
    _ = try generate.decode(&rec, l0, 3, budget, 100, 99, true, io, t0, sink);
    try rec.finish();
}

test "LogitsRecorder: the file holds every entry with the id fed after it" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [128]u8 = undefined;
    const path = try tmpPath(&tmp, "free.bin", &path_buf);

    var l0 = [_]f32{ 0, 9, 0, 0 }; // -> 1
    var l1 = [_]f32{ 0, 0, 9, 0 }; // -> 2
    var l2 = [_]f32{ 9, 0, 0, 0 }; // -> 0
    const seq = [_][]f32{ &l1, &l2, &l1 };
    var runner: CannedRunner = undefined;
    var sink: IdSink = .{};
    try recordRun(path, &l0, &seq, null, 3, &runner, &sink);

    try testing.expectEqualSlices(u32, &.{ 1, 2, 0 }, sink.buf[0..sink.n]);
    try testing.expectEqualSlices(u32, &.{ 1, 2, 0 }, runner.fed[0..runner.i]);

    var buffer: [256]u8 = undefined;
    var r = try Reader.open(io, path, &buffer);
    defer r.close(io);
    try testing.expectEqual(@as(u64, 4), r.entries);
    const want: Header = .{ .vocab = 4, .prompt_len = 3, .first_pos = 3, .prompt_hash = 0x5eed };
    try testing.expectEqual(want, r.header);

    var logits: [4]f32 = undefined;
    const want_seq = [_][]const f32{ &l0, &l1, &l2, &l1 };
    const want_ids = [_]u32{ 1, 2, 0, last_entry_id };
    for (want_seq, want_ids, 0..) |want_logits, want_id, i| {
        try testing.expectEqual(want_id, try r.readEntry(i, &logits));
        try testing.expectEqualSlices(f32, want_logits, &logits);
    }
}

test "LogitsRecorder: forced ids reach the runner, and the hash ids stay the argmax" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [128]u8 = undefined;
    const path = try tmpPath(&tmp, "forced.bin", &path_buf);

    var l0 = [_]f32{ 0, 9, 0, 0 }; // -> 1
    var l1 = [_]f32{ 0, 0, 9, 0 }; // -> 2
    const seq = [_][]f32{ &l1, &l1 };
    const forced = [_]u32{ 3, 0 };
    var runner: CannedRunner = undefined;
    var sink: IdSink = .{};
    try recordRun(path, &l0, &seq, &forced, 2, &runner, &sink);

    try testing.expectEqualSlices(u32, &.{ 1, 2 }, sink.buf[0..sink.n]);
    try testing.expectEqualSlices(u32, &forced, runner.fed[0..runner.i]);

    var buffer: [256]u8 = undefined;
    var r = try Reader.open(testing.io, path, &buffer);
    defer r.close(testing.io);
    try testing.expectEqual(@as(u64, 3), r.entries);
    try testing.expectEqual(@as(u32, 3), try r.fedId(0));
    try testing.expectEqual(@as(u32, 0), try r.fedId(1));
    try testing.expectEqual(last_entry_id, try r.fedId(2));
}

test "Reader: refuses a short header, a bad magic, a bad header, a remainder, and no entries" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [128]u8 = undefined;
    var buffer: [256]u8 = undefined;

    // A valid header of vocab 2: magic, vocab, prompt length, first position, prompt hash.
    var good: [header_bytes]u8 = undefined;
    @memcpy(good[0..8], &magic);
    std.mem.writeInt(u32, good[8..12], 2, .little);
    std.mem.writeInt(u32, good[12..16], 5, .little);
    std.mem.writeInt(u32, good[16..20], 5, .little);
    std.mem.writeInt(u64, good[20..28], 0x5eed, .little);

    const Case = struct { name: []const u8, bytes: []const u8, err: OpenError };
    var bad_magic = good;
    bad_magic[0] = 'X';
    // A file of the first format version holds the old magic.
    var old_magic = good;
    old_magic[7] = '1';
    var zero_vocab = good;
    std.mem.writeInt(u32, zero_vocab[8..12], 0, .little);
    var huge_vocab = good;
    std.mem.writeInt(u32, huge_vocab[8..12], limits.max_vocab + 1, .little);
    var zero_prompt = good;
    std.mem.writeInt(u32, zero_prompt[12..16], 0, .little);
    // One full entry of vocab 2 is 12 bytes. Five bytes past it leave a remainder.
    const remainder = good ++ [_]u8{0} ** 17;

    const cases = [_]Case{
        .{ .name = "short", .bytes = good[0 .. header_bytes - 1], .err = error.ShortHeader },
        .{ .name = "magic", .bytes = &bad_magic, .err = error.NotLogitsFile },
        .{ .name = "version", .bytes = &old_magic, .err = error.NotLogitsFile },
        .{ .name = "vocab0", .bytes = &zero_vocab, .err = error.BadHeader },
        .{ .name = "vocabmax", .bytes = &huge_vocab, .err = error.BadHeader },
        .{ .name = "prompt0", .bytes = &zero_prompt, .err = error.BadHeader },
        .{ .name = "remainder", .bytes = &remainder, .err = error.EntryRemainder },
        .{ .name = "noentries", .bytes = &good, .err = error.NoEntries },
    };
    for (cases) |c| {
        try writeFile(&tmp, c.name, c.bytes);
        const path = try tmpPath(&tmp, c.name, &path_buf);
        try testing.expectError(c.err, Reader.open(io, path, &buffer));
    }
}

test "promptHash: equals the FNV-1a 64 of the id bytes, and the order of the ids counts" {
    const ids = [_]u32{ 7, 1, 300 };
    try testing.expectEqual(std.hash.Fnv1a_64.hash(std.mem.sliceAsBytes(&ids)), promptHash(&ids));
    try testing.expect(promptHash(&ids) != promptHash(&.{ 1, 7, 300 }));
}

test "Reader: refuses a fed id outside the vocab and a last entry without the end id" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [128]u8 = undefined;
    var buffer: [256]u8 = undefined;

    var bytes: [header_bytes + 2 * 12]u8 = @splat(0);
    @memcpy(bytes[0..8], &magic);
    std.mem.writeInt(u32, bytes[8..12], 2, .little);
    std.mem.writeInt(u32, bytes[12..16], 1, .little);
    std.mem.writeInt(u32, bytes[16..20], 1, .little);
    // Entry 0 feeds id 2, outside a vocab of 2. Entry 1 holds id 0, not the end id.
    std.mem.writeInt(u32, bytes[header_bytes..][0..4], 2, .little);
    try writeFile(&tmp, "ids", &bytes);

    const path = try tmpPath(&tmp, "ids", &path_buf);
    var r = try Reader.open(io, path, &buffer);
    defer r.close(io);
    try testing.expectError(error.BadFedId, r.fedId(0));
    try testing.expectError(error.BadFedId, r.fedId(1));
}
