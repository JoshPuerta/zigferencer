//! `zigferencer serve-demo` is a scripted run against a serve instance. It shows
//! what the connector does, and it runs as a smoke test. By default it spawns
//! `serve` as a child process and exercises the whole deployment path: spawn,
//! parse READY, drive the wire, shut down. `--connect host:port` drives an
//! already-running server instead and leaves it running.
//!
//! The script walks the capabilities a client relies on:
//!   act 1  one turn on session a, streamed token by token.
//!   act 2  --turns follow-ups on the same session. The KV cache persists, so
//!          each turn prefills only its new prompt, not the whole history.
//!   act 3  the demo clones session a into b and c, which gives one prefilled
//!          context and three divergent sessions. All three then take a
//!          different turn in one decode batch. The token events interleave on
//!          the wire, and the session id demultiplexes them.
//!   act 4  a fresh session d opens with the same system prompt. The server's
//!          prefix cache restores the prefix instead of re-prefilling it, so the
//!          warm open returns in restore time. The act prints both open times
//!          side by side, and a backend without the cache shows two cold opens.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const time = @import("../core/time.zig");

const cli = @import("command.zig");
const args = @import("args.zig");
const style = @import("style.zig");
const select = @import("select.zig");
const protocol = @import("../serve/protocol.zig");
const Client = @import("../serve/client.zig").Client;
const corelog = @import("../core/log.zig");
const dtype = @import("../core/dtype.zig");
const serve_common = @import("serve_common.zig");

const log = std.log.scoped(.serve_demo);

// -- types ------------------------------------------------------------------

/// Where the spawned server's stderr lands when tracing is on, so its per-tick
/// diagnostics do not interleave with the live token stream (see `spawnServer`).
const serve_trace_file = "zigferencer-serve.log";

/// The one system prompt the demo prefills, then clones. It stays neutral on
/// purpose, because the demo shows the engine's serving capabilities and not an
/// application.
const system_prompt =
    "You are a concise technical assistant. Answer in one or two short sentences.";

/// Follow-up turns for act 2, cycled when `--turns` asks for more than the list.
const followups = [_][]const u8{
    "Now explain it like I'm five.",
    "What's the main downside?",
};

/// Per-turn headroom in tokens, held back for the turn's own prompt, which is
/// the chat-template wrapper plus the carried pending token. The generation cap
/// leaves this much room, so a turn's prompt always fits and the server never
/// returns `turn_too_long`.
const ctx_reserve: u32 = 96;

/// Below this a turn is not worth running. The session has spent its context
/// budget, and it ends its turns.
const min_turn_tokens: u32 = 16;

/// One turn's outcome: the session's new committed position, taken from the
/// server's `done`, and whether it hit a context limit and must stop taking
/// turns.
const Turn = struct { pos: u32, stopped: bool };

/// One session's turn in a batch: the session id, a short label, and the prompt.
const SessionTurn = struct { sid: u32, name: []const u8, prompt: []const u8 };

/// The flags of `serve-demo`.
const Flags = struct {
    pub_name: ?[]const u8 = null,
    connect: ?[]const u8 = null,
    turns: u32 = 2,
    max_tokens: u32 = 32,
    temperature: f32 = 0.7,
    seed: u64 = 7,
    max_sessions: u16 = 4,
    vram_slots: ?u16 = null,
    kv_ram: ?u32 = null,
    /// Parsed, never resolved: serve-demo loads no model, so the spawned
    /// server owns the capability gate and the refusal.
    kv_dtype: ?dtype.KvDtype = null,

    pub const positional = "pub_name";
    pub const help: args.Help = .{
        .usage = "zigferencer-dev serve-demo [pub_name] [flags]",
        .about =
        \\Demo the connector: a turn streamed live, follow-ups reusing the KV cache,
        \\one session cloned three ways, batched multi-session decode on one connection,
        \\and a warm re-open served by the prefix cache.
        \\
        ,
        .footer =
        \\examples:
        \\  zigferencer-dev serve-demo SmolLM2-135M-Instruct-Q8_0 --turns 3
        \\  zigferencer-dev serve-demo --connect 127.0.0.1:8000
        \\
        ,
    };
    pub const docs = .{
        .connect = args.Doc{ .group = "connection", .value = "HOST:PORT", .text = "drive a running server", .default = "spawn a child" },
        .turns = args.Doc{ .group = "script", .text = "follow-up rounds on session a, 1..8" },
        .max_tokens = args.Doc{ .group = "script", .text = "the most tokens one turn generates, 1..256" },
        .temperature = args.Doc{ .group = "script", .text = "sampling temperature. 0 = greedy" },
        .seed = args.Doc{ .group = "script", .text = "base seed. Sessions a/b/c use seed, +1, +2" },
        .max_sessions = args.Doc{ .group = "script", .text = "logical session pool for the spawned server, >=3" },
        .vram_slots = args.Doc{
            .group = "gpu (fused-GPU build only)",
            .text = "VRAM KV slots for the spawned server. Below --max-sessions\nit demos KV swap",
        },
        .kv_ram = args.Doc{
            .group = "gpu (fused-GPU build only)",
            .value = "MiB",
            .text = "RAM budget for evicted KV blobs on the spawned server.\nSmall values demo backpressure",
        },
        .kv_dtype = args.Doc{ .group = "kv", .text = "KV element type for the spawned server (the model gates it)" },
    };
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: cli.Context) !void {
    const flags = try args.parseInto(Flags, ctx) orelse return;
    const pub_name = flags.pub_name;
    const connect = flags.connect;
    const turns = flags.turns;
    const max_tokens = flags.max_tokens;
    const temperature = flags.temperature;
    const seed = flags.seed;
    const max_sessions = flags.max_sessions;
    const vram_slots = flags.vram_slots;
    const kv_ram_mib = flags.kv_ram;
    const kv_dtype = flags.kv_dtype;
    if (turns == 0 or turns > 8) return args.fail(ctx.io, "--turns must be in 1..8", .{});
    if (max_tokens == 0 or max_tokens > 256) return args.fail(ctx.io, "--max-tokens must be in 1..256", .{});
    if (!(temperature >= 0) or !std.math.isFinite(temperature)) return args.fail(ctx.io, "--temperature must be 0 or more", .{});
    // The demo opens one session and clones it twice, which makes three
    // concurrent sessions. A too-small pool fails here, not mid-script.
    if (max_sessions < 3) return args.fail(ctx.io, "--max-sessions must be >= 3 for the demo script", .{});
    if (connect != null and vram_slots != null) {
        log.info("--vram-slots is ignored with --connect (the running server set its own pool)", .{});
    }
    if (connect != null and kv_ram_mib != null) {
        log.info("--kv-ram is ignored with --connect (the running server sized its own RAM tier)", .{});
    }
    if (connect != null and kv_dtype != null) {
        log.info("--kv-dtype is ignored with --connect (the running server framed its own KV)", .{});
    }

    var stdout_buf: [4096]u8 = undefined;
    var fw: Io.File.Writer = .initStreaming(.stdout(), ctx.io, &stdout_buf);
    const w = &fw.interface;
    // Three roles, three colors, so an interleaved run reads at a glance. Cyan is
    // the input the demo sends, green is the text the model generates, and dim is
    // the demo's own annotation.
    const pal = style.Palette.init(ctx.color);

    var child: ?std.process.Child = null;
    // The spawned server is reaped on every exit path. A leaked child holds the
    // whole model mapped, which is an 8 GB orphan after one errored run. kill()
    // sends SIGKILL and reaps. On the normal path the explicit wait() below has
    // already reaped and nulled `child`, so this does nothing.
    defer if (child) |*c| c.kill(ctx.io);
    const address = if (connect) |hostport| blk: {
        const colon = std.mem.lastIndexOfScalar(u8, hostport, ':') orelse
            return args.fail(ctx.io, "--connect wants host:port", .{});
        const port = std.fmt.parseInt(u16, hostport[colon + 1 ..], 10) catch
            return args.fail(ctx.io, "--connect wants host:port", .{});
        break :blk try Io.net.IpAddress.parse(hostport[0..colon], port);
    } else blk: {
        const model = try select.model(ctx, pub_name);
        break :blk try spawnServer(ctx, w, model.pub_name, max_sessions, vram_slots, kv_ram_mib, kv_dtype, &child);
    };
    var client = try Client.connectTcp(ctx.gpa, ctx.io, address);
    defer client.deinit();

    const info = try client.hello();
    try w.print("{s}server:   max_sessions={d} batch={d} ctx={d}{s}\n\n", .{
        pal.dim, info.max_sessions, info.max_batch, info.max_seq, pal.reset,
    });

    const t0 = Io.Clock.Timestamp.now(ctx.io, .awake);
    const total = try playScript(ctx, w, pal, &client, turns, max_tokens, info.max_seq, info.max_sessions, temperature, seed);
    const wall_s = time.msBetween(t0, Io.Clock.Timestamp.now(ctx.io, .awake)) / 1000.0;
    try w.print("\n{s}done: {d} tokens in {d:.1}s ({d:.1} tok/s aggregate){s}\n", .{
        pal.dim, total, wall_s, @as(f64, @floatFromInt(total)) / wall_s, pal.reset,
    });
    try w.flush();

    // The happy path stops the server gracefully: ask it to exit, reap it, and
    // null `child` so the defer backstop does nothing. The handshake is best
    // effort, and a failed shutdown falls back to the defer's kill.
    if (child) |*c| {
        client.shutdown() catch {};
        _ = c.wait(ctx.io) catch {};
        child = null;
    }
}

// -- helpers ----------------------------------------------------------------

/// Spawn `serve` as a child the way a consuming application does: the same binary
/// through its own executable path, an ephemeral port, and the READY line parsed off the child's stdout. The child inherits
/// this process's stderr, so its diagnostics land in this terminal, unless
/// `$ZIGFERENCER_LOG` is set. Then they go to `serve_trace_file`, so the per-tick
/// traces do not split the live stream.
fn spawnServer(ctx: cli.Context, w: *Io.Writer, model_name: []const u8, max_sessions: u16, vram_slots: ?u16, kv_ram_mib: ?u32, kv_dtype: ?dtype.KvDtype, child: *?std.process.Child) !Io.net.IpAddress {
    const ms = try std.fmt.allocPrint(ctx.arena, "{d}", .{max_sessions});
    var argv: std.ArrayList([]const u8) = .empty;
    const self_exe = try std.process.executablePathAlloc(ctx.io, ctx.arena);
    // The demo owns this child and stops it with `shutdown` on the happy path.
    try argv.appendSlice(ctx.arena, &.{ self_exe, "serve", model_name, "--port", "0", "--max-sessions", ms, "--allow-shutdown" });
    if (vram_slots) |vs| {
        const vss = try std.fmt.allocPrint(ctx.arena, "{d}", .{vs});
        try argv.appendSlice(ctx.arena, &.{ "--vram-slots", vss });
    }
    if (kv_ram_mib) |kr| {
        const krs = try std.fmt.allocPrint(ctx.arena, "{d}", .{kr});
        try argv.appendSlice(ctx.arena, &.{ "--kv-ram", krs });
    }
    // Only a flag the operator gave is forwarded. A resolved default on the line
    // overrides the child's own registry lookup.
    if (kv_dtype) |d| {
        try argv.appendSlice(ctx.arena, &.{ "--kv-dtype", @tagName(d) });
    }
    // With tracing on, through a set $ZIGFERENCER_LOG, the child server's
    // per-tick diagnostics interleave with the token stream this demo prints
    // live. Two processes share one terminal, and the stream carries no
    // inter-token newline, so one child log line splits a word. The child's
    // stderr goes to a file instead, which stays ordered and greppable under
    // `tail -f`. The default keeps it inherited.
    const tracing = corelog.rawEnv() != null;
    const trace_file: ?Io.File = if (tracing)
        (Io.Dir.cwd().createFile(ctx.io, serve_trace_file, .{}) catch null)
    else
        null;
    child.* = try std.process.spawn(ctx.io, .{
        .argv = argv.items,
        .stdout = .pipe,
        .stderr = if (trace_file) |f| .{ .file = f } else .inherit,
    });
    if (trace_file) |f| f.close(ctx.io); // the child holds its own copy now
    // The child id is a HANDLE on Windows, so only a POSIX build prints it.
    if (comptime builtin.os.tag == .windows)
        try w.print("spawned:  zigferencer serve {s}\n", .{model_name})
    else
        try w.print("spawned:  zigferencer serve {s} (pid {d})\n", .{ model_name, child.*.?.id.? });
    if (trace_file != null) try w.print("traces:   {s}  (server stderr, `tail -f` to watch)\n", .{serve_trace_file});
    try w.flush();

    var read_buf: [256]u8 = undefined;
    var reader: Io.File.Reader = .init(child.*.?.stdout.?, ctx.io, &read_buf);
    // Spawn failures print on stdout, the demo's own channel. A scope-listing
    // `$ZIGFERENCER_LOG`, in the "only these" grammar, silences every other
    // scope's log.err, so a std.log report here vanishes exactly when the
    // operator turned tracing on to see more.
    const line = reader.interface.takeDelimiterExclusive('\n') catch {
        if (trace_file != null)
            w.print("server exited before READY. See {s} for its diagnostics\n", .{serve_trace_file}) catch {}
        else
            w.print("server exited before READY (set $ZIGFERENCER_LOG for its diagnostics)\n", .{}) catch {};
        w.flush() catch {};
        return error.ServerStartFailed;
    };
    // "READY tcp 127.0.0.1:36847"
    return parseReady(line) orelse {
        w.print("server printed \"{s}\" where READY was expected\n", .{line}) catch {};
        w.flush() catch {};
        return error.ServerStartFailed;
    };
}

/// The address in a "READY tcp host:port" line, null on any other shape.
fn parseReady(line: []const u8) ?Io.net.IpAddress {
    var it = std.mem.tokenizeScalar(u8, line, ' ');
    const ready = it.next() orelse return null;
    const transport = it.next() orelse return null;
    const hostport = it.next() orelse return null;
    if (!std.mem.eql(u8, ready, "READY") or !std.mem.eql(u8, transport, "tcp")) return null;
    const colon = std.mem.lastIndexOfScalar(u8, hostport, ':') orelse return null;
    const port = std.fmt.parseInt(u16, hostport[colon + 1 ..], 10) catch return null;
    return Io.net.IpAddress.parse(hostport[0..colon], port) catch null;
}

/// The four acts. It returns the total generated tokens. `max_seq` is the
/// server's per-session context budget, taken from its hello. The demo caps every
/// turn to stay inside it, so it runs the same on any model. A session that fills
/// its context ends its turns with a note instead of overflowing.
fn playScript(ctx: cli.Context, w: *Io.Writer, p: style.Palette, client: *Client, turns: u32, max_tokens: u32, max_seq: u32, max_sessions: u16, temperature: f32, seed: u64) !u64 {
    const sampling = struct {
        fn at(temp: f32, base: u64, n: u64) protocol.Sampling {
            return .{ .temperature = temp, .top_k = 40, .top_p = 0.95, .seed = base + n };
        }
    };

    const t_open = Io.Clock.Timestamp.now(ctx.io, .awake);
    const a = try client.openSession(system_prompt, sampling.at(temperature, seed, 0));
    const cold_open_ms = time.msBetween(t_open, Io.Clock.Timestamp.now(ctx.io, .awake));
    try w.print("{s}system:{s}  {s}{s}{s}\n", .{ p.dim, p.reset, p.cyan, system_prompt, p.reset });
    try w.print("{s}session: a={d} (system prompt prefilled once, server-side){s}\n", .{ p.dim, a, p.reset });
    try w.print("{s}ctx budget: {d} tokens/session. Turns generate up to {d}, capped to fit{s}\n\n", .{ p.dim, max_seq, max_tokens, p.reset });
    var total: u64 = 0;
    // The committed position. The server reports the truth in every `done`. The
    // system prompt was just prefilled, so `a` sits at a small non-zero position.
    // A low estimate is safe, because the first `done` corrects it.
    var a_pos: u32 = 0;

    // -- act 1: the first turn on session a, streamed live ------------------
    const q1 = "In one sentence, what is a KV cache in transformer inference?";
    try w.print("{s}[turn 1 | session a]{s} {s}{s}{s}\n{s}[a]{s} ", .{ p.dim, p.reset, p.cyan, q1, p.reset, p.dim, p.reset });
    try w.flush();
    const t1 = try oneTurnStreamed(w, p, client, a, q1, genCap(max_seq, a_pos, max_tokens), &total);
    a_pos = t1.pos;

    // -- act 2: follow-ups on the same session, over the kept KV ------------
    var round: u32 = 0;
    while (round < turns) : (round += 1) {
        const cap = genCap(max_seq, a_pos, max_tokens);
        if (cap == 0) {
            try w.print("{s}[session a filled its {d}-token context after {d} tokens]{s}\n", .{ p.dim, max_seq, a_pos, p.reset });
            break;
        }
        const q = followups[round % followups.len];
        const prev = a_pos;
        try w.print("\n{s}[turn {d} | session a]{s} {s}{s}{s}\n{s}[a]{s} ", .{ p.dim, round + 2, p.reset, p.cyan, q, p.reset, p.dim, p.reset });
        try w.flush();
        const t = try oneTurnStreamed(w, p, client, a, q, cap, &total);
        a_pos = t.pos;
        try w.print("{s}  (kv_pos {d} -> {d}: the prior {d} tokens stayed in the KV cache, not re-prefilled){s}\n", .{ p.dim, prev, a_pos, prev, p.reset });
        if (t.stopped) break;
    }

    // -- act 3: clone the conversation three ways, batch, demux -------------
    const cap3 = genCap(max_seq, a_pos, max_tokens);
    try w.print("\n{s}[clone] session a -> b, c (each inherits a's system prompt + history, own seed){s}\n", .{ p.dim, p.reset });
    // Everything buffered so far must reach the terminal before the demo blocks
    // on the wire. A parked server otherwise hides acts that already ran, and the
    // stall reads two acts earlier than it is.
    try w.flush();
    if (cap3 == 0) {
        try w.print("{s}[skipped: session a has used {d}/{d} ctx, no room for a batched turn]{s}\n", .{ p.dim, a_pos, max_seq, p.reset });
        return total;
    }
    const b = client.cloneSession(a, sampling.at(temperature, seed, 1)) catch |err| return noteCloneRefused(w, p, client, err, total);
    const c = client.cloneSession(a, sampling.at(temperature, seed, 2)) catch |err| return noteCloneRefused(w, p, client, err, total);
    try w.print("{s}[batch] sessions a={d} b={d} c={d} take a different turn in one decode batch:{s}\n\n", .{ p.dim, a, b, c, p.reset });
    // The demux prints only when every turn is done, so the act header goes out
    // before the first blocking read.
    try w.flush();
    total += try batchedTurnsDemuxed(ctx, w, p, client, &.{
        .{ .sid = a, .name = "a", .prompt = "Give one real-world use of caching." },
        .{ .sid = b, .name = "b", .prompt = "Name one alternative to a KV cache." },
        .{ .sid = c, .name = "c", .prompt = "What is prefill versus decode?" },
    }, &.{ cap3, cap3, cap3 });

    // -- act 4: a warm open served by the prefix cache ----------------------
    if (max_sessions < 4) return total; // a/b/c hold the whole pool
    try w.print("\n{s}[warm open] fresh session, same system prompt{s}\n", .{ p.dim, p.reset });
    try w.flush();
    const t_warm = Io.Clock.Timestamp.now(ctx.io, .awake);
    const d = try client.openSession(system_prompt, sampling.at(temperature, seed, 3));
    const warm_open_ms = time.msBetween(t_warm, Io.Clock.Timestamp.now(ctx.io, .awake));
    try w.print("{s}session d={d} opened in {d:.0} ms (cold open took {d:.0} ms). The server's stats line counts the hit{s}\n", .{ p.dim, d, warm_open_ms, cold_open_ms, p.reset });
    const q4 = "In five words, what did we discuss?";
    const cap4 = genCap(max_seq, 0, max_tokens);
    if (cap4 == 0) return total;
    try w.print("{s}[turn 1 | session d]{s} {s}{s}{s}\n{s}[d]{s} ", .{ p.dim, p.reset, p.cyan, q4, p.reset, p.dim, p.reset });
    try w.flush();
    _ = try oneTurnStreamed(w, p, client, d, q4, cap4, &total);
    return total;
}

/// Tokens a turn can generate for a session at committed position `pos`, under
/// the server's per-session budget `max_seq` and the prompt headroom. Zero means
/// the session is out of context. This is the safe-regime check: a turn run at
/// this cap satisfies `pos + ctx_reserve + cap <= max_seq`, which the call site
/// asserts. The check is model-agnostic, because `max_seq` is whatever the server
/// reported in its hello, and `pos` is the count each `done` states.
fn genCap(max_seq: u32, pos: u32, requested: u32) u32 {
    const room = max_seq -| pos -| ctx_reserve;
    if (room < min_turn_tokens) return 0;
    return @min(requested, room);
}

/// A refused fork ends the demo where it stands, with the tokens it already
/// earned. On a server refusal the client keeps the err frame's detail, so the
/// note prints the server's own reason. A transport failure has no frame, so the
/// error name stands in.
fn noteCloneRefused(w: *Io.Writer, p: style.Palette, client: *const Client, err: anyerror, total: u64) !u64 {
    const reason: []const u8 = if (err == error.ServerError)
        (if (client.last_error) |*e| e.detail() else @errorName(err))
    else
        @errorName(err);
    try w.print("{s}[clone refused: {s}. The server declined the fork, so the batched act is skipped]{s}\n", .{ p.dim, reason, p.reset });
    try w.flush();
    return total;
}

/// One whole turn on an otherwise-quiet connection. It generates up to `cap`
/// tokens and prints them as they arrive. It returns the session's new committed
/// position and whether the session hit a context limit. The cap math prevents a
/// context-limit reply, and prompts are byte-estimated, so such a reply is
/// reported and not fatal. The session stops there.
fn oneTurnStreamed(w: *Io.Writer, p: style.Palette, client: *Client, sid: u32, text: []const u8, cap: u32, total: *u64) !Turn {
    _ = try client.generate(sid, text, cap, null);
    try w.writeAll(p.green); // generated text is green until the turn's reset
    while (true) {
        const ev = try client.next();
        switch (ev.msg) {
            .token => |t| {
                try w.writeAll(t.text);
                try w.flush();
                total.* += 1;
            },
            .done => |d| {
                try w.print("{s}\n", .{p.reset});
                try w.flush();
                return .{ .pos = d.kv_pos, .stopped = d.finish_reason == .context_full };
            },
            .err => |e| {
                try w.writeAll(p.reset);
                // `turn_too_long` is the only context-class error code. A full
                // context mid-generation arrives as `done{context_full}`, in the
                // arm above. The cap math prevents this code, and prompts are
                // byte-estimated, so the demo takes it as a graceful stop.
                if (e.code == .turn_too_long) {
                    try w.print("\n{s}[session out of context: {t}]{s}\n", .{ p.dim, e.code, p.reset });
                    return .{ .pos = std.math.maxInt(u32), .stopped = true };
                }
                log.err("server error: {t} {s}", .{ e.code, e.detail });
                return error.ServerError;
            },
            else => return error.UnexpectedReply,
        }
    }
}

/// Fire every turn together and demultiplex the interleaved token events by
/// session id, which is the shape a client's session loop implements. It prints
/// each reply, and how often the wire switched sessions mid-stream, which proves
/// the turns shared decode ticks. It returns the total generated tokens.
fn batchedTurnsDemuxed(ctx: cli.Context, w: *Io.Writer, p: style.Palette, client: *Client, batch: []const SessionTurn, caps: []const u32) !u64 {
    std.debug.assert(batch.len == caps.len);
    std.debug.assert(batch.len >= 1);
    for (batch, caps) |turn, cap| _ = try client.generate(turn.sid, turn.prompt, cap, null);
    const replies = try ctx.arena.alloc(std.ArrayList(u8), batch.len);
    for (replies) |*r| r.* = .empty;
    var done_count: usize = 0;
    var switches: u64 = 0;
    var last_sid: ?u32 = null;
    var total: u64 = 0;
    while (done_count < batch.len) {
        const ev = try client.next();
        const slot = slotOf(batch, ev.header.session_id) orelse return error.UnexpectedReply;
        switch (ev.msg) {
            .token => |t| {
                try replies[slot].appendSlice(ctx.arena, t.text);
                total += 1;
                if (last_sid != null and last_sid != ev.header.session_id) switches += 1;
                last_sid = ev.header.session_id;
            },
            .done => done_count += 1,
            .err => |e| {
                if (e.code == .turn_too_long) {
                    try w.print("{s}[{s} out of context: {t}]{s}\n", .{ p.dim, batch[slot].name, e.code, p.reset });
                    done_count += 1; // that session is finished for this act
                } else {
                    log.err("server error: {t} {s}", .{ e.code, e.detail });
                    return error.ServerError;
                }
            },
            else => return error.UnexpectedReply,
        }
    }
    for (batch, replies) |turn, reply| {
        try w.print("{s}[{s}]{s} {s}{s}{s}\n{s}  ->{s} {s}{s}{s}\n", .{ p.dim, turn.name, p.reset, p.cyan, turn.prompt, p.reset, p.dim, p.reset, p.green, reply.items, p.reset });
    }
    try w.print("{s}(token events switched sessions {d} times mid-stream: all {d} turns shared the batch){s}\n", .{ p.dim, switches, batch.len, p.reset });
    try w.flush();
    return total;
}

fn slotOf(batch: []const SessionTurn, sid: u32) ?usize {
    for (batch, 0..) |turn, i| if (turn.sid == sid) return i;
    return null;
}
