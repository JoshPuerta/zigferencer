//! `zigferencer serve-demo` is a scripted run against a serve instance. By default it
//! spawns `serve` as a child and drives the whole path: spawn, parse READY, talk, shut
//! down. `--connect host:port` drives a running server and leaves it running.
//!
//! Comptime. The target OS picks how the demo reports the child process.
//!
//! The script has four acts:
//!   act 1  one turn on session a, streamed token by token.
//!   act 2  `--turns` follow-ups on session a. Each turn prefills only its new prompt.
//!   act 3  clone a into b and c, then run one turn on each in one decode batch.
//!   act 4  open a fresh session d with the same system prompt, served by the prefix cache.

const std = @import("std");
const zigferencer = @import("zigferencer");
const builtin = @import("builtin");
const Io = std.Io;
const time = zigferencer.time;

const context = zigferencer.cli.context;
const args = zigferencer.cli.args;
const style = zigferencer.cli.style;
const select = zigferencer.cli.select;
const protocol = zigferencer.serve_protocol;
const Client = zigferencer.serve_client.Client;
const corelog = zigferencer.log;
const dtype = zigferencer.dtype;

const log = std.log.scoped(.serve_demo);

// -- constants --------------------------------------------------------------

/// Where the spawned server writes its stderr when tracing is on, so its per-tick
/// diagnostics do not split the live token stream.
const serve_trace_file = "zigferencer-serve.log";

/// The one system prompt the demo prefills, then clones.
const system_prompt =
    "You are a concise technical assistant. Answer in one or two short sentences.";

/// Follow-up turns for act 2, cycled when `--turns` asks for more than the list.
const followups = [_][]const u8{
    "Now explain it like I'm five.",
    "What's the main downside?",
};

const turns_max: u32 = 8;
const max_tokens_max: u32 = 256;

/// Act 3 holds sessions a, b, and c at once, so a smaller pool fails at the flags.
const min_sessions: u16 = 3;

/// Act 4 needs a fourth session on top of a, b, and c.
const warm_open_min_sessions: u16 = 4;

const sampling_top_k: u32 = 40;
const sampling_top_p: f32 = 0.95;

/// Tokens held back per turn for its own prompt (template wrapper plus the carried
/// pending token). The generation cap leaves this room, so the server never
/// returns `turn_too_long`.
const ctx_reserve: u32 = 96;

/// Below this many tokens a turn is not worth running, and the session ends its turns.
const min_turn_tokens: u32 = 16;

/// Size of the buffer for the READY line of the spawned server.
const ready_line_bytes = 256;

// -- types ------------------------------------------------------------------

/// One turn's outcome: the new committed position from the server's `done`, and
/// whether the session hit a context limit and must stop taking turns.
const Turn = struct { pos: u32, stopped: bool };

/// Whether act 3 ran. A skipped batch also skips act 4.
const BatchOutcome = enum { ran, skipped };

/// One session's turn in a batch: the session id, a short label, and the prompt.
const SessionTurn = struct { sid: u32, name: []const u8, prompt: []const u8 };

/// The values of the flags that shape the script.
const ScriptOptions = struct {
    turns: u32,
    max_tokens: u32,
    /// The server's per-session window from its hello.
    max_seq: u32,
    max_sessions: u16,
    temperature: f32,
    seed: u64,
};

/// What the acts share: the output, the connection, and the running token count.
const Script = struct {
    ctx: context.Context,
    w: *Io.Writer,
    p: style.Palette,
    client: *Client,
    opt: ScriptOptions,
    total: u64 = 0,
};

/// The flags that choose the child server.
const SpawnOptions = struct {
    model_name: []const u8,
    max_sessions: u16,
    vram_slots: ?u16,
    kv_ram_mib: ?u32,
    kv_dtype: ?dtype.KvDtype,
};

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
        .connect = args.Doc{
            .group = "connection",
            .value = "HOST:PORT",
            .text = "drive a running server",
            .default = "spawn a child",
        },
        .turns = args.Doc{ .group = "script", .text = "follow-up rounds on session a, 1..8" },
        .max_tokens = args.Doc{
            .group = "script",
            .text = "the most tokens one turn generates, 1..256",
        },
        .temperature = args.Doc{ .group = "script", .text = "sampling temperature. 0 = greedy" },
        .seed = args.Doc{ .group = "script", .text = "base seed. Sessions a/b/c use seed, +1, +2" },
        .max_sessions = args.Doc{
            .group = "script",
            .text = "logical session pool for the spawned server, >=3",
        },
        .vram_slots = args.Doc{
            .group = "gpu (fused-GPU build only)",
            .text = "VRAM KV slots for the spawned server. Below --max-sessions\nit demos KV swap",
        },
        .kv_ram = args.Doc{
            .group = "gpu (fused-GPU build only)",
            .value = "MiB",
            .text = "RAM budget for evicted KV blobs on the spawned server.\n" ++
                "Small values demo backpressure",
        },
        .kv_dtype = args.Doc{
            .group = "kv",
            .text = "KV element type for the spawned server (the model gates it)",
        },
    };
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: context.Context) !void {
    // Validate the flags and set up the output.
    const flags = try args.parseInto(Flags, ctx) orelse return;
    try validate(ctx, flags);

    const w = &ctx.stdout.interface;

    // Cyan is the input the demo sends, green the text the model generates, dim the annotation.
    const pal = style.Palette.init(ctx.color);

    // A leaked child holds the whole model mapped. Kill reaps it on every exit path,
    // and the explicit wait below nulls `child` on the normal path.
    var child: ?std.process.Child = null;
    defer if (child) |*c| c.kill(ctx.io);

    // Connect to the server.
    const address = try serverAddress(ctx, w, flags, &child);
    var client = try Client.connectTcp(ctx.gpa, ctx.io, address);
    defer client.deinit();

    const info = try client.hello();
    try w.print("{s}server:   max_sessions={d} batch={d} ctx={d}{s}\n\n", .{
        pal.dim, info.max_sessions, info.max_batch, info.max_seq, pal.reset,
    });

    // Play the script and print the totals.
    const t0 = Io.Clock.Timestamp.now(ctx.io, .awake);
    const total = try playScript(ctx, w, pal, &client, .{
        .turns = flags.turns,
        .max_tokens = flags.max_tokens,
        .max_seq = info.max_seq,
        .max_sessions = info.max_sessions,
        .temperature = flags.temperature,
        .seed = flags.seed,
    });
    const wall_s = time.msBetween(t0, Io.Clock.Timestamp.now(ctx.io, .awake)) / 1000.0;
    try w.print("\n{s}done: {d} tokens in {d:.1}s ({d:.1} tok/s aggregate){s}\n", .{
        pal.dim, total, wall_s, @as(f64, @floatFromInt(total)) / wall_s, pal.reset,
    });
    try w.flush();

    // Stop the child. A failed shutdown falls back to the kill in the defer.
    if (child) |*c| {
        client.shutdown() catch {};
        _ = c.wait(ctx.io) catch {};
        child = null;
    }
}

// -- helpers ----------------------------------------------------------------

fn validate(ctx: context.Context, flags: Flags) !void {
    if (flags.turns == 0 or flags.turns > turns_max) {
        return args.fail(ctx.io, "--turns must be in 1..8", .{});
    }
    if (flags.max_tokens == 0 or flags.max_tokens > max_tokens_max) {
        return args.fail(ctx.io, "--max-tokens must be in 1..256", .{});
    }
    if (!(flags.temperature >= 0) or !std.math.isFinite(flags.temperature)) {
        return args.fail(ctx.io, "--temperature must be 0 or more", .{});
    }
    if (flags.max_sessions < min_sessions) {
        return args.fail(ctx.io, "--max-sessions must be >= 3 for the demo script", .{});
    }

    if (flags.connect == null) return;

    if (flags.vram_slots != null) {
        log.info(
            "--vram-slots is ignored with --connect (the running server set its own pool)",
            .{},
        );
    }

    if (flags.kv_ram != null) {
        log.info(
            "--kv-ram is ignored with --connect (the running server sized its own RAM tier)",
            .{},
        );
    }

    if (flags.kv_dtype != null) {
        log.info(
            "--kv-dtype is ignored with --connect (the running server framed its own KV)",
            .{},
        );
    }
}

/// The address of the running server from `--connect`, or of a child that this call spawns.
fn serverAddress(
    ctx: context.Context,
    w: *Io.Writer,
    flags: Flags,
    child: *?std.process.Child,
) !Io.net.IpAddress {
    if (flags.connect) |hostport| {
        const colon = std.mem.findScalarLast(u8, hostport, ':') orelse
            return args.fail(ctx.io, "--connect wants host:port", .{});
        const port = std.fmt.parseInt(u16, hostport[colon + 1 ..], 10) catch
            return args.fail(ctx.io, "--connect wants host:port", .{});
        return Io.net.IpAddress.parse(hostport[0..colon], port);
    }

    const model = try select.model(ctx, flags.pub_name);
    return spawnServer(ctx, w, .{
        .model_name = model.pub_name,
        .max_sessions = flags.max_sessions,
        .vram_slots = flags.vram_slots,
        .kv_ram_mib = flags.kv_ram,
        .kv_dtype = flags.kv_dtype,
    }, child);
}

/// Spawn `serve` from the same executable on an ephemeral port and parse the READY
/// line off its stdout. The child inherits stderr, unless `$ZIGFERENCER_LOG` is set.
/// Then its stderr goes to `serve_trace_file`.
fn spawnServer(
    ctx: context.Context,
    w: *Io.Writer,
    opt: SpawnOptions,
    child: *?std.process.Child,
) !Io.net.IpAddress {
    const argv = try serverArgv(ctx, opt);

    // Spawn the child. Two processes share one terminal, and one child log line splits a word.
    const tracing = corelog.rawEnv() != null;
    const trace_file: ?Io.File = if (tracing)
        (Io.Dir.cwd().createFile(ctx.io, serve_trace_file, .{}) catch null)
    else
        null;

    child.* = try std.process.spawn(ctx.io, .{
        .argv = argv,
        .stdout = .pipe,
        .stderr = if (trace_file) |f| .{ .file = f } else .inherit,
    });
    if (trace_file) |f| {
        f.close(ctx.io); // the child holds its own copy now
    }

    // Announce the child. The child id is a HANDLE on Windows, so only a POSIX build prints it.
    if (comptime builtin.os.tag == .windows) {
        try w.print("spawned:  zigferencer serve {s}\n", .{opt.model_name});
    } else {
        try w.print(
            "spawned:  zigferencer serve {s} (pid {d})\n",
            .{ opt.model_name, child.*.?.id.? },
        );
    }

    if (trace_file != null) {
        try w.print("traces:   {s}  (server stderr, `tail -f` to watch)\n", .{serve_trace_file});
    }

    try w.flush();

    // Read the READY line.
    var read_buffer: [ready_line_bytes]u8 = undefined;
    var reader: Io.File.Reader = .init(child.*.?.stdout.?, ctx.io, &read_buffer);

    // Failures print on stdout. A `$ZIGFERENCER_LOG` that lists scopes silences
    // `log.err` of every other scope, so a log report vanishes when tracing is on.
    const line = reader.interface.takeDelimiterExclusive('\n') catch {
        if (trace_file != null) {
            w.print(
                "server exited before READY. See {s} for its diagnostics\n",
                .{serve_trace_file},
            ) catch {};
        } else {
            w.print(
                "server exited before READY (set $ZIGFERENCER_LOG for its diagnostics)\n",
                .{},
            ) catch {};
        }

        w.flush() catch {};
        return error.ServerStartFailed;
    };

    // Parse "READY tcp 127.0.0.1:36847".
    return parseReady(line) orelse {
        w.print("server printed \"{s}\" where READY was expected\n", .{line}) catch {};
        w.flush() catch {};
        return error.ServerStartFailed;
    };
}

/// The command line of the child. Only a flag that the operator gave goes on it, so
/// a resolved default does not override the child's own registry lookup.
fn serverArgv(ctx: context.Context, opt: SpawnOptions) ![]const []const u8 {
    var argv: std.ArrayList([]const u8) = .empty;
    const self_exe = try std.process.executablePathAlloc(ctx.io, ctx.arena);
    const max_sessions = try std.fmt.allocPrint(ctx.arena, "{d}", .{opt.max_sessions});

    // The demo owns this child and stops it with `shutdown` on the normal path.
    try argv.appendSlice(ctx.arena, &.{
        self_exe,
        "serve",
        opt.model_name,
        "--port",
        "0",
        "--max-sessions",
        max_sessions,
        "--allow-shutdown",
    });
    if (opt.vram_slots) |vs| {
        const vss = try std.fmt.allocPrint(ctx.arena, "{d}", .{vs});
        try argv.appendSlice(ctx.arena, &.{ "--vram-slots", vss });
    }

    if (opt.kv_ram_mib) |kr| {
        const krs = try std.fmt.allocPrint(ctx.arena, "{d}", .{kr});
        try argv.appendSlice(ctx.arena, &.{ "--kv-ram", krs });
    }

    if (opt.kv_dtype) |d| {
        try argv.appendSlice(ctx.arena, &.{ "--kv-dtype", @tagName(d) });
    }

    return argv.items;
}

/// The address in a "READY tcp host:port" line, null on any other shape.
fn parseReady(line: []const u8) ?Io.net.IpAddress {
    var it = std.mem.tokenizeScalar(u8, line, ' ');
    const ready = it.next() orelse return null;
    const transport = it.next() orelse return null;
    const hostport = it.next() orelse return null;
    if (!std.mem.eql(u8, ready, "READY") or !std.mem.eql(u8, transport, "tcp")) return null;

    const colon = std.mem.findScalarLast(u8, hostport, ':') orelse return null;
    const port = std.fmt.parseInt(u16, hostport[colon + 1 ..], 10) catch return null;
    return Io.net.IpAddress.parse(hostport[0..colon], port) catch null;
}

/// Run the four acts and return the total generated tokens. Every turn caps its
/// generation to fit the server's window, so the script runs the same on any model.
fn playScript(
    ctx: context.Context,
    w: *Io.Writer,
    p: style.Palette,
    client: *Client,
    opt: ScriptOptions,
) !u64 {
    var s: Script = .{ .ctx = ctx, .w = w, .p = p, .client = client, .opt = opt };

    // Open session a and print the preamble.
    const t_open = Io.Clock.Timestamp.now(ctx.io, .awake);
    const a = try client.openSession(system_prompt, samplingFor(opt, 0));
    const cold_open_ms = time.msBetween(t_open, Io.Clock.Timestamp.now(ctx.io, .awake));
    try w.print("{s}system:{s}  {s}{s}{s}\n", .{ p.dim, p.reset, p.cyan, system_prompt, p.reset });
    try w.print(
        "{s}session: a={d} (system prompt prefilled once, server-side){s}\n",
        .{ p.dim, a, p.reset },
    );
    try w.print(
        "{s}ctx budget: {d} tokens/session. Turns generate up to {d}, capped to fit{s}\n\n",
        .{
            p.dim, opt.max_seq, opt.max_tokens, p.reset,
        },
    );

    // Run the acts. A skipped batch also skips the warm open.
    var a_pos = try actFirstTurn(&s, a);
    a_pos = try actFollowups(&s, a, a_pos);
    if (try actBatch(&s, a, a_pos) == .skipped) return s.total;

    if (opt.max_sessions >= warm_open_min_sessions) {
        try actWarmOpen(&s, cold_open_ms);
    }

    return s.total;
}

fn samplingFor(opt: ScriptOptions, session_index: u64) protocol.Sampling {
    return .{
        .temperature = opt.temperature,
        .top_k = sampling_top_k,
        .top_p = sampling_top_p,
        .seed = opt.seed + session_index,
    };
}

/// Act 1. Returns the committed position of session a.
fn actFirstTurn(s: *Script, a: u32) !u32 {
    const w = s.w;
    const p = s.p;
    const q1 = "In one sentence, what is a KV cache in transformer inference?";
    try w.print(
        "{s}[turn 1 | session a]{s} {s}{s}{s}\n{s}[a]{s} ",
        .{ p.dim, p.reset, p.cyan, q1, p.reset, p.dim, p.reset },
    );
    try w.flush();

    // Position 0 is a low estimate, and the first `done` corrects it.
    const t1 = try oneTurnStreamed(s, a, q1, genCap(s.opt.max_seq, 0, s.opt.max_tokens));
    return t1.pos;
}

/// Act 2. The turns run over the KV that session a keeps. Returns its new position.
fn actFollowups(s: *Script, a: u32, a_pos_start: u32) !u32 {
    const w = s.w;
    const p = s.p;
    var a_pos = a_pos_start;

    var round: u32 = 0;
    while (round < s.opt.turns) : (round += 1) {
        const cap = genCap(s.opt.max_seq, a_pos, s.opt.max_tokens);
        if (cap == 0) {
            try w.print("{s}[session a filled its {d}-token context after {d} tokens]{s}\n", .{
                p.dim, s.opt.max_seq, a_pos, p.reset,
            });
            break;
        }

        const q = followups[round % followups.len];
        const prev = a_pos;
        try w.print("\n{s}[turn {d} | session a]{s} {s}{s}{s}\n{s}[a]{s} ", .{
            p.dim, round + 2, p.reset, p.cyan, q, p.reset, p.dim, p.reset,
        });
        try w.flush();

        const t = try oneTurnStreamed(s, a, q, cap);
        a_pos = t.pos;
        try w.print(
            "{s}  (kv_pos {d} -> {d}: " ++
                "the prior {d} tokens stayed in the KV cache, not re-prefilled){s}\n",
            .{
                p.dim, prev, a_pos, prev, p.reset,
            },
        );
        if (t.stopped) break;
    }

    return a_pos;
}

/// Act 3. Returns `.skipped` when the act did not run.
fn actBatch(s: *Script, a: u32, a_pos: u32) !BatchOutcome {
    const w = s.w;
    const p = s.p;
    const client = s.client;
    const cap = genCap(s.opt.max_seq, a_pos, s.opt.max_tokens);

    try w.print(
        "\n{s}[clone] session a -> b, c (each inherits a's system prompt + history, own seed){s}\n",
        .{ p.dim, p.reset },
    );
    // A parked server must not hide the acts that already ran.
    try w.flush();
    if (cap == 0) {
        try w.print(
            "{s}[skipped: session a has used {d}/{d} ctx, no room for a batched turn]{s}\n",
            .{
                p.dim, a_pos, s.opt.max_seq, p.reset,
            },
        );
        return .skipped;
    }

    const b = client.cloneSession(
        a,
        samplingFor(s.opt, 1),
    ) catch |err| return noteCloneRefused(s, err);
    const c = client.cloneSession(
        a,
        samplingFor(s.opt, 2),
    ) catch |err| return noteCloneRefused(s, err);
    try w.print(
        "{s}[batch] sessions a={d} b={d} c={d} take a different turn in one decode batch:{s}\n\n",
        .{
            p.dim, a, b, c, p.reset,
        },
    );
    // The demux prints only when every turn is done, so the header goes out first.
    try w.flush();

    s.total += try batchedTurnsDemuxed(s, &.{
        .{ .sid = a, .name = "a", .prompt = "Give one real-world use of caching." },
        .{ .sid = b, .name = "b", .prompt = "Name one alternative to a KV cache." },
        .{ .sid = c, .name = "c", .prompt = "What is prefill versus decode?" },
    }, &.{ cap, cap, cap });
    return .ran;
}

/// Act 4. A fresh session d opens with the system prompt that the prefix cache holds.
fn actWarmOpen(s: *Script, cold_open_ms: f64) !void {
    const w = s.w;
    const p = s.p;

    try w.print("\n{s}[warm open] fresh session, same system prompt{s}\n", .{ p.dim, p.reset });
    try w.flush();

    const t_warm = Io.Clock.Timestamp.now(s.ctx.io, .awake);
    const d = try s.client.openSession(system_prompt, samplingFor(s.opt, 3));
    const warm_open_ms = time.msBetween(t_warm, Io.Clock.Timestamp.now(s.ctx.io, .awake));
    try w.print(
        "{s}session d={d} opened in {d:.0} ms (cold open took {d:.0} ms). " ++
            "The server's stats line counts the hit{s}\n",
        .{
            p.dim, d, warm_open_ms, cold_open_ms, p.reset,
        },
    );

    const q4 = "In five words, what did we discuss?";
    const cap = genCap(s.opt.max_seq, 0, s.opt.max_tokens);
    if (cap == 0) return;

    try w.print(
        "{s}[turn 1 | session d]{s} {s}{s}{s}\n{s}[d]{s} ",
        .{ p.dim, p.reset, p.cyan, q4, p.reset, p.dim, p.reset },
    );
    try w.flush();
    _ = try oneTurnStreamed(s, d, q4, cap);
}

/// Tokens a turn can generate for a session at committed position `pos`. Zero means
/// the session is out of context. A turn run at this cap keeps
/// `pos + ctx_reserve + cap <= max_seq`, whatever `max_seq` the server reported.
fn genCap(max_seq: u32, pos: u32, requested: u32) u32 {
    const room = max_seq -| pos -| ctx_reserve;
    if (room < min_turn_tokens) return 0;
    return @min(requested, room);
}

/// A refused fork ends the demo where it stands, so the caller skips act 4.
fn noteCloneRefused(s: *Script, err: anyerror) !BatchOutcome {
    const reason: []const u8 = if (err == error.ServerError)
        (if (s.client.last_error) |*e| e.detail() else @errorName(err))
    else
        @errorName(err);

    try s.w.print(
        "{s}[clone refused: {s}. The server declined the fork, so the batched act is skipped]{s}\n",
        .{
            s.p.dim, reason, s.p.reset,
        },
    );
    try s.w.flush();
    return .skipped;
}

/// One whole turn on a quiet connection. It prints up to `cap` tokens as they arrive.
/// The cap math prevents a context-limit reply, but prompts are byte-estimated, so
/// such a reply ends the session's turns and does not fail the demo.
fn oneTurnStreamed(s: *Script, sid: u32, text: []const u8, cap: u32) !Turn {
    const w = s.w;
    const p = s.p;
    _ = try s.client.generate(sid, text, cap, null);

    try w.writeAll(p.green); // generated text is green until the turn's reset
    while (true) {
        const ev = try s.client.next();
        switch (ev.msg) {
            .token => |t| {
                try w.writeAll(t.text);
                try w.flush();
                s.total += 1;
            },
            .done => |d| {
                try w.print("{s}\n", .{p.reset});
                try w.flush();
                return .{ .pos = d.kv_pos, .stopped = d.finish_reason == .context_full };
            },
            .err => |e| {
                try w.writeAll(p.reset);
                // A full context mid-generation arrives as `done{context_full}`.
                // `turn_too_long` is the only context-class error code.
                if (e.code == .turn_too_long) {
                    try w.print(
                        "\n{s}[session out of context: {t}]{s}\n",
                        .{ p.dim, e.code, p.reset },
                    );
                    return .{ .pos = std.math.maxInt(u32), .stopped = true };
                }

                log.err("server error: {t} {s}", .{ e.code, e.detail });
                return error.ServerError;
            },
            else => return error.UnexpectedReply,
        }
    }
}

/// Fire every turn together and demultiplex the token events by session id. It prints
/// each reply and how often the wire switched sessions, which shows the turns shared
/// decode ticks. Returns the generated tokens.
fn batchedTurnsDemuxed(s: *Script, batch: []const SessionTurn, caps: []const u32) !u64 {
    std.debug.assert(batch.len == caps.len);
    std.debug.assert(batch.len >= 1);

    const w = s.w;
    const p = s.p;
    const client = s.client;

    for (batch, caps) |turn, cap| {
        _ = try client.generate(turn.sid, turn.prompt, cap, null);
    }

    // Collect the token events per session.
    const replies = try s.ctx.arena.alloc(std.ArrayList(u8), batch.len);
    for (replies) |*r| {
        r.* = .empty;
    }

    var done_count: usize = 0;
    var switches: u64 = 0;
    var last_sid: ?u32 = null;
    var total: u64 = 0;
    while (done_count < batch.len) {
        const ev = try client.next();
        const slot = slotOf(batch, ev.header.session_id) orelse return error.UnexpectedReply;
        switch (ev.msg) {
            .token => |t| {
                try replies[slot].appendSlice(s.ctx.arena, t.text);
                total += 1;
                if (last_sid != null and last_sid != ev.header.session_id) {
                    switches += 1;
                }

                last_sid = ev.header.session_id;
            },
            .done => done_count += 1,
            .err => |e| {
                if (e.code != .turn_too_long) {
                    log.err("server error: {t} {s}", .{ e.code, e.detail });
                    return error.ServerError;
                }

                try w.print(
                    "{s}[{s} out of context: {t}]{s}\n",
                    .{ p.dim, batch[slot].name, e.code, p.reset },
                );
                done_count += 1; // that session is finished for this act
            },
            else => return error.UnexpectedReply,
        }
    }

    // Print the replies.
    try printBatchReplies(s, batch, replies, switches);
    try w.flush();
    return total;
}

fn printBatchReplies(
    s: *Script,
    batch: []const SessionTurn,
    replies: []const std.ArrayList(u8),
    switches: u64,
) !void {
    const w = s.w;
    const p = s.p;

    for (batch, replies) |turn, reply| {
        try w.print("{s}[{s}]{s} {s}{s}{s}\n{s}  ->{s} {s}{s}{s}\n", .{
            p.dim,
            turn.name,
            p.reset,
            p.cyan,
            turn.prompt,
            p.reset,
            p.dim,
            p.reset,
            p.green,
            reply.items,
            p.reset,
        });
    }

    try w.print(
        "{s}(token events switched sessions {d} times mid-stream: " ++
            "all {d} turns shared the batch){s}\n",
        .{
            p.dim, switches, batch.len, p.reset,
        },
    );
}

fn slotOf(batch: []const SessionTurn, sid: u32) ?usize {
    for (batch, 0..) |turn, i| {
        if (turn.sid == sid) return i;
    }

    return null;
}
