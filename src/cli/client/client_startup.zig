//! The start of a chat: the flags, the server check, and the first session. Both loops read
//! the `Startup` value this file resolves once.

const std = @import("std");
const Io = std.Io;

const context = @import("../context.zig");
const args = @import("../args.zig");
const limits = @import("../../core/limits.zig");
const chat_mod = @import("../../core/chat.zig");
const protocol = @import("../../serve/protocol.zig");
const client_mod = @import("../../serve/client.zig");
const transcript_mod = @import("client_transcript.zig");

const Client = client_mod.Client;

const client_history = @import("client_history.zig");
const client_input = @import("client_input.zig");
const client_output = @import("client_output.zig");

const Out = client_output.Out;

// -- constants --------------------------------------------------------------

const startup_fmt = client_output.session_open_fmt ++
    ". context {d} tokens per session. turn cap {d} tokens";

const keep_granted_fmt = "The server keeps session {d} for {d} ms after a drop.";
pub const keep_off_note = "The server keeps no session after a drop. A drop ends the chat.";

// -- types ------------------------------------------------------------------

/// The flags of `client`, and the sampling and per-turn fields they set.
pub const Options = struct {
    connect: ?[]const u8 = null,
    socket: ?[]const u8 = null,
    system: []const u8 = "",
    max_tokens: u32 = 0,
    /// Null takes `reserveFor` at each compaction.
    reserve: ?u32 = null,
    sampling: SamplingSource = .greedy,
    temperature: ?f32 = null,
    min_p: ?f32 = null,
    repetition_penalty: ?f32 = null,
    seed: ?u64 = null,
    thinking: chat_mod.ThinkingFlag = .auto,
    thinking_budget: ?u32 = null,
    no_stream: bool = false,
    /// The lease that the client asks for each session, in milliseconds.
    resume_ttl: u32 = 0,

    pub const help: args.Help = .{
        .usage = "zigferencer client (--connect IP:PORT | --socket PATH) [flags]",
        .about = "Chat with a running server over the binary wire. Each line you type is " ++
            "one turn.\n",
        .footer = client_input.commands_text ++
            \\
            \\examples:
            \\  zigferencer client --connect 127.0.0.1:8000
            \\  zigferencer client --socket /tmp/zigferencer.sock --system "Answer in one sentence."
            \\
        ,
    };
    pub const docs = .{
        .connect = args.Doc{
            .group = "connection",
            .value = "IP:PORT",
            .text = "connect over TCP. The host must be an IP address",
        },
        .socket = args.Doc{
            .group = "connection",
            .value = "PATH",
            .text = "connect to a server on a unix socket",
        },
        .system = args.Doc{
            .group = "session",
            .text = "system prompt of each session",
            .default = "none",
        },
        .max_tokens = args.Doc{
            .group = "session",
            .text = "the most tokens one turn generates. 0 = until the window ends",
        },
        .reserve = args.Doc{
            .group = "session",
            .text = "positions a compaction leaves free for the next turn",
            .default = std.fmt.comptimePrint(
                "the longest turn so far, from window / {d}\nto window / {d}",
                .{
                    transcript_mod.reserve_floor_divisor, transcript_mod.reserve_ceiling_divisor,
                },
            ),
        },
        .sampling = args.Doc{
            .group = "sampling",
            .text = "model for the card row of the think mode of the turns, or greedy",
        },
        .temperature = args.Doc{
            .group = "sampling",
            .text = "sampling temperature. 0 is greedy",
            .default = "0, or the card value",
        },
        .min_p = args.Doc{
            .group = "sampling",
            .text = "keep the ids with at least X times the top probability, 0..1",
            .default = "0, or the card value",
        },
        .repetition_penalty = args.Doc{
            .group = "sampling",
            .text = "factor on the ids already in the context, above 0 and at most 2.\n1 is off",
            .default = "1, or the card value",
        },
        .seed = args.Doc{
            .group = "sampling",
            .text = "sampling seed of the session. Set, every turn restarts the\nsampling " ++
                "stream at N, so a turn replays. Unset, the stream\nof the session " ++
                "continues across turns. The client has no\nfrequency penalty flag. The " ++
                "server refuses that penalty until\nit supports it",
            .default = "unset",
        },
        .thinking = args.Doc{
            .group = "thinking (models with think tags)",
            .text = "the think switch of every turn. auto sends no switch",
            .default = "auto, the server mode",
        },
        .thinking_budget = args.Doc{
            .group = "thinking (models with think tags)",
            .text = "think tokens before the server closes the block. 0 is unbounded",
            .default = "the server value",
        },
        .no_stream = args.Doc{ .group = "mode", .text = "print each reply when its turn ends" },
        .resume_ttl = args.Doc{
            .group = "reconnect (streaming mode)",
            .value = "MS",
            .text = "ask the server to keep each session for MS after a drop.\n0 keeps " ++
                "none, and a drop ends the chat. The server clamps MS\nto its own maximum. " ++
                "On a drop the client tries to reconnect\nat most 8 times, after 100 ms, " ++
                "then twice as long each time,\nat most 2 s. An attempt waits at most 10 s " ++
                "for the greeting\nand 120 s for the resume. /quit ends the wait at once. " ++
                "The\nclient then resumes the session. A turn that the server never\nran " ++
                "goes out again. If the lease is gone, the client opens a\nnew session and " ++
                "sends the kept turns",
        },
    };

    /// The think switch that every turn sends. Null sends none.
    pub fn thinkOpen(o: *const Options) ?bool {
        return switch (o.thinking) {
            .auto => null,
            .on => true,
            .off => false,
        };
    }

    pub fn turnsOpen(o: *const Options, info: Client.HelloInfo) bool {
        return o.thinkOpen() orelse (info.think_mode == .on);
    }

    /// The sampling of every session. `model` takes the card row that the
    /// server sends for the think mode of the turns.
    pub fn sessionSampling(o: *const Options, info: Client.HelloInfo) protocol.Sampling {
        var s = switch (o.sampling) {
            .greedy => protocol.Sampling.greedy,
            .model => info.recommendedFor(o.turnsOpen(info)),
        };

        if (o.temperature) |t| {
            s.temperature = t;
        }

        if (o.min_p) |m| {
            s.min_p = m;
        }

        if (o.repetition_penalty) |r| {
            s.repetition_penalty = r;
        }

        s.seed = o.seed orelse 0;
        return s;
    }

    /// The per-turn fields of every `generate`. With `--seed`, each turn
    /// asks for a reseed, so the turn replays.
    pub fn turn(
        o: *const Options,
        max_tokens: u32,
        session_sampling: protocol.Sampling,
    ) Client.TurnOptions {
        return .{
            .max_tokens = max_tokens,
            .sampling = if (o.seed != null) session_sampling else null,
            .reseed = o.seed != null,
            .think_open = o.thinkOpen(),
            .think_budget = o.thinking_budget,
        };
    }
};

const SamplingSource = enum { model, greedy };

/// Parsed once, so each reconnect dials the same server.
pub const Target = union(enum) {
    tcp: Io.net.IpAddress,
    unix: []const u8,
};

/// The lease of the first session. A zero `ttl_ms` keeps no session.
pub const Keep = struct { ttl_ms: u32, token: ?protocol.Token };

/// The values that startup resolves once and that both loops read.
pub const Startup = struct {
    out: Out,
    opts: *const Options,
    client: *Client,
    info: Client.HelloInfo,
    sampling: protocol.Sampling,
    session: Client.Opened,
    hist: *client_history.History,
};

// -- api --------------------------------------------------------------------

/// Null means `--help` printed and the command has nothing more to do.
pub fn parseOptions(ctx: context.Context) !?Options {
    const o = try args.parseInto(Options, ctx) orelse return null;
    if ((o.connect == null) == (o.socket == null)) {
        return args.fail(ctx.io, "give exactly one of --connect or --socket", .{});
    }

    if (o.temperature) |t| {
        if (!(t >= 0) or !std.math.isFinite(t)) {
            return args.fail(ctx.io, "--temperature must be 0 or more", .{});
        }
    }

    if (o.thinking_budget) |b| {
        if (b > limits.max_model_ctx) {
            return args.fail(
                ctx.io,
                "--thinking-budget must be in 0..{d}",
                .{limits.max_model_ctx},
            );
        }
    }

    if (o.min_p) |m| {
        if (!(m >= 0 and m <= 1)) return args.fail(ctx.io, "--min-p must be in 0..1", .{});
    }

    if (o.repetition_penalty) |r| {
        if (!(r > 0 and r <= limits.max_repetition_penalty)) {
            return args.fail(
                ctx.io,
                "--repetition-penalty must be above 0 and at most {d}",
                .{limits.max_repetition_penalty},
            );
        }
    }

    if (o.resume_ttl > limits.max_resume_ttl_ms) {
        return args.fail(
            ctx.io,
            "--resume-ttl must be in 0..{d} ms, the lease ceiling",
            .{limits.max_resume_ttl_ms},
        );
    }
    if (o.resume_ttl > 0 and o.no_stream) {
        return args.fail(
            ctx.io,
            "--resume-ttl needs the streaming mode, so it does not go with --no-stream",
            .{},
        );
    }
    if (o.system.len > protocol.system_prompt_bytes_max) {
        return args.fail(
            ctx.io,
            "--system holds {d} bytes. One session_open frame carries at most {d}.",
            .{
                o.system.len, protocol.system_prompt_bytes_max,
            },
        );
    }

    return o;
}

pub fn parseTarget(ctx: context.Context, opts: *const Options) !Target {
    if (opts.socket) |path| return .{ .unix = path };

    const hostport = opts.connect.?;
    const colon = std.mem.findScalarLast(u8, hostport, ':') orelse
        return args.fail(ctx.io, "--connect wants IP:PORT", .{});
    const port = std.fmt.parseInt(u16, hostport[colon + 1 ..], 10) catch
        return args.fail(ctx.io, "--connect wants IP:PORT", .{});
    const address = Io.net.IpAddress.parse(hostport[0..colon], port) catch
        return args.fail(ctx.io, "--connect wants an IP address and a port", .{});
    return .{ .tcp = address };
}

/// Refuse a server whose `hello_ack` contradicts the options. The window
/// comes from the network, and the blocking mode sizes its id buffer from it.
pub fn checkServer(out: Out, opts: *const Options, info: Client.HelloInfo) !void {
    if (info.max_seq == 0 or info.max_seq > limits.max_model_ctx) {
        try out.note(
            "The server reports a window of {d} positions. The client accepts 1 to {d}.",
            .{
                info.max_seq, limits.max_model_ctx,
            },
        );
        return error.WindowOutOfRange;
    }

    if (opts.reserve) |reserve| {
        if (reserve >= info.max_seq) {
            try out.note(
                "--reserve {d} is at or past the window of {d} positions.",
                .{ reserve, info.max_seq },
            );
            return error.ReserveOutOfRange;
        }
    }

    if (info.think_mode == .none and (opts.thinking == .on or opts.thinking_budget != null)) {
        try out.note(
            "The server model has no think tags, so --thinking on and --thinking-budget do " ++
                "not apply.",
            .{},
        );
        return error.ThinkingUnsupported;
    }
}

pub fn printSampling(
    out: Out,
    opts: *const Options,
    info: Client.HelloInfo,
    sampling: protocol.Sampling,
) !void {
    try out.dim(
        "sampling: think mode {t}, turns open {}, think budget {d}, temperature {d:.2}, " ++
            "top_k {d}, top_p {d:.2}, min_p {d:.2}, presence_penalty {d:.2}, " ++
            "repetition_penalty {d:.2}",
        .{
            info.think_mode,
            opts.turnsOpen(
                info,
            ),
            opts.thinking_budget orelse info.think_budget,
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            sampling.min_p,
            sampling.presence_penalty,
            sampling.repetition_penalty,
        },
    );
}

pub fn printStartup(out: Out, opts: *const Options, info: Client.HelloInfo, session_id: u32) !void {
    const cap = client_history.turnBudget(opts.max_tokens, info.max_seq);
    if (opts.reserve) |reserve| {
        try out.dim(
            startup_fmt ++ ", reserve {d} positions",
            .{ session_id, info.max_seq, cap, reserve },
        );
    } else {
        try out.dim(startup_fmt ++ ", reserve adaptive, from {d} to {d} positions", .{
            session_id,                                 info.max_seq, cap,
            transcript_mod.reserveFor(info.max_seq, 0),
            transcript_mod.reserveFor(
                info.max_seq,
                std.math.maxInt(u32),
            ),
        });
    }
}

pub fn connect(ctx: context.Context, target: Target, opts: *const Options, out: Out) !Client {
    return dial(ctx, target) catch |err| {
        try out.note("cannot connect to {s}: {t}", .{ opts.socket orelse opts.connect.?, err });
        return err;
    };
}

/// Keep the first session for `ttl_ms`. A server with resumption off, or a
/// refused keep, gives no lease, and the chat goes on without one.
pub fn startKeep(
    out: Out,
    client: *Client,
    info: Client.HelloInfo,
    session_id: u32,
    ttl_ms: u32,
) !Keep {
    std.debug.assert(ttl_ms > 0);

    const off: Keep = .{ .ttl_ms = 0, .token = null };

    if (info.resume_ttl_max_ms == 0) {
        try out.note(keep_off_note, .{});
        return off;
    }

    const k = client.keepSession(session_id, ttl_ms) catch |err| {
        if (err != error.ServerError) return err;

        try out.serverError(client.last_error);
        try out.note(keep_off_note, .{});
        return off;
    };
    if (k.granted_ms == 0) {
        try out.note(keep_off_note, .{});
        return off;
    }

    try out.dim(keep_granted_fmt, .{ session_id, k.granted_ms });
    return .{ .ttl_ms = k.granted_ms, .token = k.token };
}

/// A failed write of the report is dropped, so the startup error itself
/// reaches `main`.
pub fn startupFailed(out: Out, client: *const Client, err: anyerror) anyerror {
    if (err == error.ServerError) {
        out.serverError(client.last_error) catch {};
    } else {
        out.note("startup failed: {t}", .{err}) catch {};
    }

    return err;
}

// -- helpers: startup -------------------------------------------------------

fn dial(ctx: context.Context, target: Target) !Client {
    return switch (target) {
        .unix => |path| Client.connectUnix(ctx.gpa, ctx.io, path),
        .tcp => |address| Client.connectTcp(ctx.gpa, ctx.io, address),
    };
}

// -- tests ------------------------------------------------------------------

test "Options.sessionSampling follows a new hello_ack, and an explicit flag keeps its value" {
    var old = std.mem.zeroes(Client.HelloInfo);
    old.think_mode = .off;
    old.recommended_plain = protocol.Sampling.greedy;
    old.recommended_plain.temperature = 0.7;
    old.recommended_think = protocol.Sampling.greedy;
    old.recommended_think.temperature = 1.0;
    var new = old;
    new.think_mode = .on;
    new.recommended_think.temperature = 0.6;

    const o: Options = .{ .sampling = .model };
    try std.testing.expect(!o.turnsOpen(old));
    try std.testing.expect(o.turnsOpen(new));
    try std.testing.expectEqual(@as(f32, 0.7), o.sessionSampling(old).temperature);
    try std.testing.expectEqual(@as(f32, 0.6), o.sessionSampling(new).temperature);

    const pinned: Options = .{ .sampling = .model, .thinking = .off, .temperature = 0.3 };
    try std.testing.expect(!pinned.turnsOpen(new));
    try std.testing.expectEqual(@as(f32, 0.3), pinned.sessionSampling(new).temperature);
}
