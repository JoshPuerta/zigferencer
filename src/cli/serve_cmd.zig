//! `zigferencer serve <pub_name>` is the serving mode. It loads one model, binds
//! one socket, prints one `READY` line on stdout, and runs the engine loop. A
//! client can stop it with `shutdown` only under `--allow-shutdown`, because a
//! shared server must not stop on the request of one client. Without the flag
//! the process runs until a signal. A session is one conversation: the server
//! tokenizes, applies the chat template, keeps KV across turns, and advances up
//! to `--max-batch` of them per decode tick. This file is only the
//! binary-protocol connector. It parses the flags and hands a `BinaryLauncher` to
//! the shared engine builder in `serve_common.zig`. The wire format lives in
//! `serve/protocol.zig`, and the engine in `serve/engine.zig` and
//! `serve/scheduler.zig`, with `serve/server.zig` as the binary transport adapter
//! over them. The target traffic is many concurrent sessions that are short, need
//! low latency, and share a prefix. The OpenAI HTTP API (`serve-openai`) is a
//! separate connector over the same engine.
//!
//! stdout carries exactly the READY line, which the spawning process parses.
//! Every diagnostic goes through the scoped stderr logger.

const std = @import("std");
const Io = std.Io;

const cli = @import("command.zig");
const args = @import("args.zig");
const flags_mod = @import("flags.zig");
const style = @import("style.zig");
const serve_common = @import("serve_common.zig");
const kv_budget = @import("../core/kv_budget.zig");
const event = @import("../serve/event.zig");
const server_mod = @import("../serve/server.zig");
const scheduler_mod = @import("../serve/scheduler.zig");
const limits = @import("../core/limits.zig");
const chat = @import("../core/chat.zig");

const log = std.log.scoped(.serve);

// -- types ------------------------------------------------------------------

/// Drives the binary wire protocol. It builds the chat-aware codec and the framed
/// `Server` over the shared engine, announces the bind, and runs until
/// `shutdown`.
const BinaryLauncher = struct {
    transport: server_mod.Transport,
    socket_path: ?[]const u8,

    pub fn serve(
        self: BinaryLauncher,
        comptime EngineT: type,
        ctx: cli.Context,
        engine: *EngineT,
        parts: serve_common.CodecParts,
        sched_cfg: scheduler_mod.Config,
    ) !void {
        const codec: event.Codec = .{
            .vocab = parts.vocab,
            .kind = parts.kind,
            .pieces = parts.pieces,
            .eos_id = parts.eos_id,
        };
        var server = try server_mod.Server(EngineT, event.Codec).init(ctx.arena, ctx.gpa, ctx.io, engine, &codec, sched_cfg, self.transport);
        if (self.socket_path) |sp| {
            try serve_common.printReadyUnix(ctx.io, sp);
        } else {
            try serve_common.printReadyTcp(ctx.io, server.boundAddress());
        }
        try server.run();
        log.info("shutdown complete", .{});
    }
};

/// The flags of `serve`: the shared serve knobs plus the binary transport and
/// the session lease.
const Flags = struct {
    pub_name: ?[]const u8 = null,
    host: ?[]const u8 = null,
    port: ?u16 = null,
    socket: ?[]const u8 = null,
    allow_shutdown: bool = false,
    sessions_per_connection: ?u16 = null,
    resume_ttl: u32 = 0,
    serve: flags_mod.ServeFlags = .{},

    pub const positional = "pub_name";
    pub const help: args.Help = .{
        .usage = "zigferencer serve <pub_name> [flags]   (<pub_name> omitted -> pick a model)",
        .about =
        \\Serve sessions over a framed binary protocol. One READY line on stdout.
        \\A ZeroMQ DEALER client connects with tcp://HOST:PORT or ipc://PATH.
        \\
        ,
        .footer =
        \\examples:
        \\  zigferencer serve SmolLM2-135M-Instruct-Q8_0 --max-sessions 4
        \\  zigferencer serve --socket /tmp/zf.sock
        \\
        \\wire protocol: src/serve/protocol.zig
        \\
        ,
    };
    pub const docs = .{
        .host = args.Doc{ .group = "bind", .text = "TCP bind address", .default = "127.0.0.1" },
        .port = args.Doc{ .group = "bind", .text = "TCP port. 0 = ephemeral, on READY", .default = "0" },
        .socket = args.Doc{ .group = "bind", .value = "PATH", .text = "unix socket instead of TCP" },
        .allow_shutdown = args.Doc{ .group = "bind", .text = "let a client stop the server through the wire", .default = "off" },
        .sessions_per_connection = args.Doc{
            .group = "sessions",
            .text = "sessions one connection can hold",
            .default = "--max-sessions",
        },
        .resume_ttl = args.Doc{
            .group = "sessions",
            .value = "MS",
            .text = std.fmt.comptimePrint("longest lease a client can keep past a drop, cap {d}.\n0 = resumption off", .{limits.max_resume_ttl_ms}),
        },
    };
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: cli.Context) !void {
    const flags = try args.parseInto(Flags, ctx) orelse return;
    const socket_path = flags.socket;
    if (socket_path != null and (flags.host != null or flags.port != null)) {
        return args.fail(ctx.io, "--socket and --host/--port are mutually exclusive", .{});
    }
    const wire: serve_common.Wire = .{
        .allow_shutdown = flags.allow_shutdown,
        .sessions_per_connection = flags.sessions_per_connection,
        .resume_ttl_ms = flags.resume_ttl,
    };
    const host = flags.host orelse "127.0.0.1";
    const port = flags.port orelse 0;

    const transport: server_mod.Transport = if (socket_path) |sp|
        .{ .unix = sp }
    else
        .{ .tcp = .{ .host = host, .port = port } };

    return serve_common.buildAndServe(ctx, flags.pub_name, &flags.serve, wire, BinaryLauncher{ .transport = transport, .socket_path = socket_path });
}

// -- helpers ----------------------------------------------------------------
