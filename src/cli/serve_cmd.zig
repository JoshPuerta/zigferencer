//! `zigferencer serve <pub_name>` is the binary frontend. It parses the flags and hands
//! a `BinaryLauncher` to `serve_boot.buildAndServe`. A client can stop the server with
//! `shutdown` only under `--allow-shutdown`, because a shared server must not stop on
//! the request of one client. stdout holds only the READY line, which a spawner parses.

const std = @import("std");

const context = @import("context.zig");
const args = @import("args.zig");
const flags_mod = @import("flag_groups.zig");
const codec_parts = @import("boot/codec_parts.zig");
const serve_boot = @import("boot/serve_boot.zig");
const serve_checks = @import("boot/serve_checks.zig");
const limits = @import("../core/limits.zig");
const event = @import("../serve/event.zig");
const scheduler_mod = @import("../serve/scheduler.zig");
const server_mod = @import("../serve/server.zig");

const log = std.log.scoped(.serve);

// -- types ------------------------------------------------------------------

/// Builds the chat-aware codec and the framed `Server` over the shared engine,
/// announces the bind, and runs until `shutdown`.
const BinaryLauncher = struct {
    transport: server_mod.Transport,
    socket_path: ?[]const u8,

    pub fn serve(
        self: BinaryLauncher,
        comptime EngineT: type,
        ctx: context.Context,
        engine: *EngineT,
        parts: codec_parts.CodecParts,
        sched_cfg: scheduler_mod.Config,
    ) !void {
        const codec: event.Codec = .{
            .vocab = parts.vocab,
            .kind = parts.kind,
            .pieces = parts.pieces,
            .eos_id = parts.eos_id,
        };
        const ServerT = server_mod.Server(EngineT, event.Codec);
        var server = try ServerT.init(
            ctx.arena,
            ctx.gpa,
            ctx.io,
            engine,
            &codec,
            sched_cfg,
            self.transport,
        );

        if (self.socket_path) |sp| {
            try serve_boot.printReadyUnix(&ctx.stdout.interface, sp);
        } else {
            try serve_boot.printReadyTcp(&ctx.stdout.interface, server.boundAddress());
        }

        try server.run();
        log.info("shutdown complete", .{});
    }
};

/// The shared serve options plus the binary transport and the session lease.
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
        \\A ZMQ DEALER client connects with tcp://HOST:PORT or ipc://PATH.
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
        .port = args.Doc{
            .group = "bind",
            .text = "TCP port. 0 = ephemeral, on READY",
            .default = "0",
        },
        .socket = args.Doc{
            .group = "bind",
            .value = "PATH",
            .text = "unix socket instead of TCP",
        },
        .allow_shutdown = args.Doc{
            .group = "bind",
            .text = "let a client stop the server through the wire",
            .default = "off",
        },
        .sessions_per_connection = args.Doc{
            .group = "sessions",
            .text = "sessions one connection can hold",
            .default = "--max-sessions",
        },
        .resume_ttl = args.Doc{
            .group = "sessions",
            .value = "MS",
            .text = std.fmt.comptimePrint(
                "longest lease a client can keep past a drop, cap {d}.\n0 = resumption off",
                .{limits.max_resume_ttl_ms},
            ),
        },
    };
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: context.Context) !void {
    const flags = try args.parseInto(Flags, ctx) orelse return;

    const socket_path = flags.socket;
    if (socket_path != null and (flags.host != null or flags.port != null)) {
        return args.fail(ctx.io, "--socket and --host/--port are mutually exclusive", .{});
    }

    const wire: serve_checks.Wire = .{
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

    const launcher: BinaryLauncher = .{ .transport = transport, .socket_path = socket_path };

    return serve_boot.buildAndServe(ctx, flags.pub_name, &flags.serve, wire, launcher);
}
