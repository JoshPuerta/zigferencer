//! `zigferencer serve-openai <pub_name>` is the OpenAI-compatible HTTP frontend. It parses the
//! flags and hands an `OpenAiLauncher` to `serve_boot.buildAndServe`, so it batches requests over
//! the scheduler of `serve` on every backend. It serves on `std.http.Server`, which adds no
//! dependency. stdout holds only the READY line.
//!
//! Comptime. The launcher is generic over the engine type, so every backend shares one body.

const std = @import("std");

const context = @import("context.zig");
const args = @import("args.zig");
const flags_mod = @import("flag_groups.zig");
const codec_parts = @import("boot/codec_parts.zig");
const serve_boot = @import("boot/serve_boot.zig");
const limits = @import("../core/limits.zig");
const http_server = @import("../serve/openai/http_server.zig");
const map = @import("../serve/openai/map.zig");
const RawCodec = @import("../serve/openai/raw_codec.zig").RawCodec;
const scheduler_mod = @import("../serve/scheduler.zig");

const log = std.log.scoped(.serve_openai);

// -- types ------------------------------------------------------------------

/// Builds the `RawCodec` and the HTTP `Server` over the shared engine, announces the bind, and
/// runs.
const OpenAiLauncher = struct {
    host: []const u8,
    port: u16,
    default_max_tokens: u32,
    sampling_defaults: map.SamplingDefaults,

    pub fn serve(
        self: OpenAiLauncher,
        comptime EngineT: type,
        ctx: context.Context,
        engine: *EngineT,
        parts: codec_parts.CodecParts,
        sched_cfg: scheduler_mod.Config,
    ) !void {
        // Only `chatml_qwen` can resolve the thinking mode to on, and the splitter then needs tags.
        std.debug.assert((parts.think_ids != null) == (parts.kind == .chatml_qwen));

        // Build the codec from the model parts and the request defaults.
        const defaults = self.sampling_defaults;
        const codec: RawCodec = .{
            .vocab = parts.vocab,
            .pieces = parts.pieces,
            .eos_id = parts.eos_id,
            .kind = parts.kind,
            .think_ids = parts.think_ids,
            .model_id = sched_cfg.model_name,
            .max_seq = sched_cfg.max_seq,
            .default_max_tokens = map.resolveDefaultMaxTokens(
                self.default_max_tokens,
                sched_cfg.max_seq,
            ),
            .defaults = .init(defaults, sched_cfg.recommended_think, sched_cfg.recommended_plain),
            .think_forced_close = sched_cfg.think.force_len > 0,
        };
        log.info("sampling defaults: {t}, think row temperature {d}, plain row temperature {d}", .{
            defaults, codec.defaults.think.temperature, codec.defaults.plain.temperature,
        });

        // Bind, announce, and run.
        var server = try http_server.Server(EngineT).init(
            ctx.arena,
            ctx.gpa,
            ctx.io,
            engine,
            &codec,
            sched_cfg,
            .{ .tcp = .{ .host = self.host, .port = self.port } },
        );
        try serve_boot.printReadyTcp(&ctx.stdout.interface, server.boundAddress());
        try server.run();
        log.info("shutdown complete", .{});
    }
};

/// The shared serve options plus the HTTP bind and the request defaults.
const Flags = struct {
    pub_name: ?[]const u8 = null,
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    max_tokens: u32 = limits.openai_default_max_tokens,
    sampling: map.SamplingDefaults = .model,
    serve: flags_mod.ServeFlags = .{},

    pub const positional = "pub_name";
    pub const help: args.Help = .{
        .usage = "zigferencer serve-openai <pub_name> [flags]   " ++
            "(<pub_name> omitted -> pick a model)",
        .about = "OpenAI-compatible HTTP API: " ++
            "POST /v1/chat/completions (+SSE), /v1/models, /healthz.\n",
        .footer =
        \\examples:
        \\  zigferencer serve-openai SmolLM2-135M-Instruct-Q8_0 --port 8080
        \\  curl localhost:8080/v1/models
        \\
        \\Batched over the same scheduler as `serve`. Runs on every backend.
        \\
        ,
    };
    pub const docs = .{
        .host = args.Doc{ .group = "bind", .text = "TCP bind address", .default = "127.0.0.1" },
        .port = args.Doc{ .group = "bind", .text = "TCP port. 0 = ephemeral, on READY" },
        .max_tokens = args.Doc{
            .group = "requests",
            .text = "the most tokens one completion generates when a request\n" ++
                "omits max_tokens. 0 = until the window ends",
        },
        .sampling = args.Doc{
            .group = "requests",
            .text = "sampling of the fields a request omits: model (the card row\n" ++
                "of the thinking mode of the request) or greedy",
        },
        .max_sessions = args.Doc{
            .group = "sessions",
            .text = "concurrent requests batched. Cap 8 CPU, 128 fused-GPU",
        },
    };
};

// -- api --------------------------------------------------------------------

pub fn run(ctx: context.Context) !void {
    const flags = try args.parseInto(Flags, ctx) orelse return;

    const launcher: OpenAiLauncher = .{
        .host = flags.host,
        .port = flags.port,
        .default_max_tokens = flags.max_tokens,
        .sampling_defaults = flags.sampling,
    };
    return serve_boot.buildAndServe(ctx, flags.pub_name, &flags.serve, .{}, launcher);
}
