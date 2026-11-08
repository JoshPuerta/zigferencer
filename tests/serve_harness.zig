//! The shared boot for the serve tests, not a test itself.
//!
//! Comptime. `HarnessFor` builds one harness type per runner, so each runner family shares one
//! boot.
//!
//! Boot. A harness loads the SmolLM2 fixture, builds the CPU `LlamaRunner` and its session pool,
//! and starts an in-process binary server. `QwenHarness` does the same with the CPU `QwenRunner`
//! over the Qwen fixture.
//!
//! Transport. The server listens on an ephemeral TCP port or on a unix socket in a temporary
//! directory. A test drives it through the reference client and stops it over the wire or through
//! the process owner's own request.
//!
//! Fixtures. The harness reads the canonical SmolLM2 Q8_0 or Qwen3.5-2B Q8_0 from
//! `test_fixtures.path`. `boot` returns `error.SkipZigTest` only when that function does. A load
//! failure or a bind failure fails the test.
//!
//! Chat template. The kind defaults to `.none`, because template wrapping has its own unit tests
//! and the reference comparison holds for any template. A test of the turn wraps names its kind.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const zigferencer = @import("zigferencer");
const test_policy = @import("test_policy.zig");
const loader = zigferencer.loader;
const vocab_mod = zigferencer.vocab;
const bpe = zigferencer.bpe;
const LlamaRunner = zigferencer.llama_runner.LlamaRunner;
const QwenRunner = zigferencer.qwen_runner.QwenRunner;
const Sessions = zigferencer.session.Sessions;
const limits = zigferencer.limits;
const protocol = zigferencer.serve_protocol;
const event = zigferencer.serve_event;
const server_mod = zigferencer.serve_server;
const prefill_chunk = zigferencer.serve_prefill_chunk;
const Client = zigferencer.serve_client.Client;

// -- constants --------------------------------------------------------------

pub const max_seq: u32 = 64;
pub const turn_tokens_max: u32 = 12;

pub const prompts = [_][]const u8{ "The dragon flew over", "Tell me a story about" };

// -- types ------------------------------------------------------------------

pub const Endpoint = enum { tcp, unix };

pub const Options = struct {
    max_sessions: u16 = 3,
    max_batch: u16 = 3,
    max_connections: u32 = 8,
    sessions_per_connection: u16 = 0,
    allow_shutdown: bool = true,
    /// The longest lease a client can keep. 0 keeps resumption off.
    resume_ttl_max_ms: u32 = 0,
    /// The session window. The default is the small window of the stream
    /// tests.
    window: u32 = max_seq,
    kind: ChatKind = .none,
    /// The RAM budget of the prefix cache. 0 keeps the cache off.
    prefix_bytes: u64 = 0,
    /// The prefill rule. The default is the rule the server ships on the CPU
    /// runners. A test that needs slow, row-sized prefill names `.spare_rows`.
    prefill: prefill_chunk.Rule = prefill_chunk.cpu_rule,
};

pub const ChatKind = @FieldType(event.Codec, "kind");

pub const Stream = struct {
    ids: [turn_tokens_max]u32 = undefined,
    n: usize = 0,
    reason: protocol.FinishReason = .eos,
};

/// One in-process server over the SmolLM2 fixture.
pub const Harness = HarnessFor(LlamaRunner);

/// One in-process server over the Qwen fixture.
pub const QwenHarness = HarnessFor(QwenRunner);

/// One in-process server over the fixture of `Runner`. The value moves
/// freely, because the server tasks point only into the arena, and the arena
/// state itself lives in the gpa.
pub fn HarnessFor(comptime Runner: type) type {
    const ServeServer = server_mod.Server(Sessions(Runner), event.Codec);
    return struct {
        const Self = @This();

        gpa: Allocator,
        io: Io,
        endpoint: Endpoint,
        arena_state: *std.heap.ArenaAllocator,
        loaded: *loader.Loaded,
        vocab: *vocab_mod.Vocab,
        pieces: []const []const u8,
        /// The pool of the runner. It lives in the arena, so its address stays
        /// fixed while the engine loop runs.
        cpu_pool: *test_policy.Pool,
        runner: *Runner,
        engine: *Sessions(Runner),
        codec: *event.Codec,
        server: *ServeServer,
        /// The TCP address as bound at boot. Teardown frees the listener, so a
        /// fallback connection after the server stops must not read it.
        address: Io.net.IpAddress,
        run_future: Io.Future(void),
        running: bool,
        tmp: ?std.testing.TmpDir,
        /// A unix listener reports a placeholder address, so the harness keeps
        /// the socket path for `connect`.
        path_buf: [Io.net.UnixAddress.max_len]u8,
        path_len: usize,

        pub fn boot(gpa: Allocator, io: Io, endpoint: Endpoint, opts: Options) !Self {
            // The runner holds the row width of the serve default, so a lone
            // prompt prefills in full chunks as it does under the shipped server.
            std.debug.assert(opts.max_batch >= 1);
            std.debug.assert(opts.max_batch <= opts.max_sessions);

            const arena_state = try gpa.create(std.heap.ArenaAllocator);
            errdefer gpa.destroy(arena_state);

            arena_state.* = .init(gpa);
            errdefer arena_state.deinit();

            const arena = arena_state.allocator();
            const fixture_name = if (Runner == QwenRunner)
                "Qwen3.5-2B-Q8_0"
            else
                "SmolLM2-135M-Instruct-Q8_0";

            const fixture = try zigferencer.test_fixtures.path(arena, fixture_name);

            const loaded = try arena.create(loader.Loaded);
            loaded.* = try loader.load(io, arena, fixture);
            errdefer loaded.close();

            const vocab = try arena.create(vocab_mod.Vocab);
            vocab.* = try vocab_mod.buildFromParsed(arena, loaded.parsed);

            const pieces = try arena.alloc([]const u8, vocab.tokens.len);
            for (pieces, 0..) |*piece, id| {
                piece.* = try bpe.decodeAlloc(vocab, arena, &[_]u32{@intCast(id)});
                try std.testing.expect(piece.len <= limits.max_token_text_bytes);
            }

            const codec = try arena.create(event.Codec);
            codec.* = .{
                .vocab = vocab,
                .kind = opts.kind,
                .pieces = pieces,
                .eos_id = vocab.eos_id,
            };

            const policy = test_policy.policy();
            const cpu_pool = try arena.create(test_policy.Pool);
            try test_policy.startPool(cpu_pool, io, policy);
            errdefer cpu_pool.deinit();

            const runner = try arena.create(Runner);
            runner.* = try Runner.init(
                arena,
                loaded,
                zigferencer.kv_budget.pinned(opts.max_sessions, opts.window),
                prefill_chunk.stepRows(opts.prefill, opts.max_sessions, opts.max_batch),
                cpu_pool,
                policy,
                .f32,
            );
            const engine = try arena.create(Sessions(Runner));
            engine.* = try Sessions(Runner).init(arena, runner);

            var tmp: ?std.testing.TmpDir = null;
            errdefer if (tmp) |*t| t.cleanup();

            var path_buf: [Io.net.UnixAddress.max_len]u8 = undefined;
            var path_len: usize = 0;
            const transport: server_mod.Transport = switch (endpoint) {
                .tcp => .{ .tcp = .{} },
                .unix => blk: {
                    tmp = std.testing.tmpDir(.{});
                    const path = try std.fmt.bufPrint(
                        &path_buf,
                        ".zig-cache/tmp/{s}/serve.sock",
                        .{&tmp.?.sub_path},
                    );
                    path_len = path.len;
                    break :blk .{ .unix = path };
                },
            };

            // The launcher's own builder, so the harness boots the think config the server boots.
            const think = zigferencer.codec_parts.schedulerThink(
                try zigferencer.codec_parts.thinkTagsOf(arena, vocab, opts.kind),
                opts.kind,
                0,
            );

            const server = try arena.create(ServeServer);
            server.* = try ServeServer.init(arena, gpa, io, engine, codec, .{
                .max_sessions = opts.max_sessions,
                .sessions_per_connection = opts.sessions_per_connection,
                .max_batch = opts.max_batch,
                .max_seq = opts.window,
                .max_step_rows = runner.maxStepRows(),
                .prefill = opts.prefill,
                .max_connections = opts.max_connections,
                .allow_shutdown = opts.allow_shutdown,
                .resume_ttl_max_ms = opts.resume_ttl_max_ms,
                .model_name = "fixture",
                .vocab_size = loaded.card.vocab_size,
                .eos_id = vocab.eos_id,
                .cache = .{ .ram_bytes = opts.prefix_bytes, .model_fp = 0x7E57 },
                .think = think,
            }, transport);
            const run_future = try io.concurrent(runServerTask, .{server});

            return .{
                .gpa = gpa,
                .io = io,
                .endpoint = endpoint,
                .arena_state = arena_state,
                .loaded = loaded,
                .vocab = vocab,
                .pieces = pieces,
                .cpu_pool = cpu_pool,
                .runner = runner,
                .engine = engine,
                .codec = codec,
                .server = server,
                .address = server.boundAddress(),
                .run_future = run_future,
                .running = true,
                .tmp = tmp,
                .path_buf = path_buf,
                .path_len = path_len,
            };
        }

        /// A new client on the endpoint the server listens on.
        pub fn connect(self: *Self) !Client {
            return switch (self.endpoint) {
                .tcp => try Client.connectTcp(self.gpa, self.io, self.address),
                .unix => try Client.connectUnix(self.gpa, self.io, self.path_buf[0..self.path_len]),
            };
        }

        /// Stop the server over the wire, then wait for the engine loop to return.
        /// A failed handshake on `client` falls back to a fresh connection, then to the owner's
        /// request, so a failing assertion never leaves the process blocked on the server task.
        pub fn stop(self: *Self, client: *Client) void {
            if (!self.running) return;
            client.shutdown() catch self.stopOverFallback();
            self.awaitRun();
        }

        /// Stop the server through a `shutdown` on connection 0, which the server
        /// obeys whatever `allow_shutdown` says, then wait for the engine loop.
        pub fn stopFromOwner(self: *Self) void {
            if (!self.running) return;
            self.submitOwnerShutdown();
            self.awaitRun();
        }

        pub fn deinit(self: *Self) void {
            // A test that returns early skips its stop, and the engine loop must
            // end before the arena that holds its state goes away.
            self.stopFromOwner();
            self.cpu_pool.deinit();
            self.loaded.close();
            self.arena_state.deinit();
            self.gpa.destroy(self.arena_state);
            if (self.tmp) |*t| {
                t.cleanup();
            }

            self.* = undefined;
        }

        fn stopOverFallback(self: *Self) void {
            var fallback = self.connect() catch return self.submitOwnerShutdown();
            defer fallback.deinit();

            fallback.shutdown() catch self.submitOwnerShutdown();
        }

        fn submitOwnerShutdown(self: *Self) void {
            // A closed request queue means the engine loop already tears down.
            self.server.submit(
                .{ .conn = 0, .request_id = 0, .session_id = 0, .body = .shutdown },
            ) catch {};
        }

        fn awaitRun(self: *Self) void {
            self.run_future.await(self.io);
            self.running = false;
        }

        fn runServerTask(server: *ServeServer) void {
            server.run() catch {};
        }
    };
}

// -- api --------------------------------------------------------------------

/// One whole turn on an otherwise-quiet connection.
pub fn oneTurn(client: *Client, sid: u32, prompt: []const u8) !Stream {
    _ = try client.generate(sid, prompt, turn_tokens_max, null);
    var s: Stream = .{};
    var text_buf: [4096]u8 = undefined;
    const t = try client.collectTurn(s.ids[0..], &text_buf);
    s.n = @min(t.n_tokens, s.ids.len);
    s.reason = t.finish_reason;
    return s;
}

/// The serving invariant for one turn: the same token ids and the same
/// finish reason.
pub fn expectSameStream(expected: Stream, actual: Stream) !void {
    try std.testing.expectEqual(expected.reason, actual.reason);
    try std.testing.expectEqualSlices(u32, expected.ids[0..expected.n], actual.ids[0..actual.n]);
}
