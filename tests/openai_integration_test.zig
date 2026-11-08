//! End-to-end gate for the OpenAI HTTP connector over a real model and real sockets. The test
//! starts an in-process `serve-openai` on an ephemeral TCP port and drives it with
//! `std.http.Client`.
//!
//! Wire shape. The wire shape must be exact: a 200 that carries a `chat.completion` object with
//! non-empty content and a finish reason. The gate therefore catches a regression in the HTTP
//! framing, in the JSON, or along the render, tokenize and decode path. A body that names another
//! model must get a 404 that carries the `model_not_found` type, and the served id must pass the
//! same check. Two concurrent requests must also return the text each request returns alone, which
//! is the batch-equivalence rule on the HTTP path. A streamed reply must carry the text of the
//! buffered reply.
//!
//! Window. A turn whose budget passes the room fills the window and ends with `length`. A request
//! that names no budget gets the whole window. A prompt past the window gets a 400 that names both
//! numbers, and a streamed request gets the same JSON body and no event stream.
//!
//! Thinking. A request switches the thinking mode with `chat_template_kwargs`. The fixture has no
//! thinking mode, so `enable_thinking: true` gets a 400 in one sentence. `false` and an absent
//! switch give the plain reply, byte for byte.
//!
//! Sampling. The fixture serves a sampled card row as the default of each omitted field, the shape
//! of a model with a card under `--sampling model`. Every test that compares reply text therefore
//! sends `temperature: 0`. Two requests without a seed give two replies, one fixed seed gives one
//! reply, and the greedy defaults give one reply. A sampling field out of its range and a frequency
//! penalty get a 400 that names the field. A repetition penalty changes the greedy reply, so the
//! server never drops it.
//!
//! Fixture. The test reads the canonical SmolLM2 Q8_0 from `test_fixtures.path`.

const std = @import("std");
const zigferencer = @import("zigferencer");
const test_policy = @import("test_policy.zig");
const loader = zigferencer.loader;
const vocab_mod = zigferencer.vocab;
const bpe = zigferencer.bpe;
const LlamaRunner = zigferencer.llama_runner.LlamaRunner;
const Sessions = zigferencer.session.Sessions;
const limits = zigferencer.limits;
const http_server = zigferencer.serve_openai_http;
const RawCodec = zigferencer.serve_openai_raw_codec.RawCodec;

// -- fixtures ---------------------------------------------------------------

const max_seq: u32 = 128;
const reply_max_bytes = 64 * 1024;

/// The rendered prompt passes the window of 128 tokens.
const long_message = "the quick brown fox jumps over the lazy dog " ** 20;
/// The rendered prompt fits the window with a small room, so the turn fills
/// the window before the model can stop.
const near_full_message = "the quick brown fox jumps over the lazy dog " ** 9;

const OpenAiServer = http_server.Server(Sessions(LlamaRunner));
const map = zigferencer.serve_openai_map;

/// A sampled row in both modes. SmolLM2 has no card, so the test supplies one.
const card_row: zigferencer.serve_protocol.Sampling = .{
    .temperature = 1.5,
    .top_k = 40,
    .top_p = 0.95,
    .seed = 0,
};
const card_defaults: map.DefaultRows = .init(.model, card_row, card_row);

/// An in-process server over the SmolLM2 fixture. The fields point at each
/// other, so a fixture never moves after `setUp`.
const Fixture = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    arena_state: std.heap.ArenaAllocator,
    loaded: loader.Loaded,
    vocab: vocab_mod.Vocab,
    cpu_pool: test_policy.Pool,
    runner: LlamaRunner,
    sessions: Sessions(LlamaRunner),
    codec: RawCodec,
    server: OpenAiServer,
    run_future: std.Io.Future(void),
    uri_buf: [128]u8,
    uri: std.Uri,

    fn setUp(fx: *Fixture) !void {
        return fx.setUpWith(card_defaults);
    }

    fn setUpWith(fx: *Fixture, defaults: map.DefaultRows) !void {
        const io = std.testing.io;
        fx.gpa = std.testing.allocator;
        fx.io = io;
        fx.arena_state = .init(fx.gpa);
        errdefer fx.arena_state.deinit();

        const arena = fx.arena_state.allocator();
        const fixture = try zigferencer.test_fixtures.path(arena, "SmolLM2-135M-Instruct-Q8_0");

        fx.loaded = try loader.load(io, arena, fixture);
        errdefer fx.loaded.close();

        fx.vocab = try vocab_mod.buildFromParsed(arena, fx.loaded.parsed);

        const pieces = try arena.alloc([]const u8, fx.vocab.tokens.len);
        for (pieces, 0..) |*piece, id| {
            piece.* = try bpe.decodeAlloc(&fx.vocab, arena, &[_]u32{@intCast(id)});
            try std.testing.expect(piece.len <= limits.max_token_text_bytes);
        }

        const policy = test_policy.policy();
        try test_policy.startPool(&fx.cpu_pool, io, policy);
        errdefer fx.cpu_pool.deinit();

        fx.runner = try LlamaRunner.init(
            arena,
            &fx.loaded,
            zigferencer.kv_budget.pinned(4, max_seq),
            0,
            &fx.cpu_pool,
            policy,
            .f32,
        );
        fx.sessions = try Sessions(LlamaRunner).init(arena, &fx.runner);
        fx.codec = .{
            .vocab = &fx.vocab,
            .pieces = pieces,
            .eos_id = fx.vocab.eos_id,
            .kind = .chatml_smollm2,
            .think_ids = null,
            .model_id = "fixture",
            .max_seq = max_seq,
            .default_max_tokens = map.resolveDefaultMaxTokens(
                limits.openai_default_max_tokens,
                max_seq,
            ),
            .defaults = defaults,
        };
        fx.server = OpenAiServer.init(arena, fx.gpa, io, &fx.sessions, &fx.codec, .{
            .max_sessions = 4,
            .max_batch = 4,
            .max_seq = max_seq,
            .max_step_rows = 4,
            .allow_shutdown = true,
            .model_name = "fixture",
            .vocab_size = fx.loaded.card.vocab_size,
            .eos_id = fx.vocab.eos_id,
        }, .{ .tcp = .{} }) catch return error.SkipZigTest;

        fx.run_future = try io.concurrent(runServerTask, .{&fx.server});
        errdefer fx.stopServer();

        const uri_str = try std.fmt.bufPrint(
            &fx.uri_buf,
            "http://{f}/v1/chat/completions",
            .{fx.server.boundAddress()},
        );
        fx.uri = try std.Uri.parse(uri_str);
    }

    fn tearDown(fx: *Fixture) void {
        fx.stopServer();
        fx.cpu_pool.deinit();
        fx.loaded.close();
        fx.arena_state.deinit();
    }

    /// A failed assertion must not leave the test blocked on the server task.
    /// A shutdown request breaks the engine loop and closes every connection.
    fn stopServer(fx: *Fixture) void {
        fx.server.submit(
            .{ .conn = 0, .request_id = 0, .session_id = 0, .body = .shutdown },
        ) catch {};
        fx.run_future.await(fx.io);
    }
};

/// One HTTP exchange, read whole. The strings live in the caller's arena.
const Reply = struct {
    status: std.http.Status,
    content_type: []const u8,
    body: []const u8,
};

const Completion = struct {
    choices: []const struct {
        message: struct { content: []const u8 },
        finish_reason: []const u8,
    },
    usage: struct { prompt_tokens: u32, completion_tokens: u32 },
};

const ChunkReply = struct {
    choices: []const struct {
        delta: struct { role: ?[]const u8 = null, content: ?[]const u8 = null },
        finish_reason: ?[]const u8 = null,
    },
};

const ErrorReply = struct {
    @"error": struct {
        code: u16,
        message: []const u8,
        type: []const u8,
        n_prompt_tokens: u32,
        n_ctx: u32,
    },
};

const PlainErrorReply = struct {
    @"error": struct { message: []const u8, type: []const u8 },
};

/// The buffered reply of a turn with no think block, every field in wire
/// order. A parse that refuses unknown fields and a byte-equal round trip pin
/// the shape.
const PlainCompletion = struct {
    id: []const u8,
    object: []const u8,
    created: i64,
    model: []const u8,
    choices: []const struct {
        index: u32,
        message: struct { role: []const u8, content: []const u8 },
        finish_reason: []const u8,
    },
    usage: struct { prompt_tokens: u32, completion_tokens: u32, total_tokens: u32 },
};

fn runServerTask(server: *OpenAiServer) void {
    server.run() catch {};
}

fn post(fx: *const Fixture, arena: std.mem.Allocator, body: []const u8) !Reply {
    var client: std.http.Client = .{ .allocator = fx.gpa, .io = fx.io };
    defer client.deinit();

    var req = try client.request(.POST, fx.uri, .{ .keep_alive = false });
    defer req.deinit();

    try req.sendBodyComplete(try arena.dupe(u8, body));

    var redirect_buf: [4096]u8 = undefined;
    var response = try req.receiveHead(&redirect_buf);
    const status = response.head.status;
    // The head strings die when the body reader starts, so copy them first.
    const content_type = try arena.dupe(u8, response.head.content_type orelse "");
    var transfer_buf: [8192]u8 = undefined;
    const reader = response.reader(&transfer_buf);
    const reply_body = try reader.allocRemaining(arena, .limited(reply_max_bytes));
    return .{ .status = status, .content_type = content_type, .body = reply_body };
}

/// A chat body with one user message. `extra` holds more top-level fields,
/// each with a leading comma.
fn chatBody(arena: std.mem.Allocator, content: []const u8, extra: []const u8) ![]const u8 {
    return std.fmt.allocPrint(
        arena,
        "{{\"messages\":[{{\"role\":\"user\",\"content\":\"{s}\"}}]{s}}}",
        .{ content, extra },
    );
}

/// The body with its completion id replaced. Each request opens its own
/// session, and the id names the session.
fn withoutId(arena: std.mem.Allocator, body: []const u8) ![]const u8 {
    const key = "\"id\":\"";
    const start = (std.mem.find(u8, body, key) orelse return error.NoId) + key.len;
    const len = std.mem.findScalar(u8, body[start..], '"') orelse return error.NoId;
    return std.mem.replaceOwned(u8, arena, body, body[start..][0..len], "chatcmpl-id");
}

const ContentBuf = struct {
    buf: [4096]u8 = undefined,
    len: usize = 0,
    ok: bool = false,

    fn slice(self: *const ContentBuf) []const u8 {
        return self.buf[0..self.len];
    }
};

/// Run one buffered chat completion and keep the reply text. The signature is
/// concrete so `io.concurrent` can start it as a worker. A failure leaves
/// `ok = false`.
fn fetchContent(fx: *const Fixture, prompt: []const u8, out: *ContentBuf) void {
    out.* = .{};
    fetchContentErr(fx, prompt, out) catch {};
}

fn fetchContentErr(fx: *const Fixture, prompt: []const u8, out: *ContentBuf) !void {
    var arena_state = std.heap.ArenaAllocator.init(fx.gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    const reply = try post(
        fx,
        arena,
        try chatBody(arena, prompt, ",\"max_tokens\":24,\"stream\":false,\"temperature\":0"),
    );
    if (reply.status != .ok) return error.HttpStatus;

    const parsed = try std.json.parseFromSliceLeaky(
        Completion,
        arena,
        reply.body,
        .{ .ignore_unknown_fields = true },
    );
    if (parsed.choices.len == 0) return error.NoChoices;

    const content = parsed.choices[0].message.content;
    if (content.len > out.buf.len) return error.TooLong;

    @memcpy(out.buf[0..content.len], content);
    out.len = content.len;
    out.ok = true;
}

// -- tests ------------------------------------------------------------------

test "openai loopback: a chat completion returns a valid chat.completion object" {
    var fx: Fixture = undefined;
    try fx.setUp();
    defer fx.tearDown();

    var arena_state = std.heap.ArenaAllocator.init(fx.gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    const reply = try post(
        &fx,
        arena,
        try chatBody(arena, "Hello", ",\"max_tokens\":16,\"stream\":false"),
    );
    try std.testing.expectEqual(std.http.Status.ok, reply.status);
    try std.testing.expect(
        std.mem.find(u8, reply.body, "\"object\":\"chat.completion\"") != null,
    );
    try std.testing.expect(std.mem.find(u8, reply.body, "\"finish_reason\"") != null);
    try std.testing.expect(std.mem.find(u8, reply.body, "\"role\":\"assistant\"") != null);
    // A non-empty reply: usage reports at least one completion token.
    try std.testing.expect(std.mem.find(u8, reply.body, "\"completion_tokens\":0,") == null);

    // A body that names another model gets a 404 before any session opens.
    const wrong = try post(
        &fx,
        arena,
        \\{"model":"wrong","messages":[{"role":"user","content":"Hello"}],
        \\"max_tokens":16,"stream":false}
        ,
    );
    try std.testing.expectEqual(std.http.Status.not_found, wrong.status);
    try std.testing.expect(std.mem.find(u8, wrong.body, "\"type\":\"model_not_found\"") != null);

    // The served id passes the same check, so the refusal compares two strings.
    const served = try post(
        &fx,
        arena,
        \\{"model":"fixture","messages":[{"role":"user","content":"Hello"}],
        \\"max_tokens":4,"stream":false}
        ,
    );
    try std.testing.expectEqual(std.http.Status.ok, served.status);

    // Two concurrent requests batch through the shared scheduler and return the solo text.
    // The requests pin greedy, so the reply is deterministic.
    const prompt_a = "Name one primary color.";
    const prompt_b = "What is two plus two?";
    var a_solo: ContentBuf = .{};
    var b_solo: ContentBuf = .{};
    fetchContent(&fx, prompt_a, &a_solo);
    fetchContent(&fx, prompt_b, &b_solo);

    var a_conc: ContentBuf = .{};
    var b_conc: ContentBuf = .{};
    var fa = try fx.io.concurrent(fetchContent, .{ &fx, prompt_a, &a_conc });
    var fb = try fx.io.concurrent(fetchContent, .{ &fx, prompt_b, &b_conc });
    fa.await(fx.io);
    fb.await(fx.io);

    try std.testing.expect(a_solo.ok and b_solo.ok and a_conc.ok and b_conc.ok);
    try std.testing.expectEqualStrings(a_solo.slice(), a_conc.slice());
    try std.testing.expectEqualStrings(b_solo.slice(), b_conc.slice());
}

test "openai loopback: a streamed reply carries the buffered text after the role chunk" {
    var fx: Fixture = undefined;
    try fx.setUp();
    defer fx.tearDown();

    var arena_state = std.heap.ArenaAllocator.init(fx.gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    const buffered = try post(
        &fx,
        arena,
        try chatBody(arena, "Hello", ",\"max_tokens\":8,\"temperature\":0"),
    );
    try std.testing.expectEqual(std.http.Status.ok, buffered.status);
    const want = (try std.json.parseFromSliceLeaky(
        Completion,
        arena,
        buffered.body,
        .{ .ignore_unknown_fields = true },
    )).choices[0];

    const streamed = try post(
        &fx,
        arena,
        try chatBody(arena, "Hello", ",\"max_tokens\":8,\"stream\":true,\"temperature\":0"),
    );
    try std.testing.expectEqual(std.http.Status.ok, streamed.status);
    try std.testing.expectEqualStrings("text/event-stream", streamed.content_type);
    const end_event = "data: [DONE]\n\n";
    try std.testing.expect(std.mem.endsWith(u8, streamed.body, end_event));

    var text: std.ArrayList(u8) = .empty;
    var finish: []const u8 = "";
    var n_events: usize = 0;
    var events = std.mem.splitSequence(
        u8,
        streamed.body[0 .. streamed.body.len - end_event.len],
        "\n\n",
    );
    while (events.next()) |event| {
        if (event.len == 0) continue;

        try std.testing.expect(std.mem.startsWith(u8, event, "data: "));
        const chunk = try std.json.parseFromSliceLeaky(
            ChunkReply,
            arena,
            event["data: ".len..],
            .{ .ignore_unknown_fields = true },
        );
        const choice = chunk.choices[0];
        if (n_events == 0) {
            try std.testing.expectEqualStrings("assistant", choice.delta.role orelse "");
        }

        if (choice.delta.content) |content| {
            try text.appendSlice(arena, content);
        }

        if (choice.finish_reason) |reason| {
            finish = reason;
        }

        n_events += 1;
    }

    try std.testing.expectEqualStrings(want.message.content, text.items);
    try std.testing.expectEqualStrings(want.finish_reason, finish);
}

test "openai loopback: a budget past the room fills the window and ends with length" {
    var fx: Fixture = undefined;
    try fx.setUp();
    defer fx.tearDown();

    var arena_state = std.heap.ArenaAllocator.init(fx.gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    const extra = try std.fmt.allocPrint(
        arena,
        ",\"max_tokens\":{d},\"temperature\":0",
        .{max_seq},
    );
    const reply = try post(&fx, arena, try chatBody(arena, near_full_message, extra));
    try std.testing.expectEqual(std.http.Status.ok, reply.status);
    const parsed = try std.json.parseFromSliceLeaky(
        Completion,
        arena,
        reply.body,
        .{ .ignore_unknown_fields = true },
    );
    try std.testing.expect(parsed.usage.prompt_tokens < max_seq);
    try std.testing.expectEqualStrings("length", parsed.choices[0].finish_reason);
    // The prefill samples the first token, and the last token never enters the window.
    try std.testing.expectEqual(
        max_seq - parsed.usage.prompt_tokens + 1,
        parsed.usage.completion_tokens,
    );
}

test "openai loopback: a request with no budget gets the whole window" {
    var fx: Fixture = undefined;
    try fx.setUp();
    defer fx.tearDown();

    var arena_state = std.heap.ArenaAllocator.init(fx.gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    try std.testing.expectEqual(max_seq, fx.codec.default_max_tokens);
    const reply = try post(
        &fx,
        arena,
        try chatBody(arena, "Count from 1 to 200. Write every number.", ",\"temperature\":0"),
    );
    try std.testing.expectEqual(std.http.Status.ok, reply.status);
    const parsed = try std.json.parseFromSliceLeaky(
        Completion,
        arena,
        reply.body,
        .{ .ignore_unknown_fields = true },
    );
    try std.testing.expectEqualStrings("length", parsed.choices[0].finish_reason);
    try std.testing.expectEqual(
        max_seq - parsed.usage.prompt_tokens + 1,
        parsed.usage.completion_tokens,
    );
}

test "openai loopback: a buffered prompt past the window gets 400 with both numbers" {
    var fx: Fixture = undefined;
    try fx.setUp();
    defer fx.tearDown();

    var arena_state = std.heap.ArenaAllocator.init(fx.gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    const reply = try post(&fx, arena, try chatBody(arena, long_message, ",\"max_tokens\":8"));
    try std.testing.expectEqual(std.http.Status.bad_request, reply.status);
    try std.testing.expectEqualStrings("application/json", reply.content_type);
    const parsed = try std.json.parseFromSliceLeaky(ErrorReply, arena, reply.body, .{});
    const e = parsed.@"error";
    try std.testing.expectEqual(@as(u16, 400), e.code);
    try std.testing.expectEqualStrings("exceed_context_size_error", e.type);
    try std.testing.expectEqual(max_seq, e.n_ctx);
    try std.testing.expect(e.n_prompt_tokens > e.n_ctx);
    const want = try std.fmt.allocPrint(
        arena,
        "the request of {d} tokens exceeds the context of {d} tokens",
        .{ e.n_prompt_tokens, e.n_ctx },
    );
    try std.testing.expectEqualStrings(want, e.message);
}

test "openai loopback: a streamed prompt past the window gets the same 400 and no event stream" {
    var fx: Fixture = undefined;
    try fx.setUp();
    defer fx.tearDown();

    var arena_state = std.heap.ArenaAllocator.init(fx.gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    const streamed = try post(
        &fx,
        arena,
        try chatBody(arena, long_message, ",\"max_tokens\":8,\"stream\":true"),
    );
    try std.testing.expectEqual(std.http.Status.bad_request, streamed.status);
    try std.testing.expectEqualStrings("application/json", streamed.content_type);
    try std.testing.expect(!std.mem.startsWith(u8, streamed.body, "data:"));
    const parsed = try std.json.parseFromSliceLeaky(ErrorReply, arena, streamed.body, .{});
    try std.testing.expectEqualStrings("exceed_context_size_error", parsed.@"error".type);
    try std.testing.expectEqual(max_seq, parsed.@"error".n_ctx);

    const buffered = try post(&fx, arena, try chatBody(arena, long_message, ",\"max_tokens\":8"));
    try std.testing.expectEqualStrings(buffered.body, streamed.body);
}

test "openai loopback: enable_thinking true without a thinking mode gets 400" {
    var fx: Fixture = undefined;
    try fx.setUp();
    defer fx.tearDown();

    var arena_state = std.heap.ArenaAllocator.init(fx.gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    const on = ",\"max_tokens\":8,\"chat_template_kwargs\":{\"enable_thinking\":true}";
    const buffered = try post(&fx, arena, try chatBody(arena, "Hello", on));
    try std.testing.expectEqual(std.http.Status.bad_request, buffered.status);
    try std.testing.expectEqualStrings("application/json", buffered.content_type);
    const e = (try std.json.parseFromSliceLeaky(
        PlainErrorReply,
        arena,
        buffered.body,
        .{},
    )).@"error";
    try std.testing.expectEqualStrings("invalid_request_error", e.type);
    try std.testing.expectEqualStrings(
        "enable_thinking true does not apply to this model, " ++
            "because its chat template has no thinking mode",
        e.message,
    );

    // The refusal comes before any session opens, so a streamed request gets
    // the same JSON body and no event stream.
    const streamed = try post(&fx, arena, try chatBody(arena, "Hello", on ++ ",\"stream\":true"));
    try std.testing.expectEqual(std.http.Status.bad_request, streamed.status);
    try std.testing.expectEqualStrings(buffered.body, streamed.body);
}

test "openai loopback: enable_thinking false or absent gives the plain reply" {
    var fx: Fixture = undefined;
    try fx.setUp();
    defer fx.tearDown();

    var arena_state = std.heap.ArenaAllocator.init(fx.gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    const off = ",\"chat_template_kwargs\":{\"enable_thinking\":false}";
    const greedy8 = ",\"max_tokens\":8,\"temperature\":0";
    const absent = try post(&fx, arena, try chatBody(arena, "Hello", greedy8));
    try std.testing.expectEqual(std.http.Status.ok, absent.status);
    const plain = try std.json.parseFromSliceLeaky(PlainCompletion, arena, absent.body, .{});
    var round_trip: std.Io.Writer.Allocating = .init(arena);
    try std.json.Stringify.value(plain, .{}, &round_trip.writer);
    try std.testing.expectEqualStrings(absent.body, round_trip.written());

    const switched_off = try post(&fx, arena, try chatBody(arena, "Hello", greedy8 ++ off));
    try std.testing.expectEqual(std.http.Status.ok, switched_off.status);
    try std.testing.expectEqualStrings(
        try withoutId(arena, absent.body),
        try withoutId(arena, switched_off.body),
    );

    const streamed = try post(
        &fx,
        arena,
        try chatBody(arena, "Hello", greedy8 ++ ",\"stream\":true"),
    );
    const streamed_off = try post(
        &fx,
        arena,
        try chatBody(arena, "Hello", greedy8 ++ ",\"stream\":true" ++ off),
    );
    try std.testing.expectEqual(std.http.Status.ok, streamed.status);
    try std.testing.expect(std.mem.find(u8, streamed.body, "reasoning_content") == null);
    try std.testing.expectEqualStrings(
        try withoutId(arena, streamed.body),
        try withoutId(arena, streamed_off.body),
    );
}

/// The reply text of one buffered request that must succeed.
fn replyText(
    fx: *const Fixture,
    arena: std.mem.Allocator,
    prompt: []const u8,
    extra: []const u8,
) ![]const u8 {
    const reply = try post(fx, arena, try chatBody(arena, prompt, extra));
    try std.testing.expectEqual(std.http.Status.ok, reply.status);
    const parsed = try std.json.parseFromSliceLeaky(
        Completion,
        arena,
        reply.body,
        .{ .ignore_unknown_fields = true },
    );
    return parsed.choices[0].message.content;
}

const open_prompt = "Write a short story about a sea captain.";

test "openai loopback: seed omitted draws, seed fixed reproduces, temperature 0 pins greedy" {
    var fx: Fixture = undefined;
    try fx.setUp();
    defer fx.tearDown();

    var arena_state = std.heap.ArenaAllocator.init(fx.gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    // The card row samples at temperature 1.5 over 32 tokens. Two equal
    // replies from two drawn seeds therefore do not occur in practice.
    const drawn_a = try replyText(&fx, arena, open_prompt, ",\"max_tokens\":32");
    const drawn_b = try replyText(&fx, arena, open_prompt, ",\"max_tokens\":32");
    try std.testing.expect(!std.mem.eql(u8, drawn_a, drawn_b));

    const seeded_a = try replyText(&fx, arena, open_prompt, ",\"max_tokens\":32,\"seed\":7");
    const seeded_b = try replyText(&fx, arena, open_prompt, ",\"max_tokens\":32,\"seed\":7");
    try std.testing.expectEqualStrings(seeded_a, seeded_b);

    // A request field beats the card row, so temperature 0 is greedy and
    // ignores the drawn seed.
    const greedy_a = try replyText(&fx, arena, open_prompt, ",\"max_tokens\":32,\"temperature\":0");
    const greedy_b = try replyText(&fx, arena, open_prompt, ",\"max_tokens\":32,\"temperature\":0");
    try std.testing.expectEqualStrings(greedy_a, greedy_b);
}

test "openai loopback: the greedy defaults give one reply to equal requests without a seed" {
    var fx: Fixture = undefined;
    try fx.setUpWith(.init(.greedy, card_row, card_row));
    defer fx.tearDown();

    var arena_state = std.heap.ArenaAllocator.init(fx.gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    const a = try replyText(&fx, arena, open_prompt, ",\"max_tokens\":32");
    const b = try replyText(&fx, arena, open_prompt, ",\"max_tokens\":32");
    try std.testing.expectEqualStrings(a, b);
    const pinned = try replyText(&fx, arena, open_prompt, ",\"max_tokens\":32,\"temperature\":0");
    try std.testing.expectEqualStrings(a, pinned);
}

test "openai loopback: a refused sampling or budget field gets a 400 that names it" {
    var fx: Fixture = undefined;
    try fx.setUp();
    defer fx.tearDown();

    var arena_state = std.heap.ArenaAllocator.init(fx.gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    const msg_frequency = "frequency_penalty is not supported. The only accepted value is 0";
    const msg_repetition = "repetition_penalty must be more than 0 and at most 2";
    const msg_budget_range = "thinking_token_budget must be -1, 0, or a count from 1 to 262144";
    const msg_budget_no_think = "thinking_token_budget above 0 does not apply to this model, " ++
        "because its chat template has no thinking mode";
    const Case = struct { extra: []const u8, message: []const u8 };
    const cases = [_]Case{
        .{ .extra = ",\"frequency_penalty\":0.5", .message = msg_frequency },
        .{ .extra = ",\"top_k\":257", .message = "top_k must be -1 or in the range 0 to 256" },
        .{ .extra = ",\"min_p\":2", .message = "min_p must be in the range 0 to 1" },
        .{ .extra = ",\"repetition_penalty\":0", .message = msg_repetition },
        .{ .extra = ",\"thinking_token_budget\":-5", .message = msg_budget_range },
        .{ .extra = ",\"thinking_token_budget\":16", .message = msg_budget_no_think },
    };
    for (cases) |c| {
        const reply = try post(
            &fx,
            arena,
            try chatBody(
                arena,
                "Hello",
                try std.mem.concat(arena, u8, &.{ ",\"max_tokens\":4", c.extra }),
            ),
        );
        try std.testing.expectEqual(std.http.Status.bad_request, reply.status);
        const e = (try std.json.parseFromSliceLeaky(
            PlainErrorReply,
            arena,
            reply.body,
            .{},
        )).@"error";
        try std.testing.expectEqualStrings("invalid_request_error", e.type);
        try std.testing.expectEqualStrings(c.message, e.message);
    }

    // The closed modes pass on a model with no think tags: budget 0 and -1,
    // and budget 0 against enable_thinking true.
    const greedy = ",\"max_tokens\":8,\"temperature\":0";
    const plain = try replyText(&fx, arena, "Hello", greedy);
    try std.testing.expectEqualStrings(
        plain,
        try replyText(&fx, arena, "Hello", greedy ++ ",\"thinking_token_budget\":0"),
    );
    try std.testing.expectEqualStrings(
        plain,
        try replyText(&fx, arena, "Hello", greedy ++ ",\"thinking_token_budget\":-1"),
    );
    const off_wins = greedy ++
        ",\"thinking_token_budget\":0,\"chat_template_kwargs\":{\"enable_thinking\":true}";
    try std.testing.expectEqualStrings(plain, try replyText(&fx, arena, "Hello", off_wins));
}

test "openai loopback: min_p and repetition_penalty reach the sampler" {
    var fx: Fixture = undefined;
    try fx.setUp();
    defer fx.tearDown();

    var arena_state = std.heap.ArenaAllocator.init(fx.gpa);
    defer arena_state.deinit();

    const arena = arena_state.allocator();

    const prompt = "Count from 1 to 30. Write every number.";
    const greedy = ",\"max_tokens\":48,\"temperature\":0";
    const plain = try replyText(&fx, arena, prompt, greedy);
    // A greedy pick keeps the top id, and min_p never drops the top id.
    try std.testing.expectEqualStrings(
        plain,
        try replyText(&fx, arena, prompt, greedy ++ ",\"min_p\":0.5"),
    );
    // The penalty divides the logits of every id in the context, so the
    // greedy count moves off its plain path.
    const penalized = try replyText(&fx, arena, prompt, greedy ++ ",\"repetition_penalty\":2");
    try std.testing.expect(!std.mem.eql(u8, plain, penalized));
}
