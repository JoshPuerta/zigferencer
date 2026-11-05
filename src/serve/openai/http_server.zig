//! The OpenAI HTTP frontend as a serve-engine adapter. `serve/engine.zig` owns the batched engine.
//! This file is the transport: one task per connection over `std.http.Server`, mapping OpenAI JSON
//! to and from the engine's `Request` and `OutEvent` values. The format layer is pure (`types.zig`,
//! `map.zig`). This file is the imperative shell.
//!
//! Request flow. A chat request becomes an ephemeral session: `session_open("")`,
//! `generate(rendered)`, `session_close`. It reaches the shared scheduler, which batches it with
//! every other in-flight request.
//!
//! Tasks. Each connection runs one task, because request and response are half-duplex here. The
//! binary transport multiplexes over two tasks instead. The one task drains the `out_q` and writes
//! the response. A slow client parks the session engine-side, and the task wakes the engine after
//! it drains. The binary writer uses the same backpressure.
//!
//! Streaming. The SSE head goes out with the first piece or at the finish. So an engine error
//! before that point can still answer with an HTTP error status. A JSON string must be valid
//! UTF-8, so `pending` holds the tail bytes of a split codepoint until the next piece of the same
//! field completes it. An error after the head becomes one error event and then the `[DONE]`
//! event, the llama.cpp shape. It never reads as a finish.
//!
//! Teardown. The task can block in `receiveHead` or while it drains its out-queue. `beginClose`
//! closes the queue and shuts the socket down to wake a blocked read. It shuts down only after it
//! claims the fd. If the task already exited and claimed it, the task closes the fd itself, and a
//! shutdown after that close is EBADF.

const std = @import("std");
const Io = std.Io;
const net = Io.net;
const Allocator = std.mem.Allocator;
const http = std.http;
const span = @import("../../core/span.zig");

const engine_mod = @import("../engine.zig");
const protocol = @import("../protocol.zig");
const chat = @import("../../core/chat.zig");
const limits = @import("../../core/limits.zig");
const types = @import("types.zig");
const map = @import("map.zig");
const RawCodec = @import("raw_codec.zig").RawCodec;

const log = std.log.scoped(.serve_openai);

// -- constants --------------------------------------------------------------

const json_opts: std.json.Stringify.Options = .{ .emit_null_optional_fields = false };
const json_headers = [_]http.Header{.{ .name = "content-type", .value = "application/json" }};

const sse_headers = [_]http.Header{
    .{ .name = "content-type", .value = "text/event-stream" },
    .{ .name = "cache-control", .value = "no-cache" },
};

/// The body buffer of a streamed reply. A chunk waits here until the sink flushes it.
const sse_body_bytes = 4096;

// -- types ------------------------------------------------------------------

/// One turn as the request resolved it.
const TurnInput = struct {
    rendered: []const u8,
    max_tokens: u32,
    /// The request's thinking mode leaves the block open at the end of the
    /// rendered prompt.
    think_open: bool,
    /// Null takes the server budget.
    think_budget: ?u32,
};

/// One task per connection. Request and response are half-duplex, so a single
/// task reads, drives the engine, and writes.
const HttpAdapter = struct {
    pub const ConnExt = struct {
        task_future: Io.Future(void) = undefined,
    };

    pub fn onAccept(engine: anytype, conn: anytype) bool {
        const Tasks = HttpTasks(@TypeOf(engine.*));
        conn.tasks_live.store(1, .release);
        conn.ext.task_future = engine.io.concurrent(Tasks.conn, .{ engine, conn }) catch {
            conn.stream.close(engine.io);
            return false;
        };
        return true;
    }

    /// Unblock the task: close the queue, and shut the socket down. See "Teardown" in the header.
    pub fn beginClose(engine: anytype, conn: anytype) void {
        engine.closeOutput(conn);
        if (engine.claimStreamForTeardown(conn)) {
            conn.stream.shutdown(engine.io, .both) catch {};
        }
    }

    pub fn awaitConn(engine: anytype, conn: anytype) void {
        conn.ext.task_future.await(engine.io);
    }

    pub fn refuse(io: Io, stream: *net.Stream) void {
        var buf: [128]u8 = undefined;
        var w = stream.writer(io, &buf);
        w.interface.writeAll(
            "HTTP/1.1 503 Service Unavailable\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
        ) catch {};
        w.interface.flush() catch {};
        stream.close(io);
    }
};

/// The response sink of a buffered chat request. It grows one arena-owned
/// string per reply field, and the JSON response carries each one whole.
const BufSink = struct {
    arena: Allocator,
    content: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,

    fn onPiece(self: *BufSink, field: map.Field, piece: []const u8) !void {
        const buf = switch (field) {
            .content => &self.content,
            .reasoning => &self.reasoning,
        };

        try buf.appendSlice(self.arena, piece);
    }
};

/// The response sink of a streamed chat request. See "Streaming" in the header.
const SseSink = struct {
    request: *http.Server.Request,
    id: []const u8,
    model: []const u8,
    body_buf: [sse_body_bytes]u8 = undefined,
    body: ?http.BodyWriter = null,
    pending: [4]u8 = undefined,
    pending_len: usize = 0,
    pending_field: map.Field = .content,

    fn onPiece(self: *SseSink, field: map.Field, piece: []const u8) !void {
        // A think tag ends the field, so a held tail can never complete. Drop it.
        if (field != self.pending_field) {
            self.pending_len = 0;
            self.pending_field = field;
        }

        // Join the held tail and the piece, and keep any new split codepoint.
        var buf: [4 + limits.max_token_text_bytes]u8 = undefined;
        @memcpy(buf[0..self.pending_len], self.pending[0..self.pending_len]);
        @memcpy(buf[self.pending_len..][0..piece.len], piece);
        const all = buf[0 .. self.pending_len + piece.len];
        const emit_len = map.completeUtf8Len(all);
        const rest = all[emit_len..];
        @memcpy(self.pending[0..rest.len], rest);
        self.pending_len = rest.len;
        if (emit_len == 0) return;

        // Send the complete part as one chunk.
        const body = try self.open();
        try writeChunk(body, self.id, self.model, fieldDelta(field, all[0..emit_len]), null);
        // `BodyWriter.flush` sends only the connection buffer, so drain `body_buf` first.
        try body.writer.flush();
        try body.flush();
    }

    /// Write the SSE head and the role chunk once. The role chunk stays
    /// unflushed, so it reaches the client together with the first token.
    fn open(self: *SseSink) !*http.BodyWriter {
        if (self.body) |*body| return body;

        self.body = try self.request.respondStreaming(
            &self.body_buf,
            .{ .respond_options = .{ .extra_headers = &sse_headers } },
        );
        const body = &self.body.?;
        try writeChunk(body, self.id, self.model, .{ .role = "assistant" }, null);
        return body;
    }
};

/// The serve engine with the OpenAI HTTP transport. The `RawCodec` carries the model metadata.
pub fn Server(comptime Engine: type) type {
    return engine_mod.ServeEngine(Engine, RawCodec, HttpAdapter);
}

fn HttpTasks(comptime Eng: type) type {
    return struct {
        const Conn = Eng.Conn;

        fn conn(engine: *Eng, c: *Conn) void {
            span.nameThread("zf-http");
            serveConn(engine, c);
            // Tell the engine the connection dropped, so it sweeps any open session.
            engine.submit(
                .{ .conn = c.id, .request_id = 0, .session_id = 0, .body = .conn_closed },
            ) catch {};
            engine.taskExit(c);
        }

        fn serveConn(engine: *Eng, c: *Conn) void {
            var read_buf: [limits.openai_max_header_bytes]u8 = undefined;
            var write_buf: [16 * 1024]u8 = undefined;
            var sr = c.stream.reader(engine.io, &read_buf);
            var sw = c.stream.writer(engine.io, &write_buf);
            var server = http.Server.init(&sr.interface, &sw.interface);
            while (true) {
                var request = server.receiveHead() catch return;
                handleRequest(engine, c, &request) catch |err| {
                    log.debug("conn {d}: request failed: {t}", .{ c.id, err });
                    return;
                };
            }
        }

        fn handleRequest(engine: *Eng, c: *Conn, request: *http.Server.Request) !void {
            var arena_state = std.heap.ArenaAllocator.init(engine.gpa);
            defer arena_state.deinit();

            const arena = arena_state.allocator();

            const method = request.head.method;
            const target = stripQuery(request.head.target);
            log.debug("http: {t} {s}", .{ method, target });

            if (method == .GET and std.mem.eql(u8, target, "/v1/models")) {
                return respondModels(engine, request);
            }
            if (method == .GET and (std.mem.eql(u8, target, "/healthz") or std.mem.eql(
                u8,
                target,
                "/v1/health",
            ))) {
                return request.respond("{\"status\":\"ok\"}", .{ .extra_headers = &json_headers });
            }
            if (method == .POST and std.mem.eql(u8, target, "/v1/chat/completions")) {
                return handleChat(engine, c, request, arena);
            }

            return respondError(
                request,
                .{ .status = 404, .type = "invalid_request_error", .message = "unknown route" },
            );
        }

        fn handleChat(
            engine: *Eng,
            c: *Conn,
            request: *http.Server.Request,
            arena: Allocator,
        ) !void {
            const sp = span.begin(.http_chat);
            defer sp.end();

            // Read the request body under the size cap.
            const codec = engine.codec;
            const clen = request.head.content_length orelse 0;
            if (clen > limits.openai_max_request_bytes) {
                return respondError(request, .{
                    .status = 413,
                    .type = "invalid_request_error",
                    .message = "request body too large",
                });
            }

            var body_rbuf: [4096]u8 = undefined;
            const reader = request.readerExpectContinue(&body_rbuf) catch {
                return respondError(request, .{
                    .status = 400,
                    .type = "invalid_request_error",
                    .message = "bad request body",
                });
            };
            const body = try arena.alloc(u8, @intCast(clen));
            reader.readSliceAll(body) catch return error.BodyReadFailed;

            // Parse the request and refuse a model that this server does not serve.
            var diag: map.Diag = .{};
            const req = map.parseChatRequest(
                arena,
                body,
                codec.default_max_tokens,
                codec.max_seq,
                &diag,
            ) catch |e| {
                return respondError(request, map.reqErrorToHttp(e, diag));
            };
            if (req.model.len > 0 and !std.mem.eql(u8, req.model, codec.model_id)) {
                return respondError(request, .{
                    .status = 404,
                    .type = "model_not_found",
                    .message = "the requested model is not served here. " ++
                        "GET /v1/models lists the served id",
                });
            }

            // Resolve the thinking mode, the rendered prompt, and the sampling.
            const mode = map.resolveTurn(
                codec.kind,
                req.thinking,
                req.think_budget,
                codec.think_forced_close,
            ) catch |e| {
                return respondError(request, map.reqErrorToHttp(e, .{}));
            };
            const input: TurnInput = .{
                .rendered = try chat.wrapMessagesAlloc(arena, mode.kind, req.messages),
                .max_tokens = req.max_tokens,
                .think_open = chat.thinkOpen(mode.kind),
                .think_budget = mode.think_budget,
            };
            const seed = req.sampling.seed orelse drawSeed(engine.io) orelse {
                return respondError(request, .{
                    .status = 500,
                    .type = "server_error",
                    .message = "the server has no random source for a seed. " ++
                        "Send a seed field",
                });
            };
            const sampling = map.resolveSampling(
                req.sampling,
                codec.defaults.row(input.think_open),
                seed,
            );
            log.debug(
                "chat: {d} B body, stream={}, max_tokens {d}, think open {}, temperature " ++
                    "{d}, seed {d}",
                .{
                    body.len,
                    req.stream,
                    req.max_tokens,
                    input.think_open,
                    sampling.temperature,
                    sampling.seed,
                },
            );

            // Open the session. Its open seed starts the stream, so the turn needs no reseed.
            const sid = openSession(engine, c, sampling) orelse {
                return respondError(
                    request,
                    .{ .status = 503, .type = "server_error", .message = "no free session slot" },
                );
            };
            defer closeSession(engine, c, sid);

            // Run the turn, streamed or buffered.
            var id_buf: [32]u8 = undefined;
            const id_str = std.fmt.bufPrint(&id_buf, "chatcmpl-{d}", .{sid}) catch "chatcmpl-0";

            if (req.stream) {
                try streamChat(engine, c, request, arena, sid, input, id_str);
            } else {
                try bufferChat(engine, c, request, arena, sid, input, id_str);
            }
        }

        /// Open an ephemeral session with an empty system prompt, so no prefill runs. Return its
        /// protocol id, or null when the pool is exhausted or the connection dropped.
        fn openSession(engine: *Eng, c: *Conn, sampling: protocol.Sampling) ?u32 {
            const system_prompt = engine.gpa.dupe(u8, "") catch return null;
            engine.submit(.{ .conn = c.id, .request_id = 1, .session_id = 0, .body = .{
                .session_open = .{ .sampling = sampling, .system_prompt = system_prompt },
            } }) catch {
                engine.gpa.free(system_prompt);
                return null;
            };
            while (true) {
                const ev = c.out_q.getOne(engine.io) catch return null;
                switch (ev.body) {
                    .session_opened => {
                        log.debug("session open: {d}", .{ev.session_id});
                        return ev.session_id;
                    },
                    .err => return null,
                    // Only the binary reader task produces these, in its own queue.
                    .zmtp_pong, .zmtp_error => unreachable,
                    else => {},
                }

                if (c.parked_hint.swap(false, .acq_rel)) {
                    engine.wake(c.id);
                }
            }
        }

        fn closeSession(engine: *Eng, c: *Conn, sid: u32) void {
            log.debug("session close: {d}", .{sid});
            engine.submit(
                .{ .conn = c.id, .request_id = 3, .session_id = sid, .body = .session_close },
            ) catch {};
        }

        const Turn = union(enum) {
            done: struct { n: u32, reasoning: u32, finish: []const u8 },
            failed: map.HttpError,
        };

        /// Submit the generate turn and drain the out-queue, handing each piece to `sink` with its
        /// reply field. The `OutEvent` already holds the token text. It wakes the engine after a
        /// drain, so a parked session resumes.
        fn runTurn(
            engine: *Eng,
            c: *Conn,
            arena: Allocator,
            sid: u32,
            input: TurnInput,
            sink: anytype,
        ) !Turn {
            // Submit the generate request.
            const text = try engine.gpa.dupe(u8, input.rendered);
            engine.submit(.{ .conn = c.id, .request_id = 2, .session_id = sid, .body = .{
                .generate = .{
                    .max_tokens = input.max_tokens,
                    .sampling = null,
                    .text = text,
                    .think_open = input.think_open,
                    .think_budget = input.think_budget,
                },
            } }) catch {
                engine.gpa.free(text);
                return error.SubmitFailed;
            };

            // Drain the events until the turn ends.
            var split: map.ReasoningSplit = .init(engine.codec.think_ids, input.think_open);
            while (true) {
                const ev = c.out_q.getOne(engine.io) catch return error.ConnClosed;
                switch (ev.body) {
                    // The scheduler emits the terminal EOS token. OpenAI content omits it.
                    .token => |t| if (t.token_id != engine.codec.eos_id) {
                        if (split.route(t.token_id, t.text[0..t.text_len])) |p| {
                            try sink.onPiece(p.field, p.text);
                        }
                    },
                    .done => |d| {
                        log.debug(
                            "turn: session {d}, {d} tokens, finish {t}",
                            .{ sid, d.gen_tokens, d.finish_reason },
                        );
                        return .{
                            .done = .{
                                .n = d.gen_tokens,
                                .reasoning = d.reasoning_tokens,
                                .finish = finishStr(d.finish_reason),
                            },
                        };
                    },
                    .err => |e| return .{
                        .failed = try turnError(engine, arena, e.code, input.rendered),
                    },
                    // Only the binary reader task produces these, in its own queue.
                    .zmtp_pong, .zmtp_error => unreachable,
                    else => {}, // session_closed from a prior turn, etc.
                }

                if (c.parked_hint.swap(false, .acq_rel)) {
                    engine.wake(c.id);
                }
            }
        }

        fn bufferChat(
            engine: *Eng,
            c: *Conn,
            request: *http.Server.Request,
            arena: Allocator,
            sid: u32,
            input: TurnInput,
            id_str: []const u8,
        ) !void {
            var sink: BufSink = .{ .arena = arena };
            const turn = switch (try runTurn(engine, c, arena, sid, input, &sink)) {
                .done => |d| d,
                .failed => |he| return respondError(request, he),
            };

            const prompt_tokens = countPrompt(engine, input.rendered);
            const message: types.RespMessage = .{
                .content = sink.content.items,
                .reasoning_content = if (sink.reasoning.items.len > 0)
                    sink.reasoning.items
                else
                    null,
            };
            const resp: types.ChatResponse = .{
                .id = id_str,
                .model = engine.codec.model_id,
                .choices = &.{.{ .message = message, .finish_reason = turn.finish }},
                .usage = .{
                    .prompt_tokens = prompt_tokens,
                    .completion_tokens = turn.n,
                    .total_tokens = prompt_tokens + turn.n,
                    .completion_tokens_details = if (turn.reasoning > 0) .{
                        .reasoning_tokens = turn.reasoning,
                    } else null,
                },
            };
            var sbuf: [4096]u8 = undefined;
            var rbody = try request.respondStreaming(
                &sbuf,
                .{ .respond_options = .{ .extra_headers = &json_headers } },
            );
            try std.json.Stringify.value(resp, json_opts, &rbody.writer);
            try rbody.end();
        }

        /// Stream a chat reply. An error after the head becomes an error event.
        fn streamChat(
            engine: *Eng,
            c: *Conn,
            request: *http.Server.Request,
            arena: Allocator,
            sid: u32,
            input: TurnInput,
            id_str: []const u8,
        ) !void {
            var sink: SseSink = .{
                .request = request,
                .id = id_str,
                .model = engine.codec.model_id,
            };
            switch (try runTurn(engine, c, arena, sid, input, &sink)) {
                .done => |turn| {
                    const body = try sink.open();
                    try writeChunk(body, id_str, engine.codec.model_id, .{}, turn.finish);
                    try body.writer.writeAll("data: [DONE]\n\n");
                    try body.end();
                },
                .failed => |he| {
                    const body = if (sink.body) |*b| b else return respondError(request, he);
                    try writeErrorEvent(&body.writer, he);
                    try body.end();
                },
            }
        }

        fn respondModels(engine: *Eng, request: *http.Server.Request) !void {
            const data = [_]types.ModelEntry{.{ .id = engine.codec.model_id }};
            const list: types.ModelsList = .{ .data = &data };
            var sbuf: [1024]u8 = undefined;
            var rbody = try request.respondStreaming(
                &sbuf,
                .{ .respond_options = .{ .extra_headers = &json_headers } },
            );
            try std.json.Stringify.value(list, json_opts, &rbody.writer);
            try rbody.end();
        }

        /// Prompt-token count for `usage`: re-tokenize the rendered prompt. This costs little next
        /// to a decode, and it keeps the count exact without a path back through the wire.
        fn countPrompt(engine: *Eng, rendered: []const u8) u32 {
            const ids = engine.codec.turnTokens(
                engine.gpa,
                .open_first,
                rendered,
                false,
            ) catch return 0;
            defer engine.gpa.free(ids);

            return @intCast(ids.len);
        }

        /// The session opens empty, so its window is `max_seq` and the
        /// refused turn is the whole rendered prompt.
        fn turnError(
            engine: *Eng,
            arena: Allocator,
            code: protocol.ErrorCode,
            rendered: []const u8,
        ) !map.HttpError {
            return switch (code) {
                .turn_too_long => try map.exceedContextError(
                    arena,
                    countPrompt(engine, rendered),
                    engine.codec.max_seq,
                ),
                .sessions_exhausted, .session_busy => .{
                    .status = 503,
                    .type = "server_error",
                    .message = "server busy",
                },
                else => .{ .status = 500, .type = "server_error", .message = "internal error" },
            };
        }
    };
}

// -- helpers ----------------------------------------------------------------

/// Draw a seed from the secure source, the vLLM rule, so equal sampled requests differ.
/// Null refuses the request. The server never falls back to a fixed seed.
fn drawSeed(io: Io) ?u64 {
    var bytes: [8]u8 = undefined;
    io.randomSecure(&bytes) catch return null;
    return std.mem.readInt(u64, &bytes, .little);
}

fn writeErrorEvent(w: *Io.Writer, e: map.HttpError) Io.Writer.Error!void {
    try w.writeAll("data: ");
    try std.json.Stringify.value(map.errorObject(e), json_opts, w);
    try w.writeAll("\n\ndata: [DONE]\n\n");
}

fn writeChunk(
    body: *http.BodyWriter,
    id: []const u8,
    model: []const u8,
    delta: types.Delta,
    finish: ?[]const u8,
) !void {
    const chunk: types.ChatChunk = .{
        .id = id,
        .model = model,
        .choices = &.{.{ .delta = delta, .finish_reason = finish }},
    };
    try body.writer.writeAll("data: ");
    try std.json.Stringify.value(chunk, json_opts, &body.writer);
    try body.writer.writeAll("\n\n");
}

fn fieldDelta(field: map.Field, text: []const u8) types.Delta {
    return switch (field) {
        .reasoning => .{ .reasoning_content = text },
        .content => .{ .content = text },
    };
}

fn finishStr(r: protocol.FinishReason) []const u8 {
    return switch (r) {
        .eos => "stop",
        .max_tokens, .context_full => "length",
        .cancelled => "stop",
    };
}

fn stripQuery(target: []const u8) []const u8 {
    return if (std.mem.findScalar(u8, target, '?')) |i| target[0..i] else target;
}

fn respondError(request: *http.Server.Request, e: map.HttpError) !void {
    log.debug("http error {d}: {s}", .{ e.status, e.message });
    var sbuf: [1024]u8 = undefined;
    var rbody = try request.respondStreaming(&sbuf, .{ .respond_options = .{
        .status = statusEnum(e.status),
        .keep_alive = false,
        .extra_headers = &json_headers,
    } });
    try std.json.Stringify.value(map.errorObject(e), json_opts, &rbody.writer);
    try rbody.end();
}

fn statusEnum(code: u16) http.Status {
    return switch (code) {
        400 => .bad_request,
        404 => .not_found,
        413 => .payload_too_large,
        500 => .internal_server_error,
        503 => .service_unavailable,
        else => .bad_request,
    };
}

// -- tests ------------------------------------------------------------------

test "writeErrorEvent: one error event in the llama.cpp shape, then the end event" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();

    const overflow = try map.exceedContextError(arena_state.allocator(), 130, 128);

    var buf: [512]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try writeErrorEvent(&w, overflow);
    try std.testing.expectEqualStrings(
        \\data: {"error":{"code":400,"message":"the request of 130 tokens exceeds the context of 128 tokens","type":"exceed_context_size_error","n_prompt_tokens":130,"n_ctx":128}}
        \\
        \\data: [DONE]
        \\
        \\
    ,
        w.buffered(),
    );
}

test "the reply fields: reasoning_content sits next to content and vanishes when null" {
    var buf: [512]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    const split: types.RespMessage = .{ .content = "A.", .reasoning_content = "R." };
    try std.json.Stringify.value(split, json_opts, &w);
    try std.testing.expectEqualStrings(
        "{\"role\":\"assistant\",\"content\":\"A.\",\"reasoning_content\":\"R.\"}",
        w.buffered(),
    );

    w = .fixed(&buf);
    const plain: types.RespMessage = .{ .content = "A." };
    try std.json.Stringify.value(plain, json_opts, &w);
    try std.testing.expectEqualStrings("{\"role\":\"assistant\",\"content\":\"A.\"}", w.buffered());

    w = .fixed(&buf);
    try std.json.Stringify.value(fieldDelta(.reasoning, "r"), json_opts, &w);
    try std.testing.expectEqualStrings("{\"reasoning_content\":\"r\"}", w.buffered());

    w = .fixed(&buf);
    try std.json.Stringify.value(fieldDelta(.content, "c"), json_opts, &w);
    try std.testing.expectEqualStrings("{\"content\":\"c\"}", w.buffered());
}
