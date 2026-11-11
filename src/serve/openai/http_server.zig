//! The OpenAI HTTP frontend as a serve-engine adapter. `serve/engine.zig` owns the batched engine.
//! This file is the transport: one task per connection over `std.http.Server`, mapping OpenAI JSON
//! to and from the engine's `Request` and `OutEvent` values. The format layer is pure (`types.zig`,
//! `map.zig`). This file is the imperative shell.
//!
//! Comptime. Generic over the engine, so the adapter builds against any engine type.
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
//! Stop. A request with stop strings passes the reply text through `StopMatcher` first. The
//! matcher holds back a tail that can still start a stop string, and the UTF-8 hold acts on what it
//! releases. So no emitted byte belongs to a later match, and no chunk ends inside a codepoint. A
//! match cancels the turn, and the task drains the events up to its `done`.
//!
//! Body. `body.zig` decides the body form from the raw head before any body byte moves, and reads
//! the body. The std body readers never run, so the task marks the body read for the std server
//! after it reads one. The task clears the std `expect` field, so std never sends an interim
//! reply before a refusal, and the task sends `100 Continue` itself before an accepted body.
//!
//! Close. A close over unread input sends a reset, and a reset can drop the last reply. After a
//! refusal the task shuts its send side and drops input until the end of the stream,
//! `limits.close_drain_bytes`, or `limits.close_linger_ms`. After any other end it drops the input
//! that has arrived, and the close in `taskExit` sends the end of stream. The drain lowers the
//! chance of a reset, and it does not remove it.
//!
//! Routes. Each route takes one method set. Another method on a known route gets 405 with
//! `allow`, an unknown route gets 404, and a method that std does not know gets 501. HEAD on a
//! GET route answers as GET, and std leaves out the body.
//!
//! HTTP/1.0. A connection ends after one reply, and no interim reply goes out. A whole reply
//! carries its length, and a stream ends at the close, because HTTP/1.0 has no chunked coding.
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

const body_mod = @import("body.zig");
const engine_mod = @import("../engine.zig");
const timed_stream = @import("../timed_stream.zig");
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

const overhead_message = std.fmt.comptimePrint(
    "the chunk framing of the request body passes its bound of {d} bytes. Send larger chunks",
    .{limits.openai_max_chunk_overhead_bytes},
);

// -- types ------------------------------------------------------------------

/// One turn as the request resolved it.
const TurnInput = struct {
    rendered: []const u8,
    /// The byte ranges of `rendered` that came from the request, in the request arena.
    plain: []const chat.PlainRange,
    max_tokens: u32,
    /// The request's thinking mode leaves the block open at the end of the
    /// rendered prompt.
    think_open: bool,
    /// Null takes the server budget.
    think_budget: ?u32,
    /// The stop strings of the request, in the request arena.
    stop: []const []const u8,
};

/// What a reply leaves the connection in. `close` ends it after a complete reply. `refused`
/// ends it after an error reply, and the client can still be sending its request.
const Verdict = enum { keep, close, refused };

/// How the request loop of a connection ended.
const End = enum { ended, refused };

/// The routes of the server. Each route takes one method set.
const Route = enum {
    models,
    health,
    chat,

    fn of(target: []const u8) ?Route {
        if (std.mem.eql(u8, target, "/v1/models")) return .models;
        if (std.mem.eql(u8, target, "/healthz")) return .health;
        if (std.mem.eql(u8, target, "/v1/health")) return .health;
        if (std.mem.eql(u8, target, "/v1/chat/completions")) return .chat;
        return null;
    }

    fn allows(route: Route, method: http.Method) bool {
        return switch (route) {
            .models, .health => method == .GET or method == .HEAD,
            .chat => method == .POST,
        };
    }

    /// The `allow` value of a 405 reply.
    fn allowed(route: Route) []const u8 {
        return switch (route) {
            .models, .health => "GET, HEAD",
            .chat => "POST",
        };
    }
};

const ReceiveError = http.Reader.HeadError || error{ HeadInvalid, MethodUnknown };

/// The body of a request, or the refusal to send in its place.
const BodyRead = union(enum) {
    bytes: []const u8,
    refused: map.HttpError,
};

/// One task per connection. Request and response are half-duplex, so a single
/// task reads, drives the engine, and writes.
const HttpAdapter = struct {
    pub const ConnExt = struct {
        task_future: Io.Future(void) = undefined,
    };

    pub const conn_tasks: u8 = 1;

    pub fn onAccept(engine: anytype, conn: anytype) bool {
        const Tasks = HttpTasks(@TypeOf(engine.*));
        conn.ext.task_future = engine.io.concurrent(Tasks.conn, .{ engine, conn }) catch {
            conn.stream.close(engine.io);
            return false;
        };
        return true;
    }

    /// Unblock the task: close the queue, and shut the socket down.
    pub fn beginClose(engine: anytype, conn: anytype) void {
        engine.closeOutput(conn);
        if (engine.claimStreamForTeardown(conn)) {
            conn.stream.shutdown(engine.io, .both) catch {};
        }
    }

    pub fn awaitConn(engine: anytype, conn: anytype) void {
        conn.ext.task_future.await(engine.io);
    }

    /// One send that does not wait, so a client that does not read cannot stall the acceptor.
    pub fn refuse(io: Io, stream: *net.Stream) void {
        timed_stream.sendOnce(
            io,
            stream.*,
            "HTTP/1.1 503 Service Unavailable\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
        );
        timed_stream.closeRefused(io, stream);
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

/// The response sink of a streamed chat request.
const SseSink = struct {
    request: *http.Server.Request,
    id: []const u8,
    model: []const u8,
    /// An HTTP/1.0 client reads no chunked coding, so its stream ends at the close.
    close_delimited: bool,
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
        // A piece is one token text, or a matcher release that adds the held stop prefix.
        var buf: [4 + limits.max_token_text_bytes + limits.openai_max_stop_bytes]u8 = undefined;
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

        assertBodyRead(self.request);
        const options: http.Server.Request.RespondOptions = if (self.close_delimited) .{
            .keep_alive = false,
            .transfer_encoding = .none,
            .extra_headers = &sse_headers,
        } else .{ .extra_headers = &sse_headers };
        self.body = try self.request.respondStreaming(
            &self.body_buf,
            .{ .respond_options = options },
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
            engine.taskExit(c, .reader);
        }

        fn serveConn(engine: *Eng, c: *Conn) void {
            var read_buf: [limits.openai_max_header_bytes]u8 = undefined;
            var write_buf: [16 * 1024]u8 = undefined;
            const timeout_ms = engine.io_timeout_ms;
            var sr: timed_stream.Reader = .init(c.stream, engine.io, &read_buf, timeout_ms);
            var sw: timed_stream.Writer = .init(c.stream, engine.io, &write_buf, timeout_ms);
            defer if (sr.timed_out or sw.timed_out) {
                log.debug("conn {d}: closed at the io deadline", .{engine.connId(c)});
            };

            var server = http.Server.init(&sr.interface, &sw.interface);
            // After a refusal the client can still be sending, so the task drops its input for a
            // short wait before the close. After any other end, only a pipelined request can be
            // unread. That end sends no shutdown, so the end of stream comes with the close in
            // `taskExit`, after the slot leaves `live`, and a client can connect again at once.
            switch (serveRequests(engine, c, &server)) {
                .refused => {
                    const wait_ms = if (timeout_ms > 0)
                        @min(timeout_ms, limits.close_linger_ms)
                    else
                        limits.close_linger_ms;
                    timed_stream.lingerOutput(engine.io, c.stream, wait_ms);
                },
                .ended => timed_stream.dropArrived(c.stream),
            }
        }

        fn serveRequests(engine: *Eng, c: *Conn, server: *http.Server) End {
            while (true) {
                var request = receiveRequest(server) catch |err| {
                    const refusal: map.HttpError = switch (err) {
                        error.HeadInvalid => .{
                            .status = 400,
                            .type = "invalid_request_error",
                            .message = "the request head is malformed",
                        },
                        error.HttpHeadersOversize => .{
                            .status = 431,
                            .type = "invalid_request_error",
                            .message = "the request head is too large",
                        },
                        error.MethodUnknown => .{
                            .status = 501,
                            .type = "invalid_request_error",
                            .message = "the request method is not implemented",
                        },
                        error.HttpConnectionClosing,
                        error.HttpRequestTruncated,
                        error.ReadFailed,
                        => return .ended,
                    };
                    respondHeadError(server.out, refusal);
                    return .refused;
                };
                const keep_asked = keepAlive(&request);
                const verdict = handleRequest(engine, c, &request) catch |err| {
                    log.debug("conn {d}: request failed: {t}", .{ engine.connId(c), err });
                    return .ended;
                };
                // The std server still waits for a next head after a reply that said
                // `connection: close`, so the task ends the connection here.
                switch (verdict) {
                    .refused => return .refused,
                    .close => return .ended,
                    .keep => if (!keep_asked) return .ended,
                }
            }
        }

        fn handleRequest(engine: *Eng, c: *Conn, request: *http.Server.Request) !Verdict {
            var arena_state = std.heap.ArenaAllocator.init(engine.gpa);
            defer arena_state.deinit();

            const arena = arena_state.allocator();

            const method = request.head.method;
            const target = stripQuery(request.head.target);
            log.debug("http: {t} {s}", .{ method, target });

            // Decide the body form before a body read moves the head bytes.
            const head_bytes = request.head_buffer;
            const form = body_mod.framing(method, request.head.version, head_bytes);
            // RFC 9110 section 15.2: a server must not send an interim reply to an HTTP/1.0 client.
            const continue_first = form != .refused and
                request.head.version == .@"HTTP/1.1" and
                body_mod.expectsContinue(head_bytes);
            request.head.expect = null;
            switch (form) {
                .refused => |r| return respondError(request, .{
                    .status = r.status(),
                    .type = "invalid_request_error",
                    .message = r.message(),
                }),
                .none, .length, .chunked => {},
            }

            // Route the request. HEAD on a GET route answers as GET, and std drops the body.
            const route = Route.of(target) orelse return respondError(
                request,
                .{ .status = 404, .type = "invalid_request_error", .message = "unknown route" },
            );
            if (!route.allows(method)) {
                return respondErrorAllow(request, .{
                    .status = 405,
                    .type = "invalid_request_error",
                    .message = "the route does not take this method",
                }, route.allowed());
            }

            return switch (route) {
                .models => respondModels(engine, request, arena),
                .health => replyWhole(request, "{\"status\":\"ok\"}"),
                .chat => handleChat(engine, c, request, arena, form, continue_first),
            };
        }

        fn handleChat(
            engine: *Eng,
            c: *Conn,
            request: *http.Server.Request,
            arena: Allocator,
            form: body_mod.Framing,
            continue_first: bool,
        ) !Verdict {
            const sp = span.begin(.http_chat);
            defer sp.end();

            // Read the request body. The framing refused a length past the cap before any read.
            const codec = engine.codec;
            const body = switch (try readBody(request, form, continue_first, arena)) {
                .bytes => |bytes| bytes,
                .refused => |he| return respondError(request, he),
            };

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
            const prompt = try chat.promptMessagesAlloc(arena, mode.kind, req.messages);
            const input: TurnInput = .{
                .rendered = prompt.text,
                .plain = try arena.dupe(chat.PlainRange, prompt.plain()),
                .max_tokens = req.max_tokens,
                .think_open = chat.thinkOpen(mode.kind),
                .think_budget = mode.think_budget,
                .stop = req.stop,
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
                return streamChat(engine, c, request, arena, sid, input, id_str);
            } else {
                return bufferChat(engine, c, request, arena, sid, input, id_str);
            }
        }

        /// Open an ephemeral session with an empty system prompt, so no prefill runs. Return its
        /// protocol id, or null when the pool is exhausted or the connection dropped.
        fn openSession(engine: *Eng, c: *Conn, sampling: protocol.Sampling) ?u32 {
            const system_prompt = engine.gpa.dupe(u8, "") catch return null;
            const id = engine.connId(c);
            engine.submitFrom(c, .{ .conn = id, .request_id = 1, .session_id = 0, .body = .{
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
                    engine.wake(engine.connId(c));
                }
            }
        }

        /// End the turn after a stop match. The scheduler then emits the `done` of the turn.
        fn cancelTurn(engine: *Eng, c: *Conn, sid: u32) !void {
            const id = engine.connId(c);
            engine.submitFrom(
                c,
                .{ .conn = id, .request_id = 4, .session_id = sid, .body = .cancel },
            ) catch return error.ConnClosed;
        }

        fn closeSession(engine: *Eng, c: *Conn, sid: u32) void {
            log.debug("session close: {d}", .{sid});
            const id = engine.connId(c);
            engine.submitFrom(
                c,
                .{ .conn = id, .request_id = 3, .session_id = sid, .body = .session_close },
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
            const plain = engine.gpa.dupe(chat.PlainRange, input.plain) catch |e| {
                engine.gpa.free(text);
                return e;
            };
            const id = engine.connId(c);
            engine.submitFrom(c, .{ .conn = id, .request_id = 2, .session_id = sid, .body = .{
                .generate = .{
                    .max_tokens = input.max_tokens,
                    .sampling = null,
                    .text = text,
                    .plain = plain,
                    .think_open = input.think_open,
                    .think_budget = input.think_budget,
                },
            } }) catch {
                engine.gpa.free(text);
                engine.gpa.free(plain);
                return error.SubmitFailed;
            };

            // Drain the events until the turn ends. A stop match cancels the turn, and the events
            // up to its `done` then drop.
            var split: map.ReasoningSplit = .init(engine.codec.think_ids, input.think_open);
            var matcher: map.StopMatcher = .init(input.stop);
            var stopped = false;
            while (true) {
                const ev = c.out_q.getOne(engine.io) catch return error.ConnClosed;
                // A buffered reply writes nothing before the end, so only a probe sees a hang-up.
                // A streamed reply sees it as a write error. The `done` completes the reply, so
                // the probe skips it, and a client that closed its send side still gets it.
                const probe = @TypeOf(sink) == *BufSink and ev.body != .done;
                if (probe and timed_stream.peerClosed(c.stream)) {
                    log.debug("turn: session {d}, the client hung up", .{sid});
                    return error.ConnClosed;
                }

                switch (ev.body) {
                    // The scheduler emits the terminal EOS token. OpenAI content omits it.
                    .token => |t| if (!stopped and t.token_id != engine.codec.eos_id) {
                        if (split.route(t.token_id, t.text[0..t.text_len])) |p| {
                            // A stop string matches the reply text, and never the reasoning.
                            if (p.field == .reasoning or input.stop.len == 0) {
                                try sink.onPiece(p.field, p.text);
                            } else {
                                const out = matcher.feed(p.text);
                                if (out.text.len > 0) {
                                    try sink.onPiece(.content, out.text);
                                }
                                if (out.stopped) {
                                    stopped = true;
                                    try cancelTurn(engine, c, sid);
                                }
                            }
                        }
                    },
                    .done => |d| {
                        log.debug(
                            "turn: session {d}, {d} tokens, finish {t}, stop matched {}",
                            .{ sid, d.gen_tokens, d.finish_reason, stopped },
                        );
                        // A held tail that no stop string completed belongs to the reply.
                        if (!stopped and input.stop.len > 0) {
                            const rest = matcher.finish();
                            if (rest.len > 0) {
                                try sink.onPiece(.content, rest);
                            }
                        }

                        return .{
                            .done = .{
                                .n = d.gen_tokens,
                                .reasoning = d.reasoning_tokens,
                                .finish = if (stopped) "stop" else finishStr(d.finish_reason),
                            },
                        };
                    },
                    .err => |e| return .{
                        .failed = try turnError(engine, arena, e.code, input),
                    },
                    // Only the binary reader task produces these, in its own queue.
                    .zmtp_pong, .zmtp_error => unreachable,
                    else => {}, // session_closed from a prior turn, etc.
                }

                if (c.parked_hint.swap(false, .acq_rel)) {
                    engine.wake(engine.connId(c));
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
        ) !Verdict {
            // Run the turn into a buffer.
            var sink: BufSink = .{ .arena = arena };
            const turn = switch (try runTurn(engine, c, arena, sid, input, &sink)) {
                .done => |d| d,
                .failed => |he| return respondError(request, he),
            };

            // Build the response.
            const prompt_tokens = countPrompt(engine, input);
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

            // Send the reply with its length, which every HTTP version reads.
            return replyWhole(request, try std.json.Stringify.valueAlloc(arena, resp, json_opts));
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
        ) !Verdict {
            var sink: SseSink = .{
                .request = request,
                .id = id_str,
                .model = engine.codec.model_id,
                .close_delimited = request.head.version == .@"HTTP/1.0",
            };
            const after: Verdict = if (sink.close_delimited) .close else .keep;
            switch (try runTurn(engine, c, arena, sid, input, &sink)) {
                .done => |turn| {
                    const body = try sink.open();
                    try writeChunk(body, id_str, engine.codec.model_id, .{}, turn.finish);
                    try body.writer.writeAll("data: [DONE]\n\n");
                    try body.end();
                    return after;
                },
                .failed => |he| {
                    const body = if (sink.body) |*b| b else return respondError(request, he);
                    try writeErrorEvent(&body.writer, he);
                    try body.end();
                    return after;
                },
            }
        }

        fn respondModels(engine: *Eng, request: *http.Server.Request, arena: Allocator) !Verdict {
            const data = [_]types.ModelEntry{.{ .id = engine.codec.model_id }};
            const list: types.ModelsList = .{ .data = &data };
            return replyWhole(request, try std.json.Stringify.valueAlloc(arena, list, json_opts));
        }

        /// Prompt-token count for `usage`: re-tokenize the rendered prompt. This costs little next
        /// to a decode, and it keeps the count exact without a path back through the wire.
        fn countPrompt(engine: *Eng, input: TurnInput) u32 {
            const ids = engine.codec.turnTokens(
                engine.gpa,
                .open_first,
                input.rendered,
                input.plain,
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
            input: TurnInput,
        ) !map.HttpError {
            return switch (code) {
                .turn_too_long => try map.exceedContextError(
                    arena,
                    countPrompt(engine, input),
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

/// Read the body that `form` declares. The std body reader never runs, so the std server state
/// moves to `ready` here, and the next head reads as after a std body read.
fn readBody(
    request: *http.Server.Request,
    form: body_mod.Framing,
    continue_first: bool,
    arena: Allocator,
) !BodyRead {
    const in = request.server.reader.in;
    const short: map.HttpError = .{
        .status = 400,
        .type = "invalid_request_error",
        .message = "the request body ended before its declared end",
    };
    const result: BodyRead = switch (form) {
        .refused => unreachable,
        .none => .{ .bytes = "" },
        .length => |n| read: {
            if (continue_first) {
                try sendContinue(request);
            }
            const bytes = try arena.alloc(u8, n);
            in.readSliceAll(bytes) catch |err| switch (err) {
                error.EndOfStream => break :read .{ .refused = short },
                error.ReadFailed => return error.BodyReadFailed,
            };
            break :read .{ .bytes = bytes };
        },
        .chunked => read: {
            if (continue_first) {
                try sendContinue(request);
            }
            const bytes = body_mod.readChunked(in, arena) catch |err| switch (err) {
                error.ReadFailed => return error.BodyReadFailed,
                error.OutOfMemory => return error.OutOfMemory,
                error.Truncated => break :read .{ .refused = short },
                error.TooLarge => break :read .{ .refused = .{
                    .status = 413,
                    .type = "invalid_request_error",
                    .message = "request body too large",
                } },
                error.OverheadTooLarge => break :read .{ .refused = .{
                    .status = 413,
                    .type = "invalid_request_error",
                    .message = overhead_message,
                } },
                error.SizeInvalid,
                error.ExtensionTooLong,
                error.ExtensionInvalid,
                error.TrailerInvalid,
                error.LineEndInvalid,
                error.DataEndInvalid,
                error.TrailerTooLong,
                => break :read .{ .refused = .{
                    .status = 400,
                    .type = "invalid_request_error",
                    .message = "the chunked framing of the request body is malformed",
                } },
            };
            break :read .{ .bytes = bytes };
        },
    };
    request.server.reader.state = .ready;
    return result;
}

/// The interim reply of `Expect: 100-continue`, sent before the task reads an accepted body.
fn sendContinue(request: *http.Server.Request) !void {
    const out = request.server.out;
    try out.writeAll("HTTP/1.1 100 Continue\r\n\r\n");
    try out.flush();
}

/// Answer a head that the std parser refused. No `Request` exists, so the reply is written whole.
fn respondHeadError(w: *Io.Writer, e: map.HttpError) void {
    log.debug("http error {d}: {s}", .{ e.status, e.message });
    var json_buf: [512]u8 = undefined;
    var json: Io.Writer = .fixed(&json_buf);
    std.json.Stringify.value(map.errorObject(e), json_opts, &json) catch return;
    const phrase = statusEnum(e.status).phrase() orelse "";
    w.print(
        "HTTP/1.1 {d} {s}\r\ncontent-type: application/json\r\ncontent-length: {d}\r\n" ++
            "connection: close\r\n\r\n{s}",
        .{ e.status, phrase, json.end, json.buffered() },
    ) catch return;
    w.flush() catch {};
}

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

fn respondError(request: *http.Server.Request, e: map.HttpError) !Verdict {
    return respondErrorAllow(request, e, null);
}

/// Send an error reply with its length and `connection: close`. A 405 names the methods of its
/// route in `allow`.
fn respondErrorAllow(request: *http.Server.Request, e: map.HttpError, allow: ?[]const u8) !Verdict {
    log.debug("http error {d}: {s}", .{ e.status, e.message });
    var json_buf: [1024]u8 = undefined;
    var json: Io.Writer = .fixed(&json_buf);
    try std.json.Stringify.value(map.errorObject(e), json_opts, &json);
    const allow_header: http.Header = .{ .name = "allow", .value = allow orelse "" };
    const with_allow = [_]http.Header{ json_headers[0], allow_header };
    try request.respond(json.buffered(), .{
        .status = statusEnum(e.status),
        .keep_alive = false,
        .extra_headers = if (allow == null) &json_headers else &with_allow,
    });
    return .refused;
}

/// Send a whole reply with its length. It keeps the connection when the client asked for that
/// on HTTP/1.1.
fn replyWhole(request: *http.Server.Request, content: []const u8) !Verdict {
    assertBodyRead(request);
    try request.respond(content, .{
        .keep_alive = keepAlive(request),
        .extra_headers = &json_headers,
    });
    return .keep;
}

/// A reply to a method with a body comes after `readBody`. Otherwise the std server reads the
/// body itself with its own decoder before it keeps the connection.
fn assertBodyRead(request: *const http.Server.Request) void {
    if (request.head.method.requestHasBody()) {
        std.debug.assert(request.server.reader.state == .ready);
    }
}

/// True when the connection stays open after the reply. An HTTP/1.0 connection ends after one
/// reply, so its replies need no keep-alive form.
fn keepAlive(request: *const http.Server.Request) bool {
    return request.head.keep_alive and request.head.version == .@"HTTP/1.1";
}

/// Read one head and parse it, as `http.Server.receiveHead` does, but keep the parse error, so
/// an unknown method gets its own status.
fn receiveRequest(server: *http.Server) ReceiveError!http.Server.Request {
    const head_buffer = try server.reader.receiveHead();
    const head = http.Server.Request.Head.parse(head_buffer) catch |err| return switch (err) {
        error.UnknownHttpMethod => error.MethodUnknown,
        else => error.HeadInvalid,
    };
    return .{ .server = server, .head_buffer = head_buffer, .head = head };
}

fn statusEnum(code: u16) http.Status {
    return switch (code) {
        400 => .bad_request,
        404 => .not_found,
        405 => .method_not_allowed,
        413 => .payload_too_large,
        417 => .expectation_failed,
        431 => .request_header_fields_too_large,
        500 => .internal_server_error,
        501 => .not_implemented,
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
    const want_open =
        \\data: {"error":{"code":400,
    ;
    const want_message =
        \\"message":"the request of 130 tokens exceeds the context of 128 tokens",
    ;
    const want_rest =
        \\"type":"exceed_context_size_error","n_prompt_tokens":130,"n_ctx":128}}
        \\
        \\data: [DONE]
        \\
        \\
    ;
    try std.testing.expectEqualStrings(want_open ++ want_message ++ want_rest, w.buffered());
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
