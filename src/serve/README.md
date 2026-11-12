# Serving

This directory holds the serving layer. It turns the single-sequence runner into a server that
clients reach over a socket. Each conversation is a session with a KV slot, and its position
persists across turns. One forward pass per tick advances a batch of sessions. `engine.zig` and
`scheduler.zig` know no wire format. Each frontend is a thin adapter between bytes and values.

## Files

| File | Holds |
| --- | --- |
| `scheduler.zig` | `Scheduler`, the control unit: the session table and the composed tick. |
| `engine.zig` | `ServeEngine`, the engine thread, the queues, and the connection lifecycle. |
| `conn_slots.zig` | `ConnSlots`, the connection slot table and the steps of each thread. |
| `timed_stream.zig` | The socket reader and writer with a deadline, and the hang-up probe. |
| `event.zig` | `Request`, `OutEvent`, and `Codec`, the values across the two boundaries. |
| `prefill_chunk.zig` | The pure rule for the prefill rows of one pass. |
| `residency.zig` | `Residency`, the optional tiered KV pool. |
| `prefix_manager.zig` | `PrefixManager`, the serve prefix cache and its warm-up. |
| `protocol.zig` | The binary wire format: frames, verbs, and error codes. |
| `zmtp.zig` | The pure ZMTP 3.x codec under the frames. |
| `server.zig` | The binary transport adapter, a ZMTP ROUTER. |
| `client.zig` | The reference binary client, a ZMTP DEALER. |
| `openai/http_server.zig` | The OpenAI HTTP adapter. |
| `openai/body.zig` | The HTTP body: the framing rules of a head and the strict chunked decoder. |
| `openai/map.zig` | The HTTP trust boundary: parse, validate, and map a body to a value. |
| `openai/types.zig` | The OpenAI request and response structs. |
| `openai/raw_codec.zig` | `RawCodec`, the HTTP codec that encodes a render with its plain ranges. |

## How the pieces fit

A transport adapter decodes bytes into `Request` values, and the engine loop feeds them to the
scheduler. The scheduler drives the session pool of `runtime/` and emits `OutEvent` values back.

| Value | File | Holds | Made by | Read by |
| --- | --- | --- | --- | --- |
| `Request` | `event.zig` | One decoded client request, with owned text | The adapters | `Scheduler.handle` |
| `OutEvent` | `event.zig` | One fixed-size event for a connection | `Scheduler` | The adapters |
| `Codec` | `event.zig` | The chat template, the tokenizer, and the piece table | `cli/serve_cmd` | `Scheduler` |
| `Scheduler` | `scheduler.zig` | The session table, the state machines, and the config | `ServeEngine` | The engine loop |
| `ServeEngine` | `engine.zig` | One engine thread and its bounded queues | `server.Server`, `http_server.Server` | `cli/serve_cmd`, `cli/serve_openai_cmd` |
| `Residency` | `residency.zig` | The logical sessions, the LRU clock, and the RAM ledger | `cli/boot/serve_boot` | `Scheduler`, as its `Engine` |
| `PrefixManager` | `prefix_manager.zig` | The cache entries and counters over a `PromptCache` | `Scheduler` | `Scheduler` |
| `Frame` payload | `protocol.zig` | A 20-byte header and a bounded payload | `client.zig`, `server.zig` | The same two |

The scheduler takes its engine and its codec as parameters. The engine is `Residency` over
`runtime/session.Sessions`, or `Sessions` alone. The codec is `event.Codec` for the binary wire and
`openai.RawCodec` for HTTP. The scheduler imports no socket.

```
  acceptor --accept--> Adapter.onAccept(conn) --decode--> request_queue
                                                              |
                                                              v
  conn task(s) <--encode-- out_queue[i] <--emit-- ENGINE LOOP: sched.handle, sched.tick
                                                              |
                                                              v
                                         Residency --> Sessions --> runner.stepBatch
```

The adapters differ in how a connection runs. `BinaryAdapter` in `server.zig` runs a reader task and
a writer task per connection, and one ZMTP message carries one `protocol.zig` frame. `HttpAdapter`
in `openai/http_server.zig` runs one task per connection. A third transport is a third adapter, with
no engine change.

The wire layers stack from bytes to values.

```
  socket --> zmtp.zig (greeting, READY, PING) --> protocol.zig (header + payload) --> Request
  socket --> std.http.Server (head) --> openai/body.zig --> openai/map.zig      --> Request
```

The prefix cache sits beside the pool.

```
  Scheduler --asks--> PrefixManager --> runtime/prompt_cache.PromptCache (RAM tier, disk tier)
      |                                        ^
      +--restorePrefix / snapshotPrefix--> Sessions --> runner blob
```

`Residency` presents the same method surface as `Sessions`, so the scheduler only swaps its engine
type. The split with the cache is the engine boundary. The scheduler makes the calls whose result a
slot state machine consumes. The manager makes the calls whose result only the cache consumes.

## Rules

- **The serving invariant.** A batched session produces exactly the tokens of
  that session alone. On the CPU the tokens and the logits are bit-equal, and
  across backends the greedy tokens are equal. `tests/serve_integration_test.zig`
  observes it from the binary wire and `tests/openai_integration_test.zig`
  observes it over HTTP, on top of the runner equivalence tests.
- **The socket is a trust boundary.** `zmtp.zig` and the adapter check every ZMTP length first. A
  greeting has exactly 64 bytes. A command body has at most 1024 bytes. A message has at most 20
  bytes plus 64 KiB. A ping context has at most 16 bytes. Then `protocol.zig` checks the frame
  inside the message against `max_frame_payload_bytes` (64 KiB), `limits.max_token_text_bytes`
  (256), and `limits.max_model_name_bytes` (64). `openai/body.zig` checks the body framing, and
  `openai/map.zig` checks the JSON the same way. A reader must consume a payload exactly, and
  trailing bytes are malformed. A decode never panics. It returns a typed error that the adapter
  maps to an error frame or an HTTP status: 4xx for a malformed request, and 501 for an unknown
  method or transfer coding.
- **Bounded everything.** `--max-connections` sets the live connections, 8 by
  default and 64 at most (`core/limits.zig`). A binary connection costs two
  worker threads. The table holds two slots per allowed connection, and a
  binary slot holds one payload buffer of 20 bytes plus 64 KiB and one
  out-queue. `engine.zig` fixes `request_queue_capacity` at 64 and
  `out_queue_capacity` at 256 events per slot. `--max-sessions` caps at 8
  on a CPU build, and at 128 on a fused-GPU build once `--vram-slots` or
  `--kv-vram` enables the KV tier. `--vram-slots` reaches 128 and `--max-batch`
  reaches 8 (`core/limits.zig`). `--max-seq` sets the session window, from 1 to
  the model ceiling, and the default comes from a per-session byte budget (see
  "Windows" in `docs/architecture.md`). `generate.max_tokens` is a `u32`, and the
  decode rejects 0. The chat client sends the whole window by default.
- **A server shared by several clients stays fair and safe.** A session belongs
  to the connection that opened it. Decode admission is round-robin across the
  ready sessions, whatever their connection. A full out-queue parks only the
  session that emits into it. A hang-up sweeps exactly the sessions of that
  connection. The connection past the cap gets `connections_exhausted` after
  the greeting and READY of the server. `--sessions-per-connection` caps the
  share of one connection with `sessions_exhausted`, and its default equals
  `--max-sessions`. A wire `shutdown` gets `forbidden` unless `--allow-shutdown`
  is set. Connection 0 never comes from the wire, so a request of the process
  owner always stops the server. `serve-openai` has no `--allow-shutdown` and no
  share option, because HTTP has no shutdown verb and holds one session per
  connection. `tests/serve_integration_test.zig` checks the cap, the share, the
  sweep, and the refused shutdown from the wire.
- **Session leases.** A lease keeps a session through a drop. The scheduler holds it in the slot as
  a 128-bit token from the secure random source and a time to live. `serve --resume-ttl MS` sets
  `Config.resume_ttl_max_ms`, the cap on the time to live, and `limits.max_resume_ttl_ms` (one hour)
  caps the flag. The default is 0, which turns resumption off, so every drop frees as before. At a
  drop, a leased slot detaches until the drop time plus its time to live. A turn in flight takes the
  cancel rule, and an opening or append prefill runs on. A detached slot takes no decode row, emits
  no event, and no request reaches it. A resume with the token attaches the slot to the new
  connection within its share, and the token rotates. The newest holder wins, so a resume also takes
  an attached slot. A resume of a slot that is not idle waits in the slot and ends at the moment the
  slot becomes idle. Until then, the session answers `session_busy` to a generate, an append, a
  clone, a cancel, a keep, or a close. The client has no `resumed` reply yet. The scheduler frees a
  detached slot at its deadline. It scans lazily at the tick head and at each admission, with one
  clock read per scan. A live open with no free slot takes the detached slot with the earliest
  deadline, so a lease never blocks a live client. An expiry and a yield stop a prefill in flight
  before the release. A clone that tiered KV refuses after a yield loses the yielded session, and
  the log line of the yield names the loss. The serve stats line counts the expiries, the yields,
  the lost yields, and the cut prefills. The server never logs a token.
- **A lost lease reply strands the session until its time to live.** A resume rotates the token
  before the `resumed` reply goes out. A keep issues its token in the `kept` reply. If that reply is
  lost, for example because the connection drops before the delivery, no client holds a valid token.
  The session stays detached until its time to live ends, and the expiry scan then frees it. The
  client falls back to a rebuild from its own history: an open with the original system prompt and
  an append of the kept pairs. The time to live bounds the stranded slot, and a live open can take
  it earlier.
- **The engine never blocks on a client.** Emission is non-blocking. A full
  out-queue parks one event in the session's single `pending_event` cell, and the
  session leaves the batch until the draining task signals a wakeup. A client
  whose queue stays full while it keeps sending is broken, and `disconnect` is
  the bounded answer.
- **Backpressure is the only legal answer to a full RAM ledger.** An eviction that pushes the ledger
  past its budget gets `error.RamBudgetExceeded`. The scheduler then parks that turn until a slot
  frees. A blob is the only copy of its conversation, so dropping one is never legal.
- **A full window is a stop, never a shift or a cut.** The server ends a turn with
  `context_full` when the window fills. A turn or a system prompt that does
  not fit gets a `turn_too_long` refusal, and the detail names both numbers
  (`protocol.turn_too_long_fmt`, `protocol.system_too_long_fmt`). The server
  never drops, shifts, or truncates rows. The chat client continues a full
  conversation by compaction (`cli/client/client_transcript.zig`).
- **The cache paths are best-effort.** Any refusal or failure in the prefix cache degrades to a cold
  prefill. A warm restore equals a cold prefill token for token.
- **Boundary values carry the ownership rule.** `Request` is the one value that owns bytes across a
  thread hop. The adapter dupes the turn text into the gpa, and the HTTP adapter dupes the plain
  ranges of its render too. The scheduler frees both after tokenizing. `OutEvent` is fixed-size,
  so the queues hold plain copies and need no ownership rule at all.
- **Request text is plain text.** The server writes the chat template around the text of a client
  and encodes the result with `bpe.encodePromptAlloc`. A turn takes one encode, and a kept pair
  takes two: the user wrap, then the reply. A control token matches only in the template text. In
  the text of a client, a string such as `<|im_end|>` or `<think>` encodes as ordinary bytes.
  Text with no such string keeps the ids of a one-string encode, so a cached prefix stays valid.
- **The plain-text rule covers every text of a client.** On the OpenAI frontend it covers the
  content of every message role. On the binary frontend it covers the turn text, the system prompt
  of `session_open`, and both texts of a kept pair. It also covers a `--preload` text. The rule
  stops a client from forging a control token, but it does not stop prompt injection in plain words.
- **Token text is raw bytes.** One `token` event can carry a partial UTF-8
  sequence, and only the concatenation of a turn is well-formed. The reference
  client returns the bytes as they are, and `cli/client/client_streaming.zig` holds a partial
  codepoint back until the next token. The HTTP adapter holds a partial
  codepoint back too, so every SSE delta is valid UTF-8.
- **A control token sends no text.** `cli/boot/codec_parts.zig` gives each token of the GGUF type
  CONTROL the empty piece, as llama.cpp does by default. Examples are `<|im_end|>`, `<|eot_id|>`,
  and `<|python_tag|>`. A user-defined token, such as `<think>` or `<tool_call>` in the Qwen3.5
  files, keeps its text. The id, the count, and the position of a control token stay as they are,
  and a turn still ends at the EOS id. Only the text of its event is empty. So a `stop` string
  cannot match a control token. A model can sample an end token that is not the EOS id of its
  file, for example `<|eom_id|>` in the Llama 3.1 files. The turn then runs on, as before, and the
  client does not see that token.
- **The per-op GPU backend cannot serve.** `cli/boot/boot_lines.zig` returns
  `error.UnsupportedServeBackend` for it. The CPU runners, the fused runners, and
  the hybrids all serve.
- **The blob tier of the RAM ledger has three values.** `absent` means no session can be evicted.
  `bounded` carries the full need of one blob more than the evictable sessions. `unenforced` means
  `--kv-ram` is unset and the free RAM is unknown. A runner without `evictSlot` compiles `Residency`
  down to the identity, and the pool then needs `n_logical <= n_physical`.
- **The disk tier of the prefix cache shares one directory.** `--prefix-disk` adds it. The default
  is `.prompt_cache` under the working directory, and `$ZIGFERENCER_CACHE_DIR` relocates it. The
  disk budget bounds one process namespace and not the directory. An operator who shares a directory
  across models must size the disk for the sum of the active budgets.
- **The engine thread owns the runner.** The thread that calls `run` alone touches the runner, the
  sessions, and the scheduler. `run` drains the request queue, calls `sched.handle` for each
  request, and calls `sched.tick` until it returns `.idle`. It then blocks on the request queue, so
  an idle server uses no CPU.
- **A connection slot changes hands in a fixed order.** A slot is `free`, `live`, `closing`,
  `swept`, or `reclaimable` (`conn_slots.zig`). The acceptor writes a slot only in `free` or
  `reclaimable`, and in `live` when it abandons a slot whose task did not start. The engine thread
  reads a slot only in `live` or `closing`. The reader task submits `conn_closed` after its last
  request. The reply that a request gets while the engine handles it, such as a `pong` or
  `session_closed`, is then in the out-queue. The engine closes the queue, sweeps the sessions of
  the connection, and releases the slot. The writer task drains the queue and exits. A writer that
  stops early closes the queue, so its reader cannot wait on a full queue for good. The slot is
  `reclaimable` after the release and the last task exit, in either order. So an event never reaches
  the queue of a later connection. An event for a connection that holds no slot is dropped, because
  the engine closed its queue. At shutdown the request queue closes, so a `conn_closed` can be lost.
  Teardown then awaits the tasks and frees the slot.
- **A half-close is a hang-up.** A client that closes its send side ends its turns on the binary
  frontend. On Linux it also ends a buffered HTTP turn at the next token. A streamed HTTP turn runs
  to its end, because a write to a half-closed client succeeds. The sweep cancels a turn in flight,
  drops a parked event, and ends a waiting resume. A reply that the engine made before the sweep
  still arrives. A buffered HTTP reply that is complete when the probe runs still goes out. A binary
  frame error queues its error frame, and the replies to the earlier requests of the connection
  follow it before the close.
- **A slot that waits for the engine does not refuse a client.** `--max-connections` counts the
  `live` slots, where a task runs and the engine has not handled `conn_closed`. The table holds
  two slots per allowed connection, so a client can connect while the slot of an ended
  connection waits for the engine. A client that finds every spare slot waiting gets the refusal
  of the cap. A slot costs `@sizeOf(Conn)` of its frontend plus the slot record, so the table
  costs twice that per allowed connection. The debug line `listen: N connection slots of B
  bytes` of the engine prints the sum at boot.
- **A refusal never waits.** The refusal of a client past the cap is one send that does not
  wait. The server then shuts its send side, drops at most 528 KiB (`limits.close_drain_bytes`) of
  the input that has already arrived with no wait, and closes. So the refusal does not turn into a
  reset where the client has sent little.
- **A socket wait has a deadline.** `--io-timeout` sets the longest wait of one socket read or
  write, 600 seconds by default (the default of the llama.cpp server `--timeout`). 0 sets no
  deadline. Each read of an HTTP request head and body waits at most the deadline, which covers the
  idle wait between two requests on one connection. A client that sends one byte per deadline can
  still hold its connection. The deadline also bounds each read of the binary handshake and of a
  binary frame after its first byte, and each write that the client does not read. At the deadline
  the task exits, so the slot and the sessions of the connection free as after a hang-up. Only Linux
  has the deadline. On other platforms a socket wait has no bound. The input drain before a close
  and the refusal send that does not wait are Linux only too. On other platforms the server drops no
  input before a close, and the refusal send is the blocking std write.
- **An idle binary connection closes when it holds nothing.** A binary reader waits for the first
  byte of a frame with no bound while its connection holds a session or a request in the engine,
  because an idle session is normal. A connection with neither closes after one deadline with no
  frame. So only a client that keeps a session open holds a slot with no traffic. On Linux the TCP
  keepalive of an accepted socket finds a peer that vanished with no close after 120 seconds
  (`limits.zig`). That time holds for a connection with no unacknowledged output, because the kernel
  sends no probe while output waits for an acknowledgment. With output in flight, a write that
  waits for room in the send buffer ends at the `--io-timeout` deadline. Output that fits in the
  send buffer starts no wait. A binary connection that holds a session then ends only at the
  retransmission limit of the kernel. With `--io-timeout 0` the server sets no bound on a write
  wait, and only the retransmission limit of the kernel ends it.
- **A buffered reply sees a hang-up at the next token.** A buffered HTTP reply writes nothing
  before the turn ends. So at each token event of the turn the task probes the socket for a
  hang-up, and a hang-up ends the turn and closes the session. The probe skips the `done`, so a
  complete reply still goes out. A request that the client sends before it reads the reply is no
  hang-up. A prefill emits no event, so the probe waits for the first token. A streamed reply sees
  a hang-up as a write error. Only Linux has the probe.
- **An HTTP reply that says close ends the connection.** An error reply, a request with
  `Connection: close`, and every HTTP/1.0 request end the connection after the reply. A request
  that the client sent after it is not answered.
- **An HTTP connection ends in a fixed order.** A close over unread input sends a reset, and a reset
  can drop the last reply. After a refusal, the task shuts its send side. It then drops input until
  the end of the stream, 528 KiB (`limits.close_drain_bytes`), or 5 seconds
  (`limits.close_linger_ms`), whichever comes first, and the socket closes. A nonzero `--io-timeout`
  below 5 seconds shortens the wait. With `--io-timeout 0` the wait is 5 seconds. After any other
  end, the task drops at most 528 KiB of the input that has already arrived, with no wait, and the
  close sends the end of stream. The drain lowers the chance of a reset, and it does not remove it.
  Every error reply ends the connection this way, also one that follows a complete body (a bad JSON
  body, a model mismatch, a prompt past the window). The connection counts toward
  `--max-connections` until the client closes or the wait ends. On the binary frontend only the
  refusal at the cap drains: it shuts the send side and drops the input that has arrived, with no
  wait. Any other binary connection closes with no drain.
- **The two transports serve two kinds of traffic.** The binary wire serves embedded interactive
  traffic. That means many short, low-latency sessions that often share a prefilled prefix. The HTTP
  frontend serves general interop. A program that sends the request shapes of the field table in
  "The OpenAI HTTP front" works with no change. A browser page on another origin cannot call the
  server, because its CORS preflight gets 405 and no reply carries CORS headers. The server ignores
  the other fields. It logs each of the first 32 unknown names (`UnknownFields.table_cap`) once, and
  then one line that says the table is full.

## The client surface

`client.zig` holds no session state. It holds one connection, a request id
counter, one payload buffer that `next` lends out, and the last recorded error.
Every request method belongs to one of two families. A send-only method writes
one frame and returns its request id, and the caller reads frames through `next`
or `collectTurn`. A terminal-wait method writes one frame, consumes frames until
the terminal message of that request, and returns the reply as a value.
Terminal-wait methods assume one request in flight on the connection.

Each `token` frame carries its id. The frame of a control token carries no text, and it still
counts in `gen_tokens`. The server sends the end-of-sequence token as a token frame too. In the
SmolLM2, Qwen3.5, and Llama-3.1 files that token is a control token, so its frame carries no text.
`hello` records its id, `collectTurn` skips its text, and a client that renders text skips the
token whose id equals `eos_id` from `hello_ack`.

The wire is `protocol_version` 3. The server refuses a header of any other version, version 2
included, with `unsupported_version` and closes the connection. Each `token` frame carries a channel
byte: 0 content, 1 reasoning, 2 the open tag of a think block, 3 its close tag. A tag keeps its
text, so a client that ignores the channel prints the same bytes. `done` counts the reasoning
tokens. `hello_ack` names the think mode of the server (0 no tags, 1 off, 2 on) and the think budget
(0 is unbounded). It also names `resume_ttl_max_ms`, the longest lease (0 is resumption off). It
also carries two card rows, `recommended_think` and `recommended_plain`. They hold the sampling that
the model card recommends for a turn inside the think block and for a turn outside it. A model with
no card values gets greedy. A client picks the row of the mode of each turn.

`Sampling` is 36 bytes. It carries a presence penalty in [-2, 2], a repetition
penalty in (0, 2], and a min_p in [0, 1]. The identity values are 0, 1, and 0.
It also carries a frequency penalty in [-2, 2]. The server has no frequency
penalty yet. A request with a nonzero value gets `err` with `forbidden` and
the detail `frequency_penalty is not supported`, changes nothing, and the
connection lives. A value out of range or NaN is `malformed_frame`, like every
other guard. The server never drops the field.

The flags byte of `generate` holds five bits. Bit 0 says that a sampling block
follows. Bit 1 says that the turn carries a think switch, and bit 2 is its
value: 1 opens the turn inside the think block. Bit 3 says that a `u32` think
budget follows the sampling block. A budget of 0 is unbounded, and a budget is
at most `limits.max_model_ctx`. A request budget replaces the server value for
that turn only. Bit 4 restarts the sampling stream of the session at the seed
of the sampling block, so a turn replays. Without bit 4 the server ignores
that seed, and the stream of the session continues. Bit 2 without bit 1, bit 4
without bit 0, and any of bits 5 to 7 are malformed. A refused `generate`
leaves the sampling of the session unchanged. On a
model with no think tags, a switch that opens the block or any budget gets
`err` with `forbidden`. A switch that keeps the block closed passes, because it
names the only mode of such a model. The scheduler resolves the switch once,
and the wrap of the turn and the think state of the slot read that one value.
A budget above 0 on a model with no forced close also gets `err` with `forbidden`.

`next` and `send` touch disjoint fields, so one task can read while another task
writes. `deinit` runs after both tasks return. Every row of the table runs
against a live server in `tests/serve_integration_test.zig`.

| Wire verb | Client method | Family | Terminal message |
| --- | --- | --- | --- |
| `hello` | `hello` | terminal-wait | `hello_ack` |
| `session_open` | `openSession`, or `openSessionInfo` for the `kv_pos` too | terminal-wait | `session_opened` |
| `session_clone` | `cloneSession`, or `cloneSessionInfo` for the `kv_pos` too | terminal-wait | `session_opened`, the new id in the header |
| `generate` | `generate`, then `next` per frame, `collectTurn`, or `collectTurnWith` for a hook per token | send-only | `done` |
| `session_append` | `append`, or `sendAppend` then `next` | terminal-wait | `appended` with the position after each pair, or `err`. A `turn_too_long` refusal feeds no pair. A `cancel` during the append gives `err` with `cancelled`, and the fed rows stay, so the client closes the session |
| `cancel` | `cancel` | send-only | the turn's `done` with `cancelled`, under the request id of the `generate`. A cancel on an idle session produces no frame, so a turn of that session must be in flight |
| `session_close` | `closeSession` | terminal-wait | `session_closed`. Frames of an in-flight turn drain first, and the method skips them |
| `stats` | `stats` | terminal-wait | `stats_ack` |
| `ping` | `ping` | terminal-wait | `pong` |
| `shutdown` | `shutdown` | terminal-wait | `shutdown_ack`, or `err` with `forbidden` when the server refuses it |
| `session_keep` | `keepSession` | terminal-wait | `kept` with the granted time to live and the token |
| `session_resume` | `resumeSession` | terminal-wait | `resumed`, sent when the session is idle, with the session id in the header |

`connectTcp` and `connectUnix` open the connection and run the ZMTP handshake.
`hangUp` ends the connection from this side and wakes a task blocked in `next`.
`deinit` frees the client after every task that uses the client returns.

A `done` frame ends every turn with one of four finish reasons: `eos`, `max_tokens`,
`context_full`, or `cancelled`. A client can read the numbers back from a
`turn_too_long` detail with `protocol.turn_too_long_fmt` and `protocol.system_too_long_fmt`.

`generateWith` sends a `generate` with every per-turn field: the sampling
block, the think switch, and the think budget.

`session_keep{u32 ttl_ms}` keeps a session past a drop of this connection. The grant is `min(ttl_ms,
resume_ttl_max_ms)`. The `kept` reply is 40 bytes: the `u32` grant, then a 16-byte token. A grant of
0 drops the lease and carries 16 zero bytes. A positive grant carries a token that is not all zero.
Each keep issues a new token, and the old token opens nothing. `session_resume{[16]u8 token}` goes
out with header session id 0 on any connection. Any other header id is `malformed_frame`. The
`resumed` reply is 58 bytes. It holds the new token, `kv_pos`, `turns_done`, and a `has_last` byte.
It then holds the last finished request: its id, its finish reason, its generated tokens, and its
reasoning tokens. With `has_last` 0 these four fields are 0. The reply comes when the session is
idle, so `kv_pos` is final. The presented token opens nothing after the resume, so a client stores
the new one.

| Request | Condition | Answer |
| --- | --- | --- |
| `session_keep` | no such session on this connection | `unknown_session` "no such session" |
| `session_keep` with `ttl_ms` > 0 | resumption is off | `forbidden` "resumption is off". A keep with `ttl_ms` 0 always succeeds |
| `session_keep` | a resume of the session waits | `session_busy` "resume in flight" |
| `session_keep` | the secure random source fails | `internal` "no secure random source", and the lease does not change |
| `session_resume` | an unknown, rotated, or expired token | `unknown_session` "no such lease" |
| `session_resume` | the share of this connection is full | `sessions_exhausted` |
| `generate`, `session_append`, `session_clone`, `cancel`, `session_close` | a resume of the session waits | `session_busy` "resume in flight" |

One answer covers an unknown, a rotated, and an expired token, so a probe
learns nothing from the difference. The server compares tokens in constant
time. A token that a slot already holds makes the server draw again, at most 4
times and then answers `internal` "token draws collided". Two equal draws of
128 bits do not occur in practice. The token is a bearer secret,
and ZMTP NULL security sends it in clear text. Bind to 127.0.0.1 or to an `ipc`
path with tight file permissions.

`session_append` feeds kept (user, assistant) pairs into an idle session. The
server renders each pair as a live turn at the same place: the user wrap, the
closed think tail, and the reply text. It writes no end of the assistant turn,
because the next wrap writes it. A reply holds no think block. Both texts are request text, so the
server encodes each one as plain text. A live turn never encodes its reply, because the model
sampled those ids. A sampled control token sends no text to the client, so a kept reply does not
hold it. The text on both sides of that token then encodes as one plain text, so the kept pair
costs fewer positions than the live turn. A sampled user-defined token, such as a tool-call tag,
sends its text, and that text encodes as plain text in the kept pair, so the pair costs more
positions. In both cases the pair gets other ids than the live turn, and a later prefix cache
lookup misses. The server reports the real positions, and the client adopts them. One frame holds
at most `limits.max_append_pairs` pairs and 64 KiB. A longer history goes in
several frames, and the session ends in the same state as after one frame. A
pair costs the positions of its ids. With thinking off, a reply that
re-encodes to its sampled ids costs what the live turn cost. With thinking on,
the pair drops the reasoning, so it costs less. The chat client compacts a
full window with this frame. It closes the session and opens one with the
exact original system prompt, which a prefix cache can serve. It then appends
the newest pairs that fit.

The supporting values are plain data. `Event` is one decoded frame whose slices borrow the payload
buffer. `OwnedEvent` is a fixed-size copy of an `Event` for a caller that hands frames across a task
boundary. These calls return typed results. `collectTurn` and `collectTurnWith` return `Turn`.
`hello` returns `HelloInfo`. `append` returns `protocol.Appended`. `openSessionInfo` and
`cloneSessionInfo` return `Opened`. `keepSession` returns `Kept`. `resumeSession` returns
`ResumedSession`. `stats` returns `Stats`. `LastError` records the last `err` frame that a method
consumed. Its detail holds at most 64 bytes, and `truncated` marks a longer detail that the client
cut.

### A client in any language

1. Create a DEALER socket with your ZMQ binding.
2. Connect it to `tcp://HOST:PORT` or to `ipc://PATH`. The `READY` line of `serve` gives the
   address.
3. Send one message per request. The message is the 20-byte header of `protocol.zig` followed by the
   payload.
4. Receive one message per reply. Decode the header, then the payload, with the same tables.

- **Single-part messages only.** A message with the MORE flag closes the
  connection.
- **A `REQ` socket is refused.** The server answers its handshake with a ZMTP
  `ERROR`.
- **ZMTP 3.x with the NULL security method only.** The server refuses every
  other version and every other security method.
- **A message holds at most 20 bytes plus 64 KiB.** A larger message closes the
  connection.
- **The server answers a heartbeat.** A peer PING gets a PONG with the same
  context.
- **A reconnect starts with no sessions unless a lease holds one.** The server
  sweeps the sessions of a connection when that connection ends. A session
  with a lease detaches for its time to live, and `session_resume` with the
  token attaches it to the new connection (see "Session leases").
- **On `unknown_session` for a kept session, resume first.** A libzmq DEALER
  reconnects by itself and queues sends in the gap. The application can
  miss the drop. Its next request on the session then gets
  `unknown_session`. Send `session_resume` with the stored token, store the
  new token, and send the request again.
- **Sessions are private to the connection that opened them.** Another
  connection that names the session id gets `unknown_session`.

### What the server accepts, refuses, and ignores

| Peer behavior | Server answer |
| --- | --- |
| ZMTP 3.0 or 3.1 greeting, any 3.x minor | accepted |
| ZMTP 1.0 or 2.0 greeting | connection closed at the greeting |
| a security method other than NULL | ZMTP `ERROR "NULL only"`, then close |
| `READY` with `Socket-Type` DEALER, or without `Socket-Type` | accepted |
| `READY` with `REQ`, `ROUTER`, or any other type | ZMTP `ERROR "DEALER only"`, then close |
| a first command other than `READY` | ZMTP `ERROR "READY expected"`, then close |
| an `Identity` property, any value | ignored |
| an unknown `READY` property | ignored |
| `PING` | `PONG` with the same context |
| `SUBSCRIBE`, `CANCEL`, or an unknown command | ignored |
| a message with the MORE flag | `err` with `malformed_frame`, then close |
| a message shorter than 20 bytes or longer than 20 bytes plus 64 KiB | `err` with `malformed_frame` or `frame_too_large`, then close |
| a heartbeat from the server side | none, the client sets its own interval |

### Validated clients

Compatibility is a claim about named clients, and tests keep it true. The table
names each client and the way the claim holds for it.

| Client | Version | Transports | How |
| --- | --- | --- | --- |
| libzmq through the system library | 4.3.5 | `tcp` and `ipc` | `tests/zmq_interop_test.zig` opens it at run time and drives a DEALER against the in-process server, two DEALERs at once included. |
| libzmq through a Python ctypes DEALER | 4.3.5 | `tcp` and `ipc` | A DEALER drove a running `serve --resume-ttl` by hand with the heartbeat on. It kept its session, closed, and resumed it on a new DEALER. |

### The OpenAI HTTP front

`serve-openai` serves `POST /v1/chat/completions`, buffered and SSE. Each
request opens its own session, so the session-open seed starts the sampling
stream of the request, and the turn sends no reseed. The table lists what the
front does with each request field.

To try the front, start `zig-out/bin/zigferencer serve-openai MODEL --port 8080`, then send a
request:

```
curl -N http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"MODEL","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

| Field | Rule |
| --- | --- |
| `messages` | honored, at most 64 messages. A `content` is a string, an array of `text` parts, or null on an assistant message. The texts of the parts join in order with no separator. A part of another type, for example `image_url`, gets a 400 that names the type, because the model takes text only. A message holds at most 64 parts |
| `model`, `stream`, `max_tokens`, `max_completion_tokens`, `n` | honored. `n` above 1 gets a 400. A `model` that is not the served id gets a 404 |
| `stop` | honored, a string or an array of at most 4 strings of 1 to 64 bytes. The reply ends at the first token after which the reply text holds a complete stop string. Of the stop strings complete at that token, the one that starts first sets the cut, and the reply ends just before it. So when stop strings overlap, the token boundaries can decide the cut. `finish_reason` is `stop`. A stop string in the reasoning text does nothing. A stream holds back at most 63 bytes that can still start a stop string. `usage.completion_tokens` also counts the tokens that the model made after the match and before the cancel landed |
| `temperature` | honored, finite and 0 or more. 0 is greedy |
| `top_k` | honored, 0 to 256. 0 and -1 are off. A value past 256 gets a 400 |
| `top_p` | honored, more than 0 and at most 1 |
| `min_p` | honored, 0 to 1 |
| `presence_penalty` | honored, -2 to 2 |
| `repetition_penalty` | honored, more than 0 and at most 2, 1 is off. It counts the prompt and the reply |
| `frequency_penalty` | 0 is accepted. Any other value gets a 400, because the engine has no frequency penalty |
| `seed` | honored. Without it, the server draws a seed from the secure random source per request |
| `thinking_token_budget` | -1 or absent takes `--thinking-budget`. 0 turns thinking off. N from 1 to 262144 closes the block after N think tokens |
| `chat_template_kwargs.enable_thinking` | true is on, false is off, absent is the server mode |
| any other field | ignored. The server logs each of the first 32 unknown names (`UnknownFields.table_cap`) once. Examples: `tools`, `tool_choice`, `logprobs`, `response_format`, `stream_options`, `logit_bias`, and `user` |

A field refusal is a 400 of type `invalid_request_error`, and its message names
the field. A range refusal also names the range. A `model` mismatch gets 404 of
type `model_not_found`, and a prompt past the window gets 400 of type
`exceed_context_size_error`. A model with no think tags accepts only the
closed modes. `enable_thinking: true` or a budget above 0 gets a 400. A
budget of 0 or -1 passes. A model with think tags and no forced close also
refuses a budget above 0. A budget of 0 wins over `enable_thinking: true`.
A field with a value of the wrong JSON type, for example a string for
`top_k`, gets a 400 that names the field (`error.BadField` in `map.zig`).

The server decides the body form from the head before it reads a body byte (`openai/body.zig`).
A `Content-Length` of ASCII digits declares a body of that length. `Transfer-Encoding: chunked`
declares a chunked body. A request with neither has no body. So a POST to the chat route with
neither gets the 400 of an empty JSON body at once, and a POST to an unknown route gets 404. A
body holds at most 256 KiB. Each refusal below closes the connection.

| Request | Answer |
| --- | --- |
| a control byte in the target or the version | 400. A control byte in the method gets 501, because std resolves the method first |
| a field line that is not a name of token characters, a colon right after it, and a value with no control byte but tab, for example `Content-Length : 5`, a bare LF or NUL inside a line, a line with no colon, or a folded line | 400, so a proxy and this server never read two sets of fields |
| a method that std does not know | 501 |
| a known method that the route does not take | 405 with `allow` |
| `Content-Length` and `Transfer-Encoding` together | 400 |
| a transfer coding that std reads as a content coding, for example `gzip`, `gzip, chunked`, or `none` | 501, the status that RFC 9112 names for a transfer coding the server does not know. Another coding name, for example `br` or `Chunked`, gets 400 from the std parser |
| `Transfer-Encoding` on an HTTP/1.0 request | 400 |
| a `Content-Length` that is not ASCII digits, for example `+10` or `1_0` | 400 |
| two `Content-Length` headers, or another head that the std parser refuses | 400. An unknown method gets 501 |
| a head past 16 KiB | 431 |
| a `Content-Length` past 256 KiB | 413, before the server reads the body. A length past the `u64` range gets 400 from the std parser |
| a body on a method that takes none, for example GET or HEAD | 400, so the body never reads as the next request |
| an `Expect` other than `100-continue` | 417 |
| a chunked body that passes 256 KiB | 413 at the size line that passes the cap, before its data |
| chunk framing bytes past 256 KiB | 413 with a message that names the framing bound, for example a body of very small chunks |
| a malformed chunk framing, or a trailer line that is no field line | 400 |
| a body that ends before its declared length or its last chunk | 400 at the end of the stream, and no turn runs. A client that stalls inside a body is closed at the `--io-timeout` deadline with no reply |

A chunk size line holds 1 to 8 hex digits, an optional extension, and CRLF. An extension is
optional white space, a semicolon, and value bytes, at most 1024 bytes in all, and the server
skips it. Every line of the framing ends with CRLF. A trailer line is a field line, and the server
skips it. The trailer bound of 16 KiB counts the name, colon, and value bytes of the trailer
lines, and no line end. The framing bytes of one body hold at most 256 KiB, so a chunked body
costs at most 512 KiB on the wire. `--io-timeout` bounds each read.

`Expect: 100-continue` on an accepted HTTP/1.1 head gets the interim reply `100 Continue` just
before the server reads the body. A head that the server refuses gets its refusal and no interim
reply. An HTTP/1.0 client gets no interim reply, as RFC 9110 section 15.2 requires.

`GET /v1/models`, `GET /healthz`, and `GET /v1/health` also take HEAD, which gets the GET reply
with no body. An HTTP/1.0 request gets one reply, and the connection ends. A buffered reply and an
error reply carry `content-length` for every version. A stream to an HTTP/1.0 client carries no
chunked coding, and it ends at the close.

An omitted sampling field takes a default. Under `--sampling model`,
the default, it takes the field of the registry card row of the resolved
thinking mode, the vLLM priority. A model with no card row takes the greedy
row. Under `--sampling greedy`, every omitted field keeps the greedy
identity. A field that the request sends always wins over the card. A client
that needs a fixed reply sends `temperature: 0` or a fixed `seed`. One seed,
one build, and one request give one reply. Under the card defaults,
`temperature: 0` still applies the card penalties, the vLLM rule, so the
reply can differ from the pure greedy stream. `--sampling greedy`
gives the pure greedy stream.

The buffered reply adds `usage.completion_tokens_details.reasoning_tokens`
when the turn has think tokens.

A control token adds no text to `content` or `reasoning_content`, and a stream sends no chunk for
it. `usage.completion_tokens` still counts it. A user-defined token, such as `<tool_call>`, keeps
its text in the reply.

## Policy points

Every cache and residency policy is one pure function or one private scan in one
file. To change a policy, edit its one home. If a policy grows variants, promote
it to a closed enum in `scheduler.Config` and give the variants their own file
then, and not before.

| Policy | Home | Invariant it must keep |
| --- | --- | --- |
| Eviction victim (residency) | `residency.zig` `lruVictim` | never a slot in this tick's exclude set |
| Over-budget RAM behavior | `residency.zig`, the evict refusal | backpressure only, because a blob is the only copy and a drop is never legal |
| Prefix RAM admission | `runtime/prompt_cache.zig` `worthwhile` | a measured gate: the prefill must beat twice the timed restore |
| Prefix disk admission | `runtime/prompt_cache.zig` `admit`, the disk bar | adds the blob read-back at the observed disk bandwidth |
| Prefix RAM eviction | `runtime/prompt_cache.zig` `evictLru` | the budget can overshoot by at most one entry |
| Prefix disk eviction | `runtime/prompt_cache.zig` `diskEvictToFit` | own namespace only. The budget bounds this process, and the victim is its oldest file by mtime |
| Cache-insert eligibility | `prefix_manager.zig` `insert`, offered by `scheduler.zig` `maybeInsertPrefix` | position-zero completions only |
| The RAM split, blobs against prefix | `core/kv_budget.zig` `splitRam`, with the corners in `engine/device_placement.zig` | blobs get their provable need, the prefix gets the surplus |
| The blob tier of an unset `--kv-ram` | `core/kv_budget.zig` `quoteRam`, called through `engine/resolve.zig` and `engine/device_placement.zig` | the full need of `(E + 1)` blobs, or a boot refusal with both fits |
| Full window | `scheduler.zig` decode admission and the turn refusal, `cli/client/client_transcript.zig` `plan` | the server stops and refuses with numbers, the client keeps the longest suffix of turns that fits beside the system prompt and the reserve |
