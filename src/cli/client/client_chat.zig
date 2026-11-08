//! The state of the streaming control unit. `client_streaming.zig` holds its transitions, and
//! `client_reconnect.zig` holds the `reconnecting` state.
//!
//! States. `idle` waits for a line, and `streaming` holds one turn in flight. `opening`,
//! `appending`, and `keeping` wait on the request `pending_id`. `keeping` waits on the `kept` reply
//! of a new session. `reconnecting` holds the main task while it dials and resumes, so no input
//! reaches the machine in that state. `quitting` ends the loop.
//!
//! Quit. `/quit` and EOF wait for the turn in flight and every due reply, so piped input sees its
//! whole output before the hang-up.

const std = @import("std");
const Io = std.Io;

const limits = @import("../../core/limits.zig");
const protocol = @import("../../serve/protocol.zig");
const client_mod = @import("../../serve/client.zig");

const Client = client_mod.Client;

const client_history = @import("client_history.zig");
const client_input = @import("client_input.zig");
const client_output = @import("client_output.zig");
const client_startup = @import("client_startup.zig");

// -- types ------------------------------------------------------------------

pub const State = enum { opening, appending, keeping, idle, streaming, reconnecting, quitting };

/// The reply of the turn in flight as the client sees it. Each `generate` starts a new one.
pub const TurnStream = struct {
    /// A token can end inside a codepoint. The tail waits here for the next token.
    hold: [client_output.codepoint_tail_max]u8 = undefined,
    hold_len: usize = 0,
    think: client_output.ThinkView = .{},
    /// The token frames that the server counts, for `reconcile`.
    received: u32 = 0,
};

/// The prompt of the turn in flight, so that a refused turn can go out once more.
pub const Prompt = struct {
    buf: [client_input.line_bytes_max]u8 = undefined,
    len: u32 = 0,
    /// The prompt waits for the open of a compaction.
    waiting: bool = false,
    /// The turn in flight is a retry, so a second refusal is final.
    retried: bool = false,
};

/// The streaming control unit. Only the main task touches it.
pub const Chat = struct {
    io: Io,
    client: *Client,
    out: client_output.Out,
    opts: *const client_startup.Options,
    /// The text of the `info.eos_id` token frame is not part of the reply.
    info: Client.HelloInfo,
    sampling: protocol.Sampling,
    state: State = .idle,
    session: ?Client.Opened,
    /// The id of the request that the current state waits on.
    pending_id: u32 = 0,
    stream: TurnStream = .{},
    ping: ?struct { id: u32, t0: Io.Clock.Timestamp } = null,
    /// Replies to `/stats` and `session_close` that have not arrived.
    replies_due: u32 = 0,
    quit_pending: bool = false,
    lost_session: bool = false,
    hist: *client_history.History,
    cycle: ?client_history.Cycle = null,
    prompt: Prompt = .{},
    target: client_startup.Target,
    /// The lease that each new session asks for. 0 keeps no session.
    keep_ttl: u32,
    /// The token of the lease on `session`. Null falls back to a rebuild.
    lease: ?protocol.Token,
    /// The allocator of each new client.
    gpa: std.mem.Allocator,
    /// The generation of the attempt task in flight. A marker of an older one is stale.
    gen: u32 = 0,
    /// The id of the newest timer. A marker of an older one is stale.
    timer_gen: u32 = 0,
    /// The sources of `/clone`, which stay open on this connection until the quit closes them.
    sources: [limits.max_logical_sessions]u32 = undefined,
    sources_len: u32 = 0,
};
