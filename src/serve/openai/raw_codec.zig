//! The OpenAI frontend's codec: the scheduler's text-to-token boundary, without chat scaffolding.
//! The HTTP map already rendered the whole conversation with `chat.wrapMessagesAlloc`, so this
//! only tokenizes (BOS on the prefill turn).
//!
//! Model view. The codec is the frontend's whole view of the model. It carries the vocab, the
//! pieces, and the metadata that the HTTP task needs: id, kind, think tags, and budgets.
//!
//! Appends. The frontend renders whole conversations through the map, so it has no pair render.
//! The scheduler refuses an append with `error.AppendUnsupported`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const event = @import("../event.zig");
const bpe = @import("../../core/tokenizer/bpe.zig");
const vocab_mod = @import("../../core/tokenizer/vocab.zig");
const chat = @import("../../core/chat.zig");
const map = @import("map.zig");

pub const RawCodec = struct {
    vocab: *const vocab_mod.Vocab,
    pieces: []const []const u8,
    eos_id: u32,
    kind: chat.Kind, // for the HTTP map's wrapMessagesAlloc
    /// The think tags of a `chatml_qwen` kind, null for every other kind.
    think_ids: ?chat.ThinkIds,
    model_id: []const u8, // for response JSON + /v1/models
    max_seq: u32, // hard ceiling on a turn
    default_max_tokens: u32, // when a request omits a budget
    /// The sampling of each omitted request field, one row per thinking mode.
    defaults: map.DefaultRows = .greedy,
    /// True when the scheduler holds a forced think close, so a budget above 0 can close the block.
    think_forced_close: bool = false,

    /// Tokenize a turn's text without a chat wrap. The map rendered the wrap and the think tail,
    /// so `open` changes nothing. The vocab BOS opens the prefill turn.
    pub fn turnTokens(
        self: *const RawCodec,
        gpa: Allocator,
        turn: event.TurnKind,
        text: []const u8,
        open: bool,
    ) ![]u32 {
        _ = open;
        const sequence_start = turn == .system or turn == .open_first;
        return bpe.encodeAlloc(self.vocab, gpa, text, sequence_start);
    }

    pub fn pairTokens(
        self: *const RawCodec,
        gpa: Allocator,
        order: chat.PairOrder,
        user: []const u8,
        assistant: []const u8,
    ) error{AppendUnsupported}![]u32 {
        _ = self;
        _ = gpa;
        _ = order;
        _ = user;
        _ = assistant;
        return error.AppendUnsupported;
    }

    pub fn piece(self: *const RawCodec, id: u32) []const u8 {
        std.debug.assert(id < self.pieces.len);

        return self.pieces[id];
    }
};
