//! The OpenAI connector's codec: the scheduler's text->token boundary, but WITHOUT
//! chat scaffolding. The HTTP map already rendered the whole conversation with
//! `chat.wrapMessagesAlloc`, so this only tokenizes (BOS on the prefill turn).
//! It also carries the model metadata the HTTP task needs (id, kind, think
//! tags, budgets).
//! The codec is the connector's whole view of the model, the way `event.Codec`
//! carries vocab + pieces.

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
    /// The think tags of a `chatml_qwen` kind, null for every other kind. The
    /// reasoning splitter keys on them.
    think_ids: ?chat.ThinkIds,
    model_id: []const u8, // for response JSON + /v1/models
    max_seq: u32, // hard ceiling on a turn
    default_max_tokens: u32, // when a request omits a budget
    /// The sampling of each omitted request field, one row per thinking mode.
    defaults: map.DefaultRows = .greedy,
    /// The scheduler holds the ids of a forced think close, so a request
    /// budget above 0 can close the block.
    think_forced_close: bool = false,

    /// Tokenize a turn's text. No chat-wrap, because the map did it, and the
    /// map also rendered the think tail, so `open` changes nothing here.
    /// `sequence_start` (BOS, per the vocab) opens the prefill turn.
    pub fn turnTokens(self: *const RawCodec, gpa: Allocator, turn: event.TurnKind, text: []const u8, open: bool) ![]u32 {
        _ = open;
        const sequence_start = turn == .system or turn == .open_first;
        return bpe.encodeAlloc(self.vocab, gpa, text, sequence_start);
    }

    /// The connector renders whole conversations through the map, so it has
    /// no pair render, and the scheduler refuses an append with this error.
    pub fn pairTokens(self: *const RawCodec, gpa: Allocator, place: chat.PairPlace, user: []const u8, assistant: []const u8) error{AppendUnsupported}![]u32 {
        _ = self;
        _ = gpa;
        _ = place;
        _ = user;
        _ = assistant;
        return error.AppendUnsupported;
    }

    pub fn piece(self: *const RawCodec, id: u32) []const u8 {
        std.debug.assert(id < self.pieces.len);
        return self.pieces[id];
    }
};
