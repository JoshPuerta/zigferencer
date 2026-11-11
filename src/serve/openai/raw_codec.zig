//! The OpenAI frontend's codec: the scheduler's text-to-token boundary. The HTTP task already
//! rendered the whole conversation with `chat.promptMessagesAlloc`, so a turn only tokenizes, with
//! BOS on the prefill turn. The plain ranges of the render ride with the text, so the request text
//! encodes as plain text.
//!
//! System text. A session opens with no system text, except for a `--preload` text. The codec
//! wraps that text in the system block itself. For a template kind, a control token follows the
//! block in a live request, so no merge crosses its end, and its ids are a prefix of a live request
//! that opens with the same system message. Kind `none` has no such token, so this can fail there.
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
    kind: chat.Kind, // for the render of the HTTP task and the system block
    /// The think tags of a `chatml_qwen` kind, null for every other kind.
    think_ids: ?chat.ThinkIds,
    model_id: []const u8, // for response JSON + /v1/models
    max_seq: u32, // hard ceiling on a turn
    default_max_tokens: u32, // when a request omits a budget
    /// The sampling of each omitted request field, one row per thinking mode.
    defaults: map.DefaultRows = .greedy,
    /// True when the scheduler holds a forced think close, so a budget above 0 can close the block.
    think_forced_close: bool = false,

    /// Tokenize a turn's text. A `system` text gets the system block. Any other text is a render
    /// with its plain ranges, so `open` changes nothing. The vocab BOS opens the prefill turn.
    pub fn turnTokens(
        self: *const RawCodec,
        gpa: Allocator,
        turn: event.TurnKind,
        text: []const u8,
        plain: []const chat.PlainRange,
        open: bool,
    ) ![]u32 {
        _ = open;
        const sequence_start = turn == .system or turn == .open_first;
        if (turn != .system) {
            return bpe.encodePromptAlloc(self.vocab, gpa, text, plain, sequence_start);
        }

        std.debug.assert(plain.len == 0);
        const prompt = try chat.promptSystemAlloc(gpa, self.kind, text);
        defer gpa.free(prompt.text);

        return bpe.encodePromptAlloc(self.vocab, gpa, prompt.text, prompt.plain(), sequence_start);
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
