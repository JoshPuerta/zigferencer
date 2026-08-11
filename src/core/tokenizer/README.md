# src/core/tokenizer

This directory turns prompt bytes into token ids, and token ids back into text.
It is a pure datapath, the same side of the split that
[docs/architecture.md](../../../docs/architecture.md) describes for the rest of
`core/`. `vocab.zig` builds the immutable `Vocab` once at load, on the caller's
arena. `bpe.zig` is a stateless encode and decode over a `*const Vocab`, and it
takes an allocator only for the output slice. The one value that crosses out of
here is `Vocab` itself, plus the `[]u32` and `[]u8` results.

## Files

| File | Role |
| --- | --- |
| `vocab.zig` | Builds `Vocab` from the GGUF tokenizer tables: the id-to-bytes and bytes-to-id maps, the u64-keyed merge rank and merge target maps, the flat `[256]u32` byte lookup, the comptime GPT-2 byte to codepoint bijection, the specials sorted longest first, the BOS and EOS ids, and the `PreType` regime. The specials are the control tokens and the user-defined added tokens, such as `<think>`. |
| `bpe.zig` | `encodeAlloc` and `decodeAlloc` over a `*const Vocab`: the greedy longest-special match, the BPE merge, the optional BOS, the per-family pre-split paths, and the decode that undoes the byte-level mapping. |

## How the pieces fit

`vocab.zig` owns the data and `bpe.zig` owns the algorithm. The encode speed
lives in the shapes `vocab.zig` chooses, a flat `byte_to_id`, u64 pair keys, and
a length-descending `specials_sorted`, so `bpe.zig` stays a small reader over
them.

```
  io/gguf/parser.zig  --Parsed-->  vocab.buildFromParsed  --Vocab-->  bpe.encodeAlloc
   KV pairs and                     one build at load       on the     bpe.decodeAlloc
   tensor info                                              arena      output on the gpa
```

The dependency runs one way. `bpe` imports `vocab` for `Vocab`, `no_token`,
`pairKey`, and `byteLevel`. `vocab` imports `io/gguf/parser.zig`. `bpe` never
touches GGUF.

Inside encode, `vocab.pre` picks the path at the start. `.default` merges the
whole span between two special tokens as one segment. `.llama3`, `.qwen`, and
`.smollm` pre-split the text on a GPT-2 style regex first, so a merge never
crosses a pre-token boundary. `flushSegment` resolves a pending byte run through
`merge_rank` and `merge_to_id`, and `matchSpecial` walks `specials_sorted`.
Decode walks each token codepoint by codepoint through `codepoint_to_byte`.

## Rules

- `Vocab` is built once on the caller's arena and nothing mutates it afterward.
  Its token bytes borrow the GGUF map, so it lives no longer than the load.
- `encodeAlloc` prepends BOS only when the caller passes `sequence_start = true`
  and `vocab.add_bos` is set. A caller passes text and never prepends BOS itself,
  so a follow-up turn passes false and BOS appears at position 0 alone.
- `encodeAlloc` asserts `vocab.tokens.len > 0` first, and under runtime safety it
  asserts that every emitted id is below `vocab.tokens.len`.
- A byte whose GPT-2 byte-level character is absent from the vocabulary raises
  `error.UntokenizableByte`. The `no_token` sentinel marks that absence, so the
  encoder never returns a wrong id in silence.
- `vocab` pair-validates across the parser boundary. It asserts that the token
  table and the token-type table have equal length, and a comptime check
  confirms that the GPT-2 table produces exactly 68 escape codepoints.
- `decodeAlloc(vocab, encodeAlloc(vocab, s, ...))` reproduces `s` for simple
  inputs. That round trip is the invariant every pre-split path must preserve.

## Where to start

1. `vocab.zig`: the `Vocab` fields, `PreType`, and `buildFromParsed`, which is what the data looks like before an encode runs.
2. `bpe.zig`: `encodeAlloc` on the default path, then `encodePreSplit`, then `decodeAlloc`.
