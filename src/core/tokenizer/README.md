# Tokenizer

This directory turns prompt bytes into token ids, and token ids back into text. `vocab.zig` builds
the immutable `Vocab` once at load. `bpe.zig` is a stateless encode and decode over a
`*const Vocab` and takes an allocator only for its output slice.

## Files

| File | Holds |
| --- | --- |
| `vocab.zig` | `Vocab`, built from the GGUF tokenizer tables. |
| `bpe.zig` | `encodeAlloc`, `encodePromptAlloc`, and `decodeAlloc` over a `Vocab`. |

## How the pieces fit

`vocab.zig` owns the data and `bpe.zig` owns the algorithm. The encode speed lives in the shapes
that `vocab.zig` chooses, so `bpe.zig` stays a small reader over them.

| Value | File | Holds | Made by | Read by |
| --- | --- | --- | --- | --- |
| `Vocab` | `vocab.zig` | The token bytes, the merge maps, the byte lookup, the specials, BOS and EOS, and the `PreType` | `buildFromParsed` | `bpe`, `serve/event`, `serve/openai`, `cli/`, `dev/` |
| `Special` | `vocab.zig` | The id and bytes of a control or added token such as `<think>` | `buildFromParsed`, sorted longest first | `bpe.matchSpecial`, through the `Specials` cursor |
| `PreType` | `vocab.zig` | `default`, `llama3`, `qwen`, or `smollm`, the pre-split regime | `buildFromParsed`, from `tokenizer.ggml.pre` | `bpe.encodePromptAlloc` |
| `PlainRange` | `bpe.zig` | A byte range of request text, where no special token matches | `core/chat.zig`, in a `Prompt` | `bpe.encodePromptAlloc` |
| `Specials` | `bpe.zig` | The cursor of one encode over its plain ranges | each encode call | `encodePromptAlloc`, `capAtSpecial` |
| BPE state | `bpe.zig` | A pending segment of ids that `flushSegment` merges by `merge_rank` | each encode call | `bpe` alone |
| `[]u32` and `[]u8` | `bpe.zig` | The ids of an encode and the text of a decode | `encodeAlloc`, `encodePromptAlloc`, `decodeAlloc` | the CLI, the serve codecs, the benches |

The data flows from the GGUF file to the ids:

```
  io/gguf/parser   --Parsed-->  vocab.buildFromParsed  --Vocab-->  bpe.encodePromptAlloc  --> []u32
   metadata pairs                once at load, on the              bpe.decodeAlloc        --> []u8
                                 caller's arena
```

The dependency runs one way. `bpe` imports `vocab` for `Vocab`, `no_token`, `pairKey`, and
`byteLevel`, and `vocab` imports `io/gguf/parser.zig`. `bpe` never touches GGUF.

Inside encode, `vocab.pre` picks the path at the start:

```
  text --> Specials cursor, matchSpecial --> a special id, or the span between two specials
                                                          |
             .default: the whole span is one segment      |  .llama3, .qwen, .smollm:
                          |                               |  preTokenLen splits first
                          v                               v
                  flushSegment: byte_to_id, then merge_rank and merge_to_id --> ids
```

Decode walks each token codepoint by codepoint through `codepoint_to_byte`.

## Rules

- `Vocab` is built once on the caller's arena and nothing mutates it afterward. Its token bytes
  borrow the GGUF map, so it lives no longer than the load.
- `encodeAlloc` prepends BOS only when the caller passes `sequence_start = true` and `vocab.add_bos`
  is set. A caller passes text and never prepends BOS itself, so a follow-up turn passes false and
  BOS appears at position 0 alone.
- `encodePromptAlloc` takes the `PlainRange` list of a rendered prompt. A special token matches
  only when it lies wholly outside every range. `encodeAlloc` passes no range, so every special
  that `matchSpecial` finds matches.
- A merge or a whole pre-token never yields a special id. Outside a plain range the cursor matches
  each special first, so only request text can spell one there, and it must stay bytes.
- `encodeAlloc` asserts `vocab.tokens.len > 0` first, and under runtime safety it asserts that
  every emitted id is below `vocab.tokens.len`.
- A byte whose GPT-2 byte-level character is absent from the vocabulary raises
  `error.UntokenizableByte`. The `no_token` sentinel marks that absence, so the encoder never
  returns a wrong id in silence.
- `vocab` checks the tokenizer values across the parser boundary. It refuses token and token-type
  tables of different lengths, an empty token table, and a BOS or EOS id past the table, with
  `error.BadTokenizerValue`. A comptime check confirms that the GPT-2 table produces
  exactly 68 escape codepoints.
- `decodeAlloc(vocab, encodeAlloc(vocab, s, ...))` reproduces `s` for simple inputs. That round trip
  is the invariant every pre-split path must preserve.
