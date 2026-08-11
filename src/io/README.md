# src/io

This subtree is the edge between the engine and the outside world. It turns a
Hugging Face URL into bytes on disk, maps those bytes read-only, and parses them
into the typed values the runtime consumes. `download.zig` and `mmap.zig` are
shell, because they hold the I/O and the operating-system resources.
`gguf/parser.zig` is pure datapath, and `gguf/loader.zig` is the glue that joins
the map, the parser, and the semantic checks. For the layer above this one, read
[docs/architecture.md](../../docs/architecture.md).

`gguf/` carries no README of its own, because the subtree tells one story:
bytes to disk to map to `Parsed` to `ModelCard`. This document covers it end to
end.

## Files

| File | Role |
| --- | --- |
| `download.zig` | Fetch a model GGUF over HTTPS with `std.http.Client` and write it to `<root>/<local_path>/<file_name>`. The body streams in chunks behind a throttled progress bar, and `remoteSize` reads only the response head for the `fetch --dry-run` size preview. The standard library supplies the certificate-verified TLS, so there is no dependency. |
| `mmap.zig` | `Mapped`: a read-only whole-file map with no copy. `openReadOnly` prefaults every page, for a reader that consumes the whole file. `openReadOnlyLazy` faults pages in on first touch, so a metadata-only reader opens a multi-gigabyte file in milliseconds. |
| `gguf/parser.zig` | The GGUF v3 binary parser, pure: a byte slice goes in and a structured `Parsed` value, the header, the KV pairs, and the tensor info, comes out. It checks the byte-level form and rejects malformed input with a typed `Error`. |
| `gguf/loader.zig` | The glue: map, parse, then read the architecture-specific KV keys into `Loaded`, which holds the `ModelCard`, the `[]TensorRef`, and the `Parsed` KV pairs. It asserts semantic completeness and the `core/limits.zig` ceilings. A declared context past `limits.max_model_ctx` loads at that bound with one log line, and a RoPE scaling the engine does not apply fails with `error.UnsupportedRopeScaling`. |

## How the pieces fit

The arrows are the load path. Only `loader.zig` knows the whole chain. The parser
never sees a file, because the caller hands it a byte slice. `mmap.zig` never
sees the GGUF structure, because it returns raw bytes.

```
              fetch                                  load
              -----                                  ----
  registry.Model                                 loader.load(path)
        |                                               |
        v  download.fetchModel                          v
  HTTPS -> <file>.gguf.partial                  mmap.openReadOnly[Lazy]
        |  atomic rename on success                     |  []const u8, whole file
        v                                               v
  <root>/<local_path>/<file_name>  ------->      parser.parse(bytes)
        (on disk, read by a later run)                  |  Parsed { header, kv,
                                                        |           []TensorInfo }
                                                        v
                                        architecture KV extraction,
                                        limits.zig ceilings, dtype and shape
                                                        |
                                                        v
                                 Loaded { map, parsed, card, tensors }
                                   ModelCard + []TensorRef
```

`download.zig` does not feed `loader.zig` in one call. `fetch` writes the file,
and a later `load` reads it. The two meet on disk alone, at the path
`core/registry.zig` computes for both. `core/tokenizer/vocab.zig` reads the KV
tables in `parsed` later to build a `Vocab`. The loader surfaces `vocab_size` in
the `ModelCard` and leaves that build to its consumer.

## Rules

- The parser and the loader validate in pairs, and neither trusts the other. The parser checks the `"GGUF"` identifier and the version. It checks that the alignment is a power of two and that the counts do not overflow. It checks that the tensor offsets are ordered and that each is a multiple of the alignment. It checks that a name is at most 64 bytes. The loader checks that every architecture key resolves and that no card dimension is zero. It checks that the KV head count divides the head count. It checks that the dtype and shape make sense and that the `core/limits.zig` ceilings hold. A
  byte-level fault is a parser `Error` and a metadata fault is a loader `Error`.
  The one exception is the context: a file that declares more than the engine
  bound loads at the bound, and the log names both numbers.
- The download is host-pinned and fail-safe. The host is fixed to huggingface.co. The organization and repository names hold no slash. The file name is reduced to its basename and must end in `.gguf`. The on-disk path therefore cannot escape `<root>`. A non-identity content encoding fails the fetch instead of writing a compressed blob. The redirect chain is bounded. The body lands in a `<file>.partial` temporary that only a success renames. An
  interrupted fetch therefore never truncates a good GGUF.
- The map is read-only and whole-file. There are no partial maps and no writes.
  Prefault against lazy is a load-time option, not a second surface, and the
  portability fallback belongs to the standard library.
- `Loaded` owns the map, and its `close` is the single teardown for the whole
  chain. Every `TensorRef.tensor.data` borrows into that region, so the
  references stay valid exactly as long as the `Loaded` value lives.
- The values that cross out are `ModelCard`, `[]TensorRef`, and the `Parsed` KV
  pairs. All three are plain immutable structs. The map is the one owned
  resource that crosses, and `Loaded` holds it.

## Where to start

1. `gguf/parser.zig`: the byte format and the `Parsed` value. It has no I/O, so it reads on its own, and the layout diagram in its header is the specification.
2. `mmap.zig`: the read-only whole-file map and the prefault against lazy split.
3. `gguf/loader.zig`: how the map and the parser compose into a `ModelCard` and a tensor list, and where the semantic and ceiling asserts live.
4. `download.zig`: the standalone fetch path. Read it last, because it reaches the others only through the file on disk.
