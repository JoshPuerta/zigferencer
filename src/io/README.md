# I/O

This directory is the edge between the engine and the outside world. It fetches a model over HTTPS,
maps the file read-only, and parses it into the typed values that the runtime reads. `download.zig`
and `mmap.zig` hold the operating-system resources. `gguf/parser.zig` is pure, and `gguf/loader.zig`
joins the map, the parser, and the semantic checks. The `gguf/` directory has no README of its own.

## Files

| File | Holds |
| --- | --- |
| `download.zig` | The fetch of a model GGUF to the model store, and the remote size of a file. |
| `mmap.zig` | `Mapped`, the read-only whole-file map, prefaulted or lazy. |
| `gguf/parser.zig` | The GGUF v3 byte parser: a byte slice in, `Parsed` out. |
| `gguf/loader.zig` | `load`: the map, the parser, and the checks joined into `Loaded`. |

## How the pieces fit

The directory has two paths that meet only on disk. `fetch` writes a file, and a later `load`
reads it. The values on the load path are these:

| Value | File | Holds | Made by | Read by |
| --- | --- | --- | --- | --- |
| `Mapped` | `mmap.zig` | The bytes of the whole file | `openReadOnly`, `openReadOnlyLazy` | `loader`, `runtime/prompt_cache` |
| `Parsed` | `gguf/parser.zig` | The header, the `MetadataPair` values, the `TensorInfo` rows, the data offset | `parser.parse` | `loader`, `core/tokenizer/vocab` |
| `ModelCard` | `gguf/loader.zig` | The dimensions, the token ids, the chat template, and the `GdnParams` of Qwen | `loader.load` | `resolve/resolve`, `runtime/`, `cli/` |
| `TensorRef` | `gguf/loader.zig` | A tensor name and a `Tensor` that borrows the map | `loader.load` | `runtime/tensor_lookup`, the runners |
| `Loaded` | `gguf/loader.zig` | The `Mapped`, the `Parsed`, the `ModelCard`, and the `[]TensorRef` | `loader.load` | `cli/`, `dev/`, `resolve/family` |

The load path:

```
  core/registry.Model --path--> loader.load(path)
                                     |
                                     v
                               mmap.openReadOnly[Lazy] --> Mapped
                                     |  bytes
                                     v
                               parser.parse --> Parsed { header, metadata, tensors }
                                     |
                                     v  architecture keys, limits, dtype, shape
                               Loaded { map, parsed, card, tensors }
                                     |
              +----------------------+-----------------------+
              v                      v                       v
      resolve/resolve         runtime/ runners        core/tokenizer/vocab
      (ModelCard)             (TensorRef)             (Parsed metadata --> Vocab)
```

The fetch path:

```
  core/registry.Model --> download.fetchModel --> <file>.partial
                                                       |  rename on success
                                                       v
                                       <root>/<local_path>/<file_name>
  cli/fetch_cmd -------> download.remoteSize     (the size preview of fetch --dry-run)
```

Only `loader.zig` knows the whole chain. The parser never sees a file, and `mmap.zig` never sees
the GGUF structure. `core/registry.zig` computes the path that both paths use. `loader` leaves the
`Vocab` build to its consumer and surfaces `vocab_size` in the `ModelCard`.

## Rules

- The parser and the loader validate in pairs, and neither trusts the other. The parser checks the
  `"GGUF"` identifier and the version. It checks that the alignment is a power of two and that the
  counts do not overflow. It checks that the tensor offsets are ordered and aligned, and that a
  name is at most 64 bytes. The loader checks that every architecture key resolves and that no card
  dimension is zero. It checks that the KV head count divides the head count, that the dtype and
  shape make sense, and that the `core/limits.zig` ceilings hold. A byte-level fault is a parser
  `Error` and a metadata fault is a loader `Error`.
- A file that declares a context past `limits.max_model_ctx` loads at that bound, and the log names
  both numbers. A RoPE scaling the engine does not apply fails with `error.UnsupportedRopeScaling`.
- The download is host-pinned and fail-safe. The host is fixed to huggingface.co. The organization
  and repository names hold no slash. The file name is reduced to its basename and must end in
  `.gguf`, so the on-disk path cannot escape `<root>`. A non-identity content encoding fails the
  fetch. The redirect chain is bounded. The body lands in a `<file>.partial` temporary that only a
  success renames, so an interrupted fetch never truncates a good GGUF.
- The map is read-only and whole-file. There are no partial maps and no writes. Prefault against
  lazy is a load-time option, not a second surface. The lazy map opens a multi-gigabyte file in
  milliseconds for a reader that needs the metadata alone.
- `Loaded` owns the map, and its `close` is the single teardown for the whole chain. Every
  `TensorRef.tensor.data` borrows into that region, so the references stay valid exactly as long as
  the `Loaded` value lives.
- The values that cross out are `ModelCard`, `[]TensorRef`, and the `Parsed` metadata pairs. All three
  are plain immutable structs. The map is the one owned resource that crosses, and `Loaded` holds
  it.
