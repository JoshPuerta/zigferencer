# Utils

This directory holds the general services of the process: the clock arithmetic, the log, the
host probe, the Perfetto encoder, and the property test driver. These files serve any program.
None of them names an op, a model, or a runner of the inference engine. No file here imports a
file outside this directory.

## Files

| File | Holds |
| --- | --- |
| `time.zig` | Elapsed-time arithmetic. |
| `log.zig` | The process log, gated at run time. |
| `hwinfo.zig` | The host hardware card. |
| `perfetto.zig` | The protobuf encoder and the nesting walk. |
| `prop.zig` | The property test driver. |
