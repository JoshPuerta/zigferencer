# OS

This directory holds every OS condition of the engine: each `builtin.os.tag` test and each call
through `std.posix`, `std.os.linux`, or `std.os.windows`. It is the lowest layer of the tree. A
file here imports only `std`, `builtin`, and the other files of this directory. The tidy test
checks both rules.

## Two forms

- A small concern has one file. The file holds one public function for each operation, and a
  comptime `switch` on the OS selects a private function with an OS suffix, such as
  `availableBytesLinux`. The `else` case gives the result of an OS with
  no implementation: null, false, or nothing.
- The serve socket calls have one file for each OS. `socket_wait.zig` selects one file at compile
  time, and a comptime check asserts that the selected file declares every function with the
  expected type.

`builtin.os.tag` is fixed at compile time, so Zig never analyzes the case of another OS.

## Files

| File | Holds |
| --- | --- |
| `host_memory.zig` | The free RAM and the total RAM of the host. |
| `host_cores.zig` | The physical core count of the host. |
| `host_cpu_name.zig` | The CPU name that the OS states, for a CPU with no CPUID name. |
| `host_gpu.zig` | `Gpu`: whether the host can drive the vulkan backends, and its label. |
| `process.zig` | The OS name, the process id, the resident memory, and the open files. |
| `threads.zig` | The thread name in the kernel. |
| `socket_wait.zig` | The selector of the serve socket calls, with their result types. |
| `socket_wait_linux.zig` | The socket calls on Linux: the wait with a deadline, the send and drains that do not block, the hang-up probe, and the keepalive. |
| `socket_wait_none.zig` | The socket calls on every other OS: no deadline, no probe, and no keepalive. |
