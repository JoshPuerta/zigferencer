# OS

This directory holds every OS condition of the engine: each `builtin.os.tag` test and each call
through `std.posix`, `std.os.linux`, or `std.os.windows`. It is the lowest layer of the tree. A
file here imports only `std`, `builtin`, and the other files of this directory. The tidy test
checks both rules.

## Two forms

- A small concern has one file. The file holds one public function for each operation, and a
  comptime `switch` on the OS selects a private function with an OS suffix, such as
  `availableBytesLinux` or `availableBytesWindows`. The `else` case gives the result of an OS with
  no implementation: null, false, or nothing.
- The serve socket calls have one file for each OS. `socket_wait.zig` selects one file at compile
  time, and a comptime check asserts that the selected file declares every function with the
  expected type. An OS file can take a body of another OS file by name. The Windows file takes the
  waits, the probe, the keepalive, and the drains of `socket_wait_none.zig`.

`builtin.os.tag` is fixed at compile time, so Zig never analyzes the case of another OS.

## Files

| File | Holds |
| --- | --- |
| `host_memory.zig` | The free RAM and the total RAM of the host. |
| `host_cores.zig` | The physical core count of the host. |
| `host_cpu_name.zig` | The CPU name that the OS states, for a CPU with no CPUID name. |
| `host_gpu.zig` | `Gpu`: whether the host can drive the vulkan backends, and its label. |
| `file_map.zig` | The populate flag, the heap-copy test, the prefault, and the refusal text of a file map. |
| `console.zig` | The UTF-8 output code page of a console. |
| `process.zig` | The OS name, the process id, the resident memory, and the open files. |
| `threads.zig` | The stack size of a pool worker, and the thread name in the kernel. |
| `socket_wait.zig` | The selector of the serve socket calls, with their result types. |
| `socket_wait_linux.zig` | The socket calls on Linux: the std read and write, the wake by shutdown, the wait with a deadline, the send and drains that do not block, the hang-up probe, and the keepalive. |
| `socket_wait_windows.zig` | The socket calls on Windows: the read and write as requests of the project, and the wake that cancels them. No deadline, no probe, and no keepalive. |
| `socket_wait_none.zig` | The socket calls on every other OS: the std read and write, the wake by shutdown, no deadline, no probe, and no keepalive. |
| `windows_kernel32_declarations.zig` | The kernel32 functions that the standard library does not declare. |
