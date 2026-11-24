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
  expected type and every capability constant as a `bool`.

`builtin.os.tag` is fixed at compile time, so Zig never analyzes the case of another OS.

## Files

| File | Holds |
| --- | --- |
| `host_memory.zig` | The free RAM and the total RAM of the host. |
| `host_cores.zig` | The physical core count of the host. |
| `host_cpu_name.zig` | The CPU name that the OS states, for a CPU with no CPUID name. |
| `host_gpu.zig` | `Gpu`: whether the host can drive the vulkan backends, and its label. The Vulkan loader file name of the OS, or null. |
| `dynamic_library.zig` | Open a shared library by name, and find a symbol. |
| `file_map.zig` | The populate flag, the heap-copy test, the prefault, and the refusal text of a file map. |
| `console.zig` | The UTF-8 output code page of a console. |
| `process.zig` | The OS name, the process id, the resident memory, and the open files. |
| `threads.zig` | The stack size of a pool worker, and the thread name in the kernel. |
| `socket_wait.zig` | The selector of the serve socket calls, with their result types and capability constants. |
| `socket_wait_linux.zig` | The socket calls on Linux: the std read, write, and shutdown, and `poll` for each wait. |
| `socket_wait_windows.zig` | The socket calls on Windows: AFD requests of the project, and a wake that cancels them. |
| `socket_wait_none.zig` | The socket calls on every other OS: the std read, write, and shutdown, with no deadline. |
| `windows_kernel32_declarations.zig` | The kernel32 functions that the standard library does not declare. |
| `windows_afd_declarations.zig` | The AFD poll record and the hang-up event bits that the standard library does not declare. |
