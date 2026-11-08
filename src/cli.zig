//! The CLI layer. `main.zig` enters through `command`. A file under `src/dev/` reads
//! the building blocks below through `@import("zigferencer").cli`, and it reads
//! nothing else of this layer, so this list is the whole boundary between the two
//! executables.

pub const command = @import("cli/command.zig");
pub const context = @import("cli/context.zig");
pub const args = @import("cli/args.zig");
pub const flag_groups = @import("cli/flag_groups.zig");
pub const select = @import("cli/select.zig");
pub const style = @import("cli/style.zig");
pub const boot_resources = @import("cli/boot/boot_resources.zig");
pub const boot_lines = @import("cli/boot/boot_lines.zig");
