//! Whether the host can drive the vulkan backends, and a label for `info`. The file links no GPU
//! library, so a CPU build can still answer.
//!
//! Platforms. On Linux three signals decide, in order: the PCI vendor id of the primary DRM card on
//! native Linux, the paravirtual compute device `/dev/dxg` on WSL2 (the real GPU is hidden there),
//! and a render node without a card0 vendor file. Another OS reports no GPU.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const Allocator = std.mem.Allocator;

// -- types ------------------------------------------------------------------

/// Best-effort GPU presence. `present` drives the vulkan op-backends' host eligibility in `info`.
/// `label` names the vendor when sysfs exposes it.
pub const Gpu = struct {
    present: bool,
    label: []const u8,
};

// -- api --------------------------------------------------------------------

/// Detect a GPU. A vendor label that no table entry names lands on `arena`.
pub const detect = switch (builtin.os.tag) {
    .linux => detectLinux,
    else => detectNone,
};

// -- helpers ----------------------------------------------------------------

fn detectLinux(io: Io, arena: Allocator) Gpu {
    const cwd = Io.Dir.cwd();

    var vbuf: [32]u8 = undefined;
    if (cwd.readFile(io, "/sys/class/drm/card0/device/vendor", &vbuf)) |slice| {
        return .{ .present = true, .label = vendorName(arena, std.mem.trim(u8, slice, " \t\r\n")) };
    } else |_| {}

    if (exists(io, "/dev/dxg")) return .{ .present = true, .label = "WSL2 (paravirtual)" };
    if (exists(io, "/dev/dri/renderD128")) {
        return .{ .present = true, .label = "present (DRI render node)" };
    }

    return .{ .present = false, .label = "none detected" };
}

fn vendorName(arena: Allocator, vendor: []const u8) []const u8 {
    if (std.mem.eql(u8, vendor, "0x1002")) return "AMD";
    if (std.mem.eql(u8, vendor, "0x10de")) return "NVIDIA";
    if (std.mem.eql(u8, vendor, "0x8086")) return "Intel";
    if (std.mem.eql(u8, vendor, "0x1414")) return "Microsoft (virtual)";

    return arena.dupe(u8, vendor) catch "unknown vendor";
}

fn exists(io: Io, path: []const u8) bool {
    var cwd = Io.Dir.cwd();
    cwd.access(io, path, .{}) catch return false;
    return true;
}

fn detectNone(io: Io, arena: Allocator) Gpu {
    _ = io;
    _ = arena;
    return .{ .present = false, .label = "unknown (non-Linux)" };
}

// -- tests ------------------------------------------------------------------

test "detect runs without crashing on the test host" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    _ = detect(std.testing.io, arena.allocator());
}
