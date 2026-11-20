//! The CPU name of an x86_64 host from the CPUID brand string, on every OS.
//!
//! Sources. CPUID leaf 0x80000000 gives the highest extended leaf. Leaves 0x80000002 to 0x80000004
//! give 16 bytes each, in the register order eax, ebx, ecx, edx, little endian.
//!
//! Comptime. Asserts keep the reader on x86_64 and keep the three brand leaves at 48 bytes.
//!
//! Use. A caller reaches `readCpuBrandString` only under `builtin.cpu.arch == .x86_64`.

const std = @import("std");
const builtin = @import("builtin");

// -- constants --------------------------------------------------------------

const leaf_highest_extended: u32 = 0x8000_0000;
const leaf_brand_first: u32 = 0x8000_0002;
const leaf_brand_last: u32 = 0x8000_0004;

// -- types ------------------------------------------------------------------

const Registers = struct { eax: u32, ebx: u32, ecx: u32, edx: u32 };

// -- api --------------------------------------------------------------------

/// The brand string of the CPU in `buf`, without the NUL padding and the outer spaces. Null when
/// the CPU has no brand leaves or the string is empty.
pub fn readCpuBrandString(buf: *[48]u8) ?[]const u8 {
    comptime std.debug.assert(builtin.cpu.arch == .x86_64);
    comptime std.debug.assert((leaf_brand_last - leaf_brand_first + 1) * 16 == 48);

    if (cpuid(leaf_highest_extended).eax < leaf_brand_last) return null;

    var leaf = leaf_brand_first;
    while (leaf <= leaf_brand_last) : (leaf += 1) {
        const regs = cpuid(leaf);
        const part = buf[(leaf - leaf_brand_first) * 16 ..][0..16];
        std.mem.writeInt(u32, part[0..4], regs.eax, .little);
        std.mem.writeInt(u32, part[4..8], regs.ebx, .little);
        std.mem.writeInt(u32, part[8..12], regs.ecx, .little);
        std.mem.writeInt(u32, part[12..16], regs.edx, .little);
    }

    const text = std.mem.trim(u8, std.mem.sliceTo(buf, 0), " ");
    if (text.len == 0) return null;

    std.debug.assert(text.len <= buf.len);
    std.debug.assert(text[0] != ' ');
    std.debug.assert(text[text.len - 1] != ' ');
    return text;
}

// -- helpers ----------------------------------------------------------------

/// This copies the `cpuid` function of `std/zig/system/x86.zig`, without its C backend branch.
fn cpuid(leaf: u32) Registers {
    var eax: u32 = undefined;
    var ebx: u32 = undefined;
    var ecx: u32 = undefined;
    var edx: u32 = undefined;
    asm volatile ("cpuid"
        : [_] "={eax}" (eax),
          [_] "={ebx}" (ebx),
          [_] "={ecx}" (ecx),
          [_] "={edx}" (edx),
        : [_] "{eax}" (leaf),
          [_] "{ecx}" (@as(u32, 0)),
    );

    return .{ .eax = eax, .ebx = ebx, .ecx = ecx, .edx = edx };
}

// -- tests ------------------------------------------------------------------

test "readCpuBrandString: a non-empty name with no outer spaces" {
    if (builtin.cpu.arch != .x86_64) return error.SkipZigTest;

    var buf: [48]u8 = undefined;
    const name = readCpuBrandString(&buf) orelse return error.TestUnexpectedResult;
    try std.testing.expect(name.len >= 1);
    try std.testing.expect(name.len <= 48);
    try std.testing.expect(name[0] != ' ');
    try std.testing.expect(name[name.len - 1] != ' ');
}
