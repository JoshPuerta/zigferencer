//! The host hardware capabilities relevant to inference: arch and OS strings, CPU model, core
//! count, total RAM, and a bool-per-feature SIMD card.
//!
//! Role. The card is a boundary value, a plain struct with no hidden state. It feeds
//! `op_requires.isMetByHost`, which `info` uses to label eligible backends, and `info` prints it.
//!
//! GPU detection. It links no GPU library, so a CPU build can still answer whether the host can
//! drive the vulkan backends. Three signals decide, in order: the PCI vendor id of the primary
//! DRM card on native Linux, the paravirtual compute device `/dev/dxg` on WSL2 (the real GPU is
//! hidden there), and a render node without a card0 vendor file.
//!
//! Sources:
//!   - `builtin.cpu.features` for the SIMD flags. That is the BUILD target's
//!     set. On `-Dtarget=native`, the default, it matches the host. A binary
//!     copied elsewhere reports the build host's features, because no runtime
//!     cpuid runs.
//!   - `/proc/cpuinfo` (Linux, best effort) for the CPU model name.
//!   - `/proc/meminfo` (Linux) for total RAM, and for the available RAM that
//!     the hybrid placement check reads.
//!   - `std.Thread.getCpuCount` for online cores.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const Allocator = std.mem.Allocator;

// -- types ------------------------------------------------------------------

/// Per-ISA SIMD feature flags (x86 + ARM), one bool each.
pub const SimdFeatures = struct {
    sse2: bool = false,
    sse4_1: bool = false,
    sse4_2: bool = false,
    avx: bool = false,
    avx2: bool = false,
    fma: bool = false,
    f16c: bool = false,
    avx512f: bool = false,
    avx512vnni: bool = false,
    avx_vnni: bool = false,
    neon: bool = false,
    dotprod: bool = false,
    fp16: bool = false,
    sve: bool = false,
};

/// Best-effort GPU presence. `present` drives the vulkan op-backends' host eligibility in `info`.
/// `label` names the vendor when sysfs exposes it.
pub const Gpu = struct {
    present: bool,
    label: []const u8,
};

/// Host hardware snapshot.
pub const HwCard = struct {
    arch: []const u8,
    os: []const u8,
    cpu_model: ?[]const u8,
    cores_total: usize,
    ram_total_bytes: ?u64,
    simd: SimdFeatures,
    gpu: Gpu,
};

// -- api --------------------------------------------------------------------

/// Build a HwCard. Allocations land on `arena` (cpu_model + an unknown gpu label).
pub fn detect(io: Io, arena: Allocator) HwCard {
    return .{
        .arch = @tagName(builtin.cpu.arch),
        .os = @tagName(builtin.os.tag),
        .cpu_model = readCpuModel(io, arena),
        .cores_total = std.Thread.getCpuCount() catch 1,
        .ram_total_bytes = readMemTotal(io),
        .simd = detectSimd(),
        .gpu = detectGpu(io, arena),
    };
}

/// Read the build target's SIMD feature flags. It is pure (only `builtin.cpu`),
/// so it evaluates at comptime and the simd-probe bakes the result into a
/// static string.
pub fn detectSimd() SimdFeatures {
    var s: SimdFeatures = .{};
    switch (builtin.cpu.arch) {
        .x86, .x86_64 => {
            const F = std.Target.x86.Feature;
            const set = builtin.cpu.features;
            s.sse2 = std.Target.x86.featureSetHas(set, F.sse2);
            s.sse4_1 = std.Target.x86.featureSetHas(set, F.sse4_1);
            s.sse4_2 = std.Target.x86.featureSetHas(set, F.sse4_2);
            s.avx = std.Target.x86.featureSetHas(set, F.avx);
            s.avx2 = std.Target.x86.featureSetHas(set, F.avx2);
            s.fma = std.Target.x86.featureSetHas(set, F.fma);
            s.f16c = std.Target.x86.featureSetHas(set, F.f16c);
            s.avx512f = std.Target.x86.featureSetHas(set, F.avx512f);
            s.avx512vnni = std.Target.x86.featureSetHas(set, F.avx512vnni);
            s.avx_vnni = std.Target.x86.featureSetHas(set, F.avxvnni);
        },
        .aarch64, .aarch64_be => {
            const F = std.Target.aarch64.Feature;
            const set = builtin.cpu.features;
            s.neon = std.Target.aarch64.featureSetHas(set, F.neon);
            s.dotprod = std.Target.aarch64.featureSetHas(set, F.dotprod);
            s.fp16 = std.Target.aarch64.featureSetHas(set, F.fullfp16);
            s.sve = std.Target.aarch64.featureSetHas(set, F.sve);
        },
        else => {},
    }

    return s;
}

/// The RAM the kernel can give a new process without swap, from
/// `MemAvailable:` in `/proc/meminfo`. Null on a platform without the file or
/// the field, so the caller can turn its RAM check off.
pub fn readMemAvailable(io: Io) ?u64 {
    var buf: [meminfo_text_max]u8 = undefined;
    return meminfoBytes(readMeminfo(io, &buf) orelse return null, "MemAvailable");
}

/// Bytes of one `Field:   N kB` line of `/proc/meminfo` text. The match is at a
/// line start, so a field name never matches inside a longer name. A missing
/// field, a malformed line, or a count past a `u64` gives null.
pub fn meminfoBytes(text: []const u8, field: []const u8) ?u64 {
    std.debug.assert(field.len >= 1);
    std.debug.assert(std.mem.findScalar(u8, field, ':') == null);

    var lines = std.mem.splitScalar(u8, text, '\n');
    // Each pass reads one line, so the text length bounds the loop.
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, field)) continue;

        const rest = line[field.len..];
        if (rest.len == 0 or rest[0] != ':') continue;

        var fields = std.mem.tokenizeAny(u8, rest[1..], " \t\r");
        const count = fields.next() orelse return null;
        const unit = fields.next() orelse return null;
        if (fields.next() != null) return null;
        if (!std.mem.eql(u8, unit, "kB")) return null;

        const n_kb = std.fmt.parseInt(u64, count, 10) catch return null;
        return std.math.mul(u64, n_kb, 1024) catch null;
    }

    return null;
}

// -- helpers ----------------------------------------------------------------

/// Detect a GPU from sysfs and devfs signals. See "GPU detection" in the header.
fn detectGpu(io: Io, arena: Allocator) Gpu {
    if (builtin.os.tag != .linux) return .{ .present = false, .label = "unknown (non-Linux)" };

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

fn readCpuModel(io: Io, arena: Allocator) ?[]const u8 {
    if (builtin.os.tag != .linux) return null;

    var buf: [64 * 1024]u8 = undefined;
    const cwd = Io.Dir.cwd();
    const slice = cwd.readFile(io, "/proc/cpuinfo", &buf) catch return null;
    const needle = "model name";
    const idx = std.mem.find(u8, slice, needle) orelse return null;
    const after_needle = slice[idx + needle.len ..];
    const colon = std.mem.findScalar(u8, after_needle, ':') orelse return null;
    const after_colon = after_needle[colon + 1 ..];
    const end = std.mem.findScalar(u8, after_colon, '\n') orelse return null;
    const trimmed = std.mem.trim(u8, after_colon[0..end], " \t");
    return arena.dupe(u8, trimmed) catch null;
}

fn readMemTotal(io: Io) ?u64 {
    var buf: [meminfo_text_max]u8 = undefined;
    return meminfoBytes(readMeminfo(io, &buf) orelse return null, "MemTotal");
}

/// Bytes `/proc/meminfo` holds. The file is about 1.5 KiB, and both fields the
/// engine reads sit in its first lines.
const meminfo_text_max: usize = 4096;

fn readMeminfo(io: Io, buf: *[meminfo_text_max]u8) ?[]const u8 {
    if (builtin.os.tag != .linux) return null;
    const cwd = Io.Dir.cwd();
    return cwd.readFile(io, "/proc/meminfo", buf) catch null;
}

// -- tests ------------------------------------------------------------------

test "detectSimd does not panic for the current target" {
    _ = detectSimd();
}

test "detectGpu runs without crashing on the test host" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    _ = detectGpu(std.testing.io, arena.allocator());
}

test "meminfoBytes: a fixed meminfo sample, a missing field, and malformed lines" {
    const sample =
        \\MemTotal:       24606612 kB
        \\MemFree:        22687856 kB
        \\MemAvailable:   23555960 kB
        \\Buffers:           86880 kB
        \\HugePages_Total:       0
        \\
    ;
    try std.testing.expectEqual(@as(?u64, 24606612 * 1024), meminfoBytes(sample, "MemTotal"));
    try std.testing.expectEqual(@as(?u64, 23555960 * 1024), meminfoBytes(sample, "MemAvailable"));
    try std.testing.expectEqual(@as(?u64, 86880 * 1024), meminfoBytes(sample, "Buffers"));
    // A field name that is a prefix of another line never matches that line.
    try std.testing.expectEqual(@as(?u64, null), meminfoBytes(sample, "Mem"));
    try std.testing.expectEqual(@as(?u64, null), meminfoBytes(sample, "SwapTotal"));
    // A line without a unit, with a wrong unit, or with no digits is malformed.
    try std.testing.expectEqual(@as(?u64, null), meminfoBytes(sample, "HugePages_Total"));
    try std.testing.expectEqual(
        @as(?u64, null),
        meminfoBytes("MemAvailable: 12 MB\n", "MemAvailable"),
    );
    try std.testing.expectEqual(
        @as(?u64, null),
        meminfoBytes("MemAvailable: x1 kB\n", "MemAvailable"),
    );
    try std.testing.expectEqual(@as(?u64, null), meminfoBytes("MemAvailable:\n", "MemAvailable"));
    // A count whose byte figure passes a `u64` gives null, not a wrapped value.
    try std.testing.expectEqual(
        @as(?u64, null),
        meminfoBytes("MemAvailable: 18446744073709551615 kB", "MemAvailable"),
    );
    try std.testing.expectEqual(@as(?u64, null), meminfoBytes("", "MemAvailable"));
}
