//! The physical core count of the host, which sets the default thread count.
//!
//! Platforms. Linux reads the distinct `thread_siblings_list` texts in sysfs, because SMT siblings
//! share one text. Another OS gives null.
//!
//! Comptime. `physicalCoreCount` takes its cap at comptime, so its buffers live on the stack and
//! the caller keeps one cap for every OS.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

const log = std.log.scoped(.parallel);

// -- constants --------------------------------------------------------------

/// The longest `thread_siblings_list` text that the Linux read keeps.
const sibling_text_max: usize = 48;

// -- api --------------------------------------------------------------------

/// The physical cores of the host, from at most `max_cores` logical CPUs. It returns null when the
/// read fails, or when the cap cuts the read, which logs.
pub fn physicalCoreCount(io: Io, comptime max_cores: u32) ?u32 {
    comptime std.debug.assert(max_cores >= 1);

    return switch (builtin.os.tag) {
        .linux => physicalCoreCountLinux(io, max_cores),
        else => null,
    };
}

// -- helpers ----------------------------------------------------------------

/// Count the physical cores from the sibling list of each logical CPU in sysfs.
fn physicalCoreCountLinux(io: Io, comptime max_cores: u32) ?u32 {
    var texts: [max_cores][sibling_text_max]u8 = undefined;
    var lists: [max_cores][]const u8 = undefined;
    var n: u32 = 0;
    const cwd = Io.Dir.cwd();
    while (n < max_cores) : (n += 1) {
        var path_buf: [96]u8 = undefined;
        const path = std.fmt.bufPrint(
            &path_buf,
            "/sys/devices/system/cpu/cpu{d}/topology/thread_siblings_list",
            .{n},
        ) catch unreachable;
        const text = cwd.readFile(io, path, &texts[n]) catch |err| switch (err) {
            error.FileNotFound => break,
            else => return null,
        };
        if (text.len == sibling_text_max) {
            log.info(
                "cpu{d} sibling list is longer than {d} bytes. " ++
                    "The thread default uses the logical estimate",
                .{ n, sibling_text_max },
            );
            return null;
        }

        lists[n] = text;
    } else {
        log.info(
            "the host has more than {d} CPUs. The thread default uses the logical estimate",
            .{max_cores},
        );
        return null;
    }

    if (n == 0) return null;

    std.debug.assert(n <= max_cores);
    return countCores(lists[0..n]);
}

/// The number of distinct sibling-list texts in `lists`, with surrounding white space ignored.
fn countCores(lists: []const []const u8) u32 {
    std.debug.assert(lists.len >= 1);

    var distinct: u32 = 0;
    for (lists, 0..) |list, i| {
        const text = std.mem.trim(u8, list, " \t\r\n");
        const seen = for (lists[0..i]) |earlier| {
            if (std.mem.eql(u8, std.mem.trim(u8, earlier, " \t\r\n"), text)) break true;
        } else false;

        if (!seen) {
            distinct += 1;
        }
    }

    std.debug.assert(distinct >= 1);
    std.debug.assert(distinct <= lists.len);
    return distinct;
}

// -- tests ------------------------------------------------------------------

test "countCores: SMT pairs, single threads, and comma lists" {
    // An SMT-2 host lists each pair twice, once per logical CPU.
    const smt = [_][]const u8{ "0-1\n", "0-1\n", "2-3\n", "2-3\n", "4-5\n", "4-5\n" };
    try std.testing.expectEqual(@as(u32, 3), countCores(&smt));
    // A host without SMT lists each CPU alone.
    const solo = [_][]const u8{ "0\n", "1\n", "2\n", "3\n" };
    try std.testing.expectEqual(@as(u32, 4), countCores(&solo));
    // A host that numbers the siblings apart writes a comma list.
    const comma = [_][]const u8{ "0,16\n", "1,17\n", "0,16\n", "1,17\n" };
    try std.testing.expectEqual(@as(u32, 2), countCores(&comma));
}

test "physicalCoreCount: the count is nonzero and stays inside the logical CPUs" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;

    const cores = physicalCoreCount(std.testing.io, 512) orelse return error.TestUnexpectedResult;
    try std.testing.expect(cores > 0);
    try std.testing.expect(cores <= try std.Thread.getCpuCount());
}
