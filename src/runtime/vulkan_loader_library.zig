//! The Vulkan loader library, opened at run time. The binary links no Vulkan library, so a GPU
//! build starts on a host with no loader and stops with a message there.
//!
//! Lifetime. The library stays open until the process ends, so every Vulkan function pointer
//! stays valid. Each context opens it again, which gives the same library at the same address.
//!
//! Comptime. A build gate stops the compiler for a target that opens no shared library: an OS with
//! no loader name, or a static musl link.

const std = @import("std");
const vk = @import("vulkan");
const dynamic_library = @import("../os/dynamic_library.zig");
const host_gpu = @import("../os/host_gpu.zig");

const log = std.log.scoped(.gpu);

// -- constants --------------------------------------------------------------

const library_name = host_gpu.vulkan_loader_file orelse
    @compileError("runtime=vulkan supports Linux and Windows targets. Use -Druntime=cpu");

comptime {
    if (host_gpu.vulkan_loader_file != null and !dynamic_library.can_open) {
        @compileError("runtime=vulkan needs a target that opens a shared library. This build " ++
            "links a musl target statically. Use a gnu target, or -Druntime=cpu");
    }
}

/// The one function that the open finds. Every other Vulkan function loads through it.
const entry_point_name = "vkGetInstanceProcAddr";

// -- types ------------------------------------------------------------------

pub const OpenError = error{
    VulkanLoaderLibraryMissing,
    VulkanLoaderEntryPointMissing,
};

// -- api --------------------------------------------------------------------

/// Open the Vulkan loader of the OS and return its `vkGetInstanceProcAddr`. A failure logs the
/// file name.
pub fn open() OpenError!vk.PfnGetInstanceProcAddr {
    var library = dynamic_library.open(library_name) catch {
        log.err("loader: the Vulkan loader " ++ library_name ++ " does not open. " ++
            "Install the Vulkan driver of the GPU, or use a CPU build", .{});
        return error.VulkanLoaderLibraryMissing;
    };
    const symbol = dynamic_library.lookup(&library, entry_point_name) orelse {
        log.err("loader: the Vulkan loader " ++ library_name ++ " has no function " ++
            entry_point_name ++ ". Install the Vulkan driver of the GPU again", .{});
        return error.VulkanLoaderEntryPointMissing;
    };

    log.debug("loader: " ++ library_name, .{});
    return @ptrCast(@alignCast(symbol));
}
