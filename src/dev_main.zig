//! Entry point of `zigferencer-dev`: the release commands plus the developer
//! commands. The file imports the engine as the `zigferencer` module and not by
//! a relative path. A relative import puts a `src/` file into two modules, and
//! Zig rejects that.

const std = @import("std");
const zigferencer = @import("zigferencer");

pub const std_options = zigferencer.std_options;

pub fn main(init: std.process.Init) !void {
    return zigferencer.entry(init, .dev);
}
