//! An immutable tensor view, the boundary value modules pass to each other.
//! Storage (mmap, arena, scratch slab) lives outside, owned by the caller. Ops
//! never allocate, because the caller supplies the output buffer. `data` stays
//! a borrow (`[]const u8`) and allocation stays at the edges, so the kernels
//! compose with no hidden lifecycle and each op tests on its own.
//!
//! GGUF stores 2D weight shapes REVERSED vs PyTorch: a linear weight that is
//! PyTorch [out_features, in_features] arrives here as shape = [in, out].
//! Memory is row-major by OUTPUT row, so the linear kernels read w[j*in + k]
//! for output j, input k (see `ops/naive.zig`):
//!
//!   shape = [in, out]                 memory (row-major, one row per output)
//!                                       +------------------------------+
//!                              row 0 -> | w[0,0] w[0,1] ... w[0,in-1]  |
//!                              row 1 -> | w[1,0] w[1,1] ...            |
//!                                ...    |          (out rows)          |
//!                                       +------------------------------+
//!
//! Rank is fixed at 4 (`limits.max_n_dims`). `data` is the raw byte view (use
//! `bytesAsSlice` for the f32 view of an f32 tensor).

const std = @import("std");
const dtype = @import("dtype.zig");
const limits = @import("limits.zig");

pub const Tensor = struct {
    dtype: dtype.GgmlType,
    n_dims: u8,
    shape: [limits.max_n_dims]u32,
    /// Only `byte_stride[0]` has a meaning: the bytes of one row of `shape[0]`
    /// elements. No reader uses the other entries.
    byte_stride: [limits.max_n_dims]u32,
    /// Caller-owned. Tensor never frees.
    data: []const u8,

    /// Total element count (product of the active dims).
    pub fn numel(self: Tensor) usize {
        var n: usize = 1;
        for (self.shape[0..self.n_dims]) |d| {
            n *= d;
        }

        return n;
    }

    /// Bytes `data` must hold for this dtype and shape.
    pub fn bytesRequired(self: Tensor) usize {
        return dtype.bytesForNumel(self.dtype, self.numel());
    }

    /// Assert rank, positive dims, and that `data` is large enough.
    pub fn assertValid(self: Tensor) void {
        std.debug.assert(self.n_dims >= 1);
        std.debug.assert(self.n_dims <= limits.max_n_dims);

        for (self.shape[0..self.n_dims]) |d| {
            std.debug.assert(d > 0);
        }

        std.debug.assert(self.data.len >= self.bytesRequired());
    }
};

// -- tests ------------------------------------------------------------------
