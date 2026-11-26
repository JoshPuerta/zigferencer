//! Compute-kernel wrappers for the Vulkan op backend, one struct per shader. This file is the
//! table of contents: it re-exports the names of `kernels/` that callers outside it use.
//!
//! Comptime. A build gate refuses non-Vulkan builds.
//!
//! Ownership. Each wrapper owns its shader module, descriptor set layout, pipeline layout, and
//! compute pipeline, but not the buffers. A caller hands it data, and the wrapper records the
//! dispatch. Each kernel struct declares its own `requires`, because the variants span more than
//! one tolerance class.
//!
//! Shader headers. The header of each file in `shaders/` owns the math, the schedule, the memory
//! table, and the bindings of its kernel. A wrapper here states what the host must supply: a
//! precondition or a tolerance. See `docs/gpu-rdna2.md` section 5 for the wave32 pin of every
//! pipeline.
//!
//! Two paths. The per-op wrappers use host-visible coherent memory, one descriptor set per kernel,
//! and queue-wait-idle as the only synchronization. The fused runners hold device-local weights,
//! pre-bound descriptor sets, and one submit per token, and they call the record-only forms. The
//! driver holds a descriptor update until the next submit of a command buffer that reads the set.

const build_options = @import("build_options");
const spirv = @import("kernels/spirv.zig");
const pipeline = @import("kernels/pipeline.zig");
const memory = @import("kernels/memory.zig");
const submit = @import("kernels/submit.zig");
const weights = @import("kernels/weights.zig");
const elementwise = @import("kernels/elementwise.zig");
const rope = @import("kernels/rope.zig");
const attention = @import("kernels/attention.zig");
const linear = @import("kernels/linear.zig");
const gdn = @import("kernels/gdn.zig");

comptime {
    if (!build_options.use_vulkan) {
        @compileError("ops_vulkan/kernels.zig was included in a non-vulkan build. " ++
            "Either build with -Druntime=vulkan, " ++
            "or fix the caller to gate on build_options.use_vulkan.");
    }
}

// -- api --------------------------------------------------------------------

// The shader blobs and their names.
pub const spv_names = spirv.spv_names;
pub const SpvName = spirv.SpvName;

// The pipeline chain and the descriptor helpers.
pub const Pipeline = pipeline.Pipeline;
pub const createStorageBufferLayout = pipeline.createStorageBufferLayout;
pub const allocateOnePrimaryCmdBuffer = pipeline.allocateOnePrimaryCmdBuffer;
pub const updateStorageBindings = pipeline.updateStorageBindings;

// The host buffers, the device buffers, the KV store, and the VRAM arena.
pub const HostBuffer = memory.HostBuffer;
pub const DeviceBuffer = memory.DeviceBuffer;
pub const KvStore = memory.KvStore;
pub const VramArena = memory.VramArena;
pub const hostBufferCount = memory.hostBufferCount;

// The one-shot submits and the staged swaps.
pub const OneShot = submit.OneShot;
pub const Copy = submit.Copy;
pub const SubmitPhase = submit.SubmitPhase;
pub const SubmitLabel = submit.SubmitLabel;
pub const Recorded = submit.Recorded;
pub const SubmitError = submit.SubmitError;
pub const CopyRecorder = submit.CopyRecorder;
pub const SwapSplit = submit.SwapSplit;
pub const StageSide = submit.StageSide;
pub const stagingSubmitCount = submit.stagingSubmitCount;
pub const stagingWaitCheck = submit.stagingWaitCheck;
pub const drainStaging = submit.drainStaging;
pub const submitOnce = submit.submitOnce;
pub const runStagingCopies = submit.runStagingCopies;
pub const walkStage = submit.walkStage;

// The resident weights and their upload.
pub const q4_0_block_bytes = weights.q4_0_block_bytes;
pub const Q80Weight = weights.Q80Weight;
pub const UploadStage = weights.UploadStage;

// The element-wise and per-row kernels.
pub const VecAdd = elementwise.VecAdd;
pub const ResidualAdd = elementwise.ResidualAdd;
pub const SigmoidGate = elementwise.SigmoidGate;
pub const Rmsnorm = elementwise.Rmsnorm;
pub const Swiglu = elementwise.Swiglu;
pub const RmsnormBatch = elementwise.RmsnormBatch;
pub const QwenQkNorm = elementwise.QwenQkNorm;

// The rotary position kernels.
pub const Rope = rope.Rope;
pub const RopeMulti = rope.RopeMulti;
pub const RopeMultiBatch = rope.RopeMultiBatch;

// The attention kernels, the attention kernel set, and the KV commit.
pub const Attention = attention.Attention;
pub const AttentionSplit = attention.AttentionSplit;
pub const AttnOrderClass = attention.AttnOrderClass;
pub const AttnShape = attention.AttnShape;
pub const SplitAttnKernels = attention.SplitAttnKernels;
pub const AttnKernels = attention.AttnKernels;
pub const AttentionBatch = attention.AttentionBatch;
pub const AttentionMultiStream = attention.AttentionMultiStream;
pub const KvCommitF16 = attention.KvCommitF16;
pub const kvCommitF16OneShot = attention.kvCommitF16OneShot;

// The linear kernels.
pub const LinearQ80 = linear.LinearQ80;
pub const LinearQ80Reduce = linear.LinearQ80Reduce;
pub const LinearQ40Reduce = linear.LinearQ40Reduce;
pub const LinearQ80Subgroup = linear.LinearQ80Subgroup;
pub const LinearQ80Batch = linear.LinearQ80Batch;
pub const LinearQ80Gemm = linear.LinearQ80Gemm;
pub const LinearQ40Batch = linear.LinearQ40Batch;

// The Gated DeltaNet kernels.
pub const GdnDecode = gdn.GdnDecode;
pub const GdnChunkParallel = gdn.GdnChunkParallel;

// -- tests ------------------------------------------------------------------

test {
    _ = spirv;
    _ = pipeline;
    _ = memory;
    _ = submit;
    _ = weights;
    _ = elementwise;
    _ = rope;
    _ = attention;
    _ = linear;
    _ = gdn;
}
