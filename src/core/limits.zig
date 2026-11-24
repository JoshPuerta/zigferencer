//! Project-wide static engine limits. The doc of a constant states what it bounds and why. This
//! header holds the rules that span constants.
//!
//! Comptime. Asserts keep the window and session budget limits consistent with each other.
//!
//! Load bounds. The loader rejects a model that exceeds a structural bound with a typed error at
//! load. It caps a declared context at `max_model_ctx` and logs both numbers. It never truncates in
//! silence.
//!
//! Shader defines. The build feeds each tile constant to glslc, and the kernel wrapper derives its
//! grid from the same constant. So the tile the shader walks and the grid the host dispatches read
//! one number. The doc of each constant names its define.
//!
//! GPU split attention. A decode or prefill row of more than `attn_split_len` positions walks
//! `ceil(t_len / attn_split_len)` runs. A merge pass combines the runs in split order. A shorter
//! row runs a resident kernel. The scratch holds one partial record per row, query head, and split,
//! and its size follows the declared context. At 4 rows (`attn_split_rows`) the largest registry
//! scratch, the Qwen3.5-9B at 262144 positions, takes 64.5 MiB. That is inside the 128 MiB storage
//! range that Vulkan guarantees on every device.
//!
//! KV windows. Each model gets its own ceiling under `max_model_ctx` from `contextCeiling`. With no
//! window flag, `run` and `bench-model` take the larger KV budget. Each serve session takes the
//! smaller one. The costliest KV in the registry, an 8B at f32, keeps windows of 8192 and 512 under
//! these two figures. A smaller KV family gets a deeper window. The model picker shows both windows
//! of each model before a run.
//!
//! Slots. The CPU and hybrid runners cannot evict, so `max_batch` also bounds their physical KV
//! slots. The fused-GPU runners take their slot ceiling from `max_vram_slots`, because VRAM fits
//! far more slots than one tick decodes. A KV budget holds up to that many slots, of which at most
//! `max_batch` decode per tick.
//!
//! OpenAI connector. HTTP and JSON are untrusted input, so each `openai_*` bound is a hard guard at
//! the boundary, never a hint. A request past a limit gets a 4xx, never a truncation or an assert.

const std = @import("std");

// -- constants --------------------------------------------------------------

/// Tensor rank ceiling.
pub const max_n_dims: u8 = 4;

/// Positions one split of the GPU attention walks. The build feeds it to glslc as ATTN_SPLIT. The
/// resident and split shaders size their shared score row with it, 1 KiB at 256.
pub const attn_split_len: u32 = 256;

/// Rows of the split-attention scratch, the split rows that one batched step holds.
pub const attn_split_rows: u32 = 4;

/// Query heads that one workgroup of the GPU split pass serves. The build feeds it as ATTN_QUERIES.
/// One workgroup loads each K and V row once for these heads of a GQA group. G heads take `ceil(G /
/// attn_split_queries)` workgroups. A sweep of 8, 4, and 2 picked 4. At 8 the workgroup LDS halves
/// the resident workgroups. At 2 a group of 4 heads reads K and V twice.
pub const attn_split_queries: u32 = 4;

/// The most query heads that the GPU split pass can serve in one workgroup. The shader keeps one
/// named register set per query, and has sets for 8 queries. Uniform guards pick the sets.
pub const attn_split_queries_max: u32 = 8;

/// The head_dim granule of the GPU split pass, which reads a K row in 16-byte loads of 8 halves.
pub const attn_head_dim_granule: u32 = 8;

/// The widest GQA group (`n_heads / n_kv_heads`) that a CPU attention part kernel holds at once.
/// The kernel keeps one stack score row per head of a block. A wider group walks in blocks of this
/// many heads. Every registered model states a group of 3 or 4.
pub const attn_max_group: u32 = 8;

/// The widest head that the CPU attention part kernels hold in one stack row. The scalar kernel
/// widens an f16 K or V row into that row. A wider head walks in chunks of it. The vector kernel
/// falls back to the scalar kernel for a wider head. The GPU split pass stages the q row in shared
/// memory of this width, as `ATTN_D_MAX`. It takes no wider head.
pub const attn_max_head_dim: u32 = 256;

/// Positions one tile of the CPU attention row holds. The kernel keeps the scores of one tile in a
/// stack row of `[attn_max_group][attn_cpu_tile]f32`, 64 KiB at 2048. So the tile bounds the kernel
/// stack, and no score buffer grows with the window.
pub const attn_cpu_tile: u32 = 2048;

/// The part ceiling of one CPU attention row. A row splits into `min(attn_cpu_parts, tiles)` parts
/// of contiguous tiles. One KV group over one part is the unit of thread parallelism. The count is
/// a function of `t_len` alone, so the result does not depend on the thread count. The partial
/// buffer holds one record per query head and part, so this ceiling sets its size.
pub const attn_cpu_parts: u32 = 16;

/// Output-tile geometry of the batched Q8_0 matmul: 16 batch rows by 32 output features. The build
/// feeds B_TILE and N_TILE of `linear_q8_0_batch.comp`.
pub const q8_batch_b_tile: u32 = 16;

/// The workgroup width, so it must stay a multiple of 32, the wave width the kernel pins.
pub const q8_batch_n_tile: u32 = 32;

/// The same geometry for the Q4_0 prefill matmul: 8 batch rows by 32 output features. A nibble
/// weight holds half the bytes per column, so the kernel carries half the batch rows of Q8_0.
pub const q4_batch_b_tile: u32 = 8;

pub const q4_batch_n_tile: u32 = 32;

/// Output-tile geometry of the register-blocked Q8_0 GEMM: 64 by 64 per workgroup of 256 threads.
/// The build feeds BM and BN of `linear_q8_0_gemm.comp`. A workgroup stages one weight tile through
/// shared memory, and every one of the 64 rows reuses it.
pub const gemm_m_tile: u32 = 64;

pub const gemm_n_tile: u32 = 64;

/// CSUB of `gdn_chunk_scan.comp`: the sub-chunk of tokens of one GDN prefill scan solve step. It
/// aligns to the chunk start. A change of it changes the summation order of every output.
pub const gdn_scan_sub: u32 = 16;

/// W of `gdn_chunk_scan.comp`: the state columns of one workgroup and its lane count. It must stay
/// a multiple of 32, the RDNA2 wave width.
pub const gdn_scan_cols: u32 = 32;

/// DH of the GDN kernels: the value-head width the decode and chunk kernels support. Each scan lane
/// holds its whole state column in registers, which needs a compile-time width.
pub const gdn_head_dim: u32 = 128;

/// The most value heads of one GDN key head. Each lane of `gdn_decode.comp` holds one `vconv`
/// float per value head, in an array of `MAX_V_PER_K` floats. That constant must equal this one.
/// The Qwen3.5-2B runs 1 and the Qwen3.5-9B runs 2.
pub const gdn_max_v_per_k: u32 = 8;

/// The head_dim of `qwen_qk_norm.comp`, its `local_size_x`. One lane owns one head index, so the
/// shader takes this width only. That constant must equal this one. Every Qwen3.5 model runs 256.
pub const qwen_qk_head_dim: u32 = 256;

/// Input elements the subgroup matvec holds in shared memory, 16 KiB of f32 at 4096. The build
/// feeds it as LDS_CAPACITY of `linear_q8_0_subgroup.comp`. The kernel wrapper refuses a wider
/// weight rather than corrupt memory.
pub const subgroup_lds_capacity: u32 = 4096;

/// Workgroups on the x axis of a reduce matvec grid, which gives one workgroup to each output.
/// The build feeds GRID_X of the reduce shaders, and the outputs past it fold onto the y axis. It
/// is a power of two under `max_workgroups`, so the launch holds on every conformant device.
pub const reduce_grid_x: u32 = 32768;

/// Largest decoder stack the engine allocates scratch for. SmolLM2-135M has 30, Llama-3.1-8B has
/// 32, and Llama-3.1-70B has 80. 64 covers the 8B with headroom.
pub const max_layers: u32 = 64;

/// Largest vocab the engine allocates token tables for. Llama-3.1 uses 128k, SmolLM2 uses 49k, and
/// Qwen3.5 uses 248k. 262144 clears every one of them.
pub const max_vocab: u32 = 262_144;

/// Largest model context the engine accepts, and the one engine-wide window bound. Qwen3.5 declares
/// a 256K context, so it is 1<<18. It sizes the static arrays and backs the kernel asserts.
pub const max_model_ctx: u32 = 1 << 18; // 262144

/// Ceiling on `--runs` of a model bench. The bench holds one result line per run.
pub const bench_runs_max: u32 = 1000;

/// KV bytes one `run` or `bench-model` session holds when no flag names a window.
pub const run_session_kv_bytes: u64 = 2 << 30;

/// KV bytes one serve session holds when no flag names a window.
pub const serve_session_kv_bytes: u64 = 128 << 20;

/// A default window rounds down to a multiple of this token count.
pub const window_granule: u32 = 256;

/// Ceiling on rows per `stepBatch`, the conversations advanced per engine tick. It sizes every
/// `[max_batch]` stack array on the step path, so it must stay small.
pub const max_batch: u32 = 8;

/// Ceiling on physical VRAM KV slots, the conversations a device holds resident at once. 128 is the
/// largest slot count a 16 GB card holds at f16 KV for the 9B model. A raise lands in two places:
/// this constant and the clamp anchors in `core/kv_budget.zig`.
pub const max_vram_slots: u32 = 128;

/// Ceiling on a serve prefill chunk: the prompt tokens ingested in one forward pass, independent of
/// `max_batch`. It bounds `--prefill-chunk N` and the prefill rows of a runner. It caps each
/// `prefill_batch_max` (llama fused 256 rows, qwen fused 128). Serve clamps a request to the real
/// width of the runner at boot.
pub const max_prefill_chunk: u32 = 256;

/// Ceiling on the rows of one CPU step: the decode slots and the widest prefill chunk. The CPU
/// runner asserts at init that its row scratch fits. A comptime assert of `cpu_layer_kernels.zig`
/// bounds the bytes of the stack row arrays that it sizes.
pub const step_rows_max: u32 = max_batch + max_prefill_chunk;

/// Rows of one prompt chunk on the CPU runners in `run` and `bench-model`. One weight read serves
/// every row, and only the last row of the prompt runs the lm_head. The SIMD batch kernel reads
/// each input row once per weight group of the batch path. So a wide chunk leaves the L2 cache.
/// Serve sizes its own chunk.
pub const cpu_prefill_chunk: u32 = 64;

/// Ceiling on the threads of one CPU pool, the calling thread included. The pool packs the
/// participant count of a dispatch into 8 bits of its epoch word. It holds its workers in a fixed
/// array of this length.
pub const max_threads: u32 = 64;

/// Ceiling on the threads that record spans while a trace runs. A thread past it records nothing,
/// and the trace metadata counts it.
pub const span_threads_max: u32 = 64;

/// The default capacity of the span ring of one thread. A span costs 24 bytes, so the ring
/// reserves 24 MiB, and its pages fault in only as spans arrive. A full ring drops the span and
/// counts it.
pub const spans_per_thread: u32 = 1024 * 1024;

/// The capacity of the GPU submit ring. A record and its ready flag cost 281 bytes.
pub const trace_gpu_records: u32 = 64 * 1024;

/// The largest ring capacity that `trace_spans_env` sets. A thread allocates its ring at its first
/// span, so this maximum takes 96 MiB per recording thread.
pub const spans_per_thread_max: u32 = 4 * 1024 * 1024;

/// The environment variable that sets the capacity of every span ring. The GPU ring keeps
/// `trace_gpu_records`.
pub const trace_spans_env = "ZIGFERENCER_TRACE_SPANS";

/// Ceiling on logical serve sessions. A session is resident in a VRAM slot or evicted to a RAM
/// blob. The residency table is heap, so this bounds memory, not stack. Only the fused-GPU build
/// can evict, so CPU builds keep logical equal to physical. The VRAM budget in
/// `resolve/device_placement.zig` caps its slots at the logical count and `max_vram_slots`.
pub const max_logical_sessions: u32 = 128;

/// Ceiling on live serve connections, the clients one server holds at once. A binary connection
/// costs two worker threads. The slot table holds two slots per allowed connection. Each slot holds
/// one queue of 256 events and, on the binary wire, one payload buffer of 20 bytes plus 64 KiB.
/// `cli/boot/serve_checks.zig` refuses a count past it.
pub const max_connections: u32 = 64;

/// Ceiling on `--io-timeout`, one hour, the lease ceiling below. A longer wait holds a connection
/// slot for a client that sends nothing.
pub const max_io_timeout_s: u32 = 3_600;

/// Ceiling on the time to live of a session lease, one hour. A detached session holds a slot until
/// its lease deadline, so this bounds that hold.
pub const max_resume_ttl_ms: u32 = 3_600_000;

/// Ceiling on the ids the serve scheduler forces when a think budget runs out: the closing tag,
/// then the separator after it. Each slot holds a queue of this length.
pub const think_force_ids_max: u8 = 4;

/// Largest sampler top-k candidate pool, which lives on the stack in `sample`. A positive top-k
/// runs every cut within the pool, and a top-k of 0 runs them over the full vocab.
pub const max_top_k: u32 = 256;

/// Largest presence penalty, either sign: the OpenAI range. The wire guard and the model registry
/// check it.
pub const max_presence_penalty: f32 = 2;

/// Largest repetition penalty factor. The rule divides a positive logit by the factor and
/// multiplies a negative one by it. At 2 a repeated id keeps half of a positive logit, which
/// already drops it far down a typical row. Common values lie between 1 and 1.5. The wire guard and
/// the model registry check it.
pub const max_repetition_penalty: f32 = 2;

/// Longest model name the serve layer carries in a session handshake. A serve bound, not a
/// wire-format rule. The trust boundary rejects an oversize name, and never truncates it.
pub const max_model_name_bytes: usize = 64;

/// Longest detokenized piece a streamed token event carries. A serve bound, not a wire-format rule,
/// checked against the whole vocab at server start. The BPE of Llama-3 holds pieces past 64 bytes
/// (runs of dashes, multi-byte scripts). 256 clears every registry vocab with headroom.
pub const max_token_text_bytes: usize = 256;

/// Most (user, assistant) pairs in one `session_append` frame. The reply holds one u32 position per
/// pair inside a fixed-size out event. This bound keeps it inside the largest server frame. A
/// client sends a longer history in several frames.
pub const max_append_pairs: u16 = 64;

/// The largest body of an HTTP request. A larger body gets 413.
pub const openai_max_request_bytes: usize = 256 * 1024;

/// The head buffer of the HTTP server. A larger head gets 431.
pub const openai_max_header_bytes: usize = 16 * 1024;
/// The longest chunk extension in bytes, white space and semicolon included. The decoder skips
/// an extension. The bound refuses a size line that never ends, so a client cannot hold the
/// reader with extension bytes below the body cap.
pub const openai_max_chunk_ext_bytes: usize = 1024;
/// The most framing bytes of one chunked body: size lines, extensions, line ends, and trailers.
/// With this bound equal to the body cap, a chunked body costs at most twice the cap on the
/// wire. A body of chunks of 16 bytes or more, with no extension, passes it at any size.
pub const openai_max_chunk_overhead_bytes: usize = openai_max_request_bytes;
/// The most input that a close drops. A close over unread input sends a reset, and a reset can
/// drop the last reply. The bound holds the largest request the HTTP frontend reads: a head, and
/// a chunked body of the cap with the most framing bytes. That is more than one binary message.
pub const close_drain_bytes: usize = openai_max_header_bytes + openai_max_request_bytes +
    openai_max_chunk_overhead_bytes;
/// The longest wait for the input of a refused HTTP request to end. The value follows the nginx
/// default `lingering_timeout` of 5 seconds, its wait for more input in a lingering close. Here
/// it bounds the whole drain. The error reply before the drain has its own `--io-timeout`
/// deadline per write.
pub const close_linger_ms: u32 = 5_000;
/// The longest wait of a refusal at the connection cap for the end of the client input, where late
/// input drops the unread refusal. The value is a project choice, not a measurement. A refused
/// client that keeps its socket open holds the acceptor for at most this time.
pub const refuse_linger_ms: u32 = 250;
/// The most messages in one rendered conversation. A render holds one plain range per message.
pub const max_chat_messages: usize = 64;
/// The most messages in one OpenAI request, equal to the render bound.
pub const openai_max_messages: usize = max_chat_messages;
/// The most content parts in one message. It bounds the loop that joins the parts. A client splits
/// a text message into a few parts, and the body cap bounds the bytes, so the count needs no more.
pub const openai_max_content_parts: usize = 64;
/// The most stop strings in one request, the limit of the OpenAI API.
pub const openai_max_stops: usize = 4;
/// The longest stop string in bytes. It bounds the stream holdback to one byte less. A stop string
/// is a short marker, such as the end tag of a chat format, so the cap passes the usual markers
/// and keeps the holdback below `max_token_text_bytes`.
pub const openai_max_stop_bytes: usize = 64;

// The TCP keepalive of an accepted socket. A peer that vanishes with no close is found after the
// idle time plus `keepalive_probes` unanswered probes, `keepalive_interval_s` apart: 60 + 6 * 10
// = 120 seconds. Then the next socket read fails, and the slot frees. The kernel sends a probe
// only while no output waits for an acknowledgment. The Linux defaults (`tcp_keepalive_time`
// 7200, `tcp_keepalive_intvl` 75, `tcp_keepalive_probes` 9, in the kernel document
// `Documentation/networking/ip-sysctl.rst`) find a dead peer after more than two hours. A dead
// peer holds a connection slot until then, so each value replaces its default.

/// The silence before the first probe. One minute keeps the probes of an idle live client to one
/// small packet per minute.
pub const keepalive_idle_s: c_int = 60;
/// The wait between two unanswered probes. A probe that the path drops gets its retry within ten
/// seconds.
pub const keepalive_interval_s: c_int = 10;
/// The unanswered probes that end a connection. Six probes over one minute let a live client
/// pass a short loss of its path and keep its connection.
pub const keepalive_probes: c_int = 6;

/// The `max_tokens` of a request that omits it. 0 is the whole window.
pub const openai_default_max_tokens: u32 = 0;

/// Floor on a budget-derived per-session KV window. A smaller session is too cramped to be useful,
/// so the planner (`core/kv_budget.zig`) refuses such a budget loudly.
pub const min_kv_window: u32 = 128;

/// The prefix-cache RAM budget `serve` uses when `--kv-ram` names no budget.
pub const serve_prefix_ram_default: u64 = 1 << 30;

/// Ceiling on the serve prefix cache's disk-index entries, one key and token sequence per on-disk
/// entry. The boot scan skips files past it with a warning, because a capped index must never read
/// as complete.
pub const serve_prefix_disk_index_max: usize = 1024;

/// Ceiling on `--preload` files per serve boot. Each file costs a full prefill on a cold cache, so
/// the bound keeps boot time a deliberate operator choice.
pub const serve_preload_max: usize = 8;

/// Ceiling on one `--preload` file's bytes, the text scale of serve: a system prompt, not a
/// document. An oversize file fails the boot.
pub const serve_preload_text_max: usize = 64 * 1024;

/// Bytes one swap stage holds: one staged copy, so the stage RAM stays fixed at any window. The
/// rule: the smallest chunk whose evict and restore rates reach 95% of the whole-blob rate. On a
/// 67.3 MiB Qwen3.5-2B blob at depth 4096, `bench-swap --chunk-mib` took 11.80 ms evict and 444.93
/// ms restore whole. 256 MiB reached 100% and 98%. 64 MiB evicted at 85% and 16 MiB at 47%.
pub const swap_stage_chunk_bytes: u64 = 256 << 20;

/// RAM a printed fit keeps back from the usable RAM. Free RAM moves by tens of MiB between two
/// boots. So a fit at the exact edge can refuse on the re-run. The checks stay exact, and only the
/// advice keeps this margin.
pub const ram_fit_headroom_bytes: u64 = 128 << 20;

/// The RAM budget the `run` path's prompt cache uses.
pub const prompt_cache_ram_default: u64 = 4 << 30;

/// The disk budget the `run` path's prompt cache uses.
pub const prompt_cache_disk_default: u64 = 16 << 30;

/// Usable share of the live device-local budget on most drivers. The 1/8 reserve absorbs what the
/// plan omits: the upload stage, the alignment padding of the weight arenas, and driver overhead.
pub const vram_share_default: VramShare = .{ .num = 7, .den = 8 };

/// Usable share on the driver that `vulkan_ctx.driverEntry` maps to this entry, the
/// translation-layer driver of the reference setup. This entry is a heuristic from one measured
/// setup: a plan under the default line lost the device there during the weight upload, and a
/// smaller plan ran. Every other driver keeps the default entry.
pub const vram_share_vulkan_on_d3d12: VramShare = .{ .num = 3, .den = 4 };

/// The usable share of each driver. The preflight refuses a plan that cannot fit under this line.
/// A plan under the line can still fail on a driver whose budget does not show its host-visible
/// staging or its residency.
pub const vram_shares: std.EnumArray(Driver, VramShare) = .init(.{
    .vulkan_on_d3d12 = vram_share_vulkan_on_d3d12,
    .other = vram_share_default,
});

/// Usable share of the RAM that `MemAvailable` reports at boot. It covers anonymous memory: host
/// KV, swap stages, blobs, and the prefix cache. An anonymous overrun crashes the process, so the
/// planner refuses over this line with a typed error.
pub const ram_usable_num: u64 = 7;

/// The 1/8 reserve holds the OS, the process heap, and the page cache that backs the mapped
/// weights. These constants stay apart from the VRAM pair because the reserve has a different
/// reason.
pub const ram_usable_den: u64 = 8;

/// Elements one KV row can hold: the KV heads times the head width. The bound keeps every byte
/// product of a KV layout inside a `u64`.
pub const max_kv_row_elems: u32 = 1 << 20;

/// Ceiling on one layer's declared fixed per-slot cost, one recurrent state slab of a hybrid
/// model. A GDN layer holds `(time_step_rank * state_size^2 + (conv_kernel - 1) * qkv) * 4` bytes:
/// `(16 * 128^2 + 3 * 6144) * 4 = 1122304` on the Qwen3.5-2B and `(32 * 128^2 + 3 * 8192) * 4 =
/// 2195456` on the 9B, so this sits far above either. The loader refuses a card past it.
pub const max_fixed_state_bytes_per_layer: u64 = 1 << 30;

/// Ceiling on the whole model's fixed per-slot cost, every layer's slab summed. It keeps every
/// byte product of the planner inside a `u64`. The loader refuses a card past it.
pub const max_fixed_state_bytes: u64 = 1 << 32;

/// The workgroup count of one grid axis that needs no device query. The Vulkan required limits
/// table sets 65535 as the minimum of `maxComputeWorkGroupCount` on each axis.
pub const max_workgroups: u32 = 65535;

/// Elements one KV region can hold. The shaders index KV with a 32-bit value, and SPIR-V reads that
/// index as signed. So the last index of a region must stay below 2^31.
pub const max_kv_index_elems: u64 = 1 << 31;

/// The environment variable that replaces the reported storage-buffer range. A driver can report
/// the spec minimum while it reads correctly past it. An operator who proved a larger range can
/// state it here.
pub const storage_range_env = "ZIGFERENCER_VK_STORAGE_RANGE";

/// The largest value `storage_range_env` takes. Vulkan states the range as a 32-bit count.
pub const storage_range_max: u64 = std.math.maxInt(u32);

/// The environment variable that runs the Vulkan paths on a CPU device. The value `1` takes a CPU
/// device and no other, so a debug run with no GPU is a named choice.
pub const software_device_env = "ZIGFERENCER_VK_SOFTWARE";

// -- types ------------------------------------------------------------------

/// The Vulkan driver of a device, as far as a limit tells drivers apart. A boot with no device and
/// a test take `other`.
pub const Driver = enum { vulkan_on_d3d12, other };

/// The workgroup counts of a reduce matvec grid on the x and y axes.
pub const ReduceGrid = struct {
    x: u32,
    y: u32,
};

/// A share of a budget: `num / den`, with a reserve that never empties.
pub const VramShare = struct {
    num: u64,
    den: u64,
};

/// The device choices of the environment. Every command that opens a Vulkan context passes them.
pub const DeviceOptions = struct {
    /// Replaces the reported `maxStorageBufferRange`. It comes from `storage_range_env`.
    storage_range_override: ?u64 = null,
    /// Takes a CPU Vulkan device and no other. It comes from `software_device_env`.
    software: bool = false,
};

/// The storage-buffer limits of one device, read once when its context opens.
pub const BufferLimits = struct {
    /// Bytes one storage-buffer binding can address: the reported
    /// `maxStorageBufferRange`, or the value of `storage_range_env`.
    storage_range: u64,
    /// Bytes one buffer can hold: `maxMemoryAllocationSize`, or the lower
    /// `maxBufferSize` where the device reports one.
    allocation_size: u64,
    /// A storage-buffer binding offset must be a multiple of this power of two.
    offset_alignment: u64,
};

/// The first limit a buffer breaks, in the order `bufferFits` checks them.
pub const BufferVerdict = enum {
    fits,
    index_past_31_bits,
    past_allocation_size,
    past_storage_range,
    misaligned,
};

/// The state cell of one GDN scan lane: a value head and a column of its state.
pub const GdnScanCell = struct { head: u32, column: u32 };

// -- api --------------------------------------------------------------------

/// The usable share of `available_bytes` of VRAM on `driver`, rounded down.
pub fn vramUsable(driver: Driver, available_bytes: u64) u64 {
    const share = vram_shares.get(driver);
    const usable = available_bytes / share.den * share.num;
    std.debug.assert(usable <= available_bytes);
    // The reserve is never empty, so a plan never takes the whole budget.
    if (available_bytes > 0) std.debug.assert(usable < available_bytes);

    return usable;
}

/// The grid of a reduce matvec over `n` outputs: at most `reduce_grid_x` workgroups on x, and
/// `ceil(n / x)` on y. Output j runs in workgroup (j % x, j / x), and the workgroups at the last
/// y index can lie past `n`.
pub fn reduceGrid(n: u32) ReduceGrid {
    std.debug.assert(n >= 1);
    std.debug.assert(n <= max_vocab);

    const grid: ReduceGrid = .{
        .x = @min(n, reduce_grid_x),
        .y = std.math.divCeil(u32, n, reduce_grid_x) catch unreachable,
    };
    std.debug.assert(grid.x <= max_workgroups);
    std.debug.assert(grid.y <= max_workgroups);
    std.debug.assert(@as(u64, grid.x) * grid.y >= n);
    std.debug.assert(@as(u64, grid.x) * (grid.y - 1) < n);
    return grid;
}

/// The usable share of `available_bytes` of RAM, rounded down.
pub fn ramUsable(available_bytes: u64) u64 {
    const usable = available_bytes / ram_usable_den * ram_usable_num;
    std.debug.assert(usable <= available_bytes);
    // The reserve is never empty, so a plan never takes the whole RAM.
    if (available_bytes > 0) std.debug.assert(usable < available_bytes);

    return usable;
}

/// Whether a planned device-local footprint fits the usable share of `available`
/// VRAM (the live budget, or the heap size when the budget extension is absent) on `driver`.
pub fn vramPlanFits(driver: Driver, planned_bytes: u64, available_bytes: u64) bool {
    return planned_bytes <= vramUsable(driver, available_bytes);
}

/// Check a buffer of `elems` elements and `bytes` bytes against the device limits, in the
/// `BufferVerdict` order. Each bound is inclusive. A buffer that starts a binding must hold whole
/// alignment units, so the next packed region starts aligned.
pub fn bufferFits(elems: u64, bytes: u64, lim: BufferLimits) BufferVerdict {
    std.debug.assert(elems <= bytes);
    std.debug.assert(lim.storage_range >= 1);
    std.debug.assert(lim.allocation_size >= 1);
    std.debug.assert(lim.offset_alignment >= 1);
    std.debug.assert(std.math.isPowerOfTwo(lim.offset_alignment));

    if (elems > max_kv_index_elems) return .index_past_31_bits;
    if (bytes > lim.allocation_size) return .past_allocation_size;
    if (bytes > lim.storage_range) return .past_storage_range;
    if (bytes % lim.offset_alignment != 0) return .misaligned;

    return .fits;
}

/// The widest window whose KV slab of one layer keeps a 32-bit element index. The CPU KV cache
/// addresses a layer slab with a `u32`. The loader bounds the row by `max_kv_row_elems`.
pub fn kvSlabWindowMax(kv_row_elems: u32) u32 {
    std.debug.assert(kv_row_elems >= 1);
    std.debug.assert(kv_row_elems <= max_kv_row_elems);

    const window = @min(max_model_ctx, std.math.maxInt(u32) / kv_row_elems);
    std.debug.assert(window >= 1);
    std.debug.assert(@as(u64, window) * kv_row_elems <= std.math.maxInt(u32));
    return window;
}

/// The first device limit that one f32 buffer of `n` blocks of `block_elems` elements breaks. A
/// kernel indexes the buffer with a 32-bit product of a block index and the block width, and binds
/// it whole at offset 0, so the verdict is never `.misaligned`.
pub fn blocksFit(n: u32, block_elems: u32, lim: BufferLimits) BufferVerdict {
    const elems = @as(u64, n) * block_elems;
    const bytes = elems * @sizeOf(f32);
    var unaligned = lim;
    unaligned.offset_alignment = 1;
    const verdict = bufferFits(elems, bytes, unaligned);
    std.debug.assert(verdict != .misaligned);
    if (verdict == .fits) std.debug.assert(elems <= std.math.maxInt(u32));
    return verdict;
}

/// `n` rounded down to a whole number of `window_granule` positions.
pub fn floorToGranule(n: u64) u64 {
    const floored = n / window_granule * window_granule;
    std.debug.assert(floored <= n);
    std.debug.assert(floored % window_granule == 0);
    std.debug.assert(n - floored < window_granule);
    return floored;
}

/// Parse the value of `storage_range_env`: a decimal byte count in
/// `1..storage_range_max`, digits only.
pub fn parseStorageRange(text: []const u8) error{InvalidStorageRange}!u64 {
    if (text.len == 0) return error.InvalidStorageRange;

    for (text) |c| {
        if (!std.ascii.isDigit(c)) return error.InvalidStorageRange;
    }

    const value = std.fmt.parseInt(u64, text, 10) catch return error.InvalidStorageRange;
    if (value < 1 or value > storage_range_max) return error.InvalidStorageRange;

    std.debug.assert(value >= 1);
    std.debug.assert(value <= storage_range_max);
    return value;
}

/// Parse the value of `software_device_env`: `1` or `0`.
pub fn parseSoftwareDevice(text: []const u8) error{InvalidSoftwareDevice}!bool {
    if (std.mem.eql(u8, text, "1")) return true;
    if (std.mem.eql(u8, text, "0")) return false;

    return error.InvalidSoftwareDevice;
}

/// Parse the value of `trace_spans_env`: a decimal ring capacity in
/// `1..spans_per_thread_max`, digits only.
pub fn parseTraceSpans(text: []const u8) error{InvalidTraceSpans}!u32 {
    if (text.len == 0) return error.InvalidTraceSpans;

    for (text) |c| {
        if (!std.ascii.isDigit(c)) return error.InvalidTraceSpans;
    }

    const value = std.fmt.parseInt(u32, text, 10) catch return error.InvalidTraceSpans;
    if (value < 1 or value > spans_per_thread_max) return error.InvalidTraceSpans;

    std.debug.assert(value >= 1);
    std.debug.assert(value <= spans_per_thread_max);
    return value;
}

/// The window ceiling of one model: the context the file declares, capped by
/// the engine bound and then by a registry override. An override can only
/// lower the ceiling.
pub fn contextCeiling(context_length: u32, override: ?u32) u32 {
    std.debug.assert(context_length >= 1);

    if (override) |o| std.debug.assert(o >= 1);

    const engine_capped = @min(context_length, max_model_ctx);
    const ceiling = if (override) |o| @min(engine_capped, o) else engine_capped;
    std.debug.assert(ceiling >= 1);
    std.debug.assert(ceiling <= max_model_ctx);
    std.debug.assert(ceiling <= context_length);
    if (override) |o| std.debug.assert(ceiling <= o);

    return ceiling;
}

/// Shared memory of one GDN scan workgroup in bytes. It mirrors the `shared` arrays of
/// `gdn_chunk_scan.comp`: the k and q rows, the delta, two pair-product tables, four token rows.
pub fn gdnScanLdsBytes() u32 {
    const rows = 2 * gdn_scan_sub * gdn_head_dim;
    const delta = gdn_scan_sub * gdn_scan_cols;
    const pairs = 2 * gdn_scan_sub * gdn_scan_sub;
    const per_token = 4 * gdn_scan_sub;
    return (rows + delta + pairs + per_token) * @sizeOf(f32);
}

/// Workgroups of one GDN scan dispatch: `gdn_head_dim / gdn_scan_cols` column
/// blocks for each value head.
pub fn gdnScanGroups(n_v_heads: u32) u32 {
    std.debug.assert(n_v_heads >= 1);

    const groups = n_v_heads * (gdn_head_dim / gdn_scan_cols);
    std.debug.assert(groups % n_v_heads == 0);
    std.debug.assert(groups * gdn_scan_cols == n_v_heads * gdn_head_dim);
    return groups;
}

/// The cell that lane `lane` of workgroup `group` owns. It mirrors `gdn_chunk_scan.comp`. The
/// workgroups of one head are adjacent, and lane `lane` of block `c` owns column
/// `c * gdn_scan_cols + lane`.
pub fn gdnScanCell(group: u32, lane: u32) GdnScanCell {
    std.debug.assert(lane < gdn_scan_cols);

    const blocks = gdn_head_dim / gdn_scan_cols;
    const cell: GdnScanCell = .{
        .head = group / blocks,
        .column = (group % blocks) * gdn_scan_cols + lane,
    };
    std.debug.assert(cell.column < gdn_head_dim);
    std.debug.assert(cell.column % gdn_scan_cols == lane);
    return cell;
}

comptime {
    // A power of two turns the multiply by the grid width in the reduce shaders into a shift, and
    // the width fits the guaranteed count.
    std.debug.assert(std.math.isPowerOfTwo(reduce_grid_x));
    std.debug.assert(reduce_grid_x <= max_workgroups);
    // The largest vocabulary folds into a grid inside the guaranteed count.
    std.debug.assert(max_vocab / reduce_grid_x < max_workgroups);

    // Every driver keeps a reserve, so a plan never takes the whole budget.
    for (vram_shares.values) |share| {
        std.debug.assert(share.num >= 1);
        std.debug.assert(share.num < share.den);
    }

    // A default window floors at `min_kv_window` and otherwise lands on a
    // granule, so the floor must sit inside the first granule.
    std.debug.assert(min_kv_window <= window_granule);
    // The engine bound is a whole number of granules, so a default window at
    // the bound is also a granule multiple.
    std.debug.assert(max_model_ctx % window_granule == 0);
    // The serve default never passes the run default of the same model.
    std.debug.assert(serve_session_kv_bytes <= run_session_kv_bytes);
    // The help text names both budgets in whole MiB.
    std.debug.assert(run_session_kv_bytes % (1 << 20) == 0);
    std.debug.assert(serve_session_kv_bytes % (1 << 20) == 0);
    // The 64 lanes of the split shader stride a run in whole steps.
    std.debug.assert(attn_split_len % 64 == 0);
    std.debug.assert(attn_split_rows >= 1);
    // The widest head of the split pass is whole 16-byte K loads.
    std.debug.assert(attn_head_dim_granule * @sizeOf(f16) == 16);
    std.debug.assert(attn_max_head_dim % attn_head_dim_granule == 0);
    // The split pass holds the scores, q rows, and LDS trees in the 32 KiB the driver allows.
    std.debug.assert(attn_split_queries >= 1);
    std.debug.assert(attn_split_queries <= attn_split_queries_max);
    std.debug.assert(
        attn_split_queries * (attn_split_len + attn_max_head_dim + 64 + 1) * @sizeOf(
            f32,
        ) <= 32 * 1024,
    );
    // The SIMD kernel reduces the scores of a tile eight at a time, one per
    // lane of its 8-wide f32 vector.
    std.debug.assert(attn_cpu_tile >= 8);
    std.debug.assert(attn_cpu_tile % 8 == 0);
    // The stack score row of one head block stays at or below 64 KiB.
    std.debug.assert(attn_max_group >= 1);
    std.debug.assert(attn_max_group * attn_cpu_tile * @sizeOf(f32) <= 64 * 1024);
    // The engine bound is a whole number of tiles, so every tile edge of a row
    // fits in 32 bits.
    std.debug.assert(max_model_ctx % attn_cpu_tile == 0);
    std.debug.assert(attn_cpu_parts >= 1);
    // The GDN scan splits a head into whole workgroups of whole waves.
    std.debug.assert(gdn_scan_cols % 32 == 0);
    std.debug.assert(gdn_head_dim % gdn_scan_cols == 0);
    std.debug.assert(gdn_scan_sub >= 1);
    std.debug.assert(gdn_scan_sub <= gdn_scan_cols);
    std.debug.assert(gdnScanLdsBytes() <= 32 * 1024);
    // A CPU prompt chunk fits the stack row array of `prefillBatch`, which
    // holds `max_prefill_chunk` rows.
    std.debug.assert(cpu_prefill_chunk >= 1);
    std.debug.assert(cpu_prefill_chunk <= max_prefill_chunk);
    // A serve step holds every decode slot and a full prefill chunk.
    std.debug.assert(step_rows_max >= max_batch + max_prefill_chunk);
    // The epoch word of the pool holds the participant count in 8 bits.
    std.debug.assert(max_threads >= 2);
    std.debug.assert(max_threads < 0xff);
    // A lease deadline is the drop time plus at most this many milliseconds.
    // The sum stays a 64-bit count of nanoseconds, far from the clock range.
    std.debug.assert(max_resume_ttl_ms >= 1);
    std.debug.assert(@as(u64, max_resume_ttl_ms) * std.time.ns_per_ms <= std.math.maxInt(i64));
    std.debug.assert(openai_max_stop_bytes < max_token_text_bytes);
    // The keepalive doc states the detection time.
    std.debug.assert(keepalive_idle_s + keepalive_probes * keepalive_interval_s == 120);
    // The serve config holds the deadline in milliseconds in a u32.
    std.debug.assert(@as(u64, max_io_timeout_s) * std.time.ms_per_s <= std.math.maxInt(u32));
}

// -- tests ------------------------------------------------------------------

test "reduceGrid: the grid covers every output once" {
    const Case = struct { n: u32, x: u32, y: u32 };
    const cases = [_]Case{
        .{ .n = 1, .x = 1, .y = 1 },
        .{ .n = 32767, .x = 32767, .y = 1 },
        .{ .n = 32768, .x = 32768, .y = 1 },
        .{ .n = 32769, .x = 32768, .y = 2 },
        // The Llama-3.1 and the Qwen3.5 vocabularies.
        .{ .n = 128256, .x = 32768, .y = 4 },
        .{ .n = 248320, .x = 32768, .y = 8 },
    };
    for (cases) |case| {
        const grid = reduceGrid(case.n);
        try std.testing.expectEqual(case.x, grid.x);
        try std.testing.expectEqual(case.y, grid.y);
    }

    // Over every count to the largest vocabulary, the grid holds each output, and the first
    // output of the last y index lies below `n`.
    var n: u32 = 1;
    while (n <= max_vocab) : (n += 1) {
        const grid = reduceGrid(n);
        try std.testing.expect(@as(u64, grid.x) * grid.y >= n);
        try std.testing.expect(@as(u64, grid.x) * (grid.y - 1) < n);
        try std.testing.expect(grid.x <= max_workgroups);
        try std.testing.expect(grid.y <= max_workgroups);
    }
}

test "vramPlanFits: leaves a reserve below the available budget" {
    // 8 GiB available -> 7 GiB usable.
    try std.testing.expect(vramPlanFits(.other, 7 << 30, 8 << 30));
    try std.testing.expect(!vramPlanFits(.other, (7 << 30) + 1, 8 << 30));
    // A plan that wants the whole budget is refused, because the reserve is load-bearing.
    try std.testing.expect(!vramPlanFits(.other, 8 << 30, 8 << 30));
    // Zero budget (e.g. a failed query) refuses any positive plan.
    try std.testing.expect(!vramPlanFits(.other, 1, 0));
}

test "vramUsable: seven eighths of the budget, rounded down" {
    try std.testing.expectEqual(@as(u64, 7 << 30), vramUsable(.other, 8 << 30));
    try std.testing.expectEqual(@as(u64, 939_524_096), vramUsable(.other, 1 << 30));
    // The rounding happens on the eighth, so a budget under 8 bytes has none.
    try std.testing.expectEqual(@as(u64, 0), vramUsable(.other, 7));
    try std.testing.expectEqual(@as(u64, 7), vramUsable(.other, 15));
    try std.testing.expectEqual(@as(u64, 0), vramUsable(.other, 0));
}

test "vramUsable: each driver entry takes its share of the budget, rounded down" {
    const Case = struct { driver: Driver, available: u64, usable: u64 };
    const cases = [_]Case{
        .{ .driver = .other, .available = 8 << 30, .usable = 7 << 30 },
        .{ .driver = .other, .available = 15, .usable = 7 },
        .{ .driver = .vulkan_on_d3d12, .available = 8 << 30, .usable = 6 << 30 },
        // The usual budget of a 16 GB card under a translation-layer driver leaves 11676 MiB.
        .{ .driver = .vulkan_on_d3d12, .available = 15568 << 20, .usable = 11676 << 20 },
        .{ .driver = .vulkan_on_d3d12, .available = 7, .usable = 3 },
        .{ .driver = .vulkan_on_d3d12, .available = 0, .usable = 0 },
    };
    for (cases) |case| {
        try std.testing.expectEqual(case.usable, vramUsable(case.driver, case.available));
    }
}

test "vramPlanFits: a plan under the default line can be past the translation-layer line" {
    const budget: u64 = 15568 << 20;
    // The plan that lost the device at upload, and the plan that then ran.
    const lost: u64 = 13524 << 20;
    const ran: u64 = 11100 << 20;
    try std.testing.expect(vramPlanFits(.other, lost, budget));
    try std.testing.expect(!vramPlanFits(.vulkan_on_d3d12, lost, budget));
    try std.testing.expect(vramPlanFits(.vulkan_on_d3d12, ran, budget));
    try std.testing.expect(!vramPlanFits(.vulkan_on_d3d12, 1, 0));
}

test "ramUsable: seven eighths of the free RAM, rounded down, with a reserve that never empties" {
    // The anchor host: 22 GiB free leaves 20669530112 usable bytes.
    try std.testing.expectEqual(@as(u64, 20_669_530_112), ramUsable(23_622_320_128));
    try std.testing.expectEqual(@as(u64, 0), ramUsable(7));
    try std.testing.expectEqual(@as(u64, 7), ramUsable(15));
    try std.testing.expectEqual(@as(u64, 0), ramUsable(0));
    // Over a grid of 4096 figures plus the `u64` ceiling, the share stays under the input and
    // never falls as the input rises.
    var previous: u64 = 0;
    var available: u64 = 0;
    while (available < 4096) : (available += 1) {
        const usable = ramUsable(available * 1_000_003);
        if (available > 0) {
            try std.testing.expect(usable < available * 1_000_003);
        }

        try std.testing.expect(usable >= previous);
        previous = usable;
    }

    const top = ramUsable(std.math.maxInt(u64));
    try std.testing.expect(top < std.math.maxInt(u64));
    try std.testing.expectEqual(std.math.maxInt(u64) / 8 * 7, top);
}

test "floorToGranule: whole granules only, never above the input" {
    try std.testing.expectEqual(@as(u64, 0), floorToGranule(0));
    try std.testing.expectEqual(@as(u64, 0), floorToGranule(window_granule - 1));
    try std.testing.expectEqual(@as(u64, window_granule), floorToGranule(window_granule));
    try std.testing.expectEqual(@as(u64, window_granule), floorToGranule(2 * window_granule - 1));
    try std.testing.expectEqual(@as(u64, max_model_ctx), floorToGranule(max_model_ctx));
}

test "contextCeiling: the engine bound and a lower override cap the declared context" {
    // A declared context under the engine bound is its own ceiling.
    try std.testing.expectEqual(@as(u32, 8192), contextCeiling(8192, null));
    try std.testing.expectEqual(@as(u32, 1), contextCeiling(1, null));
    // The bound itself and one past it both land on the bound.
    try std.testing.expectEqual(max_model_ctx, contextCeiling(max_model_ctx, null));
    try std.testing.expectEqual(max_model_ctx, contextCeiling(max_model_ctx + 1, null));
    try std.testing.expectEqual(max_model_ctx, contextCeiling(std.math.maxInt(u32), null));
    // A lower override wins: a 131072 file capped at its original 8192.
    try std.testing.expectEqual(@as(u32, 8192), contextCeiling(131_072, 8192));
    // A higher override never raises the ceiling past the file or the bound.
    try std.testing.expectEqual(@as(u32, 8192), contextCeiling(8192, 131_072));
    try std.testing.expectEqual(max_model_ctx, contextCeiling(1 << 20, 1 << 20));
}

/// Test limits with every bound far apart, so each check fails alone.
const test_buffer_limits: BufferLimits = .{
    .storage_range = 128 << 20,
    .allocation_size = 2 << 30,
    .offset_alignment = 16,
};

test "bufferFits: each bound passes at the limit and fails one past it" {
    const lim = test_buffer_limits;
    try std.testing.expectEqual(BufferVerdict.fits, bufferFits(16, 16, lim));
    // The index bound, with the bytes inside every other bound.
    const wide: BufferLimits = .{
        .storage_range = 1 << 40,
        .allocation_size = 1 << 40,
        .offset_alignment = 16,
    };
    try std.testing.expectEqual(
        BufferVerdict.fits,
        bufferFits(max_kv_index_elems, max_kv_index_elems * 2, wide),
    );
    try std.testing.expectEqual(
        BufferVerdict.index_past_31_bits,
        bufferFits(max_kv_index_elems + 1, max_kv_index_elems * 2 + 16, wide),
    );
    // The allocation size, with a storage range that allows it.
    const deep_range: BufferLimits = .{
        .storage_range = 4 << 30,
        .allocation_size = 2 << 30,
        .offset_alignment = 16,
    };
    try std.testing.expectEqual(BufferVerdict.fits, bufferFits(1 << 29, 2 << 30, deep_range));
    try std.testing.expectEqual(
        BufferVerdict.past_allocation_size,
        bufferFits(1 << 29, (2 << 30) + 16, deep_range),
    );
    // The storage range.
    try std.testing.expectEqual(BufferVerdict.fits, bufferFits(1 << 25, 128 << 20, lim));
    try std.testing.expectEqual(
        BufferVerdict.past_storage_range,
        bufferFits(1 << 25, (128 << 20) + 16, lim),
    );
    // The alignment.
    try std.testing.expectEqual(BufferVerdict.fits, bufferFits(4, 32, lim));
    try std.testing.expectEqual(BufferVerdict.misaligned, bufferFits(4, 33, lim));
}

test "bufferFits: the first broken limit in check order wins" {
    const lim = test_buffer_limits;
    // Index and allocation both broken: the index wins.
    try std.testing.expectEqual(
        BufferVerdict.index_past_31_bits,
        bufferFits(max_kv_index_elems + 1, 16 << 30, lim),
    );
    // Allocation and storage range both broken: the allocation wins.
    try std.testing.expectEqual(
        BufferVerdict.past_allocation_size,
        bufferFits(1 << 20, (2 << 30) + 16, lim),
    );
    // Storage range and alignment both broken: the storage range wins.
    try std.testing.expectEqual(
        BufferVerdict.past_storage_range,
        bufferFits(1 << 20, (128 << 20) + 1, lim),
    );
}

test "blocksFit: the 32-bit index, the allocation, and the range each bound the blocks" {
    const wide: BufferLimits = .{
        .storage_range = 1 << 40,
        .allocation_size = 1 << 40,
        .offset_alignment = 256,
    };
    const block: u32 = 1 << 20;
    const index_max: u32 = @intCast(max_kv_index_elems / block);
    try std.testing.expectEqual(BufferVerdict.fits, blocksFit(index_max, block, wide));
    const past = blocksFit(index_max + 1, block, wide);
    try std.testing.expectEqual(BufferVerdict.index_past_31_bits, past);
    // The GDN state of 128 slots of the 9B, 32 heads of 128 by 128, fits the index.
    try std.testing.expectEqual(BufferVerdict.fits, blocksFit(128, 32 * 128 * 128, wide));

    var small = wide;
    small.allocation_size = 4 * block;
    try std.testing.expectEqual(BufferVerdict.fits, blocksFit(1, block, small));
    try std.testing.expectEqual(BufferVerdict.past_allocation_size, blocksFit(2, block, small));

    var narrow = wide;
    narrow.storage_range = 4 * block;
    try std.testing.expectEqual(BufferVerdict.past_storage_range, blocksFit(2, block, narrow));
}

test "parseSoftwareDevice: only 1 and 0 name a choice" {
    try std.testing.expectEqual(true, try parseSoftwareDevice("1"));
    try std.testing.expectEqual(false, try parseSoftwareDevice("0"));
    for ([_][]const u8{ "", "yes", "true", "2", " 1" }) |text| {
        try std.testing.expectError(error.InvalidSoftwareDevice, parseSoftwareDevice(text));
    }
}

test "parseStorageRange: digits in 1..2^32-1 only" {
    try std.testing.expectEqual(@as(u64, 1), try parseStorageRange("1"));
    try std.testing.expectEqual(@as(u64, 134_217_728), try parseStorageRange("134217728"));
    try std.testing.expectEqual(storage_range_max, try parseStorageRange("4294967295"));
    const bad = [_][]const u8{
        "",
        "0",
        "4294967296",
        "99999999999999999999999",
        "abc",
        "+5",
        "-1",
        "1_000",
        " 5",
        "5 ",
        "0x10",
    };
    for (bad) |text| {
        try std.testing.expectError(error.InvalidStorageRange, parseStorageRange(text));
    }
}

test "parseTraceSpans: digits in 1..spans_per_thread_max only" {
    try std.testing.expectEqual(@as(u32, 1), try parseTraceSpans("1"));
    try std.testing.expectEqual(spans_per_thread, try parseTraceSpans("1048576"));
    try std.testing.expectEqual(spans_per_thread_max, try parseTraceSpans("4194304"));
    const bad = [_][]const u8{
        "",
        "0",
        "4194305",
        "4294967296",
        "99999999999999999999999",
        "abc",
        "+5",
        "-1",
        "1_000",
        " 5",
        "5 ",
        "0x10",
    };
    for (bad) |text| {
        try std.testing.expectError(error.InvalidTraceSpans, parseTraceSpans(text));
    }
}

test "gdnScanCell: every state column of every head has exactly one owner lane" {
    // The 2B and the 9B value-head counts.
    for ([_]u32{ 16, 32 }) |n_v| {
        const groups = gdnScanGroups(n_v);
        try std.testing.expectEqual(n_v * (gdn_head_dim / gdn_scan_cols), groups);
        var owners = [_]u8{0} ** (32 * gdn_head_dim);
        var group: u32 = 0;
        while (group < groups) : (group += 1) {
            var lane: u32 = 0;
            while (lane < gdn_scan_cols) : (lane += 1) {
                const cell = gdnScanCell(group, lane);
                try std.testing.expect(cell.head < n_v);
                owners[cell.head * gdn_head_dim + cell.column] += 1;
            }
        }

        for (owners[0 .. n_v * gdn_head_dim]) |count| {
            try std.testing.expectEqual(@as(u8, 1), count);
        }

        for (owners[n_v * gdn_head_dim ..]) |count| {
            try std.testing.expectEqual(@as(u8, 0), count);
        }
    }

    try std.testing.expectEqual(@as(u32, 64), gdnScanGroups(16));
    try std.testing.expectEqual(@as(u32, 20_736), gdnScanLdsBytes());
}
