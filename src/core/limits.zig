//! Project-wide static engine limits.
//! The loader rejects a model that exceeds a structural bound with a typed
//! error at load. It caps a declared context at `max_model_ctx` and logs the
//! cut. It never truncates in silence (see `io/gguf/loader.zig`).

const std = @import("std");

// -- constants --------------------------------------------------------------

/// Tensor rank ceiling.
pub const max_n_dims: u8 = 4;

/// Positions one split of the GPU attention walks. A decode or prefill row of
/// more positions splits its walk into `ceil(t_len / attn_split_len)` runs,
/// and a merge pass combines the runs in split order. A row of at most this
/// length runs a resident kernel. The build feeds the value to glslc as
/// ATTN_SPLIT, and the resident and split shaders size their shared score row
/// with it, 1 KiB at 256.
pub const attn_split_len: u32 = 256;

/// Rows of the split-attention scratch: the split rows that one batched step
/// holds between their split pass and their merge pass. The scratch holds one
/// partial record per row, query head, and split, and it is sized at the
/// declared context of the model. At 4 rows the largest registry scratch, the
/// Qwen3.5-9B at 262144 positions, takes 64.5 MiB. That is inside the 128 MiB
/// storage range that Vulkan guarantees on every device.
pub const attn_split_rows: u32 = 4;

/// Query heads that one workgroup of the GPU split pass serves. The heads of
/// one GQA group read the same K and V rows, so one workgroup loads each row
/// once for up to this many heads. A group of G heads takes
/// `ceil(G / attn_split_queries)` workgroups. The build feeds the value to
/// glslc as ATTN_QUERIES, and the value must not pass
/// `attn_split_queries_max`. A grid sweep of 8, 4, and 2 picked 4: at 8 the LDS of a workgroup
/// halves the
/// resident workgroups, and at 2 a group of 4 heads reads K and V twice. A
/// group wider than 4 reads K and V once per 4 heads.
pub const attn_split_queries: u32 = 4;

/// The most query heads that the GPU split pass can serve in one workgroup.
/// The shader keeps one named register set per query and picks the sets with
/// uniform guards, so it has sets for 8 queries and no more.
pub const attn_split_queries_max: u32 = 8;

/// The head_dim granule of the GPU split pass. The pass reads a K row in
/// 16-byte loads, and one load holds 8 halves.
pub const attn_head_dim_granule: u32 = 8;

/// The widest GQA group (`n_heads / n_kv_heads`) that the CPU attention part
/// kernels hold at once. Every registered model states 3 or 4. The kernels keep
/// one stack score row per head of a block, so this is the block width, and a
/// wider group walks in blocks of this many heads.
pub const attn_max_group: u32 = 8;

/// The widest head that the CPU attention part kernels hold in one stack row.
/// The scalar kernel widens an f16 K or V row into a stack row of this width,
/// and a wider head walks in chunks of this width. The vector kernel takes
/// every head up to this width, and a wider head falls back to the scalar
/// kernel. The GPU split pass stages the q row of a head in shared memory of
/// this width, as `ATTN_D_MAX`, and takes no wider head.
pub const attn_max_head_dim: u32 = 256;

/// Positions one tile of the CPU attention row holds. The CPU kernel folds a
/// row tile by tile, and it keeps the scores of one tile in a stack row of
/// `[attn_max_group][attn_cpu_tile]f32`, 64 KiB at 2048. So the tile bounds the
/// kernel stack, and no score buffer grows with the window.
pub const attn_cpu_tile: u32 = 2048;

/// The part ceiling of one CPU attention row. A row splits into
/// `min(attn_cpu_parts, tiles)` parts of contiguous tiles. The unit of thread
/// parallelism is one KV group over one part. The part count is a function of
/// `t_len` alone, so the result does not depend on the thread count. The
/// partial buffer holds one record per query head and part, so this ceiling
/// also sets its size.
pub const attn_cpu_parts: u32 = 16;

/// Output-tile geometry of the batched Q8_0 matmul: 16 batch rows by 32 output
/// features per workgroup. The build feeds the pair to glslc as the B_TILE and
/// N_TILE defines of `linear_q8_0_batch.comp`, and the kernel wrapper derives
/// its grid from the same pair, so the tile the shader walks and the grid the
/// host dispatches read one number. `N_TILE` is the workgroup width, so it must
/// stay a multiple of 32, the wave width the kernel pins.
pub const q8_batch_b_tile: u32 = 16;

pub const q8_batch_n_tile: u32 = 32;

/// The same geometry for the Q4_0 prefill matmul, 8 batch rows by 32 output
/// features. The nibble weight holds half the bytes per column, so this kernel
/// carries half the batch rows of the Q8_0 sibling.
pub const q4_batch_b_tile: u32 = 8;

pub const q4_batch_n_tile: u32 = 32;

/// Output-tile geometry of the register-blocked Q8_0 GEMM, 64 by 64 per
/// workgroup of 256 threads. The build feeds the pair to glslc as the BM and BN
/// defines of `linear_q8_0_gemm.comp`. A workgroup stages one weight tile
/// through shared memory and every one of the 64 rows reuses it.
pub const gemm_m_tile: u32 = 64;

pub const gemm_n_tile: u32 = 64;

/// Geometry of the chunked GDN prefill scan. The build feeds the pair to glslc
/// as the CSUB and W defines of `gdn_chunk_scan.comp`. CSUB is the sub-chunk of
/// tokens that one solve step covers, aligned to the chunk start. A change of
/// CSUB changes the summation order of every output. W is the state columns of
/// one workgroup and its lane count, so it must stay a multiple of 32, the
/// RDNA2 wave width.
pub const gdn_scan_sub: u32 = 16;

pub const gdn_scan_cols: u32 = 32;

/// The value-head width the GDN chunk kernels support. The build feeds it to
/// glslc as the DH define of the scan. Each lane of the scan holds its whole
/// state column in registers, which needs a compile-time width.
pub const gdn_head_dim: u32 = 128;

/// Input elements the subgroup matvec holds in shared memory. The build feeds
/// it to glslc as the LDS_CAPACITY define of `linear_q8_0_subgroup.comp`, and
/// the kernel wrapper refuses a weight wider than this rather than corrupt
/// memory. 4096 f32 elements occupy 16 KiB of the 32 KiB budget.
pub const subgroup_lds_capacity: u32 = 4096;

/// Largest decoder stack the engine allocates scratch for. SmolLM2-135M has 30,
/// Llama-3.1-8B has 32, and Llama-3.1-70B has 80. 64 covers the 8B with headroom.
pub const max_layers: u32 = 64;

/// Largest vocab the engine allocates token tables for. Llama-3.1 uses 128k,
/// SmolLM2 uses 49k, and Qwen3.5 uses 248k. 262144 clears every one of them.
pub const max_vocab: u32 = 262_144;

/// Largest model context the engine accepts. Qwen3.5 declares a 256K context,
/// so the bound is 1<<18. A file that declares more loads with its context
/// capped here, and the loader logs both numbers. It is the one engine-wide
/// window bound: it sizes the static arrays and backs the kernel asserts. Each
/// model gets its own ceiling under it from `contextCeiling`.
pub const max_model_ctx: u32 = 1 << 18; // 262144

/// Ceiling on `--runs` of a model bench. The bench holds one result row per run.
pub const bench_runs_max: u32 = 1000;

/// KV bytes one session can hold when no flag names a window. `run` and
/// `bench-model` take the larger budget. Each serve session takes the smaller
/// one. The costliest KV
/// in the registry, an 8B at f32, keeps windows of 8192 and 512 under these two
/// figures. A cheaper KV family gets a deeper window. The model picker shows
/// both windows of each model before a run.
pub const run_session_kv_bytes: u64 = 2 << 30;

pub const serve_session_kv_bytes: u64 = 128 << 20;

/// A default window rounds down to a multiple of this token count.
pub const window_granule: u32 = 256;

/// Ceiling on rows per `stepBatch`, the conversations advanced per engine tick.
/// It also sizes every `[max_batch]` stack array on the step path, so it must
/// stay small. On the CPU and hybrid runners it also bounds physical KV slots,
/// because those runners cannot evict. The fused-GPU runners take their slot
/// ceiling from `max_vram_slots` instead.
pub const max_batch: u32 = 8;

/// Ceiling on *physical* VRAM KV slots, the conversations a device holds
/// resident at once. VRAM fits far more slots than one tick decodes, so this is
/// a separate bound from `max_batch`, the decode width. The fused-GPU runners
/// split the two: a KV budget buys up to this many slots, of which at most
/// `max_batch` decode per tick. 128 is the largest slot count a 16 GB card
/// funds at f16 KV for the 9B model. A raise lands in two places: this
/// constant, and the clamp anchors in `core/kv_budget.zig`.
pub const max_vram_slots: u32 = 128;

/// Ceiling on a serve PREFILL chunk: prompt tokens ingested in one forward
/// pass, independent of `max_batch`, the decode-concurrency dial. Decode rows
/// stay bounded by `max_batch`. A prefilling session can fill up to this many
/// rows, so a long prompt is not dribbled in `max_batch`-sized chunks. It
/// bounds the runner's row scratch and caps each family's `prefill_batch_max`
/// (llama fused holds 256 rows, qwen fused 128, and serve clamps a request to
/// the runner's real width at boot). An explicit `--prefill-chunk N`
/// is bounded by it.
pub const max_prefill_chunk: u32 = 256;

/// Rows of one prompt chunk on the CPU runners in `run` and `bench-model`. One
/// weight read serves every row of a chunk, and only the last row runs the
/// lm_head. The SIMD batch kernel reads each input row once per 4-row weight
/// group, so a wide chunk leaves the L2 cache. Serve sizes its own chunk.
pub const cpu_prefill_chunk: u32 = 64;

/// Ceiling on the threads of one CPU pool, the calling thread included. The
/// pool packs the participant count of a dispatch into 8 bits of its epoch
/// word, and it holds its workers in a fixed array of this size.
pub const max_threads: u32 = 64;

/// Ceiling on the threads that record spans while a trace runs. A thread past
/// it records nothing, and the trace metadata counts it.
pub const span_threads_max: u32 = 64;

/// The default capacity of one trace ring: the spans of one thread, and the
/// GPU submit records of the process. A span costs 24 bytes, so the default
/// ring of one thread holds 384 KiB. A full ring drops the entry and counts
/// it.
pub const spans_per_thread: u32 = 16 * 1024;

/// The largest ring capacity that `trace_spans_env` sets. A thread allocates
/// its ring at its first span, so this value prices 96 MiB per recording thread.
pub const spans_per_thread_max: u32 = 4 * 1024 * 1024;

/// The environment variable that sets the capacity of every trace ring. Per-op
/// spans under `-Dtrace` fill the default ring in seconds.
pub const trace_spans_env = "ZIGFERENCER_TRACE_SPANS";

/// Ceiling on *logical* serve sessions, the conversations the scheduler tracks
/// at once. Under the tiered KV residency (`serve/residency.zig`) a logical
/// session is either resident in a physical VRAM slot, up to `max_vram_slots`,
/// or evicted to a RAM blob, so the logical count decouples from VRAM. The
/// residency table is heap, so this bounds memory, not stack. Only the
/// fused-GPU build can over-subscribe, because only it can evict. CPU builds
/// keep logical == physical. The budget arm takes the min of this constant and
/// `max_vram_slots` in `cli/serve_common.zig`.
pub const max_logical_sessions: u32 = 128;

/// Ceiling on live serve connections, the clients one server holds at once. A
/// binary connection costs two worker threads and one payload buffer of 64 KiB,
/// and every connection holds one queue of 256 events. The cap therefore bounds
/// threads and memory. `--max-connections` sets the live count, and
/// `cli/serve_common.zig` refuses a count past this cap.
pub const max_connections: u32 = 64;

/// Ceiling on the time to live of a session lease, one hour. A detached
/// session holds a slot until its deadline, so the cap bounds that hold. The
/// server maximum and every granted time to live stay at or below it.
pub const max_resume_ttl_ms: u32 = 3_600_000;

/// Ceiling on the ids the serve scheduler forces when a think budget runs out:
/// the closing tag, then the separator after it. Each slot holds a queue of
/// this length, so the bound sizes a per-slot array.
pub const think_force_ids_max: u8 = 4;

/// Largest sampler top-k candidate pool (stack-allocated in `sample`). A
/// positive top-k runs every cut within this pool. A top-k of 0 runs them
/// over the full vocab.
pub const max_top_k: u32 = 256;

/// Largest presence penalty, either sign: the OpenAI range. The wire guard
/// and the model registry check against it.
pub const max_presence_penalty: f32 = 2;

/// Largest repetition penalty factor. The rule divides a positive logit by
/// the factor and multiplies a negative one by it. At 2 a repeated id keeps
/// half of a positive logit, which already drops it far down a typical row.
/// The common values lie between 1 and 1.5. The wire guard and the model
/// registry check against it.
pub const max_repetition_penalty: f32 = 2;

/// Longest model name the serve layer carries in a session handshake. Bounds
/// the inline name array in `serve/event.zig` and the handshake reply. A name
/// beyond this is rejected at the trust boundary, never truncated. A serve
/// bound, not a wire-format fact, so it lives here rather than in the framing.
pub const max_model_name_bytes: usize = 64;

/// Longest detokenized piece a streamed token event carries. Checked against
/// the whole vocab at server start, never truncated in silence. Llama-3's BPE
/// holds pieces past 64 bytes (runs of dashes, multi-byte scripts). 256 clears
/// every vocab in the registry with headroom. Bounds the inline text array in
/// `serve/event.zig`. A serve bound, not a wire-format fact.
pub const max_token_text_bytes: usize = 256;

/// Most (user, assistant) pairs in one `session_append` frame. The reply holds
/// one u32 position per pair inside a fixed-size out event. This bound keeps
/// that reply inside the largest server frame. A client sends a longer history
/// in several frames.
pub const max_append_pairs: u16 = 64;

/// OpenAI HTTP connector (`serve-openai`) bounds. HTTP + JSON is untrusted
/// input, so each is a hard guard at the boundary, never a hint. A request
/// past a limit gets a 4xx, never a truncation or an assert.
pub const openai_max_request_bytes: usize = 256 * 1024; // POST body cap (413 past it)

pub const openai_max_header_bytes: usize = 16 * 1024; // http.Server head buffer (431 past it)
pub const openai_max_messages: usize = 64; // messages[] length cap

/// The `max_tokens` of a request that omits it. 0 is the whole window.
pub const openai_default_max_tokens: u32 = 0;

/// Floor on a budget-derived per-session KV window. Below this a session is too
/// cramped to be useful. The planner (`core/kv_budget.zig`) refuses a budget
/// whose derived window falls under it, loudly, rather than allocating a
/// near-useless context.
pub const min_kv_window: u32 = 128;

/// The prefix-cache RAM budget `serve` uses when `--kv-ram` names no budget.
pub const serve_prefix_ram_default: u64 = 1 << 30;

/// Ceiling on the serve prefix cache's disk-index rows. The index holds one
/// key + token sequence per on-disk entry, and the boot scan skips files past
/// this bound with a warning. A capped index must never read as complete.
pub const serve_prefix_disk_index_max: usize = 1024;

/// Ceiling on `--preload` files per serve boot. Each one costs a full prefill
/// on a cold cache, so the bound keeps boot time a deliberate operator choice.
pub const serve_preload_max: usize = 8;

/// Ceiling on one `--preload` file's bytes. Matches the serve layer's text
/// scale (a system prompt, not a document). An oversize file fails the boot.
pub const serve_preload_text_max: usize = 64 * 1024;

/// Bytes one swap stage holds: the chunk that one staged copy moves between a
/// KV slot and a host blob. A swap walks the blob in chunks of this size, so
/// the stage RAM stays fixed at any window. The rule, fixed before the sweep:
/// the smallest chunk whose evict and restore rates both reach 95% of the
/// whole-blob rate. The `bench-swap --chunk-mib` sweep on a 67.3 MiB blob of
/// Qwen3.5-2B at depth 4096 measured the whole copy at 11.80 ms evict and
/// 444.93 ms restore. 256 MiB reached 100% and 98%. 64 MiB evicted at 85% and
/// 16 MiB at 47%, so both fail the rule.
pub const swap_stage_chunk_bytes: u64 = 256 << 20;

/// RAM a printed fit keeps back from the usable RAM. Free RAM moves by tens of
/// MiB between two boots, so a fit at the exact edge can refuse on the re-run.
/// The checks stay exact, and only the advice keeps this margin.
pub const ram_fit_headroom_bytes: u64 = 128 << 20;

/// The RAM budget the `run` path's prompt cache uses.
pub const prompt_cache_ram_default: u64 = 4 << 30;

/// The disk budget the `run` path's prompt cache uses.
pub const prompt_cache_disk_default: u64 = 16 << 30;

/// Usable share of the *live* device-local VRAM budget that a fused-GPU runner
/// can plan to occupy. `VK_EXT_memory_budget` reports that budget: what the
/// desktop leaves free on a shared GPU, not the physical heap size. The 1/8
/// reserve absorbs what the planned total omits or cannot predict. Those are
/// the transient weight-upload staging buffer, allocator fragmentation, the
/// activation scratch, and the driver's own overhead. The Qwen runner drives
/// that fragmentation with one device allocation per weight, about 200 of them.
/// The device destabilizes well before the heap is full. The runner therefore
/// refuses over this line with a typed error, rather than letting
/// `vkAllocateMemory` run out mid-init and remove the device.
pub const vram_usable_num: u64 = 7;

pub const vram_usable_den: u64 = 8;

/// Usable share of the RAM that `MemAvailable` reports at boot, which a hybrid
/// placement can plan to occupy with anonymous memory: host KV, swap stages,
/// blobs, and the prefix cache. The 1/8 reserve stays for what the plan does
/// not price. Those are the OS, the process heap, and the page cache that backs
/// the mapped weights. An anonymous overrun crashes the process, so the planner
/// refuses over this line with a typed error. The rule matches the VRAM rule,
/// but the constants stay separate because the reserve has a different reason.
pub const ram_usable_num: u64 = 7;

pub const ram_usable_den: u64 = 8;

/// Elements one KV row can hold: the KV heads times the head width. The bound
/// keeps every byte product of a KV layout inside a `u64`.
pub const max_kv_row_elems: u32 = 1 << 20;

/// Elements one KV region can hold. The shaders index KV with a 32-bit value,
/// and SPIR-V reads an access-chain index as signed, so the last index must
/// stay below 2^31.
pub const max_kv_index_elems: u64 = 1 << 31;

/// The environment variable that replaces the reported storage-buffer range.
/// A driver can report the spec minimum while it reads correctly past it. An
/// operator who proved a larger range on a device can state that range here.
pub const storage_range_env = "ZIGFERENCER_VK_STORAGE_RANGE";

/// The largest value `storage_range_env` takes. Vulkan states the range as a
/// 32-bit count.
pub const storage_range_max: u64 = std.math.maxInt(u32);

// -- types ------------------------------------------------------------------

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

/// The usable share of `available_bytes` of VRAM, rounded down.
pub fn vramUsable(available_bytes: u64) u64 {
    const usable = available_bytes / vram_usable_den * vram_usable_num;
    std.debug.assert(usable <= available_bytes);
    // The reserve is never empty, so a plan never takes the whole budget.
    if (available_bytes > 0) std.debug.assert(usable < available_bytes);

    return usable;
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
/// VRAM (the live budget, or the heap size when the budget extension is absent).
pub fn vramPlanFits(planned_bytes: u64, available_bytes: u64) bool {
    return planned_bytes <= vramUsable(available_bytes);
}

/// Check a buffer of `elems` elements and `bytes` bytes against the device
/// limits. The checks run in the `BufferVerdict` order, and each bound is
/// inclusive. A buffer that starts a binding must hold a whole number of
/// alignment units, so the next region packed after it starts aligned.
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

/// Shared memory of one GDN scan workgroup in bytes: the k and q rows of a
/// sub-chunk, the delta of the owned columns, the two pair-product tables, and
/// four per-token rows. It mirrors the `shared` arrays of `gdn_chunk_scan.comp`.
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

/// The cell that lane `lane` of workgroup `group` owns. It mirrors the index
/// arithmetic at the top of `gdn_chunk_scan.comp`. The workgroups of one head
/// are adjacent, and lane `lane` of column block `c` owns column
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
    // The split pass holds the scores, the q rows, and the LDS trees of its
    // queries in the 32 KiB of shared memory that the target driver allows.
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
    // The GDN scan splits a head into whole workgroups of whole waves. Its
    // shared memory holds the k and q rows of a sub-chunk, the delta of the
    // owned columns, the two pair products, and four per-token rows.
    std.debug.assert(gdn_scan_cols % 32 == 0);
    std.debug.assert(gdn_head_dim % gdn_scan_cols == 0);
    std.debug.assert(gdn_scan_sub >= 1);
    std.debug.assert(gdn_scan_sub <= gdn_scan_cols);
    std.debug.assert(gdnScanLdsBytes() <= 32 * 1024);
    // A CPU prompt chunk fits the stack row array of `prefillBatch`, which
    // holds `max_prefill_chunk` rows.
    std.debug.assert(cpu_prefill_chunk >= 1);
    std.debug.assert(cpu_prefill_chunk <= max_prefill_chunk);
    // The epoch word of the pool holds the participant count in 8 bits.
    std.debug.assert(max_threads >= 2);
    std.debug.assert(max_threads < 0xff);
    // A lease deadline is the drop time plus at most this many milliseconds.
    // The sum stays a 64-bit count of nanoseconds, far from the clock range.
    std.debug.assert(max_resume_ttl_ms >= 1);
    std.debug.assert(@as(u64, max_resume_ttl_ms) * std.time.ns_per_ms <= std.math.maxInt(i64));
}

// -- tests ------------------------------------------------------------------

test "vramPlanFits: leaves a reserve below the available budget" {
    // 8 GiB available -> 7 GiB usable.
    try std.testing.expect(vramPlanFits(7 << 30, 8 << 30));
    try std.testing.expect(!vramPlanFits((7 << 30) + 1, 8 << 30));
    // A plan that wants the whole budget is refused, because the reserve is
    // load-bearing.
    try std.testing.expect(!vramPlanFits(8 << 30, 8 << 30));
    // Zero budget (e.g. a failed query) refuses any positive plan.
    try std.testing.expect(!vramPlanFits(1, 0));
}

test "vramUsable: seven eighths of the budget, rounded down" {
    try std.testing.expectEqual(@as(u64, 7 << 30), vramUsable(8 << 30));
    try std.testing.expectEqual(@as(u64, 939_524_096), vramUsable(1 << 30));
    // The rounding happens on the eighth, so a budget under 8 bytes has none.
    try std.testing.expectEqual(@as(u64, 0), vramUsable(7));
    try std.testing.expectEqual(@as(u64, 7), vramUsable(15));
    try std.testing.expectEqual(@as(u64, 0), vramUsable(0));
}

test "ramUsable: seven eighths of the free RAM, rounded down, with a reserve that never empties" {
    // The anchor host: 22 GiB free leaves 20669530112 usable bytes.
    try std.testing.expectEqual(@as(u64, 20_669_530_112), ramUsable(23_622_320_128));
    try std.testing.expectEqual(@as(u64, 0), ramUsable(7));
    try std.testing.expectEqual(@as(u64, 7), ramUsable(15));
    try std.testing.expectEqual(@as(u64, 0), ramUsable(0));
    // The bound over a fixed grid of 4096 figures plus the `u64` ceiling:
    // the share stays under the input, and it never falls as the input rises.
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
    try std.testing.expectEqual(spans_per_thread, try parseTraceSpans("16384"));
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
