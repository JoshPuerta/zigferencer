//! Hand-curated table of models the engine knows by short name. Each entry
//! carries the on-disk path, the HF resolve URL, the chat-template family the
//! model expects, the KV dtype it is verified at, and a lower context ceiling
//! where the file cannot support its declared context. It is the canonical
//! list of the models the engine is verified against. `fetch <pub_name>` and the
//! run/inspect/bench commands resolve a short name through it. An ad-hoc
//! download still works through `fetch <org>/<repo>/<file>` (see
//! `cli/fetch_cmd.zig`) and does not appear here.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const chat = @import("chat.zig");
const dtype = @import("dtype.zig");
const limits = @import("limits.zig");
const log = std.log.scoped(.run);

// -- constants --------------------------------------------------------------

const models = [_]Model{
    // Primary small model. Llama-family, ByteLevel BPE tokenizer.
    // The upstream HuggingFaceTB GGUF repo is gated. bartowski's mirror is public.
    .{
        .pub_name = "SmolLM2-135M-Instruct-Q8_0",
        .local_path = "bartowski_SmolLM2-135M-Instruct-GGUF",
        .file_name = "SmolLM2-135M-Instruct-Q8_0.gguf",
        .download_url = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf",
        .chat_template_kind = .chatml_smollm2,
    },
    // Stretch target. The simd_pf CPU backend runs it end to end.
    .{
        .pub_name = "Meta-Llama-3.1-8B-Instruct-Q8_0",
        .local_path = "bartowski_Meta-Llama-3.1-8B-Instruct-GGUF",
        .file_name = "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        .download_url = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        .chat_template_kind = .llama3,
        // f16 KV is the default: outputs are byte identical to f32 on every
        // arm, latency is at parity or ahead, and the 32-token reference hash
        // does not move. Half the KV bytes buy double the serve residency and
        // 1.78x swap.
        .kv_dtype_default = .f16,
    },
    // Q4_0 variants for the GPU bandwidth-win measurement (decode is
    // memory-bound, and Q4_0 halves the weight stream). bartowski ships plain
    // Q4_0 for SmolLM2. For Llama-3.1-8B bartowski only has the ARM-repacked
    // Q4_0_4_4/_4_8/_8_8 (incompatible layout), so the entry uses QuantFactory's
    // plain Q4_0.
    .{
        .pub_name = "SmolLM2-135M-Instruct-Q4_0",
        .local_path = "bartowski_SmolLM2-135M-Instruct-GGUF",
        .file_name = "SmolLM2-135M-Instruct-Q4_0.gguf",
        .download_url = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_0.gguf",
        .chat_template_kind = .chatml_smollm2,
    },
    .{
        .pub_name = "Meta-Llama-3.1-8B-Instruct-Q4_0",
        .local_path = "QuantFactory_Meta-Llama-3.1-8B-Instruct-GGUF",
        .file_name = "Meta-Llama-3.1-8B-Instruct.Q4_0.gguf",
        .download_url = "https://huggingface.co/QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct.Q4_0.gguf",
        .chat_template_kind = .llama3,
        // The file declares 131072 positions but carries no llama3 RoPE
        // factors, so positions past the original 8192 context are outside
        // its training.
        .context_limit = 8192,
    },
    // Qwen3.5 hybrid Gated-DeltaNet + full-attention models. The 2B is the
    // dev/correctness target (small, fast, low VRAM). The 9B is the
    // measurement target. ChatML template (see `core/chat.zig`). Only
    // k-quants + Q8_0 ship (no Q4_0).
    .{
        .pub_name = "Qwen3.5-2B-Q8_0",
        .local_path = "lmstudio-community_Qwen3.5-2B-GGUF",
        .file_name = "Qwen3.5-2B-Q8_0.gguf",
        .download_url = "https://huggingface.co/lmstudio-community/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q8_0.gguf",
        // The model file's template thinks only when enable_thinking is true.
        .chat_template_kind = .{ .chatml_qwen = .off },
        // f16 KV is the default: outputs are byte identical across backends
        // and dtypes at depth, fused prefill is 7% ahead, and every other arm
        // is at parity. The fixed GDN state stays f32, so blobs shrink to
        // 0.64x.
        .kv_dtype_default = .f16,
        // The card values for text tasks, from
        // https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/README.md.
        .sampling = .{
            .thinking = .{
                .temperature = 1.0,
                .top_k = 20,
                .top_p = 0.95,
                .presence_penalty = 1.5,
                .min_p = 0,
                .repetition_penalty = 1,
            },
            .non_thinking = .{
                .temperature = 1.0,
                .top_k = 20,
                .top_p = 1.0,
                .presence_penalty = 2.0,
                .min_p = 0,
                .repetition_penalty = 1,
            },
        },
    },
    .{
        .pub_name = "Qwen3.5-9B-Q8_0",
        .local_path = "lmstudio-community_Qwen3.5-9B-GGUF",
        .file_name = "Qwen3.5-9B-Q8_0.gguf",
        .download_url = "https://huggingface.co/lmstudio-community/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q8_0.gguf",
        // The model file's template thinks unless enable_thinking is false.
        .chat_template_kind = .{ .chatml_qwen = .on },
        // f16 KV is the default: the 2B shape repeats at four times the scale,
        // and about 110 slots fit in free VRAM where f32 fits about 88.
        .kv_dtype_default = .f16,
        // The card values for general tasks, from
        // https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/README.md.
        .sampling = .{
            .thinking = .{
                .temperature = 1.0,
                .top_k = 20,
                .top_p = 0.95,
                .presence_penalty = 1.5,
                .min_p = 0,
                .repetition_penalty = 1,
            },
            .non_thinking = .{
                .temperature = 0.7,
                .top_k = 20,
                .top_p = 0.8,
                .presence_penalty = 1.5,
                .min_p = 0,
                .repetition_penalty = 1,
            },
        },
    },
};

// The card values cross the wire in every `hello_ack`, and the client
// decoder applies the wire guards to them. A card outside the guards makes
// every hello malformed, so the build refuses it.
comptime {
    for (models) |m| {
        const card = m.sampling orelse continue;
        for ([_]Recommended{ card.thinking, card.non_thinking }) |r| {
            std.debug.assert(r.temperature >= 0);
            std.debug.assert(r.top_p > 0 and r.top_p <= 1);
            std.debug.assert(r.top_k <= limits.max_top_k);
            std.debug.assert(@abs(r.presence_penalty) <= limits.max_presence_penalty);
            std.debug.assert(r.min_p >= 0 and r.min_p <= 1);
            std.debug.assert(
                r.repetition_penalty > 0 and r.repetition_penalty <= limits.max_repetition_penalty,
            );
        }
    }
}

/// Environment variable naming a secondary models store on another (e.g.
/// bigger) disk: `ZIGFERENCER_MODELS_DIR=/data/big/models`. The per-model
/// subdirectory (`<org>_<repo>/`) is appended underneath it. Reads check the
/// repo-local `models/` first and then this store (`locate`). When the
/// variable is set, `fetch` downloads into the store rather than `models/`.
pub const models_dir_env = "ZIGFERENCER_MODELS_DIR";

pub const default_models_root = "models";

// -- types ------------------------------------------------------------------

/// The sampling that a model card recommends for one thinking mode. The
/// engine has no frequency penalty, so a card value for it stays out. The
/// card values at their neutral settings change nothing.
pub const Recommended = struct {
    temperature: f32,
    top_k: u32,
    top_p: f32,
    presence_penalty: f32,
    /// 0 is neutral.
    min_p: f32,
    /// 1 is neutral.
    repetition_penalty: f32,
};

/// The card sampling of each thinking mode, for the text tasks of the card.
pub const SamplingCard = struct {
    thinking: Recommended,
    non_thinking: Recommended,
};

/// Static descriptor for a model the engine knows how to fetch and run.
/// All fields are string slices into the binary's read-only data.
pub const Model = struct {
    /// Stable name the user types on the CLI. Match the `gguf` filename
    /// stem (without the `.gguf` extension) where reasonable.
    pub_name: []const u8,
    /// Per-model subdirectory under the models root (see `modelDir`).
    /// Convention: `<org>_<repo>`. The root is `models/` by default, or
    /// `$ZIGFERENCER_MODELS_DIR` when set.
    local_path: []const u8,
    /// Bare filename of the GGUF on disk.
    file_name: []const u8,
    /// Raw download URL. For Hugging Face, this must be the
    /// `/resolve/<rev>/<file>` form, not the `/blob/<rev>/<file>` form.
    download_url: []const u8,
    /// Which prompt wrap `--chat` applies. `.none` means pass-through (callers
    /// can inline their own template). A Qwen entry carries the thinking mode
    /// that its model file defaults to. See `core/chat.zig`.
    chat_template_kind: chat.Kind = .none,
    /// The KV dtype this model is verified to run when no flag names one. An
    /// entry changes only on its own verification record, which re-anchors that
    /// model's no-flag token hashes alone. It follows the `chat_template_kind`
    /// precedent: one behavioral default per entry, recorded where the verified
    /// facts already live.
    kv_dtype_default: dtype.KvDtype = .f32,
    /// A lower window ceiling than the context the file declares, for a file
    /// that cannot support its declared context. Null keeps the declared
    /// context. `limits.contextCeiling` applies it.
    context_limit: ?u32 = null,
    /// The sampling the model card recommends. Null is a model with no card
    /// values, and a client then keeps greedy.
    sampling: ?SamplingCard = null,
};

// -- api --------------------------------------------------------------------

/// Apply the default policy to a raw env value: null (unset) or empty maps
/// to `models`. The env itself is read once at startup (the Io model threads
/// it through `std.process.Init.environ_map`), and the resolved root goes with
/// the CLI Context. These path helpers therefore stay pure string joins.
pub fn resolveRoot(env_value: ?[]const u8) []const u8 {
    const v = env_value orelse return default_models_root;
    return if (v.len == 0) default_models_root else v;
}

/// Directory holding `model`'s GGUF under `root`: `<root>/<local_path>`.
pub fn modelDir(arena: Allocator, root: []const u8, model: Model) Allocator.Error![]const u8 {
    return std.fs.path.join(arena, &.{ root, model.local_path });
}

/// Full path where `model`'s GGUF is written / read under `root`.
pub fn modelFilePath(arena: Allocator, root: []const u8, model: Model) Allocator.Error![]const u8 {
    return std.fs.path.join(arena, &.{ root, model.local_path, model.file_name });
}

/// The path `model`'s GGUF is actually on disk at, or null when it is absent.
/// Checks the default `models/` location first, then the configured `root` (the
/// `$ZIGFERENCER_MODELS_DIR` store) when that differs, so a repo-local file
/// wins over the external store. A non-null result guarantees the file exists
/// there. It is not a guess at where a download lands.
pub fn locate(
    arena: Allocator,
    io: Io,
    root: []const u8,
    model: Model,
) Allocator.Error!?[]const u8 {
    var cwd = Io.Dir.cwd();
    const local = try std.fs.path.join(
        arena,
        &.{ default_models_root, model.local_path, model.file_name },
    );
    if (fileExists(io, &cwd, local)) {
        log.debug("model resolve: {s} -> local {s}", .{ model.pub_name, local });
        return local;
    }

    const configured = try modelFilePath(arena, root, model);
    if (!std.mem.eql(u8, configured, local) and fileExists(io, &cwd, configured)) {
        log.debug("model resolve: {s} -> store {s}", .{ model.pub_name, configured });
        return configured;
    }

    log.debug("model resolve: {s} -> absent", .{model.pub_name});
    return null;
}

/// Path to LOAD `model` from: where it is, or, when it is absent, the
/// configured target a fresh download lands in, so a load failure names that
/// location.
pub fn resolveLoadPath(
    arena: Allocator,
    io: Io,
    root: []const u8,
    model: Model,
) Allocator.Error![]const u8 {
    return (try locate(arena, io, root, model)) orelse try modelFilePath(arena, root, model);
}

pub fn all() []const Model {
    return &models;
}

pub fn lookup(pub_name: []const u8) ?*const Model {
    // Index-based on purpose: `&m` from `for (models) |m|` points at a stack
    // copy of the loop variable, which dangles when the function returns.
    var i: usize = 0;
    while (i < models.len) : (i += 1) {
        if (std.mem.eql(u8, models[i].pub_name, pub_name)) return &models[i];
    }

    return null;
}

/// The card sampling of the model `pub_name` for one thinking mode. Returns
/// null for an unknown name and for an entry with no card values.
pub fn recommendedSampling(pub_name: []const u8, thinking: bool) ?Recommended {
    const m = lookup(pub_name) orelse return null;
    const card = m.sampling orelse return null;
    const r = if (thinking) card.thinking else card.non_thinking;
    std.debug.assert(r.temperature >= 0);
    std.debug.assert(r.top_p > 0 and r.top_p <= 1);
    return r;
}

// -- helpers ----------------------------------------------------------------

fn fileExists(io: Io, dir: *Io.Dir, path: []const u8) bool {
    const f = dir.openFile(io, path, .{}) catch return false;
    f.close(io);
    return true;
}

// -- tests ------------------------------------------------------------------

test "lookup returns null for unknown" {
    try std.testing.expect(lookup("does-not-exist") == null);
}

test "the KV and context defaults match each entry's verification record" {
    // Every no-flag byte and token hash stands on these rows. A change to one
    // must break this test on purpose and cite its own verification record.
    // Every entry the list omits stays f32.
    const flipped = [_][]const u8{
        "Meta-Llama-3.1-8B-Instruct-Q8_0",
        "Qwen3.5-2B-Q8_0",
        "Qwen3.5-9B-Q8_0",
    };
    // Every entry the list omits keeps the context its file declares.
    const limited = [_]struct { name: []const u8, limit: u32 }{
        .{ .name = "Meta-Llama-3.1-8B-Instruct-Q4_0", .limit = 8192 },
    };
    for (all()) |m| {
        var want: dtype.KvDtype = .f32;
        for (flipped) |name| {
            if (std.mem.eql(u8, m.pub_name, name)) {
                want = .f16;
            }
        }

        try std.testing.expectEqual(want, m.kv_dtype_default);
        var want_limit: ?u32 = null;
        for (limited) |entry| {
            if (std.mem.eql(u8, m.pub_name, entry.name)) {
                want_limit = entry.limit;
            }
        }

        try std.testing.expectEqual(want_limit, m.context_limit);
    }
}

test "lookup finds first model by exact name" {
    const m = lookup(models[0].pub_name) orelse return error.TestUnexpectedNull;
    try std.testing.expectEqualStrings(models[0].pub_name, m.pub_name);
    try std.testing.expectEqual(@intFromPtr(&models[0]), @intFromPtr(m));
}

test "the qwen entries carry the card sampling of each mode, and the rest none" {
    for (all()) |m| {
        const qwen = std.mem.startsWith(u8, m.pub_name, "Qwen3.5-");
        try std.testing.expectEqual(qwen, m.sampling != null);
    }

    const think = recommendedSampling("Qwen3.5-2B-Q8_0", true).?;
    try std.testing.expectEqual(
        Recommended{
            .temperature = 1.0,
            .top_k = 20,
            .top_p = 0.95,
            .presence_penalty = 1.5,
            .min_p = 0,
            .repetition_penalty = 1,
        },
        think,
    );
    const plain = recommendedSampling("Qwen3.5-2B-Q8_0", false).?;
    try std.testing.expectEqual(
        Recommended{
            .temperature = 1.0,
            .top_k = 20,
            .top_p = 1.0,
            .presence_penalty = 2.0,
            .min_p = 0,
            .repetition_penalty = 1,
        },
        plain,
    );
    const big = recommendedSampling("Qwen3.5-9B-Q8_0", false).?;
    try std.testing.expectEqual(
        Recommended{
            .temperature = 0.7,
            .top_k = 20,
            .top_p = 0.8,
            .presence_penalty = 1.5,
            .min_p = 0,
            .repetition_penalty = 1,
        },
        big,
    );
    try std.testing.expectEqual(think, recommendedSampling("Qwen3.5-9B-Q8_0", true).?);
    try std.testing.expect(recommendedSampling("SmolLM2-135M-Instruct-Q8_0", false) == null);
    try std.testing.expect(recommendedSampling("does-not-exist", true) == null);
}
