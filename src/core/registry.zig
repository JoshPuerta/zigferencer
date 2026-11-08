//! Hand-curated table of models the engine knows by short name. It is the canonical list of the
//! models the engine is verified against.
//!
//! Comptime. Asserts keep every sampling card inside the `limits` ranges.
//!
//! Entries. Each entry carries the on-disk path, the HF resolve URL, the chat-template family the
//! model expects, the KV dtype it is verified at, and a lower context ceiling where the file cannot
//! support its declared context. `fetch <pub_name>` and the run, inspect, and bench commands
//! resolve a short name through it. An ad-hoc download uses `fetch <org>/<repo>/<file>` (see
//! `cli/fetch_cmd.zig`) and does not appear here.
//!
//! Models store. `ZIGFERENCER_MODELS_DIR` names a secondary store on another disk, such as
//! `/data/big/models`. The per-model subdirectory (`<org>_<repo>/`) goes underneath it. Reads
//! check the repo-local `models/` first and then the store (`locate`). When the variable is set,
//! `fetch` downloads into the store.
//!
//! KV defaults. An f16 default means outputs are byte identical to f32 and the reference hashes do
//! not move. Llama-3.1-8B Q8_0: latency at parity or ahead, double the serve residency, and 1.78x
//! swap. Qwen 2B: byte identical across backends and dtypes at depth, fused prefill 7 % ahead,
//! every other path at parity, and blobs shrink to 0.64x because the GDN state stays f32. Qwen 9B:
//! the 2B shape at four times the scale, with about 110 slots in free VRAM where f32 fits about 88.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const chat = @import("chat.zig");
const dtype = @import("dtype.zig");
const limits = @import("limits.zig");
const log = std.log.scoped(.run);

// -- constants --------------------------------------------------------------

const models = [_]Model{
    // Primary small model. The upstream HuggingFaceTB GGUF repo is gated, so the entry uses the
    // public bartowski mirror.
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
        .kv_dtype_default = .f16,
    },
    // Q4_0 halves the weight stream of the decode. Bartowski ships only ARM-repacked Q4_0 for
    // Llama-3.1-8B (incompatible layout), so that entry uses the plain Q4_0 of QuantFactory.
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
        // The file declares 131072 positions but carries no llama3 RoPE factors, so positions
        // past the original 8192 context are outside its training.
        .context_limit = 8192,
    },
    // Qwen3.5 hybrid models. The 2B is the correctness target, the 9B the measurement target.
    // Only k-quants and Q8_0 ship, with no Q4_0.
    .{
        .pub_name = "Qwen3.5-2B-Q8_0",
        .local_path = "lmstudio-community_Qwen3.5-2B-GGUF",
        .file_name = "Qwen3.5-2B-Q8_0.gguf",
        .download_url = "https://huggingface.co/lmstudio-community/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q8_0.gguf",
        // The model file's template thinks only when enable_thinking is true.
        .chat_template_kind = .{ .chatml_qwen = .off },
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

/// Environment variable that names the secondary models store.
pub const models_dir_env = "ZIGFERENCER_MODELS_DIR";

pub const default_models_root = "models";

// -- types ------------------------------------------------------------------

/// The sampling that a model card recommends for one thinking mode. The engine has no frequency
/// penalty, so a card value for it stays out.
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
    /// Stable name the user types on the CLI. It matches the `gguf` filename stem where possible.
    pub_name: []const u8,
    /// Per-model subdirectory under the models root, as `<org>_<repo>`. See `modelDir`.
    local_path: []const u8,
    /// Bare filename of the GGUF on disk.
    file_name: []const u8,
    /// Raw download URL. For Hugging Face, it must use the `/resolve/<rev>/<file>` form.
    download_url: []const u8,
    /// Which prompt wrap `--chat` applies. `.none` passes the prompt through. See `core/chat.zig`.
    chat_template_kind: chat.Kind = .none,
    /// The KV dtype this model is verified at when no flag names one. A change re-anchors that
    /// model's no-flag token hashes.
    kv_dtype_default: dtype.KvDtype = .f32,
    /// A lower window ceiling than the declared context. Null keeps it. `limits.contextCeiling`
    /// applies it.
    context_limit: ?u32 = null,
    /// The sampling the model card recommends. Null keeps a client greedy.
    sampling: ?SamplingCard = null,
};

// -- api --------------------------------------------------------------------

/// Map a raw env value to the models root: null or empty gives `models`. The caller reads the env
/// once at startup, so the path helpers stay pure string joins.
pub fn resolveRoot(env_value: ?[]const u8) []const u8 {
    const v = env_value orelse return default_models_root;
    return if (v.len == 0) default_models_root else v;
}

/// Directory holding `model`'s GGUF under `root`: `<root>/<local_path>`.
pub fn modelDir(arena: Allocator, root: []const u8, model: Model) Allocator.Error![]const u8 {
    return std.Io.Dir.path.join(arena, &.{ root, model.local_path });
}

/// Full path where `model`'s GGUF is written and read under `root`.
pub fn modelFilePath(arena: Allocator, root: []const u8, model: Model) Allocator.Error![]const u8 {
    return std.Io.Dir.path.join(arena, &.{ root, model.local_path, model.file_name });
}

/// The path `model`'s GGUF is on disk at, or null when it is absent. The repo-local `models/`
/// wins over the store at `root`. A non-null result names a file that exists.
pub fn locate(
    arena: Allocator,
    io: Io,
    root: []const u8,
    model: Model,
) Allocator.Error!?[]const u8 {
    var cwd = Io.Dir.cwd();
    const local = try std.Io.Dir.path.join(
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

/// Path to load `model` from: where it is, or the target of a fresh download when it is absent,
/// so a load failure names that location.
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
    // Index the table. `&m` from `for (models) |m|` points at a stack copy that dangles.
    for (0..models.len) |i| {
        if (std.mem.eql(u8, models[i].pub_name, pub_name)) return &models[i];
    }

    return null;
}

/// The card sampling of model `pub_name` for one thinking mode. Null for an unknown name or an
/// entry with no card values.
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
    // Every no-flag byte and token hash stands on these rows. A change to one must break this test
    // on purpose and cite its own verification record. Every entry the list omits stays f32.
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
