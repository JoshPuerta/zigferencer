//! The Zig build script. It declares the release executable, the developer
//! executable and its `dev` step, the `test` step (unit and integration tests,
//! with `zig fmt --check` wired in), and the `bench` step. It
//! also declares the `check` step (a type-check alone, for ZLS) and the
//! `check-fmt` helper. It reads `-Dops-backend=<name>` and resolves it
//! against the CPU features of the build target. Under `-Druntime=vulkan` it
//! also adds vulkan-zig, links `libvulkan`, and compiles every
//! `src/ops_vulkan/shaders/*.comp` to SPIR-V through `glslc`, for `@embedFile`
//! to use.
//!
//! Backend selection is a build-time decision, on purpose. `resolveOpsBackend`
//! picks the strongest variant the target supports (`auto`). It panics here when
//! the request names an unsupported variant, so the failure lands at build time
//! and never as a runtime SIGILL. The GPU runtime follows the same rule:
//! `-Druntime=vulkan` owns the shader compiles and the dependency wiring, so a
//! CPU-only build never touches glslc or libvulkan.
//!
//! Common invocations. The Makefile wraps most of these as short targets, and
//! `check-fmt` has no target of its own:
//!   zig build                                Debug, auto-picks strongest CPU backend
//!   zig build -Doptimize=ReleaseFast         release build (best perf)
//!   zig build install dev                    both executables from one build
//!   zig build -Dops-backend=naive            force the scalar correctness oracle
//!   zig build test --summary all             unit + integration tests + fmt check
//!   zig build bench                          op micro-benchmarks (kernels follow -Doptimize)
//!   zig build check                          type-check only, no artifact (used by ZLS)
//!   zig build check-fmt                      fail if any source is unformatted
//!   zig build test-install -Dtest-bin=unit -Dtest-filter="<name>" -Dllvm=true
//!                                            install one test binary at zig-out/test/
//!                                            for a debugger (the stem of a tests/ file
//!                                            names an integration test)
//!   zig build -Druntime=vulkan -Dops-backend=vulkan_fused
//!                                            GPU build (needs glslc + libvulkan-dev)

const std = @import("std");
const engine_limits = @import("src/core/limits.zig");

// -- types ------------------------------------------------------------------

/// One source for the split length of attention. The resident kernels
/// `attention.comp`, `attention_multi_stream.comp`, and `attention_batch.comp`
/// and the split pass size their LDS score row with `limits.attn_split_len`.
/// The split pass and the merge pass derive the split geometry of a row from
/// it, as the host does.
const attn_split_define = std.fmt.comptimePrint("ATTN_SPLIT={d}", .{engine_limits.attn_split_len});

/// One source for the widest head of the split pass. `attention_split.comp`
/// stages the q row of a head in LDS sized by `limits.attn_max_head_dim`, and
/// `AttentionSplit.recordSplit` asserts the same bound on the host.
const attn_d_max_define = std.fmt.comptimePrint("ATTN_D_MAX={d}", .{engine_limits.attn_max_head_dim});

/// One source for the query heads of one split workgroup. The split pass
/// sizes its LDS rows and picks its named registers by
/// `limits.attn_split_queries`, and `AttentionSplit.recordSplit` derives its
/// grid from the same constant.
const attn_queries_define = std.fmt.comptimePrint("ATTN_QUERIES={d}", .{engine_limits.attn_split_queries});

/// One source for each matmul tile geometry, for the GDN scan geometry, and for
/// the shared-memory capacity of the subgroup matvec: the constants of `src/core/limits.zig`. Every value
/// below reaches `glslc` as a `-D` define, and the kernel wrappers read the same
/// constant, so the tile the shader walks and the grid the host dispatches read
/// one number. A shader whose define is absent fails the compile, so the pair
/// cannot fall out of step in silence.
const q8_batch_defines = [_][]const u8{
    std.fmt.comptimePrint("B_TILE={d}u", .{engine_limits.q8_batch_b_tile}),
    std.fmt.comptimePrint("N_TILE={d}u", .{engine_limits.q8_batch_n_tile}),
};
const q4_batch_defines = [_][]const u8{
    std.fmt.comptimePrint("B_TILE={d}u", .{engine_limits.q4_batch_b_tile}),
    std.fmt.comptimePrint("N_TILE={d}u", .{engine_limits.q4_batch_n_tile}),
};
const gemm_defines = [_][]const u8{
    std.fmt.comptimePrint("BM={d}u", .{engine_limits.gemm_m_tile}),
    std.fmt.comptimePrint("BN={d}u", .{engine_limits.gemm_n_tile}),
};
const gdn_scan_defines = [_][]const u8{
    std.fmt.comptimePrint("CSUB={d}u", .{engine_limits.gdn_scan_sub}),
    std.fmt.comptimePrint("W={d}u", .{engine_limits.gdn_scan_cols}),
    std.fmt.comptimePrint("DH={d}u", .{engine_limits.gdn_head_dim}),
};
const subgroup_defines = [_][]const u8{
    std.fmt.comptimePrint("LDS_CAPACITY={d}", .{engine_limits.subgroup_lds_capacity}),
};

const OpsBackendOption = enum { auto, naive, simd, simd_pf, vulkan, vulkan_fused };
const OpsBackendResolved = enum { naive, simd, simd_pf, vulkan, vulkan_fused };
const RuntimeOption = enum { cpu, vulkan };
/// The GPU attention kits. Each entry names one kit in `src/ops_vulkan/kernels.zig`.
const AttnKernelOption = enum { split };

/// The shaders that `-Druntime=vulkan` compiles at build time. The name on the
/// left is the anonymous-import name that source code passes to `@embedFile`.
/// The name in the middle is the path under `src/ops_vulkan/shaders/`. To add a
/// kernel, add one line here and the `.comp` file. Nothing else changes.
///
/// `defines` reaches `glslc` as `-D<name>`, and it carries two kinds of value.
/// It selects a variant, so one source can produce several SPIR-V blobs: the
/// KV-dtype attention variants differ only in the element type of two buffers,
/// and a single source keeps the arithmetic of the variants from drifting
/// apart. It also carries the tile geometry the host shares with the shader, so
/// a grid and the tile it walks read one number.
const vulkan_shaders = [_]struct {
    import_name: []const u8,
    path: []const u8,
    defines: []const []const u8 = &.{},
}{
    .{ .import_name = "vec_add.spv", .path = "src/ops_vulkan/shaders/vec_add.comp" },
    .{ .import_name = "linear_q8_0.spv", .path = "src/ops_vulkan/shaders/linear_q8_0.comp" },
    .{ .import_name = "linear_q8_0_reduce.spv", .path = "src/ops_vulkan/shaders/linear_q8_0_reduce.comp" },
    .{ .import_name = "linear_q4_0_reduce.spv", .path = "src/ops_vulkan/shaders/linear_q4_0_reduce.comp" },
    .{
        .import_name = "linear_q8_0_subgroup.spv",
        .path = "src/ops_vulkan/shaders/linear_q8_0_subgroup.comp",
        .defines = &subgroup_defines,
    },
    .{
        .import_name = "linear_q8_0_batch.spv",
        .path = "src/ops_vulkan/shaders/linear_q8_0_batch.comp",
        .defines = &q8_batch_defines,
    },
    .{
        .import_name = "linear_q8_0_gemm.spv",
        .path = "src/ops_vulkan/shaders/linear_q8_0_gemm.comp",
        .defines = &gemm_defines,
    },
    .{
        .import_name = "linear_q4_0_batch.spv",
        .path = "src/ops_vulkan/shaders/linear_q4_0_batch.comp",
        .defines = &q4_batch_defines,
    },
    .{ .import_name = "rmsnorm_batch.spv", .path = "src/ops_vulkan/shaders/rmsnorm_batch.comp" },
    .{ .import_name = "rope_multi_batch.spv", .path = "src/ops_vulkan/shaders/rope_multi_batch.comp" },
    .{
        .import_name = "attention_batch.spv",
        .path = "src/ops_vulkan/shaders/attention_batch.comp",
        .defines = &.{attn_split_define},
    },
    .{
        .import_name = "attention_multi_stream.spv",
        .path = "src/ops_vulkan/shaders/attention_multi_stream.comp",
        .defines = &.{attn_split_define},
    },
    .{ .import_name = "residual_add.spv", .path = "src/ops_vulkan/shaders/residual_add.comp" },
    .{ .import_name = "swiglu.spv", .path = "src/ops_vulkan/shaders/swiglu.comp" },
    .{ .import_name = "rmsnorm.spv", .path = "src/ops_vulkan/shaders/rmsnorm.comp" },
    .{ .import_name = "rope.spv", .path = "src/ops_vulkan/shaders/rope.comp" },
    .{ .import_name = "rope_multi.spv", .path = "src/ops_vulkan/shaders/rope_multi.comp" },
    .{
        .import_name = "attention.spv",
        .path = "src/ops_vulkan/shaders/attention.comp",
        .defines = &.{attn_split_define},
    },
    .{ .import_name = "gdn_decode.spv", .path = "src/ops_vulkan/shaders/gdn_decode.comp" },
    .{ .import_name = "gdn_chunk_conv.spv", .path = "src/ops_vulkan/shaders/gdn_chunk_conv.comp" },
    .{
        .import_name = "gdn_chunk_scan.spv",
        .path = "src/ops_vulkan/shaders/gdn_chunk_scan.comp",
        .defines = &gdn_scan_defines,
    },
    .{ .import_name = "gdn_chunk_norm.spv", .path = "src/ops_vulkan/shaders/gdn_chunk_norm.comp" },
    .{ .import_name = "sigmoid_gate.spv", .path = "src/ops_vulkan/shaders/sigmoid_gate.comp" },
    .{ .import_name = "qwen_qk_norm.spv", .path = "src/ops_vulkan/shaders/qwen_qk_norm.comp" },
    .{ .import_name = "kv_commit_f16.spv", .path = "src/ops_vulkan/shaders/kv_commit_f16.comp" },
    .{
        .import_name = "attention_f16.spv",
        .path = "src/ops_vulkan/shaders/attention.comp",
        .defines = &.{ "KV_F16", attn_split_define },
    },
    .{
        .import_name = "attention_batch_f16.spv",
        .path = "src/ops_vulkan/shaders/attention_batch.comp",
        .defines = &.{ "KV_F16", attn_split_define },
    },
    .{
        .import_name = "attention_multi_stream_f16.spv",
        .path = "src/ops_vulkan/shaders/attention_multi_stream.comp",
        .defines = &.{ "KV_F16", attn_split_define },
    },
    .{
        .import_name = "attention_split.spv",
        .path = "src/ops_vulkan/shaders/attention_split.comp",
        .defines = &.{ attn_split_define, attn_d_max_define, attn_queries_define },
    },
    .{
        .import_name = "attention_split_f16.spv",
        .path = "src/ops_vulkan/shaders/attention_split.comp",
        .defines = &.{ "KV_F16", attn_split_define, attn_d_max_define, attn_queries_define },
    },
    .{
        .import_name = "attention_merge.spv",
        .path = "src/ops_vulkan/shaders/attention_merge.comp",
        .defines = &.{attn_split_define},
    },
};

// -- api --------------------------------------------------------------------

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const requested_backend = b.option(
        OpsBackendOption,
        "ops-backend",
        "Op kernel backend. 'auto' (default) picks the strongest variant whose OpRequires the target satisfies. 'naive' = scalar f32 reference (the correctness oracle). 'simd' = @Vector(8) AVX2+FMA path. The build fails if the target lacks AVX2.",
    ) orelse .auto;

    const runtime = b.option(
        RuntimeOption,
        "runtime",
        "Dispatch runtime. 'cpu' (default) = the ops/ variants alone. 'vulkan' = also compile src/ops_vulkan/shaders/*.comp to SPIR-V with glslc, link libvulkan, and pull vulkan-zig as a build dependency. It needs glslc on PATH and libvulkan-dev installed.",
    ) orelse .cpu;

    const ops_backend = resolveOpsBackend(requested_backend, target, runtime);

    const attn_kernel = b.option(
        AttnKernelOption,
        "attn-kernel",
        "GPU attention kit. 'split' (default) = the resident kernels inside one split and the split and merge passes past it. A CPU build ignores it.",
    ) orelse .split;

    const vk_registry = b.option(
        []const u8,
        "vk-registry",
        "Path to the Vulkan registry XML (vk.xml). The build reads it only under -Druntime=vulkan. The default is the Ubuntu and Debian libvulkan-dev location.",
    ) orelse "/usr/share/vulkan/registry/vk.xml";

    const trace_hot = b.option(
        bool,
        "trace",
        "Compile hot-path traces (the layer, dispatch, and token log scopes). The default is true for Debug and false for a release mode. Output still needs $ZIGFERENCER_LOG to turn the scope on.",
    ) orelse (optimize == .Debug);

    // GPU tests run one at a time on a shared device, so the suite needs a way
    // to run a single named test. A filtered-out test never executes, so its
    // binary touches no device.
    const test_filters = b.option(
        []const []const u8,
        "test-filter",
        "Run only tests whose name contains this string (repeatable).",
    ) orelse &.{};

    // The pool size of every runner test whose thread count is incidental.
    // The default of 4 runs the fan-out path in the suite. A value of 1 runs
    // the same tests serially.
    const test_threads = b.option(
        u32,
        "test-threads",
        "Pool threads of the runner tests whose thread count is incidental (default 4, at least 1). Serial reference legs keep 1 thread.",
    ) orelse 4;
    if (test_threads == 0) @panic("-Dtest-threads must be at least 1");
    const test_opts = b.addOptions();
    test_opts.addOption(u32, "test_threads", test_threads);
    const test_options_mod = test_opts.createModule();

    // The default x86_64 Debug backend steps by line in a debugger but emits
    // no local variables, so a debugging session builds with LLVM.
    const use_llvm = b.option(
        bool,
        "llvm",
        "Build both executables and every test binary with the LLVM code generator (default: the Zig choice for the target). A debugger shows local variables only in an LLVM Debug build.",
    );

    // A debugger runs one test binary as a normal executable. The name selects
    // the binary, and only the selected binary compiles for `test-install`.
    const test_bin = b.option(
        []const u8,
        "test-bin",
        "The test binary that test-install puts at zig-out/test/<name>: unit (the tests of src/) or the stem of a tests/*_test.zig file (default unit).",
    ) orelse "unit";

    // The backend ladder is cumulative, not exclusive. A GPU binary still owns
    // engines that the GPU does not cover, such as the qwen session engine on a
    // fused build. Those engines must run on the strongest CPU variant the
    // target supports, and never silently on naive. `cpu_ops` is that variant,
    // resolved on every build. `ops_backend` stays the PRIMARY engine tag.
    const cpu_ops: OpsBackendResolved = switch (ops_backend) {
        .naive, .simd, .simd_pf => ops_backend,
        .vulkan, .vulkan_fused => if (simdBackendSupportedBy(target.result.cpu)) .simd_pf else .naive,
    };

    const build_opts = b.addOptions();
    build_opts.addOption(bool, "cpu_ops_simd", cpu_ops == .simd);
    build_opts.addOption(bool, "cpu_ops_simd_pf", cpu_ops == .simd_pf);
    build_opts.addOption(bool, "use_vulkan_ops", ops_backend == .vulkan);
    build_opts.addOption(bool, "use_vulkan_fused_ops", ops_backend == .vulkan_fused);
    // Backend tag string for runtime self-reporting (bench-model prints it).
    build_opts.addOption([]const u8, "ops_backend", @tagName(ops_backend));
    build_opts.addOption(bool, "use_vulkan", runtime == .vulkan);
    build_opts.addOption([]const u8, "runtime", @tagName(runtime));
    build_opts.addOption(bool, "trace_hot", trace_hot);
    build_opts.addOption(AttnKernelOption, "attn_kernel", attn_kernel);
    const build_options_mod = build_opts.createModule();

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "build_options", .module = build_options_mod }},
        // The perf profiler walks the call stacks by the frame pointer.
        .omit_frame_pointer = false,
    });

    if (runtime == .vulkan) wireVulkan(b, exe_mod, vk_registry);

    const exe = b.addExecutable(.{
        .name = "zigferencer",
        .root_module = exe_mod,
        .use_llvm = use_llvm,
    });
    // A build id lets perf cache each binary by its id.
    exe.build_id = .fast;
    b.installArtifact(exe);

    // `zig build dev`: the developer executable, the release commands plus the
    // benches, the kernel checks, the demo, and the tokenizer dump. Its root
    // imports `exe_mod` and builds with the same target and optimize mode, so
    // the engine code of the two binaries is equal. The default install step
    // does not build it.
    const dev_mod = b.createModule(.{
        .root_source_file = b.path("src/dev_main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zigferencer", .module = exe_mod }},
        // The perf profiler walks the call stacks by the frame pointer.
        .omit_frame_pointer = false,
    });
    const dev_exe = b.addExecutable(.{ .name = "zigferencer-dev", .root_module = dev_mod, .use_llvm = use_llvm });
    dev_exe.build_id = .fast;
    const dev_step = b.step("dev", "Build and install zigferencer-dev");
    dev_step.dependOn(&b.addInstallArtifact(dev_exe, .{}).step);

    // `zig build check`: type-check the whole project without producing an
    // artifact. ZLS runs this step for diagnostics, so a named module import
    // such as `@import("zigferencer")` or `@import("build_options")` resolves
    // cleanly in the editor.
    const check_step = b.step("check", "Type-check without producing artifacts (used by ZLS)");
    const check_exe = b.addExecutable(.{ .name = "zigferencer-check", .root_module = exe_mod });
    check_step.dependOn(&check_exe.step);
    const check_dev_exe = b.addExecutable(.{ .name = "zigferencer-dev-check", .root_module = dev_mod });
    check_step.dependOn(&check_dev_exe.step);

    const run_step = b.step("run", "Run zigferencer");
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    run_step.dependOn(&run_cmd.step);

    const test_step = b.step("test", "Run zigferencer tests");
    // `test-build` compiles the test artifacts of the current flags and runs
    // none of them. It is the compile-check for a test that cannot run where
    // the build runs, such as a `-Druntime=vulkan` GPU equivalence test on a
    // host without a device. The flags also select the list: a Vulkan build
    // below `-Dops-backend=vulkan_fused` omits the fused-runner tests, because
    // that build re-exports no fused runner.
    const test_build_step = b.step("test-build", "Compile all tests without running them");
    const test_install_step = b.step("test-install", "Install the test binary of -Dtest-bin at zig-out/test/<name> for a debugger");
    var test_bin_names: std.ArrayList(u8) = .empty;
    var test_bin_found = false;
    const exe_tests = b.addTest(.{ .name = "unit", .root_module = exe_mod, .filters = test_filters, .use_llvm = use_llvm });
    test_bin_names.appendSlice(b.allocator, exe_tests.name) catch @panic("OOM");
    if (std.mem.eql(u8, test_bin, exe_tests.name)) {
        test_install_step.dependOn(&installTestBin(b, exe_tests).step);
        test_bin_found = true;
    }
    const run_exe_tests = b.addRunArtifact(exe_tests);
    // The model tests read files and environment variables that the build graph
    // does not see, so a cached pass can hide a missing model. Every test run
    // therefore runs again.
    run_exe_tests.has_side_effects = true;
    test_step.dependOn(&run_exe_tests.step);
    test_build_step.dependOn(&exe_tests.step);
    // The unit tests alone run in seconds in Debug, so a gate that runs the
    // whole suite in an optimized mode keeps their Debug checks here.
    const test_unit_step = b.step("test-unit", "Run the unit tests of src/ only");
    test_unit_step.dependOn(&run_exe_tests.step);

    // Integration tests live under tests/ and exercise the engine end-to-end
    // against real on-disk fixtures. Each test finds its fixture through
    // `test_fixtures`. A missing file skips the test when no models directory is
    // set, and fails it when the set directory lacks the file.
    const cpu_integration_tests = [_][]const u8{
        "tests/gguf_smollm2_test.zig",
        "tests/tokenizer_smollm2_test.zig",
        "tests/tokenizer_llama3_test.zig",
        "tests/tokenizer_qwen_test.zig",
        "tests/snapshot_smollm2_test.zig",
        "tests/llama_session_equivalence_test.zig",
        "tests/llama_prefix_runner_test.zig",
        "tests/serve_integration_test.zig",
        "tests/serve_resume_test.zig",
        "tests/client_compaction_test.zig",
        "tests/zmq_interop_test.zig",
        "tests/openai_integration_test.zig",
        "tests/qwen_session_equivalence_test.zig",
        "tests/qwen_prefix_runner_test.zig",
        "tests/cpu_prefill_chunk_test.zig",
        "tests/tidy_test.zig",
    };
    // The Vulkan lists follow the two levels of the Vulkan aliases in
    // `src/main.zig`. The list below names the tests that need the context and
    // the kernels alone. Each one references the `vulkan` module and the SPIR-V
    // anonymous imports, so it compiles under `-Druntime=vulkan` and a CPU build
    // skips it.
    const vulkan_integration_tests = [_][]const u8{
        "tests/vulkan_vec_add_test.zig",
        "tests/vulkan_kv_commit_f16_test.zig",
        "tests/vulkan_linear_q8_0_test.zig",
        "tests/vulkan_attention_test.zig",
        "tests/vulkan_prefill_kernels_test.zig",
        "tests/vulkan_stream_ring_test.zig",
    };
    // The list below names the tests that reach a fused or a hybrid runner.
    // `src/main.zig` re-exports those runners under `use_vulkan_fused_ops`
    // alone, and their device code can run under no other backend, so the build
    // appends this list only for `-Dops-backend=vulkan_fused`.
    const fused_integration_tests = [_][]const u8{
        "tests/llama_vulkan_serve_equivalence_test.zig",
        "tests/qwen_vulkan_serve_equivalence_test.zig",
        "tests/llama_hybrid_residency_test.zig",
        "tests/llama_hybrid_serve_test.zig",
    };
    // A test file that no list names never runs, and nothing else says so.
    requireListed(b, &(cpu_integration_tests ++ vulkan_integration_tests ++ fused_integration_tests));
    const integration_tests_paths: []const []const u8 = if (runtime != .vulkan)
        &cpu_integration_tests
    else if (ops_backend == .vulkan_fused)
        &(cpu_integration_tests ++ vulkan_integration_tests ++ fused_integration_tests)
    else
        &(cpu_integration_tests ++ vulkan_integration_tests);
    for (integration_tests_paths) |t_path| {
        const t_mod = b.createModule(.{
            .root_source_file = b.path(t_path),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zigferencer", .module = exe_mod },
                .{ .name = "test_options", .module = test_options_mod },
            },
        });
        // The interop test opens the system libzmq with `dlopen`. The pure-Zig
        // ELF loader does not resolve that library's own dependencies.
        if (std.mem.eql(u8, t_path, "tests/zmq_interop_test.zig")) t_mod.link_libc = true;
        const t_step = b.addTest(.{
            .name = std.fs.path.stem(t_path),
            .root_module = t_mod,
            .filters = test_filters,
            .use_llvm = use_llvm,
        });
        test_bin_names.print(b.allocator, ", {s}", .{t_step.name}) catch @panic("OOM");
        if (std.mem.eql(u8, test_bin, t_step.name)) {
            test_install_step.dependOn(&installTestBin(b, t_step).step);
            test_bin_found = true;
        }
        const run_t = b.addRunArtifact(t_step);
        run_t.has_side_effects = true;
        // The tidy test reads the source tree through relative paths.
        if (std.mem.eql(u8, t_path, "tests/tidy_test.zig")) run_t.setCwd(b.path("."));
        test_step.dependOn(&run_t.step);
        test_build_step.dependOn(&t_step.step);
    }
    // An unknown name fails the step with the valid names, and the other
    // steps of the build stay usable.
    if (!test_bin_found) test_install_step.dependOn(&b.addFail(b.fmt(
        "-Dtest-bin={s} names no test binary of this build. Valid names: {s}",
        .{ test_bin, test_bin_names.items },
    )).step);

    // `zig build check-fmt`: fail if anything is unformatted. The test step
    // depends on it, so unformatted code blocks the test suite.
    const check_fmt_step = b.step("check-fmt", "Verify all source is zig-fmt clean");
    const fmt_check = b.addFmt(.{
        .paths = &.{ "src", "tests", "tools", "build.zig", "build.zig.zon" },
        .check = true,
    });
    check_fmt_step.dependOn(&fmt_check.step);
    test_step.dependOn(&fmt_check.step);

    // `zig build bench`: micro-benchmarks for the op backends. The bench root
    // is always ReleaseFast, but the kernels come from the zigferencer module
    // and follow `-Doptimize`. A Debug build times Debug kernels, so the step
    // prints a note then. The bench imports `build_options` to gate its
    // Vulkan code on `use_vulkan` at comptime, so a CPU build omits it.
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("tools/bench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "zigferencer", .module = exe_mod },
            .{ .name = "build_options", .module = build_options_mod },
        },
    });
    const bench_exe = b.addExecutable(.{ .name = "zigferencer-bench", .root_module = bench_mod });
    bench_exe.build_id = .fast;
    const bench_step = b.step("bench", "Run op micro-benchmarks");
    const run_bench = b.addRunArtifact(bench_exe);
    if (optimize == .Debug) run_bench.step.dependOn(debugNoteStep(b, "bench times Debug kernels. Pass -Doptimize=ReleaseFast for real numbers."));
    if (b.args) |args| run_bench.addArgs(args);
    bench_step.dependOn(&run_bench.step);
    // `bench-build` compiles the bench binary and runs none of it. It is the
    // compile-check for bench code that cannot run where the build runs, such as
    // the `-Druntime=vulkan` kernel rows on a host without a device. Some
    // kernels have their only caller in the bench, so no other step type-checks
    // those calls.
    const bench_build_step = b.step("bench-build", "Compile the bench without running it");
    bench_build_step.dependOn(&bench_exe.step);

    // `zig build simd-probe`: build the SIMD-verification probe with the CPU
    // ops alone. The probe root is ReleaseFast, but the kernel bodies come from
    // the zigferencer module and follow `-Doptimize`, so the step prints a note
    // under Debug. The probe pins every op kernel to
    // a standalone symbol, so `tools/simd-report.sh` can read each one. The step
    // installs the artifact under zig-out and never runs it. The header of the
    // probe states why the shipping binary cannot serve instead.
    const probe_mod = b.createModule(.{
        .root_source_file = b.path("tools/simd_probe.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "zigferencer", .module = exe_mod },
            .{ .name = "build_options", .module = build_options_mod },
        },
    });
    const probe_exe = b.addExecutable(.{ .name = "zigferencer-simd-probe", .root_module = probe_mod });
    probe_exe.build_id = .fast;
    const probe_step = b.step("simd-probe", "Emit per-op target-native assembly (for tools/simd-report.sh)");
    // The emitted `.s` is target-native for any backend Zig can compile, so the
    // report works under cross-compilation and needs no multi-arch objdump. The
    // install path is fixed, so the script finds the file.
    const probe_asm = b.addInstallFile(probe_exe.getEmittedAsm(), "zigferencer-simd-probe.s");
    probe_step.dependOn(&probe_asm.step);
    if (optimize == .Debug) probe_step.dependOn(debugNoteStep(b, "simd-probe emits Debug assembly. Pass -Doptimize=ReleaseFast for the real report."));
}

// -- helpers ----------------------------------------------------------------

/// A step that prints `note` when the build graph runs it.
fn debugNoteStep(b: *std.Build, comptime note: []const u8) *std.Build.Step {
    const Note = struct {
        fn make(_: *std.Build.Step, _: std.Build.Step.MakeOptions) anyerror!void {
            std.debug.print("note: " ++ note ++ "\n", .{});
        }
    };
    const step = b.allocator.create(std.Build.Step) catch @panic("OOM");
    step.* = std.Build.Step.init(.{ .id = .custom, .name = "debug note", .owner = b, .makeFn = Note.make });
    return step;
}

/// Install a test binary at `zig-out/test/<name>`, apart from the release
/// executables in `zig-out/bin`.
fn installTestBin(b: *std.Build, t: *std.Build.Step.Compile) *std.Build.Step.InstallArtifact {
    std.debug.assert(t.kind == .@"test");
    return b.addInstallArtifact(t, .{ .dest_dir = .{ .override = .{ .custom = "test" } } });
}

/// Fail the build when a `tests/*_test.zig` file sits in none of the lists.
fn requireListed(b: *std.Build, listed: []const []const u8) void {
    const io = b.graph.io;
    // A fetched package can omit tests/. Then the check has nothing to read.
    var dir = b.build_root.handle.openDir(io, "tests", .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return,
        else => std.debug.panic("cannot open tests/: {s}", .{@errorName(err)}),
    };
    defer dir.close(io);
    var it = dir.iterate();
    while (it.next(io) catch |err| std.debug.panic("cannot read tests/: {s}", .{@errorName(err)})) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, "_test.zig")) continue;
        const found = for (listed) |path| {
            if (std.mem.eql(u8, path["tests/".len..], entry.name)) break true;
        } else false;
        if (!found) std.debug.panic("tests/{s} is in no test list in build.zig", .{entry.name});
    }
}

/// Map the user's `-Dops-backend=...` to a concrete variant. `.auto` picks the
/// strongest variant whose published `OpRequires` the target satisfies. On
/// x86_64 with AVX2+FMA that is `simd_pf`, a register-blocked microkernel with
/// software prefetch. Otherwise it is `naive`. An explicit choice on an
/// unsupported target fails the build rather than emitting an illegal
/// instruction at run time.
fn resolveOpsBackend(
    requested: OpsBackendOption,
    target: std.Build.ResolvedTarget,
    runtime: RuntimeOption,
) OpsBackendResolved {
    const cpu = target.result.cpu;
    const avx2_fma_ok = simdBackendSupportedBy(cpu);
    return switch (requested) {
        .naive => .naive,
        .simd => if (avx2_fma_ok) .simd else @panic("ops-backend=simd needs AVX2+FMA on x86_64. This target lacks it. Use -Dops-backend=naive or =auto."),
        .simd_pf => if (avx2_fma_ok) .simd_pf else @panic("ops-backend=simd_pf needs AVX2+FMA on x86_64. This target lacks it. Use -Dops-backend=naive or =auto."),
        // The vulkan backend wraps the GPU kernels behind the CPU op API. It
        // works only with the vulkan runtime, which owns the shader compiles
        // and the libvulkan link. The build fails here rather than at a
        // confusing missing-symbol link error.
        .vulkan => if (runtime == .vulkan) .vulkan else @panic("ops-backend=vulkan needs -Druntime=vulkan to wire the shaders and the libvulkan link."),
        // The vulkan-fused backend records the whole forward pass into one
        // command buffer and submits once per token. It needs the same runtime
        // and drives the same kernel set. It uses device-local scratch buffers
        // and a process-wide KV cache in place of a per-call host-visible
        // memcpy.
        .vulkan_fused => if (runtime == .vulkan) .vulkan_fused else @panic("ops-backend=vulkan_fused needs -Druntime=vulkan to wire the shaders and the libvulkan link."),
        .auto => if (avx2_fma_ok) .simd_pf else .naive,
    };
}

/// Local mirror of the simd-variant requirements in `core/op_requires.zig`:
/// `arch = x86_64`, `min_simd = avx2`, `needs_fma = true`. The `pub const
/// requires` of the variant file is the source of truth at run time and inside
/// the binary. This function is the build-time evaluator. The two must agree.
fn simdBackendSupportedBy(cpu: std.Target.Cpu) bool {
    if (cpu.arch != .x86_64) return false;
    const F = std.Target.x86.Feature;
    if (!std.Target.x86.featureSetHas(cpu.features, F.avx2)) return false;
    if (!std.Target.x86.featureSetHas(cpu.features, F.fma)) return false;
    return true;
}

/// Add vulkan-zig as a build dependency, link `libvulkan`, and compile every
/// `.comp` that `vulkan_shaders` lists to SPIR-V with `glslc`. The function adds
/// each compiled blob as an anonymous import under its `.spv` name, so source
/// can call `@embedFile("vec_add.spv")` wherever the build wrote the artifact.
/// The build calls this function only under `-Druntime=vulkan`.
fn wireVulkan(
    b: *std.Build,
    exe_mod: *std.Build.Module,
    vk_registry: []const u8,
) void {
    // vulkan-zig generates a single `vk.zig` from the registry XML. The
    // dependency exposes it as the `vulkan-zig` module once the build gives it a
    // registry path. See the vulkan-zig README, "Generation with the package
    // manager from build.zig".
    const vk_dep = b.dependency("vulkan", .{
        .registry = @as([]const u8, vk_registry),
    });
    const vk_mod = vk_dep.module("vulkan-zig");
    exe_mod.addImport("vulkan", vk_mod);

    // libvulkan is the loader entry point. The BaseDispatch of vulkan-zig calls
    // vkGetInstanceProcAddr from it. The libvulkan1 and libvulkan-dev packages
    // provide this system library.
    //
    // `link_libc = true` is load-bearing here. The Vulkan loader uses TLS, and
    // pthreads and malloc with it, and it segfaults inside its own data setup
    // when the executable does not link against libc. With `-lc` the TLS init
    // of the C runtime runs before main, and `libvulkan.so.1`, loaded as a
    // DT_NEEDED dependency, finds the slots it expects. Without it every
    // dispatched Vulkan call crashes at the first internal heap or TLS access.
    exe_mod.linkSystemLibrary("vulkan", .{});
    exe_mod.link_libc = true;

    // glslc compiles GLSL to SPIR-V at build time. `--target-env=vulkan1.2`
    // keeps the output within Vulkan 1.2, the version the target driver
    // advertises. `-O` is the optimizing compile, for size and speed.
    for (vulkan_shaders) |shader| {
        const glslc = b.addSystemCommand(&.{
            "glslc",
            "--target-env=vulkan1.2",
            "-O",
        });
        for (shader.defines) |define| glslc.addArg(b.fmt("-D{s}", .{define}));
        glslc.addFileArg(b.path(shader.path));
        glslc.addArg("-o");
        const spv: std.Build.LazyPath = glslc.addOutputFileArg(shader.import_name);
        exe_mod.addAnonymousImport(shader.import_name, .{
            .root_source_file = spv,
        });
    }
}
