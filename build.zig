//! The Zig build script: the release and developer executables, the tests, the benches, and the
//! SIMD probe. The ops backend is a build-time decision, so an unsupported target fails here and
//! never as a runtime SIGILL. Only `-Druntime=vulkan` touches glslc and libvulkan.
//!
//! Comptime. `debugNoteStep` takes its note at comptime, so the print string joins at build time.
//!
//! Vulkan link. The Vulkan loader uses TLS, pthreads, and malloc. It segfaults in its own data
//! setup when the executable does not link libc. With `-lc` the TLS init of the C runtime runs
//! before `main`, so `libvulkan.so.1` finds the slots it expects.
//!
//! Common invocations. The Makefile wraps most of these as short targets, and
//! `check-fmt` has no target of its own:
//!   zig build                                Debug, auto-picks strongest CPU backend
//!   zig build -Doptimize=ReleaseFast         release build (best perf)
//!   zig build install dev                    both executables from one build
//!   zig build -Dops-backend=naive            force the scalar correctness reference
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
const limits = @import("src/core/limits.zig");

const Compile = std.Build.Step.Compile;
const Module = std.Build.Module;
const Step = std.Build.Step;
const print = std.fmt.comptimePrint;

// -- constants: shader defines ----------------------------------------------

// Each value below reaches `glslc` as a `-D` define, and the host reads the same constant of
// `src/core/limits.zig`. The tile a shader walks and the grid the host dispatches read one number.
// A shader whose define is absent fails the compile, so the pair cannot silently fall out of step.

/// The split length of attention. It sizes the LDS score row of the resident kernels and of the
/// split pass, and the split and merge passes derive the split geometry of a row from it.
const attn_split_define = print("ATTN_SPLIT={d}", .{limits.attn_split_len});

/// The widest head of the split pass. It sizes the LDS q row of a head, and
/// `AttentionSplit.recordSplit` asserts the same bound on the host.
const attn_d_max_define = print("ATTN_D_MAX={d}", .{limits.attn_max_head_dim});

/// The query heads of one split workgroup. It sizes the LDS rows of the split pass and picks its
/// named registers, and `AttentionSplit.recordSplit` derives its grid from the same constant.
const attn_queries_define = print("ATTN_QUERIES={d}", .{limits.attn_split_queries});

/// The KV element type of the attention variants. One source keeps their arithmetic from drifting.
const kv_f16_define = "KV_F16";

const attention_defines = [_][]const u8{attn_split_define};
const attention_f16_defines = [_][]const u8{ kv_f16_define, attn_split_define };

const attention_split_defines = [_][]const u8{
    attn_split_define,
    attn_d_max_define,
    attn_queries_define,
};
const attention_split_f16_defines = [_][]const u8{kv_f16_define} ++ attention_split_defines;

const q8_batch_defines = [_][]const u8{
    print("B_TILE={d}u", .{limits.q8_batch_b_tile}),
    print("N_TILE={d}u", .{limits.q8_batch_n_tile}),
};
const q4_batch_defines = [_][]const u8{
    print("B_TILE={d}u", .{limits.q4_batch_b_tile}),
    print("N_TILE={d}u", .{limits.q4_batch_n_tile}),
};
const gemm_defines = [_][]const u8{
    print("BM={d}u", .{limits.gemm_m_tile}),
    print("BN={d}u", .{limits.gemm_n_tile}),
};
const gdn_scan_defines = [_][]const u8{
    print("CSUB={d}u", .{limits.gdn_scan_sub}),
    print("W={d}u", .{limits.gdn_scan_cols}),
    print("DH={d}u", .{limits.gdn_head_dim}),
};
const subgroup_defines = [_][]const u8{
    print("LDS_CAPACITY={d}", .{limits.subgroup_lds_capacity}),
    print("GRID_X={d}u", .{limits.reduce_grid_x}),
};

const reduce_defines = [_][]const u8{
    print("GRID_X={d}u", .{limits.reduce_grid_x}),
};

// -- constants: shader table ------------------------------------------------

/// The shaders that `-Druntime=vulkan` compiles. To add a kernel, add one row and the `.comp` file.
/// A variant compiles the source of another shader with other defines.
const vulkan_shaders = [_]Shader{
    .{ .name = "vec_add" },
    .{ .name = "linear_q8_0" },
    .{ .name = "linear_q8_0_reduce", .defines = &reduce_defines },
    .{ .name = "linear_q4_0_reduce", .defines = &reduce_defines },
    .{ .name = "linear_q8_0_subgroup", .defines = &subgroup_defines },
    .{ .name = "linear_q8_0_batch", .defines = &q8_batch_defines },
    .{ .name = "linear_q8_0_gemm", .defines = &gemm_defines },
    .{ .name = "linear_q4_0_batch", .defines = &q4_batch_defines },
    .{ .name = "rmsnorm_batch" },
    .{ .name = "rope_multi_batch" },
    .{ .name = "attention_batch", .defines = &attention_defines },
    .{ .name = "attention_multi_stream", .defines = &attention_defines },
    .{ .name = "residual_add" },
    .{ .name = "swiglu" },
    .{ .name = "rmsnorm" },
    .{ .name = "rope" },
    .{ .name = "rope_multi" },
    .{ .name = "attention", .defines = &attention_defines },
    .{ .name = "gdn_decode" },
    .{ .name = "gdn_chunk_conv" },
    .{ .name = "gdn_chunk_scan", .defines = &gdn_scan_defines },
    .{ .name = "gdn_chunk_norm" },
    .{ .name = "sigmoid_gate" },
    .{ .name = "qwen_qk_norm" },
    .{ .name = "kv_commit_f16" },
    .variant("attention_f16", "attention", &attention_f16_defines),
    .variant("attention_batch_f16", "attention_batch", &attention_f16_defines),
    .variant("attention_multi_stream_f16", "attention_multi_stream", &attention_f16_defines),
    .{ .name = "attention_split", .defines = &attention_split_defines },
    .variant("attention_split_f16", "attention_split", &attention_split_f16_defines),
    .{ .name = "attention_merge", .defines = &attention_defines },
};

// -- constants: integration tests -------------------------------------------

// Integration tests live under tests/ and run the engine against real files on disk. A test finds
// its fixture through `test_fixtures`. A missing file skips the test when no models directory is
// set, and fails it when the set directory lacks the file.

const cpu_integration_tests = [_][]const u8{
    "tests/gguf_smollm2_test.zig",
    "tests/tokenizer_smollm2_test.zig",
    "tests/tokenizer_llama3_test.zig",
    "tests/tokenizer_qwen_test.zig",
    "tests/chat_prompt_test.zig",
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

/// The tests that need the Vulkan context and the kernels alone. Each one references the `vulkan`
/// module and the SPIR-V imports, so it compiles only under `-Druntime=vulkan`.
const vulkan_integration_tests = [_][]const u8{
    "tests/vulkan_vec_add_test.zig",
    "tests/vulkan_kv_commit_f16_test.zig",
    "tests/vulkan_linear_q8_0_test.zig",
    "tests/vulkan_attention_test.zig",
    "tests/vulkan_prefill_kernels_test.zig",
    "tests/vulkan_stream_ring_test.zig",
};

/// The tests that reach a fused or a hybrid runner. `src/main.zig` re-exports those runners under
/// `use_vulkan_fused_ops` alone, so these tests compile only under `-Dops-backend=vulkan_fused`.
const fused_integration_tests = [_][]const u8{
    "tests/llama_vulkan_serve_equivalence_test.zig",
    "tests/qwen_vulkan_serve_equivalence_test.zig",
    "tests/llama_hybrid_residency_test.zig",
    "tests/llama_hybrid_serve_test.zig",
};

const all_integration_tests =
    cpu_integration_tests ++ vulkan_integration_tests ++ fused_integration_tests;

// -- constants: option help -------------------------------------------------

const ops_backend_help = "Op kernel backend. 'auto' (default) picks the strongest variant whose " ++
    "OpRequires the target satisfies. 'naive' = scalar f32 reference. 'simd' = @Vector(8) " ++
    "AVX2+FMA path. The build fails if the target lacks AVX2.";

const runtime_help = "Dispatch runtime. 'cpu' = the ops/ variants alone. " ++
    "'vulkan' = also compile src/ops_vulkan/shaders/*.comp to SPIR-V with glslc, " ++
    "link libvulkan, and pull vulkan-zig as a build dependency. " ++
    "It needs glslc on PATH and libvulkan-dev installed. " ++
    "The default is 'vulkan' for a GPU ops backend, otherwise 'cpu'.";

const attn_kernel_help = "GPU attention kernel set. " ++
    "'split' (default) = the resident kernels inside one split and the split " ++
    "and merge passes past it. A CPU build ignores it.";

const vk_registry_help = "Path to the Vulkan registry XML (vk.xml). " ++
    "The build reads it only under -Druntime=vulkan. " ++
    "The default is the Ubuntu and Debian libvulkan-dev location.";

const trace_help = "Compile the trace: the span sites, the per-op CPU timer, " ++
    "and the layer, dispatch, and token log scopes. The default is false in every mode. " ++
    "A trace still needs $ZIGFERENCER_TRACE_OUT, and a log scope needs $ZIGFERENCER_LOG.";

// GPU tests run one at a time on a shared device, so the suite needs to run a single named test.
// A filtered-out test never executes, so its binary touches no device.
const test_filter_help = "Run only tests whose name contains this string (repeatable).";

const test_threads_help = "Pool threads of the runner tests whose thread count is incidental " ++
    "(default 4, at least 1). Serial reference runs keep 1 thread.";

const llvm_help =
    "Build both executables and every test binary with the LLVM code generator (default: " ++
    "the Zig choice for the target). " ++
    "A debugger shows local variables only in an LLVM Debug build.";

const test_bin_help = "The test binary that test-install puts at zig-out/test/<name>: " ++
    "unit (the tests of src/ outside src/dev/), dev_unit (the tests of src/dev/), " ++
    "or the stem of a tests/*_test.zig file (default unit).";

// -- types ------------------------------------------------------------------

const OpsBackendOption = enum { auto, naive, simd, simd_pf, vulkan, vulkan_fused };
const OpsBackendResolved = enum { naive, simd, simd_pf, vulkan, vulkan_fused };
const RuntimeOption = enum { cpu, vulkan };
const AttnKernelOption = enum { split };

/// One row of `vulkan_shaders`. The import name that source passes to `@embedFile` is
/// `<name>.spv`, and the file is `src/ops_vulkan/shaders/<source>.comp`.
const Shader = struct {
    name: []const u8,
    /// The `.comp` stem, when it differs from `name`.
    source: ?[]const u8 = null,
    defines: []const []const u8 = &.{},

    fn variant(name: []const u8, source: []const u8, defines: []const []const u8) Shader {
        return .{ .name = name, .source = source, .defines = defines };
    }
};

/// The build options, resolved once by `readConfig`.
const Config = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    runtime: RuntimeOption,
    ops_backend: OpsBackendResolved,
    attn_kernel: AttnKernelOption,
    vk_registry: []const u8,
    trace_hot: bool,
    test_filters: []const []const u8,
    test_threads: u32,
    use_llvm: ?bool,
    test_bin: []const u8,
};

const Modules = struct {
    build_options: *Module,
    test_options: *Module,
    exe: *Module,
    dev: *Module,
};

// -- api --------------------------------------------------------------------

pub fn build(b: *std.Build) void {
    const config = readConfig(b);
    const modules = addModules(b, config);

    if (config.runtime == .vulkan) {
        wireVulkan(b, modules, config.vk_registry);
    }

    const exe = addExecutables(b, config, modules);
    addCheck(b, modules);
    addRun(b, exe);

    const test_step = addTests(b, config, modules);
    addFmtCheck(b, test_step);

    addBench(b, config, modules);
    addSimdProbe(b, config, modules);
    addShaderLint(b, config, test_step);
}

// -- helpers: options -------------------------------------------------------

/// Read every build option. `--help` lists the options in this order.
fn readConfig(b: *std.Build) Config {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const requested_backend =
        b.option(OpsBackendOption, "ops-backend", ops_backend_help) orelse .auto;
    const requested_runtime = b.option(RuntimeOption, "runtime", runtime_help);
    // A GPU backend cannot run without the Vulkan runtime, so it selects the runtime by itself.
    const runtime: RuntimeOption = requested_runtime orelse switch (requested_backend) {
        .vulkan, .vulkan_fused => .vulkan,
        else => .cpu,
    };

    return .{
        .target = target,
        .optimize = optimize,
        .runtime = runtime,
        .ops_backend = resolveOpsBackend(requested_backend, target, runtime),
        .attn_kernel = b.option(AttnKernelOption, "attn-kernel", attn_kernel_help) orelse .split,
        .vk_registry = b.option([]const u8, "vk-registry", vk_registry_help) orelse
            "/usr/share/vulkan/registry/vk.xml",
        .trace_hot = b.option(bool, "trace", trace_help) orelse false,
        .test_filters = b.option([]const []const u8, "test-filter", test_filter_help) orelse &.{},
        .test_threads = checkTestThreads(b.option(u32, "test-threads", test_threads_help) orelse 4),
        .use_llvm = b.option(bool, "llvm", llvm_help),
        .test_bin = b.option([]const u8, "test-bin", test_bin_help) orelse "unit",
    };
}

fn checkTestThreads(test_threads: u32) u32 {
    if (test_threads == 0) @panic("-Dtest-threads must be at least 1");
    return test_threads;
}

/// Map `-Dops-backend` to a concrete variant. `.auto` picks the strongest variant that the target
/// satisfies: `simd_pf` on x86_64 with AVX2+FMA. An explicit choice on an unsupported target fails
/// the build.
fn resolveOpsBackend(
    requested: OpsBackendOption,
    target: std.Build.ResolvedTarget,
    runtime: RuntimeOption,
) OpsBackendResolved {
    const needs_avx2_fma =
        " needs AVX2+FMA on x86_64. This target lacks it. Use -Dops-backend=naive or =auto.";
    // The vulkan runtime owns the shader compiles and the libvulkan link. Without this refusal the
    // build fails later at a confusing missing-symbol link error.
    const needs_vulkan = " needs -Druntime=vulkan to wire the shaders and the libvulkan link.";

    const avx2_fma_ok = simdBackendSupportedBy(target.result.cpu);
    const vulkan_ok = runtime == .vulkan;
    return switch (requested) {
        .naive => .naive,
        .simd => if (avx2_fma_ok) .simd else @panic("ops-backend=simd" ++ needs_avx2_fma),
        .simd_pf => if (avx2_fma_ok) .simd_pf else @panic("ops-backend=simd_pf" ++ needs_avx2_fma),
        .vulkan => if (vulkan_ok) .vulkan else @panic("ops-backend=vulkan" ++ needs_vulkan),
        .vulkan_fused => if (vulkan_ok)
            .vulkan_fused
        else
            @panic("ops-backend=vulkan_fused" ++ needs_vulkan),
        .auto => if (avx2_fma_ok) .simd_pf else .naive,
    };
}

/// The build-time mirror of the simd requirements in `core/op_requires.zig`: x86_64, AVX2, and
/// FMA. The `pub const requires` of the variant file is the source of truth. The two must agree.
fn simdBackendSupportedBy(cpu: std.Target.Cpu) bool {
    if (cpu.arch != .x86_64) return false;

    const F = std.Target.x86.Feature;
    if (!std.Target.x86.featureSetHas(cpu.features, F.avx2)) return false;
    if (!std.Target.x86.featureSetHas(cpu.features, F.fma)) return false;

    return true;
}

/// The CPU variant of a binary. A GPU binary still owns runners that the GPU does not cover, such
/// as the qwen session runner on a fused build. They must run on the strongest CPU variant.
fn cpuOpsBackend(config: Config) OpsBackendResolved {
    const simd_ok = simdBackendSupportedBy(config.target.result.cpu);
    return switch (config.ops_backend) {
        .naive, .simd, .simd_pf => config.ops_backend,
        .vulkan, .vulkan_fused => if (simd_ok) .simd_pf else .naive,
    };
}

// -- helpers: modules -------------------------------------------------------

fn addModules(b: *std.Build, config: Config) Modules {
    const test_options = addTestOptions(b, config);
    const build_options = addBuildOptions(b, config);

    const exe = createExeModule(b, config, "src/main.zig", &.{
        .{ .name = "build_options", .module = build_options },
    });
    // The developer executable builds on `exe` with the same target and optimize mode, so the
    // engine code of the two binaries is equal.
    const dev = createExeModule(b, config, "src/dev/dev_main.zig", &.{
        .{ .name = "zigferencer", .module = exe },
        .{ .name = "build_options", .module = build_options },
    });

    // A dependent package reads the engine with `dep.module("zigferencer")`.
    b.modules.put(b.allocator, "zigferencer", exe) catch @panic("OOM");

    return .{
        .build_options = build_options,
        .test_options = test_options,
        .exe = exe,
        .dev = dev,
    };
}

fn createExeModule(
    b: *std.Build,
    config: Config,
    root_source_file: []const u8,
    imports: []const Module.Import,
) *Module {
    return b.createModule(.{
        .root_source_file = b.path(root_source_file),
        .target = config.target,
        .optimize = config.optimize,
        .imports = imports,
        // The perf profiler walks the call stacks by the frame pointer.
        .omit_frame_pointer = false,
    });
}

fn addBuildOptions(b: *std.Build, config: Config) *Module {
    const cpu_ops = cpuOpsBackend(config);

    const options = b.addOptions();
    options.addOption(bool, "cpu_ops_simd", cpu_ops == .simd);
    options.addOption(bool, "cpu_ops_simd_pf", cpu_ops == .simd_pf);
    options.addOption(bool, "use_vulkan_ops", config.ops_backend == .vulkan);
    options.addOption(bool, "use_vulkan_fused_ops", config.ops_backend == .vulkan_fused);
    options.addOption([]const u8, "ops_backend", @tagName(config.ops_backend));
    options.addOption(bool, "use_vulkan", config.runtime == .vulkan);
    options.addOption([]const u8, "runtime", @tagName(config.runtime));
    options.addOption(bool, "trace_hot", config.trace_hot);
    options.addOption(AttnKernelOption, "attn_kernel", config.attn_kernel);
    return options.createModule();
}

fn addTestOptions(b: *std.Build, config: Config) *Module {
    const options = b.addOptions();
    options.addOption(u32, "test_threads", config.test_threads);
    return options.createModule();
}

// -- helpers: executables ---------------------------------------------------

/// Add the release executable to the default install step and the developer executable to `dev`.
/// The developer executable holds the release commands plus the benches, the kernel checks, the
/// demo, and the tokenizer dump.
fn addExecutables(b: *std.Build, config: Config, modules: Modules) *Compile {
    const exe = b.addExecutable(.{
        .name = "zigferencer",
        .root_module = modules.exe,
        .use_llvm = config.use_llvm,
    });
    // A build id lets perf cache each binary by its id.
    exe.build_id = .fast;
    b.installArtifact(exe);

    const dev_exe = b.addExecutable(.{
        .name = "zigferencer-dev",
        .root_module = modules.dev,
        .use_llvm = config.use_llvm,
    });
    dev_exe.build_id = .fast;
    const dev_step = b.step("dev", "Build and install zigferencer-dev");
    dev_step.dependOn(&b.addInstallArtifact(dev_exe, .{}).step);

    return exe;
}

// -- helpers: check and run -------------------------------------------------

/// ZLS runs this step for diagnostics, so that named imports such as `@import("build_options")`
/// resolve in the editor.
fn addCheck(b: *std.Build, modules: Modules) void {
    const check_step = b.step("check", "Type-check without producing artifacts (used by ZLS)");

    const check_exe = b.addExecutable(.{ .name = "zigferencer-check", .root_module = modules.exe });
    check_step.dependOn(&check_exe.step);

    const check_dev_exe = b.addExecutable(.{
        .name = "zigferencer-dev-check",
        .root_module = modules.dev,
    });
    check_step.dependOn(&check_dev_exe.step);
}

fn addRun(b: *std.Build, exe: *Compile) void {
    const run_step = b.step("run", "Run zigferencer");
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    run_step.dependOn(&run_cmd.step);
}

// -- helpers: tests ---------------------------------------------------------

/// Add the test steps and return `test`, so that later areas can add to it.
fn addTests(b: *std.Build, config: Config, modules: Modules) *Step {
    // Declare the steps.
    const test_step = b.step("test", "Run zigferencer tests");
    // This step compiles a test that cannot run where the build runs, such as a GPU test on a host
    // without a device.
    const test_build_step = b.step("test-build", "Compile all tests without running them");
    const test_install_step = b.step(
        "test-install",
        "Install the test binary of -Dtest-bin at zig-out/test/<name> for a debugger",
    );
    // The unit tests alone run in seconds in Debug, so a gate that runs the whole suite in an
    // optimized mode keeps their Debug checks here.
    const test_unit_step = b.step("test-unit", "Run the unit tests of src/ only");

    // Add the unit tests of both executables.
    var binaries: std.ArrayList(*Compile) = .empty;

    const unit_tests = b.addTest(.{
        .name = "unit",
        .root_module = modules.exe,
        .filters = config.test_filters,
        .use_llvm = config.use_llvm,
    });
    const dev_unit_tests = b.addTest(.{
        .name = "dev_unit",
        .root_module = modules.dev,
        .filters = config.test_filters,
        .use_llvm = config.use_llvm,
    });

    for ([_]*Compile{ unit_tests, dev_unit_tests }) |binary| {
        binaries.append(b.allocator, binary) catch @panic("OOM");

        const run_binary = addTestRun(b, binary);
        test_step.dependOn(&run_binary.step);
        test_build_step.dependOn(&binary.step);
        test_unit_step.dependOn(&run_binary.step);
    }

    // Add the shader lint tests.
    const lint_tests = b.addTest(.{
        .name = "shader_lint",
        .root_module = createLintModule(b, config.target, config.optimize),
        .filters = config.test_filters,
        .use_llvm = config.use_llvm,
    });
    binaries.append(b.allocator, lint_tests) catch @panic("OOM");

    const run_lint_tests = addTestRun(b, lint_tests);
    test_step.dependOn(&run_lint_tests.step);
    test_build_step.dependOn(&lint_tests.step);
    test_unit_step.dependOn(&run_lint_tests.step);

    // Add the integration tests.
    requireListed(b, &all_integration_tests);

    for (integrationTestPaths(config)) |path| {
        const binary = addIntegrationTest(b, config, modules, path);
        binaries.append(b.allocator, binary) catch @panic("OOM");

        const run_binary = addTestRun(b, binary);
        // The tidy test reads the source tree through relative paths.
        if (std.mem.eql(u8, path, "tests/tidy_test.zig")) {
            run_binary.setCwd(b.path("."));
        }

        test_step.dependOn(&run_binary.step);
        test_build_step.dependOn(&binary.step);
    }

    // Wire the install step.
    wireTestInstall(b, test_install_step, config.test_bin, binaries.items);
    return test_step;
}

/// The lists follow the two levels of the Vulkan aliases in `src/main.zig`.
fn integrationTestPaths(config: Config) []const []const u8 {
    if (config.runtime != .vulkan) return &cpu_integration_tests;
    if (config.ops_backend == .vulkan_fused) return &all_integration_tests;
    return &(cpu_integration_tests ++ vulkan_integration_tests);
}

fn addIntegrationTest(b: *std.Build, config: Config, modules: Modules, path: []const u8) *Compile {
    const module = b.createModule(.{
        .root_source_file = b.path(path),
        .target = config.target,
        .optimize = config.optimize,
        .imports = &.{
            .{ .name = "zigferencer", .module = modules.exe },
            .{ .name = "test_options", .module = modules.test_options },
        },
    });
    // The interop test opens the system libzmq with `dlopen`. The pure-Zig ELF loader does not
    // resolve the dependencies of that library.
    if (std.mem.eql(u8, path, "tests/zmq_interop_test.zig")) {
        module.link_libc = true;
    }

    return b.addTest(.{
        .name = std.fs.path.stem(path),
        .root_module = module,
        .filters = config.test_filters,
        .use_llvm = config.use_llvm,
    });
}

fn addTestRun(b: *std.Build, binary: *Compile) *Step.Run {
    const run = b.addRunArtifact(binary);
    // The model tests read files and environment variables that the build graph does not see, so
    // a cached pass can hide a missing model.
    run.has_side_effects = true;
    return run;
}

/// Make `test-install` install the binary that `-Dtest-bin` names, so only that binary compiles for
/// the step. An unknown name fails the step alone, and the other steps of the build stay usable.
fn wireTestInstall(
    b: *std.Build,
    test_install_step: *Step,
    test_bin: []const u8,
    binaries: []const *Compile,
) void {
    var names: std.ArrayList(u8) = .empty;
    for (binaries, 0..) |binary, index| {
        if (std.mem.eql(u8, test_bin, binary.name)) {
            test_install_step.dependOn(&installTestBin(b, binary).step);
            return;
        }

        const separator = if (index == 0) "" else ", ";
        names.print(b.allocator, "{s}{s}", .{ separator, binary.name }) catch @panic("OOM");
    }

    const message = b.fmt(
        "-Dtest-bin={s} names no test binary of this build. Valid names: {s}",
        .{ test_bin, names.items },
    );
    test_install_step.dependOn(&b.addFail(message).step);
}

/// Install a test binary at `zig-out/test/<name>`, apart from the release executables.
fn installTestBin(b: *std.Build, binary: *Compile) *Step.InstallArtifact {
    std.debug.assert(binary.kind == .@"test");

    return b.addInstallArtifact(binary, .{ .dest_dir = .{ .override = .{ .custom = "test" } } });
}

/// Fail the build when a `tests/*_test.zig` file sits in none of the lists. Such a file never runs,
/// and nothing else says so.
fn requireListed(b: *std.Build, listed: []const []const u8) void {
    const io = b.graph.io;

    // A fetched package can omit tests/. Then the check has nothing to read.
    var dir = b.build_root.handle.openDir(io, "tests", .{ .iterate = true }) catch |err| {
        switch (err) {
            error.FileNotFound => return,
            else => std.debug.panic("cannot open tests/: {s}", .{@errorName(err)}),
        }
    };
    defer dir.close(io);

    var it = dir.iterate();
    while (it.next(io) catch |err| {
        std.debug.panic("cannot read tests/: {s}", .{@errorName(err)});
    }) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, "_test.zig")) continue;

        const found = for (listed) |path| {
            if (std.mem.eql(u8, path["tests/".len..], entry.name)) break true;
        } else false;

        if (!found) {
            std.debug.panic("tests/{s} is in no test list in build.zig", .{entry.name});
        }
    }
}

// -- helpers: format check --------------------------------------------------

/// `test` depends on the format check, so unformatted code blocks the test suite.
fn addFmtCheck(b: *std.Build, test_step: *Step) void {
    const check_fmt_step = b.step("check-fmt", "Verify all source is zig-fmt clean");
    const fmt_check = b.addFmt(.{
        .paths = &.{ "src", "tests", "tools", "build.zig", "build.zig.zon" },
        .check = true,
    });
    check_fmt_step.dependOn(&fmt_check.step);
    test_step.dependOn(&fmt_check.step);
}

// -- helpers: bench and probe -----------------------------------------------

fn addBench(b: *std.Build, config: Config, modules: Modules) void {
    const bench_exe = addToolExe(b, config, modules, .{
        .name = "zigferencer-bench",
        .root_source_file = "tools/bench.zig",
    });

    const bench_step = b.step("bench", "Run op micro-benchmarks");
    const run_bench = b.addRunArtifact(bench_exe);
    if (config.optimize == .Debug) {
        run_bench.step.dependOn(debugNoteStep(
            b,
            "bench times Debug kernels. Pass -Doptimize=ReleaseFast for real numbers.",
        ));
    }

    if (b.args) |args| {
        run_bench.addArgs(args);
    }

    bench_step.dependOn(&run_bench.step);

    // Some kernels have their only caller in the bench, so no other step type-checks those calls.
    // This step also compiles the Vulkan rows on a host without a device.
    const bench_build_step = b.step("bench-build", "Compile the bench without running it");
    bench_build_step.dependOn(&bench_exe.step);
}

/// The probe pins every op kernel to a standalone symbol, so `tools/simd-report.sh` can read each
/// one. The step installs the assembly and never runs the probe. The probe source says why the
/// release binary cannot serve instead.
fn addSimdProbe(b: *std.Build, config: Config, modules: Modules) void {
    const probe_exe = addToolExe(b, config, modules, .{
        .name = "zigferencer-simd-probe",
        .root_source_file = "tools/simd_probe.zig",
    });

    const probe_step = b.step(
        "simd-probe",
        "Emit per-op target-native assembly (for tools/simd-report.sh)",
    );
    // The emitted `.s` is target-native for any backend Zig can compile, so the report works under
    // cross-compilation and needs no multi-arch objdump. The script finds the fixed install path.
    const probe_asm = b.addInstallFile(probe_exe.getEmittedAsm(), "zigferencer-simd-probe.s");
    probe_step.dependOn(&probe_asm.step);
    if (config.optimize == .Debug) {
        probe_step.dependOn(debugNoteStep(
            b,
            "simd-probe emits Debug assembly. Pass -Doptimize=ReleaseFast for the real report.",
        ));
    }
}

/// A tool root is always ReleaseFast, but the kernels come from the zigferencer module and follow
/// `-Doptimize`. A Debug build therefore gets a note from each tool step.
fn addToolExe(
    b: *std.Build,
    config: Config,
    modules: Modules,
    options: struct { name: []const u8, root_source_file: []const u8 },
) *Compile {
    const module = b.createModule(.{
        .root_source_file = b.path(options.root_source_file),
        .target = config.target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "zigferencer", .module = modules.exe },
            .{ .name = "build_options", .module = modules.build_options },
        },
    });

    const exe = b.addExecutable(.{ .name = options.name, .root_module = module });
    exe.build_id = .fast;
    return exe;
}

/// A step that prints `note` when the build graph runs it.
fn debugNoteStep(b: *std.Build, comptime note: []const u8) *Step {
    const Note = struct {
        fn make(_: *Step, _: Step.MakeOptions) anyerror!void {
            std.debug.print("note: " ++ note ++ "\n", .{});
        }
    };

    const step = b.allocator.create(Step) catch @panic("OOM");
    step.* = Step.init(.{ .id = .custom, .name = "debug note", .owner = b, .makeFn = Note.make });
    return step;
}

// -- helpers: vulkan --------------------------------------------------------

/// Add vulkan-zig, link `libvulkan`, and compile `vulkan_shaders` to SPIR-V. Each blob becomes an
/// anonymous import, so source can call `@embedFile("vec_add.spv")` wherever the build wrote it.
fn wireVulkan(b: *std.Build, modules: Modules, vk_registry: []const u8) void {
    const exe_mod = modules.exe;
    // A lazy dependency is null on the first pass. The build runner then fetches it and runs
    // `build` again.
    const vk_dep = b.lazyDependency("vulkan", .{ .registry = vk_registry }) orelse return;
    exe_mod.addImport("vulkan", vk_dep.module("vulkan-zig"));
    modules.dev.addImport("vulkan", vk_dep.module("vulkan-zig"));

    // `link_libc` is load-bearing.
    exe_mod.linkSystemLibrary("vulkan", .{});
    exe_mod.link_libc = true;

    for (vulkan_shaders) |shader| {
        addShaderImport(b, exe_mod, shader);
    }
}

fn addShaderImport(b: *std.Build, exe_mod: *Module, shader: Shader) void {
    const import_name = b.fmt("{s}.spv", .{shader.name});
    const spv = addGlslc(b, shader, false);
    exe_mod.addAnonymousImport(import_name, .{ .root_source_file = spv });
}

/// Compile one shader row and return the blob. The release build passes `strict = false`, because
/// a newer `glslc` can add a warning and the build must not depend on that. The lint passes true.
fn addGlslc(b: *std.Build, shader: Shader, strict: bool) std.Build.LazyPath {
    // `--target-env=vulkan1.2` keeps the output within the version the target driver advertises.
    const glslc = b.addSystemCommand(&.{ "glslc", "--target-env=vulkan1.2", "-O" });
    if (strict) {
        glslc.addArg("-Werror");
    }

    for (shader.defines) |define| {
        glslc.addArg(b.fmt("-D{s}", .{define}));
    }

    const source = shader.source orelse shader.name;
    glslc.addFileArg(b.path(b.fmt("src/ops_vulkan/shaders/{s}.comp", .{source})));
    glslc.addArg("-o");

    return glslc.addOutputFileArg(b.fmt("{s}.spv", .{shader.name}));
}

// -- helpers: shader lint ---------------------------------------------------

fn createLintModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *Module {
    return b.createModule(.{
        .root_source_file = b.path("tools/shader_lint.zig"),
        .target = target,
        .optimize = optimize,
    });
}

/// `lint-shaders` runs the style rules of `tools/shader_lint.zig` and needs no Vulkan SDK. Under
/// `-Druntime=vulkan` it also compiles each shader with `glslc -Werror` and runs `spirv-val`.
/// The step names its inputs, so `--watch` runs it again on a save. `test` depends on the style
/// rules alone, so a default `zig build test` stays free of a Vulkan SDK.
fn addShaderLint(b: *std.Build, config: Config, test_step: *Step) void {
    const lint_step = b.step(
        "lint-shaders",
        "Check the shader rules (with -Druntime=vulkan: also glslc -Werror and spirv-val)",
    );
    const lint_exe = b.addExecutable(.{
        .name = "shader-lint",
        .root_module = createLintModule(b, b.graph.host, .ReleaseSafe),
    });

    const run_lint = b.addRunArtifact(lint_exe);
    run_lint.addDirectoryArg(b.path("src/ops_vulkan/shaders"));
    run_lint.addDirectoryArg(b.path("src/ops_vulkan/kernels"));
    run_lint.expectExitCode(0);
    lint_step.dependOn(&run_lint.step);
    test_step.dependOn(&run_lint.step);

    if (config.runtime != .vulkan) return;

    for (vulkan_shaders) |shader| {
        const spirv_val = b.addSystemCommand(&.{ "spirv-val", "--target-env", "vulkan1.2" });
        spirv_val.addFileArg(addGlslc(b, shader, true));
        lint_step.dependOn(&spirv_val.step);
    }
}
