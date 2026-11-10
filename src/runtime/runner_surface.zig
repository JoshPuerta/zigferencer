//! The build-time check of a runner. A family row states the check for each of its runners, and
//! `Sessions` states the check for the part it reads.
//!
//! Comptime. The whole file runs at build time, so a runner that drifts from the table in
//! `README.md` stops the build and never reaches a device.

const std = @import("std");
const StepRow = @import("step_row.zig").StepRow;

// -- api --------------------------------------------------------------------

/// The whole surface of one runner row: the single-row step, the swap hooks of a runner that
/// serves, and the plan value it takes at `init`.
pub fn requireSurface(comptime T: type, comptime Plan: type) void {
    requireStepSurface(T);
    if (@hasDecl(T, "evictSlot")) {
        requireSwapSurface(T);
    }

    requirePlanValue(T, Plan);
}

/// The marker itself. Without it, `requireRecurrentHooks` passes on a runner that dropped both.
pub fn requireRecurrentState(comptime T: type) void {
    if (!@hasDecl(T, "has_recurrent_state") or !T.has_recurrent_state) {
        @compileError("runner surface: " ++ @typeName(T) ++ " holds recurrent state, " ++
            "so it must declare 'pub const has_recurrent_state = true'");
    }
}

/// The marker of a runner without recurrent state: absent or false. A generic runner states a
/// value for every family, so the check reads the value.
pub fn requireNoRecurrentState(comptime T: type) void {
    if (@hasDecl(T, "has_recurrent_state") and T.has_recurrent_state) {
        @compileError("runner surface: " ++ @typeName(T) ++ " holds no recurrent state, " ++
            "so its 'has_recurrent_state' must be false or absent");
    }
}

/// The surface `Sessions` reads. The optional rows can be absent. `evictSlot` brings the swap
/// surface with it.
pub fn requireSessionsSurface(comptime T: type) void {
    requireFn(T, "stepBatch", &.{[]const StepRow}, void);
    requireFn(T, "maxStepRows", &.{}, u32);
    requireField(T, "logits", []f32);
    requireField(T, "n_slots", u32);
    requireField(T, "max_seq", u32);
    optionalFn(T, "logitRowsMax", &.{}, u32);
    optionalFn(T, "resetSlotState", &.{u32}, void);
    optionalFn(T, "cloneSlotState", &.{ u32, u32, u32 }, void);
    requireRecurrentHooks(T);
    if (@hasDecl(T, "evictSlot")) {
        requireSwapSurface(T);
    }
}

// -- helpers ----------------------------------------------------------------

/// One field of the surface: the name and the exact type a consumer reads.
fn requireField(comptime T: type, comptime name: []const u8, comptime Want: type) void {
    if (!@hasField(T, name)) {
        @compileError("runner surface: " ++ @typeName(T) ++ " has no field '" ++ name ++ "'");
    }
    if (@FieldType(T, name) != Want) {
        @compileError("runner surface: " ++ @typeName(T) ++ "." ++ name ++ " is " ++
            @typeName(@FieldType(T, name)) ++ ", the surface states " ++ @typeName(Want));
    }
}

/// One method of the surface.
fn requireFn(
    comptime T: type,
    comptime name: []const u8,
    comptime params: []const type,
    comptime Payload: type,
) void {
    if (!@hasDecl(T, name)) {
        @compileError("runner surface: " ++ @typeName(T) ++ " has no declaration '" ++ name ++ "'");
    }

    checkFn(T, name, params, Payload);
}

/// A method that `Sessions` gates on `@hasDecl`. Absence is legal, and a wrong shape is not.
fn optionalFn(
    comptime T: type,
    comptime name: []const u8,
    comptime params: []const type,
    comptime Payload: type,
) void {
    if (@hasDecl(T, name)) {
        checkFn(T, name, params, Payload);
    }
}

/// The receiver, the arguments after it, and the value the method yields.
/// An error union counts as its payload, because each family declares its own error set.
fn checkFn(
    comptime T: type,
    comptime name: []const u8,
    comptime params: []const type,
    comptime Payload: type,
) void {
    // Resolve the function.
    const where = "runner surface: " ++ @typeName(T) ++ "." ++ name ++ " ";
    const info = @typeInfo(@TypeOf(@field(T, name)));
    if (info != .@"fn") @compileError(where ++ "is not a function");

    // Check the arity and the receiver.
    const f = info.@"fn";
    if (f.params.len != params.len + 1) {
        @compileError(where ++ "takes " ++ std.fmt.comptimePrint("{d}", .{f.params.len}) ++
            " arguments, the surface states " ++ std.fmt.comptimePrint("{d}", .{params.len + 1}));
    }

    const receiver = f.params[0].type orelse @compileError(where ++ "has a generic receiver");
    if (receiver != *T and receiver != *const T) {
        @compileError(
            where ++ "takes " ++ @typeName(receiver) ++ ", not a pointer to its own runner",
        );
    }

    // Check the type of each argument.
    inline for (params, 0..) |Want, i| {
        const got = f.params[i + 1].type orelse @compileError(where ++ "has a generic argument");
        if (got != Want) {
            @compileError(
                where ++ "argument " ++ std.fmt.comptimePrint("{d}", .{i + 1}) ++ " is " ++
                    @typeName(got) ++ ", the surface states " ++ @typeName(Want),
            );
        }
    }

    // Check the payload type.
    const ret = f.return_type orelse @compileError(where ++ "has a generic return type");
    const payload = switch (@typeInfo(ret)) {
        .error_union => |eu| eu.payload,
        else => ret,
    };

    if (payload != Payload) {
        @compileError(
            where ++ "yields " ++ @typeName(
                payload,
            ) ++ ", the surface states " ++ @typeName(Payload),
        );
    }
}

/// The slot hooks of a runner with recurrent state. The `has_recurrent_state` marker makes
/// the check require both hooks by name.
fn requireRecurrentHooks(comptime T: type) void {
    if (!@hasDecl(T, "has_recurrent_state")) return;
    if (@TypeOf(T.has_recurrent_state) != bool) {
        @compileError("runner surface: " ++ @typeName(T) ++ ".has_recurrent_state is " ++
            @typeName(@TypeOf(T.has_recurrent_state)) ++ ", the surface states bool");
    }
    if (!T.has_recurrent_state) return;

    inline for ([_][]const u8{ "resetSlotState", "cloneSlotState" }) |hook| {
        if (!@hasDecl(T, hook)) {
            @compileError("runner surface: " ++ @typeName(T) ++ " declares has_recurrent_state, " ++
                "so it must declare '" ++ hook ++ "'");
        }
    }
}

/// The pool surface plus the single-row `step` that `generate.decode` and `run` call.
fn requireStepSurface(comptime T: type) void {
    requireSessionsSurface(T);
    requireFn(T, "step", &.{ u32, u32 }, []f32);
}

/// The swap surface of a tiered pool: size, evict, and restore.
fn requireSwapSurface(comptime T: type) void {
    requireFn(T, "evictSlotBytesFor", &.{u32}, u64);
    requireFn(T, "evictSlot", &.{ u32, u32, []u8 }, void);
    requireFn(T, "restoreSlot", &.{ u32, u32, []const u8 }, void);
}

/// A runner takes its resource plan as a value at `init`, alone or as the `plan` field of an
/// options value, and cannot outgrow it. The argument position differs per family, so the check
/// finds the plan by type.
fn requirePlanValue(comptime T: type, comptime Plan: type) void {
    if (!@hasDecl(T, "init")) {
        @compileError("runner surface: " ++ @typeName(T) ++ " has no declaration 'init'");
    }

    const info = @typeInfo(@TypeOf(@field(T, "init")));
    if (info != .@"fn") {
        @compileError("runner surface: " ++ @typeName(T) ++ ".init is not a function");
    }

    comptime var reads_plan = false;
    inline for (info.@"fn".params) |p| {
        if (p.type) |ty| {
            const in_options = @typeInfo(ty) == .@"struct" and @hasField(ty, "plan") and
                @FieldType(ty, "plan") == Plan;
            if (ty == Plan or in_options) {
                reads_plan = true;
            }
        }
    }

    if (!reads_plan) {
        @compileError("runner surface: " ++ @typeName(T) ++ ".init takes no " ++ @typeName(Plan) ++
            ", so its shape is not a value the caller planned");
    }
}
