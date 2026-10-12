# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Render a bufferized TIR call graph as a C++ FTT package.

The descriptor this consumes is the same one ``codegen/triton`` builds: the
buffer plan, the call ABI, the index/predicate algebra and the event schedule
are all language-neutral facts about the artifact. Only the emission differs.
Reusing the descriptor is deliberate -- a second derivation of the same
addresses would be a second thing to keep correct.

What this renderer does *not* do is fall back to Triton for a kernel it cannot
emit. A package is entirely C++ or entirely Triton; a kernel family without a
C++ template raises, so an unsupported model fails at compile time rather than
silently producing a mixed package nobody asked for.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from triton.flagmega.codegen.cpp.templates import (
    CppKernelTemplateSpec,
    CppTemplateRegistry,
)
from triton.flagmega.codegen.triton.tir_package import describe_tir_package
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import IRModule, verify_buffer_plan, verify_module

if TYPE_CHECKING:
    from triton.flagmega.diagnostics import Dumper


CPP_RENDERER_VERSION = "flagmega-cpp-renderer/v1"

CPP_PACKAGE_DESCRIPTOR_SCHEMA = "flagmega.cpp-package-descriptor/v1"

# Threads per CTA for a generated C++ entry.
#
# The Triton backend derives its block size from `num_warps`, which the
# autotuner owns. A C++ kernel has no autotuner, so the block size is part of
# the generated source and must be stated here.
#
# 1024 = 16 warps x 64 lanes, matching what the Triton path compiles this model
# to. Measured on BI-V150 with the 28-layer Qwen3-1.7B batch-1 decode model
# (median of 5 reps x 32 tokens at a fixed 128-token context, eager):
#
#     256 threads   96.6 ms/token
#     512 threads   73.7
#    1024 threads   73.7
#    2048 threads   78.7
#    4096 threads   launch rejected (invalid configuration)
#
# So this saturates at 512 and regresses past 1024; the device's advertised
# 4096 max threads/block is not launchable for a kernel with this register
# footprint. Decode at batch 1 is memory-bound, which is why the curve is flat
# between 512 and 1024 rather than scaling.
CPP_BLOCK_THREADS = 1024

# Independent K-walk depth for the vectorized (tiled) GEMV path: how many K
# groups one thread keeps in flight at once.
#
# Separate from `k_streams`, which is tuned for the scalar walk where a stream
# is one 2-byte load. In the tiled path a stream is three 16-byte loads, so the
# two want different values. This matters because the machine is latency-bound,
# not bandwidth-bound: the whole model runs at 183 GB/s against a ~670 GB/s
# scalar ceiling, so the figure of merit is independent loads in flight per
# thread. Measured end to end at context 1 (see the tutorial's
# .local/perf-iteration/cpp-backend-iteration.zh.md):
#
#     depth 2    20.06 ms/token
#     depth 4    20.04            (= the old coupled k_streams value)
#     depth 8    19.89            <- optimum
#     depth 16   20.38
#     depth 32   21.15
#
# The optimum differs from `k_streams`' 4, which is the point of separating
# them: 8 x 3 wide loads = 24 in flight per thread, against 12 before.
#
# `FLAGMEGA_CPP_TILED_DEPTH` overrides it for sweeps without touching the
# scalar path's own tuning.
CPP_TILED_DEPTH = int(os.environ.get("FLAGMEGA_CPP_TILED_DEPTH", "2"))

# Use the SME G2S engine with a batched producer/consumer loop in the GEMV
# kernels. Off by default until it beats the vectorized scalar path end to end.
CPP_USE_SME = os.environ.get("FLAGMEGA_CPP_SME", "0") == "1"

# Load the packed weight runs as 4-byte transactions instead of 16-byte ones.
#
# On BI-V150 a 128-bit load is issue-capped near one per cycle per MP (~46% of
# the per-die peak) while 32-bit loads reach ~82%, so narrowing can raise
# throughput even though it issues four times as many instructions. Measured
# neutral on this package (10.380 vs 10.378 ms/token): the advantage shows up
# at high occupancy, and these grid-barrier megakernels are pinned to 16 CTAs,
# one per MP. Off by default; kept because the ordering inverts with occupancy.
CPP_NARROW_LOADS = os.environ.get("FLAGMEGA_CPP_NARROW_LOADS", "0") == "1"


def _target_warp_size(architecture: str) -> int:
    """Warp width for the rendered target, from its registered capability.

    Warp geometry belongs to the target, not to FTT: the reductions fold inside
    a warp before touching shared memory, so the shared slot count follows this
    number. Reading it from the registry means a target with a different warp
    width renders correct buffer sizes instead of silently mis-sizing them.
    """

    from triton.flagmega.targets import get_target, target_names

    for name in target_names():
        target = get_target(name)
        if str(getattr(target, "codegen_architecture", "")) != architecture:
            continue
        width = getattr(getattr(target, "capability", None), "warp_size", None)
        if width:
            return int(width)
    raise CodegenError(
        f"No registered target reports a warp size for architecture "
        f"{architecture!r}; the C++ reductions cannot size their shared buffers."
    )


def describe_cpp_package(module: IRModule) -> dict[str, object]:
    """Extend the shared TIR descriptor with the C++ package's own facts."""

    descriptor = dict(describe_tir_package(module))
    platform = str(descriptor.get("platform", ""))
    architecture = str(descriptor.get("target_architecture", ""))
    registry = CppTemplateRegistry()
    specs = tuple(
        CppKernelTemplateSpec(
            str(value["kernel"]),
            str(value["variant"]),
            platform or None,
            architecture or None,
        )
        for value in descriptor.get("kernel_template_specs", ())
    )
    missing = tuple(
        f"{spec.kernel}/{spec.variant}"
        for spec in specs
        if not registry.available(spec)
    )
    if missing:
        raise CodegenError(
            "The C++ backend has no template for kernel(s) "
            f"{list(missing)}. Add them under {registry.root}, or compile this "
            "model with the Triton backend.",
            stage=module.stage,
        )
    _reject_unsupported_constructs(descriptor, module)
    descriptor.update({
        "schema": CPP_PACKAGE_DESCRIPTOR_SCHEMA,
        "kernel_templates": tuple(registry.resolve(spec) for spec in specs),
        "entry_template": "entrypoints/call_graph.cpp.jinja",
        "block_threads": CPP_BLOCK_THREADS,
        "tiled_depth": CPP_TILED_DEPTH,
        # Warp geometry is a target property, not an FTT constant: the
        # reductions fold inside a warp before touching shared memory, so the
        # slot count follows the target's warp size. Taken from the registered
        # target's capability rather than hardcoded, so a target with a
        # different warp width renders correctly instead of silently
        # mis-sizing the buffers.
        "warp_size": _target_warp_size(architecture),
        "warps": CPP_BLOCK_THREADS // _target_warp_size(architecture),
        "sme": CPP_USE_SME,
        "run_load": "load_run_narrow" if CPP_NARROW_LOADS else "load_run",
        "grid_total_blocks": _grid_total_blocks(descriptor),
        "grid_barrier_counter_slots": _grid_barrier_counter_slots(descriptor),
        "grid_barrier_group_bases": _barrier_group_bases(descriptor),
        "argument_types": _argument_types(descriptor, module),
    })
    _gemv_k_streams(descriptor)
    return descriptor


def _grid_barrier_counter_slots(descriptor: Mapping[str, object]) -> int:
    """Size the generated package's grid-barrier counter array.

    Slot 0 is the whole-grid barrier. Each axis group then needs one slot per
    *instance* of that group -- the CTAs sharing a group's axes rendezvous
    independently of the CTAs at other coordinates on the axes it excludes, so
    they cannot share a counter. The instance count is the product of the mesh
    extents the group does not span.
    """

    return 1 + sum(
        instances for _, instances in _barrier_group_bases(descriptor).values()
    )


def _barrier_group_bases(
    descriptor: Mapping[str, object]
) -> dict[str, tuple[int, int]]:
    """Per-group ``(counter base, instance count)`` in the counter array.

    Groups are laid out after slot 0 in descriptor order, each occupying as
    many consecutive slots as it has instances. Without a per-group base, two
    groups' instance indices would collide and one group's arrivals would
    release the other.
    """

    hierarchy = tuple(int(value) for value in descriptor.get("mesh_hierarchy", ()))
    bases: dict[str, tuple[int, int]] = {}
    cursor = 1
    for group in descriptor.get("grid_barrier_axis_groups", ()):
        axes = {int(value) for value in group.get("axes", ())}
        instances = 1
        for axis, extent in enumerate(hierarchy):
            if axis not in axes:
                instances *= extent
        bases[str(group["key"])] = (cursor, instances)
        cursor += instances
    return bases


def _gemv_k_streams(descriptor: Mapping[str, object]) -> None:
    """Annotate each GEMV call with how many independent K streams to use.

    These kernels are latency-bound, not bandwidth-bound. A K loop with one
    dependent load chain reaches ~94 GB/s on BI-V150 against a 819 GB/s peak,
    because the generated code waits on each load before forming the next
    address. Eight independent streams over the same bytes reach ~317 GB/s.

    Measured non-factors, each confirmed in the device assembly rather than
    assumed: coalescing the lane-to-address mapping (73.7 -> 70.1 ms/token,
    noise) and widening the loads to 16 bytes (73.7 -> 71.8, noise, despite
    `dwordx4` genuinely rising from 196 to 366). So the stream count is the
    only knob here worth setting.

    Four streams is the measured optimum for the vectorized GEMV form, where
    each iteration already issues two or three 16-byte loads, so the depth that
    eight scalar streams bought is reached with less register pressure. End to
    end on the 28-layer Qwen3-1.7B batch-1 decode model at context 1:

        2 streams   24.01 ms/token
        4 streams   20.04
        8 streams   20.62
       16 streams   23.80

    The scalar form's own saturation point was eight (~317 GB/s against ~94 for
    a single dependent chain), which is why the count changed with the shape of
    the loop rather than because the earlier number was wrong.

    The stream count must *divide* the K extent. A bounds check inside the
    unrolled stream body re-serializes the loads completely -- measured 98.6
    vs 316.9 GB/s for the same 8-stream loop with and without a `break`,
    because the branch collapses the generated load cluster from 6 to 1. So
    the count is chosen to divide evenly and the guard is omitted, rather than
    emitted and relied upon to be cheap.

    The same reasoning applies to the call's own `source_active` predicate. It
    is usually just `k < k_capacity`, which the loop bound already guarantees,
    so `k_predicate_redundant` records when it can be dropped. A predicate
    that is *not* provably implied stays, and the stream count drops to 1 --
    better to lose the pipelining than to emit a branch that silently costs
    3x, or to drop a guard that was actually doing something.
    """

    import re

    for call in descriptor.get("render_calls", ()):
        if call.get("family") not in {
            "dense_matmul",
            "dense_matmul_glu",
            "qkv_parallel_linear",
        }:
            continue
        capacity = int(call.get("local_k_capacity", 0) or 0)
        prefix = {
            "dense_matmul_glu": "dense_glu",
            "qkv_parallel_linear": "qkv",
        }.get(call["family"], "dense")
        # Accept the shapes the renderers actually emit for a pure bounds
        # check, normalizing whitespace and redundant parentheses. Anything
        # else is treated as a real predicate.
        predicate = re.sub(r"\s+", "", str(call.get("source_active", "")))
        expected = {
            f"(0<(1))&(({prefix}_local_k_offsets)<({capacity}))",
            f"({prefix}_local_k_offsets)<({capacity})",
            "True",
        }
        redundant = predicate in expected
        call["k_predicate_redundant"] = redundant
        value = 4 if redundant else 1
        while value > 1 and (capacity < value * 2 or capacity % value):
            value //= 2
        call["k_streams"] = value
        lane = _weight_k_lane(call, prefix)
        call["weight_k_lane"] = lane
        block = _weight_block(call, prefix, lane, capacity)
        if block:
            call["weight_block_n"], call["weight_block_k"] = block


def _weight_block(
    call: Mapping[str, object],
    prefix: str,
    lane: int,
    capacity: int,
) -> tuple[int, int] | None:
    """Return the packed layout's contiguous `(block_n, block_k)` tile, or None.

    `_weight_k_lane` finds the contiguous K run for one N. That run is not the
    whole contiguous region: in `k_major_n8_k16` the layout stores a whole
    `block_n x block_k` tile contiguously (8 N by 16 K = one 128-element,
    256-byte block), N-major inside it, and only then strides to the next K
    group. Knowing the tile -- not just the run -- is what lets a thread read
    `block_n * block_k` elements as a sequence of *independent* 16-byte loads.

    That distinction is the whole point. Loading one 32-byte run per iteration
    is wide but leaves a single dependent chain, which measured 30.1 ms/token
    against the narrow 8-stream loop's 21.8: this device is limited by loads in
    flight, and width alone trades that away. A whole tile is both wide *and*
    16 loads deep, so the two effects compose instead of competing.

    Derived from the call's own address formula by checking the tile identity

        offset(k0 + kl, n0 + ni) == offset(k0, n0) + ni * block_k + kl

    at several `(k0, n0)` tile origins rather than trusting the layout name, so
    a differently interleaved layout falls back to the scalar path instead of
    reading the wrong bytes.
    """

    import re

    if lane < 8 or lane % 8 or capacity <= 0 or capacity % lane:
        return None
    expression = call.get("weight_offset") or call.get("gate_weight_offset")
    if not isinstance(expression, str) or not expression:
        return None
    flat = re.sub(r"\[\s*(?:None\s*,\s*:|:\s*,\s*None)\s*\]", "", expression)

    def offset(k: int, n: int) -> int:
        return eval(  # noqa: S307 - renderer-generated integer algebra
            flat,
            {"__builtins__": {}, "tl": None},
            {f"{prefix}_global_k": k, f"{prefix}_global_n": n, "one": 1},
        )

    try:
        # Widen the N extent while the tile stays contiguous. `lane` is the K
        # extent; N grows until the next N no longer continues the block.
        block_n = 1
        while block_n < 64 and offset(0, block_n) == offset(0, 0) + block_n * lane:
            block_n *= 2
        if block_n < 2:
            return None
        # Verify the full tile identity at origins the kernel actually walks,
        # including a shard-shifted one, rather than only at (0, 0).
        for k0 in (0, lane, capacity - lane):
            for n0 in (0, block_n, block_n * 3):
                base = offset(k0, n0)
                for ni in range(block_n):
                    for kl in range(lane):
                        if offset(k0 + kl, n0 + ni) != base + ni * lane + kl:
                            return None
        return block_n, lane
    except Exception:
        return None


def _weight_k_lane(call: Mapping[str, object], prefix: str) -> int:
    """Length of the weight layout's contiguous K run, or 0 if unknown.

    This is the `k_lane` of the packed layout: the weight stores `k_lane`
    consecutive K values for one N, and the N groups follow each other, so the
    payload slot `s` maps to `(n = s / k_lane, k = s % k_lane)` and consecutive
    slots are consecutive addresses.

    Walking that order is what makes adjacent *threads* read adjacent
    addresses. Walking the logical `(n, k)` order instead -- one thread per N,
    stepping K -- puts adjacent threads one whole run apart, which is the
    32 KB-strided scatter the Triton backend's own notes identify as the
    difference between ~16-25 GB/s and full bandwidth (ITERATION.md Trial 79).

    Measured from the call's own address formula rather than assumed: the ABI's
    innermost stride for these buffers is 128, not 1, so the run length is a
    property of the formula and not of any single ABI axis.
    """

    import re

    expression = call.get("weight_offset") or call.get("gate_weight_offset")
    if not isinstance(expression, str) or not expression:
        return 0
    flat = re.sub(r"\[\s*(?:None\s*,\s*:|:\s*,\s*None)\s*\]", "", expression)

    def offset(k: int, n: int) -> int:
        return eval(  # noqa: S307 - renderer-generated integer algebra
            flat,
            {"__builtins__": {}, "tl": None},
            {f"{prefix}_global_k": k, f"{prefix}_global_n": n, "one": 1},
        )

    try:
        base = offset(0, 0)
        run = 1
        while run < 64 and offset(run, 0) == base + run:
            run *= 2
        if run < 2:
            return 0
        # The payload order only holds if the next N group begins exactly where
        # this one ends. Verify rather than assume; a layout that interleaves
        # differently keeps the logical-order path.
        if offset(0, 1) != base + run:
            return 0
        return run
    except Exception:
        return 0


def _argument_types(
    descriptor: Mapping[str, object], module: IRModule
) -> dict[str, str]:
    """Resolve the C++ element type of every generated pointer parameter.

    A Triton kernel parameter arrives already typed: the JIT reads the dtype
    off the tensor for an external buffer, and a byte pool carries an explicit
    ``.to(tl.pointer_type(...))`` cast in the renderer's own pointer
    expressions. C++ has neither, so the element type has to come from
    somewhere -- and the only authority on it is the buffer plan.

    This reads it back out of the call ABI: every buffer binding names the
    runtime argument it resolves against and the scalar dtype it views that
    argument with. Pooled storage keeps its byte-pointer parameter, because
    the renderer's cast already retypes it at each use. A non-pooled argument
    viewed through two different dtypes raises rather than picking one.

    An unresolved argument also raises. An earlier revision defaulted to
    ``unsigned char *``, which compiled cleanly and then read a bf16 buffer
    one byte at a time -- wrong results, no diagnostic. There is no safe
    default here, so there is none.
    """

    from triton.flagmega.codegen.cpp.expressions import cpp_scalar_type

    byte_pointer = "unsigned char"
    resolved: dict[str, dict[str, object]] = {}

    def record(argument: str, abi: Mapping[str, object]) -> None:
        storage = str(abi.get("storage"))
        spelling = (
            byte_pointer
            if bool(abi.get("pooled", False))
            or storage in {"rdata", "workspace"}
            else cpp_scalar_type(str(abi.get("scalar_dtype")))
        )
        # A compile-time/runtime scalar is passed by value; every other
        # argument is a pointer into a buffer.
        entry = {"type": spelling, "pointer": storage != "scalar"}
        previous = resolved.setdefault(argument, entry)
        if previous != entry:
            raise CodegenError(
                f"Runtime argument {argument!r} is viewed as both "
                f"{previous} and {entry}; the C++ backend cannot give it one "
                "parameter type.",
                stage=module.stage,
            )

    # `render_calls` is every call in the package, from the entry and from each
    # reusable device function, with each buffer binding naming the formal of
    # its own enclosing function. Reading it covers device-function parameters
    # as well as entry parameters.
    for call in descriptor.get("render_calls", ()):
        # `parameters` holds the kernel's non-buffer template parameters; only
        # inputs and outputs carry buffer bindings.
        for group in (*call.get("inputs", ()), *call.get("outputs", ())):
            for buffer in group.get("buffers", ()):
                abi = buffer.get("abi")
                argument = buffer.get("runtime_argument")
                if isinstance(abi, Mapping) and isinstance(argument, str) and argument:
                    record(argument, abi)
    required = [
        *(str(name) for name in descriptor.get("signature_arguments", ())),
        *(
            str(name)
            for function in descriptor.get("device_functions", ())
            for name in function.get("signature_arguments", ())
        ),
    ]
    types: dict[str, dict[str, object]] = {}
    for name in required:
        if name not in resolved:
            raise CodegenError(
                f"C++ codegen cannot type argument {name!r}: no kernel call "
                "binds it to a buffer ABI.",
                stage=module.stage,
            )
        types[name] = resolved[name]
    return types


def render_cpp_package(
    descriptor: Mapping[str, object], renderer_version: str
) -> str:
    context = {**descriptor, "renderer_version": renderer_version}
    return CppTemplateRegistry().render("module.cpp.jinja", context)


def _grid_total_blocks(descriptor: Mapping[str, object]) -> int:
    grid = descriptor.get("grid")
    if not isinstance(grid, (tuple, list)) or len(grid) != 3:
        raise CodegenError("C++ package descriptor has no three-dimensional grid.")
    total = 1
    for extent in grid:
        total *= int(extent)
    return total


def _reject_unsupported_constructs(
    descriptor: Mapping[str, object], module: IRModule
) -> None:
    """Fail on descriptor features the C++ templates do not implement yet.

    Each of these is a real feature of the Triton backend, not a corner case,
    and each would silently change behavior if ignored: a dropped fusion
    epilogue computes the wrong value, a dropped warp-specialized pipeline
    computes the right value by the wrong schedule, and a dropped host tensor
    descriptor reads unbound memory. Raising keeps "compiled with the C++
    backend" an honest statement about what the source does.
    """

    if descriptor.get("pipeline_schedule") is not None:
        raise CodegenError(
            "The C++ backend does not implement warp-specialized "
            "producer/consumer pipelines; compile this model without the "
            "SMEM pipeline implementations or use the Triton backend.",
            stage=module.stage,
        )
    # A device function whose schedule has only consumer events is an ordinary
    # reusable function -- the whole CTA runs its body. One with producer or
    # auxiliary events is warp-specialized and needs the pipeline machinery
    # rejected above, so check per function rather than rejecting the list.
    specialized = tuple(
        str(function.get("symbol"))
        for function in descriptor.get("device_functions", ())
        if function.get("schedule", {}).get("producer_events")
        or function.get("schedule", {}).get("auxiliary_events")
    )
    if specialized:
        raise CodegenError(
            "The C++ backend does not implement warp-specialized device "
            f"function(s) {list(specialized)}.",
            stage=module.stage,
        )
    if descriptor.get("host_tensor_descriptor_specs"):
        raise CodegenError(
            "The C++ backend does not implement host tensor descriptors "
            "(TMA-style descriptor arguments).",
            stage=module.stage,
        )
    fused = tuple(
        str(call.get("call"))
        for call in descriptor.get("render_calls", ())
        if call.get("fusion_lines")
    )
    if fused:
        raise CodegenError(
            f"The C++ backend does not implement fused epilogues on call(s) "
            f"{list(fused)}.",
            stage=module.stage,
        )
    # Intermediate dtype round-trips applied to an addend before a fused add.
    # Each one is a rounding step that changes the result, and the C++
    # templates cannot name an arbitrary Triton dtype spelling, so a non-empty
    # list must raise rather than be skipped.
    recast = tuple(
        str(call.get("call"))
        for call in descriptor.get("render_calls", ())
        if call.get("addend_cast_types")
    )
    if recast:
        raise CodegenError(
            "The C++ backend does not implement intermediate addend casts on "
            f"call(s) {list(recast)}.",
            stage=module.stage,
        )


def render_cpp_source_package(
    module: IRModule,
    output_dir: str | Path,
    *,
    dumper: Dumper | None = None,
) -> dict[str, object]:
    """Write one generated ``generated_kernels.cpp`` and describe it."""

    verify_module(
        module,
        expected_stage="bufferized_tir",
        expected_dialect="bufferized_tir",
    )
    verify_buffer_plan(module)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    descriptor = describe_cpp_package(module)
    source = render_cpp_package(descriptor, CPP_RENDERER_VERSION)
    source_path = destination / "generated_kernels.cpp"
    source_path.write_text(source, encoding="utf-8")
    package = {
        "renderer": CPP_RENDERER_VERSION,
        "renderer_spec": "bufferized-tir-cpp",
        "kind": "tir_call_graph_cpp/v1",
        "source": source_path.name,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "symbol": descriptor["symbol"],
        "grid": descriptor["grid"],
        "block_threads": descriptor["block_threads"],
        "dynamic_argument_indices": descriptor.get(
            "dynamic_argument_indices", ()
        ),
        **{
            key: value
            for key, value in descriptor.items()
            if key not in {
                "symbol",
                "grid",
                "kernel_templates",
                "kernel_template_specs",
                "render_calls",
                "entry_events",
                "entry_template",
                "block_threads",
            }
        },
    }
    if dumper is not None:
        from triton.flagmega.codegen.triton.diagnostics import (
            dump_codegen,
            dump_schedules,
        )

        dump_schedules(module, dumper)
        dump_codegen(module, package, source, dumper)
    return package


__all__ = [
    "CPP_BLOCK_THREADS",
    "CPP_PACKAGE_DESCRIPTOR_SCHEMA",
    "CPP_RENDERER_VERSION",
    "describe_cpp_package",
    "render_cpp_package",
    "render_cpp_source_package",
]
