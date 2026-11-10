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
import ast
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


CPP_RENDERER_VERSION = "flagmega-cpp-renderer/v3"

CPP_PACKAGE_DESCRIPTOR_SCHEMA = "flagmega.cpp-package-descriptor/v1"

# Threads per CTA for a generated C++ entry.
#
# The Triton backend derives its block size from `num_warps`, which the
# autotuner owns. A C++ kernel has no autotuner, so the block size is part of
# the generated source and must be stated here.
#
# 1024 = 16 physical warps x 64 lanes. The shared descriptor's `num_warps`
# describes the Triton launch; the C++ host shim uses CPP_BLOCK_THREADS.
# Measured on BI-V150 with the 28-layer Qwen3-1.7B batch-1 decode model
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
#
# Re-confirmed 2026-10-13 with the SME staging path enabled: 2048 still fails
# the launch outright (backend error 1), so this is not a limit the SME work
# relaxed. Wider CTAs would help the staging engine -- at 16 CTAs it measures
# 628 GB/s with 12 warps of a 1024-thread block against 669 with 32 warps of a
# 4096-thread block -- but the register footprint, not the thread count, is
# what the driver rejects, and cutting registers means giving up the in-flight
# depth the vector path depends on. `FLAGMEGA_CPP_BLOCK_THREADS` overrides it
# for re-measurement if that footprint ever shrinks.
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
# kernels. Target selection below additionally requires the flat SME engine.
CPP_USE_SME = os.environ.get("FLAGMEGA_CPP_SME", "1") == "1"

# Two disjoint batches of staging slots. G2S count waits complete the oldest
# batch while retaining the newer one in flight; a full drain after each tile
# loses that issue depth. Four slots hold two groups per batch and cost 96 KB
# at the largest GLU site, within its 112 KB staging budget. Each lane also
# completes its shared reads before the producer may overwrite consumed slots.
CPP_SME_SLOTS = int(os.environ.get("FLAGMEGA_CPP_SME_SLOTS", "4"))

# Padding elements appended to each staged tile, to shift its bank alignment.
# A standalone probe showed +53% on this access pattern from padding, but that
# probe disagreed with the kernel's own achieved bandwidth by 2x, so this is a
# switch to be measured in the real kernel rather than a believed win.
CPP_SME_PAD = int(os.environ.get("FLAGMEGA_CPP_SME_PAD", "0"))

# Chunked payload walk in the fused QKV GEMV. Off because it loses, and the
# reason has now been checked twice. First measured at 20.30 against 20.04
# ms/token: the kernel writes three outputs from one K loop, so a chunk of 8
# triples its live register count. Re-measured after the packed bf16 dot
# halved the inner loop's register needs, in case that was the binding
# constraint -- still slower, 9.391 against 9.245. Register pressure from the
# three fused outputs is the cause, not the arithmetic form.
#
# This leaves QKV the least utilized GEMV site (405 GB/s, 49.5% of peak
# against GLU's 65.4%), so it is where the headroom is -- but it needs fewer
# live outputs per loop, not a wider payload walk.
CPP_QKV_TILED = os.environ.get("FLAGMEGA_CPP_QKV_TILED", "0") == "1"

# Hold the QKV activation element in a register across the payload items one
# thread owns, instead of reloading it per item.
#
# The premise holds: a thread's items are `block_threads` apart and that is a
# whole multiple of `k_lane`, so every item it owns reads a byte-identical
# activation element, and an off-device oracle confirms the hoisted walk
# multiplies the identical (weight, source) multiset for all 128 outputs while
# emitting 25% fewer loads. It is still a measured loss -- 10.31 against 9.56
# ms/token, reproduced in both orders on one card -- so it is off.
#
# The reason is why it matters, because it bounds the whole family: moving the
# activation out of the step loop puts the N items *inside* the unrolled body,
# and the compiler fully unrolls `items * kg` instead of `kg`. That doubled
# body measured +3584 `ml_fma_f32` and +3584 `ml_cvt_f32_bf16`, 27% more
# arithmetic, with spills unchanged at 2386 and address arithmetic unchanged at
# 3 `ml_mad_u32_u16`. The loads it removed were L2 hits on a 4 KB activation,
# while the instructions it added occupy the issue slots this kernel is
# actually limited by -- so it trades cheap loads for expensive issue.
#
# This is the same ordering that makes `CPP_QKV_TILED` lose, by a different
# mechanism: that one widens the weight load and pays in registers, this one
# narrows the load count and pays in issue slots. The site is at 405 GB/s
# (49.5% of peak) because its scalar walk converts each 2-byte weight
# separately -- `ftt::to_float(weight) * ftt::to_float(source)`, no packed
# dot -- and neither knob addresses that.
CPP_QKV_HOIST = os.environ.get("FLAGMEGA_CPP_QKV_HOIST", "0") == "1"

# Reduce an adjacent K pair per thread in the fused QKV GEMV, so the packed
# bf16 dot applies there too.
#
# The scalar walk advances one thread per K element, so a thread's consecutive
# elements are `k_lane` apart and never adjacent -- which is why this is the
# one GEMV family still emitting `to_float(w) * to_float(s)` per element after
# the packed dot landed everywhere else, and why it measures lowest at
# 405 GB/s (49.5% of peak). Pairing halves the work items, and here that is
# exactly `block_threads`, so residency stays at 100% rather than falling off
# the cliff a pair of 4 would reach.
CPP_QKV_PAIR = os.environ.get("FLAGMEGA_CPP_QKV_PAIR", "1") == "1"

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


def _target_capability(architecture: str):
    from triton.flagmega.targets import get_target, target_names
    for name in target_names():
        target = get_target(name)
        if str(getattr(target, "codegen_architecture", "")) == architecture:
            return target.capability
    raise CodegenError(f"No registered C++ capability for {architecture!r}.")


def _cpp_block_threads(descriptor, capability) -> int:
    # Preserve the measured COREX default. NVIDIA follows the saved physical
    # launch plan instead of inheriting a 16-warp, 64-lane Iluvatar setting.
    default = (CPP_BLOCK_THREADS if descriptor["platform"] == "corex"
               else int(descriptor["num_warps"]) * int(capability.warp_size))
    try:
        threads = int(os.environ.get("FLAGMEGA_CPP_BLOCK_THREADS", default))
    except ValueError as error:
        raise CodegenError("FLAGMEGA_CPP_BLOCK_THREADS must be an integer.") from error
    if threads <= 0 or threads % capability.warp_size or threads > capability.max_threads_per_block:
        raise CodegenError(
            f"C++ block_threads={threads} must be a positive multiple of the target "
            f"warp size {capability.warp_size}, at most {capability.max_threads_per_block}.")
    return threads


def describe_cpp_package(module: IRModule) -> dict[str, object]:
    """Extend the shared TIR descriptor with the C++ package's own facts."""

    # The C++ device-function ABI passes one pointer per buffer and types every
    # formal from the ABI of the kernel call that binds it. The pooled calling
    # convention instead folds provable buffer formals into a pool base plus a
    # displacement read from a generated offset table, which leaves those
    # formals with no binding to be typed from and adds a table argument that
    # is not a buffer at all. Ask the shared descriptor for the unpooled
    # convention rather than teaching every template to undo it.
    descriptor = dict(describe_tir_package(module, pool_function_arguments=False))
    if 'counter_epoch_plan' in module.metadata and (
            not descriptor.get('programs') or descriptor.get('platform') != 'nvidia'):
        raise CodegenError('Counter epoch plans require a serial NVIDIA program package.', stage=module.stage)
    if descriptor.get("programs"):
        from .programs import describe_cpp_programs
        return describe_cpp_programs(module, descriptor, _describe_cpp_entry)
    return _describe_cpp_entry(module, descriptor)


def _describe_cpp_entry(module, descriptor):
    descriptor = dict(descriptor)
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
    if descriptor.get("distributed_entry") and tuple(descriptor["grid"][1:]) != (1, 1):
        raise CodegenError("C++ distributed mesh addressing requires a one-dimensional physical launch grid.")
    # Derive the per-call GEMV knobs first: the arena size below reads the
    # weight group stride this fills in.
    _gemv_k_streams(descriptor)
    from triton.flagmega.codegen.cpp.readonly_layout import select_readonly_layouts
    # The flat SME ISA and FIFO count waits are an ivcore11 capability.
    capability = _target_capability(architecture)
    # Lowering freezes the actual target capability, including injected
    # machine overrides. The registry is only a fallback for legacy IR.
    snapshot = module.metadata.get("target_capability")
    if isinstance(snapshot, Mapping):
        from types import SimpleNamespace
        capability = SimpleNamespace(**snapshot)
    block_threads = _cpp_block_threads(descriptor, capability)
    warp_size = int(capability.warp_size)
    if platform == "nvidia" and warp_size != 32:
        raise CodegenError("NVIDIA FTT requires the target's physical 32-lane warp contract.")
    use_sme = CPP_USE_SME and getattr(capability, "supports_flat_sme_count_wait", False)
    if use_sme:
        descriptor.update(block_threads=block_threads, warp_size=warp_size,
                          sme_slots=CPP_SME_SLOTS, sme_pad=CPP_SME_PAD)
        descriptor["readonly_device_layout"] = select_readonly_layouts(module, descriptor)
        for call in descriptor.get("render_calls", ()):
            if call.get("cpp_shard_major_weight") and call.get("variant") == "packed_k_major_gemv_norm_stats":
                groups = call["local_k_capacity"] // call["weight_k_lane"]
                tiles = call["local_n_capacity"] * call["weight_k_lane"] // 512
                concurrency = block_threads // warp_size // tiles
                if groups % (6 * concurrency) == 0:
                    call["sme_slots"] = 6
            if call.get("family") == "qkv_parallel_linear":
                outputs = call.get("outputs", ())
                source_walk = _affine_source_walk(call.get("source_offset", ""), "qkv_local_k_offsets")
                if source_walk is not None:
                    call["cpp_sme_source_base"], call["cpp_sme_source_stride"] = source_walk
                call["cpp_fused_sme"] = (
                    block_threads == 1024 and CPP_SME_PAD == 0
                    and call.get("k_lane") == 16
                    and call.get("n_lane") == 4
                    and [o["local_n_capacity"] for o in outputs] == [128, 64, 64]
                    and [o["weight_group_start"] for o in outputs] == [0, 32, 48]
                    and call.get("k_predicate_redundant")
                    and call.get("local_k_capacity", 0) % 256 == 0
                    and call.get("weight_n_group_stride") == 64
                    and call.get("weight_k_group_stride") == 4096
                    and source_walk is not None and source_walk[1] == 1
                    and call.get("weight_owner_stride") == (call.get("local_k_capacity", 0) // 16) * 4096)
        # A schedule barrier immediately after the call drains every CTA's
        # partial readers before any following call can reuse shared storage.
        # Standalone calls and schedules without that proof retain the exit
        # rendezvous in the fused template.
        _mark_shared_exit_barriers(descriptor)
    descriptor.update({
        "schema": CPP_PACKAGE_DESCRIPTOR_SCHEMA,
        "kernel_templates": tuple(registry.resolve(spec) for spec in specs),
        "entry_template": "entrypoints/call_graph.cpp.jinja",
        "block_threads": block_threads,
        "tiled_depth": CPP_TILED_DEPTH,
        # Warp geometry is a target property, not an FTT constant: the
        # reductions fold inside a warp before touching shared memory, so the
        # slot count follows the target's warp size. Taken from the registered
        # target's capability rather than hardcoded, so a target with a
        # different warp width renders correctly instead of silently
        # mis-sizing the buffers.
        "warp_size": warp_size,
        "warps": block_threads // warp_size,
        "launch_admission": "cuda-occupancy-cooperative/v1" if platform == "nvidia" else "corex-sm-bound/v1",
        "launch_abi_version": 2 if platform == "nvidia" else 1,
        "launch_scratch": ({
            "kind": "grid-barrier-counters/v1",
            "count": _grid_barrier_counter_slots(descriptor),
            "nbytes": 4 * _grid_barrier_counter_slots(descriptor),
            "dtype": "int32",
            "reset": "per-launch-stream",
        } if platform == "nvidia" and descriptor.get("distributed_entry") else None),
        "launch_resource_limits": {
            "max_threads_per_block": capability.max_threads_per_block,
            "max_shared_memory_bytes": capability.max_shared_memory_bytes,
            "max_registers_per_sm": capability.max_registers_per_sm,
            "compute_capability": list(getattr(capability, "compute_capability", (7, 1))),
        },
        "sme": use_sme,
        "sme_slots": CPP_SME_SLOTS,
        "sme_pad": CPP_SME_PAD,
        "qkv_tiled": CPP_QKV_TILED,
        "qkv_hoist": CPP_QKV_HOIST,
        "qkv_pair": CPP_QKV_PAIR,
        "run_load": "load_run_narrow" if CPP_NARROW_LOADS else "load_run",
        "sme_arena_elems": _sme_arena_elems(
            descriptor, block_threads // warp_size),
        "grid_total_blocks": _grid_total_blocks(descriptor),
        "grid_barrier_counter_slots": _grid_barrier_counter_slots(descriptor),
        "grid_barrier_group_bases": _barrier_group_bases(descriptor),
        "argument_types": _argument_types(descriptor, module),
    })
    from triton.flagmega.codegen.triton.gdn_head_gates import validate_call as validate_head_gates
    for call in descriptor.get("render_calls", ()):
        validate_head_gates(call, descriptor)
    if use_sme and descriptor["sme_arena_elems"]:
        descriptor["shared_arena_reuse"] = True
        _select_attention_staging(descriptor, capability)
        shared_floats = max((
            int(call.get("cpp_attention_shared_bytes", 0)) // 4
            or (2 * int(call.get("token_tile", 0)) + 4 + block_threads)
            for call in descriptor["render_calls"] if call["family"] == "paged_attention_partial"), default=0)
        if any(call["family"] == "greedy_sample" for call in descriptor["render_calls"]):
            shared_floats = max(shared_floats, 3 * block_threads)
        descriptor["sme_arena_elems"] = max(descriptor["sme_arena_elems"], 2 * shared_floats)
        if descriptor["sme_arena_elems"] * 2 > capability.max_shared_memory_bytes:
            raise CodegenError("C++ shared arena exceeds the target per-block memory budget.")
    if not use_sme:
        descriptor["sme_arena_elems"] = 0
    return descriptor


def _select_attention_staging(descriptor, capability):
    """Prove the uniform single-query, contiguous GQA-pair staging domain.

    Four head groups share a CTA and its barriers. Pair members cooperate on
    one KV tile, so neither differing query lengths nor a split through a KV
    head's query group is legal. Other domains use the general scalar template.
    """
    if descriptor["block_threads"] != 1024 or descriptor["warp_size"] != 64:
        return

    def affine(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value, {}
        if isinstance(node, ast.Name):
            return 0, {node.id: 1}
        if isinstance(node, ast.BinOp):
            a, av = affine(node.left)
            b, bv = affine(node.right)
            if isinstance(node.op, ast.Add):
                merged = dict(av)
                for key, value in bv.items():
                    merged[key] = merged.get(key, 0) + value
                return a + b, merged
            if isinstance(node.op, ast.Mult) and not (av and bv):
                return a * b, {key: value * b for key, value in av.items()} | {
                    key: value * a for key, value in bv.items()}
        raise ValueError

    for call in descriptor["render_calls"]:
        if call["family"] != "paged_attention_partial":
            continue
        if (call["variant"] != "decode_t16" or call["token_tile"] != 16
                or call["head_dim"] != 128 or call["query_dim_stride"] != call["query_lanes"]
                or call["query_lanes"] % 4 or call["num_query_heads"] != 2 * call["num_kv_heads"]
                or tuple(call["semantic_attrs"]["layout"]) != ("seq", "head", "dim")):
            continue
        inputs = {group["formal"]: group["buffers"] for group in call["inputs"]}
        query = inputs["q"][0]["abi"]
        cache = inputs["state"][0]["abi"]
        state = call["outputs"][0]["buffers"][0]["abi"]
        shape = tuple(state["local_capacity_shape"])
        if (query["scalar_dtype"] != "bfloat16" or query["logical_shape"][0] != 1
                or cache["scalar_dtype"] != "bfloat16" or shape[0] != 1 or shape[2] != 1
                or shape[1] % 4 or call["local_capacity"] != shape[1]):
            continue
        scalar_shape = list(cache["logical_shape"])
        scalar_shape[-1] *= cache["scalar_lane_count"]
        strides, stride = [], 1
        for extent in reversed(scalar_shape):
            strides.append(stride)
            stride *= extent
        # Packed dimension strides count physical vectors; scalar lanes fill
        # the final vector. The other dimensions are scalar storage strides.
        strides[0] = cache["scalar_lane_count"]
        if tuple(reversed(strides)) != tuple(cache["scalar_storage_strides"]):
            continue
        try:
            origin, coefficients = affine(ast.parse(
                state["logical_coordinate_expressions"][1], mode="eval").body)
        except (ValueError, SyntaxError):
            continue
        if (origin % 2 or coefficients.pop("local_coord_1", 0) != 1
                or any(value % 2 for value in coefficients.values())):
            continue
        heads, tile, padding, groups = 4, 64, 32, 8
        shared_bytes = 4 * (heads * (2 * tile + 16 + 2 * groups * 128)) + (
            heads // 2 * 2 * tile * (128 + padding) * 2)
        if shared_bytes > capability.max_shared_memory_bytes:
            continue
        call["cpp_attention_staging"] = {
            "heads": heads, "token_tile": tile, "padding": padding,
            "groups": groups, "gqa": 2, "softmax_tile": 16,
            "value_buffers": 2,
        }
        call["cpp_attention_shared_bytes"] = shared_bytes


def _affine_source_walk(expression: str, variable: str) -> tuple[int, int] | None:
    """Prove an entire source expression is base + stride * local K."""
    def affine(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value, 0
        if isinstance(node, ast.Name) and node.id == variable:
            return 0, 1
        if not isinstance(node, ast.BinOp):
            raise ValueError
        a, b = affine(node.left), affine(node.right)
        if isinstance(node.op, ast.Add):
            return a[0] + b[0], a[1] + b[1]
        if isinstance(node.op, ast.Mult) and not (a[1] and b[1]):
            return a[0] * b[0], a[1] * b[0] + b[1] * a[0]
        raise ValueError
    try:
        base, stride = affine(ast.parse(expression.strip(), mode="eval").body)
        return (base, stride) if base >= 0 and base % 8 == 0 and stride > 0 else None
    except (SyntaxError, ValueError, AttributeError):
        return None


def _mark_shared_exit_barriers(descriptor):
    """Merge an exit rendezvous only when every enclosing call reaches one."""
    functions = {f["function"]: f for f in descriptor.get("device_functions", ())}
    schedules = [(None, descriptor["entry_events"]), *(
        (name, f["schedule"]["consumer_events"]) for name, f in functions.items())]

    def follows_barrier(events, index, enclosing, seen=()):
        if index + 1 < len(events):
            return events[index + 1].get("kind") == "barrier"
        if enclosing is None or enclosing in seen:
            return False
        callers = [(owner, events, i) for owner, events in schedules
                   for i, event in enumerate(events)
                   if event.get("kind") == "function_call" and event.get("callee") == enclosing]
        return bool(callers) and all(follows_barrier(events, i, owner, (*seen, enclosing))
                                     for owner, events, i in callers)

    for call in descriptor["render_calls"]:
        if not (call.get("cpp_fused_sme") or (
                call.get("family") == "dense_matmul"
                and call.get("variant") == "packed_k_major_gemv_norm_stats")):
            continue
        uses = [(owner, events, i) for owner, events in schedules
                for i, event in enumerate(events) if event.get("symbol") == call["symbol"]]
        if uses and all(follows_barrier(events, i, owner) for owner, events, i in uses):
            call["cpp_shared_exit_barrier_in_schedule"] = True


def _sme_arena_elems(descriptor: Mapping[str, object], warps: int) -> int:
    """Elements the shared SME staging arena needs, or 0 if unused.

    Sized for the largest single staging site rather than the sum: the sites
    are separated by schedule barriers, so their live staging data can share
    one allocation. The completed arena is checked against the registered
    target's per-block shared-memory budget.
    """

    if not CPP_USE_SME:
        return 0
    largest = 0
    for call in descriptor.get("render_calls", ()):
        family = call.get("family")
        slots = int(call.get("sme_slots", CPP_SME_SLOTS))
        if family == "qkv_parallel_linear":
            if call.get("cpp_fused_sme"):
                largest = max(largest, warps * 8 * 512)
            # Q, K and V stage concurrently across K even when the package
            # contains no dense/GLU site to supply an arena on their behalf.
            lane = int(call.get("k_lane") or 0)
            for output in call.get("outputs", ()):
                group = int(output.get("local_n_capacity") or 0) * lane
                if group and group % 512 == 0:
                    tiles = group // 512
                    if tiles < warps and warps % tiles == 0:
                        largest = max(
                            largest, warps * CPP_SME_SLOTS * (512 + CPP_SME_PAD))
            continue
        if family not in {"dense_matmul", "dense_matmul_glu"}:
            continue
        lane = int(call.get("weight_k_lane") or 0)
        capacity = int(call.get("local_n_capacity") or 0)
        if not lane or not capacity or not call.get("weight_group_stride"):
            continue
        group = capacity * lane
        if group % 512 or (family == "dense_matmul" and group // 512 > warps):
            # Large N may contain complete tiles plus a bounded tail.
            if family == "dense_matmul" and group >= 512:
                largest = max(largest, warps * CPP_SME_SLOTS * (512 + CPP_SME_PAD))
            continue
        tiles = group // 512
        if tiles > warps:
            continue
        # `CPP_SME_SLOTS` ring slots per projection the kernel stages.
        projections = 2 if family == "dense_matmul_glu" else 1
        if family == "dense_matmul" and tiles < warps:
            # A site whose k-group holds fewer tiles than there are warps
            # stages several groups at once instead, so that every warp owns a
            # tile: `warps // tiles` groups concurrently, `warps` tiles in all.
            # Sizing this from `tiles` alone would under-allocate the arena,
            # and an undersized arena does not fail the launch -- it silently
            # reads garbage.
            staged = (warps // tiles) * tiles
        else:
            staged = tiles
        largest = max(
            largest,
            staged * slots * projections * (512 + CPP_SME_PAD))
    return largest


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
            stride = _weight_group_stride(call, prefix, lane)
            if stride:
                call["weight_group_stride"] = stride


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


def _weight_group_stride(
    call: Mapping[str, object],
    prefix: str,
    lane: int,
) -> int | None:
    """Elements between consecutive K groups of the packed weight, or None.

    An SME producer stages a whole K group per transaction batch, so it needs
    the distance from one group to the next. Derived from the call's own
    address formula (the offset at `k = lane` minus the offset at `k = 0`)
    rather than from the layout name, and only accepted when the step is
    uniform across several groups -- a layout that interleaves differently
    falls back to the non-staged path instead of staging wrong bytes.
    """

    import re

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
        base = offset(0, 0)
        step = offset(lane, 0) - base
        if step <= 0:
            return None
        for group in range(1, 4):
            if offset(group * lane, 0) - base != group * step:
                return None
        return int(step)
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
) -> dict[str, dict[str, object]]:
    """Resolve the C++ element type of every generated pointer parameter.

    A Triton kernel parameter arrives already typed: the JIT reads the dtype
    off the tensor for an external buffer, and a byte pool carries an explicit
    ``.to(tl.pointer_type(...))`` cast in the renderer's own pointer
    expressions. C++ has neither, so the element type has to come from
    somewhere -- and the only authority on it is the buffer plan.

    This reads it back out of the call and runtime ABIs: every buffer binding names the
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
    from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
    from triton.flagmega.codegen.triton.runtime_binding import describe_function_runtime_binding
    from triton.flagmega.ir import verify_buffer_plan

    byte_pointer = "unsigned char"
    resolved: dict[str, dict[str, object]] = dict(descriptor.get("native_argument_types", {}))

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
    # A reference or bank field can enter only through a reusable function,
    # or remain an unused external parameter. Its typed physical buffer still
    # belongs to the runtime signature even when no kernel consumes it here.
    # Resolve those roots through the same binding/BufferPlan authority used
    # by source scheduling, without inventing a byte-pointer fallback.
    plan = verify_buffer_plan(module)
    root_binding = descriptor.get("runtime_binding")
    bindings = [root_binding if isinstance(root_binding, Mapping) else
                describe_function_runtime_binding(module)]
    bindings.extend(describe_function_runtime_binding(module, function_name=str(function["function"]))
                    for function in descriptor.get("device_functions", ()))
    for binding in bindings:
        for value in binding.get("values", ()):
            argument, buffer_id = value.get("runtime_argument"), value.get("buffer")
            if isinstance(argument, str) and argument and buffer_id in plan.buffer_map:
                record(argument, describe_local_buffer_abi(plan.buffer_map[buffer_id], plan))
        for pool in binding.get("pools", ()):
            # Runtime pools are explicit byte allocations. Their scalar views
            # are supplied at each use, independently of any external dtype.
            if pool.get("storage") not in plan.memory_space_map:
                raise CodegenError("C++ runtime pool has no physical memory-space contract.", stage=module.stage)
            record(str(pool["name"]), {"storage": pool["storage"], "pooled": True})
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
                "or runtime signature binds it to a physical buffer ABI.",
                stage=module.stage,
            )
        types[name] = resolved[name]
    return types


def render_cpp_package(
    descriptor: Mapping[str, object], renderer_version: str
) -> str:
    if descriptor.get("cpp_programs"):
        from .programs import render_program_package
        return render_program_package(descriptor, renderer_version)
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

    if descriptor.get('programs'):
        raise CodegenError(
            'The C++ backend does not implement separate kernel programs and their per-program execution meshes.',
            stage=module.stage,
        )
    unsupported_encodings = tuple(
        str(call.get("call"))
        for call in descriptor.get("render_calls", ())
        if ((call.get("family") in {"block_fp8", "matmul_glu", "interleaved_projection"}
             and call.get("weight_encoding") not in {"IQ4_XS", "Q4_K"})
            or (call.get("family") == "embedding"
                and call.get("weight_encoding") not in {None, "IQ4_XS", "Q4_K"}))
    )
    if unsupported_encodings:
        raise CodegenError(
            "C++ kernels do not implement the weight encoding on call(s) "
            f"{list(unsupported_encodings)}.", stage=module.stage)
    for call in descriptor.get("render_calls", ()):
        if call.get("family") == "matmul_glu" and any(
                call.get(key) not in (None, "None") for key in ("gate_clamp_max", "up_clamp")):
            raise CodegenError("C++ native GLU does not implement projection clamps.", stage=module.stage)
        if (call.get("family") == "gdn_recurrent"
                and (call.get("b_weight_encoding") or call.get("a_weight_encoding"))
                and int(call.get("projection_dim", 0)) % 256):
            raise CodegenError("C++ packed projection requires a whole number of 256-element blocks.",
                               stage=module.stage)
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
    from triton.flagmega.codegen.cpp.fusion import validate_call
    for call in descriptor.get("render_calls", ()):
        validate_call(call)


def render_cpp_source_package(
    module: IRModule,
    output_dir: str | Path,
    *,
    dumper: Dumper | None = None,
    cpp_operator_profile: str | None = None,
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
    if cpp_operator_profile is not None:
        from triton.flagmega.codegen.cpp.operator_profile import configure_descriptor
        descriptor = configure_descriptor(descriptor, cpp_operator_profile)
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
