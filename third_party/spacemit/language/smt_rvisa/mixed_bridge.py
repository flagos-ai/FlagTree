# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""Mixed-mode bridge injector: graft LLVM-direct sibling `llvm.func`s and the
host→sibling call bridges into the LOWERED `ll.mlir`, built entirely with the
upstream MLIR Python bindings (`mlir.dialects.llvm`).

Background. A "mixed" kernel combines the builder path (SmtMLIRBuilderCodegen,
which emits `vector`/`linalg` ops inlined into a `func.func` host) with
`smt_rvisa.call(...)` siblings emitted by the LLVM-direct backend as standalone
`llvm.func`s. spine-opt cannot lower raw `llvm.*` ops sitting inside the host
`func.func` (they trip BufferDeallocation / ConvertToScalableVector), so the
siblings are NOT inlined at the TTIR/linalg layer. Instead:

  1. `call_registry.py` emits, at each call site, an *anchor* — an empty
     `tle.dsl_region` named `__spine_bridge_pt_N` — plus records the sibling's
     text (`mixed_llvm_funcs`) and the per-call ABI bridge spec
     (`mixed_llvm_calls`) into the kernel metadata.
  2. spine-opt lowers the host to a uniform `llvm.func` in `ll.mlir`; the anchor
     survives as `llvm.call @__spine_bridge_pt_N() : () -> ()` plus a private
     no-body stub `llvm.func @__spine_bridge_pt_N()`.
  3. THIS module runs post-spine-opt, pre-mlir-translate: it replaces each
     anchor call, in place, with the real bridge that adapts the host's lowered
     memref-descriptor / scalar args to the sibling's flat `i64` ABI, then drops
     the dead stub, and appends the sibling `llvm.func`(s) + any needed runtime
     symbol declarations to the module.

Why bindings and not text. The bridge is real IR that must be verifier-clean
(correct descriptor type, correct SSA dominance, valid symbol references). The
former implementation spliced it in with regex over the `ll.mlir` text; that is
fragile (SSA-name collisions, whitespace, partial matches) and is exactly the
"text path" this project removed. Here every op is built through bindings op
builders and `Module.get_asm()` refuses to print IR that fails verification.

ABI notes (must match backend/driver.py `_launch` and llvm_direct.py):
  * The new spert ABI marks kernels `__require_context__`; spine-opt injects a
    leading `%arg0: i64` context handle into the lowered `llvm.func` that is NOT
    present in the linalg `func.func` signature `host_arg_is_memref` describes.
    Every lowered arg index is therefore shifted by `_CTX = 1`.
  * Each host memref param lowers to TWO ll args: `(i64 rank, !llvm.ptr desc)`
    where `desc` points to a StridedMemRefType
    `{allocated, aligned, offset, sizes[1], strides[1]}`. The aligned data
    pointer is field `[1]`; the sibling recovers it via `llvm.inttoptr`.
  * Each host scalar is an `i32`; the sibling's flat ABI wants `i64`, so bridge
    with `llvm.sext`.
  * The sibling also receives the host ctx handle (`%arg0`) as its last arg so
    `program_id()` (which calls `@spine_grid(ctx, axis)`) works.
"""
from __future__ import annotations

from ._mlir_loader import ensure_mlir_on_path

ensure_mlir_on_path()
try:
    from mlir import ir
    from mlir.dialects import llvm as dllvm
except ImportError as _e:  # pragma: no cover - environment guard
    raise ImportError(
        "smt_rvisa mixed-mode bridge injector requires the MLIR Python bindings "
        "(importable 'mlir' package). They are vendored into "
        "triton/backends/<backend>/mlir_core by setup.py at package time; if that "
        "copy is missing, rebuild the wheel (third_party/spacemit/scripts/install_flagtree_plugin.sh) from an "
        "LLVM install that has python_packages/mlir_core.") from _e

# The new spert ABI injects a leading `%arg0: i64` context handle at the ll.mlir
# layer (not present in the linalg func.func signature host_arg_is_memref is
# from), so every lowered arg index is shifted by +1.
_CTX = 1

# Lowered host memref descriptor: spine-opt lowers each memref<*xT> param to
# (i64 rank, !llvm.ptr desc), where desc points to a StridedMemRefType
# {allocated, aligned, offset, sizes[1], strides[1]}. The aligned data ptr is
# field [1]. (This is the HOST-side 5-field descriptor; the sibling-side
# llvm_direct.py uses a distinct 3-field rank-0 descriptor — different things.)
_LL_DESC_TEXT = "!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"

# Sibling kernels call runtime symbols provided by libspert.so at dlopen time.
# `@spine_grid` backs program_id (spine_grid(ctx, axis)). spine-opt's e2e
# pipeline may already declare it when the host uses tl.program_id, so we only
# add it when absent (re-declaring → "redefinition of symbol named 'spine_grid'").
_SPINE_GRID_DECL = "llvm.func @spine_grid(i64, i64) -> i64"


def _ttir_pos_to_ll_argidx(host_arg_is_memref: list[bool]) -> list[int]:
    """Map each TTIR host-arg position → its start index in the LOWERED ll.mlir
    signature. Each memref param expands to 2 ll args (i64 rank, !llvm.ptr);
    each scalar stays 1. Returns list[int] of start indices (ordered)."""
    starts = []
    ll = 0
    for is_mem in host_arg_is_memref:
        starts.append(ll)
        ll += 2 if is_mem else 1
    return starts


def _find_top_func(mod, name):
    """Return the top-level `llvm.func` with sym_name == name, else None."""
    for op in mod.body.operations:
        if isinstance(op, dllvm.LLVMFuncOp) and op.sym_name.value == name:
            return op
    return None


def _call_callee_name(op):
    """For an `llvm.call`, return its callee symbol name (str), else None.

    The attribute-map entry is already a FlatSymbolRefAttr; `.value` is the str.
    Gotcha: wrapping it in `ir.SymbolRefAttr(attr).value` yields a LIST of
    strings, which never equals a str — use FlatSymbolRefAttr.value directly.
    """
    if op.operation.name != "llvm.call":
        return None
    return ir.FlatSymbolRefAttr(op.operation.attributes["callee"]).value


def _find_call_in(host_fn, callee_name):
    for blk in host_fn.regions[0].blocks:
        for op in blk.operations:
            if _call_callee_name(op) == callee_name:
                return op
    return None


def _find_first_return(host_fn):
    for blk in host_fn.regions[0].blocks:
        for op in blk.operations:
            if op.operation.name == "llvm.return":
                return op
    return None


def _build_bridge(spec, ll_start, entry_args, DESC, PTR, I64):
    """Emit the host→sibling bridge ops for one call spec at the current
    insertion point: adapt each referenced host arg to the sibling's i64 ABI,
    append the ctx handle (%arg0), then `llvm.call @callee(...) -> ()`."""
    operands = []
    for item in spec["arg_bridge"]:
        pos, kind = item["pos"], item["kind"]
        base = ll_start[pos]
        if kind == "ptr":
            # memref: (rank=base, desc ptr=base+1); load desc, take aligned [1].
            desc_ptr = entry_args[base + 1]
            d = dllvm.load(DESC, desc_ptr)
            p = dllvm.extractvalue(PTR, d, [1])
            operands.append(dllvm.ptrtoint(I64, p))
        else:  # scalar i32 → sext to i64
            operands.append(dllvm.sext(I64, entry_args[base]))
    # Forward the host ctx handle (%arg0) so sibling program_id() works.
    operands.append(entry_args[0])
    # void callee: result must be None (not []) — bindings append a result only
    # when `result is not None`, and passing [] raises "Result 0 must be a Type".
    dllvm.call(None, operands, [], [], callee=spec["callee"])


def _ensure_spine_grid(mod, ctx):
    """Add the `@spine_grid` declaration to `mod` if not already present."""
    if _find_top_func(mod, "spine_grid") is not None:
        return
    decl_mod = ir.Module.parse("module { " + _SPINE_GRID_DECL + " }", ctx)
    for dop in list(decl_mod.body.operations):
        dop.operation.detach_from_parent()
        mod.body.append(dop.operation)


def _append_siblings(mod, ctx, llvm_funcs):
    """Parse the sibling llvm.func texts and move them into `mod`.

    A sibling may reference `@spine_grid` (program_id); parsing it standalone
    would fail symbol verification ("does not reference a symbol in the current
    scope"). So parse all siblings together with a spine_grid decl in scope,
    then move every op EXCEPT that decl (mod already carries its own spine_grid
    via _ensure_spine_grid)."""
    if not llvm_funcs:
        return
    wrapper = "module {\n" + _SPINE_GRID_DECL + "\n" + "\n".join(llvm_funcs) + "\n}"
    sib_mod = ir.Module.parse(wrapper, ctx)
    for sop in list(sib_mod.body.operations):
        if isinstance(sop, dllvm.LLVMFuncOp) and sop.sym_name.value == "spine_grid":
            continue
        sop.operation.detach_from_parent()
        mod.body.append(sop.operation)


def inject_mixed_llvm_llmlir(llmlir: str, func_name: str, host_arg_is_memref: list[bool], llvm_funcs,
                             llvm_calls) -> str:
    """Graft llvm.func siblings + host→sibling bridges into the LOWERED ll.mlir.

    Returns the injected module as MLIR text (get_asm). Structured replacement
    for the former regex splice; the produced module translates to byte-identical
    LLVM IR (verified against real mixed kernels).
    """
    with ir.Context() as ctx, ir.Location.unknown():
        mod = ir.Module.parse(llmlir, ctx)
        DESC = ir.Type.parse(_LL_DESC_TEXT, ctx)
        PTR = ir.Type.parse("!llvm.ptr", ctx)
        I64 = ir.IntegerType.get_signless(64, ctx)

        host_fn = _find_top_func(mod, func_name)
        if host_fn is None:
            raise RuntimeError(f"mixed-mode: host llvm.func @{func_name} not found in ll.mlir")
        entry_args = list(host_fn.regions[0].blocks[0].arguments)
        ll_start = [s + _CTX for s in _ttir_pos_to_ll_argidx(host_arg_is_memref)]

        # Preferred: replace each positional anchor in place — preserves source
        # order so svector stages can sit before AND after a bridge — then drop
        # the now-dead private stub.
        anchors_replaced = 0
        for n, spec in enumerate(llvm_calls):
            anchor_name = f"__spine_bridge_pt_{n}"
            anchor_op = _find_call_in(host_fn, anchor_name)
            if anchor_op is None:
                continue
            with ir.InsertionPoint(anchor_op):
                _build_bridge(spec, ll_start, entry_args, DESC, PTR, I64)
            anchor_op.operation.erase()
            stub = _find_top_func(mod, anchor_name)
            if stub is not None:
                stub.erase()
            anchors_replaced += 1

        if anchors_replaced == 0:
            # Fallback (kernels compiled before anchors existed): insert all
            # bridges before the host func's FIRST llvm.return.
            ret = _find_first_return(host_fn)
            if ret is None:
                raise RuntimeError("mixed-mode: no llvm.return in host func to anchor llvm.call")
            with ir.InsertionPoint(ret):
                for spec in llvm_calls:
                    _build_bridge(spec, ll_start, entry_args, DESC, PTR, I64)

        _ensure_spine_grid(mod, ctx)
        _append_siblings(mod, ctx, llvm_funcs)

        return mod.operation.get_asm()
