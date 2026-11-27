from __future__ import annotations
from typing import Any


def _to_handle(v, builder, ty):
    """Convert a JIT input value to an MLIR Value handle.

    ``ty`` is the parameter's structured Ty (types.py). tl.constexpr (triton
    specializes small ints, e.g. P=1, as constexpr inside @triton.jit) has no
    .handle; emit arith.constant instead, materialising the constant's MLIR
    type structurally via ``ty.build(builder)`` (no type text).
    """
    from .types import ScalarTy
    if hasattr(v, "handle"):
        return v.handle
    # tl.constexpr case
    val = v.value if hasattr(v, "value") else v
    if isinstance(val, int):
        # index / iN params both lower to an index constant here (matches the
        # pre-refactor behaviour: the i32/i64 "fallback" also used index).
        return builder.create_arith_constant_index(val)
    if isinstance(val, float):
        if isinstance(ty, ScalarTy) and ty.is_float:
            ft = ty.build(builder)
        else:
            ft = builder.get_f32_type()
        return builder.create_arith_constant_float(val, ft)
    raise TypeError(f"smt_rvisa.call: cannot convert constexpr {val!r} "
                    f"(param type {ty!r}) to IR handle")


# LLVM-direct handoff: the @triton.jit body (which calls call()) runs during
# make_ir, strictly before the "ttir" stage's make_ttir reads it. Triton
# compiles one kernel at a time, so a process-global holder is a safe, C++-free
# channel from call() → make_ttir (avoids binding get_module/set_attr, which
# don't exist in this libtriton API and would need a full riscv64 rebuild).
_PENDING_LLVM_DIRECT_MODULE: dict[str, Any] = {}


def _clear_pending_llvm_direct_module():
    """Clear all pending state — call at the start of make_ttir to prevent stale data."""
    _PENDING_LLVM_DIRECT_MODULE.clear()


def take_pending_llvm_direct_module():
    """make_ttir calls this to retrieve + clear a pending llvm-direct module.

    Returns (module_text, emitted_symbol_name), or (None, None) if the
    just-compiled kernel was not a llvm-direct kernel.
    """
    text = _PENDING_LLVM_DIRECT_MODULE.pop("text", None)
    name = _PENDING_LLVM_DIRECT_MODULE.pop("name", None)
    return text, name


def take_pending_llvm_funcs():
    """make_ttir calls this to retrieve + clear pending llvm.func siblings for mixed mode.

    Returns list of llvm.func text strings (no module wrapper), or empty list.
    Used when host contains both normal ops and llvm-direct calls.
    """
    funcs = _PENDING_LLVM_DIRECT_MODULE.pop("llvm_funcs", [])
    return funcs


def take_pending_llvm_calls():
    """Retrieve + clear pending mixed-mode llvm.call bridge specs.

    Returns a list of {"callee": str, "arg_bridge": [{"pos": int,
    "kind": "ptr"|"scalar"}, ...]}, one per _sr_call to an llvm-direct kernel
    inside a host that also does other work. `pos` is the host func.func
    argument index; `kind` selects the bridge (memref→data-ptr-i64 vs i32→i64).
    Empty list if the just-compiled kernel had no mixed llvm-direct calls.
    """
    return _PENDING_LLVM_DIRECT_MODULE.pop("llvm_calls", [])


def take_pending_host_arg_kinds():
    """Retrieve + clear the ordered per-host-arg is-memref flags for mixed mode.

    Returns list[bool], one entry per host func.func parameter in signature
    order: True = memref (a ptr param), False = scalar. Recorded structurally
    from the host TTIR entry-block arg types (Value.get_type) at call() time,
    so compiler.py can map each TTIR arg position → its lowered
    (i64 rank, !llvm.ptr) descriptor slots WITHOUT re-parsing the func.func
    signature text. Empty list if the kernel had no mixed llvm-direct calls.
    """
    return _PENDING_LLVM_DIRECT_MODULE.pop("host_arg_kinds", [])


def call(fn, outputs=None, inputs=None, _semantic=None):
    """Inside @triton.jit: emit tle.dsl_region TTIR op holding the raw kernel body.

    Uses create_tle_dsl_region_direct — body_builder builds ops via C++ builder API
    with no MLIR text round trip.

    LLVM-direct bypass: if fn has _llvm_direct=True, emit llvm.func module text and pass it
    via module attr to metadata (compiler.py reads it in make_ttir).

    When _semantic is None (interpreter mode / outside JIT) the call is a no-op.
    """
    if _semantic is None:
        return
    if inputs is None:
        inputs = []

    # LLVM-direct bypass: detect and emit
    if getattr(fn, '_llvm_direct', False):
        from .llvm_direct import emit_llvm_func_for_inline
        from .codegen import _parse_signature
        # emit_llvm_func_for_inline needs the raw Python function, not the JIT wrapper
        raw_fn = fn._fn if hasattr(fn, '_fn') else fn

        # Arity guard: inputs must match kernel signature
        n_params = len(_parse_signature(raw_fn))
        if len(inputs) != n_params:
            raise ValueError(f"smt_rvisa.call: LLVM-direct kernel {raw_fn.__name__!r} declares "
                             f"{n_params} parameter(s) but got {len(inputs)} input(s).")

        # Emit the sibling llvm.func (no module wrapper). param_types is the
        # sibling ABI (every param → i64: memref=data-ptr-as-i64, scalar=i64).
        llvm_func_text, param_types = emit_llvm_func_for_inline(raw_fn)
        if "llvm_funcs" not in _PENDING_LLVM_DIRECT_MODULE:
            _PENDING_LLVM_DIRECT_MODULE["llvm_funcs"] = []
        _PENDING_LLVM_DIRECT_MODULE["llvm_funcs"].append(llvm_func_text)

        # Mixed-mode host bridge is emitted by compiler.py at the *linalgdir*
        # stage (func.func form), where memrefs exist and llvm.call is legal —
        # not here at TTIR (tt.ptr, no bridge ops). We can't reference the host
        # func.func's SSA args from here, but they map 1:1 by POSITION to the
        # host's entry-block args (verified: tt.func user params → func.func
        # %arg0.. in the same order). So record each input's host-arg index.
        builder = _semantic.builder
        entry = builder.get_insertion_block()
        n_block_args = entry.get_num_arguments()
        # NOTE: .id is a bound method on this libtriton build (pybind11), not a
        # property — call it. Using the method object as a dict key silently never
        # matches, so every input would look like a non-host-arg. (K3-verified.)
        argid_to_pos = {entry.get_argument(i).id(): i for i in range(n_block_args)}

        # Structurally record each host arg's is-memref flag from its TTIR type:
        # ptr params (`!tt.ptr<...>`) lower to memref → (i64 rank, !llvm.ptr) pairs;
        # scalars stay one lowered slot. compiler._inject_mixed_llvm_llmlir consumes
        # this ordered bool list to map TTIR arg position → lowered arg index,
        # instead of re-parsing the func.func signature text. The host entry block
        # is identical across all _sr_call sites in one kernel, so overwrite freely.
        # Prefix check on the printed type (a ptr arg is exactly `!tt.ptr<elt>`):
        # a plain substring test would also match types that merely EMBED a tt.ptr
        # (e.g. tensor<128x!tt.ptr<f32>>). tt.ptr is an unregistered-dialect type
        # in this libtriton build (no Python class to isinstance against), so the
        # printed-form prefix is the structural check available here.
        host_arg_kinds = [str(entry.get_argument(i).get_type()).startswith("!tt.ptr") for i in range(n_block_args)]
        _PENDING_LLVM_DIRECT_MODULE["host_arg_kinds"] = host_arg_kinds

        params = _parse_signature(raw_fn)  # [(pname, ann), ...]
        arg_bridge = []  # per-input: {"pos": int, "kind": "ptr"|"scalar"}
        for (pname, ann), v in zip(params, inputs):
            if not hasattr(v, "handle"):
                raise ValueError(f"smt_rvisa.call: LLVM-direct kernel {raw_fn.__name__!r} in "
                                 f"mixed mode requires every input to be a host launch arg "
                                 f"(a tt.func parameter); got a computed/constexpr value for "
                                 f"{pname!r}. Compute derived values INSIDE the kernel from "
                                 f"sr.program_id(axis).")
            pos = argid_to_pos.get(v.handle.id())
            if pos is None:
                raise ValueError(f"smt_rvisa.call: input for {pname!r} of {raw_fn.__name__!r} "
                                 f"is not a host entry-block argument. In mixed mode inputs must "
                                 f"be the host's own launch parameters (bridged to the sibling "
                                 f"llvm.func by position at the linalgdir stage).")
            kind = "ptr" if ann.kind == "mem" else "scalar"
            arg_bridge.append({"pos": pos, "kind": kind})

        if "llvm_calls" not in _PENDING_LLVM_DIRECT_MODULE:
            _PENDING_LLVM_DIRECT_MODULE["llvm_calls"] = []
        _PENDING_LLVM_DIRECT_MODULE["llvm_calls"].append({
            "callee": raw_fn.__name__, "arg_bridge": arg_bridge,  # ordered per sibling param
        })

        # Positional anchor: emit an empty tle.dsl_region right HERE, at this
        # call's program point, so the bridge lands in source order and svector
        # stages can sit before AND after it. TLEToLinalg lowers the anchor to a
        # func.call @__spine_bridge_pt_N (private no-arg stub); it survives to
        # ll.mlir as `llvm.call @__spine_bridge_pt_N`, which
        # compiler._inject_mixed_llvm_llmlir text-replaces with the real bridge.
        # create_tle_dsl_region_direct auto-appends spine_ext.return, so an empty
        # body_builder is valid. Without the anchor the bridge would be forced to
        # llvm.return (all bridges last), forbidding svector-after-bridge.
        bridge_idx = len(_PENDING_LLVM_DIRECT_MODULE["llvm_calls"]) - 1
        anchor_name = f"__spine_bridge_pt_{bridge_idx}"
        builder.create_tle_dsl_region_direct(anchor_name, [], [], lambda b, ba: None)
        return  # sibling llvm.func text recorded; anchor marks the call site

    # Normal path
    param_tys, body_builder = fn.make_body_builder()
    builder = _semantic.builder
    param_types = [ty.build(builder) for ty in param_tys]
    handles = [_to_handle(v, builder, ty) for v, ty in zip(inputs, param_tys)]
    builder.create_tle_dsl_region_direct(
        fn.__name__,
        handles,
        param_types,
        body_builder,
    )


# Mark as triton builtin so JIT AST visitor injects _semantic automatically.
call.__triton_builtin__ = True
