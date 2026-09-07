# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Independent execution-verifier UTs; no importer, codegen or GPU required."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.tir.execution import verify_execution_functions


def _buffer(name, function, start):
    return fm.T.buffer(name, "bfloat16", fm.MemSpan(fm.PhysicalBuffer(
        f"{function}.{name}", "shared", 32, 16, start=start, function=function,
    )), (16,), (1,))


def _module(*, nested=False, reused=False):
    tensor = fm.tensor_type("bfloat16", (16,))
    shared = tuple(_buffer(f"stage{index}", "kernel", index * 32) for index in range(2))
    dispatch = fm.T.kernel_dispatch(
        semantic_op="test.two_shared", arguments=(), outputs=("out",), writes=("out",),
        microkernel=fm.T.microkernel_selection(
            "test.two_shared", "test", "two_shared",
            shared_workspaces=tuple(fm.T.shared_workspace_descriptor(b.name, tensor, 16) for b in shared),
        ), shared_workspace_buffers=shared,
    )
    primitive = fm.T.prim_function("kernel", "triton", (
        fm.T.prim_parameter("out", tensor, fm.T.PrimParameterRole.OUTPUT),
    ), fm.T.sequential((dispatch,)), fm.T.return_((
        fm.T.return_binding(fm.T.value_ref("out", tensor), "out"),
    )))
    owner = "worker" if nested else "main"
    first = fm.T.prim_function_call("first", "kernel", shared_workspace_buffers=tuple(
        _buffer(f"first.stage{index}", owner, index * 32) for index in range(2)
    ))
    function = fm.T.execution_function(owner, (), (), fm.T.sequential((first,)))
    functions = [function]
    if nested:
        second = fm.T.prim_function_call("second", "kernel", shared_workspace_buffers=tuple(
            _buffer(f"second.stage{index}", owner, (0 if reused else 64) + index * 32)
            for index in range(2)
        ))
        functions[0] = replace(function, body=fm.T.sequential((first, second)))
        actuals = tuple(_buffer(f"invoke.{b.name}", "main", b.mem_span.absolute_start.fixed_value)
                        for call in functions[0].body.fields for b in call.shared_workspace_buffers)
        functions.append(fm.T.execution_function("main", (), (), fm.T.sequential((
            fm.T.prim_function_call("invoke", "worker", shared_workspace_buffers=actuals),
        ))))
    module = fm.IRModule("bufferized_tir", "scheduled_tir", (),
                         tuple(fm.Function(f.name, (), ()) for f in functions), "main",
                         prim_functions=(primitive,), execution_functions=tuple(functions))
    verify_execution_functions(module)
    return module


def _edit_entry(module, edit):
    entry = module.execution_function_map["main"]
    call = entry.body.fields[0]
    edited = replace(entry, body=fm.T.sequential((replace(call, shared_workspace_buffers=edit(call.shared_workspace_buffers)),)))
    return replace(module, execution_functions=tuple(edited if f.name == entry.name else f for f in module.execution_functions))


def _start(buffer, start):
    return replace(buffer, mem_span=replace(buffer.mem_span, buffer=buffer.mem_span.buffer.with_start(start)))


@pytest.mark.parametrize("edit", [
    lambda values: values[:1],
    lambda values: (replace(values[0], dimensions=(8,)), values[1]),
    lambda values: (replace(values[0], strides=(0,)), values[1]),
    lambda values: (replace(values[0], mem_span=replace(values[0].mem_span,
        buffer=replace(values[0].mem_span.buffer, alignment=8))), values[1]),
    lambda values: (_start(values[0], 8), values[1]),
    lambda values: (values[0], _start(values[1], 16)),
], ids=("arity", "shape", "stride", "alignment", "misaligned_start", "overlap"))
def test_execution_shared_abi_rejects_illegal_leaf_edits(edit):
    edited = _edit_entry(_module(), edit)
    with pytest.raises(IRVerificationError, match="Shared"):
        verify_execution_functions(edited)


def test_nested_shared_binding_cannot_introduce_new_interference():
    module = _module(nested=True)
    edited = _edit_entry(module, lambda values: (*values[:2], _start(values[2], 0), _start(values[3], 32)))
    with pytest.raises(IRVerificationError, match="Shared.*overlap"):
        verify_execution_functions(edited)


@pytest.mark.parametrize("nested,reused", [(False, False), (True, False), (True, True)])
def test_shared_abi_allows_aligned_rebase_and_existing_temporal_reuse(nested, reused):
    module = _module(nested=nested, reused=reused)
    edited = _edit_entry(module, lambda values: tuple(_start(b, b.mem_span.absolute_start.fixed_value + 128) for b in values))
    verify_execution_functions(edited)


def test_shared_abi_allows_removing_interference():
    module = _module(nested=True, reused=True)
    edited = _edit_entry(module, lambda values: (*values[:2], _start(values[2], 64), _start(values[3], 96)))
    verify_execution_functions(edited)


@pytest.mark.parametrize("edit", [
    lambda b: _start(b, 8),
    lambda b: replace(b, mem_span=fm.MemSpan(replace(b.mem_span.buffer, size=48), 8, 32)),
    lambda b: replace(b, strides=(0,)),
], ids=("allocation_alignment", "view_alignment", "strides"))
def test_selected_shared_formal_must_preserve_descriptor_layout(edit):
    primitive = _module().prim_functions[0]
    fm.verify_prim_function(primitive)
    dispatch = fm.kernel_dispatch_of(primitive)
    changed = replace(dispatch, shared_workspace_buffers=(edit(dispatch.shared_workspace_buffers[0]),
                                                         dispatch.shared_workspace_buffers[1]))
    primitive = replace(primitive, body=fm.T.sequential((changed,)))
    with pytest.raises(IRVerificationError, match="shared workspace"):
        fm.verify_prim_function(primitive)


def test_caller_cannot_exceed_shared_memory_space_capacity():
    from triton.flagmega.ir.tir.shared_call_abi import verify_shared_call_abi

    module = _module()
    call = module.execution_function_map["main"].body.fields[0]
    formals = fm.kernel_dispatch_of(module.prim_functions[0]).shared_workspace_buffers
    moved = replace(call, shared_workspace_buffers=tuple(_start(b, b.mem_span.absolute_start.fixed_value + 64)
                                                         for b in call.shared_workspace_buffers))
    with pytest.raises(IRVerificationError, match="Shared.*capacity"):
        verify_shared_call_abi(moved, formals, stage=module.stage, leaf=True, maximum_bytes=64)
