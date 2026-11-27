# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A formal buffer publishes its planned alignment, not one implied by dtype.

``PlanStorageAlignments`` records what each legal implementation of a kernel
requires and writes it into ``PrimParameter.alignment_bytes``; ``BufferPlanner``
allocates against exactly those recorded requirements.  A formal buffer that
instead derived a floor from its element type would demand an alignment nobody
planned for, which no caller storage can satisfy whenever the actual buffer is
a legitimately weaker-aligned view -- a vector reinterpretation of an offset
subspan being the case that reaches the 45-layer GLM-5.3-Flash model.
"""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.tir import bind_prim_function_buffers


def _unary_prim_function(value_type, *, alignment_bytes):
    dispatch = fm.T.kernel_dispatch(
        semantic_op="math.silu",
        semantic_candidate="tir.unary.scalar",
        arguments=("input",),
        outputs=("result",),
        semantic_attrs={},
        reads=("input",),
        writes=("result",),
    )
    parameters = (
        fm.T.prim_parameter(
            "input", value_type, fm.T.PrimParameterRole.INPUT,
            alignment_bytes=alignment_bytes,
        ),
        fm.T.prim_parameter(
            "result", value_type, fm.T.PrimParameterRole.OUTPUT,
            alignment_bytes=alignment_bytes,
        ),
    )
    return fm.T.prim_function(
        "unary", "triton", parameters, fm.T.sequential((dispatch,)),
        fm.T.return_((
            fm.T.return_binding(fm.T.value_ref("result", value_type), "result"),
        )),
    )


def _vector_view_of_offset_subspan(*, alignment_bytes, offset=1):
    """Feed a kernel the vector reinterpretation of a 2-byte-aligned subspan."""

    parent_type = fm.tensor_type("bfloat16", (2, 33))
    slice_type = fm.tensor_type("bfloat16", (1, 32))
    packed_type = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 4))
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    source = builder.var("source", parent_type, id="source")
    span = builder.call(
        "tir.buffer_subspan", (source,), slice_type, id="span",
        attrs={"offsets": (0, offset), "shape": (1, 32)},
    )
    view = builder.call(
        "tir.buffer_view", (span,), packed_type, id="view",
        attrs={"alias_kind": "vector_reinterpret"},
    )
    builder.prim_function(_unary_prim_function(packed_type, alignment_bytes=alignment_bytes))
    result = builder.call(
        "tir.call", (view,), packed_type, id="call", attrs={"callee": "unary"}
    )
    builder.function("main", (source,), (result,))
    return fm.verify_module(builder.build(entry="main"))


def _bufferize(module):
    plan = fm.make_buffer_plan(module)
    bound = bind_prim_function_buffers(module, plan=plan)
    bufferized = fm.verify_module(replace(
        bound,
        stage="bufferized_tir",
        metadata={**bound.metadata, "buffer_plan": plan.to_data()},
    ))
    return bufferized, plan, bound


@pytest.mark.parametrize("alignment_bytes", [1, None])
def test_vector_view_of_an_offset_subspan_satisfies_its_formal_abi(alignment_bytes):
    module = _vector_view_of_offset_subspan(alignment_bytes=alignment_bytes)
    bufferized, plan, bound = _bufferize(module)

    view = plan.buffer_map[dict(plan.function_map["main"].values)["view"][0]]
    # A 16-byte element type does not make its storage 16-byte aligned.
    assert view.dtype == fm.vector_type("bfloat16", (8,))
    assert view.dtype.itemsize == 16
    assert view.alignment == 2

    formal = bound.kernel_callable_map["unary"].parameter_map["input"].buffers[0]
    assert formal.mem_span.buffer.alignment == (alignment_bytes or 1)
    fm.verify_buffer_plan(bufferized)


def test_a_declared_contract_reaches_the_formal_buffer():
    module = _vector_view_of_offset_subspan(alignment_bytes=16, offset=0)
    _, _, bound = _bufferize(module)
    formal = bound.kernel_callable_map["unary"].parameter_map["input"].buffers[0]
    assert formal.mem_span.buffer.alignment == 16


def test_an_unsatisfiable_declared_contract_is_still_rejected():
    # A declared contract is a real requirement, so planning must still refuse
    # storage that cannot honour it.  Only the invented dtype floor is gone.
    module = _vector_view_of_offset_subspan(alignment_bytes=16, offset=1)
    with pytest.raises(IRVerificationError, match="does not satisfy consumer alignment"):
        fm.make_buffer_plan(module)
