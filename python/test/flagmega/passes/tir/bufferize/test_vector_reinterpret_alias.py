# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.ir.bufferization import AliasKind


def test_vector_reinterpret_view_preserves_the_exact_memspan():
    packed_type = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2, 4))
    unpacked_type = fm.tensor_type("bfloat16", (2, 32))
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    value = builder.var("value", packed_type, id="value")
    view = builder.call(
        "tir.buffer_view",
        (value,),
        unpacked_type,
        id="view",
        attrs={"alias_kind": "vector_reinterpret"},
    )
    builder.function("main", (value,), (view,))
    module = builder.build(entry="main")

    plan = fm.make_buffer_plan(module)
    source = plan.buffer_map[dict(plan.entry_inputs)["value"][0]]
    result = plan.buffer_map[dict(plan.entry_outputs)["view"][0]]

    assert source.mem_span.must_alias(result.mem_span)
    assert source.nbytes == result.nbytes
    assert source.dtype == fm.vector_type("bfloat16", (8,))
    assert result.dtype == fm.DType.BFLOAT16
    assert result.alias.kind is AliasKind.VIEW


def test_reshape_of_coarser_shard_backing_preserves_bytes():
    mesh = fm.Placement((2, 2), 'yx', 'bb')
    tensor = fm.tensor_type(fm.vector_type('bfloat16', 8), (1, 4, 1, 8))
    parent = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.split_contiguous(
        (0, )), fm.SBP.broadcast(), fm.SBP.broadcast()), mesh)
    narrow = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.split_contiguous(
        (0, )), fm.SBP.broadcast(), fm.SBP.split_contiguous((1, ))), mesh)
    output = fm.DistributedType(fm.tensor_type(fm.vector_type('bfloat16', 8), (1, 4, 8)),
                                (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, )), fm.SBP.split_contiguous(
                                    (1, ))), mesh)
    builder = fm.IRBuilder(dialect='semantic_tir', stage='packaged_tir')
    value = builder.var('value', parent, id='value')
    produced = builder.call('math.silu', (value, ), parent, id='produced')
    view = builder.call('distributed.sharded_view', (produced, ), narrow, id='view', attrs={'new_type': narrow})
    reshaped = builder.call('tir.buffer_view', (view, ), output, id='reshaped', attrs={'alias_kind': 'reshape'})
    # An internal consumer makes the coarser local backing observable before
    # the program's canonical output convention is applied.
    result = builder.call('math.silu', (reshaped, ), output, id='result')
    builder.function('main', (value, ), (result, ))
    plan = fm.make_buffer_plan(builder.build(entry='main'))
    source, actual = plan.buffer_map['view'], plan.buffer_map['reshaped']
    assert source.mem_span.must_alias(actual.mem_span)
    assert source.nbytes == actual.nbytes == 4 * 8 * 16
