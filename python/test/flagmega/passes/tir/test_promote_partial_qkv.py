# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Closed QKV sums retain FP32 partials and one BF16 materialization boundary."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir.promote_partial_matmul import promote_partial_matmul


def qkv_module(*, escaped=False, shared=False):
    mesh = fm.Placement((2, 2), "yx", "bb")
    broad = fm.SBP.broadcast()
    source = fm.DistributedType(fm.tensor_type("bfloat16", (1, 32)), (broad, fm.SBP.split_contiguous((0, ), 16)), mesh)
    weights = tuple(
        fm.DistributedType(fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (2, n)), (
            fm.SBP.split_contiguous((0, ), 1), fm.SBP.split_block_cyclic((1, ), 8)), mesh) for n in (16, 8, 8))

    class Graph(fm.Module):

        def forward(self):
            x = self.input("x", source)
            w = tuple(self.input(name, typ) for name, typ in zip(("q", "k", "v"), weights))
            none = fm.F.builtin.none()
            qkv = fm.F.ntt.packed_qkv_parallel_linear(x, *w, *([none] * 9), num_heads=2, num_kv_heads=1,
                                                      output_data_type="bfloat16", name="qkv")
            materialized = fm.TupleType(tuple(replace(t, partial=None) for t in qkv.type.fields))
            combined = fm.F.distributed.boxing(qkv, materialized, name="combine")
            outputs = [combined]
            if shared:
                outputs.append(fm.F.distributed.boxing(qkv, materialized, name="second"))
            if escaped:
                outputs.append(qkv)
            self.function("main", (x, *w), outputs)

    return Graph(dialect="high_level", stage="distributed", entry="main").build()


def test_private_qkv_tuple_promotes_all_partial_fields_and_preserves_result_abi(tmp_path):
    module = qkv_module(shared=True)
    result = promote_partial_matmul(module)
    assert [t.tensor.dtype.elem_type for t in result.node_map["qkv"].type.fields] == [fm.DType.FLOAT32] * 3
    assert result.function_map["main"] == module.function_map["main"]
    for name in ("combine", "second"):
        assert result.node_map[name].type == module.node_map[name].type
        assert result.node_map[name].op == "ntt.packed_qkv_parallel_linear_combine"
    assert [t.partial
            for t in result.node_map["qkv"].type.fields] == [t.partial for t in module.node_map["qkv"].type.fields]
    assert promote_partial_matmul(result) is result
    assert fm.load_module(fm.emit_module(result, tmp_path / "wide.py")).semantic_hash == result.semantic_hash


def test_exported_partial_tuple_keeps_its_declared_function_abi():
    module = qkv_module(escaped=True)
    assert promote_partial_matmul(module) is module


def test_combine_rounds_logical_fp32_result_before_any_consumer():
    torch = pytest.importorskip("torch")
    from triton.flagmega.ir.ops.ntt.packed_qkv_parallel_linear_combine import PackedQKVParallelLinearCombine
    from triton.flagmega.rules.neutral._utility import make_node
    module = qkv_module()
    qkv = module.node_map["qkv"]
    wide = make_node(qkv.op, qkv.id, tuple(module.node_map[i] for i in qkv.inputs),
                     {**qkv.attrs, "output_data_type": "float32"}, {})
    combined = make_node("ntt.packed_qkv_parallel_linear_combine", "combined", (wide, ),
                         {"output_type": module.node_map["combine"].type}, {})
    # Evaluator partials represent the complete logical sum. A value halfway
    # below the next BF16 step must round here, not remain widened FP32.
    values = tuple(torch.full((1, n, 8), 1.0 + 2.0**-9) for n in (16, 8, 8))
    context = SimpleNamespace(torch_dtype=lambda dtype: getattr(torch, dtype.elem_type.value))
    actual = PackedQKVParallelLinearCombine.evaluate(combined, (values, ), context)
    assert all(v.dtype == torch.bfloat16 and torch.equal(v, torch.ones_like(v)) for v in actual)


def test_unfused_shared_materializations_lower_with_rounding_after_sum():
    from triton.flagmega.passes.tir.fuse_distributed_ops import fuse_distributed_ops
    module = qkv_module(shared=True)
    result = fuse_distributed_ops(module)
    assert result.function_map['main'] == module.function_map['main']
    assert not any(n.op == 'ntt.packed_qkv_parallel_linear_combine' for n in result.nodes)
    for name in ('combine', 'second'):
        output = result.node_map[name]
        assert output.type == module.node_map[name].type
        for value in output.inputs:
            cast = result.node_map[value]
            assert cast.op == 'ntt.vectorized_cast'
            assert cast.attrs['rounding_contract'] == 'preserve'
            field = result.node_map[cast.inputs[0]]
            boxing = result.node_map[field.inputs[0]]
            assert boxing.op == 'distributed.boxing'
            assert all(t.partial is None and t.tensor.dtype.elem_type == fm.DType.FLOAT32 for t in boxing.type.fields)
