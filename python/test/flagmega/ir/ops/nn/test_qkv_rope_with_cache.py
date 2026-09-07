# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn._paged_attention_state import (
    PagedAttentionStateConfig,
)
from triton.flagmega.ir.ops.tensors.pack import Pack


def test_scalar_inputs_infer_cache_native_vector_query_type():
    config = PagedAttentionStateConfig(
        1, 1, 16, block_size=256, num_blocks=1, lanes=8
    )
    qkv = fm.TupleType(
        (
            fm.tensor_type("bfloat16", (8, 2, 16)),
            fm.tensor_type("bfloat16", (8, 1, 16)),
            fm.tensor_type("bfloat16", (8, 1, 16)),
        )
    )

    result = _infer(qkv, config.ref_type)

    assert result == fm.TupleType(
        (
            fm.tensor_type(
                fm.VectorType(fm.DType.BFLOAT16, (8,)), (2, 2, 8)
            ),
            config.ref_type,
        )
    )


def test_cache_native_vector_inputs_preserve_the_same_boundary_type():
    config = PagedAttentionStateConfig(
        1, 1, 16, block_size=256, num_blocks=1, lanes=8
    )
    scalar_qkv = (
        fm.tensor_type("bfloat16", (8, 2, 16)),
        fm.tensor_type("bfloat16", (8, 1, 16)),
        fm.tensor_type("bfloat16", (8, 1, 16)),
    )
    packed_qkv = fm.TupleType(
        tuple(_pack(value, (8,), (2,)) for value in scalar_qkv)
    )
    parameter = fm.tensor_type("bfloat16", (16,))
    packed_parameter = _pack(parameter, (8,), (0,))
    trig = fm.tensor_type("float32", (8, 1, 16))
    packed_trig = _pack(trig, (2, 8), (2, 2))

    result = _infer(
        packed_qkv,
        config.ref_type,
        parameter_type=packed_parameter,
        trig_type=packed_trig,
    )

    assert result.fields[0] == fm.tensor_type(
        fm.VectorType(fm.DType.BFLOAT16, (8,)), (2, 2, 8)
    )
    assert result.fields[1] == config.ref_type


def test_fused_op_requires_a_configured_paged_attention_reference():
    qkv = fm.TupleType(
        tuple(fm.tensor_type("bfloat16", (1, 1, 16)) for _ in range(3))
    )

    with pytest.raises(IRSchemaError, match="configured paged-attention cache"):
        _infer(qkv, fm.RefType("paged_attention_kv_cache"))


def _infer(
    qkv_type,
    state_type,
    *,
    parameter_type=None,
    trig_type=None,
):
    parameter_type = parameter_type or fm.tensor_type("bfloat16", (16,))
    trig_type = trig_type or fm.tensor_type("float32", (8, 1, 16))
    inputs = (
        _typed("qkv", qkv_type),
        _typed("q_scale", parameter_type),
        _typed("k_scale", parameter_type),
        _typed("q_bias", parameter_type),
        _typed("k_bias", parameter_type),
        _typed("cos", trig_type),
        _typed("sin", trig_type),
        _typed("state", state_type),
        _typed("layer_id", fm.tensor_type("int32", ())),
        _typed("advance", fm.tensor_type("bool", ())),
    )
    return fm.get_definition("nn.qkv_rope_with_cache").infer_type(
        inputs,
        {
            "q_axis": 2,
            "q_epsilon": 1e-6,
            "q_use_mean": False,
            "k_axis": 2,
            "k_epsilon": 1e-6,
            "k_use_mean": False,
            "qkv_layout": ("seq", "head", "dim"),
            "attention_layout": ("head", "dim", "seq"),
        },
    )


def _pack(value_type, lanes, axes):
    return Pack.infer_type(
        (_typed("input", value_type),), {"lanes": lanes, "axes": axes}
    )


def _typed(name: str, value_type: fm.IRType) -> fm.Node:
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})
