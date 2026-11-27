# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.math.packed_block_scaled_matmul import (
    PackedBlockScaledMatMul,
)


def _node(name, value_type):
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def test_packed_fp8_matmul_scales_physical_k_split_to_logical_units():
    placement = fm.Placement((8,), "x", "b")
    value = fm.DistributedType(
        fm.tensor_type("bfloat16", (1, 1024)),
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0,), 64)),
        placement,
    )
    weight = fm.DistributedType(
        fm.tensor_type(
            fm.VectorType(fm.DType.FLOAT8_E4M3FN, (2, 16)),
            (1024, 32),
        ),
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0,), 2)),
        placement,
    )
    scale = fm.DistributedType(
        fm.tensor_type("float32", (8, 8)),
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )

    result = PackedBlockScaledMatMul.infer_type(
        (
            _node("value", value),
            _node("weight", weight),
            _node("scale", scale),
        ),
        {
            "weight_block_n": 128,
            "weight_block_k": 128,
            "k_pack": 2,
            "k_vector": 16,
            "packed_layout": "n_major_k_packed",
        },
    )

    assert result.partial == fm.SBP.partial((0,))
