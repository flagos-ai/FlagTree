# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import PagedAttentionStateConfig
from triton.flagmega.ir.ops.nn.rotary_embedding import RotaryEmbedding
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


@pytest.mark.parametrize("lanes", [(), (8, ), (2, 4), (4, 2)])
def test_vector_result_types_roundtrip_python_ir(tmp_path, lanes):
    module = primitive_module(RotaryEmbedding, (fm.tensor_type(fm.vector_type("bfloat16", (8, )),
                                                               (3, 4)), PagedAttentionStateConfig(1, 2, 32).ref_type),
                              head_dim=32, theta=10000., output_lanes=lanes)
    fm.emit_module(module, tmp_path / "rotary.py")
    restored = fm.load_module(tmp_path / "rotary.py")
    assert restored.semantic_hash == module.semantic_hash
    dtype = fm.vector_type("float32", lanes) if lanes else fm.DType.FLOAT32
    assert restored.node_map["output"].type.fields[0].dtype == dtype


@pytest.mark.parametrize("lanes", [(3, ), (0, ), (-1, ), (True, ), (1.5, )])
def test_invalid_vector_width_rejected(lanes):
    with pytest.raises(IRSchemaError, match="output_lanes"):
        RotaryEmbedding.normalize_attrs({"head_dim": 32, "theta": 10000., "output_lanes": lanes})
