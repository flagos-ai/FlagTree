# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.microkernels.attention_primitives import _validate_semantic_contract
from triton.flagmega.errors import CodegenError


@pytest.mark.parametrize("op", ["nn.rope", "ntt.vectorized_rope"])
@pytest.mark.parametrize("attrs", [{}, {"rotary_dim": None}, {"rotary_dim": 64}])
def test_tir_rope_accepts_the_ir_attribute_contract(op, attrs):
    _validate_semantic_contract(op, attrs)


@pytest.mark.parametrize("op", ["nn.rope", "ntt.vectorized_rope"])
@pytest.mark.parametrize("attrs", [{"rotary_dim": 0}, {"rotary_dim": 3}, {"rotary_dim": True}, {"unknown": 64}])
def test_tir_rope_rejects_invalid_or_unrecognized_attributes(op, attrs):
    with pytest.raises(CodegenError):
        _validate_semantic_contract(op, attrs)
