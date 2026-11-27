# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.physical_access import (
    emit_buffer_pointer,
    emit_scalar_immediate,
    emit_storage_pointer,
)
from triton.flagmega.errors import CodegenError


def _abi(**overrides) -> dict[str, object]:
    return {
        "storage": "workspace",
        "storage_kind": "compact_local",
        "scalar_dtype": "bfloat16",
        "scalar_itemsize": 2,
        "pool_byte_offset": 256,
        "component_stride_scalar_elements": 0,
        **overrides,
    }


def test_pooled_pointer_applies_typed_byte_offset():
    assert emit_buffer_pointer(_abi(), "workspace") == (
        "(workspace.to(tl.pointer_type(tl.bfloat16)) + 128)"
    )


def test_compact_per_owner_pointer_applies_component_after_pool_offset():
    assert emit_buffer_pointer(
        _abi(
            storage="rdata",
            storage_kind="compact_per_owner",
            component_stride_scalar_elements=64,
        ),
        "rdata",
        owner_index="owner",
    ) == (
        "((rdata.to(tl.pointer_type(tl.bfloat16)) + 128) + "
        "(owner) * 64)"
    )


def test_storage_pointer_does_not_preapply_callee_owner_offset():
    assert emit_storage_pointer(
        _abi(
            storage="rdata",
            storage_kind="compact_per_owner",
            component_stride_scalar_elements=64,
        ),
        "rdata",
    ) == "(rdata.to(tl.pointer_type(tl.bfloat16)) + 128)"


def test_external_pointer_is_not_recast():
    assert emit_buffer_pointer(
        _abi(
            storage="input",
            pool_byte_offset=0,
            storage_kind="compact_local",
        ),
        "hidden",
    ) == "hidden"


def test_invalid_pool_offset_and_scalar_pointer_fail_explicitly():
    with pytest.raises(CodegenError, match="not aligned"):
        emit_buffer_pointer(_abi(pool_byte_offset=1), "workspace")
    with pytest.raises(CodegenError, match="not buffer pointers"):
        emit_buffer_pointer(_abi(storage="scalar"), "layer")


def test_scalar_immediate_is_a_runtime_value_not_a_constexpr_argument():
    scalar = _abi(storage="scalar", scalar_dtype="int32")

    assert emit_scalar_immediate(scalar, "7") == "tl.full((), 7, tl.int32)"
