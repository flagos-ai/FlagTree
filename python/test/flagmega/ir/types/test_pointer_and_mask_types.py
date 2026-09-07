# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir.types import data_type_from_data, data_type_to_data


def test_pointer_type_is_a_64_bit_scalar_abi_handle():
    pointer = fm.PointerType(fm.vector_type("bfloat16", (8,)))
    value_type = fm.tensor_type(pointer, ())

    assert pointer.itemsize == 8
    assert pointer.value == "*bfloat16<8>"
    assert data_type_from_data(data_type_to_data(pointer)) == pointer
    assert fm.is_pointer().match_leaf(value_type)


def test_pointer_tensor_rejects_non_scalar_shape():
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value = builder.var(
        "pointer",
        fm.tensor_type(fm.PointerType("bfloat16"), (4,)),
        id="pointer",
    )
    builder.function("main", (value,), (value,))

    with pytest.raises(IRVerificationError, match="Pointer tensors must be scalar"):
        fm.verify_module(builder.build(entry="main"))


@pytest.mark.parametrize("style", [fm.MaskVectorStyle.FAT, fm.MaskVectorStyle.SLIM])
def test_mask_vector_has_explicit_physical_bytes_and_python_round_trip(tmp_path, style):
    mask = fm.MaskVectorType(style, element_bits=32, lanes=8)
    value_type = fm.tensor_type(mask, (2,))
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    value = builder.var("mask", value_type, id="mask")
    builder.function("main", (value,), (value,))
    module = fm.verify_module(builder.build(entry="main"))

    assert mask.itemsize == 32
    assert fm.is_mask_vector().match_leaf(value_type)
    loaded = fm.load_module(fm.emit_module(module, tmp_path / f"{style.value}.py"))
    assert loaded.semantic_hash == module.semantic_hash


def test_mask_vector_rejects_bit_packed_implicit_width():
    with pytest.raises(IRSchemaError, match="byte-sized"):
        fm.MaskVectorType(fm.MaskVectorStyle.SLIM, element_bits=1, lanes=32)
