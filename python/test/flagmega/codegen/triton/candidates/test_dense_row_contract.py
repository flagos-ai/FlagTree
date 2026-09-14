# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates.dense_matmul import _supports_local_rows
from triton.flagmega.targets.nvidia.implementations import sm90_triton_implementation_model


@pytest.mark.parametrize("rows", [0, 1, 3])
def test_dense_row_loop_is_an_explicit_implementation_contract(rows):
    tensor = fm.tensor_type("bfloat16", (rows, 2048))
    implementations = [impl for impl in sm90_triton_implementation_model().implementations
                       if impl.family == "dense_matmul"]
    for impl in implementations:
        assert _supports_local_rows(impl, tensor) == (rows == 1 or impl.contract.get("supports_local_row_loop", False))
        if impl.contract.get("supports_local_row_loop"):
            assert impl.transfer_pipeline is None
            assert impl.contract["epilogue"] == "none"


def test_dense_single_row_contract_uses_capacity_not_global_token_count():
    tensor = fm.tensor_type("bfloat16", (4, 2048))
    value = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast()),
                               fm.Placement((4,), "x", "b"))
    impl = next(impl for impl in sm90_triton_implementation_model().implementations
                if impl.id == "tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv")
    assert _supports_local_rows(impl, value)
    assert not _supports_local_rows(impl, tensor)
