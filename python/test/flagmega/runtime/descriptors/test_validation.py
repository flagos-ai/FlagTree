# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.runtime import TensorDescriptorCache

from .helpers import single_spec, table_spec


def _materialize(spec, storage):
    return TensorDescriptorCache().materialize_many(
        "kernel", (spec,), {"rdata": storage}
    )


def test_descriptor_schema_is_exact_and_source_must_be_bound():
    torch = pytest.importorskip("torch")
    storage = torch.empty((3, 4), dtype=torch.float32)

    with pytest.raises(RuntimeContractError, match="unexpected=.*extra"):
        _materialize(single_spec(extra=True), storage)
    with pytest.raises(RuntimeContractError, match="unbound source"):
        TensorDescriptorCache().materialize_many(
            "kernel", (single_spec(),), {}
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"source_shape_axes": ((0,), (0,))}, "invalid/reused source axis"),
        ({"strides": (5, 1)}, "outer stride"),
        ({"strides": (4, 2)}, "last dimension"),
        ({"block_shape": (1, 3)}, "powers of two"),
        ({"offset_bytes": -1}, "invalid offset/padding"),
    ],
)
def test_invalid_physical_descriptor_contract_is_rejected(updates, message):
    torch = pytest.importorskip("torch")
    storage = torch.empty((3, 8), dtype=torch.float32)

    with pytest.raises(RuntimeContractError, match=message):
        _materialize(single_spec(**updates), storage)


def test_descriptor_view_cannot_escape_source_storage():
    torch = pytest.importorskip("torch")
    storage = torch.empty((3, 4), dtype=torch.float32)

    with pytest.raises(RuntimeContractError, match="exceeds storage size"):
        _materialize(
            single_spec(shape=(4, 4), source_shape_axes=((), ())),
            storage,
        )


def test_nan_padding_rejects_non_floating_storage_view():
    torch = pytest.importorskip("torch")
    storage = torch.empty((32,), dtype=torch.uint8)

    with pytest.raises(RuntimeContractError, match="cannot NaN-pad"):
        _materialize(single_spec(
            dtype="uint8",
            shape=(2, 16),
            strides=(16, 1),
            block_shape=(1, 16),
            source_shape_axes=((), ()),
            padding="nan",
        ), storage)


def test_table_validates_entries_before_requiring_cuda_storage():
    torch = pytest.importorskip("torch")
    storage = torch.empty((32,), dtype=torch.uint8)
    invalid_entry = dict(table_spec()["entries"][0])
    invalid_entry["unexpected"] = True

    with pytest.raises(RuntimeContractError, match="unexpected=.*unexpected"):
        _materialize(table_spec(entries=(invalid_entry,)), storage)
    with pytest.raises(RuntimeContractError, match="requires CUDA storage"):
        _materialize(table_spec(), storage)
