# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.runtime import TensorDescriptorCache

from .helpers import single_spec


def test_single_descriptor_resolves_dynamic_shape_and_reuses_exact_storage():
    torch = pytest.importorskip("torch")
    storage = torch.empty((3, 4), dtype=torch.float32)
    cache = TensorDescriptorCache()

    first = cache.materialize_many(
        "kernel", (single_spec(),), {"rdata": storage}
    )[0]
    second = cache.materialize_many(
        "kernel", (single_spec(),), {"rdata": storage}
    )[0]

    assert first is second
    assert first.base.data_ptr() == storage.data_ptr()
    assert first.shape == [3, 4]
    assert first.strides == [4, 1]
    assert first.block_shape == [1, 4]


def test_storage_or_resolved_shape_change_rebuilds_descriptor():
    torch = pytest.importorskip("torch")
    cache = TensorDescriptorCache()
    first_storage = torch.empty((3, 4), dtype=torch.float32)
    second_storage = torch.empty((5, 4), dtype=torch.float32)

    first = cache.materialize_many(
        "kernel", (single_spec(),), {"rdata": first_storage}
    )[0]
    second = cache.materialize_many(
        "kernel", (single_spec(),), {"rdata": second_storage}
    )[0]

    assert first is not second
    assert second.shape == [5, 4]
    assert second.base.data_ptr() == second_storage.data_ptr()


def test_clear_invalidates_persistent_descriptor_binding():
    torch = pytest.importorskip("torch")
    storage = torch.empty((3, 4), dtype=torch.float32)
    cache = TensorDescriptorCache()
    first = cache.materialize_many(
        "kernel", (single_spec(),), {"rdata": storage}
    )[0]

    cache.clear()
    second = cache.materialize_many(
        "kernel", (single_spec(),), {"rdata": storage}
    )[0]

    assert first is not second
