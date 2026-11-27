# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.runtime import TensorDescriptorCache
from .helpers import single_spec


def device_spec(**updates):
    return single_spec(storage="device", swizzle_mode=0, box_shape=(1, 4), **updates)


def test_device_handle_preserves_single_descriptor_schema():
    assert TensorDescriptorCache.validate_specs((device_spec(),)) == ("weight_descriptor",)


@pytest.mark.parametrize("update", [
    {"storage": "unknown"}, {"swizzle_mode": 4}, {"box_shape": (1, 8)},
])
def test_device_descriptor_rejects_inconsistent_storage_encoding(update):
    spec = device_spec()
    spec.update(update)
    with pytest.raises(RuntimeContractError):
        TensorDescriptorCache.validate_specs((spec,))


def test_device_handle_cache_identity_tracks_source_shape_and_storage():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    storage = torch.empty((3, 4), dtype=torch.float32, device="cuda")
    cache = TensorDescriptorCache()
    spec = device_spec()
    first, = cache.materialize_many("kernel", (spec,), {"rdata": storage})
    again, = cache.materialize_many("kernel", (spec,), {"rdata": storage})
    assert first is again
    assert first.dtype == torch.uint8 and first.numel() == 128
    assert first.data_ptr() % 128 == 0
    reshaped, = cache.materialize_many("kernel", (spec,), {"rdata": storage[:2]})
    assert reshaped is not first
    replaced, = cache.materialize_many("kernel", (spec,), {"rdata": storage.clone()})
    assert replaced is not reshaped
    cache.clear()
    fresh, = cache.materialize_many("kernel", (spec,), {"rdata": storage})
    assert fresh is not first
