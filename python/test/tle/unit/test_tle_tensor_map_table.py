# Copyright 2026 FlagOS Contributors

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from .._tensor_map_utils import make_tensor_map_table


@pytest.mark.skip_if_tle_unsupported("gpu.tensor_map_fenceproxy_acquire", "gpu.reinterpret_tensor_map")
def test_nvidia_advertises_both_tensor_map_primitives():
    from triton.backends.nvidia.tle_supported import TLE_SUPPORTED_PRIMITIVES
    from triton.experimental.tle import primitive_name

    for primitive in (tle.gpu.tensor_map_fenceproxy_acquire, tle.gpu.reinterpret_tensor_map):
        assert primitive_name(primitive) in TLE_SUPPORTED_PRIMITIVES


def test_descriptor_table_indexing_does_not_extend_tle_api():
    assert not hasattr(tle.gpu, "tensor_map_table_entry")
    assert not hasattr(tle.gpu.core, "tensor_map_table_entry")


@triton.jit
def _copy_selected_tensor_map(table, output, OWNER_ELEMENTS: tl.constexpr):
    owner = tl.program_id(0)
    shared = tle.gpu.alloc(
        [8, 8],
        dtype=tl.float16,
        scope=tle.gpu.smem,
    )
    # The uint8 table contains 128-byte CUDA tensor-map descriptors.
    entry = table + owner * 128
    tle.gpu.tensor_map_fenceproxy_acquire(entry)
    descriptor = tle.gpu.reinterpret_tensor_map(entry, shared)
    tle.gpu.copy(descriptor, shared, [8, 8], [0, 0])
    rows = tl.arange(0, 8)[:, None]
    columns = tl.arange(0, 8)[None, :]
    values = tl.load(tle.gpu.local_ptr(shared, (rows, columns), [8, 8]))
    tl.store(output + owner * OWNER_ELEMENTS + rows * 8 + columns, values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skip_if_tle_unsupported("gpu.reinterpret_tensor_map")
def test_tensor_map_table_selects_distinct_owner_descriptors():
    owners = 2
    source = torch.arange(owners * 64, device="cuda", dtype=torch.float16).reshape(owners, 8, 8)
    table = make_tensor_map_table(source.unbind(0))
    assert table.data_ptr() % 128 == 0
    output = torch.empty_like(source)

    _copy_selected_tensor_map[(owners, )](table, output, 64, num_warps=4)

    torch.testing.assert_close(output, source)
