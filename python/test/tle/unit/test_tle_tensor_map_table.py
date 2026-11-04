# Copyright 2026 FlagOS Contributors

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from triton.flagmega.runtime.nvidia_tensor_map import encode_tma_descriptor


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
@pytest.mark.require_tle("gpu.reinterpret_tensor_map")
def test_tensor_map_table_selects_distinct_owner_descriptors():
    owners = 2
    source = torch.arange(
        owners * 64, device="cuda", dtype=torch.float16
    ).reshape(owners, 8, 8)
    payload = bytearray()
    for owner in range(owners):
        payload.extend(
            encode_tma_descriptor(
                source[owner].data_ptr(),
                0,
                source.element_size(),
                6,
                (8, 8),
                (8, 8),
                (8, 1),
                0,
            )
        )
    table = torch.frombuffer(payload, dtype=torch.uint8).to(device="cuda")
    assert table.data_ptr() % 128 == 0
    output = torch.empty_like(source)

    _copy_selected_tensor_map[(owners,)](table, output, 64, num_warps=4)

    torch.testing.assert_close(output, source)
