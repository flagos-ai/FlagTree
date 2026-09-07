# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.descriptor_abi import device_descriptor_request
from triton.flagmega.errors import CodegenError


@pytest.mark.parametrize("shape,capacity,matrix,swizzle,box", [
    ((16, 1024), 2, True, 3, (16, 64)),
    ((4, 128), 2, True, 3, (4, 64)),
    ((1, 128), 2, True, 0, (1, 128)),
    ((8, 32), 2, True, 2, (8, 32)),
    ((8, 16), 2, True, 1, (8, 16)),
    ((512, 128), 2, False, 0, (256, 128)),
    ((1, 1, 64, 1, 128), 2, True, 3, (1, 1, 64, 1, 64)),
    ((64, 2, 2, 64), 4, True, 3, (64, 2, 2, 64)),
])
def test_hardware_box_matches_typed_shared_encoding(shape, capacity, matrix, swizzle, box):
    request = {
        "kind": "single", "parameter": "map", "block_shape": shape,
        "dtype": "bfloat16", "shape": (1000,) * len(shape),
        "offset_bytes": 4096, "source_shape_axes": ((),) * len(shape),
    }
    workspace = {"shape": (capacity, *shape), "matrix_compatible": matrix}
    encoded = device_descriptor_request(request, (workspace,))
    assert encoded == {**request, "storage": "device", "swizzle_mode": swizzle, "box_shape": box}


def test_owner_table_storage_does_not_change_its_coordinate_contract():
    request = {"kind": "table", "entries": ("owner_zero", "owner_one")}
    assert device_descriptor_request(request, ()) is request


@pytest.mark.parametrize("workspaces", [
    (),
    ({"shape": (2, 8, 64), "matrix_compatible": True},
     {"shape": (2, 8, 64), "matrix_compatible": False}),
])
def test_descriptor_requires_a_unique_matching_shared_encoding(workspaces):
    request = {"kind": "single", "parameter": "map", "block_shape": (8, 64), "dtype": "bfloat16"}
    with pytest.raises(CodegenError, match="one typed Shared encoding"):
        device_descriptor_request(request, workspaces)
