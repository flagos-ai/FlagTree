# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega.runtime.module import _typed_byte_view


def test_structured_vector_rdata_uses_trailing_physical_lanes():
    storage = torch.zeros(32, dtype=torch.uint8)
    view = _typed_byte_view(
        storage, 0, 32,
        {"kind": "vector", "elem_type": "float8_e4m3fn", "lanes": [2, 16]},
        (1, 1),
    )
    assert view.dtype == torch.float8_e4m3fn
    assert view.shape == (1, 1, 2, 16)
