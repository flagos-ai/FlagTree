# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest
import torch
from triton_kernels.tensor_details.layout import BlackwellMXScaleLayout, BlackwellActMXScaleLayout
from triton_kernels.tensor import make_ragged_tensor_metadata

# ------------------------------------------------------------
# Torch tests
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (3, 4096, 1024),
        (10, 254, 60),
        (1, 320, 160),
        (2, 16, 512),
        (3, 2, 36),
    ],
)
def test_mxfp4_scale_roundtrip(shape):
    x = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    layout = BlackwellMXScaleLayout()
    transformation = layout.make_transformation(x.shape, is_fp4=False)
    res = transformation.unswizzle_data(transformation.swizzle_data(x))
    assert (res == x).all()


@pytest.mark.parametrize("shape", [(2, 256, 192), (1, 128, 64)])
def test_act_scale_roundtrip_batched(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    layout = BlackwellActMXScaleLayout(ragged_metadata=None)
    transformation = layout.make_transformation(x.shape, is_fp4=False)
    res = transformation.unswizzle_data(transformation.swizzle_data(x))
    torch.testing.assert_close(res, x)


@pytest.mark.parametrize(
    "slice_sizes, m, k, align_m",
    [
        ([17, 0, 33, 5], 100, 94, 8),
        ([1, 2, 3, 4, 5], 50, 15, 16),
    ],
)
def test_act_scale_roundtrip_ragged(slice_sizes, m, k, align_m):
    slice_sizes = torch.tensor(slice_sizes, device="cuda", dtype=torch.int32)
    m = max(m, slice_sizes.sum().item())  # there can be padded tokens in the input
    ragged_metadata = make_ragged_tensor_metadata(slice_sizes, m)
    x = torch.randn((m, k), device="cuda", dtype=torch.float32)
    layout = BlackwellActMXScaleLayout(ragged_metadata=ragged_metadata)
    transformation = layout.make_transformation(x.shape, is_fp4=False)
    res = transformation.unswizzle_data(transformation.swizzle_data(x))

    x_useful_rows = x[ragged_metadata.slice_offs[:-1], :]
    res_useful_rows = res[ragged_metadata.slice_offs[:-1], :]
    torch.testing.assert_close(res_useful_rows, x_useful_rows)
