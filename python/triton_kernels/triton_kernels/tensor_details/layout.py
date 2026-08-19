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

from .layout_details.base import Layout
from .layout_details.blackwell_scale import BlackwellMXScaleLayout
from .layout_details.blackwell_scale import BlackwellActMXScaleLayout
from .layout_details.blackwell_value import BlackwellMXValueLayout
from .layout_details.hopper_scale import HopperMXScaleLayout
from .layout_details.hopper_value import HopperMXValueLayout
from .layout_details.cdna4_scale import CDNA4MXScaleLayout
from .layout_details.strided import StridedLayout
from ..target_info import cuda_capability_geq, is_hip_cdna4

__all__ = [
    "Layout",
    "BlackwellMXValueLayout",
    "BlackwellMXScaleLayout",
    "HopperMXScaleLayout",
    "HopperMXValueLayout",
    "CDNA4MXScaleLayout",
    "StridedLayout",
    "BlackwellActMXScaleLayout",
]


def make_default_matmul_mxfp4_w_layout(mx_axis: int):
    if cuda_capability_geq(10):
        return BlackwellMXValueLayout()
    elif cuda_capability_geq(9):
        return HopperMXValueLayout(mx_axis=mx_axis, mma_version=3)
    else:
        return StridedLayout(-2)


def make_default_matmul_mxfp4_w_scale_layout(mx_axis: int, num_warps: int = 8):
    if is_hip_cdna4():
        return CDNA4MXScaleLayout()
    else:
        if cuda_capability_geq(10):
            return BlackwellMXScaleLayout()
        elif cuda_capability_geq(9):
            return HopperMXScaleLayout(mx_axis=mx_axis, num_warps=num_warps)

    return StridedLayout(-2)


def make_default_matmul_mxfp8_act_scale_layout(ragged_metadata):
    if cuda_capability_geq(10):
        return BlackwellActMXScaleLayout(ragged_metadata)
    return StridedLayout(-2)
