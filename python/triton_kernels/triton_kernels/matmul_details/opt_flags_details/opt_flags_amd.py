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

import torch
import triton
from triton_kernels.target_info import get_cdna_version, get_rdna_version


def compute_block_nk(n, block_m, grid_m, num_xcds, lhs_dtype, rhs_dtype, precision_config):
    lhs_width = lhs_dtype.bitwidth / 8
    rhs_width = rhs_dtype.bitwidth / 8

    # block_n:
    n_cu = torch.cuda.get_device_properties(0).multi_processor_count
    if n is not None:
        if 0 < n <= 128 and (n & (n - 1)) == 0:
            block_n = n
        else:
            max_n = 64 if get_cdna_version() == 4 else 256
            block_n = max(32, min(max_n, triton.next_power_of_2(grid_m * n * num_xcds // n_cu)))
    elif block_m > 64:
        block_n = 256
    else:
        block_n = 128

    if get_rdna_version() in (3, 4) and block_m == 64:
        block_n = 256

    # block_k needs to match the cacheline size (128B)
    block_k = int(128 // min(lhs_width, rhs_width))

    # TODO: block_k = 128 seems to work better for now.
    #       perhaps due to increased number of k loops to pipeline
    if precision_config.b_mx_scale is not None:
        if get_cdna_version() != 4:
            block_k = 128

        if get_rdna_version() in (3, 4) and block_m == 64:
            block_k = 64
    return block_n, block_k
