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

import os


def get_corex_sme(args, specialization):
    import torch

    if getattr(torch, "corex", False) is not True or os.getenv("TRITON_DISABLE_SME", "0") == "1":
        return 0

    use_sme = 0
    operand_index = 0
    sme_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.int8)
    for arg, spec in zip(args, specialization):
        # Constexpr arguments are not runtime function operands and therefore do
        # not consume a bit in the use_sme operand mask.
        if spec[0] == "constexpr":
            continue

        if torch.is_tensor(arg) and arg.dtype in sme_dtypes and arg.dim() >= 2:
            dim_m = arg.shape[-2]
            dim_k = arg.shape[-1]
            if dim_m != 1 and dim_k != 1:
                sme_dim = 64 // arg.element_size()
                is_row_major = arg.is_contiguous() and dim_k % sme_dim == 0
                is_col_major = not arg.is_contiguous() and dim_m % sme_dim == 0
                can_use_sme = is_col_major if arg.dtype == torch.int8 else is_row_major or is_col_major
                if can_use_sme:
                    use_sme |= 1 << operand_index
        operand_index += 1
    return use_sme


def jit_specialize_options(args, specialization, options):
    del options
    return {"use_sme": get_corex_sme(args, specialization)}
