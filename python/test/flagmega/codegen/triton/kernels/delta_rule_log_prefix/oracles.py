# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Independent target elementary log2, followed by a CPU scan topology.

Approximate log2 is not required to be exact even on powers of two. Measure
that primitive separately so the scan is still checked bit for bit, without
using the candidate's scan or CPU log2 as a target-instruction oracle.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _elementary_log(Input, Output, Count: tl.constexpr, Epsilon: tl.constexpr, Fast: tl.constexpr):
    index = tl.program_id(0) * 256 + tl.arange(0, 256)
    alpha = tl.load(Input + index, index < Count, other=1.)
    value = libdevice.fast_log2f(alpha + Epsilon) if Fast else tl.log2(alpha + Epsilon)
    tl.store(Output + index, value, index < Count)


def scan_oracle(alpha, block, group, mode, *, torch, epsilon=1e-10):
    source = alpha.cuda()
    logs = torch.empty_like(source)
    _elementary_log[(triton.cdiv(source.numel(), 256), )](source, logs, source.numel(), epsilon, mode == "fast")
    logs = logs.cpu()
    torch.testing.assert_close(logs, torch.log2(alpha + epsilon), rtol=1e-6, atol=1e-6)
    tokens, heads = alpha.shape
    result = torch.zeros(((tokens + block - 1) // block, heads, block))
    for chunk in range(result.shape[0]):
        count = min(block, tokens - chunk * block)
        result[chunk, :, :count] = logs[chunk * block:chunk * block + count].T
        for start in range(0, block, group):
            values = result[chunk, :, start:start + group]
            distance = 1
            while distance < group:
                previous = values.clone()
                values[:, distance:] = previous[:, distance:] + previous[:, :-distance]
                distance *= 2
            if start:
                values += result[chunk, :, start - 1:start]
    return result
