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

import triton.language.core as tl


def _normalize_static_int_sequence(values, name, *, require_positive=False):
    values = tl._unwrap_if_constexpr(values)
    if isinstance(values, tl.tuple):
        values = tuple(values.values)
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"mthreads TLE warp_specialize {name} must be a static sequence")

    normalized = []
    for index, value in enumerate(values):
        value = tl._unwrap_if_constexpr(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"mthreads TLE warp_specialize {name}[{index}] must be a compile-time integer")
        if require_positive and value <= 0:
            raise ValueError(f"mthreads TLE warp_specialize {name}[{index}] must be positive")
        normalized.append(value)
    return normalized


def normalize_config(worker_num_warps, worker_num_regs):
    worker_num_warps = _normalize_static_int_sequence(
        worker_num_warps,
        "worker_num_warps",
        require_positive=True,
    )
    worker_num_regs = _normalize_static_int_sequence(
        worker_num_regs,
        "worker_num_regs",
    )
    return worker_num_warps, worker_num_regs


def partition_function_caller(generator):
    # Mthreads TLE regions are isolated from above and use direct SSA capture
    # remapping, so partition functions must be emitted inline.
    return generator.inline_JitFunction


def create_op(builder, result_types, worker_num_warps):
    # Mthreads keeps explicit captures on the isolated partitions holder rather
    # than on the outer ttg.warp_specialize operation.
    return builder.create_warp_specialize(result_types, worker_num_warps)


def create_partitions(builder, explicit_captures, num_partitions):
    return builder.create_warp_specialize_partitions(explicit_captures, num_partitions)
