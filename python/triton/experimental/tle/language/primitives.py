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

TLE_PRIMITIVES = frozenset({
    "raw.call",
    "raw.call_smem",
    "load",
    "extract_tile",
    "insert_tile",
    "range",
    "device_mesh",
    "sharding",
    "shard_id",
    "distributed_barrier",
    "remote",
    "cumsum",
    "pipe",
    "pipe.reader",
    "pipe.reader.wait",
    "pipe.reader.release",
    "pipe.writer",
    "pipe.writer.acquire",
    "pipe.writer.commit",
    "pipe.writer.close",
    "gpu.alloc",
    "gpu.copy",
    "gpu.local_ptr",
    "gpu.memory_space",
    "gpu.range",
    "gpu.pipeline",
    "gpu.set_layout",
    "gpu.warp_specialize",
    "gpu.alloc_barrier",
    "gpu.alloc_barriers",
    "gpu.barrier_arrive",
    "gpu.barrier_wait",
    "gpu.wgmma",
    "gpu.wgmma_wait",
    "gpu.buffered_tensor.slot",
    "ext.make_tensor_view",
    "ext.make_partition_view",
    "ext.make_view",
    "ext.dim",
    "ext.load_view_tko",
    "ext.store_view_tko",
    "ext.create_mem_token",
    "ext.join_mem_tokens",
})

__all__ = ["TLE_PRIMITIVES"]
