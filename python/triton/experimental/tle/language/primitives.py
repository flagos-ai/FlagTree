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
    ## TLE-Raw
    "raw.call",
    "raw.call_smem",

    ## TLE-Lite
    "load",
    "extract_tile",
    "insert_tile",
    "range",
    "device_mesh",
    "sharding",
    "shard_id",
    "n_pes",
    "distributed_barrier",
    "remote",
    "signal",
    "signal_wait",
    "cumsum",
    "pipe",
    "pipe.reader",
    "pipe.reader.wait",
    "pipe.reader.release",
    "pipe.writer",
    "pipe.writer.acquire",
    "pipe.writer.commit",
    "pipe.writer.close",

    ## TLE-Struct GPU
    "gpu.buffered_tensor.load",  # CommonIR
    "gpu.buffered_tensor.store",  # CommonIR
    "gpu.buffered_tensor.slot",
    "gpu.buffered_tensor.reshape",
    "gpu.buffered_tensor.subslice",
    "gpu.alloc",
    "gpu.copy",
    "gpu.local_ptr",
    "gpu.memory_space",
    "gpu.set_layout",
    "gpu.tensor_map_fenceproxy_acquire",
    "gpu.reinterpret_tensor_map",
    "gpu.warp_specialize",
    "gpu.alloc_barrier",
    "gpu.alloc_barriers",
    "gpu.barrier_wait",
    "gpu.barrier_arrive",
    "gpu.wgmma",
    "gpu.wgmma_wait",

    ## TLE-Struct DSA
    "dsa.alloc",
    "dsa.copy",
    "dsa.local_ptr",
    "dsa.memory_space",
    "dsa.to_tensor",
    "dsa.to_buffer",
    "dsa.add",
    "dsa.sub",
    "dsa.mul",
    "dsa.div",
    "dsa.max",
    "dsa.min",
    "dsa.pipeline",
    "dsa.parallel",
    "dsa.hint",
    "dsa.extract_slice",
    "dsa.insert_slice",
    "dsa.extract_element",
    "dsa.subview",
    "dsa.tile_alloc",
    "dsa.tile_copy",
    "dsa.tile_subview",
    "dsa.tile_to_tensor",
    "dsa.tile_set_flag",
    "dsa.tile_wait_flag",
    "dsa.tile_pipe_barrier",
    "dsa.tile_gm_offset",
    "dsa.tile_cube_launch",
    "dsa.tile_cube_wait",
    "dsa.workspace",

    ## TLE-Struct DSA (tsingmicro)
    "dsa.tsingmicro.randgen",
    "dsa.tsingmicro.rand",
    "dsa.tsingmicro.randn",

    ## TLE-Struct DSA (thrive): inter-die RMA and synchronization
    "dsa.thrive.putmem",
    "dsa.thrive.getmem",
    "dsa.thrive.wait",
    "dsa.thrive.notify",
    "dsa.thrive.fence",
    "dsa.thrive.sync",

    # TLE-Lite (tileir): token/tko
    "create_mem_token",
    "join_mem_tokens",
    "load_view_tko",
    "store_view_tko",
    "load_tko",
    "store_tko",
    "atomic_cas_tko",
    "atomic_xchg_tko",

    # TLE-Lite (tileir): view
    "dim",
    "make_tensor_view",
    "make_partition_view",
    "make_view",
})

__all__ = ["TLE_PRIMITIVES"]
