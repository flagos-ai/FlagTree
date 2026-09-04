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

# flagtree tle
import triton.language as _language
from typing import TYPE_CHECKING

from .core import (
    cumsum,
    extract_tile,
    insert_tile,
    load,
)
from .pipe import (
    pipe,
    pipe_reader,
    pipe_slot,
    pipe_value,
    pipe_wait_result,
    pipe_writer,
)
from .distributed import (
    B,
    P,
    S,
    ShardedTensor,
    ShardingSpec,
    device_mesh,
    MeshConfig,
    BarrierKind,
    MemoryOrder,
    GroupKind,
    signal,
    signal_wait,
    distributed_barrier,
    distributed_dot,
    _infer_submesh_barrier_group,
    _mesh_to_cluster_dims,
    make_sharded_tensor,
    _normalize_remote_shard_id,
    remote,
    reshard,
    _resolve_launch_axis,
    shard_id,
    sharding,
)
from . import communication
from .communication import get_mem_pool, cleanup_communicator


def create_dist_tensor(buf_tensor):
    """Create a distributed tensor context and bind it to ``buf_tensor``."""
    ctx = communication.create_dist_tensor(buf_tensor)
    ctx.register_buffer(buf_tensor)
    return ctx


_EXTENSION_APIS = frozenset({
    "make_tensor_view",
    "make_partition_view",
    "make_view",
    "dim",
    "load_view_tko",
    "store_view_tko",
    "create_mem_token",
    "join_mem_tokens",
})


# Keep ordinary TLE imports independent of backend-specific extensions.
def __getattr__(name):
    if name not in _EXTENSION_APIS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    language_extensions = getattr(_language, "ext", None)
    if language_extensions is None:
        raise RuntimeError(f"tle.{name} requires a backend providing tl.ext.{name}")
    return getattr(language_extensions, name)


__all__ = [
    "load",
    "cumsum",
    "extract_tile",
    "insert_tile",
    "MeshConfig",
    "pipe",
    "pipe_reader",
    "pipe_slot",
    "pipe_value",
    "pipe_wait_result",
    "pipe_writer",
    "device_mesh",
    "S",
    "P",
    "B",
    "sharding",
    "ShardingSpec",
    "ShardedTensor",
    "make_sharded_tensor",
    "reshard",
    "remote",
    "shard_id",
    "signal",
    "signal_wait",
    "distributed_barrier",
    "distributed_dot",
    "distributed",
    "gpu",
    "raw",
    "mem_pool",
    "get_mem_pool",
    "create_dist_tensor",
    "cleanup_communicator",
    "BarrierKind",
    "MemoryOrder",
    "GroupKind",
]

from . import distributed, gpu, raw

# TLE-specific loop iterator: tl.range plus the `reorder` extension hint.
from .gpu import range

if TYPE_CHECKING:
    from triton.experimental.tle.language.gpu.semantic import TLESemantic as TLESemantic

__all__.append("range")
