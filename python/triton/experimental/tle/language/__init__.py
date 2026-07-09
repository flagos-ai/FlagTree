# flagtree tle
import triton.language as _language

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
from .communication import get_mem_pool, create_comm_tensor, cleanup_communicator

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
    "distributed_barrier",
    "distributed_dot",
    "distributed",
    "gpu",
    "raw",
    "mem_pool",
    "get_mem_pool",
    "create_comm_tensor",
    "cleanup_communicator",
]

from . import distributed, gpu, raw
