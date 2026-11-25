# Triton 3.6 XPU TLE language module (FlagTree XPU overlay)
#
# Available: `raw` (tle.raw on the cluster path), `gpu` (tle.gpu buffers and
# continuous DMA copy), and `dsa` with its `pipe` (SDNN-backed on-chip buffers).
from . import dsa
from . import gpu
from . import raw
from .dsa.pipe import pipe
from .distributed import (remote, signal, signal_wait, device_mesh, MeshConfig,
                          distributed_barrier, shard_id, n_pes,
                          _get_local_rank, _get_world_rank,
                          BarrierKind, MemoryOrder, MemoryScope, GroupKind)
from .sharding import (B, S, P, ShardingSpec, ShardedTensor, sharding,
                       make_sharded_tensor, reshard, distributed_dot)
from .communication import get_mem_pool, create_dist_tensor, cleanup_communicator


__all__ = ["gpu", "dsa", "pipe", "device_mesh", "MeshConfig", "distributed_barrier",
           "shard_id", "n_pes", "_get_local_rank", "_get_world_rank", "remote",
           "signal", "signal_wait", "raw", "get_mem_pool", "create_dist_tensor",
           "cleanup_communicator", "B", "S", "P", "sharding", "ShardingSpec",
           "ShardedTensor", "make_sharded_tensor", "reshard", "distributed_dot",
           "BarrierKind", "MemoryOrder", "MemoryScope", "GroupKind"]
