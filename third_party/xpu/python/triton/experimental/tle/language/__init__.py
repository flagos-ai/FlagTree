# Triton 3.6 XPU TLE language module (FlagTree XPU overlay)
#
# Available: `raw` (tle.raw on the cluster path), `gpu` (tle.gpu buffers and
# continuous DMA copy), and `dsa` with its `pipe` (SDNN-backed on-chip buffers).
from . import dsa
from . import gpu
from . import raw
from .dsa.pipe import pipe
from .distributed import (B, P, S, MeshConfig, ShardedTensor, ShardingSpec, _get_local_rank, device_mesh,
                          distributed_barrier, make_sharded_tensor, n_pes, remote, sharding, shard_id)

__all__ = [
    "gpu", "dsa", "pipe", "raw", "MeshConfig", "device_mesh", "B", "S", "P", "sharding", "ShardingSpec",
    "ShardedTensor", "make_sharded_tensor", "distributed_barrier", "shard_id", "n_pes", "remote"
]

from triton.backends.xpu.distributed import (clear_implicit_context, flagcx_rt_conf, register_implicit_context)

if flagcx_rt_conf.is_available:
    from . import communication
    from .communication import get_mem_pool

    def create_dist_tensor(buf_tensor):
        ctx = communication.create_dist_tensor(buf_tensor)
        ctx.register_buffer(buf_tensor)
        register_implicit_context(ctx)
        return ctx

    def cleanup_communicator():
        try:
            communication.cleanup_communicator()
        finally:
            clear_implicit_context()

    __all__ += ["get_mem_pool", "create_dist_tensor", "cleanup_communicator"]
