# Triton 3.6 XPU TLE language module (FlagTree XPU overlay)
#
# Available: `raw` (tle.raw on the cluster path), `gpu` (tle.gpu buffers and
# continuous DMA copy), and `dsa` with its `pipe` (SDNN-backed on-chip buffers).
from . import dsa
from . import gpu
from . import raw
from .dsa.pipe import pipe


def device_mesh(*args, **kwargs):
    raise NotImplementedError("tle.device_mesh is not implemented for the XPU TLE backend")


def distributed_barrier(*args, **kwargs):
    raise NotImplementedError("tle.distributed_barrier is not implemented for the XPU TLE backend")


def shard_id(*args, **kwargs):
    raise NotImplementedError("tle.shard_id is not implemented for the XPU TLE backend")


def remote(*args, **kwargs):
    raise NotImplementedError("tle.remote is not implemented for the XPU TLE backend")


for _fn in [device_mesh, distributed_barrier, shard_id, remote]:
    _fn.__triton_builtin__ = True

del _fn

__all__ = ["gpu", "dsa", "pipe", "device_mesh", "distributed_barrier", "shard_id", "remote", "raw"]
