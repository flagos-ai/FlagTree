# Copyright 2026- FlagOS Contributors
# Copyright 2026- Xcoresigma Technology Co., Ltd

from .core import (UB, L1, L0A, L0B, L0C, PIPE, SyncSpec, sub_vec_id, sub_vec_num, sync_block_set, sync_block_wait,
                   sync_block_all, compile_hint, raw, multibuffer)
from .pipe import (PipeScheduler, pipe_reader, pipe_slot, pipe_value, pipe_wait_result, pipe_writer, pipe_scheduler,
                   run_pipe, reset_pipe_event_allocator)
from .distributed import (MeshConfig, device_mesh, remote, shard_id, distributed_barrier, swizzle2d_Nz,
                          gemm_swizzle2d_Nz)
from triton.language.extra.cann.extension.scope import scope
from . import custom_ops

# ``custom_ops`` and ``core`` import the CANN language extension, which
# registers its own ``tensor.extract_slice`` / ``tensor.insert_slice`` members.
# Those require explicit sizes/strides and cannot take constexpr-tuple offsets.
# Register the TLE slice frontends afterwards so the member forms follow the
# portable ``tle.dsa`` API on Ascend as well.
from triton._flagtree_backend import FLAGTREE_BACKEND

if FLAGTREE_BACKEND == "ascend":
    import triton.language.core as _tl
    from . import compat as _compat

    _tl._tensor_member_fn(_compat.extract_slice)
    _tl._tensor_member_fn(_compat.insert_slice)

__all__ = [
    "UB",
    "L1",
    "L0A",
    "L0B",
    "L0C",
    "PIPE",
    "SyncSpec",
    "sub_vec_id",
    "sub_vec_num",
    "sync_block_set",
    "sync_block_wait",
    "sync_block_all",
    "compile_hint",
    "raw",
    "scope",
    "multibuffer",
    "PipeScheduler",
    "pipe_reader",
    "pipe_slot",
    "pipe_value",
    "pipe_wait_result",
    "pipe_writer",
    "pipe_scheduler",
    "run_pipe",
    "reset_pipe_event_allocator",
    "MeshConfig",
    "device_mesh",
    "remote",
    "shard_id",
    "distributed_barrier",
    "swizzle2d_Nz",
    "gemm_swizzle2d_Nz",
    "custom_ops",
]
