# Copyright 2026- Xcoresigma Technology Co., Ltd

from __future__ import annotations


def _detect_backend() -> str:
    from triton._flagtree_backend import FLAGTREE_BACKEND as _flagtree_backend

    return _flagtree_backend.strip().lower() if _flagtree_backend else "nvidia"


def _build_ascend_impl() -> dict[str, ...]:
    from triton.experimental.tle.language.dsa.ascend.communication import (
        cleanup_communicator,
        create_dist_tensor,
        init_communicator,
    )
    from triton.experimental.tle.language.dsa.ascend.distributed import (
        MeshConfig,
        device_mesh,
        distributed_barrier,
        gemm_swizzle2d_Nz,
        remote,
        shard_id,
        swizzle2d_Nz,
    )

    return {
        "init_communicator": init_communicator,
        "create_dist_tensor": create_dist_tensor,
        "cleanup_communicator": cleanup_communicator,
        "device_mesh": device_mesh,
        "MeshConfig": MeshConfig,
        "remote": remote,
        "shard_id": shard_id,
        "distributed_barrier": distributed_barrier,
        "swizzle2d_Nz": swizzle2d_Nz,
        "gemm_swizzle2d_Nz": gemm_swizzle2d_Nz,
    }


def _build_nvidia_impl() -> dict[str, ...]:
    from triton.experimental.tle.distributed import (
        B,
        P,
        S,
        ShardedTensor,
        ShardingSpec,
        device_mesh,
        distributed_barrier,
        distributed_dot,
        make_sharded_tensor,
        remote,
        reshard,
        shard_id,
        sharding,
    )

    return {
        "B": B,
        "P": P,
        "S": S,
        "ShardedTensor": ShardedTensor,
        "ShardingSpec": ShardingSpec,
        "device_mesh": device_mesh,
        "distributed_barrier": distributed_barrier,
        "distributed_dot": distributed_dot,
        "make_sharded_tensor": make_sharded_tensor,
        "remote": remote,
        "reshard": reshard,
        "shard_id": shard_id,
        "sharding": sharding,
    }


# Once at import time: build the selected backend's op table and expose every
# op as a plain module attribute.
BACKEND = _detect_backend()
IMPL = _build_ascend_impl() if BACKEND == "ascend" else _build_nvidia_impl()
globals().update(IMPL)


def ops() -> list[str]:
    return sorted(IMPL.keys())
