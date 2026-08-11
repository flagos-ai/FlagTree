# Copyright 2026- Xcoresigma Technology Co., Ltd

import os
import torch
import torch.distributed as dist
import shmem as ash

_g_pe = None
_g_world_size = None
_g_mem_count = 0


def _get_ash_ip_port():
    # default to 8666 if ASH_MASTER_PORT is not set
    ash_port = os.environ.get("ASH_MASTER_PORT", "8666")
    # default to 127.0.0.1 if ASH_MASTER_ADDR is not set
    ash_addr = os.environ.get("ASH_MASTER_ADDR", "127.0.0.1")
    return f"tcp://{ash_addr}:{ash_port}"


def init_communicator(ip_port=None, ash_size=None, engine_type=None):
    global _g_pe, _g_world_size, _g_mem_count
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", rank=local_rank)
    _g_pe = dist.get_rank()
    _g_world_size = dist.get_world_size()
    dist.barrier()

    ash.set_conf_store_tls(False, "")
    attributes = ash.InitAttr()
    attributes.my_rank = _g_pe
    attributes.n_ranks = _g_world_size
    attributes.local_mem_size = ash_size if ash_size is not None else 1024 * 1024 * 1024
    attributes.ip_port = ip_port if ip_port is not None else _get_ash_ip_port()
    # ROCE/RDMA is the inter-node (multi-machine) protocol; MTE is intra-node only.
    if engine_type is None:
        engine_type = ash.OpEngineType.MTE
    attributes.option_attr.data_op_engine_type = engine_type
    ret = ash.aclshmem_init(attributes)
    if ret != 0:
        raise ValueError("[ERROR] aclshmem_init failed")
    _g_mem_count = 0


def create_dist_tensor(buf_tensor):
    """Allocate share memory for distributed communication.

    Each successful allocation increments the global `_g_mem_count` by 1.
    The communicator is finalized only when `cleanup_communicator` brings the
    count back to 0.
    """
    global _g_mem_count
    if isinstance(buf_tensor, torch.Tensor):
        peer_mem = ash.aclshmem_create_tensor([buf_tensor.numel()], dtype=buf_tensor.dtype, device_id=_g_pe)
        _g_mem_count += 1
        return peer_mem
    else:
        raise ValueError("[ERROR] buf_tensor must be a tensor")


def cleanup_communicator(peer_mem):
    """Free one distributed tensor.

    Decrements the global `_g_mem_count` by 1. Only when the count reaches 0
    (i.e. all tensors created by `create_dist_tensor` have been freed) is
    `aclshmem_finalize()` called.
    """
    global _g_mem_count
    torch.npu.synchronize()
    dist.barrier()
    ash.aclshmem_free_tensor(peer_mem)
    _g_mem_count -= 1
    if _g_mem_count < 0:
        raise ValueError("[ERROR] cleanup_communicator called more times than create_dist_tensor")
    if _g_mem_count == 0:
        ash.aclshmem_finalize()
