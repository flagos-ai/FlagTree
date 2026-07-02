import ctypes
from pathlib import Path

import os
import torch
import datetime
import triton.knobs as knobs


def load_library(library_path):
    library_path = Path(library_path).expanduser().resolve()
    return ctypes.CDLL(str(library_path))


def install_cumodule_hook(common):

    def hook(*args, **kwargs):
        key = kwargs["key"]
        function = kwargs["fn"].jit_function
        device = kwargs["compile"]["device"]
        kernel = function.device_caches[device][0].get(key)
        assert kernel is not None
        kernel._init_handles()
        result = common.nvshmemx_cumodule_init_wrapper(ctypes.c_void_p(kernel.module))
        assert result == 0, f"nvshmemx_cumodule_init failed: {result}"

    knobs.runtime.jit_post_compile_hook = hook


def init_torch_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", local_rank),
        timeout=datetime.timedelta(seconds=1800),
    )
    group = torch.distributed.new_group(ranks=list(range(world_size)), backend="nccl")
    torch.distributed.barrier(group=group)
    return group


def init_nvshmem_by_torch_pg(common, group):
    rank = group.rank()
    world_size = group.size()
    uid_size = 1024
    uid_buffer = ctypes.create_string_buffer(uid_size)

    if rank == 0:
        result = common.nvshmem_get_unique_id_bytes(uid_buffer, uid_size)
        assert result == 0, f"nvshmemx_get_uniqueid failed: {result}"
        uid = bytes(uid_buffer.raw)
    else:
        uid = bytes(uid_size)

    objects = [uid]
    torch.distributed.broadcast_object_list(
        objects,
        src=torch.distributed.get_global_rank(group, 0),
        group=group,
    )
    uid_buffer = ctypes.create_string_buffer(objects[0], uid_size)
    result = common.nvshmem_init_from_torch_distributed(
        rank,
        world_size,
        int(os.environ["LOCAL_RANK"]),
        uid_buffer,
        uid_size,
    )
    assert result == 0, f"NVSHMEM init failed: {result}"
    torch.distributed.barrier(group=group)


def tensor_from_pointer(pointer, shape, dtype, device):
    elements = 1
    for extent in shape:
        elements *= extent
    storage = torch._C._construct_storage_from_data_pointer(
        pointer.value,
        device,
        elements * dtype.itemsize,
    )
    return torch.empty(0, dtype=dtype, device=device).set_(storage).view(shape)


def _env_flag_enabled(name):
    value = os.environ.get(name, "")
    return value.strip().lower() in ("1", "true", "yes", "on")


def prepare_clang_bitcode(
    common,
    local_rank,
    bitcode_path,
    source_path,
    dialect_function,
    public_api_names=None,
):
    bitcode_path = Path(bitcode_path).expanduser().resolve()
    source_path = Path(source_path).expanduser().resolve()
    assert source_path.is_file(), f"missing device source: {source_path}"

    force = _env_flag_enabled("FORCE_BITCODE")
    should_build = force or not bitcode_path.is_file()
    reason = "forced" if force else "missing bitcode"

    if not should_build and source_path.stat().st_mtime_ns > bitcode_path.stat().st_mtime_ns:
        should_build = True
        reason = "device source is newer"

    if local_rank == 0:
        if should_build:
            print(f"[build] {bitcode_path} ({reason})", flush=True)
            generated = dialect_function.make_bc(public_api_names)
            assert generated == bitcode_path
        else:
            print(f"[reuse] {bitcode_path}", flush=True)

    common.nvshmem_barrier_all_wrapper()
    assert bitcode_path.is_file(), f"missing device bitcode: {bitcode_path}"
    return {bitcode_path.stem: str(bitcode_path)}
