import os

import pytest
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend
from triton.compiler.errors import CompilationError
from triton.runtime import DistributedRtContext


@triton.jit
def _transfer(buffer, peer, src_offset, dst_offset, valid_n, ctx: tl.constexpr, SPACE: tl.constexpr, MODE: tl.constexpr,
              IS_PUT: tl.constexpr, CONTEXT_ID: tl.constexpr, BLOCK_SIZE: tl.constexpr, STRIDE: tl.constexpr = 1):
    if CONTEXT_ID is None:
        remote = tle.remote(ctx, shard_id=peer, space=SPACE, dtype=tl.float32, mode=MODE)
    else:
        remote = tle.remote(ctx, peer, None, SPACE, tl.float32, None, None, CONTEXT_ID, MODE)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < valid_n
    copy_offsets = offsets * STRIDE
    if IS_PUT:
        values = tl.load(buffer + src_offset + copy_offsets, mask=mask)
        tl.store(remote + dst_offset + copy_offsets, values, mask=mask)
    else:
        values = tl.load(remote + src_offset + copy_offsets, mask=mask)
        tl.store(buffer + dst_offset + copy_offsets, values, mask=mask)


@triton.jit
def _remote_access(buffer, ctx: tl.constexpr, SPACE: tl.constexpr, MODE: tl.constexpr, OFFSET: tl.constexpr):
    remote = tle.remote(ctx, shard_id=1, space=SPACE, dtype=tl.float32, offset=OFFSET, mode=MODE)
    value = tl.load(remote)
    tl.store(buffer, value + 1)


def _compile_ttgir(kernel, signature, **constants):
    target = GPUTarget("cuda", 90, 32)
    backend = make_backend(target)
    options = backend.parse_options({})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    ctx = DistributedRtContext()
    # The runtime packs the registered-memory handle before the communicator.
    ctx._comm_ptr = 0x2000
    ctx._mem_ptr = 0x1000
    source = ASTSource(kernel, signature, constexprs={"ctx": ctx, **constants})
    module = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                            context)
    metadata = {}
    module = backend.make_ttir(module, metadata, options, target.arch)
    module = backend.make_ttgir(module, metadata, options, target.arch)
    return str(module), metadata


@pytest.mark.parametrize("space,mode,context_id", [("device", "transfer", None), ("device", "transfer", 2),
                                                    ("node", None, 3), ("node", "transfer", 1)])
@pytest.mark.parametrize("is_put", [True, False], ids=["put", "get"])
def test_transfer_fusion(space, mode, context_id, is_put):
    ttgir, metadata = _compile_ttgir(
        _transfer, {"buffer": "*fp32", "peer": "i32", "src_offset": "i64", "dst_offset": "i64", "valid_n": "i32"},
        SPACE=space, MODE=mode, IS_PUT=is_put, CONTEXT_ID=context_id, BLOCK_SIZE=16, STRIDE=1)

    direction = "put" if is_put else "get"
    assert f'"tle.node_{direction}"' in ttgir
    if space == "device":
        assert "team_kind = 0 : i32" in ttgir
    assert f"context_id = {context_id or 0} : i32" in ttgir
    assert "elem_bytes = 4 : i64" in ttgir
    assert "tt.load" not in ttgir
    assert "tt.store" not in ttgir
    assert metadata["tle_node_buffer_bindings"] == ("s:0:8192" if is_put else "d:0:8192")


def test_default_device_mode_keeps_pointer_access():
    ttgir, metadata = _compile_ttgir(_remote_access, {"buffer": "*fp32"}, SPACE="device", MODE=None, OFFSET=2)
    assert 'space = "device"' in ttgir
    assert "tt.load" in ttgir
    assert "tt.store" in ttgir
    assert not metadata.get("tle_node_buffer_bindings")


@pytest.mark.parametrize("space,mode,offset,error", [
    ("device", "pointer", None, "mode must be None or"),
    ("cluster", "transfer", None, "cluster space does not support"),
    ("device", "transfer", 0, "in transfer mode does not accept offset"),
])
def test_invalid_remote_mode_arguments(space, mode, offset, error):
    with pytest.raises(CompilationError, match=error):
        _compile_ttgir(_remote_access, {"buffer": "*fp32"}, SPACE=space, MODE=mode, OFFSET=offset)


def test_device_transfer_requires_a_paired_copy(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_ttgir(_remote_access, {"buffer": "*fp32"}, SPACE="device", MODE="transfer", OFFSET=None)
    assert "could not fuse device remote pointer" in capfd.readouterr().err


def test_device_transfer_rejects_strided_copy(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_ttgir(
            _transfer, {"buffer": "*fp32", "peer": "i32", "src_offset": "i64", "dst_offset": "i64", "valid_n": "i32"},
            SPACE="device", MODE="transfer", IS_PUT=True, CONTEXT_ID=None, BLOCK_SIZE=16, STRIDE=2)
    assert "must use a scalar offset plus a contiguous" in capfd.readouterr().err


def _run_two_gpu_test():
    """Run with torchrun --standalone --nproc-per-node=2 and a P2P-enabled FlagCX runtime."""
    import torch
    import torch.distributed as dist

    if int(os.environ["WORLD_SIZE"]) != 2 or int(os.environ["LOCAL_WORLD_SIZE"]) != 2:
        raise RuntimeError("this test requires two ranks on one node")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    pool = tle.get_mem_pool()
    peer = 1 - local_rank
    src_offset, dst_offset, valid_n, block_size = 3, 21, 9, 16
    with torch.cuda.use_mem_pool(pool):
        buffer = torch.empty(48, dtype=torch.float32, device="cuda")
    ctx = tle.create_dist_tensor(buffer)
    try:
        for is_put in (True, False):
            payload = torch.arange(16, device="cuda", dtype=torch.float32)
            buffer.fill_(-1)
            buffer[:16] = payload + 100 * local_rank
            torch.cuda.synchronize()
            dist.barrier()

            _transfer[(1, )](buffer, peer, src_offset, dst_offset, valid_n, ctx, "device", "transfer", is_put, None,
                             block_size)
            torch.cuda.synchronize()
            dist.barrier()

            expected = torch.full_like(buffer, -1)
            expected[:16] = payload + 100 * local_rank
            expected[dst_offset:dst_offset + valid_n] = payload[src_offset:src_offset + valid_n] + 100 * peer
            matches = (buffer == expected).all().to(torch.int32)
            dist.all_reduce(matches, op=dist.ReduceOp.MIN)
            direction = "PUT" if is_put else "GET"
            assert matches.item() == 1, f"node-local {direction} copied incorrect data"
    finally:
        tle.cleanup_communicator()


if __name__ == "__main__":
    _run_two_gpu_test()
