import torch
import torch_npu  # noqa: F401

import triton
import triton.language as tl


@triton.jit
def _indirect_load_from_i64_ptr_kernel(
    block_table_ptrs,
    positions,
    out,
    BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)

    block_table_ptr = tl.load(block_table_ptrs).to(tl.pointer_type(tl.int32))

    pos = tl.load(positions + offsets)
    block_indices = pos // BLOCK_SIZE

    # Regression point:
    # int64 runtime address -> pointer<i32> -> vector indirect load.
    block_numbers = tl.load(block_table_ptr + block_indices)

    slot_ids = block_numbers * BLOCK_SIZE + pos % BLOCK_SIZE
    tl.store(out + offsets, slot_ids.to(tl.int64))


def test_indirect_load_pointer_cast_precise_size_e2e():
    block_size = 4
    block = 8

    block_table_cpu = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    positions_cpu = torch.tensor(
        [0, 1, 4, 7, 8, 11, 12, 15],
        dtype=torch.int32,
    )

    block_table = block_table_cpu.npu()
    positions = positions_cpu.npu()
    out = torch.empty((block, ), dtype=torch.int64, device="npu")

    block_table_ptrs = torch.tensor(
        [block_table.data_ptr()],
        dtype=torch.int64,
    ).npu()

    _indirect_load_from_i64_ptr_kernel[(1, )](
        block_table_ptrs,
        positions,
        out,
        BLOCK_SIZE=block_size,
        BLOCK=block,
    )

    torch.npu.synchronize()

    expected = (block_table_cpu[positions_cpu // block_size] * block_size + positions_cpu % block_size).to(torch.int64)

    assert torch.equal(out.cpu(), expected.cpu())
