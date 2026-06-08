import torch
import triton
import triton.language as tl
import triton.language.extra.triton_thrive.dshmem as dshmem
import triton.experimental.tle.language as tle

import pytest


@triton.jit
def dshmem_all_gather(
    src_ptr,  # *Pointer* to source vector.
    dst_ptr,  # *Pointer* to destination vector.
    element_per_rank,
):
    # `pe` in libshmem is actually `die`.
    die_id = dshmem.my_pe()
    die_num = dshmem.n_pes()

    # Each die put their data to the others
    for dst_id in range(0, die_num):
        die_offset = die_id * element_per_rank
        dshmem.putmem_nbi_block(dst_ptr + die_offset, src_ptr, element_per_rank, dst_id)


@pytest.mark.skip(reason="not supported on thrive")
@pytest.mark.parametrize("NUM_TP, element_per_rank", [(4, 500), (2, 512)])
def test_all_gather(NUM_TP, element_per_rank):
    # thrive die is mapped to tle device
    mesh = tle.device_mesh(topology={"device": NUM_TP})
    spec = tle.sharding(mesh, split=[tle.S("device")], partial=[])

    src_cpu = torch.rand(NUM_TP, element_per_rank)
    src_sharded = tle.make_sharded_tensor(src_cpu, sharding=spec, shape=[NUM_TP, element_per_rank])

    dst_cpu = torch.empty(NUM_TP, NUM_TP, element_per_rank)
    dst_sharded = tle.make_sharded_tensor(dst_cpu, sharding=spec, shape=[NUM_TP, NUM_TP, element_per_rank])

    # kernel-side communication via dshmem
    dshmem_all_gather[(1, )](src_sharded, dst_sharded, element_per_rank)

    # check
    dst_cpu = dst_sharded.cpu()
    for i in range(NUM_TP):
        torch.testing.assert_close(dst_cpu[i], src_cpu)


if __name__ == "__main__":
    test_all_gather(4, 512)
