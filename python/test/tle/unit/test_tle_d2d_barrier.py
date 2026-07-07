import triton.experimental.tle.language as tle
import torch
import triton
import triton.language as tl

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))


@triton.jit
def _barrier_d2d_kernel(dev_comm_dptr, dev_mem_dptr, out_ptr, mesh: tl.constexpr, BLOCK: tl.constexpr):
    tle.distributed_barrier(comm_ptr=dev_comm_dptr, space="device", group_kind="block", order="acqrel",
                            barrier_kind="sync")


class TestD2DBarrier:

    def test_tle_d2d_barrier(self):
        block = 64
        grid = 2

        N = 64

        with torch.cuda.use_mem_pool(tle.get_mem_pool()):
            x = torch.randn((N, N), dtype=torch.float32, device="cuda")
        y = torch.empty_like(x)

        dev_comm_dptr, dev_mem_dptr = tle.create_comm_tensor(x)

        compiled = _barrier_d2d_kernel.warmup(
            dev_comm_dptr=dev_comm_dptr,
            dev_mem_dptr=dev_mem_dptr,
            out_ptr=y,
            mesh=DEVICE_MESH,
            BLOCK=block,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert "distributed_barrier" in compiled.asm["ttgir"]
        assert "flagcxIntraBarrier" in compiled.asm['ptx']

        _barrier_d2d_kernel[(grid, )](dev_comm_dptr=dev_comm_dptr, dev_mem_dptr=dev_mem_dptr, out_ptr=y,
                                      mesh=DEVICE_MESH, BLOCK=block)

        tle.cleanup_communicator()


TestD2DBarrier().test_tle_d2d_barrier()
