import triton.experimental.tle.language as tle
import torch
import triton
import triton.language as tl

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))


@triton.jit
def _remote_peer_d2d_kernel(in_ptr, out_ptr, mesh: tl.constexpr, BLOCK: tl.constexpr):

    remote_mem = tle.remote(in_ptr, space="device", dtype=tl.float32, shard_id=0)
    x = tl.load(remote_mem)
    tl.store(out_ptr, x)


class TestDeviceToDevice:

    def test_tle_d2d_remote(self):
        block = 64
        grid = 2

        N = 64

        with torch.cuda.use_mem_pool(tle.get_mem_pool()):
            x = torch.randn((N, N), dtype=torch.float32, device="cuda")
        y = torch.empty_like(x)

        dis_tensor_ptr = tle.create_comm_tensor(x)

        compiled = _remote_peer_d2d_kernel.warmup(
            in_ptr=dis_tensor_ptr,
            out_ptr=y,
            mesh=DEVICE_MESH,
            BLOCK=block,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert "remote_pointers" in compiled.asm["ttgir"]
        assert "flagcxGetIntraPointerC" in compiled.asm['ptx']

        _remote_peer_d2d_kernel[(grid, )](in_ptr=dis_tensor_ptr, out_ptr=y, mesh=DEVICE_MESH, BLOCK=block)

        tle.cleanup_communicator()


TestDeviceToDevice().test_tle_d2d_remote()
