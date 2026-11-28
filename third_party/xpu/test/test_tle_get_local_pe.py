"""Check the XPU FlagCX local rank with two local torchrun ranks."""

import os

import pytest
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))


@triton.jit
def _tle_local_pe_kernel(out_ptr, device_dptr, mesh: tl.constexpr):
    pid = tl.program_id(0)
    local_rank = tle.shard_id(mesh, "device", device_dptr=device_dptr)
    tl.store(out_ptr + pid, local_rank)


class TestLocalPeCount:

    @pytest.mark.skipif(
        os.environ.get("WORLD_SIZE") != "2" or os.environ.get("LOCAL_WORLD_SIZE") != "2",
        reason="requires two local torchrun ranks with the XPU FlagCX runtime",
    )
    def test_tle_local_pe_kernel(self):
        grid = 2
        local_rank = int(os.environ["LOCAL_RANK"])
        with torch.cuda.use_mem_pool(tle.get_mem_pool()):
            y = torch.full((grid, ), -1, dtype=torch.int32, device="cuda")
        device_dptr = tle.create_dist_tensor(y)

        try:
            compiled = _tle_local_pe_kernel[(grid, )](
                out_ptr=y,
                device_dptr=device_dptr,
                mesh=DEVICE_MESH,
                num_ctas=1,
                num_warps=4,
            )
            torch.cuda.synchronize()
            actual = y.cpu()
            expected = torch.full((grid, ), local_rank, dtype=torch.int32)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            assert compiled.metadata.xpu_uses_flagcx
            ttxir = compiled.asm.get("ttxir")
            if ttxir:
                assert "triton_xpu.get_device_id" in ttxir
                assert ttxir.count("triton_xpu.device_init") == 1
                assert ttxir.index("triton_xpu.device_init") < ttxir.index("scf.for")
            llir = compiled.asm.get("llir")
            if llir:
                assert "call i32 @flagcxDevCommGetIntraRank(" in llir
            print(f"[Rank {local_rank}] PASSED: local PE {actual.tolist()}", flush=True)
        finally:
            tle.cleanup_communicator()


if __name__ == "__main__":
    TestLocalPeCount().test_tle_local_pe_kernel()
