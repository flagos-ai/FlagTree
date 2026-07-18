# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import triton.experimental.tle.language as tle
import torch
import triton
import triton.language as tl

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))


@triton.jit
def _tle_local_pe_kernel(out_ptr, device_dptr: tl.constexpr, mesh: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)  # noqa: F841
    local_rank = tle.shard_id(mesh, 'device', device_dptr=device_dptr)
    n_rank = mesh.shape[0]
    peer = (local_rank + 1) % n_rank  # noqa: F841


class TestLocalPeCount:

    def test_tle_local_pe_kernel(self):
        block = 64
        grid = 2
        N = 64
        with torch.cuda.use_mem_pool(tle.get_mem_pool()):
            x = torch.randn((N, N), dtype=torch.float32, device="cuda")
        y = torch.empty_like(x)
        device_dptr = tle.create_dist_tensor(x)

        compiled = _tle_local_pe_kernel.warmup(
            out_ptr=y,
            device_dptr=device_dptr,
            mesh=DEVICE_MESH,
            BLOCK=block,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert "get_device_id" in compiled.asm["ttgir"]

        _tle_local_pe_kernel[(grid, )](out_ptr=y, device_dptr=device_dptr, mesh=DEVICE_MESH, BLOCK=block)

        tle.cleanup_communicator()


TestLocalPeCount().test_tle_local_pe_kernel()
