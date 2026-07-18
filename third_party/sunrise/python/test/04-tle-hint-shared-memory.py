# flagtree tle
"""
TLE #@hint: shared_memory end-to-end test (sunrise)
===================================================

Verifies the source-comment hint path on sunrise:

    x = tl.load(x_ptr + offs)  #@hint: shared_memory

The SunriseHintHandler parses the `#@hint: shared_memory` comment and injects it
as the load's `flagtree_hints` attr; the `add_process_shared_memory_hint` TTGIR
pass then rewrites that load into the async copy -> shared -> local_load chain
(local_alloc + async_copy_global_to_local + async_commit_group + async_wait +
local_load). On sunrise these lower to async_load.* + STVM::Barrier0Op, so the
result must be numerically identical to a plain load.

Requires the sunrise hint handler to be registered in
triton/compiler/hint_manager.py (backend 'sunrise', aliases 'ptpu'/'tang').
"""

import torch
import triton
import triton.language as tl
import pytest

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def hinted_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    # The hint comment must sit on the load line so the handler maps it by lineno.
    x = tl.load(x_ptr + offs, mask=mask)  #@hint: shared_memory
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@pytest.mark.parametrize("n_elements", [128, 256, 512])
def test_hint_shared_memory(n_elements):
    torch.manual_seed(0)
    BLOCK = 128
    x = torch.randn(n_elements, device=DEVICE, dtype=torch.float32)
    y = torch.randn(n_elements, device=DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)

    grid = (triton.cdiv(n_elements, BLOCK), )
    hinted_add_kernel[grid](x, y, out, n_elements, BLOCK)

    torch.testing.assert_close(out.cpu(), (x + y).cpu(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
