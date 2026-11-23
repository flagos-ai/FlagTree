"""Raw-bit MP31 TME roundtrip with option off/on and live stage/phase reuse."""

import pytest
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from test_tle_utils import require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _issue_reused_phases(src, dst, epochs, BLOCK: tl.constexpr, STAGES: tl.constexpr):
    ring = tle.gpu.alloc((STAGES, BLOCK), tl.float16, nv_mma_shared_layout=False)
    pipe = tle.pipe(capacity=STAGES, scope="cta", data=ring)
    writer = pipe.writer()
    reader = pipe.reader()
    # Runtime loop: every physical slot goes through both phase values repeatedly.
    for epoch in tl.range(0, epochs, num_stages=1):
        # Launch every stage before waiting for the first one.
        for stage in tl.static_range(STAGES):
            step = epoch * STAGES + stage
            slot = writer.acquire(step)
            tle.gpu.copy(src, slot.data, (BLOCK, ), ((step + 1) * BLOCK, ))
            writer.commit(step)
        for stage in tl.static_range(STAGES):
            step = epoch * STAGES + stage
            ready = reader.wait(step)
            tle.gpu.copy(ready.slot.data, dst, (BLOCK, ), ((step + 1) * BLOCK, ))
            reader.release(step)


@pytest.mark.parametrize("collective", [False, True], ids=["lane", "warp"])
@pytest.mark.parametrize("stages", [1, 2, 3], ids=["stage1", "stage2", "stage3"])
def test_musa_tme_issue_reused_phases(collective, stages):
    pytest.importorskip("torch_musa")
    if not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    if triton.runtime.driver.active.get_current_target().backend != "musa":
        pytest.skip("MUSA must be the active Triton backend")
    if tuple(torch.musa.get_device_capability()) != (3, 1):
        pytest.skip("TME warp issue regression targets MP31")

    block, epochs = 128, 11
    elements = (epochs * stages + 2) * block
    generator = torch.Generator(device="cpu").manual_seed(328431)
    source_bits = torch.randint(0, 65536, (elements, ), generator=generator, dtype=torch.int32).to(torch.int16)
    source = source_bits.to(device="musa").view(torch.float16)
    output = torch.empty_like(source)
    src = TensorDescriptor.from_tensor(source, [block])
    dst = TensorDescriptor.from_tensor(output, [block])
    opts = dict(BLOCK=block, STAGES=stages, num_warps=4, num_stages=1, tme_issue_warp=collective)
    compiled = _issue_reused_phases.warmup(src, dst, epochs, grid=(1, ), **opts)
    assert compiled.metadata.tme_issue_warp == collective
    ir = compiled.asm["llir"]
    for intrinsic in ("tme.ld.tile.1d", "tme.st.1d", "async.add.trans", "async.arrive.none.phaseid"):
        assert "call void @llvm.musa." + intrinsic in ir, ir

    sentinel = -137
    for n_epochs in (epochs, epochs - 2, epochs):
        output.view(torch.int16).fill_(sentinel)
        _issue_reused_phases[(1, )](src, dst, n_epochs, **opts)
        torch.musa.synchronize()
        end = (n_epochs * stages + 1) * block
        expected = torch.full_like(source_bits, sentinel)
        expected[block:end] = source_bits[block:end]
        torch.testing.assert_close(output.view(torch.int16).cpu(), expected, rtol=0, atol=0)
