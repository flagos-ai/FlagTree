"""The COREX FMAD pass must honor each compilation's FP contract."""
import pytest
import triton
import triton.language as tl

@triton.jit
def _product_sum(p, v, out, TAG: tl.constexpr):
    h = tl.arange(0, 4)[:, None, None]
    t = tl.arange(0, 16)[None, :, None]
    d = tl.arange(0, 128)[None, None, :]
    probability = tl.load(p + h * 16 + t)
    value = tl.load(v + h * 2048 + t * 128 + d)
    product = probability * value
    halves = tl.trans(tl.reshape(product, (4, 2, 8, 128)), (0, 2, 3, 1))
    low, high = tl.split(halves)
    paired = low + high
    halves = tl.trans(tl.reshape(paired, (4, 2, 4, 128)), (0, 2, 3, 1))
    low, high = tl.split(halves)
    paired = low + high
    halves = tl.trans(tl.reshape(paired, (4, 2, 2, 128)), (0, 2, 3, 1))
    low, high = tl.split(halves)
    paired = low + high
    low, high = tl.split(tl.trans(paired, (0, 2, 1)))
    tl.store(out + tl.arange(0, 4)[:, None] * 128 + tl.arange(0, 128)[None, :], low + high)


def test_separate_product_rounding_after_fused_compilation():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    torch.manual_seed(101)
    probability = torch.randn((4, 16), device="cuda")
    value = torch.randn((4, 16, 128), device="cuda")
    result = torch.empty((4, 128), device="cuda")
    # Changing contracts in one process must affect both directions; no
    # process-global compiler option may leak from the previous kernel.
    for index, fusion in enumerate((False, True, False, True, False)):
        compiled = _product_sum[(1,)](
            probability, value, result, index, num_warps=32, enable_fp_fusion=fusion
        )
        torch.cuda.synchronize()
        if not fusion:
            reference = probability.cpu()[:, :, None] * value.cpu()
            for half in (8, 4, 2, 1):
                reference = reference[:, :half] + reference[:, half:]
            assert torch.equal(result.cpu(), reference[:, 0])
