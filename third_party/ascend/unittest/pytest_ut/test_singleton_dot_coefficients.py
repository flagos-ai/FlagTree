"""MIX singleton coefficients preserve value and memory ordering without sharing."""
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def _singleton_dot(A, B, W, Out, MUTATE: tl.constexpr):
    batch = tl.program_id(0)
    row = tl.arange(0, 16)[:, None]
    col = tl.arange(0, 16)[None, :]
    a = tl.load(A + batch * 256 + row * 16 + col)
    b = tl.load(B + batch * 256 + row * 16 + col)
    product = tl.dot(a, b, input_precision="ieee")
    index = tl.arange(0, 32)
    weights = tl.load(W + batch * 32 + index)
    if MUTATE:
        # Independent of weights: every use of the read is still a projection.
        # Moving projected reads after this store would change the result.
        tl.store(W + batch * 32 + index, 31.)
    for i in tl.static_range(8):
        coefficient = al.extract_slice(weights, [2 * i + 1], [1], [1])
        value = tl.reshape(product * coefficient, (256, ))
        tl.store(Out + batch * 2048 + i * 256 + tl.arange(0, 256), value)


@pytest.mark.parametrize('bind', [False, True])
@pytest.mark.parametrize('mutate', [False, True])
def test_singleton_dot_coefficients(bind, mutate):
    batch, guard = 3, 128
    torch.manual_seed(111127)
    ah, bh, wh = torch.randn(batch, 16, 16), torch.randn(batch, 16, 16), torch.randn(batch, 32)
    a, b = ah.npu(), bh.npu()
    weights_guard = torch.full((batch * 32 + 2 * guard, ), 23456., device='npu')
    weights = weights_guard[guard:-guard]
    result_guard = torch.full((batch * 2048 + 2 * guard, ), 12345., device='npu')
    out = result_guard[guard:-guard]
    for scale in (1., 0.001):
        a.copy_(ah * scale)
        expected = ((ah.double() * scale) @ bh.double())[:, None] * wh[:, 1:16:2].double()[:, :, None, None]

        def launch():
            _singleton_dot[(batch, )](a, b, weights, out, mutate, num_warps=4, num_stages=1, multibuffer=False,
                                      enable_auto_blockify=True, enable_auto_bind_sub_block=bind)

        weights.copy_(wh.flatten())
        launch()
        torch.npu.synchronize()
        for replay in (False, True):
            if replay:
                weights.copy_(wh.flatten())
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    launch()
                weights.copy_(wh.flatten())
                out.fill_(12345.)
                graph.replay()
                torch.npu.synchronize()
            got = out.cpu().reshape_as(expected).double()
            error = torch.linalg.vector_norm(got - expected) / torch.linalg.vector_norm(expected)
            assert error < 5e-7, float(error)
            assert torch.isfinite(got).all()
            assert (result_guard[:guard].cpu() == 12345.).all()
            assert (result_guard[-guard:].cpu() == 12345.).all()
            assert (weights_guard[:guard].cpu() == 23456.).all()
            assert (weights_guard[-guard:].cpu() == 23456.).all()
            expected_weights = torch.full_like(wh, 31.) if mutate else wh
            assert torch.equal(weights.cpu().reshape_as(wh), expected_weights)
            assert torch.equal(a.cpu(), ah * scale)
            assert torch.equal(b.cpu(), bh)
