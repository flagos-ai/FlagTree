"""MMA reduction hoisting must preserve the accumulator recurrence."""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['steps'])
def _recurrence(A, B, Scale, Out, Observed, steps, SEED: tl.constexpr,
                MODE: tl.constexpr):
    m = tl.arange(0, 16)
    n = tl.arange(0, 32)
    k = tl.arange(0, 16)
    a = tl.load(A + m[:, None] * 16 + k[None, :])
    acc = tl.full((16,), SEED, tl.float32)
    for step in range(steps):
        b = tl.load(B + step * 512 + k[:, None] * 32 + n[None, :])
        value = tl.dot(a, b)
        alpha = tl.load(Scale + step * 16 + m)
        if MODE == 'observed':
            tl.store(Observed + step * 16 + m, acc)
        if MODE == 'nonlinear':
            acc = acc * acc + tl.sum(value, 1)
        else:
            acc = acc * alpha + tl.sum(value, 1)
    tl.store(Out + m, acc)


@pytest.mark.parametrize('steps', [0, 1, 3])
@pytest.mark.parametrize('seed', [0.0, 1.0, 5.0])
@pytest.mark.parametrize('mode', ['linear', 'observed', 'nonlinear'])
def test_mma_reduction_preserves_initial_value_and_in_loop_observers(steps, seed, mode):
    if not torch.cuda.is_available():
        pytest.skip('requires an Iluvatar GPU')
    from triton._C import libtriton
    if not hasattr(libtriton, 'iluvatar'):
        pytest.skip('requires the Iluvatar MMA reduction pass')
    rows = torch.arange(1, 17, device='cuda', dtype=torch.float32)
    a = rows[:, None].expand(16, 16).contiguous().bfloat16()
    b = torch.stack([torch.full((16, 32), i + 1, device='cuda', dtype=torch.bfloat16) for i in range(3)])
    scale = torch.stack([torch.zeros_like(rows), torch.full_like(rows, .5), torch.full_like(rows, 2.)])
    output = torch.empty((16,), device='cuda', dtype=torch.float32)
    observed = torch.empty((3, 16), device='cuda', dtype=torch.float32)
    kernel = _recurrence[(1,)](a, b, scale, output, observed, steps, seed, mode,
                              num_warps=4, num_stages=1)
    expected = torch.full_like(rows, seed)
    history = []
    for i in range(steps):
        history.append(expected.clone())
        contribution = rows * (512 * (i + 1))
        expected = expected.square() + contribution if mode == 'nonlinear' else expected * scale[i] + contribution
    torch.testing.assert_close(output, expected, rtol=1e-6 if mode == 'nonlinear' else 0, atol=0)
    if mode == 'observed' and steps:
        torch.testing.assert_close(observed[:steps], torch.stack(history), rtol=0, atol=0)
    if mode == 'linear':
        # Keep the optimization: register-local partials are carried through
        # the loop, followed by one cross-lane reduction on loop exit.
        assert 'tensor<16x16x2xf32' in kernel.asm['ttgir']
