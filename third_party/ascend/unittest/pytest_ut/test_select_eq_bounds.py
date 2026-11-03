"""Equality-mask select lowering must handle empty and live ranges exactly."""
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _select_eq(X, Y, Out, State, ROWS: tl.constexpr, COLS: tl.constexpr, AXIS: tl.constexpr, SHIFT: tl.constexpr,
               BOUND: tl.constexpr, MODE: tl.constexpr):
    if ROWS == 0:
        offset = tl.arange(0, COLS)
        index = offset
    else:
        r = tl.arange(0, ROWS)[:, None]
        c = tl.arange(0, COLS)[None, :]
        offset = r * COLS + c
        if AXIS == 0:
            index = r
        else:
            index = c
    x = tl.load(X + offset)
    y = tl.load(Y + offset)
    if MODE == 2:
        shift = tl.load(State)
    else:
        shift = SHIFT
    if MODE == 1:
        bound = tl.load(State + 1)
    else:
        bound = BOUND
    tl.store(Out + offset, tl.where(index + shift == bound, x, y))


def _run(rows, cols, axis, shift, bound, mode, auto):
    shape = (cols, ) if rows == 0 else (rows, cols)
    size = max(rows, 1) * cols
    x = (torch.arange(size).float() * .125 + 1).reshape(shape)
    y = -x - .25
    # Check NaN classification and exact finite values, including signed zero.
    x.flatten().view(torch.int32)[2] = 0x7FC12345
    y.flatten()[4] = -0.
    y.flatten().view(torch.int32)[6] = 0x7FC54321
    source, other = x.npu(), y.npu()
    storage = torch.full((size + 128, ), 31415.25, device='npu')
    output = storage[64:-64]
    state = torch.zeros(2, dtype=torch.int32, device='npu')
    extent = cols if rows == 0 or axis == 1 else rows
    index = torch.arange(extent, dtype=torch.int32)
    if rows:
        index = index[:, None] if axis == 0 else index[None, :]
    values = [(shift, bound)]
    if mode == 1:
        values = [(shift, shift + i) for i in (-1, 0, 7, extent - 1, extent, extent + 9)]
    elif mode == 2:
        values = [(i, bound) for i in (-extent, -7, 0, 1, extent)]
    options = dict(num_warps=8, num_stages=1, multibuffer=False, enable_auto_blockify=auto, enable_select_analysis=True)
    for actual_shift, actual_bound in values:
        state.copy_(torch.tensor([actual_shift, actual_bound], dtype=torch.int32).npu())
        output.fill_(float('nan'))
        _select_eq[(1, )](source, other, output, state, rows, cols, axis, shift, bound, mode, **options)
        torch.npu.synchronize()
        expected = torch.where(index + actual_shift == actual_bound, x, y)
        actual = output.cpu()
        expected = expected.flatten()
        nan = torch.isnan(expected)
        assert torch.equal(torch.isnan(actual), nan)
        assert torch.equal(actual.view(torch.int32)[~nan], expected.view(torch.int32)[~nan])
        assert torch.all(storage[:64].cpu() == 31415.25)
        assert torch.all(storage[-64:].cpu() == 31415.25)
    assert torch.equal(source.cpu().view(torch.int32), x.view(torch.int32))
    assert torch.equal(other.cpu().view(torch.int32), y.view(torch.int32))


_STATIC = [(r, c, axis, 17, 17 + i)
           for r, c, axis in [(0, 32, 0), (0, 64, 0), (4, 32, 1), (32, 8, 0)]
           for i in (-1, 0, 7, (c if r == 0 or axis == 1 else r) - 1, c if r == 0 or axis == 1 else r,
                     (c if r == 0 or axis == 1 else r) + 9)]
_STATIC += [(0, 32, 0, -17, -17 + i) for i in (-1, 0, 7, 31, 32, 41)]

_STATIC += [(0, 32, 0, 2147483647, -2147483648), (0, 32, 0, 2147483630, -2147483648),
            (0, 32, 0, -2147483648, -2147483648)]


@pytest.mark.parametrize('auto', [False, True])
@pytest.mark.parametrize('rows,cols,axis,shift,bound', _STATIC)
def test_static_eq_bounds(rows, cols, axis, shift, bound, auto):
    _run(rows, cols, axis, shift, bound, 0, auto)


@pytest.mark.parametrize('auto', [False, True])
@pytest.mark.parametrize('rows,cols,axis', [(0, 32, 0), (4, 32, 1)])
@pytest.mark.parametrize('mode', [1, 2])
def test_dynamic_eq_bounds(rows, cols, axis, mode, auto):
    _run(rows, cols, axis, 17, 0, mode, auto)


@triton.jit
def _memory_eq(X, Out, SHIFT: tl.constexpr, BOUND: tl.constexpr, STORE: tl.constexpr):
    i = tl.arange(0, 32)
    mask = i + SHIFT == BOUND
    if STORE:
        value = tl.load(X + i)
        tl.store(Out + i, value, mask)
    else:
        value = tl.load(X + i, mask, other=-3.25)
        tl.store(Out + i, value)


@pytest.mark.parametrize('auto', [False, True])
@pytest.mark.parametrize('store', [False, True])
# The existing tail-singleton load exception is isolated in the handoff;
# its unchanged frontend IR and device failure reproduce with the baseline.
@pytest.mark.parametrize('shift,bound', [(17, 0), (17, 17), (17, 24), (17, 49), (2147483647, -2147483648)])
def test_eq_masked_memory(shift, bound, store, auto):
    cpu = torch.arange(32, dtype=torch.float32) * .125 + 1
    source = cpu.npu()
    storage = torch.full((160, ), -17.25, device='npu')
    output = storage[64:-64]
    _memory_eq[(1, )](source, output, shift, bound, store, num_warps=8, num_stages=1, multibuffer=False,
                      enable_auto_blockify=auto)
    torch.npu.synchronize()
    mask = torch.arange(32, dtype=torch.int32) + shift == bound
    expected = torch.where(mask, cpu, -17.25 if store else -3.25)
    assert torch.equal(output.cpu(), expected)
    assert torch.all(storage[:64].cpu() == -17.25)
    assert torch.all(storage[-64:].cpu() == -17.25)
    assert torch.equal(source.cpu(), cpu)
