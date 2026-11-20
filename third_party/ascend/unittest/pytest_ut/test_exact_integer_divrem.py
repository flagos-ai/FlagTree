"""Full-width i32 division/remainder, independent of bounded FFT indices."""
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _divrem(X, D, Q, R, N: tl.constexpr, UNSIGNED: tl.constexpr, CONSTANT: tl.constexpr, DIVISOR: tl.constexpr,
            B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    n = tl.load(X + i, i < N, 0)
    if CONSTANT:
        d = tl.full((B, ), DIVISOR, tl.uint32 if UNSIGNED else tl.int32)
    else:
        d = tl.load(D + i, i < N, 1)
    if UNSIGNED:
        n = n.to(tl.uint32)
        d = d.to(tl.uint32)
    tl.store(Q + i, (n // d).to(tl.int32), i < N)
    tl.store(R + i, (n % d).to(tl.int32), i < N)


def _reference(x, d, unsigned):
    a, b = x.long(), d.long()
    if unsigned:
        a, b = a & 0xffffffff, b & 0xffffffff
    q = torch.div(a, b, rounding_mode='trunc')
    return q.int(), (a - q * b).int()


def _guarded(n):
    buf = torch.full((n + 128, ), 0x13579bdf, dtype=torch.int32, device='npu')
    return buf, buf[64:-64]


def _verify(q, r, qbuf, rbuf, ref):
    torch.npu.synchronize()
    assert torch.equal(q.cpu(), ref[0])
    assert torch.equal(r.cpu(), ref[1])
    for buf in (qbuf, rbuf):
        assert (buf[:64].cpu() == 0x13579bdf).all()
        assert (buf[-64:].cpu() == 0x13579bdf).all()


SIGNED = [1, -1, 2, -2, 3, 15, -15, 65535, -(1 << 31), (1 << 31) - 1]
UNSIGNED = [1, 2, 3, 15, 65535, 0x80000001, 0xffffffff]
CASES = [(False, d, True) for d in SIGNED] + [(True, d, True) for d in UNSIGNED]
CASES += [(False, 15, False), (True, 15, False)]


@pytest.mark.parametrize('unsigned,divisor,auto', CASES)
def test_constant_full_width_divrem(unsigned, divisor, auto):
    torch.manual_seed(92811)
    n = 4093
    x = torch.randint(-(1 << 31), 1 << 31, (n, ), dtype=torch.int64).int()
    boundaries = [
        -(1 << 31), -(1 << 31) + 1, -16777217, -16777216, -31, -30, -29, -1, 0, 1, 14, 15, 16, 16777215, 16777216,
        16777217, (1 << 31) - 2, (1 << 31) - 1
    ]
    # Add quotient-transition boundaries for each constant, with Python's
    # unbounded integer arithmetic as the construction oracle.
    for k in [-1024, -3, -1, 0, 1, 3, 1024]:
        boundaries += [k * divisor + delta for delta in [-1, 0, 1] if -(1 << 31) <= k * divisor + delta < 1 << 31]
    x[:len(boundaries)] = torch.tensor(boundaries, dtype=torch.int32)
    if divisor == -1:
        x[x == -(1 << 31)] += 1  # INT_MIN / -1 is source undefined behavior.
    d = torch.full((n, ), divisor, dtype=torch.int64).int()
    xn, dn = x.npu(), d.npu()
    qb, q = _guarded(n)
    rb, r = _guarded(n)
    for data in (x, x.roll(19)):
        xn.copy_(data.npu())
        k = _divrem[(triton.cdiv(n, 256), )](xn, dn, q, r, n, unsigned, True, divisor, 256, num_warps=4, num_stages=1,
                                             multibuffer=False, enable_auto_blockify=auto)
        _verify(q, r, qb, rb, _reference(data, d, unsigned))
        assert torch.equal(xn.cpu(), data)
        assert torch.equal(dn.cpu(), d)
    # Constant strength reduction must not be a floating approximation or
    # another tensor divide, including a widened scalarized divide.
    ir = k.asm['ttadapter']
    assert 'arith.divsi' not in ir and 'arith.divui' not in ir
    assert 'arith.sitofp' not in ir and 'arith.uitofp' not in ir
    if abs(divisor) == 15:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            _divrem[(triton.cdiv(n, 256), )](xn, dn, q, r, n, unsigned, True, divisor, 256, num_warps=4, num_stages=1,
                                             multibuffer=False, enable_auto_blockify=auto)
        xn.copy_(x.npu())
        graph.replay()
        _verify(q, r, qb, rb, _reference(x, d, unsigned))


@pytest.mark.parametrize('unsigned', [False, True])
@pytest.mark.parametrize('auto', [False, True])
def test_dynamic_full_width_divrem(unsigned, auto):
    torch.manual_seed(92812)
    n = 4093
    x = torch.randint(-(1 << 31), 1 << 31, (n, ), dtype=torch.int64).int()
    d = torch.randint(-(1 << 31), 1 << 31, (n, ), dtype=torch.int64).int()
    x[:8] = torch.tensor([-(1 << 31), (1 << 31) - 1, -1, -1, 16777217, -16777217, 0, 1])
    d[:8] = torch.tensor([15, 15, 1, -1, 1, 1, -1, -(1 << 31)])
    d[d == 0] = 1
    if not unsigned:
        d[(x == -(1 << 31)) & (d == -1)] = 1
    xn, dn = x.npu(), d.npu()
    qb, q = _guarded(n)
    rb, r = _guarded(n)

    def run():
        return _divrem[(triton.cdiv(n, 256), )](xn, dn, q, r, n, unsigned, False, 1, 256, num_warps=4, num_stages=1,
                                                multibuffer=False, enable_auto_blockify=auto)

    run()
    _verify(q, r, qb, rb, _reference(x, d, unsigned))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for data in (x.roll(7), x):
        # Keep source division defined in every replay.
        ds = d.clone()
        if not unsigned:
            ds[(data == -(1 << 31)) & (ds == -1)] = 1
        xn.copy_(data.npu())
        dn.copy_(ds.npu())
        graph.replay()
        _verify(q, r, qb, rb, _reference(data, ds, unsigned))
        assert torch.equal(xn.cpu(), data) and torch.equal(dn.cpu(), ds)
