"""Restructured ND loads must keep their logical axes and padding at dot."""
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _segmented_dot(X, W, Out, Side, WIDTH: tl.constexpr, LEFT: tl.constexpr, MASK: tl.constexpr, OTHER: tl.constexpr,
                   SIDE: tl.constexpr):
    row = tl.arange(0, 16)[:, None]
    col = tl.arange(0, WIDTH)[None, :]
    segment = col // (WIDTH // 16)
    idx = ((row + 1) * 15 + segment) * 5200 + col % (WIDTH // 16)
    if MASK:
        raw = tl.load(X + idx, segment < 15, OTHER)
    else:
        raw = tl.load(X + idx)
    k = tl.arange(0, 16)[None, :]
    basis = tl.load(W + row * 16 + k)
    if SIDE:
        tl.store(Side + row * WIDTH + col, raw)
    if LEFT:
        result = tl.dot(tl.trans(raw), basis, input_precision='ieee')
        dst = col.T * 16 + row.T
    else:
        result = tl.dot(basis, raw, input_precision='ieee')
        dst = row * WIDTH + col
    tl.store(Out + tl.program_id(0) * 16 * WIDTH + dst, result)


@triton.jit
def _dynamic_mask_dot(X, W, Out, valid, WIDTH: tl.constexpr):
    row = tl.arange(0, 16)[:, None]
    col = tl.arange(0, WIDTH)[None, :]
    segment = col // (WIDTH // 16)
    idx = ((row + 1) * 15 + segment) * 5200 + col % (WIDTH // 16)
    raw = tl.load(X + idx, segment < valid, 0.)
    k = tl.arange(0, 16)[None, :]
    basis = tl.load(W + row * 16 + k)
    result = tl.dot(basis, raw, input_precision='ieee')
    tl.store(Out + tl.program_id(0) * 16 * WIDTH + row * WIDTH + col, result)


def _raw(x, width, valid=16, other=0.):
    col = torch.arange(width)[None, :]
    idx = ((torch.arange(16)[:, None] + 1) * 15 + col // (width // 16)) * 5200 + col % (width // 16)
    return torch.where(col // (width // 16) < valid, x[idx], other)


def _guarded(count):
    backing = torch.full((count + 128, ), 31415.25, device='npu')
    out = backing[64:-64]
    out.fill_(float('nan'))
    return backing, out


def _check(backing, out, expected, programs):
    torch.npu.synchronize()
    got = out.cpu().reshape(programs, *expected.shape)
    assert torch.isfinite(got).all()
    assert (backing[:64].cpu() == 31415.25).all()
    assert (backing[-64:].cpu() == 31415.25).all()
    ref = expected.double().expand_as(got)
    error = torch.linalg.vector_norm(got.double() - ref) / torch.linalg.vector_norm(ref)
    assert error < 5e-7, float(error)
    zero_columns = (expected == 0).all(dim=0)
    if zero_columns.any():
        assert (got[..., zero_columns] == 0).all()
    zero_rows = (expected == 0).all(dim=1)
    if zero_rows.any():
        assert (got[:, zero_rows, :] == 0).all()


@pytest.mark.parametrize('width', [256, 1024])
@pytest.mark.parametrize('left', [False, True])
@pytest.mark.parametrize('auto', [False, True])
@pytest.mark.parametrize('masked,other,side', [(False, 0., False), (True, 0., False), (True, 1.5, False),
                                               (True, 0., True)])
def test_segmented_load_dot(width, left, auto, masked, other, side):
    if width == 1024 and left and side:
        pytest.skip('CANN 9.0 needs 1606144 UB bits for this combined graph; 1572864 are available')
    torch.manual_seed(20261107)
    x, w = torch.randn(1331200), torch.randn(16, 16)
    xn, wn = x.npu(), w.npu()
    side_guard, side_out = _guarded(16 * width)
    for programs in (1, 4):
        guard, out = _guarded(programs * 16 * width)
        for scale in (1., 1e-3, 1e3):
            value = x * scale
            xn.copy_(value.npu())
            out.fill_(float('nan'))
            compiled = _segmented_dot[(programs, )](xn, wn, out, side_out, width, left, masked, other, side,
                                                    num_warps=4, num_stages=1, multibuffer=False,
                                                    enable_fp_fusion=False, enable_auto_blockify=auto)
            raw = _raw(value, width, 15 if masked else 16, other)
            expected = raw.double().T @ w.double() if left else w.double() @ raw.double()
            _check(guard, out, expected.float(), programs)
            if side:
                assert torch.equal(side_out.cpu().reshape_as(raw), raw)
                assert (side_guard[:64].cpu() == 31415.25).all()
                assert (side_guard[-64:].cpu() == 31415.25).all()
            assert torch.equal(xn.cpu(), value)
            assert torch.equal(wn.cpu(), w)
        assert 'flagtree.vector_to_cube' in compiled.asm['ttadapter']


@pytest.mark.parametrize('width', [256, 1024])
def test_dynamic_segment_mask_and_graph_replay(width):
    torch.manual_seed(20261108)
    x, w = torch.randn(1331200), torch.randn(16, 16)
    xn, wn = x.npu(), w.npu()
    guard, out = _guarded(4 * 16 * width)
    for valid in (15, 7, 16):
        _dynamic_mask_dot[(4, )](xn, wn, out, valid, width, num_warps=4, num_stages=1, multibuffer=False,
                                 enable_fp_fusion=False, enable_auto_blockify=True)
        expected = (w.double() @ _raw(x, width, valid).double()).float()
        _check(guard, out, expected, 4)
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        with torch.npu.graph(graph, stream=stream):
            _dynamic_mask_dot[(4, )](xn, wn, out, 15, width, num_warps=4, num_stages=1, multibuffer=False,
                                     enable_fp_fusion=False, enable_auto_blockify=True)
    torch.npu.synchronize()
    for scale in (1e-3, 1e3, 1.):
        value = x * scale
        xn.copy_(value.npu())
        out.fill_(float('nan'))
        graph.replay()
        expected = (w.double() @ _raw(value, width, 15).double()).float()
        _check(guard, out, expected, 4)
        assert torch.equal(xn.cpu(), value)
