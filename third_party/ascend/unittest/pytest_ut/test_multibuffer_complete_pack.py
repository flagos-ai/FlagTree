"""Complete slice packs remain correct with real 910B double buffering."""
import math
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def split4(v, N: tl.constexpr, SPAN: tl.constexpr):
    G: tl.constexpr = N // SPAN
    Q: tl.constexpr = SPAN // 4
    p = tl.permute(tl.reshape(v, (G, 2, 2, Q)), 0, 3, 1, 2)
    # Flatten before split: keep both channels in compact one-dimensional
    # storage instead of creating a strided slice in a small middle axis.
    a, b = tl.split(tl.reshape(p, (N // 2, 2)))
    x0, x2 = tl.split(tl.reshape(a, (N // 4, 2)))
    x1, x3 = tl.split(tl.reshape(b, (N // 4, 2)))
    return (tl.reshape(x0, (N // 4, )), tl.reshape(x1, (N // 4, )), tl.reshape(x2,
                                                                               (N // 4, )), tl.reshape(x3, (N // 4, )))


@triton.jit
def merge4(a, b, c, d, N: tl.constexpr, SPAN: tl.constexpr):
    p = tl.join(tl.join(a, c), tl.join(b, d))
    p = tl.reshape(p, (N // SPAN, SPAN // 4, 2, 2))
    return tl.reshape(tl.permute(p, 0, 2, 3, 1), (N, ))


@triton.jit
def stage4(r, i, W, N: tl.constexpr, SPAN: tl.constexpr, S: tl.constexpr, SORT: tl.constexpr = False,
           SLICE: tl.constexpr = False, C: tl.constexpr = 1, KEEP: tl.constexpr = False):
    if SLICE:
        x0r, x1r, x2r, x3r = split4_slice(r, N, SPAN, KEEP)
        x0i, x1i, x2i, x3i = split4_slice(i, N, SPAN, KEEP)
    else:
        x0r, x1r, x2r, x3r = split4(r, N, SPAN)
        x0i, x1i, x2i, x3i = split4(i, N, SPAN)
    t0r, t0i = x0r + x2r, x0i + x2i
    t1r, t1i = x0r - x2r, x0i - x2i
    t2r, t2i = x1r + x3r, x1i + x3i
    t3r, t3i = x1r - x3r, x1i - x3i
    g0r, g0i = t0r + t2r, t0i + t2i
    a1r, a1i = t1r + t3i, t1i - t3r
    a2r, a2i = t0r - t2r, t0i - t2i
    a3r, a3i = t1r - t3i, t1i + t3r
    if SPAN > 4 * C:
        j = tl.arange(0, N // 4)
        w1r = tl.load(W + S * 6 * (N // 4) + j)
        w1i = tl.load(W + S * 6 * (N // 4) + N // 4 + j)
        w2r = tl.load(W + S * 6 * (N // 4) + 2 * (N // 4) + j)
        w2i = tl.load(W + S * 6 * (N // 4) + 3 * (N // 4) + j)
        w3r = tl.load(W + S * 6 * (N // 4) + 4 * (N // 4) + j)
        w3i = tl.load(W + S * 6 * (N // 4) + 5 * (N // 4) + j)
        if KEEP:
            w1r = tl.reshape(w1r, (N // SPAN, SPAN // 4))
            w1i = tl.reshape(w1i, (N // SPAN, SPAN // 4))
            w2r = tl.reshape(w2r, (N // SPAN, SPAN // 4))
            w2i = tl.reshape(w2i, (N // SPAN, SPAN // 4))
            w3r = tl.reshape(w3r, (N // SPAN, SPAN // 4))
            w3i = tl.reshape(w3i, (N // SPAN, SPAN // 4))
        g1r, g1i = a1r * w1r - a1i * w1i, a1r * w1i + a1i * w1r
        g2r, g2i = a2r * w2r - a2i * w2i, a2r * w2i + a2i * w2r
        g3r, g3i = a3r * w3r - a3i * w3i, a3r * w3i + a3i * w3r
    else:
        g1r, g1i, g2r, g2i, g3r, g3i = a1r, a1i, a2r, a2i, a3r, a3i
    if SORT:
        return pack4(g0r, g1r, g2r, g3r, N), pack4(g0i, g1i, g2i, g3i, N)
    return (merge4(g0r, g1r, g2r, g3r, N, SPAN), merge4(g0i, g1i, g2i, g3i, N, SPAN))


@triton.jit
def split4_slice(v, N: tl.constexpr, SPAN: tl.constexpr, KEEP: tl.constexpr):
    p = tl.reshape(v, (N // SPAN, SPAN))
    a = al.extract_slice(p, [0, 0], [N // SPAN, SPAN // 4], [1, 1])
    b = al.extract_slice(p, [0, SPAN // 4], [N // SPAN, SPAN // 4], [1, 1])
    c = al.extract_slice(p, [0, SPAN // 2], [N // SPAN, SPAN // 4], [1, 1])
    d = al.extract_slice(p, [0, 3 * SPAN // 4], [N // SPAN, SPAN // 4], [1, 1])
    if KEEP:
        return a, b, c, d
    else:
        return (tl.reshape(a, (N // 4, )), tl.reshape(b, (N // 4, )), tl.reshape(c,
                                                                                 (N // 4, )), tl.reshape(d, (N // 4, )))


@triton.jit
def pack4(a, b, c, d, N: tl.constexpr):
    a = tl.reshape(a, (N // 4, ))
    b = tl.reshape(b, (N // 4, ))
    c = tl.reshape(c, (N // 4, ))
    d = tl.reshape(d, (N // 4, ))
    v = tl.zeros((N, ), tl.float32)
    v = al.insert_slice(v, a, [0], [N // 4], [1])
    v = al.insert_slice(v, b, [N // 4], [N // 4], [1])
    v = al.insert_slice(v, c, [N // 2], [N // 4], [1])
    return al.insert_slice(v, d, [3 * N // 4], [N // 4], [1])


@triton.jit
def _npu_power_cmul(ar, ai, br, bi):
    return ar * br - ai * bi, ai * br + ar * bi


@triton.jit
def _npu_power_rad8(r0, r1, r2, r3, r4, r5, r6, r7, i0, i1, i2, i3, i4, i5, i6, i7):
    e02r = r0 + r4
    e02i = i0 + i4
    d02r = r0 - r4
    d02i = i0 - i4
    e13r = r2 + r6
    e13i = i2 + i6
    d13r = r2 - r6
    d13i = i2 - i6

    E0r = e02r + e13r
    E0i = e02i + e13i
    E2r = e02r - e13r
    E2i = e02i - e13i
    E1r = d02r + d13i
    E1i = d02i - d13r
    E3r = d02r - d13i
    E3i = d02i + d13r

    o02r = r1 + r5
    o02i = i1 + i5
    od02r = r1 - r5
    od02i = i1 - i5
    o13r = r3 + r7
    o13i = i3 + i7
    od13r = r3 - r7
    od13i = i3 - i7

    O0r = o02r + o13r
    O0i = o02i + o13i
    O2r = o02r - o13r
    O2i = o02i - o13i
    O1r = od02r + od13i
    O1i = od02i - od13r
    O3r = od02r - od13i
    O3i = od02i + od13r

    T0r = O0r
    T0i = O0i
    T1r, T1i = _npu_power_cmul(O1r, O1i, 0.7071067811865475, -0.7071067811865475)
    T2r, T2i = _npu_power_cmul(O2r, O2i, 0.0, -1.0)
    T3r, T3i = _npu_power_cmul(O3r, O3i, -0.7071067811865475, -0.7071067811865475)

    y0r = E0r + T0r
    y0i = E0i + T0i
    y4r = E0r - T0r
    y4i = E0i - T0i
    y1r = E1r + T1r
    y1i = E1i + T1i
    y5r = E1r - T1r
    y5i = E1i - T1i
    y2r = E2r + T2r
    y2i = E2i + T2i
    y6r = E2r - T2r
    y6i = E2i - T2i
    y3r = E3r + T3r
    y3i = E3i + T3i
    y7r = E3r - T3r
    y7i = E3i - T3i

    return (
        y0r,
        y1r,
        y2r,
        y3r,
        y4r,
        y5r,
        y6r,
        y7r,
        y0i,
        y1i,
        y2i,
        y3i,
        y4i,
        y5i,
        y6i,
        y7i,
    )


@triton.jit
def power_terminal8(r, i, N: tl.constexpr, C: tl.constexpr):
    re = tl.reshape(r, (N // (8 * C), 8 * C))
    im = tl.reshape(i, (N // (8 * C), 8 * C))
    r0 = tl.reshape(al.extract_slice(re, [0, 0 * C], [N // (8 * C), C], [1, 1]), (N // 8, ))
    i0 = tl.reshape(al.extract_slice(im, [0, 0 * C], [N // (8 * C), C], [1, 1]), (N // 8, ))
    r1 = tl.reshape(al.extract_slice(re, [0, 1 * C], [N // (8 * C), C], [1, 1]), (N // 8, ))
    i1 = tl.reshape(al.extract_slice(im, [0, 1 * C], [N // (8 * C), C], [1, 1]), (N // 8, ))
    r2 = tl.reshape(al.extract_slice(re, [0, 2 * C], [N // (8 * C), C], [1, 1]), (N // 8, ))
    i2 = tl.reshape(al.extract_slice(im, [0, 2 * C], [N // (8 * C), C], [1, 1]), (N // 8, ))
    r3 = tl.reshape(al.extract_slice(re, [0, 3 * C], [N // (8 * C), C], [1, 1]), (N // 8, ))
    i3 = tl.reshape(al.extract_slice(im, [0, 3 * C], [N // (8 * C), C], [1, 1]), (N // 8, ))
    r4 = tl.reshape(al.extract_slice(re, [0, 4 * C], [N // (8 * C), C], [1, 1]), (N // 8, ))
    i4 = tl.reshape(al.extract_slice(im, [0, 4 * C], [N // (8 * C), C], [1, 1]), (N // 8, ))
    r5 = tl.reshape(al.extract_slice(re, [0, 5 * C], [N // (8 * C), C], [1, 1]), (N // 8, ))
    i5 = tl.reshape(al.extract_slice(im, [0, 5 * C], [N // (8 * C), C], [1, 1]), (N // 8, ))
    r6 = tl.reshape(al.extract_slice(re, [0, 6 * C], [N // (8 * C), C], [1, 1]), (N // 8, ))
    i6 = tl.reshape(al.extract_slice(im, [0, 6 * C], [N // (8 * C), C], [1, 1]), (N // 8, ))
    r7 = tl.reshape(al.extract_slice(re, [0, 7 * C], [N // (8 * C), C], [1, 1]), (N // 8, ))
    i7 = tl.reshape(al.extract_slice(im, [0, 7 * C], [N // (8 * C), C], [1, 1]), (N // 8, ))
    r0, r1, r2, r3, r4, r5, r6, r7, i0, i1, i2, i3, i4, i5, i6, i7 = _npu_power_rad8(
        r0, r1, r2, r3, r4, r5, r6, r7, i0, i1, i2, i3, i4, i5, i6, i7)
    re = tl.zeros((N, ), tl.float32)
    im = tl.zeros((N, ), tl.float32)
    re = al.insert_slice(re, r0, [0 * (N // 8)], [N // 8], [1])
    im = al.insert_slice(im, i0, [0 * (N // 8)], [N // 8], [1])
    re = al.insert_slice(re, r1, [1 * (N // 8)], [N // 8], [1])
    im = al.insert_slice(im, i1, [1 * (N // 8)], [N // 8], [1])
    re = al.insert_slice(re, r2, [2 * (N // 8)], [N // 8], [1])
    im = al.insert_slice(im, i2, [2 * (N // 8)], [N // 8], [1])
    re = al.insert_slice(re, r3, [3 * (N // 8)], [N // 8], [1])
    im = al.insert_slice(im, i3, [3 * (N // 8)], [N // 8], [1])
    re = al.insert_slice(re, r4, [4 * (N // 8)], [N // 8], [1])
    im = al.insert_slice(im, i4, [4 * (N // 8)], [N // 8], [1])
    re = al.insert_slice(re, r5, [5 * (N // 8)], [N // 8], [1])
    im = al.insert_slice(im, i5, [5 * (N // 8)], [N // 8], [1])
    re = al.insert_slice(re, r6, [6 * (N // 8)], [N // 8], [1])
    im = al.insert_slice(im, i6, [6 * (N // 8)], [N // 8], [1])
    re = al.insert_slice(re, r7, [7 * (N // 8)], [N // 8], [1])
    im = al.insert_slice(im, i7, [7 * (N // 8)], [N // 8], [1])
    return re, im


@triton.jit
def _pipeline_pack(X, TW, Y, C: tl.constexpr):
    batch = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, 32)[:, None]
    q = tl.program_id(0) * C + tl.arange(0, C)[None, :]
    j = tl.program_id(0) // (1024 // C)
    pf = (tl.program_id(0) % (1024 // C) * C + tl.arange(0, C))[None, :]
    off = d * 16384 + q
    rr = tl.load(X + batch * 1049600 + off)
    ii = tl.load(X + batch * 1049600 + 524800 + off)
    wr = tl.load(TW + off)
    wi = tl.load(TW + 524288 + off)
    re = tl.reshape(rr * wr - ii * wi, (32 * C, ))
    im = tl.reshape(rr * wi + ii * wr, (32 * C, ))
    r, i = stage4(re, im, TW + 1048576, 32 * C, 32 * C, 0, True, True, C)
    r, i = power_terminal8(r, i, 32 * C, C)
    dst = (j * 32 + d) * 1024 + pf
    tl.store(Y + batch * 1049600 + dst, tl.reshape(r, (32, C)))
    tl.store(Y + batch * 1049600 + 524800 + dst, tl.reshape(i, (32, C)))


@pytest.mark.parametrize("columns", [64, 128])
@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("batch", [1, 3, 64])
def test_multibuffer_complete_pack(columns, stages, batch):
    n, physical, guard = 524288, 524800, 128
    sample = torch.arange(512, dtype=torch.float64)[:, None]
    digit = (torch.arange(1024) // 32)[None, :]
    angle = -2 * math.pi * sample * digit / 16384
    j = torch.arange(8 * columns) // columns
    rows = []
    for branch in (1, 2, 3):
        theta = -2 * math.pi * branch * j.double() / 32
        rows.extend((theta.cos().float(), theta.sin().float()))
    weights = torch.cat((angle.cos().float().flatten(), angle.sin().float().flatten(), *rows))
    tw = weights.npu()
    phase = torch.complex(weights[:n].double(), weights[n:2 * n].double()).reshape(32, 16, 1024)
    torch.manual_seed(111102)
    initial = torch.randn(batch, 2, physical)
    x = initial.npu()
    guarded = torch.full((2 * batch * physical + 2 * guard, ), 31415.25, device="npu")
    out = guarded[guard:-guard]

    def launch():
        return _pipeline_pack[(16384 // columns, batch)](x, tw, out, columns, num_warps=4, num_stages=stages,
                                                         multibuffer=stages == 2, enable_auto_blockify=True,
                                                         enable_fp_fusion=True)

    for scale in (1., 0.001):
        source = initial * scale
        x.copy_(source)
        z = torch.complex(source[:, 0, :n].double(), source[:, 1, :n].double())
        expected = torch.fft.fft(z.reshape(batch, 32, 16, 1024) * phase, dim=1)
        expected = expected.permute(0, 2, 1, 3).reshape(batch, n)
        out.fill_(31415.25)
        compiled = launch()
        torch.npu.synchronize()
        if stages == 2:
            assert compiled.metadata.num_stages == 2 and compiled.metadata.multibuffer
            assert "tensor.concat" in compiled.asm["ttadapter"]
        for replay in (False, True):
            if replay:
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    launch()
                out.fill_(31415.25)
                graph.replay()
                torch.npu.synchronize()
            result = out.cpu().reshape(batch, 2, physical)
            got = torch.complex(result[:, 0, :n].double(), result[:, 1, :n].double())
            error = torch.linalg.vector_norm(got - expected) / torch.linalg.vector_norm(expected)
            assert error < 5e-7, float(error)
            assert torch.isfinite(got).all()
            assert (result[:, :, n:] == 31415.25).all()
            assert (guarded[:guard].cpu() == 31415.25).all()
            assert (guarded[-guard:].cpu() == 31415.25).all()
            assert torch.equal(x.cpu(), source)
            assert torch.equal(tw.cpu(), weights)
