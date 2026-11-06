import pytest
import torch

import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource


@gluon.jit
def _fp4_decode_1d(X, Y, N: gl.constexpr, DTYPE: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    i = gl.arange(0, 512, layout=layout)
    x = gl.load(X + i, i < N, other=0)
    y = gl.fp4_to_fp(x, DTYPE, 0)
    j = gl.arange(0, 1024, layout=y.type.layout)
    gl.store(Y + j, y, j < 2 * N)


@gluon.jit
def _fp4_decode_2d(X, Y, AXIS: gl.constexpr, SIZE: gl.constexpr, DTYPE: gl.constexpr):
    if AXIS == 1:
        layout: gl.constexpr = gl.BlockedLayout([1, SIZE], [4, 16], [4, 1], [1, 0])
    else:
        layout: gl.constexpr = gl.BlockedLayout([SIZE, 1], [16, 4], [1, 4], [0, 1])
    i = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
    j = gl.arange(0, 16, layout=gl.SliceLayout(0, layout))
    x = gl.load(X + i[:, None] * 16 + j[None, :])
    y = gl.fp4_to_fp(x, DTYPE, AXIS)
    out_layout: gl.constexpr = y.type.layout
    if AXIS == 1:
        a = gl.arange(0, 16, layout=gl.SliceLayout(1, out_layout))
        b = gl.arange(0, 32, layout=gl.SliceLayout(0, out_layout))
        gl.store(Y + a[:, None] * 32 + b[None, :], y)
    else:
        a = gl.arange(0, 32, layout=gl.SliceLayout(1, out_layout))
        b = gl.arange(0, 16, layout=gl.SliceLayout(0, out_layout))
        gl.store(Y + a[:, None] * 16 + b[None, :], y)


@pytest.mark.parametrize("dtype,out_dtype", [(gl.float16, torch.float16), (gl.bfloat16, torch.bfloat16),
                                             (gl.float16, torch.float32), (gl.bfloat16, torch.float32),
                                             (gl.float16, torch.float64), (gl.bfloat16, torch.float64)])
@pytest.mark.parametrize("signed", [False, True])
def test_metax_fp4_to_fp_1d(dtype, out_dtype, signed, metax_device):
    raw = torch.arange(259, dtype=torch.int64).remainder(256).to(torch.uint8)
    x = raw.to(metax_device)
    if signed:
        x = x.view(torch.int8)
    output = torch.full((582, ), 11., dtype=out_dtype, device=metax_device)
    kernel = _fp4_decode_1d[(1, )](x, output[32:550], 259, dtype)
    # The table is independent of the lowering and retains both zero signs.
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=out_dtype)
    codes = torch.stack((raw & 15, raw >> 4), dim=1).flatten().long()
    expected = torch.full((582, ), 11., dtype=out_dtype)
    expected[32:550] = table[codes]
    assert torch.equal(output.cpu().view(torch.uint8), expected.view(torch.uint8))
    assert "fp4_to_fp" not in kernel.asm["llir"]
    assert "llvm.mxc.cvt.pk4.f4" not in kernel.asm["llir"]


@pytest.mark.parametrize("dtype", [gl.float16, gl.bfloat16])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("size", [1, 4])
@pytest.mark.parametrize("signed", [False, True])
def test_metax_fp4_to_fp_2d(dtype, axis, size, signed, metax_device):
    raw = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    x = raw.to(metax_device)
    if signed:
        x = x.view(torch.int8)
    out_dtype = torch.float16 if dtype == gl.float16 else torch.bfloat16
    output = torch.full((576, ), 11., dtype=out_dtype, device=metax_device)
    _fp4_decode_2d[(1, )](x, output[32:544], axis, size, dtype)
    codes = torch.stack((raw & 15, raw >> 4), dim=axis + 1).flatten(axis, axis + 1).long()
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=out_dtype)
    expected = torch.full((576, ), 11., dtype=out_dtype)
    expected[32:544] = table[codes].flatten()
    assert torch.equal(output.cpu().view(torch.uint8), expected.view(torch.uint8))


def test_metax_fp4_to_fp_diagnostic(fresh_triton_cache, monkeypatch, capfd):
    monkeypatch.setenv("MLIR_ENABLE_DIAGNOSTICS", "warnings")
    source = GluonASTSource(_fp4_decode_1d, {"X": "*u8", "Y": "*fp16"}, {"N": 259, "DTYPE": gl.float16})
    kernel = triton.compile(source, target=GPUTarget("maca", 80, 64), options={"num_warps": 4})
    assert "using non-native FP4 conversion on MetaX" in capfd.readouterr().err
    assert "llvm.mxc.cvt.pk4.f4" not in kernel.asm["llir"]


@pytest.mark.parametrize("dtype", [gl.float32, gl.float64])
def test_metax_fp4_to_fp_invalid_output(dtype, capfd):
    source = GluonASTSource(_fp4_decode_1d, {"X": "*u8", "Y": "*fp32"}, {"N": 259, "DTYPE": dtype})
    with pytest.raises(RuntimeError, match="error encountered during parsing"):
        triton.compile(source, target=GPUTarget("maca", 80, 64), options={"num_warps": 4})
    assert "only bf16 or f16 is supported" in capfd.readouterr().err
