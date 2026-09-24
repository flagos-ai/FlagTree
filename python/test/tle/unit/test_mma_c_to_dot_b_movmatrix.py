import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@pytest.fixture(scope="module", autouse=True)
def _require_nvidia_gpu():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "cuda" or int(target.arch) < 80:
        pytest.skip("requires NVIDIA Ampere or newer")


@gluon.jit
def _mma_dot_b_roundtrip(x, out, ROWS: gl.constexpr, COLS: gl.constexpr, LAYOUT: gl.constexpr):
    rows = gl.arange(0, ROWS, layout=gl.SliceLayout(1, LAYOUT))
    cols = gl.arange(0, COLS, layout=gl.SliceLayout(0, LAYOUT))
    offsets = rows[:, None] * COLS + cols[None, :]
    values = gl.load(x + offsets)
    # Side effects keep the compiler from folding the two layout conversions
    # together or moving them across the loads and stores.
    values = gl.inline_asm_elementwise("mov.b16 $0, $1;", "=h,h", [values], dtype=values.dtype, is_pure=False, pack=1)
    dot_b = gl.convert_layout(values, gl.DotOperandLayout(operand_index=1, parent=LAYOUT, k_width=2))
    dot_b = gl.inline_asm_elementwise("mov.b16 $0, $1;", "=h,h", [dot_b], dtype=dot_b.dtype, is_pure=False, pack=1)
    result = gl.convert_layout(dot_b, LAYOUT)
    gl.store(out + offsets, result)


@pytest.mark.parametrize(
    "shape,warps,dtype,uses_movmatrix",
    [
        ((16, 16), (1, 4), torch.bfloat16, True),
        ((16, 64), (1, 4), torch.bfloat16, True),
        ((64, 16), (1, 4), torch.bfloat16, True),
        ((128, 128), (1, 4), torch.bfloat16, True),
        ((128, 128), (1, 1), torch.bfloat16, True),
        ((32, 128), (1, 8), torch.bfloat16, True),
        ((16, 32), (2, 2), torch.bfloat16, True),
        ((32, 64), (2, 2), torch.bfloat16, False),
        ((64, 16), (4, 1), torch.bfloat16, False),
        ((16, 16), (1, 4), torch.float16, True),
        ((128, 128), (1, 4), torch.float16, True),
        ((8, 16), (1, 4), torch.bfloat16, False),
    ],
)
def test_mma_c_to_dot_b_preserves_values(shape, warps, dtype, uses_movmatrix):
    rows, cols = shape
    # Unique finite 16-bit patterns catch incorrect register or lane permutations
    # without rounding many logical positions to the same value.
    bits = (torch.arange(rows * cols, device="cuda", dtype=torch.int32) + 0x2000).to(torch.int16)
    x = bits.view(dtype).reshape(shape)
    out = torch.empty_like(x)
    layout = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=list(warps), instr_shape=[16, 8])
    compiled = _mma_dot_b_roundtrip[(1, )](x, out, rows, cols, layout, num_warps=warps[0] * warps[1])
    torch.testing.assert_close(out.view(torch.int16), x.view(torch.int16), rtol=0, atol=0)
    assert ("movmatrix.sync.aligned.m8n8.trans.b16" in compiled.asm["ptx"]) == uses_movmatrix
