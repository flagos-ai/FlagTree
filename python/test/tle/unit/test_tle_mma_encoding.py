"""The Hopper instruction tile has M/N/K dimensions independently of tensor rank."""
import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C import libtriton
from triton._filecheck import run_parser
from triton.backends.compiler import GPUTarget


@triton.jit
def _hopper_scale_layout_kernel(layout: tl.constexpr):
    values = tl.arange(0, 64).to(tl.float32)
    tle.gpu.set_layout(values, layout)


@pytest.mark.parametrize("warps", ([4, 1], [8, 1], [1, 4, 1]))
def test_hopper_mma_instruction_shape_has_three_dimensions(warps):
    encoding = tle.gpu.MmaEncoding([3, 0], warps, [16, 8, 32])
    assert encoding.rank == len(warps)
    assert encoding.instr_shape == [16, 8, 32]
    assert encoding.warps_per_cta == list(warps)


@pytest.mark.parametrize("shape", ([16, 8], [16, 8, 32, 1]))
def test_hopper_mma_rejects_non_mnk_instruction_shapes(shape):
    with pytest.raises(ValueError, match="three M/N/K dimensions"):
        tle.gpu.MmaEncoding([3, 0], [4, 1], shape)


def test_ampere_mma_retains_tensor_rank_validation():
    assert tle.gpu.MmaEncoding([2, 0], [4, 1], [16, 8]).rank == 2
    with pytest.raises(ValueError, match="rank must match warps_per_cta"):
        tle.gpu.MmaEncoding([2, 0], [4, 1], [16, 8, 32])


def test_hopper_mma_cga_basis_uses_tensor_rank():
    encoding = tle.gpu.MmaEncoding([3, 0], [4, 1], [16, 8, 32], [[1, 0]])
    assert encoding.cga_layout == [[1, 0]]
    with pytest.raises(ValueError, match="cga_layout basis rank mismatch"):
        tle.gpu.MmaEncoding([3, 0], [4, 1], [16, 8, 32], [[1, 0, 0]])


@pytest.mark.skipif(not hasattr(libtriton.ir.builder, "ensure_ttg_layout_attrs"),
                    reason="requires the TLE explicit layout builder")
def test_hopper_mma_slice_encoding_frontend():
    parent = tle.gpu.MmaEncoding([3, 0], [8, 1], [16, 8, 32])
    layout = tle.gpu.SlicedEncoding(1, parent)
    module = run_parser(_hopper_scale_layout_kernel,
                        kwargs={"layout": layout, "num_warps": 8},
                        target=GPUTarget("cuda", 90, 32))
    text = module.str_nodebug()
    assert "versionMajor = 3" in text
    assert "instrShape = [16, 8, 32]" in text
    assert "#ttg.slice<{dim = 1" in text
    assert "tle.gpu.set_layout" in text
