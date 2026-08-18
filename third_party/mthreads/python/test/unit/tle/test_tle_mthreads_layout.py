import os
import re
import subprocess
import sys

import pytest
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.experimental.tle.language.gpu.mthreads.layout import (
    MusaDotOperandEncoding,
    MusaSqmmaEncoding,
    MusaWmmaEncoding,
)


class _FakeBuilder:

    def __init__(self):
        self.calls = []

    def get_musa_wmma_layout(self, version, warps_per_cta, cga_layout, instr_shape):
        self.calls.append(("wmma", version, warps_per_cta, cga_layout, instr_shape))
        return "wmma_attr"

    def get_musa_sqmma_layout(self, version, warps_per_cta, cga_layout, instr_shape):
        self.calls.append(("sqmma", version, warps_per_cta, cga_layout, instr_shape))
        return "sqmma_attr"

    def get_mma_layout(self, version, warps_per_cta, cga_layout, instr_shape):
        self.calls.append(("nvidia", version, warps_per_cta, cga_layout, instr_shape))
        return "nvidia_attr"

    def get_dot_operand_layout(self, operand_index, parent, k_width):
        self.calls.append(("dot_operand", operand_index, parent, k_width))
        return "dot_operand_attr"


def test_mthreads_layout_is_available_only_from_explicit_submodule():
    assert MusaWmmaEncoding is tle.gpu.mthreads.layout.MusaWmmaEncoding
    assert MusaSqmmaEncoding is tle.gpu.mthreads.layout.MusaSqmmaEncoding
    assert MusaDotOperandEncoding is tle.gpu.mthreads.layout.MusaDotOperandEncoding
    assert not hasattr(tle.gpu, "MusaWmmaEncoding")
    assert not hasattr(tle.gpu, "MusaSqmmaEncoding")
    assert not hasattr(tle.gpu, "MusaDotOperandEncoding")
    assert not hasattr(tle.gpu.mthreads, "MusaWmmaEncoding")
    assert not hasattr(tle.gpu.mthreads, "MusaSqmmaEncoding")
    assert not hasattr(tle.gpu.mthreads, "MusaDotOperandEncoding")


def test_generic_gpu_import_does_not_load_mthreads_layout():
    env = os.environ.copy()
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    script = ("import sys; "
              "import triton.experimental.tle.language as tle; "
              "assert 'triton.experimental.tle.language.gpu.mthreads.layout' not in sys.modules")
    subprocess.run([sys.executable, "-c", script], env=env, check=True, capture_output=True, text=True)


def test_explicit_mthreads_layout_import_does_not_load_torch_musa():
    env = os.environ.copy()
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    script = ("import sys; "
              "from triton.experimental.tle.language.gpu.mthreads.layout import MusaWmmaEncoding; "
              "assert MusaWmmaEncoding; "
              "assert 'torch_musa' not in sys.modules")
    subprocess.run([sys.executable, "-c", script], env=env, check=True, capture_output=True, text=True)


@pytest.mark.parametrize(
    "encoding_type,warps_per_cta,instr_shape,expected_kind",
    [
        (MusaWmmaEncoding, (4, 1), (16, 16, 16), "wmma"),
        (MusaSqmmaEncoding, (4, 1), (64, 64, 32), "sqmma"),
        (MusaWmmaEncoding, (2, 2, 1), (16, 8, 8), "wmma"),
        (MusaSqmmaEncoding, (4, 2, 1), (32, 64, 16), "sqmma"),
    ],
)
def test_mthreads_layout_construction_and_builder_contract(encoding_type, warps_per_cta, instr_shape, expected_kind):
    encoding = encoding_type(
        tl.tuple((tl.constexpr(3), tl.constexpr(1))),
        tl.tuple(tuple(tl.constexpr(value) for value in warps_per_cta)),
        tl.tuple(tuple(tl.constexpr(value) for value in instr_shape)),
    )
    assert encoding.version == [3, 1]
    assert encoding.warps_per_cta == list(warps_per_cta)
    assert encoding.instr_shape == list(instr_shape)
    assert encoding.cga_layout == []
    assert encoding.rank == len(warps_per_cta)

    builder = _FakeBuilder()
    assert encoding.to_ir(builder) == f"{expected_kind}_attr"
    assert builder.calls == [(expected_kind, [3, 1], list(warps_per_cta), [], list(instr_shape))]


def test_mthreads_layout_cga_repr_equality_and_hash():
    lhs = MusaWmmaEncoding([3, 1], [2, 2, 1], [16, 16, 16], [[0, 1, 0]])
    rhs = MusaWmmaEncoding((3, 1), (2, 2, 1), (16, 16, 16), ((0, 1, 0), ))
    other = MusaSqmmaEncoding([3, 1], [4, 1, 1], [16, 16, 16], [[0, 1, 0]])
    assert lhs == rhs
    assert hash(lhs) == hash(rhs)
    assert lhs != other
    assert repr(lhs) == ("MusaWmmaEncoding(version=[3, 1], warps_per_cta=[2, 2, 1], "
                         "instr_shape=[16, 16, 16], cga_layout=[[0, 1, 0]])")


@pytest.mark.parametrize(
    "encoding_type,args,error",
    [
        (MusaWmmaEncoding, ([3], [4, 1], [16, 16, 16]), "must contain major and minor"),
        (MusaWmmaEncoding, ([3, 0], [4, 1], [16, 16, 16]), "must be positive"),
        (MusaWmmaEncoding, ([3, 2], [4, 1], [16, 16, 16]), "only MUSA PH1 version"),
        (MusaWmmaEncoding, ([3, True], [4, 1], [16, 16, 16]), "compile-time integer"),
        (MusaWmmaEncoding, ([3, 1], [4], [16, 16, 16]), "rank must be 2 or 3"),
        (MusaWmmaEncoding, ([3, 1], [3, 1], [16, 16, 16]), "positive powers of two"),
        (MusaWmmaEncoding, ([3, 1], [2, 2, 2], [16, 16, 16]), "must be [warps_m, warps_n, 1]"),
        (MusaWmmaEncoding, ([3, 1], [4, 1], [16, 16]), "logical (M, N, K)"),
        (MusaWmmaEncoding, ([3, 1], [4, 1], [12, 16, 16]), "M must be"),
        (MusaWmmaEncoding, ([3, 1], [4, 1], [16, 12, 16]), "N must be"),
        (MusaWmmaEncoding, ([3, 1], [4, 1], [16, 16, 6]), "K must be"),
        (MusaSqmmaEncoding, ([3, 1], [2, 2], [16, 16, 16]), "multiple of 4"),
        (MusaSqmmaEncoding, ([3, 1], [4, 1], [8, 16, 16]), "M must be at least 16"),
        (MusaSqmmaEncoding, ([3, 1], [4, 1], [16, 12, 16]), "N must be"),
        (MusaSqmmaEncoding, ([3, 1], [4, 1], [16, 16, 4]), "K must be"),
    ],
)
def test_mthreads_layout_rejects_invalid_mma_configuration(encoding_type, args, error):
    with pytest.raises(ValueError, match=re.escape(error)):
        encoding_type(*args)


@pytest.mark.parametrize(
    "cga_layout,error",
    [
        (1, "static sequence of basis vectors"),
        ([[0]], "rank must match layout rank 2"),
        ([[0, -1]], "entries must be non-negative"),
        ([[0, 1.0]], "compile-time integer"),
    ],
)
def test_mthreads_layout_rejects_invalid_cga_layout(cga_layout, error):
    with pytest.raises(ValueError, match=re.escape(error)):
        MusaWmmaEncoding([3, 1], [4, 1], [16, 16, 16], cga_layout)


def test_nvidia_mma_encoding_builder_contract_is_unchanged():
    encoding = tle.gpu.MmaEncoding([2, 0], [4, 1], [16, 8])
    builder = _FakeBuilder()
    assert encoding.to_ir(builder) == "nvidia_attr"
    assert builder.calls == [("nvidia", [2, 0], [4, 1], [], [16, 8])]


@pytest.mark.parametrize("parent_type", [MusaWmmaEncoding, MusaSqmmaEncoding])
def test_mthreads_dot_operand_encoding_accepts_musa_parent_and_zero_k_width(parent_type):
    parent = parent_type([3, 1], [4, 1], [16, 16, 16])
    encoding = MusaDotOperandEncoding(0, parent)
    builder = _FakeBuilder()

    assert encoding.to_ir(builder) == "dot_operand_attr"
    assert builder.calls[-1] == ("dot_operand", 0,
                                 f"{parent_type.__name__.removeprefix('Musa').removesuffix('Encoding').lower()}_attr",
                                 0)


def test_mthreads_dot_operand_encoding_rejects_invalid_contracts():
    parent = MusaWmmaEncoding([3, 1], [4, 1], [16, 16, 16])
    nvidia_parent = tle.gpu.MmaEncoding([2, 0], [4, 1], [16, 8])

    with pytest.raises(ValueError, match="operand_index must be 0 or 1"):
        MusaDotOperandEncoding(2, parent)
    with pytest.raises(ValueError, match="parent must be a MusaWmmaEncoding or MusaSqmmaEncoding"):
        MusaDotOperandEncoding(0, nvidia_parent)
    with pytest.raises(ValueError, match="requires k_width=0"):
        MusaDotOperandEncoding(0, parent, 1)


def test_nvidia_dot_operand_k_width_contract_is_unchanged():
    parent = tle.gpu.MmaEncoding([2, 0], [4, 1], [16, 8])
    assert tle.gpu.DotOperandEncoding(0, parent, 2).k_width == 2
    with pytest.raises(ValueError, match="DotOperandEncoding k_width must be positive"):
        tle.gpu.DotOperandEncoding(0, parent, 0)


def test_generic_dot_operand_does_not_absorb_mthreads_k_width_contract():
    parent = MusaWmmaEncoding([3, 1], [4, 1], [16, 16, 16])
    with pytest.raises(ValueError, match="DotOperandEncoding k_width must be positive"):
        tle.gpu.DotOperandEncoding(0, parent, 0)
