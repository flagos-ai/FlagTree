# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import test_common
from unittest.mock import MagicMock
from triton.language.semantic import TritonSemantic


@pytest.fixture(scope="function")
def restore_npu_hf32_setting():
    original_allow_hf32 = torch_npu.npu.matmul.allow_hf32
    try:
        torch_npu.npu.matmul.allow_hf32 = True
        yield
    finally:
        torch_npu.npu.matmul.allow_hf32 = original_allow_hf32


def torch_dot_None(x0, x1):
    res = torch.matmul(x0, x1)
    return res


@triton.jit
def triton_dot_2_None(output_ptr, x_ptr, y_ptr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)

    x_mask = (bidx[:, None] < B) & (cidx[None, :] < C)
    y_mask = (cidx[:, None] < C) & (didx[None, :] < D)
    out_mask = (bidx[:, None] < B) & (didx[None, :] < D)
    Xidx = bidx[:, None] * C + cidx[None, :]
    Yidx = cidx[:, None] * D + didx[None, :]
    X = tl.load(x_ptr + Xidx, mask=x_mask, other=0.0)
    Y = tl.load(y_ptr + Yidx, mask=y_mask, other=0.0)
    ret = tl.dot(X, Y, input_precision="hf32")
    oidx = bidx[:, None] * D + didx[None, :]
    tl.store(output_ptr + oidx, ret, mask=out_mask)


@triton.jit
def triton_dot_2_allow_tf32(output_ptr, x_ptr, y_ptr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)

    x_mask = (bidx[:, None] < B) & (cidx[None, :] < C)
    y_mask = (cidx[:, None] < C) & (didx[None, :] < D)
    out_mask = (bidx[:, None] < B) & (didx[None, :] < D)
    Xidx = bidx[:, None] * C + cidx[None, :]
    Yidx = cidx[:, None] * D + didx[None, :]
    X = tl.load(x_ptr + Xidx, mask=x_mask, other=0.0)
    Y = tl.load(y_ptr + Yidx, mask=y_mask, other=0.0)
    # when allow_tf32=True; input_precision=tf32
    # After removing invasive modifications,input_precision needs to be passed.
    ret = tl.dot(X, Y, input_precision="tf32")
    oidx = bidx[:, None] * D + didx[None, :]
    tl.store(output_ptr + oidx, ret, mask=out_mask)


@triton.jit
def triton_dot_2_input_tf32(output_ptr, x_ptr, y_ptr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)

    x_mask = (bidx[:, None] < B) & (cidx[None, :] < C)
    y_mask = (cidx[:, None] < C) & (didx[None, :] < D)
    out_mask = (bidx[:, None] < B) & (didx[None, :] < D)
    Xidx = bidx[:, None] * C + cidx[None, :]
    Yidx = cidx[:, None] * D + didx[None, :]
    X = tl.load(x_ptr + Xidx, mask=x_mask, other=0.0)
    Y = tl.load(y_ptr + Yidx, mask=y_mask, other=0.0)
    ret = tl.dot(X, Y, input_precision="tf32")
    oidx = bidx[:, None] * D + didx[None, :]
    tl.store(output_ptr + oidx, ret, mask=out_mask)


@triton.jit
def triton_dot_2_ignore_tf32(output_ptr, x_ptr, y_ptr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)

    x_mask = (bidx[:, None] < B) & (cidx[None, :] < C)
    y_mask = (cidx[:, None] < C) & (didx[None, :] < D)
    out_mask = (bidx[:, None] < B) & (didx[None, :] < D)
    Xidx = bidx[:, None] * C + cidx[None, :]
    Yidx = cidx[:, None] * D + didx[None, :]
    X = tl.load(x_ptr + Xidx, mask=x_mask, other=0.0)
    Y = tl.load(y_ptr + Yidx, mask=y_mask, other=0.0)
    ret = tl.dot(X, Y, input_precision="hf32")
    oidx = bidx[:, None] * D + didx[None, :]
    tl.store(output_ptr + oidx, ret, mask=out_mask)


testlist1 = [
    (10, 13, 35, 39),
]

testlist2 = [(16, 32, 16)]

typelist = [
    'float32',
]


@pytest.mark.parametrize("B, C, D", testlist2)
@pytest.mark.parametrize("sigtype", typelist)
def test_dot_2(restore_npu_hf32_setting, sigtype, B, C, D):
    x = test_common.generate_tensor((B, C), sigtype).npu()
    y = test_common.generate_tensor((C, D), sigtype).npu()
    z_ref = torch_dot_None(x, y).to(torch.float32)
    z = torch.zeros((B, D), dtype=torch.float32).npu()
    triton_dot_2_None[1, 1, 1](z, x, y, B, C, D)
    torch.testing.assert_close(z, z_ref, rtol=3e-02, atol=3e-02, equal_nan=True)


@pytest.mark.parametrize("B, C, D", testlist2)
@pytest.mark.parametrize("sigtype", typelist)
def test_dot_2_allow_tf32(restore_npu_hf32_setting, sigtype, B, C, D):
    x = test_common.generate_tensor((B, C), sigtype).npu()
    y = test_common.generate_tensor((C, D), sigtype).npu()
    z_ref = torch_dot_None(x, y).to(torch.float32)
    z = torch.zeros((B, D), dtype=torch.float32).npu()
    triton_dot_2_allow_tf32[1, 1, 1](z, x, y, B, C, D)
    torch.testing.assert_close(z, z_ref, rtol=3e-02, atol=3e-02, equal_nan=True)


@pytest.mark.parametrize("B, C, D", testlist2)
@pytest.mark.parametrize("sigtype", typelist)
def test_dot_2_input_tf32(restore_npu_hf32_setting, sigtype, B, C, D):
    x = test_common.generate_tensor((B, C), sigtype).npu()
    y = test_common.generate_tensor((C, D), sigtype).npu()
    z_ref = torch_dot_None(x, y).to(torch.float32)
    z = torch.zeros((B, D), dtype=torch.float32).npu()
    triton_dot_2_input_tf32[1, 1, 1](z, x, y, B, C, D)
    torch.testing.assert_close(z, z_ref, rtol=3e-02, atol=3e-02, equal_nan=True)


@pytest.mark.parametrize("B, C, D", testlist2)
@pytest.mark.parametrize("sigtype", typelist)
def test_dot_2_ignore_tf32(sigtype, B, C, D):
    input_type = "bfloat16"
    x = test_common.generate_tensor((B, C), input_type).npu()
    y = test_common.generate_tensor((C, D), input_type).npu()
    z = torch.zeros((B, D), dtype=torch.float32).npu()

    original_allow_hf32 = torch_npu.npu.matmul.allow_hf32
    try:
        torch_npu.npu.matmul.allow_hf32 = False
        z_ref = torch_dot_None(x.to(torch.float32), y.to(torch.float32)).to(torch.float32)

    finally:
        torch_npu.npu.matmul.allow_hf32 = original_allow_hf32

    triton_dot_2_ignore_tf32[1, 1, 1](z, x, y, B, C, D)
    torch.testing.assert_close(z, z_ref, rtol=3e-02, atol=3e-02, equal_nan=True)


# =============================================================================
# Unit tests for the dot monkey-patch (third_party/ascend/backend/__init__.py)
# =============================================================================


def _make_mock_builder():
    builder = MagicMock()
    builder.options.default_dot_input_precision = "ieee"
    builder.options.allowed_dot_input_precisions = ("ieee", "hf32")
    builder.options.max_num_imprecise_acc_default = 0
    builder.codegen_fns = {"min_dot_size": MagicMock(return_value=(1, 1, 1))}
    builder.create_dot = MagicMock(return_value=MagicMock())
    builder.create_splat = MagicMock(return_value=MagicMock())
    builder.get_fp32 = MagicMock(return_value=MagicMock())
    builder.get_fp16 = MagicMock(return_value=MagicMock())
    builder.get_fp64 = MagicMock(return_value=MagicMock())
    builder.get_int32 = MagicMock(return_value=MagicMock())
    return builder


def _mock_tensor(dtype):
    t = MagicMock()
    t.dtype = dtype
    t.handle = MagicMock()
    dim = MagicMock()
    dim.value = 64
    t.shape = [dim, dim]
    type_mock = MagicMock()
    type_mock.is_block = MagicMock(return_value=True)
    type_mock.scalar = dtype
    type_mock.shape = [64, 64]
    t.type = type_mock
    return t


class TestDotAscendPatch:
    """Verify the monkey-patched TritonSemantic.dot is behaviourally
    equivalent to the original intrusive code."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        if getattr(TritonSemantic, "_ascend_dot_patch_applied", False):
            return
        try:
            from triton.backends.ascend import _apply_ascend_patch as _apply
        except ImportError:
            pytest.skip("Ascend backend unavailable")
        _apply()

    @pytest.fixture
    def semantic(self):
        semantic = MagicMock(spec=TritonSemantic)
        semantic.builder = _make_mock_builder()
        semantic._str_to_dot_input_precision = (lambda ip: TritonSemantic._str_to_dot_input_precision(semantic, ip))
        semantic.tensor = MagicMock(return_value=MagicMock())
        semantic.cast = MagicMock(side_effect=lambda x, dtype: x)
        semantic.bitcast = MagicMock(side_effect=lambda x, dtype: x)
        return semantic

    # ---- HF32 guard ----

    def test_hf32_fp32_inputs_keep_hf32(self, semantic):
        """fp32 x fp32 + hf32 → hf32 is preserved."""
        patched = TritonSemantic.dot
        mock_original = MagicMock(return_value=MagicMock())
        cell = patched.__closure__[0]
        saved = cell.cell_contents
        cell.cell_contents = mock_original
        try:
            lhs = _mock_tensor(tl.float32)
            rhs = _mock_tensor(tl.float32)
            patched(semantic, lhs, rhs, None, "hf32", None, tl.float32)
            assert mock_original.call_args[0][4] == "hf32"
        finally:
            cell.cell_contents = saved

    # ---- max_num_imprecise_acc ----

    def test_max_imprecise_explicit_forced_to_zero(self, semantic, capsys):
        """Explicit max_num_imprecise_acc: warns, reaches create_dot as 0."""
        lhs = _mock_tensor(tl.float16)
        rhs = _mock_tensor(tl.float16)
        TritonSemantic.dot(semantic, lhs, rhs, None, "ieee", 16, tl.float32)

        assert "max_num_imprecise_acc" in capsys.readouterr().out
        assert semantic.builder.create_dot.call_args[0][4] == 0

    def test_max_imprecise_none_no_warning_and_zero(self, semantic, capsys):
        """None (default): no warning, reaches create_dot as 0."""
        lhs = _mock_tensor(tl.float16)
        rhs = _mock_tensor(tl.float16)
        TritonSemantic.dot(semantic, lhs, rhs, None, "ieee", None, tl.float32)

        assert "max_num_imprecise_acc" not in capsys.readouterr().out
        assert semantic.builder.create_dot.call_args[0][4] == 0
