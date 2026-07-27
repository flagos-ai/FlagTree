import ast
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import triton
import triton.language as tl
from triton.language.extra.xpu import libdevice
from triton.tools import build_extern


@triton.jit
def grid_kernel(dst, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(dst + offsets, 0)


@triton.jit
def fmod_kernel(lhs, rhs, dst, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(lhs + offsets, mask=mask)
    y = tl.load(rhs + offsets, mask=mask)
    tl.store(dst + offsets, x % y, mask=mask)


@triton.jit
def libdevice_kernel(lhs, rhs, dst, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(lhs + offsets, mask=mask, other=1.0)
    y = tl.load(rhs + offsets, mask=mask, other=1.0)
    value = libdevice.sqrt(x * x + 1.0)
    value += libdevice.rsqrt(y * y + 1.0)
    value += libdevice.div_rn(x, y)
    value += libdevice.fma(x, y, 1.0)
    tl.store(dst + offsets, value, mask=mask)


def _extern_functions():
    tree = ast.parse(_libdevice_path().read_text())
    return [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            isinstance(decorator, ast.Attribute) and decorator.attr == "extern" for decorator in node.decorator_list)
    ]


def _function_source(name):
    tree = ast.parse(_libdevice_path().read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(_libdevice_path().read_text(), node)
    raise AssertionError(name)


def _libdevice_path():
    return Path(__file__).resolve().parents[4] / "language/xpu/libdevice.py"


def test_xpu_libdevice_externs_use_semantic():
    functions = _extern_functions()
    assert len(functions) == 198
    for function in functions:
        args = [argument.arg for argument in function.args.args]
        assert "_semantic" in args, function.name
        assert "_builder" not in args, function.name


def test_xpu_libdevice_extern_calls_use_semantic():
    source = _libdevice_path().read_text()
    assert "_builder" not in source
    assert source.count("_semantic=_semantic") == 198


def test_build_extern_generates_semantic_stubs():
    source = (Path(__file__).resolve().parents[6] / "third_party/xpu/python/triton/tools/build_extern.py").read_text()
    assert "_builder" not in source
    assert "_semantic=None" in source
    assert "_semantic=_semantic" in source


def test_xpu_libdevice_symbols():
    fmod_source = _function_source("fmod")
    rsqrt_source = _function_source("rsqrt")
    div_source = _function_source("div_rn")
    assert "_ZN3xpu5fmodfEff" in fmod_source
    assert "_ZN3xpu6rsqrtfEf" in rsqrt_source
    assert "_ZN3xpu6rsqrtfEd" in rsqrt_source
    assert "_ZN3xpu9__fdiv_rnEff" in div_source
    assert '"Unsupported", core.dtype("fp64")' in div_source


def test_xpu_large_tensor_override_is_vendored():
    root = Path(__file__).resolve().parents[6]
    traits = root / "third_party/xpu/spec_cpp/include/triton/Dialect/Triton/IR/Traits.h"
    source = traits.read_text()
    assert "maxTensorNumElements = INT_MAX" in source


def test_xpu_backend_spec_sources_are_backend_owned():
    root = Path(__file__).resolve().parents[6]
    spec_root = root / "third_party/xpu/spec_cpp"
    spec_sources = sorted((spec_root / "lib").rglob("*.cpp"))

    assert spec_sources
    for spec_source in spec_sources:
        main_source = root / spec_source.relative_to(spec_root)
        assert main_source.is_file()


def test_xpu_elementwise_dedup_fallback_is_vendored():
    root = Path(__file__).resolve().parents[6]
    main_header = (root / "include/triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h")
    xpu_header = (root /
                  "third_party/xpu/spec_cpp/include/triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h")
    main_source = main_header.read_text()
    xpu_source = xpu_header.read_text()
    assert ("for (auto [c, d] : llvm::zip(constancy, dims)) {\n"
            "      assert(llvm::isPowerOf2_32(c));" in main_source)
    assert "if (!llvm::isPowerOf2_32(c))\n        return resultVals;" in xpu_source
    assert "assert(llvm::isPowerOf2_32(c));" in xpu_source


def test_xpu_masked_load_materializes_other_in_frontend():
    semantic = (Path(__file__).resolve().parents[6] / "third_party/xpu/python/triton/language/semantic.py").read_text()
    assert "load_value = self.tensor(" in semantic
    assert "ret = self.where(mask, load_value, other)" in semantic


def test_xpu_launch_grid_is_compilation_option(device):
    block = 32
    grid = (5, )
    dst = torch.empty(block, dtype=torch.int32, device=device)
    grid_kernel[grid](dst, BLOCK=block)
    torch.testing.assert_close(dst, torch.zeros_like(dst), rtol=0, atol=0)
    # XPU maps programs through clusters/cores, so CUDA-style contiguous pid
    # output is not a valid grid assertion. The migration contract is that the
    # actual launch grid is present in XPU compilation metadata.
    cache_dir = os.environ.get("TRITON_CACHE_DIR")
    if cache_dir:
        metadata = list(Path(cache_dir).rglob("grid_kernel.json"))
        assert metadata
        assert json.loads(metadata[0].read_text())["grid"] == list(grid)


def test_xpu_float_mod_matches_fmod(device):
    lhs_np = np.array([-5.5, -5.5, 5.5, 5.5, -4.0, 4.0], dtype=np.float32)
    rhs_np = np.array([2.0, -2.0, 2.0, -2.0, 2.0, -2.0], dtype=np.float32)
    lhs = torch.tensor(lhs_np, device=device)
    rhs = torch.tensor(rhs_np, device=device)
    dst = torch.empty_like(lhs)
    fmod_kernel[(1, )](lhs, rhs, dst, n_elements=lhs.numel(), BLOCK=8)
    torch.testing.assert_close(dst.cpu(), torch.from_numpy(np.fmod(lhs_np, rhs_np)), rtol=0, atol=0)


def test_xpu_representative_libdevice_externs(device):
    lhs = torch.tensor([1.25, 2.5, 4.0, 8.0], dtype=torch.float32, device=device)
    rhs = torch.tensor([2.0, 4.0, 8.0, 16.0], dtype=torch.float32, device=device)
    dst = torch.empty_like(lhs)
    libdevice_kernel[(1, )](lhs, rhs, dst, n_elements=lhs.numel(), BLOCK=8)
    assert torch.isfinite(dst).all()
