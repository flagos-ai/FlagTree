# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""MUSA pointer-copy policies must reach native LSU operations without losing masks."""

import dataclasses
import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.errors import CompilationError
from triton.experimental.tle.language.gpu.mthreads import LSUCachePolicy, TMECachePolicy

from test_tle_utils import compile_musa, compile_to_ttir, require_mthreads_libtriton

if not tl.core._HAS_NATIVE_CACHE_POLICY:
    pytest.skip("native cache policy disabled", allow_module_level=True)

require_mthreads_libtriton()

POLICIES = [LSUCachePolicy()] + [LSUCachePolicy(i, 5 - i, i % 2) for i in range(6)]


@triton.jit
def _pointer_copy(src, dst, n, stride, LOAD: tl.constexpr, STORE: tl.constexpr, MASK_STORE: tl.constexpr):
    offsets = tl.arange(0, 128)
    smem = tle.gpu.alloc((128, ), src.dtype.element_ty, nv_mma_shared_layout=False)
    tle.gpu.copy(src + offsets * stride, smem, (128, ), mask=offsets < n, other=-9, cache_policy=LOAD)
    if MASK_STORE:
        tle.gpu.copy(smem, dst + offsets, (128, ), mask=offsets < n, cache_policy=STORE)
    else:
        tle.gpu.copy(smem, dst + offsets, (128, ), cache_policy=STORE)


def _source(policy, dtype="fp32"):
    return ASTSource(
        _pointer_copy, signature={
            "src": f"*{dtype}", "dst": f"*{dtype}", "n": "i32", "stride": "i32", "LOAD": "constexpr", "STORE":
            "constexpr", "MASK_STORE": "constexpr"
        }, constexprs={"LOAD": policy, "STORE": policy, "MASK_STORE": True})


@pytest.mark.parametrize("policy", POLICIES)
def test_lsu_cache_policy_reaches_native_copy(policy):
    compiled = compile_musa(_pointer_copy, signature=_source(policy).signature,
                            constexprs={"LOAD": policy, "STORE": policy, "MASK_STORE": True})
    for stage in ("ttir", "ttgir"):
        assert "tt.cache_policy" not in compiled.asm[stage]
        for is_load in (True, False):
            fields = policy.to_attributes(is_load)
            cache = (f"musa_i{fields['inner_persistence']}_"
                     f"o{fields['outer_persistence']}_l{fields['l2_policy']}")
            assert f"cacheModifier = {cache}" in compiled.asm[stage]
    assert "llvm.musa.memcpy.g2s" not in compiled.asm["llir"]
    for is_load, name in ((True, "ld"), (False, "st")):
        fields = policy.to_attributes(is_load)
        calls = [
            line for line in compiled.asm["llir"].splitlines()
            if "call " in line and f"llvm.musa.lsu.{name}.cache.hint.i" in line
        ]
        assert calls
        tail = (rf"i32 {fields['inner_persistence']}, i32 {fields['outer_persistence']}, "
                rf"i32 {0 if is_load else 1}, i32 {fields['l2_policy']}\)")
        assert all(re.search(tail, call) for call in calls), calls


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int8, torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize("mask_store,stride", [(True, 1), (False, 2)])
@pytest.mark.parametrize("hinted", [False, True])
@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_lsu_cache_policy_masked_copy_runtime(dtype, mask_store, stride, hinted):
    src = torch.arange(117 * stride, dtype=torch.int32, device="musa").to(dtype)
    dst = torch.full((128, ), -23, dtype=dtype, device="musa")
    load = LSUCachePolicy(3, 1, 0) if hinted else None
    store = LSUCachePolicy(1, 3, 1) if hinted else None
    _pointer_copy[(1, )](src, dst, 117, stride, load, store, mask_store, num_warps=4, num_stages=1)
    reference = torch.full((128, ), -23 if mask_store else -9, dtype=dtype)
    reference[:117] = src.cpu()[::stride]
    torch.testing.assert_close(dst.cpu(), reference, rtol=0, atol=0)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_lsu_cache_policy_native_modes_runtime(policy):
    src = torch.arange(117, dtype=torch.float32, device="musa")
    dst = torch.full((128, ), -23, dtype=torch.float32, device="musa")
    first = _pointer_copy[(1, )](src, dst, 117, 1, policy, policy, True, num_warps=4, num_stages=1)
    equal = dataclasses.replace(policy)
    second = _pointer_copy[(1, )](src, dst, 117, 1, equal, equal, True, num_warps=4, num_stages=1)
    assert first is second
    torch.testing.assert_close(dst.cpu()[:117], src.cpu(), rtol=0, atol=0)
    assert torch.all(dst.cpu()[117:] == -23)


@pytest.mark.parametrize("field,value", [("inner_persistence", -1),
                                         ("inner_persistence", 6), ("outer_persistence", True),
                                         ("outer_persistence", "persist"), ("l2_policy", 2), ("l2_policy", 0.5)])
def test_lsu_cache_policy_rejects_invalid_field(field, value):
    with pytest.raises(ValueError, match="LSUCachePolicy"):
        LSUCachePolicy(**{field: value})


def test_lsu_cache_policy_direction_defaults_and_immutability():
    policy = LSUCachePolicy()
    assert policy.to_attributes(True)["l2_policy"] == 0
    assert policy.to_attributes(False)["l2_policy"] == 1
    assert "coherent" not in policy.to_attributes(True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.l2_policy = 1
    with pytest.raises(ValueError, match="pointer copies"):
        policy.validate_copy(True, True)
    with pytest.raises(ValueError, match="descriptor copies"):
        TMECachePolicy().validate_copy(False, False)


def test_lsu_cache_policy_rejects_qy2(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        triton.compile(_source(LSUCachePolicy()), target=GPUTarget("musa", 22, 128))
    assert "LSU copy cache policy requires MUSA mp_31" in capfd.readouterr().err


@triton.jit
def _ordinary_memory(src, dst, LOAD: tl.constexpr, STORE: tl.constexpr):
    offsets = tl.arange(0, 128)
    values = tl.load(src + offsets, cache_policy=LOAD)
    tl.store(dst + offsets, values, cache_policy=STORE)


def _compile_memory(arch, load=None, store=None):
    source = ASTSource(_ordinary_memory,
                       signature={"src": "*fp32", "dst": "*fp32", "LOAD": "constexpr", "STORE":
                                  "constexpr"}, constexprs={"LOAD": load, "STORE": store})
    return triton.compile(source, target=GPUTarget("musa", arch, 32)).asm["llir"]


def test_ordinary_lsu_policy_uses_native_instructions():
    policy = LSUCachePolicy(3, 3, 1)
    hinted = _compile_memory(31, policy, policy)
    assert "llvm.musa.lsu.ld.cache.hint" in hinted
    assert "llvm.musa.lsu.st.cache.hint" in hinted


@pytest.mark.parametrize("access", ["load", "store"])
def test_unsupported_ordinary_lsu_policy_reports_target(access):
    assert "llvm.musa.lsu." not in _compile_memory(32)
    with pytest.raises(CompilationError, match=rf"tl\.{access} with LSUCachePolicy requires MUSA mp_31 .*mp_32"):
        _compile_memory(32, **{access: LSUCachePolicy()})


def test_tme_policy_cannot_silently_select_lsu():
    with pytest.raises(CompilationError, match="cache_policy"):
        compile_to_ttir(_pointer_copy, signature=_source(TMECachePolicy()).signature,
                        constexprs={"LOAD": TMECachePolicy(), "STORE": None, "MASK_STORE": False})


@triton.jit
def _whole_copy(src, dst, POLICY: tl.constexpr):
    offsets = tl.arange(0, 1024)
    smem = tle.gpu.alloc((1024, ), src.dtype.element_ty, nv_mma_shared_layout=False)
    all_true = tl.full((1024, ), True, tl.int1)
    tle.gpu.copy(src + offsets, smem, (1024, ), mask=all_true, cache_policy=POLICY)
    tle.gpu.copy(smem, dst + offsets, (1024, ), mask=all_true, cache_policy=POLICY)


@pytest.mark.parametrize(
    "dtype",
    [torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_lsu_cache_policy_full_mask_and_vector_copy(dtype):
    src = torch.arange(1024, dtype=torch.int32, device="musa").to(dtype)
    dst = torch.empty_like(src)
    default = _whole_copy[(1, )](src, dst, None, num_warps=4, num_stages=1)
    compiled = _whole_copy[(1, )](src, dst, LSUCachePolicy(), num_warps=4, num_stages=1)
    assert "llvm.musa.memcpy.g2s" in default.asm["llir"]
    assert compiled.asm["llir"].count("llvm.musa.memcpy.g2s") == default.asm["llir"].count("llvm.musa.memcpy.g2s")
    assert "llvm.musa.lsu.ld.cache.hint.i" not in compiled.asm["llir"]
    assert "llvm.musa.lsu.st.cache.hint.i" in compiled.asm["llir"]
    torch.testing.assert_close(dst.cpu(), src.cpu(), rtol=0, atol=0)
