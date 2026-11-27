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
"""HCU cache policy validation for global-memory and buffer-op lowerings."""
from dataclasses import FrozenInstanceError
import re

import pytest
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental.tle import language as tle
from triton.experimental.tle.language.gpu.hcu import CachePolicy

if not getattr(tl.core, "_HAS_NATIVE_CACHE_POLICY", False):
    pytest.skip("native cache policy disabled", allow_module_level=True)

LOAD_MODIFIERS = ("", ".ca", ".cg", ".cs", ".cv")
STORE_MODIFIERS = ("", ".wb", ".cg", ".cs")


def test_policy_contract():
    policy = CachePolicy(tl.constexpr(".cs"))
    assert policy == CachePolicy(".cs")
    assert hash(policy) == hash(CachePolicy(".cs"))
    assert policy.type.mangle() != CachePolicy(".cg").type.mangle()
    with pytest.raises(FrozenInstanceError):
        policy.cache_modifier = ".cg"
    for invalid in (None, 3, ".wt", ".lu", "evict_last"):
        with pytest.raises(ValueError):
            CachePolicy(invalid)
    with pytest.raises(ValueError, match="descriptor"):
        policy.validate_copy(True, True)
    with pytest.raises(ValueError, match="store"):
        CachePolicy(".cv").validate_copy(False, False)
    with pytest.raises(ValueError, match="load"):
        CachePolicy(".wb").validate_copy(False, True)


@triton.jit
def copy_kernel(a, b, LOAD: tl.constexpr, STORE: tl.constexpr):
    tile = tle.gpu.alloc((128, ), tl.float32, nv_mma_shared_layout=False)
    x = tl.arange(0, 128)
    tle.gpu.copy(a + x, tile, (128, ), cache_policy=LOAD)
    tle.gpu.copy(tile, b + x, (128, ), cache_policy=STORE)


@triton.jit
def masked_copy_kernel(a, b, N: tl.constexpr, POLICY: tl.constexpr):
    tile = tle.gpu.alloc((128, ), tl.float32, nv_mma_shared_layout=False)
    x = tl.arange(0, 128)
    tle.gpu.copy(a + x, tile, (128, ), mask=x < N, other=0, cache_policy=POLICY)
    tle.gpu.copy(tile, b + x, (128, ), mask=x < N, cache_policy=POLICY)


@triton.jit
def ordinary_memory_kernel(a, b, POLICY: tl.constexpr):
    x = tl.arange(0, 128)
    tl.store(b + x, tl.load(a + x, cache_policy=POLICY), cache_policy=POLICY)


@triton.jit
def masked_buffer_kernel(a, b, POLICY: tl.constexpr):
    x = tl.arange(0, 128)
    value = tl.load(a + x, mask=x < 117, other=-7, cache_policy=POLICY)
    tl.store(b + x, value, mask=x < 121, cache_policy=POLICY)


@pytest.mark.parametrize("policy", (None, CachePolicy(".cs")))
def test_default_buffer_descriptor_on_device(policy, monkeypatch):
    pytest.importorskip("triton._C.libtriton.hcu")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or triton.runtime.driver.active.get_current_target().backend != "hip":
        pytest.skip("requires an HCU device")
    monkeypatch.delenv("AMDGCN_USE_BUFFER_OPS", raising=False)
    src = torch.arange(140, dtype=torch.int32, device="cuda")
    dst = torch.full_like(src, -19)
    kernel = masked_buffer_kernel[(1, )](src[3:], dst[5:], policy)
    assembly = kernel.asm["amdgcn"]
    for op in ("buffer_load_", "buffer_store_"):
        instructions = [line for line in assembly.splitlines() if op in line]
        assert instructions and all(("glc slc" in line) == (policy is not None) for line in instructions)
    expected = torch.full_like(src, -19)
    expected[5:122] = src[3:120]
    expected[122:126] = -7
    torch.testing.assert_close(dst, expected)


@pytest.mark.parametrize("arch", ("gfx928", "gfx936", "gfx938"))
def test_ordinary_streaming_load_and_store_on_hcu(arch, monkeypatch):
    pytest.importorskip("triton._C.libtriton.hcu")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    source = ASTSource(ordinary_memory_kernel, {"a": "*fp32", "b": "*fp32"}, {"POLICY": CachePolicy(".cs")})
    assembly = triton.compile(source, target=GPUTarget("hip", arch, 64)).asm["amdgcn"]
    for op in ("buffer_load_", "buffer_store_"):
        instructions = [line for line in assembly.splitlines() if op in line]
        assert instructions and all("glc slc" in line for line in instructions)


@pytest.mark.parametrize("n", (1, 127, 128))
@pytest.mark.parametrize("arch", ("gfx936", "gfx938"))
def test_uniform_mask_keeps_vector_cache_control(arch, n, monkeypatch):
    pytest.importorskip("triton._C.libtriton.hcu")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "0")
    compiled = triton.compile(
        ASTSource(masked_copy_kernel, {"a": "*fp32", "b": "*fp32"}, {"N": n, "POLICY": CachePolicy(".cs")}),
        target=GPUTarget("hip", arch, 64), options={"num_warps": 4})
    loads = [line for line in compiled.asm["amdgcn"].splitlines() if "global_load_" in line]
    assert loads and all("glc slc" in line for line in loads)
    assert "load volatile" not in compiled.asm["llir"]
    assert "buffer_load" not in compiled.asm["amdgcn"]


@pytest.mark.parametrize("load", LOAD_MODIFIERS)
@pytest.mark.parametrize("store", STORE_MODIFIERS)
def test_gfx936_global_cache_flags(load, store, monkeypatch):
    pytest.importorskip("triton._C.libtriton.hcu")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "0")
    compiled = triton.compile(
        ASTSource(copy_kernel, {"a": "*fp32", "b": "*fp32"}, {"LOAD": CachePolicy(load), "STORE": CachePolicy(store)}),
        target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 4})
    assembly = compiled.asm["amdgcn"]
    assert "buffer_load" not in assembly and "buffer_store" not in assembly
    loads = [line for line in assembly.splitlines() if re.search(r"global_load_", line)]
    stores = [line for line in assembly.splitlines() if re.search(r"global_store_", line)]
    assert loads and stores
    if load in (".cg", ".cs"):
        assert all("glc slc" in line for line in loads)
    elif load == ".cv":
        assert all("glc" in line and "slc" not in line for line in loads)
    else:
        assert all("glc" not in line and "slc" not in line for line in loads)
    assert all(("glc slc" in line) == (store == ".cs") for line in stores)


@pytest.mark.parametrize("load", LOAD_MODIFIERS)
@pytest.mark.parametrize("store", STORE_MODIFIERS)
def test_gfx936_buffer_ops_preserve_transport(load, store, monkeypatch):
    pytest.importorskip("triton._C.libtriton.hcu")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    compiled = triton.compile(
        ASTSource(copy_kernel, {"a": "*fp32", "b": "*fp32"}, {"LOAD": CachePolicy(load), "STORE": CachePolicy(store)}),
        target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 4})
    assembly = compiled.asm["amdgcn"]
    assert "global_load_" not in assembly and "global_store_" not in assembly
    loads = [line for line in assembly.splitlines() if re.search(r"buffer_load_", line)]
    stores = [line for line in assembly.splitlines() if re.search(r"buffer_store_", line)]
    assert loads and stores
    if load in (".cg", ".cs"):
        assert all("glc slc" in line for line in loads)
    elif load == ".cv":
        assert all("glc" in line and "slc" not in line for line in loads)
    else:
        assert all("glc" not in line and "slc" not in line for line in loads)
    assert all(("glc slc" in line) == (store == ".cs") for line in stores)
