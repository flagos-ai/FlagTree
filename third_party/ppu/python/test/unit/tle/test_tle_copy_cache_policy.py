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
"""Per-copy PPU hints must reach TIX without touching shared-memory accesses."""

from dataclasses import FrozenInstanceError
import re

import pytest
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental.tle.language.gpu.ppu import CachePolicy

tle = pytest.importorskip("triton.experimental.tle.language")

LOAD_MODIFIERS = ("", ".ca", ".cg", ".cs", ".cv", ".lu", ".nc", ".bl")
STORE_MODIFIERS = ("", ".wb", ".cg", ".cs", ".wt", ".bl")


def test_policy_value_and_direction_contract():
    policy = CachePolicy(tl.constexpr(".ba"))
    assert policy == CachePolicy(".ba")
    assert hash(policy) == hash(CachePolicy(".ba"))
    assert policy.type.mangle() != CachePolicy(".bl").type.mangle()
    with pytest.raises(FrozenInstanceError):
        policy.cache_modifier = ".bl"
    for value in (None, 3, 7, ".ce", "evict_last", ".kp7"):
        with pytest.raises(ValueError):
            CachePolicy(value)
    with pytest.raises(ValueError, match="descriptor"):
        policy.validate_copy(True, True)
    with pytest.raises(ValueError, match="store"):
        CachePolicy(".nc").validate_copy(False, False)
    with pytest.raises(ValueError, match="load"):
        CachePolicy(".wt").validate_copy(False, True)
    with pytest.raises(ValueError, match="visibility"):
        CachePolicy(".ba").validate_copy(False, False)
    with pytest.raises(ValueError, match="visibility"):
        CachePolicy(".ba").validate_copy(False, True)


@triton.jit
def _copy(a, b, LOAD: tl.constexpr, STORE: tl.constexpr, SYNC: tl.constexpr = False):
    tile = tle.gpu.alloc((128, ), tl.float32, nv_mma_shared_layout=False)
    x = tl.arange(0, 128)
    if SYNC:
        # A nonzero masked fill structurally requires a register load.
        tle.gpu.copy(a + x, tile, (128, ), mask=x < 125, other=-9, cache_policy=LOAD)
    else:
        tle.gpu.copy(a + x, tile, (128, ), cache_policy=LOAD)
    tle.gpu.copy(tile, b + x, (128, ), cache_policy=STORE)


def _compile(load, store, sync=False):
    pytest.importorskip("triton._C.libtriton.ppu")
    compiled = triton.compile(
        ASTSource(_copy, {"a": "*fp32", "b": "*fp32"}, {"LOAD": load, "STORE": store, "SYNC": sync}),
        target=GPUTarget("cuda", 80, 32), options={"num_warps": 4})
    return compiled.asm["llir"]


@pytest.mark.parametrize("modifier", LOAD_MODIFIERS)
@pytest.mark.parametrize("sync", [False, True])
def test_load_cache_modifier_follows_selected_transport(modifier, sync):
    ir = _compile(CachePolicy(modifier), None, sync)
    if sync:
        assert f"ppu.ld.global{modifier}." in ir
        assert not re.search(r"cp\.async\.(?:ca|cg)\.shared\.global", ir)
    else:
        copies = re.findall(r"cp\.async\.(?:ca|cg)\.shared\.global", ir)
        assert copies
        assert copies == re.findall(r"cp\.async\.(?:ca|cg)\.shared\.global", _compile(None, None))
    assert not re.search(r"ppu\.(?:ld|st)\.shared\.(?:ca|cg|cs|cv|lu|nc|bl|ba)\.", ir)


@pytest.mark.parametrize("modifier", STORE_MODIFIERS)
def test_store_cache_modifier_reaches_only_global_store(modifier):
    ir = _compile(None, CachePolicy(modifier))
    assert f"ppu.st.global{modifier}." in ir
    assert not re.search(r"ppu\.(?:ld|st)\.shared\.(?:wb|cg|cs|wt|bl|ba)\.", ir)


def test_independent_load_and_store_policies():
    ir = _compile(CachePolicy(5), CachePolicy(6), sync=True)
    assert "ppu.ld.global.lu." in ir
    assert "ppu.st.global.bl." in ir


@pytest.mark.parametrize("load", [True, False])
def test_ba_rejected_by_ir_parser_without_python_validation(tmp_path, capfd, load):
    compiled = triton.compile(
        ASTSource(_copy, {"a": "*fp32", "b": "*fp32"},
                  {"LOAD": CachePolicy(".bl") if load else None, "STORE": None if load else CachePolicy(".bl")}),
        target=GPUTarget("cuda", 80, 32), options={"num_warps": 4})
    ir = compiled.asm["ttir"]
    assert "cacheModifier = bl" in ir
    path = tmp_path / "unsafe.ttir"
    path.write_text(ir.replace("cacheModifier = bl", "cacheModifier = ba"))
    with pytest.raises(RuntimeError):
        triton.compile(str(path), target=GPUTarget("cuda", 80, 32), options={"num_warps": 4})
    assert "cache" in capfd.readouterr().err.lower()
