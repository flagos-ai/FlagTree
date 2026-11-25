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
"""AIU copy hints preserve the descriptor, synchronization, and native kp mode."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess

import pytest
import triton
import triton.language as tl

if not getattr(tl.core, "_HAS_NATIVE_CACHE_POLICY", False):
    pytest.skip("native cache policy disabled", allow_module_level=True)
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental.tle import language as tle
from triton.experimental.tle.language.gpu.ppu import AIUCachePolicy


def test_aiu_policy_contract():
    policy = AIUCachePolicy(kp=tl.constexpr(4))
    assert policy == AIUCachePolicy(kp=4)
    assert policy.to_attributes(True) == {"backend": "ppu", "kind": "aiu", "kp": 4}
    assert policy.type.mangle() != AIUCachePolicy(kp=1).type.mangle()
    with pytest.raises(FrozenInstanceError):
        policy.kp = 1
    for kp in (0, 8, True, 1.0, "4"):
        with pytest.raises(ValueError):
            AIUCachePolicy(kp=kp)
    # An AIU mode is not an LSU modifier; the field does not exist on this type.
    with pytest.raises(TypeError, match="cache_modifier"):
        AIUCachePolicy(cache_modifier=".cs", kp=4)
    policy.validate_copy(True, True)
    for descriptor, load in ((True, False), (False, False)):
        with pytest.raises(ValueError, match="global-to-shared loads"):
            policy.validate_copy(descriptor, load)
    with pytest.raises(ValueError, match="GPU-produced input can be stale"):
        AIUCachePolicy(kp=7).validate_copy(True, True)


@triton.jit
def _copy(A, B, M: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, OM: tl.constexpr,
          OK: tl.constexpr, POLICY: tl.constexpr):
    row = tl.program_id(0) * BM + OM
    col = tl.program_id(1) * BK + OK
    source = tl.make_block_ptr(A, shape=(M, K), strides=(K, 1), offsets=(row, col), block_shape=(BM, BK), order=(1, 0))
    r = tl.arange(0, BM)
    c = tl.arange(0, BK)
    program = tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)
    output = B + program * BM * BK + r[:, None] * BK + c[None, :]
    if POLICY is None:
        tl.store(output, tle.load(source, is_async=True))
    else:
        tile = tle.gpu.alloc((BM, BK), A.dtype.element_ty, nv_mma_shared_layout=False)
        tle.gpu.copy(source, tile, (BM, BK), cache_policy=POLICY)
        tle.gpu.copy(tile, output, (BM, BK))


def _check_native_mode(kernel, kp, tmp_path):
    from triton.backends.ppu.compiler import get_ppu_llc
    binary = tmp_path / "copy.hgbin"
    binary.write_bytes(kernel.asm["hgbin"])
    objdump = Path(get_ppu_llc()).parent / "hgobjdump"
    isa = subprocess.check_output([str(objdump), "--dump-isa", str(binary)], text=True)
    lines = [line for line in isa.splitlines() if "vmem.aiu.ld.tsm" in line]
    expected = 1 if kp == 0 else 2 if kp == 3 else kp
    assert lines and all(f".kp{expected}" in line for line in lines)
    assert "__flagtree_ppu_aiu_copy" not in isa
    assert "vmem.acp.commit.grp" in isa and "commit_group(0)" in isa


@triton.jit
def _descriptor_and_lsu(A, Side, Out):
    descriptor = tl.make_tensor_descriptor(A, [32, 32], [32, 1], [32, 32])
    tile = tle.gpu.alloc((32, 32), tl.float16, nv_mma_shared_layout=False)
    tle.gpu.copy(descriptor, tile, (32, 32), (0, 0), cache_policy=tle.gpu.ppu.AIUCachePolicy(kp=5))
    index = tl.arange(0, 32)
    offsets = index[:, None] * 32 + index[None, :]
    side = tl.load(Side + offsets, cache_modifier=".cg")
    tl.store(Out + offsets, tl.load(tle.gpu.local_ptr(tile)) + side)


def test_aiu_descriptor_preserves_neighboring_lsu(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton._C.libtriton.ppu")
    a = torch.arange(1024, device="cuda", dtype=torch.float16)
    side = torch.ones_like(a)
    out = torch.empty_like(a)
    for _ in range(2):
        side.add_(1)
        out.fill_(-1)
        kernel = _descriptor_and_lsu[(1, )](a, side, out)
        assert torch.equal(out, a + side)
    _check_native_mode(kernel, 5, tmp_path)
    assert "ppu.ld.global.cg" in kernel.asm["llir"]


@pytest.mark.parametrize("kp", range(7))
@pytest.mark.parametrize("M,K,BM,BK,OM,OK,grid,dtype", [
    (128, 128, 32, 32, 0, 0, (4, 4), "float16"),
    (128, 160, 32, 32, 16, 32, (2, 3), "float32"),
    (93, 150, 32, 32, 0, 0, (3, 5), "int32"),
    (64, 128, 32, 64, 0, 0, (2, 2), "int8"),
    (64, 64, 16, 32, 0, 0, (4, 2), "float16"),
    (64, 1024, 32, 512, 0, 0, (2, 2), "float16"),
])
def test_aiu_copy_runtime(M, K, BM, BK, OM, OK, grid, dtype, kp, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton._C.libtriton.ppu")
    dtype = getattr(torch, dtype)
    a = (torch.arange(M * K, device="cuda", dtype=torch.int32) % 97 - 48).to(dtype).reshape(M, K)
    expected = torch.zeros((grid[0] * grid[1], BM, BK), dtype=dtype)
    for mi in range(grid[0]):
        for ki in range(grid[1]):
            r, c = mi * BM + OM, ki * BK + OK
            mm, kk = min(BM, M - r), min(BK, K - c)
            expected[mi * grid[1] + ki, :mm, :kk] = a[r:r + mm, c:c + kk].cpu()
    b = torch.full(expected.shape, -111, dtype=dtype, device="cuda")
    policy = None if kp == 0 else AIUCachePolicy(kp=kp)
    kernel = _copy[grid](a, b, M, K, BM, BK, OM, OK, policy, num_warps=4)
    torch.cuda.synchronize()
    assert torch.equal(b.cpu(), expected)
    _check_native_mode(kernel, kp, tmp_path)


@triton.jit
def _loop(A, B, M, K, POLICY: tl.constexpr, TRANS: tl.constexpr, STAGES: tl.constexpr):
    r = tl.arange(0, 32)
    c = tl.arange(0, 32)
    tile = tle.gpu.alloc((32, 32), A.dtype.element_ty, nv_mma_shared_layout=False)
    for index in tl.range(0, 4, num_stages=STAGES):
        if TRANS:
            source = tl.make_block_ptr(A, shape=(M, K), strides=(1, M), offsets=(index * 32, 32), block_shape=(32, 32),
                                       order=(0, 1))
        else:
            source = tl.make_block_ptr(A, shape=(M, K), strides=(K, 1), offsets=(index * 32, 32), block_shape=(32, 32),
                                       order=(1, 0))
        tle.gpu.copy(source, tile, (32, 32), cache_policy=POLICY)
        tle.gpu.copy(tile, B + index * 1024 + r[:, None] * 32 + c[None, :], (32, 32))


@pytest.mark.parametrize("kp", range(1, 7))
@pytest.mark.parametrize("trans,stages,dtype", [(False, 1, "float16"), (False, 3, "float16"), (True, 1, "int16"),
                                                (True, 3, "bfloat16")])
def test_aiu_loop_dynamic_shapes(trans, stages, dtype, kp, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton._C.libtriton.ppu")
    dtype = getattr(torch, dtype)
    a = (torch.arange(128 * 96, device="cuda", dtype=torch.int32) % 101 - 50).to(dtype).reshape(128, 96)
    if trans:
        a = a.T.contiguous().T
    expected = a[:, 32:64].reshape(4, 32, 32).cpu()
    b = torch.full(expected.shape, -111, dtype=dtype, device="cuda")
    kernel = _loop[(1, )](a, b, 128, 96, AIUCachePolicy(kp=kp), trans, stages, num_warps=4)
    torch.cuda.synchronize()
    assert torch.equal(b.cpu(), expected)
    _check_native_mode(kernel, kp, tmp_path)


@triton.jit
def _invalid(A, B, POLICY: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, STRIDE: tl.constexpr, PLAIN: tl.constexpr):
    r = tl.arange(0, BM)
    c = tl.arange(0, BK)
    tile = tle.gpu.alloc((BM, BK), A.dtype.element_ty, nv_mma_shared_layout=False)
    if PLAIN:
        source = A + r[:, None] * 128 + c[None, :]
    else:
        source = tl.make_block_ptr(A, shape=(128, 128), strides=(STRIDE, 1), offsets=(0, 0), block_shape=(BM, BK),
                                   order=(1, 0))
    tle.gpu.copy(source, tile, (BM, BK), cache_policy=POLICY)
    tle.gpu.copy(tile, B + r[:, None] * BK + c[None, :], (BM, BK))


@pytest.mark.parametrize("dtype,BM,BK,stride,plain,arch,diagnostic", [
    ("fp32", 32, 32, 128, True, 80, "rank-2 block pointer"),
    ("fp64", 32, 32, 128, False, 80, "8/16/32-bit"),
    ("fp32", 32, 32, 256, False, 80, "contiguous block-pointer strides"),
    ("fp32", 8, 32, 128, False, 80, "16..2048 rows"),
    ("fp16", 32, 8, 128, False, 80, "32 contiguous bytes"),
    ("fp32", 32, 32, 128, False, 89, "PPU0010"),
])
def test_invalid_aiu_copy(dtype, BM, BK, stride, plain, arch, diagnostic, capfd):
    pytest.importorskip("triton._C.libtriton.ppu")
    constants = {"POLICY": AIUCachePolicy(kp=4), "BM": BM, "BK": BK, "STRIDE": stride, "PLAIN": plain}
    with pytest.raises(Exception):
        triton.compile(ASTSource(_invalid, {"A": "*" + dtype, "B": "*" + dtype}, constants),
                       target=GPUTarget("cuda", arch, 32), options={"num_warps": 4})
    assert diagnostic in capfd.readouterr().err


@triton.jit
def _narrow_stride(A, B, width):
    row_stride = width.to(tl.uint16).to(tl.int64)
    source = tl.make_block_ptr(A, (32, width), (row_stride, 1), (0, 0), (32, 32), (1, 0))
    tile = tle.gpu.alloc((32, 32), tl.int32, nv_mma_shared_layout=False)
    tle.gpu.copy(source, tile, (32, 32), cache_policy=tle.gpu.ppu.AIUCachePolicy(kp=5))
    i = tl.arange(0, 32)
    tle.gpu.copy(tile, B + i[:, None] * 32 + i[None, :], (32, 32))


def test_aiu_narrowed_stride_requires_packed_guard():
    pytest.importorskip("triton._C.libtriton.ppu")
    kernel = triton.compile(ASTSource(_narrow_stride, {"A": "*i32", "B": "*i32", "width": "i32"}, {}),
                            target=GPUTarget("cuda", 80, 32))
    assert "PPU AIU cache policy requires contiguous block-pointer strides matching its order" in kernel.asm["ttgir"]


@triton.jit(noinline=True)
def _untraceable_copy(source, output, POLICY: tl.constexpr):
    tile = tle.gpu.alloc((32, 32), tl.float32, nv_mma_shared_layout=False)
    tle.gpu.copy(source, tile, (32, 32), cache_policy=POLICY)
    r = tl.arange(0, 32)
    c = tl.arange(0, 32)
    tle.gpu.copy(tile, output + r[:, None] * 32 + c[None, :], (32, 32))


@triton.jit
def _call_untraceable(A, B, POLICY: tl.constexpr):
    source = tl.make_block_ptr(A, shape=(128, 128), strides=(128, 1), offsets=(0, 0), block_shape=(32, 32),
                               order=(1, 0))
    _untraceable_copy(source, B, POLICY)


def test_untraceable_aiu_pointer_is_diagnosed(capfd):
    pytest.importorskip("triton._C.libtriton.ppu")
    with pytest.raises(Exception):
        triton.compile(ASTSource(_call_untraceable, {"A": "*fp32", "B": "*fp32"}, {"POLICY": AIUCachePolicy(kp=4)}),
                       target=GPUTarget("cuda", 80, 32), options={"num_warps": 4})
    assert "traceable make_block_ptr" in capfd.readouterr().err
