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

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from triton.experimental.tle.language.gpu import CachePolicy, FractionalEvictionPolicy

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="requires NVIDIA Hopper or newer",
)


@triton.jit
def _pointer_copy(X, Y, N: tl.constexpr, LOAD: tl.constexpr, STORE: tl.constexpr):
    offsets = tl.arange(0, 512)
    dtype: tl.constexpr = tl.int8 if X.dtype.element_ty == tl.int1 else X.dtype.element_ty
    smem = tle.gpu.alloc([512], dtype, nv_mma_shared_layout=False)
    tle.gpu.copy(X + offsets, smem, [512], mask=offsets < N, cache_policy=LOAD)
    tle.gpu.copy(smem, Y + offsets, [512], mask=offsets < N, cache_policy=STORE)


@triton.jit
def _ordinary_load(X, Y, N: tl.constexpr, POLICY: tl.constexpr):
    offsets = tl.arange(0, 512)
    value = tl.load(X + offsets, offsets < N, other=0, cache_policy=POLICY)
    tl.store(Y + offsets, value + 1, offsets < N)


@triton.jit
def _descriptor_copy(X, Y, ROWS: tl.constexpr, LOAD: tl.constexpr, STORE: tl.constexpr):
    x_desc = tl.make_tensor_descriptor(X, [ROWS, 64], [64, 1], [ROWS, 64])
    smem = tle.gpu.alloc([ROWS, 64], tl.float32)
    barrier = tle.gpu.alloc_barrier(expect_bytes=ROWS * 64 * 4)
    tle.gpu.copy(x_desc, smem, [ROWS, 64], [0, 0], barrier=barrier, cache_policy=LOAD)
    tle.gpu.barrier_wait(barrier, phaseIdx=0)
    y_desc = tl.make_tensor_descriptor(Y, [ROWS, 64], [64, 1], [ROWS, 64])
    tle.gpu.copy(smem, y_desc, [ROWS, 64], [0, 0], cache_policy=STORE)


LOAD_POLICIES = (
    [CachePolicy(cache_modifier=m) for m in ("ca", "cg", "cs", "lu", "cv")] +
    [CachePolicy(l1=p) for p in ("evict_normal", "evict_first", "evict_last", "evict_unchanged", "no_allocate")] + [
        CachePolicy(l2=FractionalEvictionPolicy(p))
        for p in ("evict_normal", "evict_first", "evict_last", "evict_unchanged")
    ])
STORE_POLICIES = (
    [CachePolicy(cache_modifier=m) for m in ("wb", "cg", "cs", "wt")] +
    [CachePolicy(l1=p) for p in ("evict_normal", "evict_first", "evict_last", "evict_unchanged", "no_allocate")] + [
        CachePolicy(l2=FractionalEvictionPolicy(p))
        for p in ("evict_normal", "evict_first", "evict_last", "evict_unchanged")
    ])


def _assert_policy(ptx, policy, is_load, is_tma=False):
    if is_tma:
        marker = "global.shared" if not is_load else "shared::cluster.global"
        lines = [line for line in ptx.splitlines() if "cp.async.bulk.tensor" in line and marker in line]
    elif is_load:
        lines = [
            line for line in ptx.splitlines()
            if "ld.global" in line or ("cp.async." in line and ".shared.global" in line)
        ]
    else:
        lines = [line for line in ptx.splitlines() if "st.global" in line]
    assert lines, ptx
    for line in lines:
        if policy.cache_modifier:
            assert f".{policy.cache_modifier}." in line, line
        if policy.l1:
            assert f".L1::{policy.l1}." in line, line
        if policy.l2:
            assert ".L2::cache_hint" in line, line
    if policy.l2:
        assert f"createpolicy.fractional.L2::{policy.l2.primary}" in ptx


@pytest.mark.parametrize("policy", LOAD_POLICIES)
@pytest.mark.parametrize("n", [509, 512])
def test_ordinary_load(policy, n):
    x = torch.arange(512, device="cuda", dtype=torch.float32)
    y = torch.full_like(x, -17)
    kernel = _ordinary_load[(1, )](x, y, n, policy)
    torch.testing.assert_close(y[:n], x[:n] + 1, rtol=0, atol=0)
    assert torch.all(y[n:] == -17)
    _assert_policy(kernel.asm["ptx"], policy, True)
    assert "cp.async." not in kernel.asm["ptx"]


@pytest.mark.parametrize("n,modifier", [(509, None), (512, None), (509, "ca"), (512, "ca"), (512, "cg")])
def test_supported_pointer_copy_keeps_async(n, modifier):
    x = torch.arange(512, device="cuda", dtype=torch.float32)
    y = torch.full_like(x, -17)
    policy = CachePolicy(cache_modifier=modifier) if modifier else None
    kernel = _pointer_copy[(1, )](x, y, n, policy, None)
    torch.testing.assert_close(y[:n], x[:n], rtol=0, atol=0)
    assert torch.all(y[n:] == -17)
    assert "cp.async." in kernel.asm["ptx"]
    if policy:
        _assert_policy(kernel.asm["ptx"], policy, True)


@pytest.mark.parametrize(
    "policy", [CachePolicy(cache_modifier=m) for m in ("cs", "lu", "cv", "cg")] +
    [CachePolicy(l1=p) for p in ("evict_normal", "evict_first", "evict_last", "evict_unchanged", "no_allocate")])
@pytest.mark.parametrize("n", [509, 512])
def test_pointer_copy_converts_hints_without_changing_transport(policy, n):
    x = torch.arange(512, device="cuda", dtype=torch.float32)
    y = torch.full_like(x, -17)
    default = _pointer_copy[(1, )](x, y, n, None, None)
    kernel = _pointer_copy[(1, )](x, y, n, policy, None)
    torch.testing.assert_close(y[:n], x[:n], rtol=0, atol=0)
    assert torch.all(y[n:] == -17)
    copies = [line for line in kernel.asm["ptx"].splitlines() if "cp.async." in line and "shared.global" in line]
    baseline = [line for line in default.asm["ptx"].splitlines() if "cp.async." in line and "shared.global" in line]
    assert copies and len(copies) == len(baseline)
    assert all(".L1::" not in line and ".L2::cache_hint" not in line for line in copies)
    assert all("cp.async.ca." in line or "cp.async.cg." in line for line in copies)
    # Reuse resident allocations after a GPU update, and overwrite warm output.
    x.add_(11)
    y.fill_(-17)
    _pointer_copy[(1, )](x, y, n, policy, None)
    torch.testing.assert_close(y[:n], x[:n], rtol=0, atol=0)
    assert torch.all(y[n:] == -17)


@pytest.mark.parametrize("ptx_version", [87, 94])
def test_pointer_l2_policy_tracks_selected_ptx(ptx_version, monkeypatch, tmp_path):
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.compiler import ASTSource

    # Inspect PTX without requiring an assembler that supports PTX 9.4.
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(CUDABackend, "make_cubin", lambda *args: b"")
    policy = CachePolicy(l2=FractionalEvictionPolicy("evict_last"))
    source = ASTSource(
        _pointer_copy,
        signature={"X": "*fp32", "Y": "*fp32", "N": "constexpr", "LOAD": "constexpr", "STORE":
                   "constexpr"}, constexprs={"N": 512, "LOAD": policy, "STORE": None})
    kernel = triton.compile(source, target=GPUTarget("cuda", 90, 32), options={"ptx_version": ptx_version})
    ptx = kernel.asm["ptx"]
    assert "cp.async." in ptx
    assert ("createpolicy.fractional" in ptx) == (ptx_version >= 94)
    assert (".L2::cache_hint" in ptx) == (ptx_version >= 94)


@pytest.mark.parametrize("policy", STORE_POLICIES)
def test_pointer_store(policy):
    x = torch.arange(512, device="cuda", dtype=torch.float32)
    y = torch.full_like(x, -17)
    kernel = _pointer_copy[(1, )](x, y, 509, None, policy)
    torch.testing.assert_close(y[:509], x[:509], rtol=0, atol=0)
    assert torch.all(y[509:] == -17)
    _assert_policy(kernel.asm["ptx"], policy, False)


@pytest.mark.parametrize("primary", ["evict_normal", "evict_first", "evict_last", "evict_unchanged"])
@pytest.mark.parametrize("rows", [16, 32])
def test_tma_bidirectional(primary, rows, with_allocator):
    load = CachePolicy(l2=FractionalEvictionPolicy(primary))
    store = CachePolicy(l2=FractionalEvictionPolicy("evict_last"))
    x = torch.arange(rows * 64, device="cuda", dtype=torch.float32).reshape(rows, 64)
    y = torch.full_like(x, -17)
    kernel = _descriptor_copy[(1, )](x, y, rows, load, store)
    torch.testing.assert_close(y, x, rtol=0, atol=0)
    _assert_policy(kernel.asm["ptx"], load, True, True)
    _assert_policy(kernel.asm["ptx"], store, False, True)


@triton.jit
def _logical_tma_gemm(A, B, C, POLICY: tl.constexpr, POINTER: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(A, [64, 256], [256, 1], [64, 256])
    a = tle.gpu.alloc([64, 256], tl.float16)
    b = tle.gpu.alloc([80, 256], tl.float16)
    a_bar = tle.gpu.alloc_barrier(expect_bytes=64 * 256 * 2)
    tle.gpu.copy(a_desc, a, [64, 256], [0, 0], barrier=a_bar)
    tle.gpu.barrier_wait(a_bar, phaseIdx=0)
    if POINTER:
        rows = tl.arange(0, 128)
        cols = tl.arange(0, 256)
        tle.gpu.copy(B + rows[:, None] * 256 + cols[None, :], b, [80, 256], mask=rows[:, None] < 80,
                     cache_policy=POLICY)
    else:
        b_desc = tl.make_tensor_descriptor(B, [80, 256], [256, 1], [80, 256])
        b_bar = tle.gpu.alloc_barrier(expect_bytes=80 * 256 * 2)
        tle.gpu.copy(b_desc, b, [80, 256], [0, 0], barrier=b_bar, cache_policy=POLICY)
        tle.gpu.barrier_wait(b_bar, phaseIdx=0)
    result = tle.gpu.wgmma(a, b, trans_b=True, out_dtype=tl.float32)
    result = tle.gpu.wgmma_wait(0, result)
    rows = tl.arange(0, 64)
    cols = tl.arange(0, 128)
    tl.store(C + rows[:, None] * 80 + cols[None, :], result, cols[None, :] < 80)


@pytest.mark.parametrize("priority", ["evict_first", "evict_last"])
@pytest.mark.parametrize("pointer", [False, True])
def test_logical_policy_survives_split(priority, pointer, with_allocator):
    torch.manual_seed(17)
    a = torch.randn((64, 256), device="cuda", dtype=torch.float16)
    b = torch.randn((80, 256), device="cuda", dtype=torch.float16)
    c = torch.empty((64, 80), device="cuda", dtype=torch.float32)
    policy = CachePolicy(l2=FractionalEvictionPolicy(priority))
    kernel = _logical_tma_gemm[(1, )](a, b, c, policy, pointer)
    torch.testing.assert_close(c.cpu(), a.cpu().float() @ b.cpu().float().T, rtol=1e-3, atol=1e-3)
    copies = [line for line in kernel.asm["ptx"].splitlines() if "cp.async.bulk.tensor" in line]
    if pointer:
        _assert_policy(kernel.asm["ptx"], policy, True)
    else:
        assert sum(".L2::cache_hint" in line for line in copies) > 1
    assert any(".L2::cache_hint" not in line for line in copies)
    assert f"createpolicy.fractional.L2::{priority}" in kernel.asm["ptx"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int32, torch.int8, torch.bool])
def test_pointer_dtypes(dtype):
    x = torch.randint(0, 2, (512, ), device="cuda").to(dtype)
    y = torch.zeros_like(x)
    load = CachePolicy(cache_modifier="ca")
    store = CachePolicy(cache_modifier="cs")
    kernel = _pointer_copy[(1, )](x, y, 509, load, store)
    torch.testing.assert_close(y[:509], x[:509], rtol=0, atol=0)
    _assert_policy(kernel.asm["ptx"], load, True)
    _assert_policy(kernel.asm["ptx"], store, False)


def test_structurally_synchronous_copy_preserves_policy():
    # A one-element fp16 mask tail cannot use the minimum 4-byte cp.async.
    x = torch.arange(512, device="cuda", dtype=torch.float16)
    y = torch.full_like(x, -17)
    policy = CachePolicy(l1="evict_last")
    kernel = _pointer_copy[(1, )](x, y, 509, policy, None)
    torch.testing.assert_close(y[:509], x[:509], rtol=0, atol=0)
    assert torch.all(y[509:] == -17)
    assert "cp.async." not in kernel.asm["ptx"]
    _assert_policy(kernel.asm["ptx"], policy, True)


@triton.jit
def _block_copy(X, Y, POLICY: tl.constexpr):
    source = tl.make_block_ptr(X, [16, 32], [32, 1], [0, 0], [16, 32], [1, 0])
    dest = tl.make_block_ptr(Y, [16, 32], [32, 1], [0, 0], [16, 32], [1, 0])
    smem = tle.gpu.alloc([16, 32], tl.float32, nv_mma_shared_layout=False)
    tle.gpu.copy(source, smem, [16, 32], cache_policy=POLICY)
    tle.gpu.copy(smem, dest, [16, 32], cache_policy=POLICY)


def test_block_pointer_omits_unsafe_async_l2():
    x = torch.arange(512, device="cuda", dtype=torch.float32).reshape(16, 32)
    y = torch.empty_like(x)
    policy = CachePolicy(l2=FractionalEvictionPolicy("evict_last"))
    kernel = _block_copy[(1, )](x, y, policy)
    torch.testing.assert_close(y, x, rtol=0, atol=0)
    copies = [line for line in kernel.asm["ptx"].splitlines() if "cp.async." in line and "shared.global" in line]
    assert copies and all(".L2::cache_hint" not in line for line in copies)
    _assert_policy(kernel.asm["ptx"], policy, False)


@triton.jit
def _loop_copy(X, Y, LOAD: tl.constexpr, STORE: tl.constexpr, STAGES: tl.constexpr, ALL_TRUE: tl.constexpr):
    offsets = tl.arange(0, 512)
    for i in tl.range(0, 4, num_stages=STAGES):
        smem = tle.gpu.alloc([512], tl.float32, nv_mma_shared_layout=False)
        if ALL_TRUE:
            mask = tl.full([512], True, tl.int1)
        else:
            mask = offsets < 509
        tle.gpu.copy(X + i * 512 + offsets, smem, [512], mask=mask, other=7, cache_policy=LOAD)
        tle.gpu.copy(smem, Y + i * 512 + offsets, [512], mask=tl.full([512], True, tl.int1), cache_policy=STORE)


@pytest.mark.parametrize("stages", [1, 3])
@pytest.mark.parametrize("all_true", [False, True])
def test_loop_and_constant_masks(stages, all_true):
    x = torch.arange(2048, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    load = CachePolicy(l1="evict_first")
    store = CachePolicy(cache_modifier="cs", l2=FractionalEvictionPolicy("evict_last"))
    kernel = _loop_copy[(1, )](x, y, load, store, stages, all_true)
    expected = x.reshape(4, 512).clone()
    if not all_true:
        expected[:, 509:] = 7
    torch.testing.assert_close(y.reshape(4, 512), expected, rtol=0, atol=0)
    if all_true:
        assert "cp.async." in kernel.asm["ptx"]
        assert ".L1::evict_first" not in kernel.asm["ptx"]
    else:
        _assert_policy(kernel.asm["ptx"], load, True)
    _assert_policy(kernel.asm["ptx"], store, False)


@triton.jit
def _shared_copy_rejected(X, Y, POLICY: tl.constexpr):
    offsets = tl.arange(0, 128)
    a = tle.gpu.alloc([128], tl.float32, nv_mma_shared_layout=False)
    b = tle.gpu.alloc([128], tl.float32, nv_mma_shared_layout=False)
    tle.gpu.copy(X + offsets, a, [128])
    pointer = tle.gpu.local_ptr(a, [offsets])
    tle.gpu.copy(pointer, b, [128], cache_policy=POLICY)
    tle.gpu.copy(b, Y + offsets, [128])


def test_policy_rejects_shared_tensor_operand():
    x = torch.zeros((128, ), device="cuda")
    with pytest.raises(triton.CompilationError, match="requires a global-memory pointer"):
        _shared_copy_rejected[(1, )](x, x, CachePolicy(l1="evict_last"))


@triton.jit
def _pipelined_dot(A, B, C, POLICY: tl.constexpr, MODIFIER: tl.constexpr, EVICT: tl.constexpr):
    m = tl.arange(0, 32)
    n = tl.arange(0, 64)
    k = tl.arange(0, 32)
    acc = tl.zeros([32, 64], tl.float32)
    for block in tl.range(0, 4, num_stages=3):
        a = tl.load(A + m[:, None] * 128 + (block * 32 + k[None, :]), cache_policy=POLICY, cache_modifier=MODIFIER,
                    eviction_policy=EVICT)
        b = tl.load(B + (block * 32 + k[:, None]) * 64 + n[None, :])
        acc += tl.dot(a, b)
    tl.store(C + m[:, None] * 64 + n[None, :], acc)


@pytest.mark.parametrize("policy,modifier,evict", [
    (CachePolicy(cache_modifier="cv"), "", ""),
    (None, ".cv", ""),
    (CachePolicy(cache_modifier="cs"), "", ""),
    (None, ".cg", ""),
    (CachePolicy(l1="evict_last"), "", ""),
    (None, "", "evict_last"),
    (None, "", "evict_first"),
])
def test_pipeliner_converts_native_and_legacy_hints(policy, modifier, evict):
    torch.manual_seed(17)
    a = torch.randn((32, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 64), device="cuda", dtype=torch.float16)
    c = torch.empty((32, 64), device="cuda", dtype=torch.float32)
    default = _pipelined_dot[(1, )](a, b, c, None, "", "", ptx_version=87)
    kernel = _pipelined_dot[(1, )](a, b, c, policy, modifier, evict, ptx_version=87)
    assert "cp.async." in default.asm["ptx"]
    assert "cp.async." in kernel.asm["ptx"]
    assert ".L2::cache_hint" not in kernel.asm["ptx"]
    assert ".L1::evict_last" not in kernel.asm["ptx"]
    torch.testing.assert_close(c, a.float() @ b.float(), rtol=1e-3, atol=1e-3)
