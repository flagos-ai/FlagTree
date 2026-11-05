"""Native policy construction, default copies, visibility and worker forwarding."""
import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._flagtree_backend import FLAGTREE_BACKEND

BACKEND = FLAGTREE_BACKEND or "nvidia"
DEVICE = "musa" if BACKEND == "mthreads" else "cuda"
POLICY_ATTR = {
    "nvidia": "evictionPolicy = l2_evict_first",
    "mthreads": "cacheModifier = musa_i3_o3_",
    "hcu": "cacheModifier = cs",
    "metax": "cacheModifier = cs",
    "ppu": "cacheModifier = cs",
}[BACKEND]
if BACKEND == "mthreads":
    import torch_musa  # noqa: F401


@triton.jit
def _copy(X, Y, N: tl.constexpr, POLICY: tl.constexpr, IN_JIT: tl.constexpr, OMIT: tl.constexpr = False):
    x = tl.program_id(0) * 512 + tl.arange(0, 512)
    buf = tle.gpu.alloc((512, ), tl.float32, nv_mma_shared_layout=False)
    if IN_JIT:
        policy: tl.constexpr = tle.gpu.CachePolicy(cache_modifier=POLICY.cache_modifier, l1=POLICY.l1, l2=POLICY.l2,
                                                   l2_prefetch_size=POLICY.l2_prefetch_size)
    else:
        policy: tl.constexpr = POLICY
    if OMIT:
        tle.gpu.copy(X + x, buf, (512, ), mask=x < N, other=7)
        tle.gpu.copy(buf, Y + x, (512, ), mask=x < N)
    else:
        tle.gpu.copy(X + x, buf, (512, ), mask=x < N, other=7, cache_policy=policy)
        tle.gpu.copy(buf, Y + x, (512, ), mask=x < N, cache_policy=policy)


def _memory_code(kernel):
    text = kernel.asm.get("ptx", kernel.asm["llir"])
    return [
        re.sub(r",? !dbg !\d+", "", line.strip()) for line in text.splitlines() if any(
            s in line
            for s in ("ld.global", "st.global", "cp.async", "createpolicy", "lsu.ld", "lsu.st", "nontemporal", "l2rp"))
    ]


def _policy():
    return {
        "nvidia": tle.gpu.CachePolicy(l2=tle.gpu.FractionalEvictionPolicy("evict_first")),
        "mthreads": tle.gpu.mthreads.LSUCachePolicy(inner_persistence=3, outer_persistence=3),
        "hcu": tle.gpu.hcu.CachePolicy(".cs"),
        "metax": tle.gpu.metax.CachePolicy("cs"),
        "ppu": tle.gpu.ppu.CachePolicy(4),
    }[BACKEND]


@pytest.mark.parametrize("n", [509, 1 << 20])
def test_none_matches_omission_and_preserves_gpu_visibility(n):
    x = torch.zeros(n + 3, device=DEVICE)
    y = torch.full_like(x, -11)
    baseline = _copy[(triton.cdiv(n, 512), )](x, y, n, None, False, True)
    for value in (13., 29.):
        x.fill_(value)
        y.fill_(-11)
        kernel = _copy[(triton.cdiv(n, 512), )](x, y, n, None, False)
        consumed = y.clone().cpu()
        assert torch.all(consumed[:n] == value)
        assert torch.all(consumed[n:] == -11)
    assert _memory_code(kernel) == _memory_code(baseline)
    assert POLICY_ATTR not in kernel.asm["ttir"]


def test_equal_native_values_share_jit_specialization():
    x = torch.zeros(512, device=DEVICE)
    first = _copy[(1, )](x, x, 512, _policy(), False)
    equal = _copy[(1, )](x, x, 512, _policy(), False)
    assert first.hash == equal.hash


def test_native_hint_tail_after_gpu_write():
    n = (1 << 20) - 3
    x = torch.zeros(n + 3, device=DEVICE)
    y = torch.full_like(x, -11)
    for value in (13., 29.):
        x.fill_(value)
        y.fill_(-11)
        kernel = _copy[(triton.cdiv(n, 512), )](x, y, n, _policy(), False)
        consumed = y.clone().cpu()
        assert torch.all(consumed[:n] == value)
        assert torch.all(consumed[n:] == -11)
    assert POLICY_ATTR in kernel.asm["ttir"]


@triton.jit
def _async_copy(X, Y, N: tl.constexpr, POLICY: tl.constexpr):
    x = tl.program_id(0) * 512 + tl.arange(0, 512)
    buf = tle.gpu.alloc((512, ), tl.float32, nv_mma_shared_layout=False)
    tle.gpu.copy(X + x, buf, (512, ), mask=x < N, other=0, cache_policy=POLICY)
    tle.gpu.copy(buf, Y + x, (512, ), mask=x < N)


@triton.jit
def _legacy_shared_fill(X, Y, N, FILL: tl.constexpr):
    i = tl.arange(0, 512)
    buf = tle.gpu.alloc((512, ), tl.float32, nv_mma_shared_layout=False)
    tle.gpu.copy(X + i, buf, (512, ), mask=i < N, other=FILL)
    tle.gpu.copy(buf, Y + i, (512, ))


@pytest.mark.skipif(BACKEND != "mthreads", reason="MUSA memcpy.g2s zero-fills masked lanes")
@pytest.mark.parametrize("fill", [0, 7])
def test_musa_legacy_shared_copy_preserves_masked_fill(fill):
    x = torch.arange(512, dtype=torch.float32, device=DEVICE)
    y = torch.full_like(x, -1)
    kernel = _legacy_shared_fill[(1, )](x, y, 504, fill)
    torch.testing.assert_close(y[:504], x[:504], rtol=0, atol=0)
    assert torch.all(y[504:] == fill)
    assert ("ttg.async_copy_global_to_local" in kernel.asm["ttgir"]) == (fill == 0)


@pytest.mark.parametrize("n", [509, 512])
def test_native_hint_preserves_selected_copy_transport(n):
    x = torch.zeros(n + 3, device=DEVICE)
    y = torch.full_like(x, -11)
    baseline = _async_copy[(triton.cdiv(n, 512), )](x, y, n, None)
    policy = tle.gpu.CachePolicy(l1="evict_last") if BACKEND == "nvidia" else _policy()
    for value in (13., 29.):
        x.fill_(value)
        y.fill_(-11)
        kernel = _async_copy[(triton.cdiv(n, 512), )](x, y, n, policy)
        consumed = y.clone().cpu()
        assert torch.all(consumed[:n] == value)
        assert torch.all(consumed[n:] == -11)
    # Compare instruction families, not hints or debug locations. A structural
    # fallback may apply on a backend, but must apply to both variants equally.
    for operation in ("ttg.async_copy_global_to_local", "async_aiu_copy", "buffer_load", "tt.load"):
        assert kernel.asm["ttgir"].count(operation) == baseline.asm["ttgir"].count(operation)
    if BACKEND == "nvidia":
        assert "cp.async." in kernel.asm["ptx"]
        assert ".L1::evict_last" not in kernel.asm["ptx"]


@pytest.mark.skipif(BACKEND != "nvidia", reason="NVIDIA conversion diagnostic")
def test_cache_hint_adjustment_remark(monkeypatch, tmp_path, capfd):
    monkeypatch.setenv("MLIR_ENABLE_DIAGNOSTICS", "remarks")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    x = torch.zeros(512, device=DEVICE)
    y = torch.empty_like(x)
    _async_copy[(1, )](x, y, 507, tle.gpu.CachePolicy(cache_modifier="cs"))
    diagnostic = capfd.readouterr().err
    assert "cache hint on NVIDIA cp.async" in diagnostic
    assert "cache=cs" in diagnostic


@pytest.mark.skipif(BACKEND != "nvidia", reason="NVIDIA native fields")
@pytest.mark.parametrize("l1", [None, "evict_first", "evict_last", "cg"])
@pytest.mark.parametrize("l2", [None, "evict_first", "evict_last"])
def test_nvidia_host_and_jit_construction_have_equal_codegen(l1, l2):
    if l1 is None and l2 is None:
        pytest.skip("covered by default tests")
    if l1 in ("evict_first", "evict_last") and l2 is not None:
        with pytest.raises(ValueError, match="cannot be combined"):
            tle.gpu.CachePolicy(l1=l1, l2=tle.gpu.FractionalEvictionPolicy(l2))
        return
    policy = tle.gpu.CachePolicy(cache_modifier="cg" if l1 == "cg" else None, l1=l1 if l1 != "cg" else None,
                                 l2=tle.gpu.FractionalEvictionPolicy(l2) if l2 else None)
    x = torch.zeros(512, device=DEVICE)
    y = torch.zeros_like(x)
    original = _copy[(1, )](x, y, 509, policy, False)
    for value in (17., 31.):
        x.fill_(value)
        y.fill_(-1)
        kernel = _copy[(1, )](x, y, 509, policy, True)
        consumed = y.clone().cpu()
        assert torch.all(consumed[:509] == value)
        assert torch.all(consumed[509:] == -1)
    assert _memory_code(kernel) == _memory_code(original)


@triton.jit
def _producer(desc, buf, full, POLICY: tl.constexpr):
    tle.gpu.copy(desc, buf, (16, 64), (0, 0), barrier=full, cache_policy=POLICY)


@triton.jit
def _producer_string(desc, buf, full, L2: tl.constexpr):
    if L2 is None:
        policy: tl.constexpr = None
    else:
        policy: tl.constexpr = tle.gpu.CachePolicy(l2=tle.gpu.FractionalEvictionPolicy(L2))
    tle.gpu.copy(desc, buf, (16, 64), (0, 0), barrier=full, cache_policy=policy)


@triton.jit
def _consumer(buf, full, out):
    tle.gpu.barrier_wait(full, phaseIdx=0)
    value = tl.load(tle.gpu.local_ptr(buf))
    offsets = tl.arange(0, 16)[:, None] * 64 + tl.arange(0, 64)[None, :]
    tl.store(out + offsets, value)


@triton.jit
def _ws(desc, out, POLICY: tl.constexpr, STRING: tl.constexpr):
    buf = tle.gpu.alloc((16, 64), tl.float32)
    full = tle.gpu.alloc_barrier(expect_bytes=16 * 64 * 4)
    if STRING:
        if POLICY is None:
            l2: tl.constexpr = None
        else:
            l2: tl.constexpr = POLICY.l2.primary
        tle.gpu.warp_specialize([(_consumer, (buf, full, out)), (_producer_string, (desc, buf, full, l2))],
                                worker_num_warps=[4], worker_num_regs=[24])
    else:
        tle.gpu.warp_specialize([(_consumer, (buf, full, out)), (_producer, (desc, buf, full, POLICY))],
                                worker_num_warps=[4], worker_num_regs=[24])


@pytest.mark.skipif(BACKEND != "nvidia", reason="NVIDIA TMA worker")
@pytest.mark.parametrize("priority", [None, "evict_first", "evict_last"])
@pytest.mark.parametrize("string", [False, True])
def test_warp_specialize_object_string_and_none(priority, string):
    from triton.tools.tensor_descriptor import TensorDescriptor
    x = torch.arange(1024, device=DEVICE, dtype=torch.float32).reshape(16, 64)
    y = torch.empty_like(x)
    policy = None if priority is None else tle.gpu.CachePolicy(l2=tle.gpu.FractionalEvictionPolicy(priority))
    desc = TensorDescriptor.from_tensor(x, [16, 64])
    kernel = _ws[(1, )](desc, y, policy, string)
    torch.testing.assert_close(y, x, rtol=0, atol=0)
    assert "cp.async.bulk.tensor" in kernel.asm["ptx"]
    if priority is None:
        assert ".L2::cache_hint" not in kernel.asm["ptx"]
    else:
        assert ".L2::cache_hint" in kernel.asm["ptx"]
        assert f"createpolicy.fractional.L2::{priority}" in kernel.asm["ptx"]


def test_backend_mismatch_fails_before_launch():
    x = torch.zeros(512, device=DEVICE)
    wrong = tle.gpu.ppu.CachePolicy(4) if BACKEND == "nvidia" else tle.gpu.CachePolicy(l1="evict_first")
    with pytest.raises(triton.CompilationError, match="backend/path"):
        _copy[(1, )](x, x, 512, wrong, False)
