import re

import pytest
import torch
import triton
import triton.language as tl


def _tle_enabled():
    try:
        import triton.experimental.tle.language  # noqa: F401
        from triton.backends.metax import compiler as metax_compiler
    except Exception:
        return False
    return getattr(metax_compiler, "enable_mctle", False) is True


pytestmark = pytest.mark.skipif(not _tle_enabled(), reason="requires a metax build with mctle")

if _tle_enabled():
    import triton.experimental.tle.language as tle
else:  # pragma: no cover - module is skipped
    tle = None

DEVICE = "cuda"
INT32_BYTES = 4


def _ttir_has_shared_atomic(kernel, op):
    ttir = str(kernel.asm["ttir"])
    return any(op in line and "tt.ptr<i32, 3>" in line for line in ttir.splitlines())


# --------------------------------------
# test atomics on shared-memory pointers
# --------------------------------------


@triton.jit
def _smem_atomic_add_kernel(old_out, new_out, N: tl.constexpr):
    i = tl.arange(0, N)
    buf = tle.gpu.alloc([N], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    p = tle.gpu.local_ptr(buf, (i, ))
    tl.store(p, i)
    tl.debug_barrier()
    old = tl.atomic_add(p, 7, sem="relaxed", scope="cta")
    tl.debug_barrier()
    tl.store(old_out + i, old)
    tl.store(new_out + i, tl.load(p))


@triton.jit
def _smem_atomic_cas_kernel(old_out, new_out, N: tl.constexpr):
    i = tl.arange(0, N)
    buf = tle.gpu.alloc([N], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    p = tle.gpu.local_ptr(buf, (i, ))
    tl.store(p, i)
    tl.debug_barrier()
    # even lanes compare equal and are replaced, odd lanes are left alone
    cmp = tl.where(i % 2 == 0, i, -1)
    old = tl.atomic_cas(p, cmp, i + 1000, sem="relaxed", scope="cta")
    tl.debug_barrier()
    tl.store(old_out + i, old)
    tl.store(new_out + i, tl.load(p))


def test_tle_atomic_add_on_shared_pointer():
    n = 256
    old = torch.full((n, ), -1, dtype=torch.int32, device=DEVICE)
    new = torch.full_like(old, -1)
    k = _smem_atomic_add_kernel[(1, )](old, new, N=n, num_warps=4)
    torch.cuda.synchronize()
    assert _ttir_has_shared_atomic(k, "tt.atomic_rmw")
    ref = torch.arange(n, dtype=torch.int32, device=DEVICE)
    torch.testing.assert_close(old, ref)
    torch.testing.assert_close(new, ref + 7)


def test_tle_atomic_cas_on_shared_pointer():
    n = 256
    old = torch.full((n, ), -1, dtype=torch.int32, device=DEVICE)
    new = torch.full_like(old, -1)
    k = _smem_atomic_cas_kernel[(1, )](old, new, N=n, num_warps=4)
    torch.cuda.synchronize()
    assert _ttir_has_shared_atomic(k, "tt.atomic_cas")
    ref = torch.arange(n, dtype=torch.int32, device=DEVICE)
    torch.testing.assert_close(old, ref)
    torch.testing.assert_close(new, torch.where(ref % 2 == 0, ref + 1000, ref))


# -----------------------------------
# test two live shared-memory buffers
# -----------------------------------


# Loads through shared-memory pointers are masked and the masked-out slot is
# never compared: unmasked scalar-index loads and `other` on masked loads are
# separate, still-open plugin issues.
@triton.jit
def _two_buffers_tensor_index_kernel(out_a, out_b, N: tl.constexpr):
    i = tl.arange(0, N)
    mask = i < N - 1
    a = tle.gpu.alloc([N], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    pa = tle.gpu.local_ptr(a, (i, ))
    tl.store(pa, i + 100, mask=mask)
    tl.debug_barrier()
    # allocated after a's last direct use; a is still read below through pa
    b = tle.gpu.alloc([N], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    pb = tle.gpu.local_ptr(b, (i, ))
    tl.store(pb, i + 10000, mask=mask)
    tl.debug_barrier()
    tl.store(out_a + i, tl.load(pa, mask=mask, other=0))
    tl.store(out_b + i, tl.load(pb, mask=mask, other=0))


@triton.jit
def _two_buffers_pointer_chain_kernel(out_a, out_b, N: tl.constexpr):
    i = tl.arange(0, N)
    mask = i < N - 1
    a = tle.gpu.alloc([N], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    qa = tle.gpu.local_ptr(a, (0, )) + i  # tt.splat + tt.addptr over a scalar local pointer
    tl.store(qa, i + 100, mask=mask)
    tl.debug_barrier()
    b = tle.gpu.alloc([N], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    qb = tle.gpu.local_ptr(b, (0, )) + i
    tl.store(qb, i + 10000, mask=mask)
    tl.debug_barrier()
    tl.store(out_a + i, tl.load(qa, mask=mask, other=0))
    tl.store(out_b + i, tl.load(qb, mask=mask, other=0))


@pytest.mark.parametrize("kernel, n", [
    (_two_buffers_tensor_index_kernel, 512),
    (_two_buffers_pointer_chain_kernel, 256),
], ids=["tensor_index", "pointer_chain"])
def test_tle_two_live_buffers_do_not_overlap(kernel, n):
    out_a = torch.full((n, ), -1, dtype=torch.int32, device=DEVICE)
    out_b = torch.full_like(out_a, -1)
    k = kernel[(1, )](out_a, out_b, N=n, num_warps=4)
    torch.cuda.synchronize()
    assert k.metadata.shared >= 2 * n * INT32_BYTES, f"shared={k.metadata.shared}"
    ref = torch.arange(n - 1, dtype=torch.int32, device=DEVICE)
    torch.testing.assert_close(out_a[:-1], ref + 100)
    torch.testing.assert_close(out_b[:-1], ref + 10000)


# -----------------------------------------------
# test a buffer live across shared-memory scratch
# -----------------------------------------------


@triton.jit
def _buffer_across_histogram_kernel(inp, out, N: tl.constexpr, BINS: tl.constexpr):
    i = tl.arange(0, N)
    buf = tle.gpu.alloc([N], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    v = tl.load(inp + i)
    tl.store(tle.gpu.local_ptr(buf, (i, )), v)
    tl.debug_barrier()
    h = tl.histogram(v % BINS, BINS)
    tl.debug_barrier()
    back = tl.load(tle.gpu.local_ptr(buf, (i, )))
    tl.store(out + i, back + tl.sum(h, axis=0) * 0)


@triton.jit
def _buffer_across_reductions_kernel(out, flags, N: tl.constexpr):
    i = tl.arange(0, N)
    buf = tle.gpu.alloc([N], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    p = tle.gpu.local_ptr(buf, (i, ))
    tl.store(p, i + 5)
    tl.debug_barrier()
    total = tl.sum(i, axis=0)  # cross-warp reductions use shared-memory scratch
    prefix = tl.cumsum(i, axis=0)
    tl.debug_barrier()
    tl.store(out + i, tl.load(p))
    tl.store(flags, (total == N * (N - 1) // 2).to(tl.int32))
    tl.store(flags + 1, (tl.sum((prefix != i * (i + 1) // 2).to(tl.int32), axis=0) == 0).to(tl.int32))


def test_tle_buffer_live_across_histogram():
    n, bins = 4096, 256
    inp = torch.randint(0, 1 << 16, (n, ), dtype=torch.int32, device=DEVICE)
    out = torch.full_like(inp, -1)
    k = _buffer_across_histogram_kernel[(1, )](inp, out, N=n, BINS=bins, num_warps=8)
    torch.cuda.synchronize()
    assert k.metadata.shared >= n * INT32_BYTES, f"shared={k.metadata.shared}"
    torch.testing.assert_close(out, inp)


def test_tle_buffer_live_across_reductions():
    n = 512
    out = torch.full((n, ), -1, dtype=torch.int32, device=DEVICE)
    flags = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    k = _buffer_across_reductions_kernel[(1, )](out, flags, N=n, num_warps=8)
    torch.cuda.synchronize()
    assert k.metadata.shared > n * INT32_BYTES, f"shared={k.metadata.shared}"
    torch.testing.assert_close(out, torch.arange(n, dtype=torch.int32, device=DEVICE) + 5)
    assert flags.tolist() == [1, 1]


# -------------------------------------------
# test a pointer that is not the first result
# -------------------------------------------


@triton.jit
def _pointer_as_second_result_kernel(out, N: tl.constexpr):
    idx = tl.arange(0, N)
    mask = idx < N - 1
    a = tle.gpu.alloc([N], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    pa = tle.gpu.local_ptr(a, (idx, ))
    tl.store(pa, idx + 100, mask=mask)
    tl.debug_barrier()
    # result 0 is an int, result 1 forwards pa
    tag, forwarded = tl.inline_asm_elementwise(
        asm="",
        constraints="=r,=r,1,0,~{memory}",
        args=[pa, idx],
        dtype=(tl.int32, pa.dtype),
        is_pure=False,
        pack=1,
    )
    b = tle.gpu.alloc([N], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    pb = tle.gpu.local_ptr(b, (idx, ))
    tl.store(pb, tag + 10000, mask=mask)
    tl.debug_barrier()
    tl.store(out + idx, tl.load(forwarded, mask=mask, other=0))


def test_tle_pointer_as_second_result_keeps_buffer_live():
    n = 256
    out = torch.full((n, ), -1, dtype=torch.int32, device=DEVICE)
    k = _pointer_as_second_result_kernel[(1, )](out, N=n, num_warps=4)
    torch.cuda.synchronize()
    assert re.search(r"inline_asm", str(k.asm["ttir"]))
    assert k.metadata.shared >= 2 * n * INT32_BYTES, f"shared={k.metadata.shared}"
    torch.testing.assert_close(out[:-1], torch.arange(n - 1, dtype=torch.int32, device=DEVICE) + 100)
