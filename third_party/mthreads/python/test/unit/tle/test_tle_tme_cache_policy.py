"""Per-copy TME cache persistence, including completion and pipe paths."""

import itertools
import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler.errors import CompilationError
from triton.experimental.tle.language.gpu.mthreads import TMECachePolicy
from triton.tools.tensor_descriptor import TensorDescriptor

from test_tle_utils import compile_musa, compile_to_ttir, require_mthreads_libtriton

require_mthreads_libtriton()

POLICIES = ("none", "once", "normal", "persist")
PAIRS = [TMECachePolicy(inner, outer) for inner, outer in itertools.product(POLICIES, repeat=2)]


@triton.jit
def _copy_kernel(desc, out, row, POLICY: tl.constexpr, EXPLICIT: tl.constexpr, OMIT: tl.constexpr):
    smem = tle.gpu.alloc((16, 64), desc.dtype, nv_mma_shared_layout=False)
    if EXPLICIT:
        completion = tle.gpu.alloc_barrier(expect_bytes=16 * 64 * desc.dtype.primitive_bitwidth // 8)
        tle.gpu.copy(desc, smem, (16, 64), (row, 0), barrier=completion, cache_policy=POLICY)
        tle.gpu.barrier_wait(completion, phaseIdx=0)
    elif OMIT:
        tle.gpu.copy(desc, smem, (16, 64), (row, 0))
    else:
        tle.gpu.copy(desc, smem, (16, 64), (row, 0), cache_policy=POLICY)
    values = tl.load(tle.gpu.local_ptr(smem))
    tl.store(out + tl.arange(0, 16)[:, None] * 64 + tl.arange(0, 64)[None, :], values)


def _compile_copy(policy=None, explicit=False, omit=False):
    return compile_musa(
        _copy_kernel,
        signature={
            "desc": "tensordesc<fp16[16,64]>", "out": "*fp16", "row": "i32", "POLICY": "constexpr", "EXPLICIT":
            "constexpr", "OMIT": "constexpr"
        },
        constexprs={"POLICY": policy, "EXPLICIT": explicit, "OMIT": omit},
    )


def _tme_loads(compiled):
    return [line for line in compiled.asm["llir"].splitlines() if "call void @llvm.musa.tme.ld.tile.2d(" in line]


@pytest.mark.parametrize("policy", PAIRS)
@pytest.mark.parametrize("explicit", [False, True])
def test_tme_cache_policy_reaches_intrinsic(policy, explicit):
    compiled = _compile_copy(policy, explicit)
    inner, outer = (POLICIES.index(value) for value in (policy.inner, policy.outer))
    ttir = compiled.asm["ttir"]
    assert f"musa.tme.inner_persistence = {inner} : i32" in ttir
    assert f"musa.tme.outer_persistence = {outer} : i32" in ttir
    calls = _tme_loads(compiled)
    assert calls
    # The final operands are inner persistence, outer persistence, and l2 hint.
    assert all(re.search(rf"i32 {inner}, i32 {outer}, i32 0\)", line) for line in calls), calls
    if explicit:
        assert "musa.tme.explicit_completion" in compiled.asm["ttgir"]
        assert "musa.tme.issue_thread = 0 : i32" in compiled.asm["ttgir"]


def test_tme_cache_policy_default_is_unchanged():
    omitted = _compile_copy(omit=True)
    normal = _compile_copy(TMECachePolicy())
    assert "musa.tme.inner_persistence" not in omitted.asm["ttir"]
    # Compare the emitted load operands, ignoring debug metadata.
    loads = lambda compiled: [line.split(", !dbg")[0] for line in _tme_loads(compiled)]
    assert loads(omitted) == loads(normal)


@triton.jit
def _independent_policy_kernel(a, b):
    a_smem = tle.gpu.alloc((16, 64), a.dtype, nv_mma_shared_layout=False)
    b_smem = tle.gpu.alloc((16, 64), b.dtype, nv_mma_shared_layout=False)
    tle.gpu.copy(a, a_smem, (16, 64), (0, 0),
                 cache_policy=tle.gpu.mthreads.TMECachePolicy(inner="persist", outer="normal"))
    tle.gpu.copy(b, b_smem, (16, 64), (0, 0), cache_policy=tle.gpu.mthreads.TMECachePolicy(inner="once", outer="none"))


def test_tme_cache_policy_is_per_copy():
    compiled = compile_musa(_independent_policy_kernel,
                            signature={"a": "tensordesc<fp16[16,64]>", "b": "tensordesc<fp16[16,64]>"})
    calls = _tme_loads(compiled)
    assert len(calls) == 2
    assert re.search(r"i32 3, i32 2, i32 0\)", calls[0]), calls
    assert re.search(r"i32 1, i32 0, i32 0\)", calls[1]), calls


@triton.jit
def _grouped_policy_kernel(a, b, out):
    aa = tle.gpu.alloc((2, 16, 64), a.dtype, nv_mma_shared_layout=False)
    bb = tle.gpu.alloc((2, 16, 64), b.dtype, nv_mma_shared_layout=False)
    full = tle.gpu.alloc_barriers(2, expect_bytes=2 * 16 * 64 * a.dtype.primitive_bitwidth // 8)
    pipe = tle.pipe(capacity=2, a=aa, b=bb)
    writer = pipe.writer()
    reader = pipe.reader()
    for i in tl.static_range(4):
        slot = writer.acquire(i)
        tle.gpu.copy(a, slot.a, (16, 64), (i * 16, 0), barrier=full[i % 2],
                     cache_policy=tle.gpu.mthreads.TMECachePolicy(inner="persist", outer="normal"))
        tle.gpu.copy(b, slot.b, (16, 64), (i * 16, 0), barrier=full[i % 2],
                     cache_policy=tle.gpu.mthreads.TMECachePolicy(inner="once", outer="none"))
        writer.commit(i)
        result = reader.wait(i)
        av = tl.load(tle.gpu.local_ptr(result.slot.a)).to(tl.float32)
        bv = tl.load(tle.gpu.local_ptr(result.slot.b)).to(tl.float32)
        offsets = (i * 16 + tl.arange(0, 16))[:, None] * 64 + tl.arange(0, 64)[None, :]
        tl.store(out + offsets, av + bv)
        reader.release(i)


def test_tme_cache_policy_preserves_grouped_completion():
    compiled = compile_musa(_grouped_policy_kernel,
                            signature={"a": "tensordesc<fp16[16,64]>", "b": "tensordesc<fp16[16,64]>", "out": "*fp32"})
    calls = _tme_loads(compiled)
    assert len(calls) == 8
    assert all(re.search(r"i32 3, i32 2, i32 0\)", line) for line in calls[::2]), calls
    assert all(re.search(r"i32 1, i32 0, i32 0\)", line) for line in calls[1::2]), calls
    ttgir = compiled.asm["ttgir"]
    # The two copies share one 4096-byte transaction per iteration, even
    # though their cache hints differ. Reuse the same two completion slots.
    assert ttgir.count("ttmg.barrier_add_trans") == 4
    assert ttgir.count("ttmg.arrive_barrier_noret") == 4
    add_trans = [line for line in compiled.asm["llir"].splitlines() if "call void @llvm.musa.async.add.trans(" in line]
    assert len(add_trans) == 4
    assert all("i32 4096)" in line for line in add_trans)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_tme_cache_policy_grouped_completion_runtime(dtype):
    a = torch.randn((64, 64), device="musa", dtype=dtype)
    b = torch.randn_like(a)
    out = torch.empty((64, 64), device="musa", dtype=torch.float32)
    _grouped_policy_kernel[(1, )](TensorDescriptor.from_tensor(a, [16, 64]), TensorDescriptor.from_tensor(b, [16, 64]),
                                  out, num_warps=4, num_stages=1)
    torch.testing.assert_close(out.cpu(), a.cpu().float() + b.cpu().float(), rtol=0, atol=0)


@pytest.mark.parametrize("policy", [(), ("once", ), ("once", "none", "normal"), "once", 1, ("invalid", "none"),
                                    ("once", 0), ("once", "last_use"), ("once", "none"), ["once", "none"]])
def test_tme_cache_policy_rejects_invalid_values(policy):
    with pytest.raises(CompilationError, match="cache_policy"):
        _compile_copy(policy)


@triton.jit
def _dynamic_policy_kernel(desc, policy):
    smem = tle.gpu.alloc((16, 64), desc.dtype, nv_mma_shared_layout=False)
    tle.gpu.copy(desc, smem, (16, 64), (0, 0), cache_policy=tle.gpu.mthreads.TMECachePolicy(inner=policy, outer="none"))


def test_tme_cache_policy_must_be_compile_time():
    with pytest.raises(CompilationError, match="TMECachePolicy inner must be a compile-time string"):
        compile_to_ttir(_dynamic_policy_kernel, signature={"desc": "tensordesc<fp16[16,64]>", "policy": "i32"})


@triton.jit
def _unsupported_copy_kernel(desc, ptr, STORE: tl.constexpr):
    smem = tle.gpu.alloc((16, 64), desc.dtype, nv_mma_shared_layout=False)
    if STORE:
        tle.gpu.copy(smem, desc, (16, 64), (0, 0),
                     cache_policy=tle.gpu.mthreads.TMECachePolicy(inner="once", outer="none"))
    else:
        offsets = tl.arange(0, 16)[:, None] * 64 + tl.arange(0, 64)[None, :]
        tle.gpu.copy(ptr + offsets, smem, (16, 64),
                     cache_policy=tle.gpu.mthreads.TMECachePolicy(inner="once", outer="none"))


@pytest.mark.parametrize("store", [False, True])
def test_tme_cache_policy_rejects_unsupported_direction(store):
    with pytest.raises(CompilationError, match="only supported for descriptor-to-shared copies"):
        compile_to_ttir(_unsupported_copy_kernel,
                        signature={"desc": "tensordesc<fp16[16,64]>", "ptr": "*fp16", "STORE":
                                   "constexpr"}, constexprs={"STORE": store})


@pytest.mark.parametrize("enabled,common_ir", [(False, False), (True, True)])
def test_tme_cache_policy_rejects_other_backends(monkeypatch, enabled, common_ir):
    from triton.experimental.tle.language.gpu import core
    monkeypatch.setattr(core.mthreads_common, "enabled", lambda: enabled)
    monkeypatch.setattr(core.tle_semantic, "COMMON_IR_ENABLED", common_ir)
    with pytest.raises(ValueError, match="copy cache_policy"):
        core.copy(None, None, (), cache_policy=tle.gpu.mthreads.TMECachePolicy(inner="once", outer="none"),
                  _semantic=object())


@triton.jit
def _pipe_copy_kernel(src, dst):
    layout: tl.constexpr = tle.gpu.swizzled_shared_layout(16, 1, 8, [2, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                                                          [2, 1, 0])
    smem = tle.gpu.alloc((2, 32, 256), src.dtype, layout=layout)
    full = tle.gpu.alloc_barriers(2, expect_bytes=32 * 256 * 2)
    pipe = tle.pipe(capacity=2, name="cache_policy", data=smem)
    writer = pipe.writer()
    reader = pipe.reader()
    for iteration in tl.static_range(4):
        slot = writer.acquire(iteration)
        tle.gpu.copy(src, slot.data, (32, 256), ((iteration + 1) * 32, 128), barrier=full[iteration % 2],
                     cache_policy=tle.gpu.mthreads.TMECachePolicy(inner="once", outer="none"))
        writer.commit(iteration)
        wait = reader.wait(iteration)
        tle.gpu.copy(wait.slot.data, dst, (32, 256), ((iteration + 1) * 32, 128))
        reader.release(iteration)


def test_tme_cache_policy_survives_pipe_and_swizzle():
    compiled = compile_musa(_pipe_copy_kernel,
                            signature={"src": "tensordesc<fp16[32,256]>", "dst": "tensordesc<fp16[32,256]>"})
    calls = _tme_loads(compiled)
    # Each 256-column tile is segmented into two 128-column TME loads.
    assert len(calls) == 8
    assert all("<2 x i32> <i32 128, i32 32>" in line for line in calls)
    assert all(re.search(r"i32 1, i32 0, i32 0\)", line) for line in calls), calls
    ttgir = compiled.asm["ttgir"]
    assert "maxPhase = 8" in ttgir
    assert ttgir.count("ttmg.barrier_add_trans") == 4
    assert ttgir.count("ttmg.arrive_barrier_noret") == 4


@triton.jit
def _ws_producer(desc, smem, full, POLICY: tl.constexpr):
    tle.gpu.copy(desc, smem, (16, 64), (0, 0), barrier=full, cache_policy=POLICY)


@triton.jit
def _ws_producer_split(desc, smem, full, INNER: tl.constexpr, OUTER: tl.constexpr):
    tle.gpu.copy(desc, smem, (16, 64), (0, 0), barrier=full,
                 cache_policy=tle.gpu.mthreads.TMECachePolicy(inner=INNER, outer=OUTER))


@triton.jit
def _ws_consumer(smem, full, out):
    tle.gpu.barrier_wait(full, phaseIdx=0)
    values = tl.load(tle.gpu.local_ptr(smem))
    offsets = tl.arange(0, 16)[:, None] * 64 + tl.arange(0, 64)[None, :]
    tl.store(out + offsets, values)


@triton.jit
def _ws_copy_kernel(desc, out, POLICY: tl.constexpr, SPLIT_POLICY: tl.constexpr):
    smem = tle.gpu.alloc((16, 64), desc.dtype, nv_mma_shared_layout=False)
    full = tle.gpu.alloc_barrier(expect_bytes=16 * 64 * desc.dtype.primitive_bitwidth // 8)
    if SPLIT_POLICY:
        inner: tl.constexpr = POLICY.inner
        outer: tl.constexpr = POLICY.outer
        tle.gpu.warp_specialize([(_ws_consumer, (smem, full, out)),
                                 (_ws_producer_split, (desc, smem, full, inner, outer))], worker_num_warps=[4],
                                worker_num_regs=[24])
    else:
        tle.gpu.warp_specialize([(_ws_consumer, (smem, full, out)), (_ws_producer, (desc, smem, full, POLICY))],
                                worker_num_warps=[4], worker_num_regs=[24])


@pytest.mark.parametrize("policy", PAIRS)
@pytest.mark.parametrize("split_policy", [False, True], ids=["object", "strings"])
def test_tme_cache_policy_preserves_warp_specialization(policy, split_policy):
    compiled = compile_musa(
        _ws_copy_kernel, signature={
            "desc": "tensordesc<fp16[16,64]>", "out": "*fp16", "POLICY": "constexpr", "SPLIT_POLICY": "constexpr"
        }, constexprs={"POLICY": policy, "SPLIT_POLICY": split_policy})
    calls = _tme_loads(compiled)
    assert len(calls) == 1
    inner, outer = (POLICIES.index(value) for value in (policy.inner, policy.outer))
    assert re.search(rf"i32 {inner}, i32 {outer}, i32 0\)", calls[0]), calls
    assert "musa.tme.issue_thread = 128 : i32" in compiled.asm["ttgir"]
    assert "call void @llvm.musa.barrier0()" not in compiled.asm["llir"]


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
@pytest.mark.parametrize("policy", PAIRS)
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_tme_cache_policy_runtime(policy, explicit, dtype):
    source = torch.randn((48, 64), device="musa", dtype=dtype)
    out = torch.empty((16, 64), device="musa", dtype=dtype)
    desc = TensorDescriptor.from_tensor(source, [16, 64])
    for row in (16, 32):
        _copy_kernel[(1, )](desc, out, row, policy, explicit, False, num_warps=4, num_stages=1)
        torch.testing.assert_close(out.cpu(), source[row:row + 16].cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tme_cache_policy_pipe_reuse_runtime():
    src = torch.randn((192, 512), dtype=torch.float16, device="musa")
    dst = torch.full_like(src, -23)
    src_desc = TensorDescriptor.from_tensor(src, [32, 256])
    dst_desc = TensorDescriptor.from_tensor(dst, [32, 256])
    _pipe_copy_kernel[(1, )](src_desc, dst_desc, num_warps=4, num_stages=1)
    expected = torch.full_like(src, -23)
    expected[32:160, 128:384] = src[32:160, 128:384]
    torch.testing.assert_close(dst.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
@pytest.mark.parametrize("policy", PAIRS)
@pytest.mark.parametrize("split_policy", [False, True], ids=["object", "strings"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_tme_cache_policy_warp_specialization_runtime(policy, split_policy, dtype):
    src = torch.randn((16, 64), dtype=dtype, device="musa")
    dst = torch.empty_like(src)
    desc = TensorDescriptor.from_tensor(src, [16, 64])
    _ws_copy_kernel[(1, )](desc, dst, policy, split_policy, num_warps=4, num_stages=1)
    torch.testing.assert_close(dst.cpu(), src.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tme_cache_policy_jit_cache_runtime():
    source = torch.randn((16, 64), device="musa", dtype=torch.float16)
    out = torch.empty_like(source)
    desc = TensorDescriptor.from_tensor(source, [16, 64])
    first = _copy_kernel[(1, )](desc, out, 0, TMECachePolicy("once", "none"), True, False, num_warps=4, num_stages=1)
    equal = _copy_kernel[(1, )](desc, out, 0, TMECachePolicy(inner="once", outer="none"), True, False, num_warps=4,
                                num_stages=1)
    different = _copy_kernel[(1, )](desc, out, 0, TMECachePolicy("normal", "none"), True, False, num_warps=4,
                                    num_stages=1)
    assert first is equal
    assert first is not different
    assert "musa.tme.inner_persistence = 1 : i32" in first.asm["ttir"]
    assert "musa.tme.inner_persistence = 2 : i32" in different.asm["ttir"]
    torch.testing.assert_close(out.cpu(), source.cpu(), rtol=0, atol=0)
