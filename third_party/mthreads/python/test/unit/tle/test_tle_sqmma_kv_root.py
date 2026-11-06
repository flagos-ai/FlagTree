"""A staged KV allocation can serve both K-transpose and V consumers."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.tools.tensor_descriptor import TensorDescriptor

from test_tle_utils import compile_musa
from test_tle_warp_specialize import _compile_ws_ir


@triton.jit
def shared_kv_kernel(Q, KV, P, QK, PV, DTYPE: tl.constexpr, STAGES: tl.constexpr, REVERSE: tl.constexpr = False,
                     REPEATS: tl.constexpr = 1, N: tl.constexpr = 64, K: tl.constexpr = 256,
                     TRANSPOSE: tl.constexpr = True):
    rows = tl.arange(0, 32)[:, None]
    query = tl.load(Q + rows * K + tl.arange(0, K)[None, :])
    key_value = tl.load(KV + tl.arange(0, STAGES)[:, None, None] * (N * K) + tl.arange(0, N)[None, :, None] * K +
                        tl.arange(0, K)[None, None, :])
    probability = tl.load(P + rows * N + tl.arange(0, N)[None, :])
    q = tle.gpu.alloc((32, K), DTYPE, init_value=query)
    kv = tle.gpu.alloc((STAGES, N, K), DTYPE, init_value=key_value)
    p = tle.gpu.alloc((32, N), DTYPE, init_value=probability)
    tl.debug_barrier()
    tile = kv.slot(STAGES - 1)
    scores = tl.zeros((32, N), tl.float32)
    output = tl.zeros((32, K), tl.float32)
    for _ in tl.static_range(REPEATS):
        if REVERSE:
            output = tle.gpu.wgmma(p, tile, output)
            output = tle.gpu.wgmma_wait(0, output)
            scores = tle.gpu.wgmma(q, tile, scores, trans_b=TRANSPOSE)
            scores = tle.gpu.wgmma_wait(0, scores)
        else:
            scores = tle.gpu.wgmma(q, tile, scores, trans_b=TRANSPOSE)
            scores = tle.gpu.wgmma_wait(0, scores)
            output = tle.gpu.wgmma(p, tile, output)
            output = tle.gpu.wgmma_wait(0, output)
    tl.store(QK + rows * N + tl.arange(0, N)[None, :], scores)
    tl.store(PV + rows * K + tl.arange(0, K)[None, :], output)


def compile_shared_kv(dtype, stages, reverse, repeats=1, n=64, k=256, transpose=True):
    pointer = "*fp8e4nv" if dtype == tl.float8e4nv else "*bf16"
    return _compile_ws_ir(
        shared_kv_kernel,
        {
            "Q": pointer, "KV": pointer, "P": pointer, "QK": "*fp32", "PV": "*fp32",
            **{name: "constexpr"
               for name in ("DTYPE", "STAGES", "REVERSE", "REPEATS", "N", "K", "TRANSPOSE")}
        },
        {
            "DTYPE": dtype, "STAGES": stages, "REVERSE": reverse, "REPEATS": repeats, "N": n, "K": k, "TRANSPOSE":
            transpose
        },
        consumer_warps=4,
        include_late=True,
    )


@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("repeats", [1, 2])
def test_shared_kv_consumer_order(stages, repeats):
    roots = []
    for reverse in (False, True):
        _, ttgir, lowered = compile_shared_kv(tl.float8e4nv, stages, reverse, repeats)
        assert ttgir.count(" = ttmg.squad_dot ") == 2 * repeats
        assert ttgir.count(" = ttg.local_alloc ") == 3
        assert "llvm.musa.sqmma.e4m3" in lowered
        encoding = re.search(rf"ttg.local_alloc.*!ttg.memdesc<{stages}x64x256xf8E4M3FN, (#[a-z0-9]+),", ttgir)[1]
        roots.append(re.search(rf"^{encoding} = (.*)$", ttgir, re.MULTILINE)[1])
    assert roots[0] == roots[1]


@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("repeats", [1, 2])
def test_shared_kv_rejects_incompatible_physical_layouts(stages, reverse, repeats, capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        compile_shared_kv(tl.bfloat16, stages, reverse, repeats)
    assert "SQMMA consumers require incompatible physical shared layouts" in capfd.readouterr().err


@pytest.mark.parametrize("stages", [1, 2])
def test_shared_kv_bf16_compatible_views(stages):
    # Two untransposed consumers require the same physical BF16 layout.
    _, ttgir, lowered = compile_shared_kv(tl.bfloat16, stages, False, n=64, k=64, transpose=False)
    assert ttgir.count(" = ttg.local_alloc ") == 3
    assert ttgir.count(" = ttmg.squad_dot ") == 2
    assert "llvm.musa.sqmma.bfmma" in lowered


@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("reverse,repeats", [(False, 1), (True, 2)])
@pytest.mark.parametrize("dtype,n,k,transpose", [(torch.float8_e4m3fn, 64, 256, True), (torch.bfloat16, 64, 64, False)])
def test_shared_kv_runtime_precision(stages, reverse, repeats, dtype, n, k, transpose):
    import torch_musa

    if not torch_musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.manual_seed(1450)
    torch_dtype = dtype
    query = (torch.randn((32, k)) * 0.25).to(torch_dtype)
    key_value = (torch.randn((stages, n, k)) * 0.25).to(torch_dtype)
    probability = (torch.randn((32, n)) * 0.25).to(torch_dtype)
    tile = key_value[-1].double()
    expected_qk = repeats * (query.double() @ (tile.T if transpose else tile))
    expected_pv = repeats * (probability.double() @ tile)

    qk = torch.empty((32, n), dtype=torch.float32, device="musa")
    pv = torch.empty((32, k), dtype=torch.float32, device="musa")
    kernel = shared_kv_kernel[(1, )](
        query.to("musa"),
        key_value.to("musa"),
        probability.to("musa"),
        qk,
        pv,
        tl.float8e4nv if dtype == torch.float8_e4m3fn else tl.bfloat16,
        stages,
        reverse,
        repeats,
        n,
        k,
        transpose,
        num_warps=4,
        num_stages=1,
    )
    torch.musa.synchronize()
    torch.testing.assert_close(qk.cpu(), expected_qk.float(), rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(pv.cpu(), expected_pv.float(), rtol=1e-3, atol=2e-3)
    llir = kernel.asm["llir"]
    if dtype == torch.bfloat16:
        assert "llvm.musa.sqmma.bfmma" in llir
        return
    assert "llvm.musa.sqmma.e4m3" in llir
    descriptor = "@llvm.musa.sqmma.desc.i8"
    declarations = [line for line in llir.splitlines() if line.startswith("declare") and descriptor in line]
    calls = [line.strip() for line in llir.splitlines() if " call " in line and descriptor in line]
    assert len(declarations) == 1
    assert calls
    if descriptor + "(ptr addrspace(3)" in declarations[0]:
        assert all(descriptor + "(ptr addrspace(3)" in call for call in calls)
    else:
        assert descriptor + "(i8" in declarations[0]
        assert all(descriptor + "(i8 " in call for call in calls)
        assert any("load i8, ptr addrspace(3)" in line for line in llir.splitlines())


@triton.jit
def capture_marker(Markers):
    tl.store(Markers, 1)


@triton.jit
def capture_consumer(QK, PV, tile):
    q = tle.gpu.alloc((32, 256), tl.float8e4nv)
    p = tle.gpu.alloc((32, 64), tl.float8e4nv)
    scores = tle.gpu.wgmma(q, tile, trans_b=True)
    scores = tle.gpu.wgmma_wait(0, scores)
    output = tle.gpu.wgmma(p, tile)
    output = tle.gpu.wgmma_wait(0, output)
    rows = tl.arange(0, 32)[:, None]
    tl.store(QK + rows * 64 + tl.arange(0, 64)[None, :], scores)
    tl.store(PV + rows * 256 + tl.arange(0, 256)[None, :], output)


@triton.jit
def captured_stage_kernel(QK, PV, Markers, STAGES: tl.constexpr):
    kv = tle.gpu.alloc((STAGES, 64, 256), tl.float8e4nv)
    tile = kv.slot(STAGES - 1)
    tle.gpu.warp_specialize(
        [(capture_marker, (Markers, )), (capture_consumer, (QK, PV, tile)), (capture_marker, (Markers, ))],
        worker_num_warps=[4, 4],
        worker_num_regs=[128, 32],
    )


@pytest.mark.parametrize("stages", [1, 2])
def test_shared_kv_captured_stage_across_workers(stages):
    _, ttgir, lowered = _compile_ws_ir(
        captured_stage_kernel,
        {"QK": "*fp32", "PV": "*fp32", "Markers": "*i32", "STAGES": "constexpr"},
        {"STAGES": stages},
        consumer_warps=4,
        include_late=True,
    )
    assert ttgir.count(" = ttg.local_alloc ") == 3
    assert "llvm.musa.sqmma.e4m3" in lowered


@triton.jit
def kv_writer(writer, Q, KV, P, iterations):
    for iteration in range(iterations):
        slot = writer.acquire(iteration)
        tle.gpu.copy(Q, slot.q, (32, 256), (0, 0))
        tle.gpu.copy(KV, slot.kv, (64, 256), (iteration * 64, 0))
        tle.gpu.copy(P, slot.p, (32, 64), (0, 0))
        writer.commit(iteration)


@triton.jit
def kv_reader(reader, QK, PV, iterations):
    rows = tl.arange(0, 32)[:, None]
    for iteration in range(iterations):
        ready = reader.wait(iteration)
        scores = tle.gpu.wgmma(ready.slot.q, ready.slot.kv, trans_b=True)
        scores = tle.gpu.wgmma_wait(0, scores)
        output = tle.gpu.wgmma(ready.slot.p, ready.slot.kv)
        output = tle.gpu.wgmma_wait(0, output)
        reader.release(iteration)
        tl.store(QK + iteration * (32 * 64) + rows * 64 + tl.arange(0, 64)[None, :], scores)
        tl.store(PV + iteration * (32 * 256) + rows * 256 + tl.arange(0, 256)[None, :], output)


@triton.jit(do_not_specialize=["iterations"])
def shared_kv_pipeline(Q, KV, P, QK, PV, iterations, STAGES: tl.constexpr):
    q = tle.gpu.alloc((STAGES, 32, 256), tl.float8e4nv)
    kv = tle.gpu.alloc((STAGES, 64, 256), tl.float8e4nv)
    p = tle.gpu.alloc((STAGES, 32, 64), tl.float8e4nv)
    pipe = tle.pipe(capacity=STAGES, scope="cta", name="shared_kv", q=q, kv=kv, p=p)
    tle.gpu.warp_specialize(
        [(kv_reader, (pipe.reader(), QK, PV, iterations)), (kv_writer, (pipe.writer(), Q, KV, P, iterations))],
        worker_num_warps=[4],
        worker_num_regs=[128],
    )


@pytest.mark.parametrize("stages", [1, 2])
def test_shared_kv_dynamic_pipeline_compile(stages):
    compiled = compile_musa(
        shared_kv_pipeline,
        {
            "Q": "tensordesc<fp8e4nv[32,256]>", "KV": "tensordesc<fp8e4nv[64,256]>", "P": "tensordesc<fp8e4nv[32,64]>",
            "QK": "*fp32", "PV": "*fp32", "iterations": "i32", "STAGES": "constexpr"
        },
        {"STAGES": stages},
        options={"num_warps": 4, "num_stages": 1},
    )
    assert "ttg.memdesc_index" in compiled.asm["ttgir"]
    assert len(re.findall(r" = ttg.local_alloc.*xf8E4M3FN", compiled.asm["ttgir"])) == 3
    assert "llvm.musa.sqmma.e4m3" in compiled.asm["llir"]


@pytest.mark.parametrize("stages", [1, 2])
def test_shared_kv_dynamic_pipeline_precision(stages):
    import torch_musa

    if not torch_musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.manual_seed(1450)
    iterations = 2 * stages + 1
    query = (torch.randn((32, 256)) * 0.25).to(torch.float8_e4m3fn)
    key_value = (torch.randn((iterations, 64, 256)) * 0.25).to(torch.float8_e4m3fn)
    probability = (torch.randn((32, 64)) * 0.25).to(torch.float8_e4m3fn)
    expected_qk = query.double() @ key_value.double().transpose(1, 2)
    expected_pv = probability.double() @ key_value.double()
    descriptors = [
        TensorDescriptor.from_tensor(value.to("musa"), block)
        for value, block in ((query, [32, 256]), (key_value.reshape(iterations * 64, 256), [64, 256]), (probability,
                                                                                                        [32, 64]))
    ]
    qk = torch.empty((iterations, 32, 64), dtype=torch.float32, device="musa")
    pv = torch.empty((iterations, 32, 256), dtype=torch.float32, device="musa")
    shared_kv_pipeline[(1, )](*descriptors, qk, pv, iterations, stages, num_warps=4, num_stages=1)
    torch.musa.synchronize()
    torch.testing.assert_close(qk.cpu(), expected_qk.float(), rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(pv.cpu(), expected_pv.float(), rtol=1e-3, atol=2e-3)
