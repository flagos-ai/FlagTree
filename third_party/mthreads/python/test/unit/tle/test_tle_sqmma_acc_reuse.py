"""FP32 accumulator reuse across SQMMA operand dtypes."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from test_tle_utils import compile_musa, mthreads_backend, require_mthreads_libtriton, tme_descriptor_attrs

require_mthreads_libtriton()


@triton.jit
def _alloc_sqmma(M: tl.constexpr, N: tl.constexpr, DTYPE: tl.constexpr):
    if DTYPE == 0:
        value = tle.gpu.alloc((M, N), tl.float8e4nv, layout=None, nv_mma_shared_layout=True)
    elif DTYPE == 1:
        value = tle.gpu.alloc((M, N), tl.float16, layout=None, nv_mma_shared_layout=True)
    else:
        value = tle.gpu.alloc((M, N), tl.bfloat16, layout=None, nv_mma_shared_layout=True)
    return value


@triton.jit(do_not_specialize=["ITERATIONS"])
def _cross_dtype_kernel(A0, B0, A1, B1, Out, T0: tl.constexpr, T1: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr,
                        ITERATIONS, LOOP: tl.constexpr, WAIT_IN_LOOP: tl.constexpr = False,
                        FINAL_WAIT: tl.constexpr = True):
    a0 = _alloc_sqmma(64, K0, T0)
    b0 = _alloc_sqmma(K0, 64, T0)
    a1 = _alloc_sqmma(64, K1, T1)
    b1 = _alloc_sqmma(K1, 64, T1)
    a_ready0 = tle.gpu.alloc_barrier(expect_bytes=64 * K0 * (1 if T0 == 0 else 2))
    b_ready0 = tle.gpu.alloc_barrier(expect_bytes=64 * K0 * (1 if T0 == 0 else 2))
    a_ready1 = tle.gpu.alloc_barrier(expect_bytes=64 * K1 * (1 if T1 == 0 else 2))
    b_ready1 = tle.gpu.alloc_barrier(expect_bytes=64 * K1 * (1 if T1 == 0 else 2))
    tle.gpu.copy(A0, a0, (64, K0), (0, 0), barrier=a_ready0)
    tle.gpu.copy(B0, b0, (K0, 64), (0, 0), barrier=b_ready0)
    tle.gpu.copy(A1, a1, (64, K1), (0, 0), barrier=a_ready1)
    tle.gpu.copy(B1, b1, (K1, 64), (0, 0), barrier=b_ready1)
    tle.gpu.barrier_wait(a_ready0, phaseIdx=0)
    tle.gpu.barrier_wait(b_ready0, phaseIdx=0)
    tle.gpu.barrier_wait(a_ready1, phaseIdx=0)
    tle.gpu.barrier_wait(b_ready1, phaseIdx=0)
    acc = tl.full((64, 64), 0.5, tl.float32)
    if not LOOP:
        acc = tle.gpu.wgmma(a0, b0, acc)
        acc = tle.gpu.wgmma(a1, b1, acc)
    else:
        for _ in range(ITERATIONS):
            acc = tle.gpu.wgmma(a0, b0, acc)
            acc = tle.gpu.wgmma(a1, b1, acc)
            if WAIT_IN_LOOP:
                acc = tle.gpu.wgmma_wait(0, acc)
    if FINAL_WAIT:
        acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, 64)[:, None] * 64 + tl.arange(0, 64)[None, :]
    tl.store(Out + offsets, acc)


@pytest.mark.parametrize("first,second", [
    ("float8_e4m3fn", "float16"),
    ("float16", "float8_e4m3fn"),
    ("float8_e4m3fn", "bfloat16"),
    ("bfloat16", "float8_e4m3fn"),
    ("float8_e4m3fn", "float8_e4m3fn"),
    ("float16", "float16"),
    ("float16", "bfloat16"),
    ("bfloat16", "float16"),
    ("bfloat16", "bfloat16"),
])
@pytest.mark.parametrize("repeats", [0, 1, 3])
@pytest.mark.parametrize("wait_in_loop", [False, True])
def test_accumulator_reuse_runtime(first, second, repeats, wait_in_loop):
    from triton.tools.tensor_descriptor import TensorDescriptor

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.manual_seed(917)
    formats = {
        "float8_e4m3fn": (torch.float8_e4m3fn, 0, "e4m3", 32),
        "float16": (torch.float16, 1, "fmma", 16),
        "bfloat16": (torch.bfloat16, 2, "bfmma", 16),
    }
    dtypes = [formats[first], formats[second]]
    inputs, descriptors = [], []
    for torch_type, _, _, k in dtypes:
        a_shape, b_shape = (64, k), (k, 64)
        a = (torch.randn(a_shape) * 0.25).to(torch_type)
        b = (torch.randn(b_shape) * 0.25).to(torch_type)
        inputs.append((a, b))
        descriptors.extend([
            TensorDescriptor.from_tensor(a.to("musa"), list(a_shape)),
            TensorDescriptor.from_tensor(b.to("musa"), list(b_shape))
        ])
    shape = (64, 64)
    out = torch.empty(shape, dtype=torch.float32, device="musa")
    kernel = _cross_dtype_kernel[(1, )](
        *descriptors,
        out,
        dtypes[0][1],
        dtypes[1][1],
        dtypes[0][3],
        dtypes[1][3],
        repeats,
        wait_in_loop or repeats != 1,
        WAIT_IN_LOOP=wait_in_loop,
        FINAL_WAIT=not wait_in_loop,
        num_warps=4,
        num_stages=1,
    )
    expected = torch.full(shape, 0.5, dtype=torch.float64)
    for _ in range(repeats):
        for a, b in inputs:
            expected += a.double() @ b.double()
    torch.testing.assert_close(out.cpu(), expected.float(), rtol=1e-3, atol=2e-3)
    for _, _, tag, k in dtypes:
        assert f"llvm.musa.sqmma.{tag}.m64n64k{k}.mma" in kernel.asm["llir"]
    # Descriptor construction can read A/B shared memory between the dots.
    # Check that C itself is forwarded directly, independently of those reads.
    dots = [
        line for line in kernel.asm["llir"].splitlines()
        if "call " in line and re.search(r"@llvm\.musa\.sqmma\.(?:e4m3|fmma|bfmma)\..*\.mma\(", line)
    ]
    assert len(dots) >= 2
    producer = re.match(r"\s*(%[-\w.]+) = ", dots[0]).group(1)
    assert f"<32 x i32> {producer}," in dots[1], dots[1]
    if wait_in_loop or repeats != 1:
        assert "scf.for" in kernel.asm["ttgir"]
    if wait_in_loop:
        loop_body = kernel.asm["ttgir"].split("scf.for", 1)[1].split("scf.yield", 1)[0]
        assert "ttg.convert_layout" not in loop_body
        assert "sqmma_wait" in loop_body
        assert kernel.asm["llir"].count("call void @llvm.musa.sqmma.wait()") == 1


def cross_dtype_source(wait_in_loop=False, second="fp16"):
    from triton.compiler import ASTSource

    signature = {
        "A0": "tensordesc<fp8e4nv[64,32]>",
        "B0": "tensordesc<fp8e4nv[32,64]>",
        "A1": f"tensordesc<{second}[64,16]>",
        "B1": f"tensordesc<{second}[16,64]>",
        "Out": "*fp32",
        "T0": "constexpr",
        "T1": "constexpr",
        "K0": "constexpr",
        "K1": "constexpr",
        "ITERATIONS": "i32",
        "LOOP": "constexpr",
        "WAIT_IN_LOOP": "constexpr",
        "FINAL_WAIT": "constexpr",
    }
    constants = {
        "T0": 0, "T1": 1 if second == "fp16" else 2, "K0": 32, "K1": 16, "LOOP": wait_in_loop, "WAIT_IN_LOOP":
        wait_in_loop, "FINAL_WAIT": not wait_in_loop
    }
    return ASTSource(_cross_dtype_kernel, signature, constexprs=constants, attrs=tme_descriptor_attrs(signature))


@pytest.fixture(scope="module")
def cross_dtype_ttgir():
    target, _ = mthreads_backend()
    return triton.compile(cross_dtype_source(), target=target).asm["ttgir"]


@pytest.mark.parametrize("second", ["fp16", "bf16"])
def test_wait_loop_keeps_native_accumulator_before_layout_optimization(second):
    from triton._C.libtriton import ir, passes, mthreads

    target, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    options = backend.parse_options({"num_warps": 4, "num_stages": 1})
    source = cross_dtype_source(wait_in_loop=True, second=second)
    module = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                            context)
    backend.make_ttir(module, {}, options, 31)
    pm = ir.pass_manager(context)
    passes.ttir.add_convert_to_ttgpuir(pm, "musa:31", 4, 32, 1)
    mthreads.passes.ttgpuir.add_tle_early_assign_memory_space(pm)
    mthreads.passes.ttgpuir.add_tle_select_encodings(pm)
    mthreads.passes.ttgpuir.add_tle_lower_sqmma(pm)
    pm.run(module, "wait_loop_accumulator_preparation")
    loop_body = module.str_nodebug().split("scf.for", 1)[1].split("scf.yield", 1)[0]
    assert "ttg.convert_layout" not in loop_body
    assert loop_body.count("ttmg.squad_dot ") == 2
    assert loop_body.count("ttmg.squad_dot_wait") == 1
    assert "k = 32 : i32" in loop_body
    assert "k = 16 : i32" in loop_body


@pytest.mark.parametrize("marked", [False, True])
@pytest.mark.parametrize("mutation", [None, "k", "m"])
def test_accumulator_verifier_independent_of_marker(cross_dtype_ttgir, marked, mutation, tmp_path, capfd):
    from triton._C.libtriton import ir

    source = cross_dtype_ttgir
    dots = [line for line in source.splitlines() if " = ttmg.squad_dot " in line]
    assert len(dots) == 2
    if mutation is not None:
        replacement = "k = 7 : i32" if mutation == "k" else "m = 32 : i32"
        changed, count = re.subn(rf"\b{mutation} = \d+ : i32", replacement, dots[1], count=1)
        assert count == 1
        source = source.replace(dots[1], changed)
    if not marked:
        source = source.replace("musa_tle.explicit_sqmma, ", "")
        source = source.replace(", musa_tle.explicit_sqmma", "")
        source = source.replace("musa_tle.explicit_sqmma", "")

    target, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    path = tmp_path / "accumulator.ttgir"
    path.write_text(source)
    if mutation is None:
        assert ir.parse_mlir_module(str(path), context).verify()
        compiled = triton.compile(str(path), target=target)
        assert "llvm.musa.sqmma.e4m3.m64n64k32.mma" in compiled.asm["llir"]
        assert "llvm.musa.sqmma.fmma.m64n64k16.mma" in compiled.asm["llir"]
    else:
        with pytest.raises(RuntimeError, match="Parse MLIR file failed"):
            ir.parse_mlir_module(str(path), context)
        diagnostic = ("unsupported PH1 shape/type combination"
                      if mutation == "k" else "m/n attrs must match the accumulator instrShape")
        assert diagnostic in capfd.readouterr().err


@triton.jit
def _loop_without_wait(Out, iterations):
    a = _alloc_sqmma(64, 32, 0)
    b = _alloc_sqmma(32, 64, 0)
    acc = tl.full((64, 64), 0.0, tl.float32)
    for _ in range(iterations):
        acc = tle.gpu.wgmma(a, b, acc)
        acc += 1.0
    offsets = tl.arange(0, 64)[:, None] * 64 + tl.arange(0, 64)[None, :]
    tl.store(Out + offsets, acc)


def test_loop_accumulator_arithmetic_requires_wait(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        compile_musa(_loop_without_wait, {"Out": "*fp32", "iterations": "i32"})
    assert "async result must be consumed by musa_tle.sqmma_wait" in capfd.readouterr().err


@pytest.mark.parametrize("m,n", [(32, 64), (64, 32), (32, 32)])
@pytest.mark.parametrize("encoding_k", [32, 128])
def test_native_accumulator_instruction_shapes(cross_dtype_ttgir, m, n, encoding_k, tmp_path):
    source, count = re.subn(r"instrShape = \[64, 64, 32\]", f"instrShape = [{m}, {n}, {encoding_k}]", cross_dtype_ttgir)
    assert count == 1
    source = re.sub(r"\bm = 64 : i32", f"m = {m} : i32", source)
    source = re.sub(r"\bn = 64 : i32", f"n = {n} : i32", source)
    path = tmp_path / "instruction_shapes.ttgir"
    path.write_text(source)
    target, _ = mthreads_backend()
    kernel = triton.compile(str(path), target=target)
    assert f"llvm.musa.sqmma.e4m3.m{m}n{n}k32.mma" in kernel.asm["llir"]
    assert f"llvm.musa.sqmma.fmma.m{m}n{n}k16.mma" in kernel.asm["llir"]


@triton.jit
def _incompatible_async_layout_kernel(Out):
    a = _alloc_sqmma(64, 32, 0)
    b = _alloc_sqmma(32, 64, 0)
    acc = tle.gpu.wgmma(a, b, tl.full((64, 64), 0.0, tl.float32))
    acc = tl.trans(acc)
    acc = tle.gpu.wgmma(a, b, acc)
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, 64)[:, None] * 64 + tl.arange(0, 64)[None, :]
    tl.store(Out + offsets, acc)


def test_async_accumulator_relayout_requires_wait(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        compile_musa(_incompatible_async_layout_kernel, {"Out": "*fp32"})
    assert "async result must be consumed by musa_tle.sqmma_wait" in capfd.readouterr().err
