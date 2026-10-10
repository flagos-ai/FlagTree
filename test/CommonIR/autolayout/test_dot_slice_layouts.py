"""Keep a dot-produced tensor in registers across legal logical slices."""

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C import libtriton
from triton._common_ir import ENABLED
from triton.backends.compiler import GPUTarget
from triton.backends.metax.autolayout import (
    apply_dot_layout_candidate, finalize_dot_layouts, prepare_dot_layouts,
)
from triton.compiler.compiler import ASTSource, make_backend

pytestmark = pytest.mark.skipif(
    not ENABLED or not hasattr(libtriton, "metax"),
    reason="requires a MetaX FLAGTREE_COMMON_IR build",
)


def operations(module, name):
    result = []
    module.walk(lambda op: result.append(op) if op.get_name() == name else None)
    return result


def choose(module, col_major, warps):
    domains = libtriton.metax.autolayout.enumerate_dot_layout_plans(
        module, capability=80, num_warps=4)
    candidate = {}
    for domain in domains:
        matches = [plan for plan in domain["plans"] if all(
            token in str(plan["mma"]) for token in (
                f"colMajor = {col_major}",
                f"warpsPerCTA = [{warps[0]}, {warps[1]}]",
                "elementsMNK = [1, 1, 4]",
            ))]
        assert len(matches) == 1
        candidate[domain["dot_id"]] = matches[0]
    return candidate


@triton.jit
def _chain(a, b, v, out, REUSE: tl.constexpr):
    m = tl.arange(0, 128)
    k = tl.arange(0, 32)
    av = tl.load(a + m[:, None] * 32 + k[None, :])
    bv = tl.load(b + k[:, None] * 32 + k[None, :])
    qk = tl.dot(av, bv)
    p = tl.exp2(qk - tl.max(qk, 1)[:, None]).to(tl.float16)
    p0 = tle.extract_tile(p, (0, 0), (128, 16))
    if REUSE:
        p1 = p0
    else:
        p1 = tle.extract_tile(p, (0, 1), (128, 16))
    r = tl.arange(0, 16)
    vv = tl.load(v + r[:, None] * 32 + k[None, :])
    result = tl.dot(p1, vv, tl.dot(p0, vv))
    tl.store(out + m[:, None] * 32 + k[None, :], result)


def prepare(tmp_path, reuse=False):
    target = GPUTarget("maca", 80, 64)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 4})
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(_chain, {"a": "*fp16", "b": "*fp16", "v": "*fp16", "out": "*fp32",
                               "REUSE": "constexpr"}, {"REUSE": reuse})
    module = source.make_ir(target, options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)
    module.context = context
    module = backend.make_ttir(module, {}, options)
    prepare_dot_layouts(module, options, 80)
    return module


@pytest.mark.parametrize("col_major", [0, 1])
def test_softmax_slices_require_register_compatible_dot_operand(tmp_path, col_major):
    module = prepare(tmp_path)
    apply_dot_layout_candidate(module, choose(module, col_major, warps=(4, 1)), 80, 4)
    assert module.verify()
    extracts = operations(module, "ttg.extract_tensor")
    assert len(extracts) == 2
    # colMajor=0 can be sliced, but its C -> dot A conversion needs shared.
    # Only colMajor=1 satisfies the complete register-only path.
    carrier = "maca_mma" if col_major else "blocked"
    for op in extracts:
        assert carrier in str(op.get_operand(0).get_type())
        assert carrier in str(op.get_result(0).get_type())
    finalize_dot_layouts(module, 80)
    assert module.verify()
    # Check the P fragments independently of the other dot inputs.
    p_allocs = [op for op in operations(module, "ttg.local_alloc")
                if "128x16xf16" in str(op.get_result(0).get_type())]
    assert bool(p_allocs) == (col_major == 0)


def test_illegal_mma_slice_preserves_fallback(tmp_path):
    module = prepare(tmp_path)
    # Two warps along N require a 32-column MMA tile; a 16-column slice
    # cannot retain that thread distribution.
    apply_dot_layout_candidate(module, choose(module, 1, warps=(2, 2)), 80, 4)
    assert module.verify()
    extracts = operations(module, "ttg.extract_tensor")
    assert len(extracts) == 2
    assert all("blocked" in str(op.get_result(0).get_type()) for op in extracts)
    finalize_dot_layouts(module, 80)
    assert module.verify()


@pytest.mark.parametrize("same_layout", [False, True])
def test_reused_fragment_preserves_consumer_requirements(tmp_path, same_layout):
    module = prepare(tmp_path, reuse=True)
    candidate = choose(module, 1, warps=(4, 1))
    other = choose(module, 1, warps=(2, 2))
    last = list(candidate)[-1]
    if not same_layout:
        candidate[last] = other[last]
    apply_dot_layout_candidate(module, candidate, 80, 4)
    assert module.verify()
    extracts = operations(module, "ttg.extract_tensor")
    assert len(extracts) == 1
    assert "maca_mma" in str(extracts[0].get_result(0).get_type())
    finalize_dot_layouts(module, 80)
    assert module.verify()
    assert operations(module, "ttg.convert_layout")


@pytest.mark.parametrize("col_major,warps,reuse,mixed", [
    (1, (4, 1), False, False),
    (0, (4, 1), False, False),
    (1, (2, 2), False, False),
    (1, (4, 1), True, True),
])
def test_gpu_dot_slices_preserve_values(tmp_path, col_major, warps, reuse, mixed):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires a MetaX GPU")
    module = prepare(tmp_path, reuse)
    candidate = choose(module, col_major, warps=warps)
    if mixed:
        other = choose(module, col_major, warps=(2, 2))
        last = list(candidate)[-1]
        candidate[last] = other[last]
    apply_dot_layout_candidate(module, candidate, 80, 4)
    finalize_dot_layouts(module, 80)
    path = tmp_path / "dot-slices.ttgir"
    path.write_text(module.str())
    kernel = triton.compile(str(path), target=GPUTarget("maca", 80, 64), options={"num_warps": 4})
    for seed in (0, 17):
        generator = torch.Generator().manual_seed(seed)
        a, b, v = [(torch.randn(shape, generator=generator) * 0.25).half()
                   for shape in ((128, 32), (32, 32), (16, 32))]
        qk = a.float() @ b.float()
        p = torch.exp2(qk - qk.max(dim=1, keepdim=True).values).half().float()
        first, second = p[:, :16], p[:, :16] if reuse else p[:, 16:]
        expected = first @ v.float() + second @ v.float()
        result = torch.empty((128, 32), device="cuda", dtype=torch.float32)
        kernel[(1, 1, 1)](a.cuda(), b.cuda(), v.cuda(), result)
        torch.testing.assert_close(result.cpu(), expected, atol=3e-3, rtol=3e-3)
