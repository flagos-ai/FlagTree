"""Explicit TLE layouts must be eliminated by Iluvatar's TTIR conversion."""

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def _explicit_load(source, destination, POINTER: tl.constexpr):
    row = tl.arange(0, 4)[:, None]
    column = tl.arange(0, 128)[None, :]
    pointers = source + row * 128 + column
    layout: tl.constexpr = tle.gpu.BlockEncoding([1, 2], [1, 64], [4, 1], [1, 0])
    if POINTER:
        values = tl.load(tle.gpu.set_layout(pointers, layout))
    else:
        values = tle.gpu.set_layout(tl.load(pointers), layout)
    tl.store(destination + row * 128 + column, values)


@pytest.mark.parametrize("pointer", [False, True])
def test_explicit_layout_compiles_for_values_and_pointers(pointer):
    compiled = triton.compile(
        ASTSource(_explicit_load, signature={"source": "*fp32", "destination": "*fp32"},
                  constexprs={"POINTER": pointer}),
        target=GPUTarget("corex", 71, 64), options={"num_warps": 4},
    )
    assert "tle.gpu.set_layout" not in compiled.asm["ttgir"]
    assert "define" in compiled.asm["llir"]
    assert '"nvvm.reqntid"="256"' in compiled.asm["llir"]


@pytest.mark.parametrize("pointer", [False, True])
def test_explicit_layout_preserves_values_on_gpu(pointer):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    source = torch.arange(512, device="cuda", dtype=torch.float32).reshape(4, 128)
    destination = torch.empty_like(source)
    _explicit_load[(1,)](source, destination, pointer, num_warps=4)
    assert torch.equal(destination, source)


@triton.jit
def _explicit_reduction(source, factor, destination):
    head = tl.arange(0, 4)[:, None, None]
    token = tl.arange(0, 16)[None, :, None]
    dimension = tl.arange(0, 128)[None, None, :]
    value = tl.load(source + head * 2048 + token * 128 + dimension)
    value = tle.gpu.set_layout(
        value, tle.gpu.BlockEncoding([1, 1, 1], [1, 1, 64], [4, 8, 1], [2, 1, 0])
    )
    multiplier = tl.load(factor + tl.arange(0, 128))
    total = tl.sum(value * multiplier[None, None, :], 2)
    tl.store(destination + tl.arange(0, 4)[:, None] * 16 + tl.arange(0, 16)[None, :], total)


def test_explicit_layout_survives_elementwise_and_reduction_on_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    # Integer-valued FP32 arithmetic keeps this check independent of a
    # backend-specific floating-point reduction tree.
    source = (torch.arange(8192, device="cuda") % 17).float().reshape(4, 16, 128)
    factor = (torch.arange(128, device="cuda") % 3).float()
    destination = torch.empty((4, 16), device="cuda")
    compiled = _explicit_reduction[(1,)](source, factor, destination, num_warps=32)
    assert torch.equal(destination, (source * factor).sum(-1))
    assert "warpsPerCTA = [4, 8, 1]" in compiled.asm["ttgir"]


@triton.jit
def _two_layouts(source, factor, destination):
    head = tl.arange(0, 4)[:, None, None]
    token = tl.arange(0, 16)[None, :, None]
    dimension = tl.arange(0, 128)[None, None, :]
    value = tl.load(source + head * 2048 + token * 128 + dimension)
    layout: tl.constexpr = tle.gpu.BlockEncoding(
        [1, 1, 1], [1, 1, 64], [4, 8, 1], [2, 1, 0]
    )
    value = tle.gpu.set_layout(value, layout)
    # The second hint is an identity conversion during TTIR adaptation.
    # Its contract must survive reconciliation of temporary adaptors.
    value = tle.gpu.set_layout(value * tl.load(factor + dimension), layout)
    value = tle.gpu.set_layout(
        value, tle.gpu.BlockEncoding([1, 1, 1], [1, 1, 64], [1, 16, 2], [2, 1, 0])
    )
    total = tl.sum(value, 1)
    tl.store(destination + tl.arange(0, 4)[:, None] * 128 + tl.arange(0, 128)[None, :], total)


def test_intermediate_layout_contract_survives_identity_materialization():
    compiled = triton.compile(
        ASTSource(_two_layouts,
                  signature={"source": "*fp32", "factor": "*fp32", "destination": "*fp32"}),
        target=GPUTarget("corex", 71, 64), options={"num_warps": 32},
    )
    ir = compiled.asm["ttgir"]
    assert 'tle.explicit_encoding.0' in ir
    assert 'warpsPerCTA = [4, 8, 1]' in ir
    assert 'warpsPerCTA = [1, 16, 2]' in ir
    # The last explicit contract must remain attached to the value used by
    # the token reduction rather than only a dead load or temporary cast.
    import re
    explicit = re.findall(r'(%[\w.]+) = .*tle.explicit_encoding.0.*tensor<4x16x128xf32, (#[\w]+)>', ir)
    assert any(f'"tt.reduce"({value})' in ir for value, _ in explicit), ir


@triton.jit
def _explicit_loop(source, destination):
    row = tl.arange(0, 4)[:, None]
    column = tl.arange(0, 128)[None, :]
    layout: tl.constexpr = tle.gpu.BlockEncoding([1, 2], [1, 64], [4, 1], [1, 0])
    value = tle.gpu.set_layout(tl.load(source + row * 128 + column), layout)
    for step in range(3):
        value = tle.gpu.set_layout(value + step, layout)
    tl.store(destination + row * 128 + column, value)


def test_explicit_layout_loop_carried_value_compiles():
    compiled = triton.compile(
        ASTSource(_explicit_loop, signature={"source": "*fp32", "destination": "*fp32"}),
        target=GPUTarget("corex", 71, 64), options={"num_warps": 4},
    )
    assert "tle.gpu.set_layout" not in compiled.asm["ttgir"]


def test_explicit_layout_loop_carried_value_on_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    source = torch.arange(512, device="cuda", dtype=torch.float32).reshape(4, 128)
    destination = torch.empty_like(source)
    _explicit_loop[(1,)](source, destination, num_warps=4)
    assert torch.equal(destination, source + 3)


@triton.jit
def _explicit_reshape(source, destination):
    values = tl.load(source + tl.arange(0, 512)).reshape((4, 128))
    values = tle.gpu.set_layout(
        values, tle.gpu.BlockEncoding([1, 2], [1, 64], [4, 1], [1, 0])
    )
    values = values.reshape((4, 2, 64)).trans(0, 2, 1)
    low, high = tl.split(values)
    index = tl.arange(0, 4)[:, None] * 64 + tl.arange(0, 64)[None, :]
    tl.store(destination + index, low + high)


def test_explicit_layout_reshape_and_transpose_on_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    source = torch.arange(512, device="cuda", dtype=torch.float32).reshape(4, 128)
    destination = torch.empty((4, 64), device="cuda")
    _explicit_reshape[(1,)](source, destination, num_warps=4)
    assert torch.equal(destination, source[:, :64] + source[:, 64:])


@triton.jit
def _explicit_constant_loop(source, destination):
    row = tl.arange(0, 4)[:, None]
    column = tl.arange(0, 128)[None, :]
    layout: tl.constexpr = tle.gpu.BlockEncoding([1, 2], [1, 64], [4, 1], [1, 0])
    value = tle.gpu.set_layout(tl.zeros((4, 128), tl.float32), layout)
    for step in range(3):
        value = tle.gpu.set_layout(value + tl.load(source + row * 128 + column + step * 512), layout)
    tl.store(destination + row * 128 + column, value)


def test_explicit_constant_rematerialization_terminates():
    compiled = triton.compile(
        ASTSource(_explicit_constant_loop, signature={"source": "*fp32", "destination": "*fp32"}),
        target=GPUTarget("corex", 71, 64), options={"num_warps": 4},
    )
    assert "tle.explicit_encoding.0" in compiled.asm["ttgir"]


@triton.jit
def _explicit_sliced_reduction(source, destination):
    row = tl.arange(0, 128)[:, None]
    column = tl.arange(0, 32)[None, :]
    layout: tl.constexpr = tle.gpu.BlockEncoding(
        [1, 8], [16, 4], [4, 1], [1, 0]
    )
    values = tl.load(tle.gpu.set_layout(source + row * 32 + column, layout))
    total = tl.sum(values, 1)
    pointers = tle.gpu.set_layout(
        destination + tl.arange(0, 128), tle.gpu.SlicedEncoding(1, layout)
    )
    tl.store(pointers, total)


def test_explicit_sliced_reduction_compiles():
    compiled = triton.compile(
        ASTSource(_explicit_sliced_reduction,
                  signature={"source": "*fp32", "destination": "*fp32"}),
        target=GPUTarget("corex", 71, 64), options={"num_warps": 4},
    )
    assert "tle.gpu.set_layout" not in compiled.asm["ttgir"]
    assert "ttg.slice<{dim = 1" in compiled.asm["ttgir"]


def test_explicit_sliced_reduction_preserves_values_on_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    source = (torch.arange(4096, device="cuda") % 17).float().reshape(128, 32)
    destination = torch.empty((128,), device="cuda")
    _explicit_sliced_reduction[(1,)](source, destination, num_warps=4)
    assert torch.equal(destination, source.sum(-1))


@triton.jit(noinline=True)
def _explicit_tensor_result(source):
    index = tl.arange(0, 512).reshape(4, 128)
    layout: tl.constexpr = tle.gpu.BlockEncoding([1, 2], [1, 64], [4, 1], [1, 0])
    return tle.gpu.set_layout(tl.load(source + index), layout) + 1


@triton.jit
def _explicit_tensor_call(source, destination):
    values = _explicit_tensor_result(source)
    tl.store(destination + tl.arange(0, 512).reshape(4, 128), values)


def test_explicit_tensor_function_result_compiles():
    compiled = triton.compile(
        ASTSource(_explicit_tensor_call,
                  signature={"source": "*fp32", "destination": "*fp32"}),
        target=GPUTarget("corex", 71, 64), options={"num_warps": 4},
    )
    assert "tle.gpu.set_layout" not in compiled.asm["ttgir"]


def test_explicit_tensor_function_result_on_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    source = torch.arange(512, device="cuda", dtype=torch.float32)
    destination = torch.empty_like(source)
    _explicit_tensor_call[(1,)](source, destination, num_warps=4)
    assert torch.equal(destination, source + 1)


@triton.jit(noinline=True)
def _explicit_tensor_multiple_results(source):
    index = tl.arange(0, 512).reshape(4, 128)
    layout: tl.constexpr = tle.gpu.BlockEncoding([1, 2], [1, 64], [4, 1], [1, 0])
    values = tle.gpu.set_layout(tl.load(source + index), layout)
    return values * 2, tl.sum(tl.sum(values, 1), 0)


@triton.jit
def _explicit_tensor_multiple_call(source, destination, total):
    index = tl.arange(0, 512).reshape(4, 128)
    doubled, reduced = _explicit_tensor_multiple_results(source)
    tl.store(destination + index, doubled)
    tl.store(total, reduced)


def test_explicit_tensor_function_multiple_results_compiles():
    compiled = triton.compile(
        ASTSource(_explicit_tensor_multiple_call, signature={
            "source": "*fp32", "destination": "*fp32", "total": "*fp32",
        }), target=GPUTarget("corex", 71, 64), options={"num_warps": 4},
    )
    assert "tle.gpu.set_layout" not in compiled.asm["ttgir"]


def test_explicit_tensor_function_multiple_results_on_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    source = torch.arange(512, device="cuda", dtype=torch.float32)
    destination = torch.empty_like(source)
    total = torch.empty((), device="cuda", dtype=torch.float32)
    _explicit_tensor_multiple_call[(1,)](source, destination, total, num_warps=4)
    assert torch.equal(destination, source * 2)
    assert torch.equal(total, source.sum())
