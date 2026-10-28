"""PPU explicit layouts: offline compiler checks and separate device tests.

The offline tests stop after LLVM IR without invoking the PPU SDK or loading a
kernel. Runtime tests additionally require a PPU device and a working SDK.
Missing TLE bindings or whitelist entries in a PPU build are failures, not skips.
"""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C import libtriton
from triton._flagtree_backend import FLAGTREE_BACKEND
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, make_backend
from triton.compiler.errors import CompilationError
from triton.experimental.tle import get_supported_primitives

BLOCK = 256
TARGET = GPUTarget("cuda", 80, 32)


def _blocked(size=1):
    return tle.gpu.BlockEncoding([size], [32], [4], [0])


def _matrix_layout():
    return tle.gpu.BlockEncoding([1, 1], [4, 8], [4, 1], [1, 0])


def _layout(kind):
    if kind == "blocked":
        return _blocked()
    return tle.gpu.SlicedEncoding(0, tle.gpu.BlockEncoding([1, 2], [1, 32], [1, 4], [1, 0]))


def _encoding(layout):
    if isinstance(layout, tle.gpu.SlicedEncoding):
        return f"#ttg.slice<{{dim = {layout.dim}, parent = {_encoding(layout.parent)}}}>"
    return (f"#ttg.blocked<{{sizePerThread = {layout.size_per_thread}, "
            f"threadsPerWarp = {layout.threads_per_warp}, "
            f"warpsPerCTA = {layout.warps_per_cta}, order = {layout.order}}}>")


@pytest.fixture(scope="module")
def ppu_backend():
    if FLAGTREE_BACKEND != "ppu":
        pytest.skip("requires a FlagTree PPU build")
    assert hasattr(libtriton, "ppu"), "PPU build is missing native PPU bindings"
    assert hasattr(libtriton, "tle"), "PPU build is missing native TLE bindings"
    assert hasattr(libtriton.ir.builder, "create_tle_gpu_set_layout")
    assert "gpu.set_layout" in get_supported_primitives("ppu")
    backend = make_backend(TARGET)
    assert type(backend).__name__ == "PPUBackend"
    assert backend.binary_ext == "hgbin"
    return backend


@pytest.fixture
def ppu_runtime(ppu_backend):
    if not torch.cuda.is_available():
        pytest.skip("runtime test requires a visible PPU device; offline tests do not")
    backend = make_backend(triton.runtime.driver.active.get_current_target())
    assert type(backend).__name__ == "PPUBackend"


def _compile_offline(backend, kernel, signature, constexprs, **option_overrides):
    """Use the real PPU pipeline without backend.hash()/ppu-llc or a launcher."""
    options = backend.parse_options({"num_warps": 4, "num_ctas": 1, **option_overrides})
    source = ASTSource(kernel, signature, constexprs)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    backend.load_dialects(context)
    module = backend.make_ir(source, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                             context)
    metadata = {"target": TARGET, **vars(options)}
    capability = int(options.arch.removeprefix("sm"))
    module = backend.make_ttir(module, metadata, options, capability)
    result = {"ttir": str(module)}
    module = backend.make_ttgir(module, metadata, options, capability)
    result["ttgir"] = str(module)
    result["llir"] = backend.make_llir(module, metadata, options, capability)
    return result


def _expand_aliases(text, aliases):
    for _ in range(len(aliases) + 1):
        expanded = re.sub(r"#[A-Za-z_]\w*", lambda match: aliases.get(match[0], match[0]), text)
        if expanded == text:
            return re.sub(r"\s+", "", text)
        text = expanded
    raise AssertionError("recursive layout aliases in TTGIR")


def _assert_layout(ttgir, operation, shape, element, layout):
    aliases = dict(re.findall(r"^(#\w+)\s*=\s*(.+)$", ttgir, re.MULTILINE))
    lines = [line for line in ttgir.splitlines() if re.search(r"\b" + re.escape(operation) + r"\b", line)]
    assert lines, f"missing {operation} in TTGIR"
    expected = re.sub(r"\s+", "", f"tensor<{shape}x{element}, {_encoding(layout)}>")
    # Look at the operation's actual tensor types, not an unused alias or an
    # explicit_encoding attribute that could have survived on the wrong type.
    type_lines = lines if operation == "scf.for" else [line.split(" : ", 1)[-1] for line in lines]
    assert all(expected in _expand_aliases(line, aliases) for line in type_lines), "\n".join(lines)


def _assert_lowered(assembly):
    assert "tle.gpu.set_layout" in assembly["ttir"]
    assert "tle.gpu.set_layout" not in assembly["ttgir"]
    assert re.search(r'ttg\.target\s*=\s*"ppu:', assembly["ttgir"])
    assert assembly["llir"]
    assert "tle.gpu.set_layout" not in assembly["llir"]


@triton.jit
def _baseline(src, out, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.load(src + offsets, mask, other=0.0)
    tl.store(out + offsets, values + 7.0, mask)


@triton.jit
def _pointwise(src, out, n: tl.constexpr, LAYOUT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    values = tle.gpu.set_layout(tl.load(src + offsets, mask, other=0.0), LAYOUT)
    pointers = tle.gpu.set_layout(out + offsets, LAYOUT)
    tl.store(pointers, values + 7.0, mask)


@triton.jit
def _sequential(src, out_a, out_b, LAYOUT_A: tl.constexpr, LAYOUT_B: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    a = tle.gpu.set_layout(tl.load(src + offsets), LAYOUT_A)
    b = tle.gpu.set_layout(a, LAYOUT_B)
    # Both results must remain live, so this exercises two requested domains.
    tl.store(tle.gpu.set_layout(out_a + offsets, LAYOUT_A), a + 1.0)
    tl.store(tle.gpu.set_layout(out_b + offsets, LAYOUT_B), b + 2.0)


@triton.jit
def _loop(src, out, repeats, LAYOUT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tle.gpu.set_layout(tl.load(src + offsets), LAYOUT)
    for _ in range(repeats):
        values = values + 1.0
    tl.store(tle.gpu.set_layout(out + offsets, LAYOUT), values)


@triton.jit
def _reduce(src, out, LAYOUT: tl.constexpr, RESULT_LAYOUT: tl.constexpr):
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 32)
    values = tl.load(src + rows[:, None] * 32 + cols[None, :])
    values = tle.gpu.set_layout(values, LAYOUT)
    result = tl.sum(values, axis=1)
    tl.store(tle.gpu.set_layout(out + rows, RESULT_LAYOUT), result)


@triton.jit
def _invalid(out, LAYOUT: tl.constexpr):
    values = tle.gpu.set_layout(tl.arange(0, 256), LAYOUT)
    tl.store(out + tl.arange(0, 256), values)


@triton.jit
def _conflict(src, out, LAYOUT_A: tl.constexpr, LAYOUT_B: tl.constexpr):
    offsets = tl.arange(0, 256)
    values = tl.load(src + offsets)
    a = tle.gpu.set_layout(values, LAYOUT_A)
    b = tle.gpu.set_layout(values, LAYOUT_B)
    # Merging incompatible hard layouts without an explicit conversion is an
    # error; sequential set_layout in _sequential is a legal conversion.
    tl.store(out + offsets, a + b)


def test_baseline_without_set_layout_compiles_offline(ppu_backend):
    assembly = _compile_offline(ppu_backend, _baseline, {"src": "*fp32", "out": "*fp32"}, {"n": 125, "BLOCK": BLOCK})
    assert all(assembly[stage] for stage in ("ttir", "ttgir", "llir"))
    assert all("tle." not in text for text in assembly.values())
    assert re.search(r'ttg\.target\s*=\s*"ppu:', assembly["ttgir"])
    assert "arith.addf" in assembly["ttgir"]
    assert "tt.store" in assembly["ttgir"]


@pytest.mark.parametrize("kind", ["blocked", "sliced"])
def test_set_layout_compiles_offline(ppu_backend, kind):
    layout = _layout(kind)
    assembly = _compile_offline(ppu_backend, _pointwise, {"src": "*fp32", "out": "*fp32"},
                                {"n": 125, "LAYOUT": layout, "BLOCK": BLOCK})
    _assert_lowered(assembly)
    _assert_layout(assembly["ttgir"], "arith.addf", str(BLOCK), "f32", layout)
    _assert_layout(assembly["ttgir"], "tt.store", str(BLOCK), "!tt.ptr<f32>", layout)


def test_set_layout_sequential_compiles_offline(ppu_backend):
    a, b = _blocked(1), _blocked(2)
    assembly = _compile_offline(ppu_backend, _sequential, {"src": "*fp32", "out_a": "*fp32", "out_b": "*fp32"},
                                {"LAYOUT_A": a, "LAYOUT_B": b, "BLOCK": BLOCK})
    _assert_lowered(assembly)
    ttgir = assembly["ttgir"]
    # Do not require a convert instruction: legal rematerialization can remove
    # it. The two final stores must still use their respective encodings.
    stores = [line for line in ttgir.splitlines() if "tt.store " in line]
    assert len(stores) == 2
    aliases = dict(re.findall(r"^(#\w+)\s*=\s*(.+)$", ttgir, re.MULTILINE))
    store_types = [_expand_aliases(line.split(" : ", 1)[-1], aliases) for line in stores]
    for layout in (a, b):
        expected = re.sub(r"\s+", "", f"tensor<{BLOCK}x!tt.ptr<f32>, {_encoding(layout)}>")
        assert any(expected in types for types in store_types), "\n".join(stores)


def test_set_layout_loop_compiles_offline(ppu_backend):
    layout = _blocked()
    assembly = _compile_offline(ppu_backend, _loop, {"src": "*fp32", "out": "*fp32", "repeats": "i32"},
                                {"LAYOUT": layout, "BLOCK": BLOCK})
    _assert_lowered(assembly)
    _assert_layout(assembly["ttgir"], "scf.for", str(BLOCK), "f32", layout)
    _assert_layout(assembly["ttgir"], "arith.addf", str(BLOCK), "f32", layout)
    _assert_layout(assembly["ttgir"], "tt.store", str(BLOCK), "!tt.ptr<f32>", layout)


def test_set_layout_reduce_compiles_offline(ppu_backend):
    layout = _matrix_layout()
    result_layout = tle.gpu.SlicedEncoding(1, layout)
    assembly = _compile_offline(ppu_backend, _reduce, {"src": "*fp32", "out": "*fp32"},
                                {"LAYOUT": layout, "RESULT_LAYOUT": result_layout})
    _assert_lowered(assembly)
    ttgir = assembly["ttgir"]
    assert '"tt.reduce"' in ttgir
    aliases = dict(re.findall(r"^(#\w+)\s*=\s*(.+)$", ttgir, re.MULTILINE))
    # Generic reduce syntax places its input/result types after its region.
    signature = (f"(tensor<16x32xf32, {_encoding(layout)}>) -> "
                 f"tensor<16xf32, {_encoding(result_layout)}>")
    assert re.sub(r"\s+", "", signature) in _expand_aliases(ttgir, aliases)
    _assert_layout(assembly["ttgir"], "tt.store", "16", "!tt.ptr<f32>", result_layout)


@pytest.mark.parametrize("case,diagnostic", [
    ("rank", "must match source tensor rank"),
    ("shared", "must be a distributed encoding"),
    ("mma", "only supports blocked layouts and slices of blocked layouts"),
    ("threads", "requires 32 threads per warp"),
    ("warps", "warps per CTA, but the context requires"),
])
def test_set_layout_rejects_invalid_layout_offline(ppu_backend, capfd, case, diagnostic):
    layouts = {
        "rank": _matrix_layout(),
        "shared": tle.gpu.swizzled_shared_layout.make_default(1),
        "mma": tle.gpu.SlicedEncoding(0, tle.gpu.MmaEncoding([2, 0], [4, 1], [16, 8])),
        "threads": tle.gpu.BlockEncoding([1], [16], [4], [0]),
        "warps": tle.gpu.BlockEncoding([1], [32], [8], [0]),
    }
    with pytest.raises((CompilationError, RuntimeError, ValueError)) as error:
        _compile_offline(ppu_backend, _invalid, {"out": "*i32"}, {"LAYOUT": layouts[case]})
    assert diagnostic in str(error.value) + capfd.readouterr().err


def test_set_layout_rejects_conflicting_hard_layouts_offline(ppu_backend, capfd):
    with pytest.raises((CompilationError, RuntimeError, ValueError)) as error:
        _compile_offline(ppu_backend, _conflict, {"src": "*fp32", "out": "*fp32"},
                         {"LAYOUT_A": _blocked(1), "LAYOUT_B": _blocked(2)})
    assert "conflicting TLE encoding hints" in str(error.value) + capfd.readouterr().err


@pytest.mark.parametrize("kind", ["blocked", "sliced"])
@pytest.mark.parametrize("n", [125, 256, 513])
def test_set_layout_pointwise_runtime(ppu_runtime, kind, n):
    layout = _layout(kind)
    reference = torch.arange(n, dtype=torch.float32)
    source = reference.cuda()
    grid = triton.cdiv(n, BLOCK)
    guard = 16
    allocation = torch.full((guard + grid * BLOCK + guard, ), float("nan"), dtype=torch.float32, device="cuda")
    compiled = _pointwise[(grid, )](source, allocation[guard:], n, layout, BLOCK, num_warps=4, num_ctas=1)
    torch.cuda.synchronize()
    _assert_lowered(compiled.asm)
    assert compiled.asm["hgbin"]
    _assert_layout(compiled.asm["ttgir"], "arith.addf", str(BLOCK), "f32", layout)
    _assert_layout(compiled.asm["ttgir"], "tt.store", str(BLOCK), "!tt.ptr<f32>", layout)
    actual = allocation.cpu()
    torch.testing.assert_close(actual[guard:guard + n], reference + 7, rtol=0, atol=0)
    assert torch.isnan(actual[:guard]).all()
    assert torch.isnan(actual[guard + n:]).all()


def test_set_layout_sequential_runtime(ppu_runtime):
    reference = torch.arange(BLOCK, dtype=torch.float32)
    source = reference.cuda()
    a = torch.full_like(source, float("nan"))
    b = torch.full_like(source, float("nan"))
    _sequential[(1, )](source, a, b, _blocked(1), _blocked(2), BLOCK, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(a.cpu(), reference + 1, rtol=0, atol=0)
    torch.testing.assert_close(b.cpu(), reference + 2, rtol=0, atol=0)


def test_set_layout_loop_runtime(ppu_runtime):
    reference = torch.arange(BLOCK, dtype=torch.float32)
    source = reference.cuda()
    out = torch.full_like(source, float("nan"))
    _loop[(1, )](source, out, 5, _blocked(), BLOCK, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.cpu(), reference + 5, rtol=0, atol=0)


def test_set_layout_reduce_runtime(ppu_runtime):
    reference = torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32)
    source = reference.cuda()
    out = torch.full((16, ), float("nan"), dtype=torch.float32, device="cuda")
    layout = _matrix_layout()
    _reduce[(1, )](source, out, layout, tle.gpu.SlicedEncoding(1, layout), num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.cpu(), reference.sum(dim=1), rtol=0, atol=0)
