"""Iluvatar explicit layouts: offline compiler checks and separate device tests.

The offline tests stop after LLVM IR or target assembly without invoking the
COREX SDK or loading a kernel. Runtime tests require an Iluvatar device and SDK.
Missing TLE bindings or whitelist entries in an Iluvatar build are failures, not skips.
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
TARGET = GPUTarget("corex", 71, 64)


def _blocked(size=1):
    return tle.gpu.BlockEncoding([size], [64], [4], [0])


def _matrix_layout():
    return tle.gpu.BlockEncoding([1, 1], [4, 16], [4, 1], [1, 0])


def _layout(kind):
    if kind == "blocked":
        return _blocked()
    if kind == "recursive-sliced":
        parent = tle.gpu.BlockEncoding([1, 1, 2], [1, 1, 64], [1, 1, 4], [2, 1, 0])
        return tle.gpu.SlicedEncoding(0, tle.gpu.SlicedEncoding(0, parent))
    return tle.gpu.SlicedEncoding(0, tle.gpu.BlockEncoding([1, 2], [1, 64], [1, 4], [1, 0]))


def _encoding(layout):
    if isinstance(layout, tle.gpu.SlicedEncoding):
        return f"#ttg.slice<{{dim = {layout.dim}, parent = {_encoding(layout.parent)}}}>"
    return (f"#ttg.blocked<{{sizePerThread = {layout.size_per_thread}, "
            f"threadsPerWarp = {layout.threads_per_warp}, "
            f"warpsPerCTA = {layout.warps_per_cta}, order = {layout.order}, "
            "isSme = false, smeMask = false, smeWarpsPerCTA = []}>")


@pytest.fixture(scope="module")
def iluvatar_backend():
    if FLAGTREE_BACKEND != "iluvatar":
        pytest.skip("requires a FlagTree Iluvatar build")
    assert hasattr(libtriton, "iluvatar"), "Iluvatar build is missing native Iluvatar bindings"
    assert hasattr(libtriton.iluvatar, "tle"), "Iluvatar build is missing native TLE bindings"
    assert hasattr(libtriton.ir.builder, "ensure_ttg_layout_attrs")
    assert hasattr(libtriton.ir.builder, "create_tle_gpu_set_layout")
    assert "gpu.set_layout" in get_supported_primitives("iluvatar")
    backend = make_backend(TARGET)
    assert type(backend).__name__ == "CorexBackend"
    assert backend.binary_ext == "cubin"
    return backend


@pytest.fixture
def iluvatar_runtime(iluvatar_backend):
    if not torch.cuda.is_available():
        pytest.skip("runtime test requires a visible Iluvatar device; offline tests do not")
    device_name = torch.cuda.get_device_name()
    if re.search(r"nvidia|geforce|tesla|quadro", device_name, re.IGNORECASE):
        pytest.skip(f"requires Iluvatar hardware; visible CUDA device is {device_name}")
    if getattr(torch.version, "hip", None):
        pytest.skip("requires Iluvatar hardware; PyTorch is using HIP")
    driver = triton.runtime.driver.active
    target = driver.get_current_target()
    # CorexBackend also accepts CUDA targets for compatibility. Testing that
    # backend class alone could incorrectly count an NVIDIA GPU as Iluvatar.
    if target.backend != "corex":
        pytest.skip(f"requires an Iluvatar corex device, active target is {target.backend}")
    assert "iluvatar" in type(driver).__module__, type(driver)
    assert target.warp_size == 64
    backend = make_backend(target)
    assert type(backend).__name__ == "CorexBackend"


def _compile_offline(backend, kernel, signature, constexprs, *, asm=False):
    """Compile through the native Iluvatar stages without creating a launcher."""
    options = backend.parse_options({"num_warps": 4, "num_ctas": 1, "warp_size": TARGET.warp_size})
    source = ASTSource(kernel, signature, constexprs)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    backend.load_dialects(context)
    module = source.make_ir(TARGET, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                            context)
    metadata = {"target": TARGET, **vars(options)}
    capability = int(options.arch.removeprefix("sm"))
    module = backend.make_ttir(module, metadata, options, capability)
    result = {"ttir": str(module)}
    module = backend.make_ttgir(module, metadata, options, capability)
    result["ttgir"] = str(module)
    assert '"ttg.threads-per-warp" = 64' in result["ttgir"]
    result["llir"] = backend.make_llir(module, metadata, options, capability)
    if asm:
        result["asm"] = backend.make_asm(result["llir"], metadata, options, capability)
    return result


def _expand_aliases(text, aliases):
    for _ in range(len(aliases) + 1):
        expanded = re.sub(r"#[A-Za-z_]\w*", lambda match: aliases.get(match[0], match[0]), text)
        if expanded == text:
            return re.sub(r"\s+", "", text)
        text = expanded
    raise AssertionError("recursive layout aliases in TTGIR")


def _assert_layout(ttgir, operation, shape, element, layout, *, contains=None):
    aliases = dict(re.findall(r"^(#\w+)\s*=\s*(.+)$", ttgir, re.MULTILINE))
    lines = [line for line in ttgir.splitlines() if re.search(r"\b" + re.escape(operation) + r"\b", line)]
    if contains is not None:
        lines = [line for line in lines if contains in line]
    assert lines, f"missing {operation} in TTGIR"
    expected = re.sub(r"\s+", "", f"tensor<{shape}x{element}, {_encoding(layout)}>")
    # Look at the operation's actual tensor types, not an unused alias or an
    # explicit_encoding attribute that could have survived on the wrong type.
    type_lines = lines if operation.startswith("scf.") else [line.split(" : ", 1)[-1] for line in lines]
    assert all(expected in _expand_aliases(line, aliases) for line in type_lines), "\n".join(lines)


def _assert_lowered(assembly):
    assert "iluvatar_tle.gpu.set_layout" in assembly["ttir"]
    assert "iluvatar_tle.gpu.set_layout" not in assembly["ttgir"]
    assert re.search(r'ttg\.target\s*=\s*"cuda:(71|80)"', assembly["ttgir"])
    assert '"ttg.threads-per-warp" = 64' in assembly["ttgir"]
    assert assembly["llir"]
    assert "iluvatar_tle.gpu.set_layout" not in assembly["llir"]


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
def _data_only(src, out, n: tl.constexpr, LAYOUT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tle.gpu.set_layout(tl.load(src + offsets, offsets < n, other=0), LAYOUT)
    # No pointer hint: the data's hard encoding must reach the store pointers
    # and mask and survive memory coalescing.
    tl.store(out + offsets, values * 2.0, offsets < n)


@triton.jit
def _sequential(src, out_a, out_b, LAYOUT_A: tl.constexpr, LAYOUT_B: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    a = tle.gpu.set_layout(tl.load(src + offsets), LAYOUT_A)
    b = tle.gpu.set_layout(a, LAYOUT_B)
    # Both results must remain live, so this exercises two requested domains.
    tl.store(tle.gpu.set_layout(out_a + offsets, LAYOUT_A), a + 1.0)
    tl.store(tle.gpu.set_layout(out_b + offsets, LAYOUT_B), b + 2.0)


@triton.jit
def _shared_producer(src, out_a, out_b, LAYOUT_A: tl.constexpr, LAYOUT_B: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    root = tl.load(src + offsets)
    shared = root + root
    a = tle.gpu.set_layout(shared, LAYOUT_A)
    b = tle.gpu.set_layout(shared, LAYOUT_B)
    tl.store(tle.gpu.set_layout(out_a + offsets, LAYOUT_A), a + 1.0)
    tl.store(tle.gpu.set_layout(out_b + offsets, LAYOUT_B), b + 2.0)


@triton.jit
def _data_only_domains(src, out_a, out_b, LAYOUT_A: tl.constexpr, LAYOUT_B: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    root = tl.load(src + offsets)
    shared = root + root
    a = tle.gpu.set_layout(shared, LAYOUT_A)
    b = tle.gpu.set_layout(shared, LAYOUT_B)
    tl.store(out_a + offsets, a + 1.0)
    tl.store(out_b + offsets, b + 2.0)


@triton.jit
def _soft_and_hard_stores(src, out_soft, out_a, out_b, LAYOUT_A: tl.constexpr, LAYOUT_B: tl.constexpr,
                          BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(src + offsets)
    a = tle.gpu.set_layout(values, LAYOUT_A)
    b = tle.gpu.set_layout(BLOCK - 1 - offsets, LAYOUT_B)
    tl.store(out_soft + offsets, values)
    tl.store(out_a + offsets, a)
    tl.store(out_b + offsets, b)


@triton.jit
def _shared_memory(src, out, LAYOUT: tl.constexpr, BLOCK: tl.constexpr, COPY: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    if COPY:
        buffer = tle.gpu.alloc((BLOCK, ), dtype=tl.float32, nv_mma_shared_layout=False)
        pointers = tle.gpu.set_layout(src + offsets, LAYOUT)
        tle.gpu.copy(pointers, buffer, (BLOCK, ))
    else:
        value = tle.gpu.set_layout(tl.load(src + offsets), LAYOUT)
        buffer = tle.gpu.alloc((BLOCK, ), dtype=tl.float32, init_value=value, nv_mma_shared_layout=False)
    tl.debug_barrier()
    pointers = tle.gpu.local_ptr(buffer, (offsets, ))
    value = tle.gpu.set_layout(tl.load(pointers), LAYOUT)
    tl.store(out + offsets, value * 2.0 - 4.0)


@triton.jit
def _loop(src, out, repeats, LAYOUT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tle.gpu.set_layout(tl.load(src + offsets), LAYOUT)
    for _ in range(repeats):
        values = values + 1.0
    tl.store(tle.gpu.set_layout(out + offsets, LAYOUT), values)


@triton.jit
def _while_loop(src, out, repeats, LAYOUT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tle.gpu.set_layout(tl.load(src + offsets), LAYOUT)
    iteration = 0
    while iteration < repeats:
        values = values + 1.0
        iteration += 1
    tl.store(out + offsets, values)


@triton.jit
def _conditional(src, out, condition, LAYOUT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tle.gpu.set_layout(tl.load(src + offsets), LAYOUT)
    if condition:
        values = values + 1.0
    else:
        values = values * 2.0
    tl.store(out + offsets, values)


@triton.jit
def _reduce(src, out, LAYOUT: tl.constexpr, RESULT_LAYOUT: tl.constexpr):
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 32)
    values = tl.load(src + rows[:, None] * 32 + cols[None, :])
    values = tle.gpu.set_layout(values, LAYOUT)
    result = tl.sum(values, axis=1)
    tl.store(tle.gpu.set_layout(out + rows, RESULT_LAYOUT), result)


@triton.jit
def _gather(src, out, SOURCE_LAYOUT: tl.constexpr, RESULT_LAYOUT: tl.constexpr, EXPLICIT_RESULT: tl.constexpr,
            BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tle.gpu.set_layout(tl.load(src + offsets), SOURCE_LAYOUT)
    indices = BLOCK - 1 - offsets
    if EXPLICIT_RESULT:
        result = tle.gpu.set_layout(tl.gather(values, indices, axis=0), RESULT_LAYOUT)
    else:
        indices = tle.gpu.set_layout(indices, RESULT_LAYOUT)
        result = tl.gather(values, indices, axis=0)
    tl.store(out + offsets, result)


@triton.jit
def _reduce_result(src, out, LAYOUT: tl.constexpr):
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 32)
    values = tl.load(src + rows[:, None] * 32 + cols[None, :])
    result = tle.gpu.set_layout(tl.sum(values, axis=1), LAYOUT)
    tl.store(out + rows, result)


@triton.jit
def _scan_result(src, out, LAYOUT: tl.constexpr):
    offsets = tl.arange(0, 256)
    values = tl.load(src + offsets)
    result = tle.gpu.set_layout(tl.cumsum(values, axis=0), LAYOUT)
    tl.store(out + offsets, result)


@triton.jit
def _dot(lhs, rhs, out, LAYOUT: tl.constexpr):
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 16)
    inner = tl.arange(0, 32)
    a = tl.load(lhs + rows[:, None] * 32 + inner[None, :])
    b = tl.load(rhs + inner[:, None] * 16 + cols[None, :])
    result = tle.gpu.set_layout(tl.dot(a, b), LAYOUT)
    tl.store(out + rows[:, None] * 16 + cols[None, :], result)


@triton.jit
def _layout_helper(src, LAYOUT: tl.constexpr, BLOCK: tl.constexpr):
    return tle.gpu.set_layout(tl.load(src + tl.arange(0, BLOCK)), LAYOUT)


@triton.jit
def _inline_helper(src, out, LAYOUT: tl.constexpr, BLOCK: tl.constexpr):
    value = _layout_helper(src, LAYOUT, BLOCK)
    tl.store(out + tl.arange(0, BLOCK), value + 3.0)


@triton.jit
def _invalid(out, LAYOUT: tl.constexpr):
    values = tle.gpu.set_layout(tl.arange(0, 256), LAYOUT)
    tl.store(out + tl.arange(0, 256), values)


@triton.jit
def _scalar(out, LAYOUT: tl.constexpr):
    tl.store(out, tle.gpu.set_layout(1, LAYOUT))


@triton.jit
def _conflict(src, out, LAYOUT_A: tl.constexpr, LAYOUT_B: tl.constexpr):
    offsets = tl.arange(0, 256)
    values = tl.load(src + offsets)
    a = tle.gpu.set_layout(values, LAYOUT_A)
    b = tle.gpu.set_layout(values, LAYOUT_B)
    # Merging incompatible hard layouts without an explicit conversion is an
    # error; sequential set_layout in _sequential is a legal conversion.
    tl.store(out + offsets, a + b)


def test_baseline_without_set_layout_compiles_offline(iluvatar_backend):
    assembly = _compile_offline(iluvatar_backend, _baseline, {"src": "*fp32", "out": "*fp32"},
                                {"n": 125, "BLOCK": BLOCK})
    assert all(assembly[stage] for stage in ("ttir", "ttgir", "llir"))
    assert all("tle." not in text for text in assembly.values())
    assert re.search(r'ttg\.target\s*=\s*"cuda:71', assembly["ttgir"])
    assert "arith.addf" in assembly["ttgir"]
    assert "tt.store" in assembly["ttgir"]


@pytest.mark.parametrize("kind", ["blocked", "sliced", "recursive-sliced"])
def test_set_layout_compiles_offline(iluvatar_backend, kind):
    layout = _layout(kind)
    assembly = _compile_offline(iluvatar_backend, _pointwise, {"src": "*fp32", "out": "*fp32"},
                                {"n": 125, "LAYOUT": layout, "BLOCK": BLOCK})
    _assert_lowered(assembly)
    _assert_layout(assembly["ttgir"], "arith.addf", str(BLOCK), "f32", layout)
    _assert_layout(assembly["ttgir"], "tt.store", str(BLOCK), "!tt.ptr<f32>", layout)


@pytest.mark.parametrize("kind", ["blocked", "sliced", "recursive-sliced"])
def test_set_layout_target_assembly_offline(iluvatar_backend, kind):
    # This runs the real ivcore11 LLVM instruction selector and register
    # allocator, without a CoreX launcher, linker discovery or GPU execution.
    layout = _layout(kind)
    assembly = _compile_offline(iluvatar_backend, _pointwise, {"src": "*fp32", "out": "*fp32"},
                                {"n": 125, "LAYOUT": layout, "BLOCK": BLOCK}, asm=True)
    _assert_lowered(assembly)
    _assert_layout(assembly["ttgir"], "tt.store", str(BLOCK), "!tt.ptr<f32>", layout)
    assert ".iluvatar_kernel _pointwise" in assembly["asm"]
    assert ".iluvatar_kernel_code_t" in assembly["asm"]


def test_set_layout_data_only_store_compiles_offline(iluvatar_backend):
    layout = _blocked(2)
    assembly = _compile_offline(iluvatar_backend, _data_only, {"src": "*fp32", "out": "*fp32"},
                                {"n": 125, "LAYOUT": layout, "BLOCK": BLOCK})
    _assert_lowered(assembly)
    _assert_layout(assembly["ttgir"], "arith.mulf", str(BLOCK), "f32", layout)
    _assert_layout(assembly["ttgir"], "tt.store", str(BLOCK), "!tt.ptr<f32>", layout)


@pytest.mark.parametrize("kernel", [_sequential, _shared_producer, _data_only_domains])
def test_set_layout_multiple_domains_compiles_offline(iluvatar_backend, kernel):
    a, b = _blocked(1), _blocked(2)
    assembly = _compile_offline(iluvatar_backend, kernel, {"src": "*fp32", "out_a": "*fp32", "out_b": "*fp32"},
                                {"LAYOUT_A": a, "LAYOUT_B": b, "BLOCK": BLOCK})
    _assert_lowered(assembly)
    ttgir = assembly["ttgir"]
    if kernel is not _sequential:
        # Specialize pure pointwise producers without duplicating their load.
        assert len([line for line in ttgir.splitlines() if "tt.load " in line]) == 1
    # Do not require a convert instruction: legal rematerialization can remove
    # it. The two final stores must still use their respective encodings.
    stores = [line for line in ttgir.splitlines() if "tt.store " in line]
    assert len(stores) == 2
    aliases = dict(re.findall(r"^(#\w+)\s*=\s*(.+)$", ttgir, re.MULTILINE))
    store_types = [_expand_aliases(line.split(" : ", 1)[-1], aliases) for line in stores]
    for layout in (a, b):
        expected = re.sub(r"\s+", "", f"tensor<{BLOCK}x!tt.ptr<f32>, {_encoding(layout)}>")
        assert any(expected in types for types in store_types), "\n".join(stores)


def test_set_layout_preserves_unconstrained_store(iluvatar_backend):
    a, b = _blocked(1), _blocked(2)
    assembly = _compile_offline(iluvatar_backend, _soft_and_hard_stores,
                                {"src": "*fp32", "out_soft": "*fp32", "out_a": "*fp32", "out_b": "*i32"},
                                {"LAYOUT_A": a, "LAYOUT_B": b, "BLOCK": BLOCK})
    _assert_lowered(assembly)
    stores = [line for line in assembly["ttgir"].splitlines() if "tt.store " in line]
    assert len(stores) == 3
    soft, = [line for line in stores if "tle.explicit_memory_encoding" not in line]
    assert "!tt.ptr<f32>" in soft
    hard_stores = "\n".join(line for line in stores if "tle.explicit_memory_encoding" in line)
    aliases = "\n".join(line for line in assembly["ttgir"].splitlines() if line.startswith("#"))
    hard_ir = aliases + "\n" + hard_stores
    _assert_layout(hard_ir, "tt.store", str(BLOCK), "!tt.ptr<f32>", a, contains="!tt.ptr<f32>")
    _assert_layout(hard_ir, "tt.store", str(BLOCK), "!tt.ptr<i32>", b, contains="!tt.ptr<i32>")


@pytest.mark.parametrize("copy", [False, True], ids=["alloc-local-ptr", "copy-local-ptr"])
def test_set_layout_shared_memory_compiles_offline(iluvatar_backend, copy):
    layout = _blocked(2)
    assembly = _compile_offline(iluvatar_backend, _shared_memory, {"src": "*fp32", "out": "*fp32"},
                                {"LAYOUT": layout, "BLOCK": BLOCK, "COPY": copy})
    _assert_lowered(assembly)
    ttgir = assembly["ttgir"]
    assert "ttg.local_alloc" in ttgir
    if "ttg.local_load" in ttgir:
        _assert_layout(ttgir, "ttg.local_load", str(BLOCK), "f32", layout)
    else:
        # A protected local pointer load can remain a tt.load in address
        # space 3 instead of becoming a ttg.local_load. Both are shared reads.
        assert "iluvatar_tle.local_pointers" in ttgir
        _assert_layout(ttgir, "tt.load", str(BLOCK), "!tt.ptr<f32, 3>", layout, contains="!tt.ptr<f32, 3>")
    _assert_layout(ttgir, "arith.mulf", str(BLOCK), "f32", layout)
    _assert_layout(ttgir, "tt.store", str(BLOCK), "!tt.ptr<f32>", layout, contains="!tt.ptr<f32>")
    assert "iluvatar_tle.local_pointers" not in assembly["llir"]


@pytest.mark.parametrize("kernel,operation", [(_loop, "scf.for"), (_while_loop, "scf.while")])
def test_set_layout_loop_compiles_offline(iluvatar_backend, kernel, operation):
    layout = _blocked()
    assembly = _compile_offline(iluvatar_backend, kernel, {"src": "*fp32", "out": "*fp32", "repeats": "i32"},
                                {"LAYOUT": layout, "BLOCK": BLOCK})
    _assert_lowered(assembly)
    _assert_layout(assembly["ttgir"], operation, str(BLOCK), "f32", layout)
    _assert_layout(assembly["ttgir"], "arith.addf", str(BLOCK), "f32", layout)
    _assert_layout(assembly["ttgir"], "tt.store", str(BLOCK), "!tt.ptr<f32>", layout)


def test_set_layout_if_compiles_offline(iluvatar_backend):
    layout = _blocked(2)
    assembly = _compile_offline(iluvatar_backend, _conditional, {"src": "*fp32", "out": "*fp32", "condition": "i1"},
                                {"LAYOUT": layout, "BLOCK": BLOCK})
    _assert_lowered(assembly)
    ttgir = assembly["ttgir"]
    # Canonicalization may replace the branch with a select; either result
    # must carry the requested encoding all the way to the store.
    operation = "scf.if" if "scf.if" in ttgir else "arith.select"
    _assert_layout(ttgir, operation, str(BLOCK), "f32", layout)
    _assert_layout(ttgir, "tt.store", str(BLOCK), "!tt.ptr<f32>", layout)


def test_set_layout_reduce_compiles_offline(iluvatar_backend):
    layout = _matrix_layout()
    result_layout = tle.gpu.SlicedEncoding(1, layout)
    assembly = _compile_offline(iluvatar_backend, _reduce, {"src": "*fp32", "out": "*fp32"},
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


@pytest.mark.parametrize("explicit_result", [False, True], ids=["indices-layout", "result-layout"])
def test_set_layout_gather_preserves_independent_source_layout(iluvatar_backend, explicit_result):
    source_layout, result_layout = _blocked(1), _blocked(2)
    assembly = _compile_offline(iluvatar_backend, _gather, {"src": "*fp32", "out": "*fp32"}, {
        "SOURCE_LAYOUT": source_layout,
        "RESULT_LAYOUT": result_layout,
        "EXPLICIT_RESULT": explicit_result,
        "BLOCK": BLOCK,
    })
    _assert_lowered(assembly)
    ttgir = assembly["ttgir"]
    aliases = dict(re.findall(r"^(#\w+)\s*=\s*(.+)$", ttgir, re.MULTILINE))
    gather, = [line for line in ttgir.splitlines() if "tt.gather " in line]
    # Gather's indices and result share a layout; its source is independent.
    signature = (f"(tensor<{BLOCK}xf32, {_encoding(source_layout)}>, "
                 f"tensor<{BLOCK}xi32, {_encoding(result_layout)}>) -> "
                 f"tensor<{BLOCK}xf32, {_encoding(result_layout)}>")
    assert re.sub(r"\s+", "", signature) in _expand_aliases(gather, aliases), gather
    _assert_layout(ttgir, "tt.store", str(BLOCK), "!tt.ptr<f32>", result_layout)


@pytest.mark.parametrize("kernel,kind,shape,operation", [
    (_reduce_result, "blocked", "16", "tt.reduce"),
    (_scan_result, "sliced", "256", "tt.scan"),
])
def test_set_layout_converts_reduce_scan_result(iluvatar_backend, kernel, kind, shape, operation):
    layout = _layout(kind)
    assembly = _compile_offline(iluvatar_backend, kernel, {"src": "*fp32", "out": "*fp32"}, {"LAYOUT": layout})
    _assert_lowered(assembly)
    # The producer can keep its own supported encoding. set_layout converts
    # its result to the requested layout before the memory operation.
    assert f'"{operation}"' in assembly["ttgir"]
    _assert_layout(assembly["ttgir"], "tt.store", shape, "!tt.ptr<f32>", layout)


def test_set_layout_blocked_dot_result_compiles_offline(iluvatar_backend):
    layout = _matrix_layout()
    assembly = _compile_offline(iluvatar_backend, _dot, {"lhs": "*fp16", "rhs": "*fp16", "out": "*fp32"},
                                {"LAYOUT": layout})
    _assert_lowered(assembly)
    # Matmul acceleration may use an internal MMA result, but its bridge to
    # the requested blocked layout must reach the store unchanged.
    assert "tt.dot" in assembly["ttgir"] or "ttng.warp_group_dot" in assembly["ttgir"]
    _assert_layout(assembly["ttgir"], "tt.store", "16x16", "!tt.ptr<f32>", layout)


def test_set_layout_normal_helper_is_inlined_offline(iluvatar_backend):
    layout = _blocked(2)
    assembly = _compile_offline(iluvatar_backend, _inline_helper, {"src": "*fp32", "out": "*fp32"},
                                {"LAYOUT": layout, "BLOCK": BLOCK})
    _assert_lowered(assembly)
    assert "tt.call" not in assembly["ttgir"]
    _assert_layout(assembly["ttgir"], "arith.addf", str(BLOCK), "f32", layout)
    _assert_layout(assembly["ttgir"], "tt.store", str(BLOCK), "!tt.ptr<f32>", layout)


def test_set_layout_rejects_direct_mma_result_offline(iluvatar_backend, capfd):
    layout = tle.gpu.MmaEncoding([2, 0], [4, 1], [16, 8])
    with pytest.raises((CompilationError, RuntimeError, ValueError)) as error:
        _compile_offline(iluvatar_backend, _dot, {"lhs": "*fp16", "rhs": "*fp16", "out": "*fp32"}, {"LAYOUT": layout})
    assert "only supports BlockEncoding and SlicedEncoding" in str(error.value) + capfd.readouterr().err


@pytest.mark.parametrize("case,diagnostic", [
    ("rank", "layout rank must match the tensor rank"),
    ("shared", "requires a distributed"),
    ("mma", "only supports BlockEncoding and SlicedEncoding"),
    ("threads", "threads per warp, but the target requires 64"),
    ("warps", "warps per CTA, but the context requires"),
    ("ctas", "requires num_ctas = 1"),
])
def test_set_layout_rejects_invalid_layout_offline(iluvatar_backend, capfd, case, diagnostic):
    layouts = {
        "rank": _matrix_layout(),
        "shared": tle.gpu.swizzled_shared_layout.make_default(1),
        "mma": tle.gpu.SlicedEncoding(0, tle.gpu.MmaEncoding([2, 0], [4, 1], [16, 8])),
        "threads": tle.gpu.BlockEncoding([1], [16], [4], [0]),
        "warps": tle.gpu.BlockEncoding([1], [64], [8], [0]),
        "ctas": tle.gpu.BlockEncoding([1], [64], [4], [0], cga_layout=[[1]]),
    }
    with pytest.raises((CompilationError, RuntimeError, ValueError)) as error:
        _compile_offline(iluvatar_backend, _invalid, {"out": "*i32"}, {"LAYOUT": layouts[case]})
    assert diagnostic in str(error.value) + capfd.readouterr().err


def test_set_layout_rejects_scalar_offline(iluvatar_backend):
    with pytest.raises((CompilationError, RuntimeError, ValueError), match="only supports block tensors"):
        _compile_offline(iluvatar_backend, _scalar, {"out": "*i32"}, {"LAYOUT": _blocked()})


def test_set_layout_rejects_conflicting_hard_layouts_offline(iluvatar_backend, capfd):
    with pytest.raises((CompilationError, RuntimeError, ValueError)) as error:
        _compile_offline(iluvatar_backend, _conflict, {"src": "*fp32", "out": "*fp32"},
                         {"LAYOUT_A": _blocked(1), "LAYOUT_B": _blocked(2)})
    assert "conflicting TLE encoding hints" in str(error.value) + capfd.readouterr().err


@pytest.mark.parametrize("kind", ["blocked", "sliced", "recursive-sliced"])
@pytest.mark.parametrize("n", [125, 256, 513])
def test_set_layout_pointwise_runtime(iluvatar_runtime, kind, n):
    layout = _layout(kind)
    reference = torch.arange(n, dtype=torch.float32)
    source = reference.cuda()
    grid = triton.cdiv(n, BLOCK)
    guard = 16
    allocation = torch.full((guard + grid * BLOCK + guard, ), float("nan"), dtype=torch.float32, device="cuda")
    compiled = _pointwise[(grid, )](source, allocation[guard:], n, layout, BLOCK, num_warps=4, num_ctas=1)
    torch.cuda.synchronize()
    _assert_lowered(compiled.asm)
    assert compiled.asm["cubin"]
    _assert_layout(compiled.asm["ttgir"], "arith.addf", str(BLOCK), "f32", layout)
    _assert_layout(compiled.asm["ttgir"], "tt.store", str(BLOCK), "!tt.ptr<f32>", layout)
    actual = allocation.cpu()
    torch.testing.assert_close(actual[guard:guard + n], reference + 7, rtol=0, atol=0)
    assert torch.isnan(actual[:guard]).all()
    assert torch.isnan(actual[guard + n:]).all()


@pytest.mark.parametrize("kernel", [_sequential, _shared_producer, _data_only_domains])
def test_set_layout_multiple_domains_runtime(iluvatar_runtime, kernel):
    reference = torch.arange(BLOCK, dtype=torch.float32)
    source = reference.cuda()
    a = torch.full_like(source, float("nan"))
    b = torch.full_like(source, float("nan"))
    kernel[(1, )](source, a, b, _blocked(1), _blocked(2), BLOCK, num_warps=4)
    torch.cuda.synchronize()
    base = reference if kernel is _sequential else reference * 2
    torch.testing.assert_close(a.cpu(), base + 1, rtol=0, atol=0)
    torch.testing.assert_close(b.cpu(), base + 2, rtol=0, atol=0)


@pytest.mark.parametrize("copy", [False, True], ids=["alloc-local-ptr", "copy-local-ptr"])
def test_set_layout_shared_memory_runtime(iluvatar_runtime, copy):
    reference = torch.arange(BLOCK, dtype=torch.float32) * 0.5 - 9.0
    source = reference.cuda()
    out = torch.full_like(source, float("nan"))
    _shared_memory[(1, )](source, out, _blocked(2), BLOCK, copy, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.cpu(), reference * 2.0 - 4.0, rtol=0, atol=0)


@pytest.mark.parametrize("kernel", [_loop, _while_loop])
def test_set_layout_loop_runtime(iluvatar_runtime, kernel):
    reference = torch.arange(BLOCK, dtype=torch.float32)
    source = reference.cuda()
    out = torch.full_like(source, float("nan"))
    kernel[(1, )](source, out, 5, _blocked(), BLOCK, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.cpu(), reference + 5, rtol=0, atol=0)


@pytest.mark.parametrize("condition", [False, True])
def test_set_layout_if_runtime(iluvatar_runtime, condition):
    reference = torch.arange(BLOCK, dtype=torch.float32)
    source = reference.cuda()
    out = torch.full_like(source, float("nan"))
    _conditional[(1, )](source, out, condition, _blocked(2), BLOCK, num_warps=4)
    torch.cuda.synchronize()
    expected = reference + 1 if condition else reference * 2
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_set_layout_data_only_runtime(iluvatar_runtime):
    n = 125
    reference = torch.arange(n, dtype=torch.float32)
    source = reference.cuda()
    out = torch.full((BLOCK, ), float("nan"), device="cuda", dtype=torch.float32)
    _data_only[(1, )](source, out, n, _blocked(2), BLOCK, num_warps=4)
    torch.cuda.synchronize()
    actual = out.cpu()
    torch.testing.assert_close(actual[:n], reference * 2, rtol=0, atol=0)
    assert torch.isnan(actual[n:]).all()


def test_set_layout_reduce_runtime(iluvatar_runtime):
    reference = torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32)
    source = reference.cuda()
    out = torch.full((16, ), float("nan"), dtype=torch.float32, device="cuda")
    layout = _matrix_layout()
    _reduce[(1, )](source, out, layout, tle.gpu.SlicedEncoding(1, layout), num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.cpu(), reference.sum(dim=1), rtol=0, atol=0)
