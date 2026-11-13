"""CPU regressions for the P800 SDNN wrapped-load lowering.

These execute the native passes without a device. Hardware tests are in
language/test_rank1_wrap.py. No files from a local experiment are required.
"""
import re

import pytest

triton = pytest.importorskip("triton")
tl = triton.language

lib = pytest.importorskip("triton._C.libtriton")
pytestmark = pytest.mark.skipif(not hasattr(lib, "xpu"), reason="XPU build required")


@triton.jit
def _wrapped_load(src, dst, PERIOD: tl.constexpr, LENGTH: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(src + x % PERIOD, x < LENGTH)
    tl.store(dst + x, value, x < LENGTH)


@pytest.mark.parametrize("period", [7, 17, 32])
def test_backend_pipeline_compiles_wrapped_load(tmp_path, monkeypatch, period):
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TRITON_REPRODUCER_PATH", str(tmp_path / "pipeline"))
    kernel = triton.compile(
        ASTSource(_wrapped_load, {"src": "*fp32", "dst": "*fp32"},
                  constexprs={"PERIOD": period, "LENGTH": 19, "BLOCK": 16}), target=GPUTarget("cuda", 3, 1),
        options={"is_sdnn": True})
    assert kernel.metadata.is_sdnn and kernel.asm["xpubin"]
    text = (tmp_path / "pipeline.make_ttxir.repro.mlir").read_text()
    capture = text.index("tritonxpu-capture-rank1-wrap")
    vendor = text.index("triton-to-linalg-experimental")
    repair = text.index("tritonxpu-repair-rank1-wrap")
    loop_grid = text.index("tritonsdnn-loop-grid")
    assert capture < vendor < repair < loop_grid


def module_text(period=7, length=19, block=16):
    return f"""module {{
  tt.func public @copy(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {{
    %period = arith.constant dense<{period}> : tensor<{block}xi32>
    %length = arith.constant dense<{length}> : tensor<{block}xi32>
    %block = arith.constant {block} : i32
    %pid = tt.get_program_id x : i32
    %base = arith.muli %pid, %block : i32
    %lane = tt.make_range {{end = {block} : i32, start = 0 : i32}} : tensor<{block}xi32>
    %bases = tt.splat %base : i32 -> tensor<{block}xi32>
    %x = arith.addi %bases, %lane : tensor<{block}xi32>
    %mask = arith.cmpi slt, %x, %length : tensor<{block}xi32>
    %index = arith.remsi %x, %period : tensor<{block}xi32>
    %src = tt.splat %arg0 : !tt.ptr<f32> -> tensor<{block}x!tt.ptr<f32>>
    %ptr = tt.addptr %src, %index : tensor<{block}x!tt.ptr<f32>>, tensor<{block}xi32>
    %value = tt.load %ptr, %mask : tensor<{block}x!tt.ptr<f32>>
    %dst = tt.splat %arg1 : !tt.ptr<f32> -> tensor<{block}x!tt.ptr<f32>>
    %out = tt.addptr %dst, %x : tensor<{block}x!tt.ptr<f32>>, tensor<{block}xi32>
    tt.store %out, %value, %mask : tensor<{block}x!tt.ptr<f32>>
    tt.return
  }}
}}
"""


def parse(tmp_path, text):
    path = tmp_path / "input.mlir"
    path.write_text(text)
    ctx = lib.ir.context()
    lib.ir.load_dialects(ctx)
    lib.xpu.load_dialects(ctx)
    return ctx, lib.ir.parse_mlir_module(str(path), ctx)


def run(ctx, mod, *steps):
    pm = lib.ir.pass_manager(ctx)
    for add, args in steps:
        add(pm, *args)
    pm.run(mod, "test")


def dump(ctx, mod, monkeypatch, tmp_path, name):
    # Reproducers work even in builds that conceal str(module).
    stem = tmp_path / name
    monkeypatch.setenv("TRITON_REPRODUCER_PATH", str(stem))
    run(ctx, mod)
    return (tmp_path / f"{name}.test.repro.mlir").read_text().split("{-#", 1)[0]


def through_repair():
    x = lib.xpu.passes.ttsdnnir
    return [
        (x.add_capture_rank1_wrap_pass, ()),
        (x.add_convert_triton_to_tritonsdnn_pass, (3, 131072)),
        (lib.passes.ttir.add_loop_aware_cse, ()),
        (x.add_linalg_to_tritonsdnn_pass, (3, )),
        (lib.passes.ttir.add_loop_aware_cse, ()),
        (x.add_tritonsdnn_legalize_pass, (3, )),
        (x.add_tritonsdnn_merge_extern_ew_pass, ()),
        (x.add_tritonsdnn_combine_before_pass, (3, )),
        (x.add_tritonsdnn_resolve_layout_conflict_pass, ()),
        (x.add_tritonsdnn_hoist_ds_pass, ()),
        (x.add_tritonsdnn_transpose_mma_pass, ()),
        (x.add_tritonsdnn_transpose_ew_pass, ()),
        (x.add_tritonsdnn_combine_before_pass, (3, )),
        (x.add_tritonsdnn_wrap_fallback_pass, ()),
        (x.add_tritonsdnn_eliminate_mma_acc_zero_pass, (3, )),
        (x.add_tritonsdnn_optimize_rc_layout_pass, (3, )),
        (x.add_tritonsdnn_dsa_copy_pass, ()),
        (x.add_tritonsdnn_bufferize_pass, (3, )),
        (x.add_tritonsdnn_mx_scale_layout_pass, ()),
        (x.add_tritonsdnn_fuse_relu_activation_pass, (3, )),
        (x.add_tritonsdnn_ew_act_table_pass, (None, )),
        (x.add_repair_rank1_wrap_pass, ()),
    ]


class AddressEvaluator:
    """Evaluate the emitted SSA, rejecting operations outside this fixture.

    The oracle is the original per-lane modulo expression, not the repair's
    segment formula. GPU tests independently check the actual copied bits.
    """

    ssa = r"%[A-Za-z0-9_]+"

    def __init__(self, definitions, pid, values=None):
        self.definitions = definitions
        self.pid = pid
        self.values = dict(values or {})

    def value(self, name):
        if re.fullmatch(r"-?\d+", name):
            return int(name)
        if name in self.values:
            return self.values[name]
        op = self.definitions[name]
        if match := re.fullmatch(r"arith.constant (-?\d+) : (?:index|i32)", op):
            return int(match[1])
        if op == "tt.get_program_id x : i32":
            return self.pid
        if match := re.fullmatch(rf"arith.index_cast ({self.ssa}) : i32 to index", op):
            return self.value(match[1])
        if match := re.fullmatch(rf"arith.(\w+) ({self.ssa}), ({self.ssa}) : (?:index|i32)", op):
            import operator
            a, b = self.value(match[2]), self.value(match[3])
            operations = {
                "addi": operator.add, "subi": operator.sub, "muli": operator.mul, "minsi": min, "maxsi": max, "minui":
                min, "divsi": operator.floordiv, "remsi": operator.mod
            }
            assert match[1] in operations, op
            if match[1] in ("divsi", "remsi", "minui"):
                assert a >= 0 and b >= (1 if match[1] != "minui" else 0)
            return operations[match[1]](a, b)
        if match := re.fullmatch(rf"arith.cmpi sgt, ({self.ssa}), ({self.ssa}) : index", op):
            return self.value(match[1]) > self.value(match[2])
        if match := re.fullmatch(rf"arith.select ({self.ssa}), ({self.ssa}), ({self.ssa}) : index", op):
            return self.value(match[2] if self.value(match[1]) else match[3])
        raise AssertionError(f"Unhandled scalar expression: {op}")

    def view(self, name):
        op = self.definitions[name]
        if match := re.match(
                r"memref.reinterpret_cast (%arg\d+) to offset: \[([^]]+)\], "
                r"sizes: \[([^]]+)\], strides: \[([^]]+)\] : ", op):
            return match[1], *(self.value(match[i]) for i in (2, 3, 4))
        if match := re.match(rf"memref.subview ({self.ssa})\[([^]]+)\] \[([^]]+)\] \[([^]]+)\] : ", op):
            root, base, extent, stride = self.view(match[1])
            offset, count, step = (self.value(match[i]) for i in (2, 3, 4))
            assert offset >= 0 and count >= 0 and offset + count <= extent
            return root, base + offset * stride, count, stride * step
        if match := re.match(rf"memref.cast ({self.ssa}) : ", op):
            return self.view(match[1])
        if match := re.match(r"memref.alloc\(\).* : memref<(\d+)xf32>$", op):
            return name, 0, int(match[1]), 1
        raise AssertionError(f"Unhandled view: {op}")


def assert_addresses(text, period, length, block):
    ssa = AddressEvaluator.ssa
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if " = scf.for " in line]
    assert len(starts) == 1
    i = starts[0]
    loop = re.fullmatch(
        rf"\s*({ssa}) = scf.for ({ssa}) = ({ssa}) to ({ssa}) step ({ssa}) "
        rf"iter_args\(({ssa}) = ({ssa})\) -> \(index\) \{{", lines[i])
    assert loop
    j = next(j for j in range(i + 1, len(lines)) if lines[j].strip() == "}")
    body = lines[i + 1:j]
    definitions = dict(re.findall(rf"^\s*({ssa}) = (.+)$", "\n".join(lines[:i] + body), re.M))
    copies = re.findall(rf"^\s*sdnn.dma in ({ssa}), ({ssa})", "\n".join(body), re.M)
    assert len(copies) == 1
    yielded = re.fullmatch(rf"\s*scf.yield ({ssa}) : index", body[-1])
    assert yielded
    for pid in range((length + block - 1) // block + 1):
        evaluator = AddressEvaluator(definitions, pid)
        lo, hi, step = (evaluator.value(loop[k]) for k in (3, 4, 5))
        assert lo == 0 and step == 1 and 0 <= hi <= block + 1
        pos = evaluator.value(loop[7])
        actual = {}
        for iteration in range(lo, hi, step):
            inner = AddressEvaluator(definitions, pid, {loop[2]: iteration, loop[6]: pos})
            src = inner.view(copies[0][0])
            dst = inner.view(copies[0][1])
            assert src[0] == "%arg0" and dst[0].startswith("%alloc")
            assert src[3] == dst[3] == 1 and src[2] == dst[2] > 0
            for lane in range(src[2]):
                assert dst[1] + lane not in actual
                assert 0 <= src[1] + lane < period
                actual[dst[1] + lane] = src[1] + lane
            pos = inner.value(yielded[1])
        valid = max(0, min(block, length - pid * block))
        assert actual == {lane: (pid * block + lane) % period for lane in range(valid)}


@pytest.mark.parametrize("period", [2, 7, 17, 31, 32, 4099])
@pytest.mark.parametrize("block", [16, 512, 1024])
@pytest.mark.parametrize("tail", [-15, -1, 0, 1, 21])
def test_wrapped_dma_addresses(tmp_path, monkeypatch, period, block, tail):
    length = block + tail
    ctx, mod = parse(tmp_path, module_text(period, length, block))
    run(ctx, mod, *through_repair())
    text = dump(ctx, mod, monkeypatch, tmp_path, "repaired")
    assert_addresses(text, period, length, block)


@pytest.mark.parametrize("change", [
    lambda t: t.replace("dense<7>", "dense<0>"),
    lambda t: t.replace("dense<7>", "dense<-7>"),
    lambda t: t.replace("arith.remsi", "arith.remui"),
    lambda t: t.replace("cmpi slt", "cmpi sle"),
    lambda t: t.replace("arith.constant 16 : i32", "arith.constant 32 : i32"),
    lambda t: t.replace("dense<19>", "dense<-1>"),
    lambda t: t.replace("dense<19>", "dense<2147483647>"),
    lambda t: t.replace("end = 16 : i32, start = 0", "end = 17 : i32, start = 1"),
    lambda t: t.replace("%ptr, %mask", "%ptr"),
    lambda t: t.replace("    %dst =", "    %extra = tt.load %ptr, %mask : tensor<16x!tt.ptr<f32>>\n    %dst ="),
])
def test_unsupported_modulo_is_rejected(tmp_path, capfd, change):
    ctx, mod = parse(tmp_path, change(module_text()))
    with pytest.raises(RuntimeError):
        run(ctx, mod, (lib.xpu.passes.ttsdnnir.add_capture_rank1_wrap_pass, ()))
    assert "unsupported P800 SDNN rank-1 modulo load" in capfd.readouterr().err


@pytest.mark.parametrize("sequence", ["repair", "capture,capture", "capture,repair"])
def test_invalid_pass_order(tmp_path, capfd, sequence):
    ctx, mod = parse(tmp_path, module_text())
    x = lib.xpu.passes.ttsdnnir
    methods = {"capture": x.add_capture_rank1_wrap_pass, "repair": x.add_repair_rank1_wrap_pass}
    with pytest.raises(RuntimeError):
        for name in sequence.split(","):
            run(ctx, mod, (methods[name], ()))
    assert "rank-1 wrap" in capfd.readouterr().err


@pytest.mark.parametrize("scalar_modulo", [False, True])
def test_contiguous_load_unchanged(tmp_path, monkeypatch, scalar_modulo):
    text = module_text().replace("tt.addptr %src, %index", "tt.addptr %src, %x")
    if scalar_modulo:
        text = text.replace("    %base =", "    %wrapped_pid = arith.remsi %pid, %block : i32\n    %base =")
        text = text.replace("arith.muli %pid, %block", "arith.muli %wrapped_pid, %block")
    ctx, mod = parse(tmp_path, text)
    before = dump(ctx, mod, monkeypatch, tmp_path, "before")
    x = lib.xpu.passes.ttsdnnir
    run(ctx, mod, (x.add_capture_rank1_wrap_pass, ()), (x.add_repair_rank1_wrap_pass, ()))
    assert dump(ctx, mod, monkeypatch, tmp_path, "after") == before
