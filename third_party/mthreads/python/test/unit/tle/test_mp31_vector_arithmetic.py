"""MP31 typed casts and FP8 SQMMA arithmetic, including fallback boundaries."""
import re

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir, llvm, mthreads
from test_tle_utils import compile_musa, mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _cast(x, y, n, BLOCK: tl.constexpr, ROUND: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x + i, i < n, other=0)
    if ROUND == "rtne" or ROUND == "rtz":
        v = v.to(y.dtype.element_ty, fp_downcast_rounding=ROUND)
    else:
        v = v.to(y.dtype.element_ty)
    tl.store(y + i, v, i < n)


@pytest.mark.parametrize("source,dest,rounding,pattern", [
    ("i32", "fp32", "", r"sitofp <4 x i32>"),
    ("u32", "fp32", "", None),
    ("i16", "fp32", "", None),
    ("fp32", "fp8e4nv", "rtne", r"call <4 x i8> @llvm.musa.f2e4m3.rn.bst4"),
    ("fp32", "fp8e4nv", "rtz", None),
    ("fp32", "fp8e5", "rtne", None),
])
def test_compile_cast(source, dest, rounding, pattern):
    signature = {"x": "*" + source, "y": "*" + dest, "n": "i32", "BLOCK": "constexpr", "ROUND": "constexpr"}
    for enabled in (False, True):
        k = compile_musa(_cast, signature, {"BLOCK": 1024, "ROUND": rounding},
                         options={"mp31_vector_arithmetic": enabled})
        text = k.asm["llir"]
        assert k.asm["mubin"] and not re.search(r'\bcall\b[^\n@]*\basm\b', text)
        if pattern and enabled:
            assert re.search(pattern, text)
        if source != "i32":
            assert "sitofp <4 x i32>" not in text
        assert ("call <4 x i8> @llvm.musa.f2e4m3.rn.bst4" in text) == (enabled and dest == "fp8e4nv"
                                                                       and rounding == "rtne")


def test_option_scope():
    _, backend = mthreads_backend()
    off = backend.parse_options({})
    assert not off.mp31_vector_arithmetic
    assert off.hash() != backend.parse_options({"mp31_vector_arithmetic": True}).hash()
    with pytest.raises(ValueError, match="must be a bool"):
        backend.parse_options({"mp31_vector_arithmetic": 1})
    with pytest.raises(ValueError, match="requires MP31"):
        backend.parse_options({"mp31_vector_arithmetic": True, "arch": 32})


def _lower(tmp_path, source):
    p = tmp_path / "arithmetic.mlir"
    p.write_text("module {\n" + source + "\n}")
    ctx = ir.context()
    ir.load_dialects(ctx)
    _, backend = mthreads_backend()
    backend.load_dialects(ctx)
    llvm_ctx = llvm.context()
    mod = llvm.to_module(ir.parse_mlir_module(str(p), ctx), llvm_ctx)
    before = str(mod)
    count = mthreads.lower_mp31_fp32_adds(mod)
    after = str(mod)
    assert " asm " not in after
    return count, before, after


@pytest.mark.parametrize("kind,width,early_wait,strict", [
    ("e4m3", 1, False, False),
    ("e4m3", 2, False, False),
    ("e5m2", 4, False, False),
    ("s8", 4, False, False),
    ("e4m3", 4, True, False),
    ("e4m3", 2, False, True),
])
def test_overlap_keeps_wait_and_rounding(tmp_path, kind, width, early_wait, strict):
    mma = f"llvm.musa.sqmma.{kind}.m32n128k128.mma"
    types = ", ".join(["i32", "i32", "vector<32xi32>"] + ["i32"] * 9)
    args = ", ".join(["%d", "%d", "%acc"] + ["%d"] * 9)
    ty = "f32" if width == 1 else f"vector<{width}xf32>"
    wait = "llvm.call @llvm.musa.sqmma.wait() : () -> ()"
    attrs = 'attributes {passthrough = ["strictfp"]}' if strict else ""
    n, before, after = _lower(
        tmp_path, f"""
      llvm.func @{mma}({types}) -> vector<32xi32>
      llvm.func @llvm.musa.sqmma.wait()
      llvm.func @test(%d: i32, %acc: vector<32xi32>, %a: {ty}, %b: {ty}) -> {ty} {attrs} {{
        %m = llvm.call @{mma}({args}) : ({types}) -> vector<32xi32>
        {wait if early_wait else ""}
        %s = llvm.fadd %a, %b : {ty}
        {"" if early_wait else wait}
        llvm.return %s : {ty}
      }}""")
    if early_wait or strict or kind == "s8":
        assert n == 0 and before == after
    else:
        assert n == 1
        assert ("@llvm.fma.f32" if width == 1 else f"@llvm.musa.add.f.bst{width}.src01") in after
        assert after.count("call void @llvm.musa.sqmma.wait()") == 1
        assert " fast " not in after and " reassoc " not in after


@pytest.mark.parametrize("mask,changed", [("1, -1, -1, -1", True), ("1, 0, -1, -1", False)])
def test_pack_pair_uses_only_single_defined_lane(tmp_path, mask, changed):
    n, before, after = _lower(
        tmp_path, f"""
      llvm.func @llvm.musa.f2e4m3.rn.bst4(vector<4xf32>) -> vector<4xi8>
      llvm.func @save(vector<4xi8>)
      llvm.func @probe(%v: vector<4xf32>) -> vector<4xf32> {{
        %p = llvm.call @llvm.musa.f2e4m3.rn.bst4(%v) : (vector<4xf32>) -> vector<4xi8>
        %nil = llvm.mlir.poison : vector<4xf32>
        %s = llvm.shufflevector %v, %nil [{mask}] : vector<4xf32>
        %sum = llvm.fadd %v, %s : vector<4xf32>
        llvm.call @save(%p) : (vector<4xi8>) -> ()
        llvm.return %sum : vector<4xf32>
      }}""")
    assert bool(n) == changed
    assert (before != after) == changed
    assert ("@llvm.fma.f32" in after) == changed
    assert after.count("call <4 x i8> @llvm.musa.f2e4m3.rn.bst4") == 1


@pytest.mark.parametrize("kind", ["i32", "e4m3"])
def test_device_cast_graph_refresh(kind):
    import os
    if os.environ.get("TRITON_MUSA_RUN_NUMERICAL_DEVICE_TESTS") != "1":
        pytest.skip("requires an isolated MP31 device")
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_musa")
    if kind == "i32":
        values = torch.tensor([0, -1, 1, 2**24 - 1, 2**24 + 1, -2**24 - 1, 2**31 - 1, -2**31], dtype=torch.int32)
        dest, rounding = torch.float32, ""
    else:
        values = torch.tensor([0., -0., 2**-10, -2**-10, 3 * 2**-10, 2**-9, 1.0625, 1.1875, -1.1875, 448., -448.])
        dest, rounding = torch.float8_e4m3fn, "rtne"
        codes = torch.arange(256, dtype=torch.uint8).view(dest).float()
        finite = torch.unique(codes[codes.isfinite()]).sort().values
        mid = (finite[:-1] + finite[1:]) * 0.5
        values = torch.cat([
            values, codes, mid,
            torch.nextafter(mid, torch.full_like(mid, -float("inf"))),
            torch.nextafter(mid, torch.full_like(mid, float("inf"))),
            torch.tensor([464., -464., 465., -465., float("inf"), -float("inf")])
        ])
    # Include a partial last CTA. The tail after n is caller-owned.
    x = torch.empty(2051, dtype=values.dtype, device="musa")
    y = torch.zeros(2055, dtype=dest, device="musa")
    control = torch.zeros_like(y)
    n = x.numel()

    def run():
        _cast[(3, )](x, y, n, 1024, rounding, mp31_vector_arithmetic=True)
        _cast[(3, )](x, control, n, 1024, rounding)

    run()
    g = torch.musa.MUSAGraph()
    with torch.musa.graph(g):
        run()
    for roll in (0, 3, 1, 0):
        raw = values.roll(roll).repeat((n + values.numel() - 1) // values.numel())[:n]
        x.copy_(raw)
        g.replay()
        torch.musa.synchronize()
        actual = y.cpu().view(torch.uint8)
        assert torch.equal(actual, control.cpu().view(torch.uint8))
        if kind == "e4m3":
            defined = raw.abs() <= 448
            assert torch.equal(y[:n].cpu().view(torch.uint8)[defined], raw[defined].to(dest).view(torch.uint8))
        else:
            assert torch.equal(y[:n].cpu().view(torch.uint8), raw.to(dest).view(torch.uint8))
        assert not actual[n * y.element_size():].any()


def packed_transform(tmp_path, width=4, dependent=False, memory=False, strict=False, int8=False, late_view=False):
    suffix = 's8' if int8 else 'e4m3'
    attrs = 'attributes {passthrough = ["strictfp"]}' if strict else ''
    dtype = f'vector<{width}xf32>'
    types = ', '.join(['i32', 'i32', 'vector<32xi32>'] + ['i32'] * 9)
    args = ', '.join(['%d0', '%d1', '%zero'] + ['%c0'] * 9)
    body = []
    for i in range(width):
        body.append(f'%idx{i} = llvm.mlir.constant({i} : i32) : i32')
        body.append(f'%x{i} = llvm.extractelement %x[%idx{i} : i32] : {dtype}')
        body.append(f'%y{i} = llvm.extractelement %y[%idx{i} : i32] : {dtype}')
        if memory and i == width // 2:
            body.append('llvm.call @observe(f32 %sum0) : (f32) -> ()'.replace('f32 %sum0', '%sum0'))
        a = f'%sum{i-1}' if dependent and i else f'%x{i}'
        if late_view and i:
            body.append(f'%view{i} = llvm.insertelement %sum{i-1}, %x[%idx0 : i32] : {dtype}')
            body.append(f'%pick{i} = llvm.extractelement %view{i}[%idx0 : i32] : {dtype}')
            a = f'%pick{i}'
        body.append(f'%sum{i} = llvm.fadd {a}, %y{i} : f32')
    body.append(f'%nil = llvm.mlir.poison : {dtype}')
    for i in range(width):
        body.append(
            f'%r{i} = llvm.insertelement %sum{i}, {"%nil" if not i else "%r"+str(i-1)}[%idx{i} : i32] : {dtype}')
    body.append('llvm.call @llvm.musa.sqmma.wait() : () -> ()')
    body.append(f'llvm.return %r{width-1} : {dtype}')
    source = f"""
      llvm.func @llvm.musa.sqmma.{suffix}.m32n128k128.mma({types}) -> vector<32xi32>
      llvm.func @llvm.musa.sqmma.wait()
      llvm.func @observe(f32)
      llvm.func @probe(%d0: i32, %d1: i32, %zero: vector<32xi32>, %x: {dtype}, %y: {dtype}) -> {dtype} {attrs} {{
        %c0 = llvm.mlir.constant(0 : i32) : i32
        %mma = llvm.call @llvm.musa.sqmma.{suffix}.m32n128k128.mma({args}) : ({types}) -> vector<32xi32>
        {chr(10).join(body)}
      }}"""
    return _lower(tmp_path, source)


@pytest.mark.parametrize("width,burst", [(2, 2), (3, 2), (4, 4), (8, 4)])
def test_packs_independent_sums(tmp_path, width, burst):
    _, _, result = packed_transform(tmp_path, width=width)
    assert f"call <{burst} x float> @llvm.musa.add.f.bst{burst}.src01" in result
    assert result.count("call void @llvm.musa.sqmma.wait") == 1


@pytest.mark.parametrize("case", ["dependent", "late_view", "strict", "int8"])
def test_respects_add_dependencies(tmp_path, case):
    _, before, result = packed_transform(tmp_path, **{case: True})
    assert "@llvm.musa.add.f.bst" not in result
    if case in ("strict", "int8"):
        assert before == result


def test_unknown_call_splits_groups(tmp_path):
    _, _, result = packed_transform(tmp_path, memory=True)
    assert result.count("call <2 x float> @llvm.musa.add.f.bst2.src01") == 2
    assert result.count("call void @observe") == 1
