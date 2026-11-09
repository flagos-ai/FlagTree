"""Data checks for the MUSA port of FlagTree #1150 / Triton #11646."""
import os
import pytest
import torch
import torch_musa
import triton

pytestmark = pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device required")


def layout(reg, lane, warp):
    return "#ttg.linear<{register=" + str(reg) + ", lane=" + str([[v, 0] for v in lane]) + ", warp=" + str(
        [[v, 0] for v in warp]) + ", block=[]}>"


def indices(prefix, encoding, m, n):
    row = f"#ttg.slice<{{dim=1, parent={encoding}}}>"
    col = f"#ttg.slice<{{dim=0, parent={encoding}}}>"
    ty = f"tensor<{m}x{n}xi32, {encoding}>"
    return f"""
    %{prefix}r = tt.make_range {{start=0 : i32, end={m} : i32}} : tensor<{m}xi32, {row}>
    %{prefix}c = tt.make_range {{start=0 : i32, end={n} : i32}} : tensor<{n}xi32, {col}>
    %{prefix}rx = tt.expand_dims %{prefix}r {{axis=1 : i32}} : tensor<{m}xi32, {row}> -> tensor<{m}x1xi32, {encoding}>
    %{prefix}cx = tt.expand_dims %{prefix}c {{axis=0 : i32}} : tensor<{n}xi32, {col}> -> tensor<1x{n}xi32, {encoding}>
    %{prefix}rb = tt.broadcast %{prefix}rx : tensor<{m}x1xi32, {encoding}> -> {ty}
    %{prefix}cb = tt.broadcast %{prefix}cx : tensor<1x{n}xi32, {encoding}> -> {ty}
    %{prefix}stride = arith.constant dense<{n}> : {ty}
    %{prefix}scaled = arith.muli %{prefix}rb, %{prefix}stride : {ty}
    %{prefix}offset = arith.addi %{prefix}scaled, %{prefix}cb : {ty}
"""


@pytest.mark.parametrize("case,dtype", [("pointer16", "int32")] +
                         [(case, dtype)
                          for case in ["mixed", "expanded", "reverse", "warp_register", "register_select", "packed"]
                          for dtype in ["int8", "float16", "bfloat16", "int32", "float64"]])
def test_broadcast_layout_shuffle(case, dtype, tmp_path, device):
    m, n, warps = 128, 4, 8
    src = layout([[0, 1], [0, 2]], [1, 2, 4, 8, 16], [32, 64, 0])
    dst = layout([[0, 1], [0, 2], [8, 0]], [0, 0, 1, 2, 4], [32, 64, 16])
    pointer = case == "pointer16"
    if pointer:
        m, n, warps = 16, 1, 4
        src = "#ttg.blocked<{sizePerThread=[1,1], threadsPerWarp=[32,1], warpsPerCTA=[4,1], order=[1,0]}>"
        dst = "#ttg.blocked<{sizePerThread=[1,8], threadsPerWarp=[4,8], warpsPerCTA=[4,1], order=[1,0]}>"
    elif case == "expanded":
        dst = layout([[0, 1], [0, 2], [4, 0], [8, 0]], [0, 0, 0, 1, 2], [32, 64, 16])
    elif case == "reverse":
        src, dst = dst, src
    elif case == "warp_register":
        m, n = 128, 1
        src, dst = layout([[32, 0]], [1, 2, 4, 8, 16], [0, 64, 0]), layout([], [1, 2, 4, 8, 16], [32, 64, 0])
    elif case == "register_select":
        m, n, warps = 64, 1, 2
        src, dst = layout([[1, 0]], [2, 4, 8, 16, 32], [0]), layout([], [1, 2, 4, 8, 16], [32])
    elif case == "packed":
        m, n, warps = 128, 1, 2
        src, dst = layout([[1, 0], [2, 0]], [4, 8, 16, 32, 64], [0]), layout([[4, 0], [1, 0], [2, 0]],
                                                                             [0, 8, 16, 32, 0], [64])
    value_type = {"int8": "i8", "float16": "f16", "bfloat16": "bf16", "int32": "i32", "float64": "f64"}[dtype]
    load_type = "i64" if pointer else value_type
    output_type = "f32" if pointer else value_type
    converted_type = "!tt.ptr<f32>" if pointer else value_type
    load_tensor = f"tensor<{m}x{n}x{load_type}, #src>"
    converted_src = f"tensor<{m}x{n}x{converted_type}, #src>"
    converted_dst = f"tensor<{m}x{n}x{converted_type}, #dst>"
    source = f"""#src = {src}
#dst = {dst}
module attributes {{"ttg.num-ctas"=1 : i32, "ttg.num-warps"={warps} : i32, "ttg.threads-per-warp"=32 : i32, ttg.target="musa:31"}} {{
  tt.func public @convert(%x: !tt.ptr<{load_type}>, %y: !tt.ptr<{output_type}>) {{
{indices("s", "#src", m, n)}
    %xb = tt.splat %x : !tt.ptr<{load_type}> -> tensor<{m}x{n}x!tt.ptr<{load_type}>, #src>
    %xp = tt.addptr %xb, %soffset : tensor<{m}x{n}x!tt.ptr<{load_type}>, #src>, tensor<{m}x{n}xi32, #src>
    %loaded = tt.load %xp : tensor<{m}x{n}x!tt.ptr<{load_type}>, #src>
"""
    value = "%loaded"
    if pointer:
        source += f"    %values = tt.int_to_ptr %loaded : {load_tensor} -> {converted_src}\n"
        value = "%values"
    source += f"    %converted = ttg.convert_layout {value} : {converted_src} -> {converted_dst}\n"
    result = "%converted"
    if pointer:
        source += f"    %result = tt.load %converted : {converted_dst}\n"
        result = "%result"
    source += indices("d", "#dst", m, n) + f"""
    %yb = tt.splat %y : !tt.ptr<{output_type}> -> tensor<{m}x{n}x!tt.ptr<{output_type}>, #dst>
    %yp = tt.addptr %yb, %doffset : tensor<{m}x{n}x!tt.ptr<{output_type}>, #dst>, tensor<{m}x{n}xi32, #dst>
    tt.store %yp, {result} : tensor<{m}x{n}x!tt.ptr<{output_type}>, #dst>
    tt.return
  }}
}}
"""
    path = tmp_path / "convert.ttgir"
    path.write_text(source)
    values = torch.arange(m * n, device=device).reshape(m, n).remainder(199) - 100
    expected = values.to(getattr(torch, "float32" if pointer else dtype))
    x = expected
    if pointer:
        offsets = torch.arange(m * n, device=device, dtype=torch.int64).reshape(m, n)
        x = offsets * expected.element_size() + expected.data_ptr()
    y = torch.empty_like(expected)
    compiled = triton.compile(str(path))
    compiled[(1, 1, 1)](x, y)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    if os.environ.get("FLAGTREE_TEST_MUSA_BROADCAST_SHUFFLE") == "1":
        if case in ("expanded", "reverse", "warp_register"):
            assert compiled.metadata.shared > 0
        else:
            assert compiled.metadata.shared == 0
            assert "shfl" in compiled.asm["llir"]
