import os
import re
import tempfile

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler.errors import CompilationError

from utils import compile_iluvatar


@triton.jit
def _tle_cumsum_masked_kernel(x_ptr, exclusive_ptr, total_ptr, n: tl.constexpr, BLOCK: tl.constexpr,
                              REVERSE: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    exclusive, total = tle.cumsum(x, axis=0, reverse=REVERSE)
    tl.store(exclusive_ptr + offs, exclusive, mask=mask)
    tl.store(total_ptr, total)


@triton.jit
def _cumsum_2d_axis_kernel(src, out, ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.arange(0, ROWS)[:, None]
    cols = tl.arange(0, COLS)[None, :]
    values = tl.load(src + rows * COLS + cols)
    exclusive, _ = tle.cumsum(values, axis=1)
    tl.store(out + rows * COLS + cols, exclusive)


_CUMSUM_SIGNATURE = {
    "x_ptr": "*fp32",
    "exclusive_ptr": "*fp32",
    "total_ptr": "*fp32",
    "n": "constexpr",
    "BLOCK": "constexpr",
    "REVERSE": "constexpr",
}


def _compile_cumsum(reverse=False):
    return compile_iluvatar(
        _tle_cumsum_masked_kernel,
        signature=_CUMSUM_SIGNATURE,
        constexprs={"n": 127, "BLOCK": 128, "REVERSE": reverse},
    )


def test_tle_cumsum_builder_binding_is_backend_local():
    from triton._C import libtriton
    from triton._C.libtriton import ir

    context = ir.context()
    ir.load_dialects(context)
    builder = ir.builder(context)

    assert hasattr(builder, "create_exclusive_cumsum")
    assert hasattr(libtriton, "iluvatar")
    assert not hasattr(libtriton, "tle")


def test_tle_cumsum_ttir_uses_iluvatar_tle_op():
    ttir = _compile_cumsum().asm["ttir"]
    assert "iluvatar_tle.exclusive_cumsum" in ttir, ttir
    assert re.search(r"(?<!iluvatar_)\btle\.exclusive_cumsum\b", ttir) is None, ttir


def test_tle_cumsum_ttgir_lowers_to_scan_reduce():
    ttgir = _compile_cumsum().asm["ttgir"]
    assert "iluvatar_tle.exclusive_cumsum" not in ttgir, ttgir
    assert '"tt.scan"' in ttgir, ttgir
    assert '"tt.reduce"' in ttgir, ttgir
    assert ("arith.subi" in ttgir) or ("arith.subf" in ttgir), ttgir


def test_tle_cumsum_reverse_ttgir_uses_reverse_scan():
    ttgir = _compile_cumsum(reverse=True).asm["ttgir"]
    assert '"tt.scan"' in ttgir, ttgir
    assert "reverse = true" in ttgir, ttgir


def test_tle_cumsum_no_iluvatar_tle_reaches_llir():
    llir = _compile_cumsum().asm["llir"]
    assert "iluvatar_tle.exclusive_cumsum" not in llir, llir


def test_tle_cumsum_rejects_2d_axis_1(capfd):
    with pytest.raises((CompilationError, RuntimeError)) as excinfo:
        compile_iluvatar(
            _cumsum_2d_axis_kernel,
            signature={
                "src": "*fp32",
                "out": "*fp32",
                "ROWS": "constexpr",
                "COLS": "constexpr",
            },
            constexprs={"ROWS": 16, "COLS": 16},
        )
    diagnostics = str(excinfo.value) + capfd.readouterr().err
    assert ("currently only rank-1 tensors are supported" in diagnostics
            or "currently only axis=0 is supported" in diagnostics
            or "PassManager::run failed" in diagnostics), diagnostics


def _pick_expected_dtype(input_dtype: torch.dtype) -> torch.dtype:
    if input_dtype in (torch.int8, torch.int16):
        return torch.int32
    if input_dtype == torch.bfloat16:
        return torch.float32
    return input_dtype


def _make_input(dtype: torch.dtype, block: int) -> torch.Tensor:
    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        return torch.randn((block, ), device="cuda", dtype=dtype)
    if dtype == torch.int8:
        return torch.randint(-32, 32, (block, ), device="cuda", dtype=dtype)
    if dtype == torch.int16:
        return torch.randint(-512, 512, (block, ), device="cuda", dtype=dtype)
    if dtype == torch.int32:
        return torch.randint(-2048, 2048, (block, ), device="cuda", dtype=dtype)
    raise AssertionError(f"unsupported dtype for test: {dtype}")


@pytest.mark.parametrize(
    "dtype, n, block, reverse, num_warps",
    [
        (torch.int8, 511, 512, False, 16),
        (torch.int16, 257, 512, False, 16),
        (torch.int32, 512, 512, False, 16),
        (torch.int32, 256, 256, True, 8),
        (torch.int32, 512, 512, True, 16),
        (torch.float16, 127, 128, False, 4),
        (torch.float32, 128, 128, True, 4),
        (torch.float32, 512, 512, True, 16),
        (torch.bfloat16, 193, 256, False, 8),
    ],
)
def test_tle_cumsum_exclusive_and_total(dtype, n, block, reverse, num_warps):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported on this GPU")

    x = _make_input(dtype, block)
    out_dtype = _pick_expected_dtype(dtype)
    exclusive = torch.empty((block, ), device="cuda", dtype=out_dtype)
    total = torch.empty((1, ), device="cuda", dtype=out_dtype)

    _tle_cumsum_masked_kernel[(1, )](
        x,
        exclusive,
        total,
        n,
        BLOCK=block,
        REVERSE=reverse,
        num_warps=num_warps,
    )

    x_valid = x[:n].to(out_dtype)
    if reverse:
        expected_exclusive = torch.flip(torch.cumsum(torch.flip(x_valid, dims=[0]), dim=0, dtype=out_dtype),
                                        dims=[0]) - x_valid
    else:
        expected_exclusive = torch.cumsum(x_valid, dim=0, dtype=out_dtype) - x_valid
    expected_total = torch.sum(x_valid, dim=0, dtype=out_dtype)

    if out_dtype in (torch.float16, torch.bfloat16):
        torch.testing.assert_close(exclusive[:n], expected_exclusive, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(total[0], expected_total, atol=2e-2, rtol=2e-2)
    elif out_dtype == torch.float32:
        # GPU parallel scan accumulation order differs from torch's sequential
        # cumsum reference, especially in reverse mode.
        atol = 1e-5 if reverse else 2e-6
        rtol = 5e-4 if reverse else 1e-5
        torch.testing.assert_close(exclusive[:n], expected_exclusive, atol=atol, rtol=rtol)
        torch.testing.assert_close(total[0], expected_total, atol=2e-6, rtol=1e-5)
    else:
        torch.testing.assert_close(exclusive[:n], expected_exclusive)
        torch.testing.assert_close(total[0], expected_total)


# ---------------------------------------------------------------------------
# OptimizeExclusiveCumsumLayouts (fold convert_layout around the op).
#
# Remains lit-style (mirrors `test_tle_optimize_exclusive_cumsum_layouts.mlir`);
# the main-repo Python cumsum tests do not cover this pass in isolation.
# ---------------------------------------------------------------------------

_BLOCKED = ("#ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], "
            "warpsPerCTA = [4], order = [0], isSme = false, smeWarpsPerCTA = [0]}>")
_BLOCKED1 = ("#ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], "
             "warpsPerCTA = [4], order = [0], isSme = false, smeWarpsPerCTA = [0]}>")

_MODULE_HEADER = ('module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, '
                  'ttg.target = "cuda:71", "ttg.threads-per-warp" = 64 : i32} {')


def _fold_cvt_cumsum_cvt_ttgir():
    return f"""#blocked = {_BLOCKED}
#blocked1 = {_BLOCKED1}
{_MODULE_HEADER}
  tt.func @fold_cvt_cumsum_cvt(%arg0: tensor<256xi32, #blocked>) -> (tensor<256xi32, #blocked>, i32) {{
    %0 = ttg.convert_layout %arg0 : tensor<256xi32, #blocked> -> tensor<256xi32, #blocked1>
    %exclusive, %total = iluvatar_tle.exclusive_cumsum %0 {{axis = 0 : i32, reverse = false}} : tensor<256xi32, #blocked1> -> tensor<256xi32, #blocked1>, i32
    %1 = ttg.convert_layout %exclusive : tensor<256xi32, #blocked1> -> tensor<256xi32, #blocked>
    tt.return %1, %total : tensor<256xi32, #blocked>, i32
  }}
}}
"""


def _keep_when_non_convert_user_exists_ttgir():
    return f"""#blocked = {_BLOCKED}
#blocked1 = {_BLOCKED1}
{_MODULE_HEADER}
  tt.func @keep_when_non_convert_user_exists(%arg0: tensor<256xi32, #blocked>) -> (tensor<256xi32, #blocked1>, i32) {{
    %0 = ttg.convert_layout %arg0 : tensor<256xi32, #blocked> -> tensor<256xi32, #blocked1>
    %exclusive, %total = iluvatar_tle.exclusive_cumsum %0 {{axis = 0 : i32, reverse = false}} : tensor<256xi32, #blocked1> -> tensor<256xi32, #blocked1>, i32
    %1 = arith.addi %exclusive, %exclusive : tensor<256xi32, #blocked1>
    tt.return %1, %total : tensor<256xi32, #blocked1>, i32
  }}
}}
"""


def _run_optimize_layouts_pass(ttgir_text):
    from triton._C.libtriton import ir, iluvatar

    context = ir.context()
    ir.load_dialects(context)
    iluvatar.load_dialects(context)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "module.ttgir")
        with open(path, "w") as f:
            f.write(ttgir_text)
        module = ir.parse_mlir_module(path, context)

    module.context = context
    pm = ir.pass_manager(context)
    iluvatar.passes.tle.add_optimize_exclusive_cumsum_layouts(pm)
    pm.run(module, "optimize_exclusive_cumsum_layouts")
    return module.str_nodebug()


def test_tle_cumsum_optimize_layouts_pass_is_exposed():
    from triton._C.libtriton import iluvatar
    assert hasattr(iluvatar.passes.tle, "add_optimize_exclusive_cumsum_layouts")


def test_tle_cumsum_fold_cvt_cumsum_cvt():
    out = _run_optimize_layouts_pass(_fold_cvt_cumsum_cvt_ttgir())
    # The cumsum op survives (it is only lowered by the later lower pass) ...
    assert "iluvatar_tle.exclusive_cumsum" in out, out
    # ... but the surrounding convert_layout sandwich must be folded away.
    assert "ttg.convert_layout" not in out, out


def test_tle_cumsum_keep_when_non_convert_user_exists():
    out = _run_optimize_layouts_pass(_keep_when_non_convert_user_exists_ttgir())
    # A non-convert consumer blocks folding: the op and input convert stay.
    assert "iluvatar_tle.exclusive_cumsum" in out, out
    assert "ttg.convert_layout" in out, out


def test_tle_cumsum_pipeline_still_lowers_after_optimize_layouts():
    # End-to-end: with the optimize pass wired before the lower pass, a normal
    # cumsum kernel must still fully lower to tt.scan/tt.reduce (no op left).
    ttgir = _compile_cumsum().asm["ttgir"]
    assert "iluvatar_tle.exclusive_cumsum" not in ttgir, ttgir
    assert '"tt.scan"' in ttgir, ttgir
    assert '"tt.reduce"' in ttgir, ttgir
