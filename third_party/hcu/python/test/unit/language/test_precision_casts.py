import pytest
import torch

import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.runtime.interpreter import _encode_fp64_to_float

from test_precision import conversion_source_values, precision_cast


@pytest.mark.parametrize("integer", [torch.int8, torch.int32])
@pytest.mark.parametrize("floating", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("to_float", [False, True])
def test_hcu_standard_integer_cast_values(integer, floating, to_float, device):
    if to_float:
        source = (torch.arange(-128, 128).to(integer) if integer == torch.int8 else torch.tensor(
            [-(2**31), 2**31 - 1, -16777217, 16777217, -65521, 65521, -128, 127, -1, 0, 1], dtype=integer))
        expected = source.to(floating)
        output = torch.full((source.numel() + 32, ), 42, dtype=floating, device=device)
    else:
        source = torch.tensor([
            -2147483520., -65504., -128.75, -128., -127.5, -1.5, -.75, -0., 0., .75, 1.5, 126.75, 127.5, 128., 65504.,
            2147483520.
        ], dtype=floating)
        truncated = source.double().trunc()
        valid = source.isfinite() & (truncated >= torch.iinfo(integer).min) & (truncated <= torch.iinfo(integer).max)
        source = source[valid]
        expected = source.to(integer)
        output = torch.full((source.numel() + 32, ), 42, dtype=integer, device=device)
    precision_cast[(triton.cdiv(source.numel(), 256), )](source.to(device), output[16:-16], source.numel(), 256)
    torch.testing.assert_close(output[16:-16].cpu(), expected, rtol=0, atol=0)
    assert torch.all(output[:16] == 42) and torch.all(output[-16:] == 42)


@pytest.mark.parametrize("to_int8", [False, True])
def test_hcu_integer_width_cast_values(to_int8, device):
    source = (torch.tensor([-(2**31), 2**31 - 1, -257, -256, -129, -128, 127, 128, 255, 256], dtype=torch.int32)
              if to_int8 else torch.arange(-128, 128, dtype=torch.int8))
    destination = torch.int8 if to_int8 else torch.int32
    expected = source.to(destination)
    output = torch.full((source.numel() + 32, ), 42, dtype=destination, device=device)
    precision_cast[(triton.cdiv(source.numel(), 256), )](source.to(device), output[16:-16], source.numel(), 256)
    torch.testing.assert_close(output[16:-16].cpu(), expected, rtol=0, atol=0)
    assert torch.all(output[:16] == 42) and torch.all(output[-16:] == 42)


@pytest.mark.parametrize("source", ["fp16", "bf16", "fp32", "fp64"])
@pytest.mark.parametrize("destination", ["fp8e4nv", "fp8e5"])
@pytest.mark.parametrize("rounding", [None, "rtne", "rtz"])
def test_hcu_fp8_downcast_compile(source, destination, rounding, fresh_triton_cache):
    triton.compile(
        ASTSource(precision_cast, {"X": "*" + source, "Y": "*" + destination},
                  constexprs={"N": 513, "ROUND": rounding, "BLOCK": 256}), target=GPUTarget("hip", "gfx936", 64),
        options={"num_warps": 4})


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("destination", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("rounding", [None, "rtne", "rtz"])
def test_hcu_fp8_downcast_values(dtype, destination, rounding, device):
    levels = torch.arange(128).to(torch.uint8).view(destination).double()
    levels = levels[levels.isfinite()]
    boundaries = ((levels[:-1] + levels[1:]) / 2).tolist() + [450., 464., 465., 57344., 61440., 65504.]
    source = conversion_source_values(dtype, boundaries)
    limit = 448 if destination == torch.float8_e4m3fn else 57344
    clipped = source.double().clamp(-limit, limit)
    fmt = tl.float8e4nv if destination == torch.float8_e4m3fn else tl.float8e5
    mode = ir.ROUNDING_MODE.RTZ if rounding == "rtz" else ir.ROUNDING_MODE.RTNE
    expected = torch.from_numpy(_encode_fp64_to_float(clipped.numpy(), fmt, mode))
    output = torch.full((source.numel() + 32, ), 165, dtype=torch.uint8, device=device).view(destination)
    precision_cast[(triton.cdiv(source.numel(), 256), )](source.to(device), output[16:-16], source.numel(), 256,
                                                         rounding)
    actual = output[16:-16].cpu()
    assert torch.equal(actual.isnan(), source.isnan())
    not_nan = ~source.isnan()
    torch.testing.assert_close(actual.view(torch.uint8)[not_nan], expected[not_nan], rtol=0, atol=0)
    assert torch.all(output[:16].view(torch.uint8) == 165) and torch.all(output[-16:].view(torch.uint8) == 165)


@pytest.mark.parametrize("source_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("destination", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_hcu_fp8_upcast_values(source_dtype, destination, device):
    source = torch.arange(256).to(torch.uint8).view(source_dtype)
    expected = source.to(destination)
    output = torch.full((256 + 32, ), 42, dtype=destination, device=device)
    precision_cast[(1, )](source.to(device), output[16:-16], 256, 256)
    actual = output[16:-16].cpu()
    assert torch.equal(actual.isnan(), expected.isnan())
    torch.testing.assert_close(actual[~expected.isnan()], expected[~expected.isnan()], rtol=0, atol=0)
    assert torch.equal(torch.signbit(actual[~expected.isnan()]), torch.signbit(expected[~expected.isnan()]))
    assert torch.all(output[:16] == 42) and torch.all(output[-16:] == 42)


@pytest.mark.parametrize("integer", [torch.int8, torch.int32])
@pytest.mark.parametrize("fp8", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("to_fp8", [False, True])
def test_hcu_fp8_integer_values(integer, fp8, to_fp8, device):
    if to_fp8:
        source = (torch.arange(-128, 128).to(integer) if integer == torch.int8 else torch.tensor(
            [-(2**31), 2**31 - 1, -16842753, 16842753, -57345, 57345, -449, 449, -1, 0, 1], dtype=integer))
        limit = 448 if fp8 == torch.float8_e4m3fn else 57344
        expected = source.float().clamp(-limit, limit).to(fp8).view(torch.uint8)
        output = torch.full((source.numel() + 32, ), 165, dtype=torch.uint8, device=device).view(fp8)
    else:
        source = torch.arange(256).to(torch.uint8).view(fp8)
        values = source.double()
        valid = values.isfinite() & (values >= torch.iinfo(integer).min) & (values <= torch.iinfo(integer).max)
        # Index raw bytes: PyTorch 2.4 has no CPU FP8 masked-index kernel.
        source = source.view(torch.uint8)[valid].view(fp8)
        expected = source.double().trunc().to(integer)
        output = torch.full((source.numel() + 32, ), 42, dtype=integer, device=device)
    precision_cast[(triton.cdiv(source.numel(), 256), )](source.to(device), output[16:-16], source.numel(), 256)
    torch.testing.assert_close(output[16:-16].cpu().view(expected.dtype), expected, rtol=0, atol=0)
    guard = output.view(torch.uint8) if to_fp8 else output
    assert torch.all(guard[:16] == (165 if to_fp8 else 42)) and torch.all(guard[-16:] == (165 if to_fp8 else 42))
