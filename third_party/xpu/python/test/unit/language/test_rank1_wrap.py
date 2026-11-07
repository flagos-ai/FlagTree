"""P800 device regressions: exact modulo-load bits and allocation guards."""
import json

import pytest
import triton
import triton.language as tl


@triton.jit
def wrapped_copy(src, dst, PERIOD: tl.constexpr, LENGTH: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = x < LENGTH
    value = tl.load(src + x % PERIOD, mask)
    tl.store(dst + x, value, mask)


@pytest.fixture(scope="module")
def p800():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("P800 device required")
    from triton.backends import backends
    from triton.runtime import driver
    xpu = backends.get("xpu")
    target = driver.active.get_current_target()
    # torch_xmlir may report the generic name "GPU" for a real P800. Identify
    # the XPU driver and its hardware model instead of that display string.
    if xpu is None or not isinstance(driver.active, xpu.driver) or target.arch != 3:
        pytest.skip("P800 device required")
    return torch


CASES = [(7, 19, 16), (17, 19, 16), (32, 19, 16), (7, 16, 16), (17, 1, 16), (2, 37, 16), (4099, 5125, 512),
         (4099, 5125, 1024)]


@pytest.mark.parametrize("period,length,block", CASES)
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_wrapped_copy_bits_and_guards(p800, tmp_path, period, length, block, dtype):
    torch = p800
    dtype = getattr(torch, dtype)
    integer = torch.int32 if dtype == torch.float32 else torch.int16
    # Retain enough physical storage even for the old, incorrect unwrapped load.
    # Reading beyond the logical source must fail the assertion, not harm a GPU.
    grid = triton.cdiv(length, block) + 1
    host = torch.full((max(period, grid * block) + 64, ), -1000, dtype=dtype)
    special = {
        torch.float32: [0, -2147483648, 2139095040, -8388608, 2143289344, 1, -2147483647, 8388607],
        torch.float16: [0, -32768, 31744, -1024, 32256, 1, -32767, 1023],
        torch.bfloat16: [0, -32768, 32640, -128, 32704, 1, -32767, 127],
    }
    checks = []
    for kind in ("ordinary", "special", "one_hot"):
        if kind == "ordinary":
            host[:period] = (100 + torch.arange(period) % 31).to(dtype)
        elif kind == "special":
            bits = special[dtype]
            host[:period] = torch.tensor((bits * ((period + len(bits) - 1) // len(bits)))[:period],
                                         dtype=integer).view(dtype)
        else:
            host[:period] = 0
            host[0] = 1
        src = host.to("cuda")
        dst = torch.full((grid * block + 64, ), -777, dtype=dtype, device="cuda")
        kernel = wrapped_copy[(grid, )](src[:period], dst[:length], PERIOD=period, LENGTH=length, BLOCK=block,
                                        is_sdnn=True)
        torch.cuda.synchronize()
        actual = dst.cpu()
        expected = host[torch.arange(length) % period]
        bad = (actual[:length].view(integer) != expected.view(integer)).nonzero().flatten().tolist()
        source_ok = torch.equal(src.cpu().view(integer), host.view(integer))
        guard_ok = bool(torch.all(actual[length:] == -777))
        checks.append(dict(kind=kind, mismatch_indices=bad, source_unchanged=source_ok, guard_unchanged=guard_ok))
        (tmp_path / "result.json").write_text(
            json.dumps(
                dict(period=period, length=length, block=block, dtype=str(dtype), checks=checks,
                     is_sdnn=kernel.metadata.is_sdnn), indent=2))
        assert kernel.metadata.is_sdnn
        assert not bad and source_ok and guard_ok
