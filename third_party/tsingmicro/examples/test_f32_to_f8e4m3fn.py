import math

import pytest
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Golden encodings vs ml_dtypes.float8_e4m3fn / arith.truncf rtne.
# E4M3FN has no Inf: overflow and Inf map to NaN (0x7F / 0xFF).
F32_TO_F8E4M3FN_CASES = [
    (0.0, 0x00),
    (-0.0, 0x80),
    (1.0, 0x38),
    (-1.0, 0xB8),
    (256.0, 0x78),
    (448.0, 0x7E),
    (464.0, 0x7E),
    (472.0, 0x7F),
    (480.0, 0x7F),
    (496.0, 0x7F),
    (512.0, 0x7F),
    (1024.0, 0x7F),
    (1e38, 0x7F),
    (float("inf"), 0x7F),
    (float("-inf"), 0xFF),
    (float("nan"), 0x7F),
    (-512.0, 0xFF),
    (-600.0, 0xFF),
    (2**-9, 0x01),
    (2**-10, 0x00),
]

# Semantic checks for the f8 -> f32 inverse (E4M3FN, bias 7).
F8E4M3FN_TO_F32_CASES = [
    (0x00, 0.0),
    (0x80, -0.0),
    (0x01, 2**-9),
    (0x04, 2**-7),
    (0x08, 2**-6),
    (0x38, 1.0),
    (0xB8, -1.0),
    (0x78, 256.0),
    (0x7E, 448.0),
    (0x7F, float("nan")),
    (0xFF, float("nan")),
]


@triton.jit
def f32_to_f8_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x.to(tl.float8e4nv, fp_downcast_rounding="rtne")
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def f8_to_f32_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x.to(tl.float32)
    tl.store(y_ptr + offsets, y, mask=mask)


def f32_to_f8e4m3fn(x):
    n_elements = x.numel()
    output = torch.empty(n_elements, dtype=torch.float8_e4m3fn, device=x.device)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    f32_to_f8_kernel[grid](x, output, n_elements, BLOCK_SIZE=32)
    return output


def f8e4m3fn_to_f32(x):
    n_elements = x.numel()
    output = torch.empty(n_elements, dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    f8_to_f32_kernel[grid](x, output, n_elements, BLOCK_SIZE=32)
    return output


def _skip_if_cpu():
    if str(DEVICE).startswith("cpu"):
        pytest.skip("f8E4M3FN software lowering is TX81-only")


def test_f32_to_f8e4m3fn():
    _skip_if_cpu()

    vals = [v for v, _ in F32_TO_F8E4M3FN_CASES]
    expected = [e for _, e in F32_TO_F8E4M3FN_CASES]
    x = torch.tensor(vals, dtype=torch.float32, device=DEVICE)
    got = f32_to_f8e4m3fn(x).to("cpu").view(torch.uint8).tolist()

    mismatches = []
    for v, exp, actual in zip(vals, expected, got):
        ok = actual == exp
        if math.isnan(v):
            ok = (actual & 0x7F) == 0x7F
        if not ok:
            mismatches.append(f"{v}: got {actual:#04x}, expected {exp:#04x}")
    assert not mismatches, "f32->f8E4M3FN mismatches:\n  " + "\n  ".join(mismatches)


def test_f8e4m3fn_to_f32():
    _skip_if_cpu()

    codes = [c for c, _ in F8E4M3FN_TO_F32_CASES]
    expected = [v for _, v in F8E4M3FN_TO_F32_CASES]
    x = torch.tensor(codes, dtype=torch.uint8, device="cpu").view(torch.float8_e4m3fn).to(DEVICE)
    got = f8e4m3fn_to_f32(x).to("cpu").tolist()

    mismatches = []
    for code, exp, actual in zip(codes, expected, got):
        if math.isnan(exp):
            ok = math.isnan(actual)
        else:
            ok = actual == exp or (exp == 0.0 and actual == 0.0
                                   and math.copysign(1.0, actual) == math.copysign(1.0, exp))
        if not ok:
            mismatches.append(f"{code:#04x}: got {actual!r}, expected {exp!r}")
    assert not mismatches, "f8E4M3FN->f32 mismatches:\n  " + "\n  ".join(mismatches)


def test_f8e4m3fn_roundtrip():
    _skip_if_cpu()

    codes = list(range(256))
    x = torch.tensor(codes, dtype=torch.uint8, device="cpu").view(torch.float8_e4m3fn).to(DEVICE)
    back = f32_to_f8e4m3fn(f8e4m3fn_to_f32(x)).to("cpu").view(torch.uint8).tolist()

    mismatches = []
    for code, actual in zip(codes, back):
        if (code & 0x7F) == 0x7F:
            ok = (actual & 0x7F) == 0x7F
        else:
            ok = actual == code
        if not ok:
            mismatches.append(f"{code:#04x}: got {actual:#04x}")
    assert not mismatches, "f8->f32->f8 mismatches:\n  " + "\n  ".join(mismatches)


if __name__ == "__main__":
    test_f32_to_f8e4m3fn()
    print("test_f32_to_f8e4m3fn passed")
    test_f8e4m3fn_to_f32()
    print("test_f8e4m3fn_to_f32 passed")
    test_f8e4m3fn_roundtrip()
    print("test_f8e4m3fn_roundtrip passed")
