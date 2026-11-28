import numpy as np
import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def _int4_unpack(X, Y, N: tl.constexpr, SIGNED: tl.constexpr, BLOCK: tl.constexpr = 128):
    b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    packed = tl.load(X + b, b < (N + 1) // 2, other=0)
    if SIGNED:
        lo = (packed << 4).to(tl.int8, bitcast=True) >> 4
        hi = packed.to(tl.int8, bitcast=True) >> 4
    else:
        lo = packed & 15
        hi = packed >> 4
    values = tl.interleave(lo, hi).to(Y.dtype.element_ty)
    i = tl.program_id(0) * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    tl.store(Y + i, values, i < N)


@triton.jit
def _int4_pack(X, Y, N: tl.constexpr, BLOCK: tl.constexpr = 128):
    i = tl.program_id(0) * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    values = tl.load(X + i, i < N, other=0).to(tl.int8)
    lo, hi = tl.split(tl.reshape(values, (BLOCK, 2)))
    packed = (((hi & 15) << 4) | (lo & 15)).to(tl.uint8)
    b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + b, packed, b < (N + 1) // 2)


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("n", [0, 1, 3, 511, 512, 513])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_metax_int4_roundtrip(signed, n, num_warps, metax_device):
    nbytes = triton.cdiv(n, 2)
    payload = torch.arange(nbytes, dtype=torch.int64).remainder(256).to(torch.uint8)
    if nbytes:
        payload[-1] = 255
    source = torch.full((nbytes + 64, ), 165, dtype=torch.uint8)
    source[32:32 + nbytes] = payload
    dtype = torch.int8 if signed else torch.uint8
    decoded = torch.full((n + 64, ), 73, dtype=dtype, device=metax_device)
    encoded = torch.full((nbytes + 64, ), 165, dtype=torch.uint8, device=metax_device)
    grid = (max(1, triton.cdiv(nbytes, 128)), )
    _int4_unpack[grid](source.to(metax_device)[32:], decoded[32:], n, signed, num_warps=num_warps)
    _int4_pack[grid](decoded[32:], encoded[32:], n, num_warps=num_warps)
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).flatten()[:n].to(torch.int16)
    expected = torch.full((n + 64, ), 73, dtype=dtype)
    expected[32:32 + n] = torch.where(codes >= 8, codes - 16, codes) if signed else codes
    assert torch.equal(decoded.cpu(), expected)
    # Encoding defines zero padding. Copying an existing byte preserves it.
    if n % 2:
        payload[-1] &= 15
    source[32:32 + nbytes] = payload
    assert torch.equal(encoded.cpu(), source)


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("dtype", [
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
])
def test_metax_int4_unpack_cast(signed, dtype, metax_device):
    payload = torch.arange(256, dtype=torch.int64).to(torch.uint8)
    output = torch.full((576, ), 73, dtype=dtype, device=metax_device)
    _int4_unpack[(2, )](payload.to(metax_device), output[32:], 512, signed)
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).flatten().to(torch.int16)
    values = torch.where(codes >= 8, codes - 16, codes) if signed else codes
    expected = torch.full((576, ), 73, dtype=dtype)
    expected[32:544] = values
    assert torch.equal(output.cpu(), expected)


@pytest.mark.parametrize("dtype", [
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
])
def test_metax_int4_pack_cast(dtype, metax_device):
    if dtype.startswith(("int", "uint")):
        info = np.iinfo(dtype)
        values = list(range(max(info.min, -256), min(info.max, 256) + 1))
        values += [int(info.min), int(info.min) + 1, int(info.max) - 1, int(info.max)]
        source = torch.from_numpy(np.array(values, dtype=dtype))
        codes = source.numpy().astype(np.uint64) & 15
    else:
        # The INT8 intermediate is defined for every truncated input here.
        values = np.concatenate((np.arange(-127, 128, dtype=np.float64), np.arange(-120, 120, dtype=np.float64) + 0.75,
                                 [-8.75, -8.5, -8.25, -0.75, -0., 0., 0.75, 7.25, 7.5, 7.75, 8.75]))
        source = torch.tensor(values, dtype=getattr(torch, dtype))
        codes = np.trunc(source.to(torch.float64).numpy()).astype(np.int64) & 15
    n = source.numel()
    if n % 2:
        codes = np.concatenate((codes, np.zeros(1, dtype=codes.dtype)))
    expected = (codes[::2] | (codes[1::2] << 4)).astype(np.uint8)
    output = torch.full((expected.size + 64, ), 165, dtype=torch.uint8, device=metax_device)
    _int4_pack[(triton.cdiv(expected.size, 128), )](source.to(metax_device), output[32:], n)
    reference = np.full(expected.size + 64, 165, dtype=np.uint8)
    reference[32:-32] = expected
    np.testing.assert_array_equal(output.cpu().numpy(), reference)
