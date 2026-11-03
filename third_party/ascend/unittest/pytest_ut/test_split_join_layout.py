"""Split/join layout regressions, including users that prevent load folding."""
import pytest
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _layout(X, Z, Y, AOUT, BOUT, TOUT, JOUT, A: tl.constexpr, B: tl.constexpr, XS: tl.constexpr, YS: tl.constexpr,
            SWAP: tl.constexpr):
    i = tl.arange(0, 2 * A * B)
    x = tl.load(X + i * XS)
    # Select a non-last axis in the original three dimensional tensor.
    t = tl.permute(tl.reshape(x, (A, 2, B)), (0, 2, 1))
    tl.store(TOUT + i, tl.reshape(t, (2 * A * B, )))
    a, b = tl.split(t)
    j = tl.arange(0, A * B)
    tl.store(AOUT + j, tl.reshape(a, (A * B, )))
    tl.store(BOUT + j, tl.reshape(b, (A * B, )))
    joined = tl.join(b, a) if SWAP else tl.join(a, b)
    tl.store(Y + i * YS, tl.reshape(joined, (2 * A * B, )))
    # An independent join, not eligible for join(split(x)) cancellation.
    z = tl.load(Z + tl.arange(0, A * B))
    other = tl.join(a, tl.reshape(z, (A, B)))
    tl.store(JOUT + i, tl.reshape(other, (2 * A * B, )))


@pytest.mark.parametrize("shape", [(1, 1), (2, 4), (4, 2), (4, 8), (8, 32), (1, 2), (2, 1)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int16, torch.int32, torch.int8])
@pytest.mark.parametrize("swap,strides", [(False, (1, 1)), (True, (3, 2))])
def test_split_join_layout(shape, dtype, swap, strides):
    a, b = shape
    xs, ys = strides
    n = 2 * a * b
    torch.manual_seed(123)
    # Random integers are exactly representable in all selected dtypes.
    storage = torch.randint(-100, 100, (n * xs + 2, ), dtype=torch.int32).to(dtype).npu()
    x = storage[1:1 + n * xs:xs]
    zcpu = torch.randint(-100, 100, (a * b, ), dtype=torch.int32).to(dtype)
    z = zcpu.npu()
    out_storage = torch.full((n * ys + 2, ), -111, dtype=dtype, device="npu")
    y = out_storage[1:1 + n * ys:ys]
    ao = torch.empty(a * b, dtype=dtype, device="npu")
    bo = torch.empty_like(ao)
    to = torch.empty(n, dtype=dtype, device="npu")
    jo = torch.empty_like(to)
    _layout[(1, )](x, z, y, ao, bo, to, jo, A=a, B=b, XS=xs, YS=ys, SWAP=swap)
    cpu = x.cpu().reshape(a, 2, b)
    expected_a, expected_b = cpu[:, 0, :], cpu[:, 1, :]
    t = cpu.permute(0, 2, 1).contiguous().reshape(-1)
    joined = torch.stack((expected_b, expected_a) if swap else (expected_a, expected_b), dim=-1).reshape(-1)
    other = torch.stack((expected_a, zcpu.reshape(a, b)), dim=-1).reshape(-1)
    for name, actual, expected in [("split0", ao, expected_a.reshape(-1)), ("split1", bo, expected_b.reshape(-1)),
                                   ("transpose", to, t), ("join", y, joined), ("independent_join", jo, other)]:
        assert torch.equal(actual.cpu(), expected), (name, actual.cpu(), expected)
    # Check all output gaps as well as both guard elements.
    guard = torch.ones(n * ys + 2, dtype=torch.bool)
    guard[1:1 + n * ys:ys] = False
    assert torch.all(out_storage.cpu()[guard] == -111)


@triton.jit
def _permute_bits(X, Y, SHAPE: tl.constexpr, ORDER: tl.constexpr, N: tl.constexpr):
    i = tl.arange(0, N)
    v = tl.reshape(tl.load(X + i), SHAPE)
    tl.store(Y + i, tl.reshape(tl.permute(v, ORDER), (N, )))


@pytest.mark.parametrize("shape,order", [
    ((4, 32), (1, 0)),
    ((32, 4), (1, 0)),
    ((2, 4, 16), (2, 0, 1)),
    ((4, 1, 32), (2, 1, 0)),
    ((1, 16, 1, 8), (2, 3, 0, 1)),
    ((32, 256), (1, 0)),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_permutation_bit_patterns(shape, order, dtype):
    # Include NaN payloads, infinities, signed zero and subnormal bit patterns.
    bits_dtype = torch.int32 if dtype == torch.float32 else torch.int16
    patterns = ([0, -2147483648, 0x7fc00001, -8388608, 0x7f800000, 1, -1, 12345]
                if dtype == torch.float32 else [0, -32768, 0x7fc1, -128, 0x7f80, 1, -1, 12345]
                if dtype == torch.bfloat16 else [0, -32768, 0x7e01, -1024, 0x7c00, 1, -1, 12345])
    n = 1
    for dim in shape:
        n *= dim
    raw = torch.tensor(patterns, dtype=bits_dtype).repeat(n // len(patterns))
    x = raw.view(dtype).npu()
    y = torch.empty_like(x)
    _permute_bits[(1, )](x, y, SHAPE=shape, ORDER=order, N=n)
    expected = raw.reshape(shape).permute(order).contiguous().reshape(-1)
    assert torch.equal(y.cpu().view(bits_dtype), expected)


@triton.jit
def _computed_split(X, Y, SIDE, N: tl.constexpr, XS: tl.constexpr, YS: tl.constexpr, VALID: tl.constexpr):
    i = tl.arange(0, 2 * N)
    v = tl.load(X + i * XS, i < VALID, other=0)
    v = (v + 1).to(X.dtype.element_ty)
    tl.store(SIDE + i, v)
    a, b = tl.split(tl.reshape(v, (N, 2)))
    j = tl.arange(0, N)
    tl.store(Y + j * YS, a)
    tl.store(Y + (N + j) * YS, b)


@pytest.mark.parametrize("n", [1, 8, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int8, torch.int16, torch.int32])
@pytest.mark.parametrize("xs,ys", [(1, 1), (3, 2)])
def test_computed_masked_split(n, dtype, xs, ys):
    valid = max(1, 2 * n - 3)
    x = torch.arange(valid * xs + 2, dtype=torch.int32).remainder(97).to(dtype).npu()[1:1 + valid * xs:xs]
    storage = torch.full((2 * n * ys + 2, ), -111, dtype=dtype, device="npu")
    y = storage[1:1 + 2 * n * ys:ys]
    side = torch.empty(2 * n, dtype=dtype, device="npu")
    _computed_split[(1, )](x, y, side, N=n, XS=xs, YS=ys, VALID=valid)
    ref = torch.zeros(2 * n, dtype=dtype)
    ref[:valid] = x.cpu()
    ref += 1
    expected = ref.reshape(n, 2).T.contiguous().reshape(-1)
    assert torch.equal(side.cpu(), ref)
    assert torch.equal(y.cpu(), expected)
    guard = torch.ones(2 * n * ys + 2, dtype=torch.bool)
    guard[1:1 + 2 * n * ys:ys] = False
    assert torch.all(storage.cpu()[guard] == -111)


@triton.jit
def _constant_split(Y, N: tl.constexpr):
    v = tl.full((N, 2), 3, Y.dtype.element_ty)
    a, b = tl.split(v)
    i = tl.arange(0, N)
    tl.store(Y + i, a)
    tl.store(Y + N + i, b)


@pytest.mark.parametrize("n", [1, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8])
def test_constant_split(n, dtype):
    y = torch.empty(2 * n, dtype=dtype, device="npu")
    _constant_split[(1, )](y, N=n)
    assert torch.all(y.cpu() == 3)
