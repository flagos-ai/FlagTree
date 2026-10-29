from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.gpu as gpu
import triton.experimental.tle.language.raw as raw
from triton.experimental.tle.raw import dialect

from triton._flagtree_backend import FLAGTREE_BACKEND

pytestmark = pytest.mark.skipif(FLAGTREE_BACKEND not in ("", "nvidia"), reason="requires the NVIDIA CUDA raw backend")


@pytest.fixture(autouse=True)
def fragment_environment():
    if torch.version.hip is not None:
        pytest.skip("requires the NVIDIA CUDA raw backend")
    if not torch.cuda.is_available():
        pytest.fail("CUDA fragment tests require an accessible NVIDIA GPU")
    if triton.runtime.driver.active.get_current_target().backend != "cuda":
        pytest.skip("requires the NVIDIA CUDA raw backend")
    with triton.knobs.autotuning.scope():
        triton.knobs.autotuning.adjust_block_size = False
        yield


SOURCE = Path(__file__).parent / "inputs" / "raw_fragment.cu"


def bind(symbol, **kwargs):

    @dialect(name="cuda", file=SOURCE, extern_func_name=symbol, **kwargs)
    def function(*args, **kw):
        ...

    return function


COPIES = {
    name: bind("Copy_" + name)
    for name in ("Bool", "I8", "U8", "I16", "U16", "I32", "U32", "I64", "U64", "F16", "BF16", "F32", "F64")
}
SCALARS = {name: bind("Scalar_" + name) for name in COPIES}
DOTS = {name: bind("Dot_" + name) for name in ("FP8", "FP16", "BF16")}
step = bind("Step")
masked_modify = bind("MaskedModify")
mixed_returns = bind("MixedReturns")
swap = bind("Swap")
stamp = bind("Stamp")
inspect_fragment = bind("InspectFragment")
store_only = bind("StoreOnly")
mixed_memory = bind("MixedMemory")
bool_byte = bind("BoolByte")

TYPES = [
    ("bool", "Bool", 1),
    ("int8", "I8", 8),
    ("uint8", "U8", 8),
    ("int16", "I16", 16),
    ("uint16", "U16", 16),
    ("int32", "I32", 32),
    ("uint32", "U32", 32),
    ("int64", "I64", 64),
    ("uint64", "U64", 64),
    ("float16", "F16", 16),
    ("bfloat16", "BF16", 16),
    ("float32", "F32", 32),
    ("float64", "F64", 64),
]


def storage_bits(name, width):
    if width == 1:
        return torch.tensor([False, True] * 128, device="cuda")
    if width == 8:
        values = list(range(256))
    else:
        values = [0, 1, (1 << (width - 1)), (1 << width) - 1, (1 << (width - 1)) - 1, (1 << (width - 1)) + 1]
        special = {
            "float16": [0x0400, 0x7bff, 0x7c00, 0xfc00, 0x7e55, 0x3555],
            "bfloat16": [0x0080, 0x7f7f, 0x7f80, 0xff80, 0x7fc5, 0x3eab],
            "float32": [0x00800000, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc01234, 0x3eaaaaab],
            "float64": [
                0x0010000000000000, 0x7fefffffffffffff, 0x7ff0000000000000, 0xfff0000000000000, 0x7ff8123456789abc,
                0x3fd5555555555555
            ],
        }
        values += special.get(name, [])
        values = (values * (256 // len(values) + 1))[:256]
    signed = [v if v < (1 << (width - 1)) else v - (1 << width) for v in values]
    return torch.tensor(signed, device="cuda", dtype=getattr(torch, f"int{width}"))


PARAMS = [pytest.param(name, symbol, width, id=name) for name, symbol, width in TYPES]
BOUNDARIES = [1, 31, 32, 33, 127, 128, 129, 1023, 1024, 1025, 4095, 4096, 4097, 16384]


@pytest.mark.parametrize("name,symbol,width", PARAMS)
@pytest.mark.parametrize("scalar_index", [None, 0, 2, 3],
                         ids=["fragment", "scalar_zero", "scalar_sign", "scalar_allbits"])
def test_bit_exact_abi(name, symbol, width, scalar_index):
    target = tl.int1 if name == "bool" else getattr(tl, name)
    x = storage_bits(name, width)
    original = x.clone()
    # Both ends are guarded; compare original integer storage, including NaN payloads.
    backing = torch.full((x.numel() + 32, ), 1, dtype=x.dtype, device="cuda")
    y = backing[16:-16]
    index = scalar_index or 0
    if width == 8 and scalar_index in (2, 3):
        index = 128 if scalar_index == 2 else 255
    scalar = x[index:index + 1]
    function = SCALARS[symbol] if scalar_index is not None else COPIES[symbol]
    abi_kernel[(1, )](triton.reinterpret(x, target), triton.reinterpret(y, target), triton.reinterpret(scalar, target),
                      x.numel(), 256, function, SCALAR=scalar_index is not None, num_warps=4)
    torch.testing.assert_close(y, scalar.expand_as(y) if scalar_index is not None else x, rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    assert torch.all(backing[:16] == 1) and torch.all(backing[-16:] == 1)


@triton.jit
def abi_kernel(X, Y, S, N: tl.constexpr, BLOCK: tl.constexpr, FN: tl.constexpr, SCALAR: tl.constexpr = False):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # ABI cases use full tiles and compare raw storage, including NaN payloads.
    x = tl.load(X + offsets)
    dst = tl.full((BLOCK, ), 0, x.dtype)
    if SCALAR:
        result = raw.call(FN, (raw.fragment(dst), tl.load(S)), output_indices=[0])
    else:
        result = raw.call(FN, (raw.fragment(dst), raw.fragment(x)), output_indices=[0])
    tl.store(Y + offsets, result, offsets < N)


@pytest.mark.parametrize("flag", [False, True])
def test_bool_bytes_are_zero_or_one(flag):
    x = torch.tensor([False, True] * 64, device="cuda")
    y = torch.empty(128, dtype=torch.int32, device="cuda")
    bool_kernel[(1, )](x, y, flag, 128, num_warps=4)
    torch.testing.assert_close(y, x.int() + 10 * int(flag), rtol=0, atol=0)


@triton.jit
def bool_kernel(X, Y, flag, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    dst = tl.full((BLOCK, ), 0, tl.int32)
    result = raw.call(bool_byte, (raw.fragment(dst), raw.fragment(x), flag.to(tl.int1)), output_indices=[0])
    tl.store(Y + offsets, result)


@pytest.mark.parametrize("dtype,symbol", [
    pytest.param(torch.float8_e4m3fn, "FP8", id="fp8"),
    pytest.param(torch.float16, "FP16", id="fp16"),
    pytest.param(torch.bfloat16, "BF16", id="bf16"),
])
@pytest.mark.parametrize("num_warps", [1, 2, 8])
@pytest.mark.parametrize("M,N", [(4, 5), (5, 4)])
@pytest.mark.parametrize("K", BOUNDARIES)
def test_float_matmul(dtype, symbol, M, N, K, num_warps):
    gen = torch.Generator().manual_seed(20261024)
    host_a = torch.randint(-16, 16, (M, K), generator=gen).float() / 4
    host_b = torch.randint(-16, 16, (K, N), generator=gen).float() / 4
    a = host_a.to(device="cuda", dtype=dtype)
    b = host_b.to(device="cuda", dtype=dtype)
    buffer = torch.full((M + 2, N + 2), -12345, device="cuda", dtype=torch.float32)
    c = buffer[1:-1, 1:-1]
    expected = buffer.clone()
    expected[1:-1, 1:-1] = a.float().cpu() @ b.float().cpu()
    BLOCK_SIZE = max(triton.next_power_of_2(K), num_warps * 32)
    LAYOUT: tl.constexpr = gpu.BlockEncoding(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[num_warps],
                                             order=[0])
    float_matmul_kernel[(M, N)](a, a.stride(0), a.stride(1), b, b.stride(0), b.stride(1), c, c.stride(0), c.stride(1),
                                K, BLOCK_SIZE, LAYOUT, DOTS[symbol], num_warps=num_warps)
    torch.testing.assert_close(buffer, expected, rtol=0, atol=0)


@triton.jit
def float_matmul_kernel(
    a_ptr,
    a_row_stride,
    a_col_stride,
    b_ptr,
    b_row_stride,
    b_col_stride,
    c_ptr,
    c_row_stride,
    c_col_stride,
    K,
    BLOCK_SIZE: tl.constexpr,
    LAYOUT: tl.constexpr,
    DOT: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    ks = tl.arange(0, BLOCK_SIZE)
    mask = ks < K
    a_row = tl.load(a_ptr + row * a_row_stride + ks * a_col_stride, mask=mask, other=0.0)
    b_col = tl.load(b_ptr + ks * b_row_stride + col * b_col_stride, mask=mask, other=0.0)
    out = c_ptr + row * c_row_stride + col * c_col_stride
    a_row = gpu.set_layout(a_row, LAYOUT)
    b_col = gpu.set_layout(b_col, LAYOUT)
    raw.call(DOT, [raw.fragment(a_row), raw.fragment(b_col), out], output_indices=[])


def test_wrapper_specializations_do_not_cross_contaminate():
    # One launch contains five raw calls, with 1, 4, 2, 4, 1 elements/thread.
    lengths = (128, 512, 256, 512, 128)
    host = torch.arange(1, 5 * 512 + 1, dtype=torch.int32).reshape(5, 512)
    x = host.cuda()
    y = torch.full_like(x, -1)
    layout = gpu.BlockEncoding([1], [32], [4], [0])

    wrapper_specializations_kernel[(1, )](x, y, layout, num_warps=4)

    # Each call has separate input/output storage; unused output stays guarded.
    expected = torch.full_like(host, -1)
    for call, n in enumerate(lengths):
        columns = torch.arange(n, dtype=torch.int32)
        expected[call, :n] = host[call, :n] + (columns % 128) * 10000 + (columns // 128) * 1000000
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(x.cpu(), host, rtol=0, atol=0)


@triton.jit
def wrapper_specializations_kernel(X, Y, LAYOUT: tl.constexpr):
    lengths: tl.constexpr = (128, 512, 256, 512, 128)
    # Static unrolling creates five call sites in the same compiled kernel.
    for call in tl.static_range(5):
        columns = tl.arange(0, lengths[call])
        offsets = call * 512 + columns
        x = gpu.set_layout(tl.load(X + offsets), LAYOUT)
        result = raw.call(stamp, (raw.fragment(x), ), output_indices=[0])
        tl.store(Y + offsets, result)


@pytest.mark.parametrize("warps", [1, 2, 4, 8])
@pytest.mark.parametrize("spt", [1, 2, 4, 8])
def test_1d_element_ownership(warps, spt):
    threads = warps * 32
    n = threads * spt * 2
    x = torch.arange(n, dtype=torch.int32, device="cuda")
    y = torch.empty_like(x)
    layout = gpu.BlockEncoding([spt], [32], [warps], [0])
    stamp_1d_kernel[(1, )](x, y, n, layout, num_warps=warps)
    expected = torch.empty(n, dtype=torch.int32)
    # Independent blocked-layout oracle: contiguous per-thread groups, then tiles.
    for thread in range(threads):
        for item in range(2 * spt):
            col = (item // spt) * threads * spt + thread * spt + item % spt
            expected[col] = col + thread * 10000 + item * 1000000
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)


@triton.jit
def stamp_1d_kernel(X, Y, BLOCK: tl.constexpr, LAYOUT: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = gpu.set_layout(tl.load(X + offsets), LAYOUT)
    result = raw.call(stamp, (raw.fragment(x), ), output_indices=[0])
    tl.store(Y + offsets, result)


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("distribution", [0, 1])
def test_2d_element_ownership(order, distribution):
    spt, tpw, wpc = ([1, 2], [4, 8], [1, 4]) if distribution == 0 else ([2, 1], [8, 4], [2, 2])
    shape = [spt[d] * tpw[d] * wpc[d] for d in range(2)]
    m, n = shape
    x = torch.arange(m * n, dtype=torch.int32, device="cuda")
    y = torch.empty_like(x)
    layout = gpu.BlockEncoding(spt, tpw, wpc, order)
    compiled = stamp_2d_kernel[(1, )](x, y, m, n, layout, num_warps=4)
    expected = torch.empty((m, n), dtype=torch.int32)
    for thread in range(128):
        for item in range(spt[0] * spt[1]):
            lane, warp, local = thread % 32, thread // 32, item
            coords = [0, 0]
            for dim in order:
                coords[dim] = local % spt[dim] + spt[dim] * (lane % tpw[dim] + tpw[dim] * (warp % wpc[dim]))
                local //= spt[dim]
                lane //= tpw[dim]
                warp //= wpc[dim]
            row, col = coords
            expected[row, col] = row * n + col + thread * 10000 + item * 1000000
    torch.testing.assert_close(y.cpu().reshape(m, n), expected, rtol=0, atol=0)
    assert "#ttg.blocked" in compiled.asm["ttgir"]


@triton.jit
def stamp_2d_kernel(X, Y, M: tl.constexpr, N: tl.constexpr, LAYOUT: tl.constexpr):
    offsets = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    x = gpu.set_layout(tl.load(X + offsets), LAYOUT)
    result = raw.call(stamp, (raw.fragment(x), ), output_indices=[0])
    tl.store(Y + offsets, result)


def test_sliced_layout():
    parent = gpu.BlockEncoding([1, 1], [4, 8], [1, 4], [1, 0])
    sliced = gpu.SlicedEncoding(0, parent)
    host = torch.arange(128, dtype=torch.int32) * 17 + 3
    x = host.cuda()
    observed = torch.full((128, 4), -1, device="cuda", dtype=torch.int32)
    lengths = torch.full((128, ), -1, device="cuda", dtype=torch.int32)
    inspect_fragment_kernel[(1, )](x, observed, lengths, 128, 4, sliced, num_warps=4)

    # Drop the parent's row coordinate. Lanes differing only in lane // 8
    # must receive identical fragments, in the same item order.
    expected = torch.empty((128, 4), dtype=torch.int32)
    for thread in range(128):
        warp, lane = thread // 32, thread % 32
        for item in range(4):
            col = item * 32 + warp * 8 + lane % 8
            expected[thread, item] = host[col]
    torch.testing.assert_close(observed.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(lengths.cpu(), torch.full((128, ), 4, dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(x.cpu(), host, rtol=0, atol=0)


@triton.jit
def inspect_fragment_kernel(X, Observed, Lengths, BLOCK: tl.constexpr, CAPACITY: tl.constexpr, LAYOUT: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = gpu.set_layout(tl.load(X + offsets), LAYOUT)
    # CUDA writes every physical thread's fragment to its own output row,
    # avoiding the deduplication of replicated elements by a Triton store.
    raw.call(inspect_fragment, (raw.fragment(x), Observed, Lengths, CAPACITY), output_indices=[])


@pytest.mark.parametrize("warps", [1, 2, 4, 8])
def test_implicit_layout(warps):
    n = 128 + 1
    x = torch.arange(3 * n, device="cuda", dtype=torch.float32).reshape(3, n) / 16
    y = torch.full_like(x, float("nan"))
    original = x.clone()
    expected = x + 0.5
    elementwise_kernel[(3, )](x, y, n, triton.next_power_of_2(n), None, num_warps=warps)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@triton.jit
def elementwise_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, LAYOUT: tl.constexpr):
    col = tl.arange(0, BLOCK)
    row = tl.program_id(0)
    x = tl.load(X + row * N + col, col < N, other=0)
    if LAYOUT is not None:
        x = gpu.set_layout(x, LAYOUT)
    result = raw.call(step, (raw.fragment(x), 0.5, True), output_indices=[0])
    tl.store(Y + row * N + col, result, col < N)


@pytest.mark.parametrize("n", BOUNDARIES, ids=lambda n: f"n{n}")
@pytest.mark.parametrize("mask_pattern", ["tail", "holes"])
@pytest.mark.parametrize("spt", [1, 2, 4, 8], ids=lambda spt: f"spt{spt}")
def test_boundaries_and_mask(n, mask_pattern, spt):
    # Keep at least one complete per-thread group.
    block = max(triton.next_power_of_2(n), spt)
    x = torch.arange(3 * block, device="cuda", dtype=torch.float32).reshape(3, block) / 16
    columns = torch.arange(block, device="cuda")[None, :]
    rows = torch.arange(3, device="cuda")[:, None]
    mask = (columns < n).expand(3, block).clone()
    if mask_pattern == "holes":
        mask &= (columns + rows) % 3 != 0
    original_x, original_mask = x.clone(), mask.clone()
    backing = torch.full((3 * block + 32, ), -12345.0, device="cuda")
    y = backing[16:-16].reshape(3, block)
    expected = torch.where(mask, x * 2 + 0.5, x)
    layout = gpu.BlockEncoding([spt], [32], [4], [0])

    masked_fragment_kernel[(3, )](x, mask, y, block, layout, num_warps=4)

    # Compare the entire tile, including masked-off elements.
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(mask, original_mask, rtol=0, atol=0)
    assert torch.all(backing[:16] == -12345) and torch.all(backing[-16:] == -12345)


@triton.jit
def masked_fragment_kernel(X, Mask, Y, BLOCK: tl.constexpr, LAYOUT: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = gpu.set_layout(tl.load(X + offsets), LAYOUT)
    mask = gpu.set_layout(tl.load(Mask + offsets), LAYOUT)
    result = raw.call(masked_modify, (raw.fragment(x), raw.fragment(mask)), output_indices=[0])
    tl.store(Y + offsets, result)


@pytest.mark.parametrize("mode,iterations", [
    pytest.param(mode, n, id=f"{n}-{mode}")
    for n in (0, 1, 2, 7)
    for mode in ("static", "range", "tl_range", "nested", "named_fragment", "multi")
])
def test_loop_carries(mode, iterations):
    x = torch.arange(256, device="cuda", dtype=torch.float32) / 16
    original = x.clone()
    y, z = torch.empty_like(x), torch.empty_like(x)
    layout = gpu.BlockEncoding([2], [32], [4], [0])
    compiled = loop_kernel[(1, )](x, y, z, iterations, 256, layout, mode, iterations, num_warps=4)
    if mode == "multi":
        ey, ez = original.cpu(), original.cpu() + 10
        delta = 1.0
        for _ in range(iterations):
            delta /= 2
            ey = ey + delta
            ez = ez - delta
            ey, ez = ez, ey
        ey, ez = ey.to(x.device), ez.to(x.device)
    else:
        increment = iterations * 0.5
        if mode == "nested":
            increment = iterations * iterations * 0.5
        ey, ez = x + increment, x + 10
    torch.testing.assert_close(y, ey, rtol=0, atol=0)
    torch.testing.assert_close(z, ez, rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    if mode != "static" and iterations in (2, 7):
        assert "scf.for" in compiled.asm["ttir"], "runtime loop was unexpectedly specialized away"


@triton.jit
def loop_kernel(X, Y, Z, iterations, BLOCK: tl.constexpr, LAYOUT: tl.constexpr, MODE: tl.constexpr,
                STATIC_N: tl.constexpr = 0):
    offsets = tl.arange(0, BLOCK)
    x = gpu.set_layout(tl.load(X + offsets), LAYOUT)
    other = x + 10.0
    if MODE == "static":
        for i in tl.static_range(STATIC_N):
            x = raw.call(step, (raw.fragment(x), 0.5, True), output_indices=[0])
    elif MODE == "range":
        for i in range(iterations):
            x = raw.call(step, (raw.fragment(x), 0.5, True), output_indices=[0])
    elif MODE == "tl_range":
        for i in tl.range(iterations):
            x = raw.call(step, (raw.fragment(x), 0.5, True), output_indices=[0])
    elif MODE == "nested":
        for i in range(iterations):
            for j in range(iterations):
                x = raw.call(step, (raw.fragment(x), 0.5, True), output_indices=[0])
    elif MODE == "named_fragment":
        fragment = raw.fragment(x)
        for i in range(iterations):
            x = raw.call(step, (fragment, 0.5, True), output_indices=[0])
            fragment = raw.fragment(x)
        x = fragment.full_tensor
    else:
        delta = 1.0
        for i in range(iterations):
            delta = delta / 2
            x = raw.call(step, (raw.fragment(x), delta, True), output_indices=[0])
            other = raw.call(step, (raw.fragment(other), delta, False), output_indices=[0])
            other, x = raw.call(swap, (raw.fragment(x), raw.fragment(other)), output_indices=[1, 0])
    tl.store(Y + offsets, x)
    tl.store(Z + offsets, other)


@pytest.mark.parametrize("iterations", [0, 1, 2, 7, 15, 16])
def test_2d_layout_dynamic_loop_permutation(iterations):
    host = torch.arange(256, dtype=torch.int32) * 17 + 3
    x = host.cuda()
    y = torch.full_like(x, -1)
    compiled = loop_permute_2d_kernel[(1, )](x, y, iterations, COPIES["I32"], num_warps=4)

    # Each layout covers (16, 16) exactly once, with two items per thread.
    # CUDA copies the same physical (warp, lane, item) from A to B.
    source_for_destination = torch.full((256, ), -1, dtype=torch.int64)
    for warp in range(4):
        for lane in range(32):
            for item in range(2):
                source_row, source_col = 4 * warp + lane // 8, 2 * (lane % 8) + item
                destination_row, destination_col = 2 * (lane // 4) + item, 4 * warp + lane % 4
                source_for_destination[destination_row * 16 + destination_col] = source_row * 16 + source_col
    torch.testing.assert_close(source_for_destination.sort().values, torch.arange(256), rtol=0, atol=0)
    expected = host.clone()
    for _ in range(iterations):
        expected = expected[source_for_destination]
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(x.cpu(), host, rtol=0, atol=0)
    if iterations > 1:
        assert "scf.for" in compiled.asm["ttir"], "runtime loop was unexpectedly specialized away"


@triton.jit
def loop_permute_2d_kernel(X, Y, iterations, copy_fn: tl.constexpr):
    source_layout: tl.constexpr = gpu.BlockEncoding([1, 2], [4, 8], [4, 1], [1, 0])
    destination_layout: tl.constexpr = gpu.BlockEncoding([2, 1], [8, 4], [1, 4], [1, 0])
    offsets = tl.arange(0, 256).reshape(16, 16)
    x = gpu.set_layout(tl.load(X + offsets), source_layout)
    for i in range(iterations):
        destination = gpu.set_layout(tl.full((16, 16), -1, tl.int32), destination_layout)
        # Achieve logical permutation by copying the physical storage.
        result = raw.call(copy_fn, (raw.fragment(destination), raw.fragment(x)), output_indices=[0])
        # Convert back to the source layout. Get a permuted x with the original layout.
        x = gpu.set_layout(result, source_layout)
    tl.store(Y + offsets, x)


@pytest.mark.parametrize("mode", ["ordered", "reverse", "duplicate"])
@pytest.mark.parametrize("sizes", [(128, 128, 128), (256, 128, 512)], ids=["equal_lengths", "unequal_lengths"])
def test_heterogeneous_results(mode, sizes):
    nx, ny, nm = sizes
    x = torch.arange(nx, device="cuda", dtype=torch.float32)
    y = torch.arange(ny, device="cuda", dtype=torch.int32)
    mask = torch.arange(nm, device="cuda") % 2 == 0
    originals = [v.clone() for v in (x, y, mask)]
    a = torch.empty_like(x)
    b = torch.empty_like(mask) if mode == "duplicate" else torch.empty_like(y)
    c = torch.empty_like(mask)
    returns_kernel[(1, )](x, y, mask, a, b, c, nx, ny, nm, mode, num_warps=4)
    torch.testing.assert_close(a, x + 0.5, rtol=0, atol=0)
    torch.testing.assert_close(c, ~mask, rtol=0, atol=0)
    expected_b = ~mask if mode == "duplicate" else (y * 2 + 3 if mode == "ordered" else y)
    torch.testing.assert_close(b, expected_b, rtol=0, atol=0)
    for current, original in zip((x, y, mask), originals):
        torch.testing.assert_close(current, original, rtol=0, atol=0)


@triton.jit
def returns_kernel(X, Y, Mask, A, B, C, BX: tl.constexpr, BY: tl.constexpr, BM: tl.constexpr, MODE: tl.constexpr):
    ix = tl.arange(0, BX)
    iy = tl.arange(0, BY)
    im = tl.arange(0, BM)
    x, y, mask = tl.load(X + ix), tl.load(Y + iy), tl.load(Mask + im)
    if MODE == "ordered":
        a, b, c = raw.call(mixed_returns, (raw.fragment(x), raw.fragment(y), raw.fragment(mask)),
                           output_indices=[0, 1, 2])
        tl.store(A + ix, a)
        tl.store(B + iy, b)
        tl.store(C + im, c)
    elif MODE == "reverse":
        c, a = raw.call(mixed_returns, (raw.fragment(x), raw.fragment(y), raw.fragment(mask)), output_indices=[2, 0])
        tl.store(A + ix, a)
        tl.store(C + im, c)
        tl.store(B + iy, y)
    else:
        c, a, again = raw.call(mixed_returns, (raw.fragment(x), raw.fragment(y), raw.fragment(mask)),
                               output_indices=[2, 0, 2])
        tl.store(A + ix, a)
        tl.store(C + im, c)
        tl.store(B + im, again)


def test_homogeneous_reordered_returns():
    x = torch.arange(128, device="cuda", dtype=torch.float32)
    y = x + 100
    a, b = torch.empty_like(x), torch.empty_like(x)
    swap_kernel[(1, )](x, y, a, b, 128, num_warps=4)
    torch.testing.assert_close(a, y, rtol=0, atol=0)
    torch.testing.assert_close(b, x, rtol=0, atol=0)


@triton.jit
def swap_kernel(X, Y, A, B, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x, y = tl.load(X + offsets), tl.load(Y + offsets)
    b, a = raw.call(swap, (raw.fragment(x), raw.fragment(y)), output_indices=[1, 0])
    tl.store(A + offsets, a)
    tl.store(B + offsets, b)


def test_no_return_with_global_pointer_side_effect():
    x = torch.arange(512, dtype=torch.int32, device="cuda")
    backing = torch.full((544, ), -12345, dtype=torch.int32, device="cuda")
    y = backing[16:-16]
    layout = gpu.BlockEncoding([1], [32], [4], [0])
    side_effect_kernel[(1, )](x, y, 512, layout, num_warps=4)
    torch.testing.assert_close(y, x + 7, rtol=0, atol=0)
    assert torch.all(backing[:16] == -12345) and torch.all(backing[-16:] == -12345)


@triton.jit
def side_effect_kernel(X, Y, BLOCK: tl.constexpr, LAYOUT: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = gpu.set_layout(tl.load(X + offsets), LAYOUT)
    raw.call(store_only, (raw.fragment(x), Y), output_indices=[])


@pytest.mark.parametrize("flag", [True, False])
def test_fragment_tensor_buffer_scalars_and_pointer(flag):
    x = torch.arange(128, device="cuda", dtype=torch.float32)
    y = x * 0.25
    a, b, c, side = [torch.empty_like(x) for _ in range(4)]
    layout = gpu.BlockEncoding([1], [32], [4], [0])
    memory_kernel[(1, )](x, y, a, b, c, side, flag, layout, num_warps=4)
    expected = (x + y * 2 + 3) * (1 if flag else -1)
    for output, offset in ((a, 0), (b, 1), (c, 2), (side, 3)):
        torch.testing.assert_close(output, expected + offset, rtol=0, atol=0)
    torch.testing.assert_close(x, torch.arange(128, device="cuda", dtype=torch.float32), rtol=0, atol=0)
    torch.testing.assert_close(y, x * 0.25, rtol=0, atol=0)


@triton.jit
def memory_kernel(X, Y, A, B, C, Side, flag, LAYOUT: tl.constexpr):
    offsets = tl.arange(0, 128)
    x = gpu.set_layout(tl.load(X + offsets), LAYOUT)
    y = tl.load(Y + offsets)
    buf = gpu.alloc([4, 32], tl.float32, layout=None, scope=gpu.smem, nv_mma_shared_layout=False)
    buf, a, b = raw.call(mixed_memory, (raw.fragment(x), y, buf, 2.0, flag.to(tl.int1), 3, Side),
                         output_indices=[2, 0, 1])
    tl.store(A + offsets, a)
    tl.store(B + offsets, b)
    r = tl.arange(0, 4)[:, None]
    c = tl.arange(0, 32)[None, :]
    values = tl.load(gpu.local_ptr(buf, (tl.broadcast_to(r, (4, 32)), tl.broadcast_to(c, (4, 32)))))
    tl.store(C + r * 32 + c, values)
