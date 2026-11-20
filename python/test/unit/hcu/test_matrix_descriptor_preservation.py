"""Block-pointer descriptor preservation and exact-index fallback contracts."""
import json
import re
import pytest
from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)

from triton.backends.hcu.compiler import HIPBackend
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler import ASTSource


def rewrite(tmp_path, body, enabled=True):
    attr = ' attributes {ttg.hcu.prefer_wide_async_copy = 1 : i32}' if enabled else ''
    path = tmp_path / "descriptor.mlir"
    path.write_text("module" + attr + " {\n" + body + "\n}")
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    manager = libtriton.ir.pass_manager(context)
    libtriton.passes.ttir.add_rewrite_tensor_pointer(manager)
    manager.run(module, "matrix_descriptor_preservation")
    return str(module)


@pytest.mark.parametrize("enabled,async_,volatile,padding,wide_shape,expected", [
    (True, True, False, 1, False, True),
    (True, True, False, None, False, True),
    (False, True, False, 1, False, False),
    (True, False, False, 1, False, True),
    (True, True, True, 1, False, False),
    (True, True, False, 2, False, False),
    (True, True, False, 1, True, False),
])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_descriptor_preserves_semantics_or_falls_back(tmp_path, enabled, async_, volatile, padding, wide_shape,
                                                      expected, dtype):
    shape = "%wide" if wide_shape else "%n64"
    pad = f", padding = {padding} : i32" if padding is not None else ""
    text = f'''tt.func public @copy(%p: !tt.ptr<{dtype}>, %n: i32,
      %wide: i64, %stride: i64, %offset: i32) -> tensor<16x16x{dtype}> {{
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i64
      %n64 = arith.extsi %n : i32 to i64
      %bp = tt.make_tensor_ptr %p, [%n64, {shape}], [%stride, %one],
        [%offset, %zero] {{order = array<i32: 1, 0>}}
        : !tt.ptr<tensor<16x16x{dtype}>>
      %x = tt.load %bp {{boundaryCheck = array<i32: 0, 1>{pad},
        tt.load.async = {str(async_).lower()}, isVolatile = {str(volatile).lower()}}}
        : !tt.ptr<tensor<16x16x{dtype}>>
      tt.return %x : tensor<16x16x{dtype}>
    }}'''
    result = rewrite(tmp_path, text, enabled)
    assert ("tt.matrix_load" in result) == expected
    assert "tt.make_tensor_ptr" not in result
    if expected:
        assert re.search(r"boundaryCheck = (?:array<i32: 0, 1>|\[0, 1\])", result)
        assert "tt.load.async = " + str(async_).lower() in result
        assert "tt.addptr" not in result
    else:
        assert "tt.load " in result and "tt.addptr" in result


@pytest.mark.parametrize("advance,initial,expected", [
    (16, 7, True),
    (8, 7, False),
    (16, 0, False),
])
def test_loop_descriptor_requires_exact_induction_recurrence(tmp_path, advance, initial, expected):
    text = f'''tt.func public @loop(%p: !tt.ptr<f16>, %n: i32,
      %stride: i64, %row: i32) -> tensor<16x16xf16> {{
      %zero = arith.constant 0 : i32
      %lower = arith.constant 7 : i32
      %initial = arith.constant {initial} : i32
      %step = arith.constant 16 : i32
      %delta = arith.constant {advance} : i32
      %one = arith.constant 1 : i64
      %n64 = arith.extsi %n : i32 to i64
      %empty = arith.constant dense<0.0> : tensor<16x16xf16>
      %bp = tt.make_tensor_ptr %p, [%n64, %n64], [%stride, %one],
        [%row, %initial] {{order = array<i32: 1, 0>}}
        : !tt.ptr<tensor<16x16xf16>>
      %result:2 = scf.for %i = %lower to %n step %step
        iter_args(%q = %bp, %acc = %empty)
        -> (!tt.ptr<tensor<16x16xf16>>, tensor<16x16xf16>) : i32 {{
        %x = tt.load %q {{boundaryCheck = array<i32: 0, 1>,
          padding = 1 : i32, tt.load.async = true}}
          : !tt.ptr<tensor<16x16xf16>>
        %next = tt.advance %q, [%zero, %delta]
          : !tt.ptr<tensor<16x16xf16>>
        %sum = arith.addf %acc, %x : tensor<16x16xf16>
        scf.yield %next, %sum
          : !tt.ptr<tensor<16x16xf16>>, tensor<16x16xf16>
      }}
      tt.return %result#1 : tensor<16x16xf16>
    }}'''
    result = rewrite(tmp_path, text)
    assert ("tt.matrix_load" in result) == expected
    assert "tt.make_tensor_ptr" not in result and "tt.advance" not in result


@triton.jit
def _unpipelined_descriptor_dot(A, B, C, BK: tl.constexpr, ASYNC: tl.constexpr, COL, A_PITCH: tl.constexpr):
    a = tl.make_block_ptr(A, (1, BK), (A_PITCH, 1), (0, 0), (128, BK), (1, 0))
    b = tl.make_block_ptr(B, (BK, 16), (16, 1), (0, COL), (BK, 128), (1, 0))
    av = tle.load(a, boundary_check=(0, 1), padding_option="zero", is_async=ASYNC)
    bv = tle.load(b, boundary_check=(0, 1), padding_option="zero", is_async=ASYNC)
    value = tl.dot(av, bv)
    m = tl.arange(0, 128)
    n = tl.arange(0, 128)
    tl.store(C + m[:, None] * 16 + n[None, :], value, (m[:, None] < 1) & (n[None, :] < 16))


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("block_k", [16, 32])
@pytest.mark.parametrize("kpack", [1, 2])
@pytest.mark.parametrize("async_", [False, True])
def test_unpipelined_descriptor_is_lowered_before_pointer_canonicalization(dtype, block_k, kpack, async_):
    source = ASTSource(
        _unpipelined_descriptor_dot,
        {
            "A": "*" + dtype, "B": "*" + dtype, "C": "*" + dtype, "BK": "constexpr", "ASYNC": "constexpr", "COL":
            "constexpr"
        },
        {"BK": block_k, "ASYNC": async_, "COL": 0, "A_PITCH": block_k},
        attrs={(i, ): [("tt.divisibility", 16), ("tt.pointer_range", 32)]
               for i in range(3)},
    )
    kernel = triton.compile(
        source,
        target=GPUTarget("hip", "gfx936", 64),
        options=dict(prefer_wide_async_copy=True, num_warps=4, num_stages=4, kpack=kpack,
                     schedule_hint="local-prefetch", mmac_warps_m=2, mmac_transposed=True, optimize_epilogue=True),
    )
    if async_:
        assert "tt.matrix_load " not in kernel.asm["ttgir"]
        assert "hcug.matrix_load_to_local" in kernel.asm["ttgir"]
        assert re.search(r"ttg.local_load[^\n]* token ", kernel.asm["ttgir"])
        assert "!noalias" in kernel.asm["llir"]
        assert '\"dereferenceable\"(ptr addrspace(3)' in kernel.asm["llir"]
    else:
        assert "hcu.structured_register" in kernel.asm["ttgir"]
        assert "llvm.amdgcn.struct.ptr.buffer.load." in kernel.asm["llir"]

    # The descriptor is not a fixed-width native MLS instruction. Honor the
    # ordinary dot option, while K16 still clamps the illegal packed K group.
    expected = 8 if kpack == 2 and block_k >= 32 else 4
    assert set(re.findall(r"kWidth = (\d+)", kernel.asm["ttgir"])) == {str(expected)}


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("pitch,structured", [(32, True), (40, True), (64, True), (8191, True), (8192, False),
                                              (16, False)])
def test_register_descriptor_uses_physical_row_stride(dtype, pitch, structured):
    source = ASTSource(
        _unpipelined_descriptor_dot,
        {"A": "*" + dtype, "B": "*" + dtype, "C": "*" + dtype},
        {"BK": 32, "ASYNC": False, "COL": 0, "A_PITCH": pitch},
        attrs={(i, ): [("tt.divisibility", 16), ("tt.pointer_range", 32)]
               for i in range(3)},
    )
    kernel = triton.compile(
        source,
        target=GPUTarget("hip", "gfx936", 64),
        options=dict(prefer_wide_async_copy=True, num_warps=4, num_stages=1, kpack=2, mmac_warps_m=2,
                     mmac_transposed=True),
    )
    assert ("hcu.structured_register" in kernel.asm["ttgir"]) == structured
    assert ("llvm.amdgcn.struct.ptr.buffer.load." in kernel.asm["llir"]) == structured
    if structured:
        # The descriptor's physical stride is independent of its logical K.
        assert re.search(r"i16 " + str(0x4000 | (2 * pitch)) + r"\b", kernel.asm["llir"])
        # Row rebasing already clamps to [0, M]. Repeating an opaque scalar
        # saturation here adds two instructions and hides constant folding.
        assert 'asm "s_sub_u32 $0, $1, $2' not in kernel.asm["llir"]


@triton.jit(do_not_specialize=["M", "ROW", "COL"])
def _register_descriptor_copy(A, C, M, ROW, COL, K: tl.constexpr, PITCH: tl.constexpr):
    block = tl.make_block_ptr(A, (M, K), (PITCH, 1), (ROW, COL * 32), (16, 32), (1, 0))
    value = tle.load(block, boundary_check=(0, 1), padding_option="zero", is_async=False)
    rows = tl.arange(0, 16)
    columns = tl.arange(0, 32)
    tl.store(C + rows[:, None] * 32 + columns[None, :], value)


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("pitch", [64, 72, 8191, 8192])
def test_register_descriptor_padded_rows_and_zero_boundaries(dtype, pitch):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    dtype = getattr(torch, {"bf16": "bfloat16", "fp16": "float16"}[dtype])
    # Poison the padding: using logical K as the row stride reads NaNs.
    allocation = torch.full((19 * pitch + 32, ), float("nan"), device="cuda", dtype=dtype)
    rows = allocation[16:-16].view(19, pitch)
    rows[:, :64] = torch.randn((19, 64), device="cuda", dtype=dtype)
    before = allocation.clone()
    output = torch.full((16 * 32 + 32, ), -123, device="cuda", dtype=dtype)
    for row, column in [(0, 0), (1, 32), (16, 32), (17, 0), (18, 32), (32, 0), (-16, 0), (-1, 0), (0, -32), (0, 64)]:
        kernel = _register_descriptor_copy[(1, )](
            rows,
            output[16:-16],
            19,
            row,
            column // 32,
            64,
            pitch,
            prefer_wide_async_copy=True,
            num_warps=4,
            num_stages=1,
        )
        assert ("hcu.structured_register" in kernel.asm["ttgir"]) == (pitch < 8192)
        expected = torch.zeros((16, 32), device="cuda", dtype=dtype)
        start, end = max(0, -row), min(16, 19 - row)
        if start < end and 0 <= column < 64:
            expected[start:end] = rows[row + start:row + end, column:column + 32]
        torch.testing.assert_close(output[16:-16].view(16, 32), expected, rtol=0, atol=0)
        assert torch.all(output[:16] == -123) and torch.all(output[-16:] == -123)
    torch.testing.assert_close(allocation, before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("pitch", [64, 72])
def test_register_descriptor_clamped_row_extremes(dtype, pitch):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("HCU device required")
    dtype = getattr(torch, {"bf16": "bfloat16", "fp16": "float16"}[dtype])
    allocation = torch.full((19 * pitch + 32, ), float("nan"), device="cuda", dtype=dtype)
    rows = allocation[16:-16].view(19, pitch)
    rows[:, :64] = torch.randn((19, 64), device="cuda", dtype=dtype)
    before = allocation.clone()
    output = torch.full((16 * 32 + 32, ), -123, device="cuda", dtype=dtype)
    for logical_rows in (-1, 0, 19):
        for row in (-(2**31), -1, 0, 18, 32, 2**31 - 1):
            kernel = _register_descriptor_copy[(1, )](
                rows,
                output[16:-16],
                logical_rows,
                row,
                0,
                64,
                pitch,
                prefer_wide_async_copy=True,
                num_warps=4,
                num_stages=1,
            )
            assert "hcu.structured_register" in kernel.asm["ttgir"]
            expected = torch.zeros((16, 32), device="cuda", dtype=dtype)
            start, end = max(0, -row), min(16, logical_rows - row)
            if start < end:
                expected[start:end] = rows[row + start:row + end, :32]
            torch.testing.assert_close(output[16:-16].view(16, 32), expected, rtol=0, atol=0)
            assert torch.all(output[:16] == -123) and torch.all(output[-16:] == -123)
    torch.testing.assert_close(allocation, before, rtol=0, atol=0, equal_nan=True)


def test_async_descriptor_with_unproved_column_alignment_falls_back():
    source = ASTSource(
        _unpipelined_descriptor_dot,
        {"A": "*bf16", "B": "*bf16", "C": "*bf16", "COL": "i32", "BK": "constexpr", "ASYNC": "constexpr"},
        {"BK": 32, "ASYNC": True, "A_PITCH": 32},
        attrs={(i, ): [("tt.divisibility", 16), ("tt.pointer_range", 32)]
               for i in range(3)},
    )
    kernel = triton.compile(
        source,
        target=GPUTarget("hip", "gfx936", 64),
        options=dict(prefer_wide_async_copy=True, num_warps=4, num_stages=4, kpack=2, schedule_hint="local-prefetch",
                     mmac_warps_m=2, mmac_transposed=True),
    )
    assert kernel.asm["ttgir"].count("hcug.matrix_load_to_local") == 1
    assert "tt.matrix_load " not in kernel.asm["ttgir"]


@triton.jit
def _aligned_loop_descriptor_dot(A, B, C, BK: tl.constexpr, STAGES: tl.constexpr = 3):
    value = tl.full((128, 128), 0, tl.float32)
    for k in tl.range(0, 128, BK, num_stages=STAGES):
        a = tl.make_block_ptr(A, (128, 128), (128, 1), (0, k), (128, BK), (1, 0))
        b = tl.make_block_ptr(B, (128, 128), (1, 128), (k, 0), (BK, 128), (0, 1))
        av = tle.load(a, boundary_check=(0, 1), padding_option="zero", is_async=True)
        bv = tle.load(b, boundary_check=(0, 1), padding_option="zero", is_async=True)
        value = tl.dot(av, bv, value)
    m = tl.arange(0, 128)
    n = tl.arange(0, 128)
    tl.store(C + m[:, None] * 128 + n[None, :], value)


def test_async_descriptor_accepts_aligned_loop_induction_in_both_orders():
    source = ASTSource(
        _aligned_loop_descriptor_dot,
        {"A": "*bf16", "B": "*bf16", "C": "*bf16", "BK": "constexpr"},
        {"BK": 32},
        attrs={(i, ): [("tt.divisibility", 16), ("tt.pointer_range", 32)]
               for i in range(3)},
    )
    kernel = triton.compile(
        source,
        target=GPUTarget("hip", "gfx936", 64),
        options=dict(prefer_wide_async_copy=True, num_warps=4, num_stages=3, kpack=2, schedule_hint="local-prefetch",
                     mmac_warps_m=2, mmac_transposed=True),
    )
    assert kernel.asm["ttgir"].count("hcug.matrix_load_to_local") >= 2


@pytest.mark.parametrize("stages,slots", [(4, 2), (5, 4), (6, 8)])
def test_async_descriptor_ring_covers_the_scheduled_matrix_writes(stages, slots):
    source = ASTSource(
        _aligned_loop_descriptor_dot,
        {"A": "*bf16", "B": "*bf16", "C": "*bf16", "BK": "constexpr", "STAGES": "constexpr"},
        {"BK": 16, "STAGES": stages},
        attrs={(i, ): [("tt.divisibility", 16), ("tt.pointer_range", 32)]
               for i in range(3)},
    )
    kernel = triton.compile(
        source,
        target=GPUTarget("hip", "gfx936", 64),
        options=dict(prefer_wide_async_copy=True, num_warps=4, num_stages=stages, kpack=2,
                     schedule_hint="local-prefetch", mmac_warps_m=2, mmac_transposed=True),
    )
    ttgir = kernel.asm["ttgir"]
    assert "hcug.matrix_load_to_local" in ttgir
    assert re.findall(r"ttg\.local_alloc[^\n]*!ttg\.memdesc<(\d+)x", ttgir) == [str(slots)] * 2


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_paired_descriptor_preserves_k_contiguity_and_reindexes_columns(dtype):
    source = ASTSource(
        _aligned_loop_descriptor_dot,
        {"A": "*" + dtype, "B": "*" + dtype, "C": "*" + dtype, "BK": "constexpr"},
        {"BK": 16},
        attrs={(i, ): [("tt.divisibility", 16), ("tt.pointer_range", 32)]
               for i in range(3)},
    )
    opts = dict(prefer_wide_async_copy=True, num_warps=4, num_stages=3, kpack=2, schedule_hint="local-prefetch",
                mmac_warps_m=2, mmac_transposed=True)
    target = GPUTarget("hip", "gfx936", 64)
    control = triton.compile(source, target=target, options=opts).asm["ttgir"]
    paired = triton.compile(source, target=target, options=opts | {"mmac_pair_columns": True}).asm["ttgir"]
    assert "mmacLayout = 5" in paired

    def b_offset_basis(text):
        charts = (
            json.loads(offset)
            for offset in re.findall(r"(?m)^#\w+ = #ttg.shared_linear<\{offset = (\[.*\])\}, alignment = 16>$", text))
        basis, = [chart for chart in charts if chart[:3] == [[1, 0], [2, 0], [4, 0]]]
        return basis

    expected = [[k, (n & ~31) | ((n & 15) << 1) | ((n >> 4) & 1)] for k, n in b_offset_basis(control)]
    assert b_offset_basis(paired) == expected


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("pitch", [128, 384, 6144])
def test_record_pitch_policy_retains_register_stride_generalization(dtype, pitch):
    source = ASTSource(
        _register_descriptor_copy,
        {
            "A": "*" + dtype, "C": "*" + dtype, "M": "i32", "ROW": "i32", "COL": "i32", "K": "constexpr", "PITCH":
            "constexpr"
        },
        {"K": 64, "PITCH": pitch},
        attrs={(0, ): [("tt.divisibility", 16)], (1, ): [("tt.divisibility", 16)]},
    )
    for enabled in (False, True):
        kernel = triton.compile(
            source, target=GPUTarget("hip", "gfx936", 64),
            options=dict(prefer_wide_async_copy=True, num_warps=4, num_stages=1, normalize_buffer_record_pitch=enabled))
        assert "hcu.structured_register" in kernel.asm["ttgir"]
        record = 1 << (pitch.bit_length() - 1) if enabled else pitch
        assert f", i16 {2 * record | 0x4000}, i64" in kernel.asm["llir"]
        # Disabling factoring must retain physical row strides and the clamp
        # cleanup; it must not restore the historical logical-K descriptor.
        assert 'asm "s_sub_u32 $0, $1, $2' not in kernel.asm["llir"]


@pytest.mark.parametrize('rank', [1, 2, 3])
@pytest.mark.parametrize('dtype,padding', [('f16', None), ('f16', 1), ('f16', 2), ('bf16', 1), ('bf16', 2), ('f32', 1),
                                           ('f32', 2), ('i32', 1)])
@pytest.mark.parametrize('bounds', ['absent', 'empty', 'last'])
@pytest.mark.parametrize('async_', [False, True])
def test_padding_follows_mask(tmp_path, rank, dtype, padding, bounds, async_):
    shape = "x".join(["8"] * rank + [dtype])
    sizes = ", ".join(["%extent"] * rank)
    strides = ", ".join(["%one"] * rank)
    offsets = ", ".join(["%zero"] * rank)
    order = ", ".join(map(str, reversed(range(rank))))
    attrs = ["tt.load.async = " + str(async_).lower()]
    if bounds == "empty": attrs.append("boundaryCheck = array<i32>")
    if bounds == "last": attrs.append("boundaryCheck = array<i32: " + str(rank - 1) + ">")
    if padding is not None: attrs.append("padding = " + str(padding) + " : i32")
    # The unknown i64 shape forces pointer expansion even with wide-copy enabled.
    text = f"""tt.func public @load(%p: !tt.ptr<{dtype}>, %extent: i64) -> tensor<{shape}> {{
 %zero = arith.constant 0 : i32
 %one = arith.constant 1 : i64
 %ptr = tt.make_tensor_ptr %p, [{sizes}], [{strides}], [{offsets}] {{order = array<i32: {order}>}} : !tt.ptr<tensor<{shape}>>
 %v = tt.load %ptr {{{", ".join(attrs)}}} : !tt.ptr<tensor<{shape}>>
 tt.return %v : tensor<{shape}>
}}"""
    result = rewrite(tmp_path, text)
    assert "tt.make_tensor_ptr" not in result and "tt.matrix_load" not in result
    line = next(line for line in result.splitlines() if " = tt.load " in line)
    arguments = line.split("{", 1)[0].split(":", 1)[0]
    expected = (2 if padding is not None else 1) if bounds == "last" else 0
    assert arguments.count(",") == expected
    assert ("arith.cmpi" in result) == (bounds == "last")
    assert "tt.load.async = " + str(async_).lower() in line
