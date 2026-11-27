# matrix shared copy
"""Matrix G2L is derived from ordinary SharedLinear memory."""
import pytest
import triton
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource
from triton.experimental.gluon.language.amd.cdna4.async_copy import commit_group, wait_group

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU bindings required", allow_module_level=True)
from triton.experimental.gluon.language.hcu import matrix_load_to_shared


@g.constexpr_function
def _matrix_shared_copy_shared_layout(transpose, bad):
    bases = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]]
    if bad: bases[0], bases[3] = bases[3], bases[0]
    if transpose: bases = [[b[1], b[0]] for b in bases]
    return gl.SharedLinearLayout(bases)


@g.jit
def _matrix_shared_copy_copy(A, O, ROW, COL, ENABLE, TRANSPOSE: gl.constexpr, BAD_CHART: gl.constexpr,
                             PITCH: gl.constexpr, COLUMNS: gl.constexpr):
    shared: gl.constexpr = _matrix_shared_copy_shared_layout(TRANSPOSE, BAD_CHART)
    if TRANSPOSE:
        shape: gl.constexpr = [32, 64]
        extents: gl.constexpr = [COLUMNS, 83]
        strides: gl.constexpr = [1, PITCH]
        origins = (COL * 8, ROW)
    else:
        shape: gl.constexpr = [64, 32]
        extents: gl.constexpr = [83, COLUMNS]
        strides: gl.constexpr = [PITCH, 1]
        origins = (ROW, COL * 8)
    dest = gl.allocate_shared_memory(A.dtype.element_ty, shape, shared)
    matrix_load_to_shared(dest, A, extents, strides, origins, ENABLE)
    commit_group()
    ready = wait_group(0, return_token=True)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    value = dest.load(layout, token=ready)
    i = gl.arange(0, shape[0], layout=gl.SliceLayout(1, layout))
    j = gl.arange(0, shape[1], layout=gl.SliceLayout(0, layout))
    gl.store(O + i[:, None] * shape[1] + j[None, :], value)


def _matrix_shared_copy_compile_copy(dtype="bf16", transpose=False, bad=False, pitch=80, columns=72):
    source = GluonASTSource(
        _matrix_shared_copy_copy, {
            "A": "*" + dtype, "O": "*" + dtype, "ROW": "i32", "COL": "i32", "ENABLE": "i32", "TRANSPOSE": "constexpr",
            "BAD_CHART": "constexpr", "PITCH": "constexpr", "COLUMNS": "constexpr"
        }, {"TRANSPOSE": transpose, "BAD_CHART": bad, "PITCH": pitch, "COLUMNS": columns},
        attrs={(0, ): [("tt.divisibility", 16)], (1, ): [("tt.divisibility", 16)]})
    return triton.compile(source, target=GPUTarget("hip", "gfx936", 64), options=dict(num_warps=4, num_stages=1))


@pytest.mark.parametrize("pitch", [80, 8192, 1 << 24])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("transpose", [False, True])
def test_plain_shared_matrix_copy(dtype, transpose, pitch):
    kernel = _matrix_shared_copy_compile_copy(dtype, transpose, pitch=pitch)
    ir = kernel.asm['ttgir']
    assert "matrix_load_to_local" in ir and "shared_linear" in ir
    intrinsic = "llvm.amdgcn.struct.ptr.buffer.load.lds" if pitch < 8192 else "llvm.amdgcn.raw.ptr.buffer.load.lds"
    assert intrinsic in kernel.asm['llir']


@pytest.mark.parametrize("kwargs,reason", [
    (dict(bad=True), "physical vectors"),
    (dict(pitch=0), "row pitch"),
    (dict(columns=71), "row pitch"),
    (dict(pitch=1 << 25), "signed31-bit"),
])
def test_unsupported_structured_copy_is_rejected(kwargs, reason, capfd):
    # Explicit target AOT: no GPU is initialized, malformed proofs cannot reach
    # a native producer. The caller can select ordinary buffer copy instead.
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _matrix_shared_copy_compile_copy(**kwargs)
    diagnostic = capfd.readouterr().err
    assert "matrix shared copy" in diagnostic and reason in diagnostic


@pytest.mark.parametrize("pitch", [72, 88, 256, 272, 768, 1280, 3072, 4096, 6144, 7936, 8176, 8184, 8192, 8200])
def test_ordinary_stride_field_and_raw_fallback(pitch):
    # Exercise the real ordinary lowering, including valid pitches below256
    # and pitches divisible by8 but not16; the retired PacketDomain is stricter
    # than this target ABI and must not constrain ordinary copies.
    kernel = _matrix_shared_copy_compile_copy(pitch=pitch)
    structured = "llvm.amdgcn.struct.ptr.buffer.load.lds" in kernel.asm["llir"]
    assert structured == (2 * pitch < 0x4000)
    assert "llvm.amdgcn.raw.ptr.buffer.load.lds" in kernel.asm["llir"] or structured
    assert "shared_linear" in kernel.asm["ttgir"]


@g.jit
def _matrix_shared_copy_copy_known_origins(A, O, ENABLE, ROW: gl.constexpr, COL: gl.constexpr, TRANSPOSE: gl.constexpr,
                                           PITCH: gl.constexpr):
    _matrix_shared_copy_copy(A, O, ROW, COL, ENABLE, TRANSPOSE, False, PITCH, 72)


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("row,col,expected_raw", [
    (0, 0, True),
    (19, 1, True),
    (20, 0, False),
    (-1, 0, False),
    (0, -1, True),
    (0, 6, True),
])
def test_proved_complete_rectangle_prefers_affine_raw(dtype, transpose, row, col, expected_raw):
    signature = {
        "A": "*" + dtype, "O": "*" + dtype, "ENABLE": "i32", "ROW": "constexpr", "COL": "constexpr", "TRANSPOSE":
        "constexpr", "PITCH": "constexpr"
    }
    source = GluonASTSource(
        _matrix_shared_copy_copy_known_origins,
        signature,
        {"ROW": row, "COL": col, "TRANSPOSE": transpose, "PITCH": 80},
        attrs={(0, ): [("tt.divisibility", 16)], (1, ): [("tt.divisibility", 16)]},
    )
    kernel = triton.compile(source, target=GPUTarget("hip", "gfx936", 64), options=dict(num_warps=4, num_stages=1))
    raw = "llvm.amdgcn.raw.ptr.buffer.load.lds" in kernel.asm["llir"]
    structured = "llvm.amdgcn.struct.ptr.buffer.load.lds" in kernel.asm["llir"]
    # Complete rows with partial columns use affine RAW plus per-vector zero fill.
    # The older descriptor-only expectation predates that lowering.
    assert raw == expected_raw
    assert structured != expected_raw


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("row,col", [(0, 0), (19, 1), (20, 0), (0, -1)])
def test_complete_rectangle_and_partial_copy_changed_replay(dtype_name, transpose, row, col):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires an HCU device")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx936":
        pytest.skip("requires gfx936")
    dtype = getattr(torch, dtype_name)
    source = torch.empty((83, 80), device="cuda", dtype=dtype)
    storage = torch.full((64 * 32 + 512, ), 7., device="cuda", dtype=dtype)
    output = storage[256:-256]
    for enabled in (1, 0):
        _matrix_shared_copy_copy_known_origins[(1, )](source, output, enabled, row, col, transpose, 80, num_warps=4,
                                                      num_stages=1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(4):
                _matrix_shared_copy_copy_known_origins[(1, )](source, output, enabled, row, col, transpose, 80,
                                                              num_warps=4, num_stages=1)
        for seed in (20261107, 20261113):
            torch.manual_seed(seed)
            source.normal_()
            saved_input = source.clone()
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            expected = torch.zeros((64, 32), dtype=dtype)
            if enabled:
                rows = torch.arange(64) + row
                cols = torch.arange(32) + col * 8
                valid = ((rows >= 0) & (rows < 83))[:, None] & ((cols >= 0) & (cols < 72))[None, :]
                actual_input = saved_input.cpu()
                values = actual_input[rows.clamp(0, 82)[:, None], cols.clamp(0, 71)[None, :]]
                expected = torch.where(valid, values, expected)
            if transpose:
                expected = expected.t().contiguous()
            assert torch.equal(output.cpu(), expected.reshape(-1))
            assert torch.equal(source, saved_input)
            assert torch.all(storage[:256] == 7).item()
            assert torch.all(storage[-256:] == 7).item()


@g.jit
def _matrix_shared_copy_copy_unsigned_origins(A, O, INDEX, ENABLE, MOD: gl.constexpr, DIV: gl.constexpr,
                                              STEP: gl.constexpr, BIAS: gl.constexpr, TRANSPOSE: gl.constexpr):
    row = (((INDEX.to(gl.uint32) % MOD) // DIV).to(gl.int32) * STEP + BIAS)
    _matrix_shared_copy_copy(A, O, row, 0, ENABLE, TRANSPOSE, False, 80, 72)


# The input may have any i32 bit pattern. Prove bounds from the unsigned
# operations themselves, including high-bit divisors, rather than a launch grid.
_matrix_shared_copy_UNSIGNED_ORIGIN_CASES = [
    (19, 1, 1, 0, True),
    (23, 3, 2, 0, True),
    (19, 1, 2, 0, False),
    (19, 1, 1, -1, False),
    (19, 1, 1, 2, False),
    (31, 3, 1, 0, True),
    (2147483647, 1073741824, 1, 0, True),
    (2147483651, 2147483648, 1, 0, True),
    (2147483651, 1, 1, 0, False),
    (19, 1, 2147483647, 0, False),
]


# Every range-proof obligation is checked on both A/B orientations. f16 and
# bf16 use the same 16-bit integer address proof; their data paths are covered
# independently by the shared-copy numerical tests below.
@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("modulus,divisor,step,bias,complete", _matrix_shared_copy_UNSIGNED_ORIGIN_CASES)
def test_unsigned_origin_range_selects_raw_without_launch_assumptions(dtype, transpose, modulus, divisor, step, bias,
                                                                      complete):
    constants = dict(MOD=modulus, DIV=divisor, STEP=step, BIAS=bias, TRANSPOSE=transpose)
    signature = dict(A="*" + dtype, O="*" + dtype, INDEX="i32", ENABLE="i32",
                     **{name: "constexpr"
                        for name in constants})
    source = GluonASTSource(
        _matrix_shared_copy_copy_unsigned_origins,
        signature,
        constants,
        attrs={(0, ): [("tt.divisibility", 16)], (1, ): [("tt.divisibility", 16)]},
    )
    kernel = triton.compile(source, target=GPUTarget("hip", "gfx936", 64), options=dict(num_warps=4, num_stages=1))
    raw = "llvm.amdgcn.raw.ptr.buffer.load.lds" in kernel.asm["llir"]
    structured = "llvm.amdgcn.struct.ptr.buffer.load.lds" in kernel.asm["llir"]
    assert raw == complete and structured != complete


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("transpose", [False, True])
def test_unsigned_origin_negative_bits_partial_rows_and_changed_replay(dtype_name, transpose):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires an HCU device")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx936":
        pytest.skip("requires gfx936")
    dtype = getattr(torch, dtype_name)
    source = torch.empty((83, 80), device="cuda", dtype=dtype)
    storage = torch.full((64 * 32 + 512, ), 7., device="cuda", dtype=dtype)
    output = storage[256:-256]
    for modulus, divisor, step, bias, _ in _matrix_shared_copy_UNSIGNED_ORIGIN_CASES:
        for index in (-2147483648, -1, 0, 2147483647):
            quotient = ((index & 0xffffffff) % modulus) // divisor
            row = ((quotient * step + bias + (1 << 31)) % (1 << 32)) - (1 << 31)
            for enabled in (0, 1):

                def launch():
                    _matrix_shared_copy_copy_unsigned_origins[(1, )](source, output, index, enabled, modulus, divisor,
                                                                     step, bias, transpose, num_warps=4, num_stages=1)

                launch()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(4):
                        launch()
                for seed in (20261107, 20261119):
                    torch.manual_seed(seed)
                    source.normal_()
                    saved_input = source.clone()
                    output.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = torch.zeros((64, 32), dtype=dtype)
                    if enabled:
                        rows = torch.arange(64, dtype=torch.int64) + row
                        cols = torch.arange(32, dtype=torch.int64)
                        valid = ((rows >= 0) & (rows < 83))[:, None]
                        expected = torch.where(valid,
                                               saved_input.cpu()[rows.clamp(0, 82)[:, None], cols[None, :]], expected)
                    if transpose:
                        expected = expected.t().contiguous()
                    assert torch.equal(output.cpu(), expected.reshape(-1))
                    assert torch.equal(source, saved_input)
                    assert torch.all(storage[:256] == 7).item()
                    assert torch.all(storage[-256:] == 7).item()


@g.constexpr_function
def _matrix_shared_copy_cohort_shared_layout(tile_columns, transpose):
    tile_rows = 2048 // tile_columns
    bases = [[0, 1 << b] for b in range(tile_columns.bit_length() - 1)]
    bases += [[1 << b, 0] for b in range(tile_rows.bit_length() - 1)]
    if transpose:
        bases = [[y, x] for x, y in bases]
    return gl.SharedLinearLayout(bases, alignment=16)


@g.jit
def _matrix_shared_copy_copy_row_cohort(A, O, ROW, COL, ENABLE, TILE_COLUMNS: gl.constexpr, ROWS: gl.constexpr,
                                        COLUMNS: gl.constexpr, PITCH: gl.constexpr, RSTEP: gl.constexpr,
                                        CSTEP: gl.constexpr, TRANSPOSE: gl.constexpr):
    tile_rows: gl.constexpr = 2048 // TILE_COLUMNS
    shared: gl.constexpr = _matrix_shared_copy_cohort_shared_layout(TILE_COLUMNS, TRANSPOSE)
    if TRANSPOSE:
        shape: gl.constexpr = [TILE_COLUMNS, tile_rows]
        extents: gl.constexpr = [COLUMNS, ROWS]
        strides: gl.constexpr = [1, PITCH]
        origins = (COL * CSTEP, ROW * RSTEP)
    else:
        shape: gl.constexpr = [tile_rows, TILE_COLUMNS]
        extents: gl.constexpr = [ROWS, COLUMNS]
        strides: gl.constexpr = [PITCH, 1]
        origins = (ROW * RSTEP, COL * CSTEP)
    dest = gl.allocate_shared_memory(A.dtype.element_ty, shape, shared)
    matrix_load_to_shared(dest, A, extents, strides, origins, ENABLE)
    commit_group()
    ready = wait_group(0, return_token=True)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    value = dest.load(layout, token=ready)
    i = gl.arange(0, shape[0], layout=gl.SliceLayout(1, layout))
    j = gl.arange(0, shape[1], layout=gl.SliceLayout(0, layout))
    gl.store(O + i[:, None] * shape[1] + j[None, :], value)


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("tile_columns", [32, 64])
@pytest.mark.parametrize("rows,columns,pitch,aligned,raw", [
    (128, 128, 128, True, True),
    (128, 128, 8192, True, True),
    (128, 128, 8200, True, True),
    (83, 128, 128, True, False),
    (128, 128, 128, False, False),
    (128, 72, 128, True, True),
])
def test_whole_row_nr0_raw_cache_policy_from_actual_wave_chart(dtype, transpose, tile_columns, rows, columns, pitch,
                                                               aligned, raw):
    import re
    constants = dict(TILE_COLUMNS=tile_columns, ROWS=rows, COLUMNS=columns, PITCH=pitch,
                     RSTEP=2048 // tile_columns if aligned else 8, CSTEP=tile_columns, TRANSPOSE=transpose)
    signature = dict(A="*" + dtype, O="*" + dtype, ROW="i32", COL="i32", ENABLE="i32",
                     **{name: "constexpr"
                        for name in constants})
    source = GluonASTSource(_matrix_shared_copy_copy_row_cohort, signature, constants,
                            attrs={(0, ): [("tt.divisibility", 16)], (1, ): [("tt.divisibility", 16)]})
    kernel = triton.compile(source, target=GPUTarget("hip", "gfx936", 64), options=dict(num_warps=4, num_stages=1))
    llir = kernel.asm["llir"]
    assert ("llvm.amdgcn.raw.ptr.buffer.load.lds" in llir) == raw
    assert ("llvm.amdgcn.struct.ptr.buffer.load.lds" in llir) != raw
    # The two ordinary charts have respectively 4 and 8 vectors per row/wave.
    # A legal raw copy must retain its byte bounds irrespective of cache policy.
    controls = re.findall(r"@llvm\.amdgcn\.make\.buffer\.rsrc[^\s(]*"
                          r"\(ptr addrspace\(1\) [^,]+, i16 (-?\d+),", llir)
    assert len(controls) == 1
    pitched_cache = 2 * pitch < 0x4000 and (not raw or tile_columns == 32)
    assert int(controls[0]) == ((2 * pitch) | 0x4000 if pitched_cache else 0)
    assert not int(controls[0]) & 0x8000  # Address swizzle is never enabled.


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("tile_columns", [32, 64])
@pytest.mark.parametrize("pitch", [144, 1664, 3072, 3104])
def test_whole_row_nr0_and_partial_fallback_changed_input_replay(dtype_name, transpose, tile_columns, pitch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires an HCU device")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx936":
        pytest.skip("requires gfx936")
    dtype = getattr(torch, dtype_name)
    tile_rows = 2048 // tile_columns
    # Include both structured partial rows and RAW partial columns. The two
    # aligned, non-power-of-two pitches exercise record/offset decomposition;
    # 3104 retains its physical pitch, and 144 covers the original small chart.
    for rows, columns, rstep in [(128, 128, tile_rows), (128, 72, tile_rows), (83, 72, 8), (83, 128, 8)]:
        backing = torch.empty((rows, pitch), device="cuda", dtype=dtype)
        source = backing[:, :columns]
        input_guard = backing[:, columns:columns + 8]
        input_guard.fill_(7.)
        storage = torch.full((2048 + 512, ), 7., device="cuda", dtype=dtype)
        output = storage[256:-256]
        origins = [
            (-1, 0, tile_columns),
            (0, -1, tile_columns),
            (1, 1, tile_columns),
            (0, columns // tile_columns, tile_columns),
            (rows // rstep + 1, 0, tile_columns),
            (2147483647, 3, tile_columns),
        ]
        if columns % tile_columns:
            origins.extend([
                (0, -1, 8),  # A negative aligned origin with a valid suffix.
                (1, columns // 8 - 1, 8),  # One valid vector at the right edge.
                (0, -(1 << 28), 8),
                (0, (1 << 28) - 1, 8),
            ])
        for ri, ci, cstep in origins:
            row = ((ri * rstep + (1 << 31)) % (1 << 32)) - (1 << 31)
            col = ((ci * cstep + (1 << 31)) % (1 << 32)) - (1 << 31)
            for enabled in (0, 1):

                def launch():
                    _matrix_shared_copy_copy_row_cohort[(1, )](source, output, ri, ci, enabled, tile_columns, rows,
                                                               columns, pitch, rstep, cstep, transpose, num_warps=4,
                                                               num_stages=1)

                launch()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(4):
                        launch()
                for seed in (20261107, 20261119):
                    torch.manual_seed(seed)
                    source.normal_()
                    saved = source.clone()
                    output.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = torch.zeros((tile_rows, tile_columns), dtype=dtype)
                    if enabled:
                        rr = torch.arange(tile_rows, dtype=torch.int64) + row
                        cc = torch.arange(tile_columns, dtype=torch.int64) + col
                        valid = ((rr >= 0) & (rr < rows))[:, None] & ((cc >= 0) & (cc < columns))[None, :]
                        values = saved.cpu()[rr.clamp(0, rows - 1)[:, None], cc.clamp(0, columns - 1)[None, :]]
                        expected = torch.where(valid, values, expected)
                    if transpose:
                        expected = expected.t().contiguous()
                    assert torch.equal(output.cpu(), expected.reshape(-1))
                    assert torch.equal(source, saved)
                    assert torch.all(input_guard == 7).item()
                    assert torch.all(storage[:256] == 7).item()
                    assert torch.all(storage[-256:] == 7).item()


@pytest.mark.parametrize("value", [-1, 2, "physical"])
def test_record_pitch_policy_requires_boolean(value):
    from triton.backends.hcu.compiler import HIPBackend

    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    with pytest.raises(ValueError, match="normalize_buffer_record_pitch must be boolean"):
        backend.parse_options({"normalize_buffer_record_pitch": value})


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("row", [0, 20])
@pytest.mark.parametrize("pitch", [128, 384, 6144, 8192])
def test_record_pitch_policy_preserves_matrix_copy_mode(dtype, transpose, row, pitch):
    import re

    source = GluonASTSource(
        _matrix_shared_copy_copy_known_origins,
        {
            "A": "*" + dtype, "O": "*" + dtype, "ENABLE": "i32", "ROW": "constexpr", "COL": "constexpr", "TRANSPOSE":
            "constexpr", "PITCH": "constexpr"
        },
        {"ROW": row, "COL": 0, "TRANSPOSE": transpose, "PITCH": pitch},
        attrs={(0, ): [("tt.divisibility", 16)], (1, ): [("tt.divisibility", 16)]},
    )
    base_options = dict(num_warps=4, num_stages=1)
    target = GPUTarget("hip", "gfx936", 64)
    default = triton.compile(source, target=target, options=base_options)
    normalized = triton.compile(source, target=target, options=dict(base_options, normalize_buffer_record_pitch=True))
    physical = triton.compile(source, target=target, options=dict(base_options, normalize_buffer_record_pitch=False))
    assert default.hash == normalized.hash
    assert physical.hash != normalized.hash
    if pitch in (128, 8192):
        assert physical.asm["llir"] == normalized.asm["llir"]
    else:
        for artifact, record in ((physical, pitch), (normalized, 1 << (pitch.bit_length() - 1))):
            fields = re.findall(r"make\.buffer\.rsrc[^\n]*, i16 (\d+), i64", artifact.asm["llir"])
            assert str(2 * record | 0x4000) in fields
    intrinsic = "struct.ptr.buffer.load.lds" if row == 20 and pitch < 8192 else "raw.ptr.buffer.load.lds"
    assert intrinsic in physical.asm["llir"] and intrinsic in normalized.asm["llir"]


@pytest.mark.parametrize("value,expected", [(0, False), (1, True), (None, True)])
def test_record_pitch_policy_uses_existing_option_cache_contract(value, expected):
    from triton.backends.hcu.compiler import HIPBackend

    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    actual = backend.parse_options({"normalize_buffer_record_pitch": value})
    canonical = backend.parse_options({"normalize_buffer_record_pitch": expected})
    assert actual.normalize_buffer_record_pitch is expected
    assert actual.hash() == canonical.hash()


def _matrix_shared_copy_matrix_copy_with_control(source, control):
    """Place the real collective under CF without executing unsafe controls."""
    import re

    lines = source.splitlines(keepends=True)
    first = next(i for i, line in enumerate(lines) if "tt.func public" in line) + 1
    last = next(i for i in range(first, len(lines)) if lines[i].startswith("  }"))
    body = "".join(lines[first:last])
    assert "tt.return" in body and "scf." not in body
    prefix = ("    %control_tid = rocdl.workitem.id.x : i32\n"
              "    %control_128 = arith.constant 128 : i32\n"
              "    %control_zero = arith.constant 0 : i32\n"
              "    %control_role = arith.cmpi ult, %control_tid, %control_128 : i32\n"
              "    %control_all = arith.cmpi ne, %ENABLE, %control_zero : i32\n")
    if control.startswith("switch-phi-"):
        value = "%ROW" if control.endswith("same") else "%COL"
        changed = (prefix + "    cf.switch %control_tid : i32, [default: ^join(%ROW : i32), " +
                   f"0: ^join({value} : i32)]\n  ^join(%chosen: i32):\n" + re.sub(r"%ROW\b", "%chosen", body))
    elif control in ("reconverged", "phi-different", "phi-same", "switch-reconverged"):
        branch = ("    cf.switch %control_tid : i32, [default: ^right, 0: ^left]\n"
                  if control == "switch-reconverged" else "    cf.cond_br %control_role, ^left, ^right\n")
        if control.startswith("phi-"):
            left = "    %left_value = arith.constant 0 : i32\n    cf.br ^join(%left_value : i32)\n"
            right_value = 8 if control == "phi-different" else 0
            right = f"    %right_value = arith.constant {right_value} : i32\n    cf.br ^join(%right_value : i32)\n"
            join = "  ^join(%chosen: i32):\n"
            body = re.sub(r"%ROW\b", "%chosen", body)
        else:
            left = right = "    cf.br ^join\n"
            join = "  ^join:\n"
        changed = (prefix + branch + "  ^left:\n    rocdl.sched.barrier 0\n" + left +
                   "  ^right:\n    rocdl.sched.barrier 1\n" + right + join + body)
    elif control in ("guard-role", "guard-uniform", "switch-guard"):
        if control == "switch-guard":
            branch = "    cf.switch %control_tid : i32, [default: ^exit, 0: ^copy]\n"
        else:
            condition = "%control_all" if control == "guard-uniform" else "%control_role"
            branch = f"    cf.cond_br {condition}, ^copy, ^exit\n"
        changed = prefix + branch + "  ^copy:\n" + body + "  ^exit:\n    tt.return\n"
    else:
        assert control == "loop-role"
        body = re.sub(r"    tt.return[^\n]*\n", "", body)
        changed = (prefix + "    cf.br ^loop\n  ^loop:\n" + body +
                   "    cf.cond_br %control_role, ^loop, ^exit\n  ^exit:\n    tt.return\n")
    return "".join(lines[:first]) + changed + "".join(lines[last:])


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("control,accepted", [
    ("reconverged", True),
    ("guard-role", False),
    ("guard-uniform", True),
    ("loop-role", False),
    ("phi-different", False),
    ("phi-same", True),
    ("switch-reconverged", True),
    ("switch-guard", False),
    ("switch-phi-same", True),
    ("switch-phi-different", False),
])
def test_matrix_copy_requires_uniform_participation(dtype, transpose, control, accepted, tmp_path, capfd):
    source = _matrix_shared_copy_compile_copy(dtype=dtype, transpose=transpose).asm["ttgir"]
    path = tmp_path / "control.ttgir"
    path.write_text(_matrix_shared_copy_matrix_copy_with_control(source, control))
    options = dict(num_warps=4, num_stages=1, use_block_pingpong=False)
    target = GPUTarget("hip", "gfx936", 64)
    if accepted:
        result = triton.compile(str(path), target=target, options=options)
        assert "buffer.load.lds" in result.asm["llir"]
    else:
        # CPU compilation only: a partial-wave collective must never run.
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            triton.compile(str(path), target=target, options=options)
        assert "matrix shared copy: unproven uniform origins" in capfd.readouterr().err


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("transpose", [False, True])
def test_private_function_scalar_is_not_a_kernel_abi_argument(dtype, transpose, tmp_path, capfd):
    # A noinline device function can receive a per-thread scalar. Treating its
    # argument as an ABI-uniform value would silently select an SGPR origin.
    source = _matrix_shared_copy_compile_copy(dtype=dtype, transpose=transpose).asm["ttgir"]
    lines = source.splitlines(keepends=True)
    first = next(i for i, line in enumerate(lines) if "tt.func public" in line)
    last = next(i for i in range(first + 1, len(lines)) if lines[i].startswith("  }"))
    helper = re.sub(r"public @[^ (]+", "private @helper", lines[first]).replace("noinline = false", "noinline = true")
    ptr = "!tt.ptr<bf16>" if dtype == "bf16" else "!tt.ptr<f16>"
    entry = (f"  tt.func public @entry(%A: {ptr} {{tt.divisibility = 16 : i32}}, "
             f"%O: {ptr} {{tt.divisibility = 16 : i32}}, %ROW: i32, %COL: i32, %ENABLE: i32) {{\n"
             "    %tid = rocdl.workitem.id.x : i32\n"
             f"    tt.call @helper(%A, %O, %tid, %COL, %ENABLE) : ({ptr}, {ptr}, i32, i32, i32) -> ()\n"
             "    tt.return\n  }\n")
    source = "".join(lines[:first]) + helper + "".join(lines[first + 1:last + 1]) + entry + "".join(lines[last + 1:])
    path = tmp_path / "private-callee.ttgir"
    path.write_text(source)
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        triton.compile(str(path), target=GPUTarget("hip", "gfx936", 64),
                       options=dict(num_warps=4, num_stages=1, use_block_pingpong=False))
    assert "requires actual aligned source argument" in capfd.readouterr().err


# matrix descriptor preservation
import json
import re

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)

from triton.backends.hcu.compiler import HIPBackend
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler import ASTSource


def _matrix_descriptor_preservation_rewrite(tmp_path, body, enabled=True):
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
    result = _matrix_descriptor_preservation_rewrite(tmp_path, text, enabled)
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
    result = _matrix_descriptor_preservation_rewrite(tmp_path, text)
    assert ("tt.matrix_load" in result) == expected
    assert "tt.make_tensor_ptr" not in result and "tt.advance" not in result


@triton.jit
def _matrix_descriptor_preservation_unpipelined_descriptor_dot(A, B, C, BK: tl.constexpr, ASYNC: tl.constexpr, COL,
                                                               A_PITCH: tl.constexpr):
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
        _matrix_descriptor_preservation_unpipelined_descriptor_dot,
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
        _matrix_descriptor_preservation_unpipelined_descriptor_dot,
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
def _matrix_descriptor_preservation_register_descriptor_copy(A, C, M, ROW, COL, K: tl.constexpr, PITCH: tl.constexpr):
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
        kernel = _matrix_descriptor_preservation_register_descriptor_copy[(1, )](
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
            kernel = _matrix_descriptor_preservation_register_descriptor_copy[(1, )](
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
        _matrix_descriptor_preservation_unpipelined_descriptor_dot,
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
def _matrix_descriptor_preservation_aligned_loop_descriptor_dot(A, B, C, BK: tl.constexpr, STAGES: tl.constexpr = 3):
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
        _matrix_descriptor_preservation_aligned_loop_descriptor_dot,
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
        _matrix_descriptor_preservation_aligned_loop_descriptor_dot,
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
        _matrix_descriptor_preservation_aligned_loop_descriptor_dot,
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
        _matrix_descriptor_preservation_register_descriptor_copy,
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
    result = _matrix_descriptor_preservation_rewrite(tmp_path, text)
    assert "tt.make_tensor_ptr" not in result and "tt.matrix_load" not in result
    line = next(line for line in result.splitlines() if " = tt.load " in line)
    arguments = line.split("{", 1)[0].split(":", 1)[0]
    expected = (2 if padding is not None else 1) if bounds == "last" else 0
    assert arguments.count(",") == expected
    assert ("arith.cmpi" in result) == (bounds == "last")
    assert "tt.load.async = " + str(async_).lower() in line


# full wave buffer copy


def _full_wave_buffer_copy_lower_copy(tmp_path, divergent):
    limit = "%lane" if divergent else "%trips"
    text = f'''#blocked = #ttg.blocked<{{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}}>
#shared = #ttg.swizzled_shared<{{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}}>
#smem = #ttg.shared_memory
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"}} {{
  tt.func public @copy(%ptr: !tt.ptr<bf16>) {{
    ttg.warp_specialize(%ptr) attributes {{requestedRegisters = array<i32: 32>}}
    default {{ ttg.warp_yield }}
    partition0(%p: !tt.ptr<bf16>) num_warps(4) {{
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      %trips = arith.constant 3 : i32
      %tile_k = arith.constant 64 : i32
      %lane = rocdl.workitem.id.x : i32
      %rows = tt.make_range {{end = 256 : i32, start = 0 : i32}}
        : tensor<256xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
      %expanded_rows = tt.expand_dims %rows {{axis = 0 : i32}}
        : tensor<256xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>> -> tensor<1x256xi32, #blocked>
      %stride = arith.constant dense<256> : tensor<1x256xi32, #blocked>
      %row_offsets = arith.muli %expanded_rows, %stride : tensor<1x256xi32, #blocked>
      %rows_broadcast = tt.broadcast %row_offsets : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %columns = tt.make_range {{end = 64 : i32, start = 0 : i32}}
        : tensor<64xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
      %expanded_columns = tt.expand_dims %columns {{axis = 1 : i32}}
        : tensor<64xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>> -> tensor<64x1xi32, #blocked>
      %columns_broadcast = tt.broadcast %expanded_columns : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %local = arith.addi %rows_broadcast, %columns_broadcast : tensor<64x256xi32, #blocked>
      %bound = arith.constant dense<255> : tensor<1x256xi32, #blocked>
      %row_mask = arith.cmpi ult, %expanded_rows, %bound : tensor<1x256xi32, #blocked>
      %mask = tt.broadcast %row_mask : tensor<1x256xi1, #blocked> -> tensor<64x256xi1, #blocked>
      %dest = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable>
      scf.for %iv = %zero to {limit} step %one : i32 {{
        %origin = arith.muli %iv, %tile_k : i32
        %base = tt.splat %origin : i32 -> tensor<64x256xi32, #blocked>
        %offset = arith.addi %base, %local : tensor<64x256xi32, #blocked>
        %copy = hcug.buffer_load_to_local %p[%offset] mask = %mask into %dest
          {{contiguity = 8 : i32}} : <bf16>[tensor<64x256xi32, #blocked>]
          -> <64x256xbf16, #shared, #smem, mutable>
        %commit = ttg.async_commit_group tokens %copy
        %wait = hcug.async_wait %commit {{num_inst = 0 : i32}}
      }}
      ttg.warp_return
    }} : (!tt.ptr<bf16>) -> ()
    tt.return
  }}
}}'''
    path = tmp_path / "copy.mlir"
    path.write_text(text)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    manager = libtriton.ir.pass_manager(context)
    target_passes = libtriton.hcu.passes.ttgpuir
    target_passes.add_warp_specialize_to_llvm(manager, "gfx936", 4, 4, False, 0, 0, 0)
    libtriton.passes.convert.add_scf_to_cf(manager)
    target_passes.add_allocate_shared_memory(manager)
    target_passes.add_to_llvmir(manager, "gfx936", False)
    manager.run(module, "full_wave_buffer_copy")
    return str(module)


@pytest.mark.parametrize("divergent", [False, True])
def test_buffer_copy_retains_ballot_for_divergent_loop(tmp_path, divergent):
    lowered = _full_wave_buffer_copy_lower_copy(tmp_path, divergent)
    assert "buffer.load.lds" in lowered
    assert ("rocdl.ballot" in lowered) == divergent


# sync matrix descriptor


@triton.jit
def _sync_matrix_descriptor_descriptor_copy(A, C, origin, runtime_stride, M: tl.constexpr, K: tl.constexpr,
                                            BM: tl.constexpr, BK: tl.constexpr, ROW: tl.constexpr, COLUMN: tl.constexpr,
                                            ROW_STRIDE: tl.constexpr, COLUMN_STRIDE: tl.constexpr,
                                            DYNAMIC_COLUMN: tl.constexpr):
    row = tl.program_id(0) * BM + ROW
    if ROW_STRIDE == -1:
        stride = runtime_stride
    elif ROW_STRIDE == 0:
        stride = K
    else:
        stride = ROW_STRIDE
    if DYNAMIC_COLUMN:
        column = origin
    else:
        column = tl.program_id(1) * BK + COLUMN
    pointer = tl.make_block_ptr(A, (M, K), (stride, COLUMN_STRIDE), (row, column), (BM, BK), (1, 0))
    value = tl.load(pointer, boundary_check=(0, 1), padding_option="zero")
    m = tl.arange(0, BM)
    n = tl.arange(0, BK)
    tl.store(C + m[:, None] * BK + n[None, :], value)


@triton.jit
def _sync_matrix_descriptor_idle():
    pass


@triton.jit
def _sync_matrix_descriptor_descriptor_copy_roles(A, C, origin, runtime_stride, M: tl.constexpr, K: tl.constexpr,
                                                  BM: tl.constexpr, BK: tl.constexpr, ROW: tl.constexpr,
                                                  COLUMN: tl.constexpr, ROW_STRIDE: tl.constexpr,
                                                  COLUMN_STRIDE: tl.constexpr, DYNAMIC_COLUMN: tl.constexpr,
                                                  DEFAULT_ROLE: tl.constexpr):
    if DEFAULT_ROLE:
        tle.gpu.warp_specialize(
            [(_sync_matrix_descriptor_descriptor_copy,
              (A, C, origin, runtime_stride, M, K, BM, BK, ROW, COLUMN, ROW_STRIDE, COLUMN_STRIDE, DYNAMIC_COLUMN)),
             (_sync_matrix_descriptor_idle, ())], [4], [32])
    else:
        tle.gpu.warp_specialize(
            [(_sync_matrix_descriptor_idle, ()),
             (_sync_matrix_descriptor_descriptor_copy,
              (A, C, origin, runtime_stride, M, K, BM, BK, ROW, COLUMN, ROW_STRIDE, COLUMN_STRIDE, DYNAMIC_COLUMN))],
            [4], [32])


# Legality boundaries, not a tuning table. Each excluded case exercises a
# distinct requirement of the native complete-window descriptor semantics.
_sync_matrix_descriptor_CASES = [
    ("complete", 64, 64, 0, 0, 1, False, True),
    ("next_window", 128, 64, 64, 0, 1, False, True),
    ("non_power_of_two_extent", 96, 32, 32, 0, 1, False, True),
    ("partial_k", 96, 64, 64, 0, 1, False, False),
    ("unaligned_window", 64, 32, 1, 0, 1, False, False),
    ("unknown_column", 128, 64, 1, 0, 1, True, False),
    ("padded_stride", 64, 64, 0, 128, 1, False, True),
    ("non_unit_column", 64, 64, 0, 128, 2, False, False),
    ("unknown_stride", 64, 64, 0, -1, 1, False, False),
    ("empty_k", 0, 64, 0, 0, 1, False, False),
    ("descriptor_stride_limit", 8192, 64, 0, 0, 1, False, False),
]


def _sync_matrix_descriptor_compile_copy(dtype, roles, case):
    name, k, bk, column, row_stride, column_stride, dynamic, expected = case
    constants = dict(M=73, K=k, BM=128, BK=bk, ROW=-3, COLUMN=column, ROW_STRIDE=row_stride,
                     COLUMN_STRIDE=column_stride, DYNAMIC_COLUMN=dynamic)
    fn = _sync_matrix_descriptor_descriptor_copy_roles if roles else _sync_matrix_descriptor_descriptor_copy
    if roles:
        constants["DEFAULT_ROLE"] = roles != "worker"
    signature = {
        name:
        "constexpr" if name in constants else "*" +
        dtype if name in ["A", "C"] else "i64" if name == "runtime_stride" else "i32"
        for name in fn.arg_names
    }
    source = ASTSource(fn, signature, constants, attrs={(i, ): [("tt.divisibility", 16), ("tt.pointer_range", 32)]
                                                        for i in [0, 1]})
    return triton.compile(source, target=GPUTarget("hip", "gfx936", 64),
                          options=dict(prefer_wide_async_copy=True, num_warps=4, num_stages=1))


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("roles", [False, True, "worker"])
@pytest.mark.parametrize("case", _sync_matrix_descriptor_CASES, ids=[case[0] for case in _sync_matrix_descriptor_CASES])
def test_sync_descriptor_or_masked_fallback(tmp_path, dtype, roles, case):
    kernel = _sync_matrix_descriptor_compile_copy(dtype, roles, case)
    for key, value in kernel.asm.items():
        if isinstance(value, (str, bytes)):
            (tmp_path / ("kernel." + key)).write_bytes(value.encode() if isinstance(value, str) else value)
    assert ("hcu.structured_register" in kernel.asm["ttgir"]) == case[-1]
    assert ("llvm.amdgcn.struct.ptr.buffer.load." in kernel.asm["llir"]) == case[-1]
    assert "hcug.matrix_load_to_local" not in kernel.asm["ttgir"]
    assert "ttg.async_copy_global_to_local" not in kernel.asm["ttgir"]
    assert "tt.matrix_load " not in kernel.asm["ttir"] or "tt.load.async = false" in kernel.asm["ttir"]


# coalesce async copy contiguity
from triton._C.libtriton import ir
try:
    from triton._C.libtriton import hcu
except ImportError:
    pytest.skip("Requires HCU backend", allow_module_level=True)

_coalesce_async_copy_contiguity_BASE = r"""#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#sM = #ttg.slice<{dim = 1, parent = #blocked}>
#sK = #ttg.slice<{dim = 0, parent = #blocked}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx936", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @outer_factor_inner_unknown_stride(%A: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %O: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %U: i32 {tt.divisibility = 8 : i32}, %stride: i32 {tt.divisibility = 16 : i32}) {
    %m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #sM>
    %k = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #sK>
    %m1 = tt.expand_dims %m {axis = 1 : i32} : tensor<128xi32, #sM> -> tensor<128x1xi32, #blocked>
    %k1 = tt.expand_dims %k {axis = 0 : i32} : tensor<32xi32, #sK> -> tensor<1x32xi32, #blocked>
    %mf = tt.broadcast %m1 : tensor<128x1xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %kf = tt.broadcast %k1 : tensor<1x32xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %pitch = arith.constant dense<1024> : tensor<128x32xi32, #blocked>
    %sixteen = arith.constant dense<32> : tensor<128x32xi32, #blocked>
    %eight = arith.constant dense<8> : tensor<128x32xi32, #blocked>
    %zero = arith.constant dense<0> : tensor<128x32xi32, #blocked>
    %stridef = tt.splat %stride : i32 -> tensor<128x32xi32, #blocked>
    %row = arith.muli %mf, %stridef : tensor<128x32xi32, #blocked>
    %v0 = arith.addi %row, %kf : tensor<128x32xi32, #blocked>
    %us = tt.splat %U : i32 -> tensor<128x32xi32, #blocked>
    %sum = arith.addi %us, %v0 : tensor<128x32xi32, #blocked>
    %kb = arith.andi %kf, %eight : tensor<128x32xi32, #blocked>
    %mask = arith.cmpi eq, %kb, %zero : tensor<128x32xi32, #blocked>
    %init = arith.constant dense<-3.000000e+00> : tensor<128x32xbf16, #blocked>
    %dst = ttg.local_alloc %init : (tensor<128x32xbf16, #blocked>) -> !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    ttg.local_barrier
    %t = hcug.buffer_load_to_local %A[%sum] into %dst  : <bf16>[tensor<128x32xi32, #blocked>] -> <128x32xbf16, #shared, #smem, mutable>
    %w = ttg.async_wait %t {num = 0 : i32}
    ttg.local_barrier
    %got = ttg.local_load %dst token %w : !ttg.memdesc<128x32xbf16, #shared, #smem, mutable> -> tensor<128x32xbf16, #blocked>
    %orow = arith.muli %mf, %sixteen : tensor<128x32xi32, #blocked>
    %oi = arith.addi %orow, %kf : tensor<128x32xi32, #blocked>
    %op = tt.splat %O : !tt.ptr<bf16> -> tensor<128x32x!tt.ptr<bf16>, #blocked>
    %optrs = tt.addptr %op, %oi : tensor<128x32x!tt.ptr<bf16>, #blocked>, tensor<128x32xi32, #blocked>
    tt.store %optrs, %got : tensor<128x32x!tt.ptr<bf16>, #blocked>
    ttg.local_dealloc %dst : !ttg.memdesc<128x32xbf16, #shared, #smem, mutable>
    tt.return
  }
}
"""


def _coalesce_async_copy_contiguity_fixture(kind, dtype):
    text = _coalesce_async_copy_contiguity_BASE.replace('bf16', dtype)
    if kind in ['width8', 'width16']:
        width = 8 if kind == 'width8' else 16
        text = text.replace('into %dst  :', 'into %dst {contiguity = ' + str(width) + ' : i32} :')
    elif kind in ['mask_coarse', 'mask_fine']:
        text = text.replace('%A[%sum] into', '%A[%sum] mask = %mask into')
        if kind == 'mask_fine':
            text = text.replace('dense<8>', 'dense<1>')
    elif kind == 'uniform_mask':
        text = text.replace('%stride: i32 {tt.divisibility = 16 : i32})',
                            '%stride: i32 {tt.divisibility = 16 : i32}, %take: i1)')
        text = text.replace(
            '    %t = hcug.buffer_load_to_local',
            '    %uniform_mask = tt.splat %take : i1 -> tensor<128x32xi1, #blocked>\n'
            '    %t = hcug.buffer_load_to_local')
        text = text.replace('%A[%sum] into', '%A[%sum] mask = %uniform_mask into')
    elif kind == 'unknown_offset':
        text = text.replace('%U: i32 {tt.divisibility = 8 : i32}', '%U: i32')
    elif kind == 'unknown_stride':
        text = text.replace('%stride: i32 {tt.divisibility = 16 : i32}', '%stride: i32')
    return text


def _coalesce_async_copy_contiguity_run_pass(tmp_path, text):
    path = tmp_path / 'coalesce-width.mlir'
    path.write_text(text)
    ctx = ir.context()
    ir.load_dialects(ctx)
    hcu.load_dialects(ctx)
    module = ir.parse_mlir_module(str(path), ctx)
    pm = ir.pass_manager(ctx)
    hcu.passes.ttgpuir.add_coalesce_async_copy(pm, 'gfx936')
    pm.run(module, 'coalesce_width')
    first = str(module)
    # Each run follows the real pipeline's fresh manager/context lifecycle.
    second_path = tmp_path / 'coalesce-width-second.mlir'
    second_path.write_text(first)
    second_ctx = ir.context()
    ir.load_dialects(second_ctx)
    hcu.load_dialects(second_ctx)
    second = ir.parse_mlir_module(str(second_path), second_ctx)
    second_pm = ir.pass_manager(second_ctx)
    hcu.passes.ttgpuir.add_coalesce_async_copy(second_pm, 'gfx936')
    second_pm.run(second, 'coalesce_width_idempotence')
    assert str(second) == first
    return first


@pytest.mark.parametrize('dtype', ['bf16', 'f16'])
@pytest.mark.parametrize(
    'kind',
    ['default', 'width8', 'width16', 'uniform_mask', 'mask_coarse', 'mask_fine', 'unknown_offset', 'unknown_stride'])
def test_existing_layout_width_and_legality(tmp_path, kind, dtype):
    text = _coalesce_async_copy_contiguity_fixture(kind, dtype)
    out = _coalesce_async_copy_contiguity_run_pass(tmp_path, text)
    # AxisInfo currently does not prove the bitwise grouped mask's alignment.
    # Preserve that conservative fallback rather than bypassing mask analysis.
    fallback = kind in ['mask_coarse', 'mask_fine', 'unknown_offset', 'unknown_stride']
    assert out.count('hcug.buffer_load_to_local') == int(not fallback)
    assert out.count('hcug.buffer_load ') == int(fallback)
    if not fallback:
        assert 'contiguity = 8 : i32' in out
    assert out.count('ttg.local_barrier') == text.count('ttg.local_barrier')
    assert out.count('ttg.async_wait') == text.count('ttg.async_wait')
