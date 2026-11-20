"""Matrix G2L is derived from ordinary SharedLinear memory, not packet IR."""
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
def shared_layout(transpose, bad):
    bases = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]]
    if bad: bases[0], bases[3] = bases[3], bases[0]
    if transpose: bases = [[b[1], b[0]] for b in bases]
    return gl.SharedLinearLayout(bases)


@g.jit
def _copy(A, O, ROW, COL, ENABLE, TRANSPOSE: gl.constexpr, BAD_CHART: gl.constexpr, PITCH: gl.constexpr,
          COLUMNS: gl.constexpr):
    shared: gl.constexpr = shared_layout(TRANSPOSE, BAD_CHART)
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


def compile_copy(dtype="bf16", transpose=False, bad=False, pitch=80, columns=72):
    source = GluonASTSource(
        _copy, {
            "A": "*" + dtype, "O": "*" + dtype, "ROW": "i32", "COL": "i32", "ENABLE": "i32", "TRANSPOSE": "constexpr",
            "BAD_CHART": "constexpr", "PITCH": "constexpr", "COLUMNS": "constexpr"
        }, {"TRANSPOSE": transpose, "BAD_CHART": bad, "PITCH": pitch, "COLUMNS": columns},
        attrs={(0, ): [("tt.divisibility", 16)], (1, ): [("tt.divisibility", 16)]})
    return triton.compile(source, target=GPUTarget("hip", "gfx936", 64), options=dict(num_warps=4, num_stages=1))


@pytest.mark.parametrize("pitch", [80, 8192, 1 << 24])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("transpose", [False, True])
def test_plain_shared_matrix_copy(dtype, transpose, pitch):
    kernel = compile_copy(dtype, transpose, pitch=pitch)
    ir = kernel.asm['ttgir']
    assert "matrix_load_to_local" in ir and "shared_linear" in ir
    assert "hcu.packet" not in ir and "mls_shared" not in ir
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
        compile_copy(**kwargs)
    diagnostic = capfd.readouterr().err
    assert "matrix shared copy" in diagnostic and reason in diagnostic


@pytest.mark.parametrize("pitch", [72, 88, 256, 272, 768, 1280, 3072, 4096, 6144, 7936, 8176, 8184, 8192, 8200])
def test_ordinary_stride_field_and_raw_fallback(pitch):
    # Exercise the real ordinary lowering, including valid pitches below256
    # and pitches divisible by8 but not16; the retired PacketDomain is stricter
    # than this target ABI and must not constrain ordinary copies.
    kernel = compile_copy(pitch=pitch)
    structured = "llvm.amdgcn.struct.ptr.buffer.load.lds" in kernel.asm["llir"]
    assert structured == (2 * pitch < 0x4000)
    assert "llvm.amdgcn.raw.ptr.buffer.load.lds" in kernel.asm["llir"] or structured
    assert "shared_linear" in kernel.asm["ttgir"]


@g.jit
def _copy_known_origins(A, O, ENABLE, ROW: gl.constexpr, COL: gl.constexpr, TRANSPOSE: gl.constexpr,
                        PITCH: gl.constexpr):
    _copy(A, O, ROW, COL, ENABLE, TRANSPOSE, False, PITCH, 72)


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
        _copy_known_origins,
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
        _copy_known_origins[(1, )](source, output, enabled, row, col, transpose, 80, num_warps=4, num_stages=1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(4):
                _copy_known_origins[(1, )](source, output, enabled, row, col, transpose, 80, num_warps=4, num_stages=1)
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
def _copy_unsigned_origins(A, O, INDEX, ENABLE, MOD: gl.constexpr, DIV: gl.constexpr, STEP: gl.constexpr,
                           BIAS: gl.constexpr, TRANSPOSE: gl.constexpr):
    row = (((INDEX.to(gl.uint32) % MOD) // DIV).to(gl.int32) * STEP + BIAS)
    _copy(A, O, row, 0, ENABLE, TRANSPOSE, False, 80, 72)


# The input may have any i32 bit pattern. Prove bounds from the unsigned
# operations themselves, including high-bit divisors, rather than a launch grid.
_UNSIGNED_ORIGIN_CASES = [
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


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("modulus,divisor,step,bias,complete", _UNSIGNED_ORIGIN_CASES)
def test_unsigned_origin_range_selects_raw_without_launch_assumptions(dtype, transpose, modulus, divisor, step, bias,
                                                                      complete):
    constants = dict(MOD=modulus, DIV=divisor, STEP=step, BIAS=bias, TRANSPOSE=transpose)
    signature = dict(A="*" + dtype, O="*" + dtype, INDEX="i32", ENABLE="i32",
                     **{name: "constexpr"
                        for name in constants})
    source = GluonASTSource(
        _copy_unsigned_origins,
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
    for modulus, divisor, step, bias, _ in _UNSIGNED_ORIGIN_CASES:
        for index in (-2147483648, -1, 0, 2147483647):
            quotient = ((index & 0xffffffff) % modulus) // divisor
            row = ((quotient * step + bias + (1 << 31)) % (1 << 32)) - (1 << 31)
            for enabled in (0, 1):

                def launch():
                    _copy_unsigned_origins[(1, )](source, output, index, enabled, modulus, divisor, step, bias,
                                                  transpose, num_warps=4, num_stages=1)

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
def _cohort_shared_layout(tile_columns, transpose):
    tile_rows = 2048 // tile_columns
    bases = [[0, 1 << b] for b in range(tile_columns.bit_length() - 1)]
    bases += [[1 << b, 0] for b in range(tile_rows.bit_length() - 1)]
    if transpose:
        bases = [[y, x] for x, y in bases]
    return gl.SharedLinearLayout(bases, alignment=16)


@g.jit
def _copy_row_cohort(A, O, ROW, COL, ENABLE, TILE_COLUMNS: gl.constexpr, ROWS: gl.constexpr, COLUMNS: gl.constexpr,
                     PITCH: gl.constexpr, RSTEP: gl.constexpr, CSTEP: gl.constexpr, TRANSPOSE: gl.constexpr):
    tile_rows: gl.constexpr = 2048 // TILE_COLUMNS
    shared: gl.constexpr = _cohort_shared_layout(TILE_COLUMNS, TRANSPOSE)
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
    source = GluonASTSource(_copy_row_cohort, signature, constants, attrs={(0, ): [("tt.divisibility", 16)], (1, ):
                                                                           [("tt.divisibility", 16)]})
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
                    _copy_row_cohort[(1, )](source, output, ri, ci, enabled, tile_columns, rows, columns, pitch, rstep,
                                            cstep, transpose, num_warps=4, num_stages=1)

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
        _copy_known_origins,
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


def _matrix_copy_with_control(source, control):
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
    source = compile_copy(dtype=dtype, transpose=transpose).asm["ttgir"]
    path = tmp_path / "control.ttgir"
    path.write_text(_matrix_copy_with_control(source, control))
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
    source = compile_copy(dtype=dtype, transpose=transpose).asm["ttgir"]
    lines = source.splitlines(keepends=True)
    first = next(i for i, line in enumerate(lines) if "tt.func public" in line)
    last = next(i for i in range(first + 1, len(lines)) if lines[i].startswith("  }"))
    helper = lines[first].replace("public @_copy", "private @helper").replace("noinline = false", "noinline = true")
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
    assert "matrix shared copy: unproven uniform origins" in capfd.readouterr().err
