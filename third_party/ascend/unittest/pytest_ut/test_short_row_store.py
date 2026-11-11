"""Concurrent short-row writes must preserve adjacent programs and GM guards."""
import hashlib

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend
import triton
import triton.language as tl


@triton.jit
def _store(X, Y, ProgramBase, ROWS: tl.constexpr, COLS: tl.constexpr, VALID_ROWS: tl.constexpr, WIDTH: tl.constexpr,
           STRIDE: tl.constexpr, JOIN: tl.constexpr):
    base = (tl.program_id(0) + tl.load(ProgramBase)) * COLS
    r = tl.arange(0, ROWS)[:, None]
    c = base + tl.arange(0, COLS)[None, :]
    offsets = r * STRIDE + c
    raw = tl.load(X + offsets, (r < VALID_ROWS) & (c < WIDTH), other=0.)
    if JOIN:
        a, b = tl.split(tl.reshape(raw, (ROWS * COLS // 2, 2)))
        value = tl.reshape(tl.join(a + 1, b + 1), (ROWS, COLS))
    else:
        value = raw + 1
    tl.store(Y + offsets, value, (r < VALID_ROWS) & (c < WIDTH))


def _check_store(dtype, rows, cols, programs, tail, offset, row_padding, join, late, serial):
    width = programs * cols - tail
    stride, start = width + row_padding, 32 + offset
    count = rows * stride
    cpu = torch.arange(count, dtype=torch.int32).remainder(97).to(dtype)
    source_storage = torch.full((start + count + 32, ), -111, dtype=dtype, device='npu')
    source = source_storage[start:start + count]
    source.copy_(cpu)
    storage = torch.full_like(source_storage, -111)
    output = storage[start:start + count]
    base = torch.zeros(1, dtype=torch.int32, device='npu')
    valid_rows = rows if rows == 1024 else max(1, rows - 3)
    expected = torch.full((count, ), -111, dtype=dtype).reshape(rows, stride)
    expected[:valid_rows, :width] = cpu.reshape(rows, stride)[:valid_rows, :width] + 1
    options = dict(num_warps=8, num_stages=1, multibuffer=False, structured_address_hoisting=late,
                   structured_store_sync_optimization=late)
    args = (source, output, base, rows, cols, valid_rows, width, stride, join)
    binary = None
    # Repeated clears and launches make a nondeterministic cross-program race
    # observable. Both dispatch policies are exercised by the test runner.
    for _ in range(5):
        storage.fill_(-111)
        kernel = _store[(programs, )](*args, **options)
        torch.npu.synchronize()
        digest = hashlib.sha256(kernel.asm['npubin']).hexdigest()
        binary = digest if binary is None else binary
        assert binary == digest
        actual = storage.cpu()
        assert torch.equal(actual[start:start + count].reshape(rows, stride), expected)
        assert torch.all(actual[:start] == -111) and torch.all(actual[start + count:] == -111)
    if serial:
        storage.fill_(-111)
        for program in range(programs):
            base.fill_(program)
            kernel = _store[(1, )](*args, **options)
            assert hashlib.sha256(kernel.asm['npubin']).hexdigest() == binary
        torch.npu.synchronize()
        actual = storage.cpu()
        assert torch.equal(actual[start:start + count].reshape(rows, stride), expected)
        assert torch.all(actual[:start] == -111) and torch.all(actual[start + count:] == -111)
    torch.npu.synchronize()
    assert torch.equal(source.cpu(), cpu)
    got_source = source_storage.cpu()
    assert torch.all(got_source[:start] == -111) and torch.all(got_source[start + count:] == -111)


@pytest.mark.parametrize('late', [False, True])
@pytest.mark.parametrize('join', [False, True])
def test_parallel_serial_same_binary(late, join):
    _check_store(torch.float32, 1024, 4, 256, 0, 0, 0, join, late, True)


@pytest.mark.parametrize('dtype,cols', [
    (torch.float16, 2),
    (torch.float16, 4),
    (torch.float16, 8),
    (torch.bfloat16, 4),
    (torch.float32, 2),
    (torch.float32, 4),
    (torch.int16, 4),
    (torch.int32, 4),
])
@pytest.mark.parametrize('programs,tail,offset,row_padding', [
    (1, 0, 0, 8),
    (2, 1, 1, 7),
    (39, 0, 4, 3),
    (40, 1, 7, 1),
    (41, 0, 1, 0),
    (257, 1, 7, 13),
])
def test_parallel_short_rows_and_guards(dtype, cols, programs, tail, offset, row_padding):
    _check_store(dtype, 32, cols, programs, tail, offset, row_padding, True, True, False)


@triton.jit
def _block_store(X, Y, STRIDE: tl.constexpr):
    base = tl.program_id(0) * 4
    src = tl.make_block_ptr(X, shape=(29, 163), strides=(STRIDE, 1), offsets=(0, base), block_shape=(32, 4),
                            order=(1, 0))
    raw = tl.load(src, boundary_check=(0, 1), padding_option='zero')
    a, b = tl.split(tl.reshape(raw, (64, 2)))
    value = tl.reshape(tl.join(a + 1, b + 1), (32, 4))
    dst = tl.make_block_ptr(Y, shape=(29, 163), strides=(STRIDE, 1), offsets=(0, base), block_shape=(32, 4),
                            order=(1, 0))
    tl.store(dst, value, boundary_check=(0, 1))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_boundary_checked_short_row_store(dtype):
    stride, start = 176, 39
    cpu = torch.arange(29 * stride, dtype=torch.int32).remainder(97).to(dtype).reshape(29, stride)
    source = cpu.flatten().npu()
    storage = torch.full((start + cpu.numel() + 32, ), -111, dtype=dtype, device='npu')
    output = storage[start:start + cpu.numel()]
    _block_store[(41, )](source, output, stride, num_warps=8, num_stages=1, multibuffer=False)
    torch.npu.synchronize()
    expected = torch.full_like(cpu, -111)
    expected[:, :163] = cpu[:, :163] + 1
    actual = storage.cpu()
    assert torch.equal(actual[start:start + cpu.numel()].reshape_as(expected), expected)
    assert torch.all(actual[:start] == -111) and torch.all(actual[start + cpu.numel():] == -111)


@triton.jit
def _unmasked_view(X, Y, JOIN: tl.constexpr):
    program = tl.program_id(0)
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 4)[None, :]
    raw = tl.load(X + program * 32 * 37 + r * 37 + c)
    if JOIN:
        a, b = tl.split(tl.reshape(raw, (64, 2)))
        value = tl.reshape(tl.join(a + 1, b + 1), (128, ))
    else:
        value = tl.reshape(raw, (128, )) + 1
    i = tl.arange(0, 128)
    tl.store(Y + program * 128 + i, value)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.int16, torch.int32])
@pytest.mark.parametrize('join', [False, True])
def test_unmasked_strided_load_packed_view(dtype, join):
    batch, start, size = 41, 39, 41 * 128
    cpu = torch.arange(batch * 32 * 37, dtype=torch.int32).remainder(97).to(dtype)
    source = cpu.npu()
    storage = torch.full((start + size + 32, ), -111, dtype=dtype, device='npu')
    compiled = _unmasked_view[(batch, )](source, storage[start:start + size], join, num_warps=8, num_stages=1,
                                         multibuffer=False)
    torch.npu.synchronize()
    expected = cpu.reshape(batch, 32, 37)[:, :, :4].contiguous().flatten() + 1
    actual = storage.cpu()
    assert torch.equal(actual[start:start + size], expected)
    assert torch.all(actual[:start] == -111) and torch.all(actual[start + size:] == -111)
    assert torch.equal(source.cpu(), cpu)
    if not join:
        # The standalone load primitive needs neither split/join nor a TLE import.
        assert 'custom_flagtree_short_row_load_' in str(compiled.asm['ttadapter'])
        assert 'custom_flagtree_split_' not in str(compiled.asm['ttadapter'])


@triton.jit
def _load_layout_probe(X, Y):
    _unmasked_view(X, Y, True)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_unmasked_load_physical_row_alignment(dtype, tmp_path, monkeypatch):
    # This must check CANN's final physical descriptors. Early padded allocs can
    # disappear while bufferization folds a reshape into the load destination.
    monkeypatch.setenv('FLAGTREE_PERMUTE_DUMP_DIR', str(tmp_path))
    cpu = torch.arange(32 * 37, dtype=torch.int32).remainder(97).to(dtype)
    source = cpu.npu()
    output = torch.full((128, ), -111, dtype=dtype, device='npu')
    with (compilation := triton.knobs.compilation).scope():
        compilation.always_compile = True
        _load_layout_probe[(1, )](source, output, num_warps=8, num_stages=1, multibuffer=False)
    torch.npu.synchronize()
    assert torch.equal(output.cpu(), cpu.reshape(32, 37)[:, :4].contiguous().flatten() + 1)
    dumps = list(tmp_path.glob('*/after.mlir'))
    assert dumps, 'Expected a fresh physical IR dump'
    text = '\n'.join(path.read_text() for path in dumps)
    # Native CANN loads into compact short rows must not reappear. The private
    # load explicitly performs aligned DMA followed by vector compaction.
    suffix = 'half' if dtype == torch.float16 else 'float'
    assert f'call @custom_flagtree_short_row_load_{suffix}' in text
    assert f'call @load_gm_to_ubuf_2d_{suffix}' not in text
