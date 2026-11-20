"""Direct UB descriptor tests. Set FLAGTREE_UB_CONTRACT_LIBRARY explicitly.

These call the packaged implementation with controlled *physical* UB layouts.
They do not infer UB strides from HBM views. Guard checks cover whole buffers.
"""
import ctypes
import os
from pathlib import Path

import pytest
import torch
import torch_npu

STORAGE = 512
LAYOUTS = {
    'packed': (1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
    'strided_source': (3, 1, 1, 0, 1, 0, 1, 0, 1, 0),
    'strided_destinations': (1, 0, 2, 1, 3, 1, 1, 0, 1, 0),
    'strided_indices': (1, 0, 1, 0, 1, 0, 2, 1, 1, 0),
    'strided_scratch': (3, 1, 1, 0, 1, 0, 1, 0, 2, 1),
    'unaligned_source': (1, 1, 1, 0, 1, 0, 1, 0, 1, 0),
    'unaligned_destination': (1, 0, 1, 1, 1, 1, 1, 0, 1, 0),
    'unaligned_indices': (1, 0, 1, 0, 1, 0, 1, 1, 1, 0),
    'unaligned_scratch': (3, 1, 1, 0, 1, 0, 1, 0, 1, 1),
}


@pytest.fixture(scope='module')
def launch():
    torch.npu.set_device(0)
    path = Path(os.environ['FLAGTREE_UB_CONTRACT_LIBRARY']).resolve()
    assert path.is_file(), path
    library = ctypes.CDLL(str(path))
    call = library.launch_ub_contract
    call.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_uint32] * 2
    call.restype = ctypes.c_uint32

    def run(x, indices, a, b, cfg, mode):
        rc = call(torch.npu.current_stream().npu_stream, x.data_ptr(), indices.data_ptr(), a.data_ptr(), b.data_ptr(),
                  cfg.data_ptr(), mode, x.element_size())
        assert rc == 0, rc
        torch.npu.synchronize()

    return run


@pytest.mark.parametrize('layout', list(LAYOUTS))
@pytest.mark.parametrize('dtype', [torch.int16, torch.int32])
@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('n', [1, 7, 8, 9, 15, 16, 17, 64])
def test_word_descriptors(launch, layout, dtype, mode, n):
    ss, so, ds, do, bs, bo, ins, ino, ts, to = LAYOUTS[layout]
    cpu = (torch.arange(STORAGE, dtype=torch.int32) * 977 - 31001).to(dtype)
    indices = torch.zeros(STORAGE, dtype=torch.int32)
    logical_indices = (torch.arange(n) * 13 + 7).remainder(2 * n)
    indices[ino:ino + n * ins:ins] = logical_indices
    x, idx = cpu.npu(), indices.npu()
    a, b = [torch.full((STORAGE, ), -91, dtype=dtype, device='npu') for _ in range(2)]
    cfg = torch.tensor([n, 3, *LAYOUTS[layout]], dtype=torch.int32).npu()
    launch(x, idx, a, b, cfg, mode)
    ea, eb = [torch.full((STORAGE, ), -91, dtype=dtype) for _ in range(2)]
    if mode == 0:
        ea[do:do + n * ds:ds] = cpu[so + logical_indices * ss]
    else:
        ea[do:do + n * ds:ds] = cpu[so:so + 2 * n * ss:2 * ss]
        eb[bo:bo + n * bs:bs] = cpu[so + ss:so + (2 * n + 1) * ss:2 * ss]
    assert torch.equal(a.cpu(), ea), (mode, layout, n, a.cpu(), ea)
    assert torch.equal(b.cpu(), eb), (mode, layout, n, b.cpu(), eb)


@pytest.mark.parametrize('layout', [
    'packed', 'strided_source', 'strided_destinations', 'unaligned_source', 'unaligned_destination', 'unaligned_scratch'
])
@pytest.mark.parametrize('n', [1, 15, 16, 17, 64])
@pytest.mark.parametrize('mode', [2, 3])
def test_byte_storage_descriptors(launch, layout, n, mode):
    ss, so, ds, do, *_ = LAYOUTS[layout]
    src_dtype, dst_dtype = (torch.int8, torch.int16) if mode == 2 else (torch.int16, torch.int8)
    cpu = (torch.arange(STORAGE, dtype=torch.int32) * 37 + 211).to(torch.int8).to(src_dtype)
    x = cpu.npu()
    idx = torch.zeros(STORAGE, dtype=torch.int32, device='npu')
    a, b = [torch.full((STORAGE, ), -91, dtype=dst_dtype, device='npu') for _ in range(2)]
    cfg = torch.tensor([n, 3, *LAYOUTS[layout]], dtype=torch.int32).npu()
    launch(x, idx, a, b, cfg, mode)
    expected = torch.full((STORAGE, ), -91, dtype=dst_dtype)
    expected[do:do + n * ds:ds] = cpu[so:so + n * ss:ss].to(dst_dtype)
    assert torch.equal(a.cpu(), expected), (mode, layout, n)


@pytest.fixture(scope='module')
def launch_store():
    torch.npu.set_device(0)
    library = ctypes.CDLL(str(Path(os.environ['FLAGTREE_UB_CONTRACT_LIBRARY']).resolve()))
    call = library.launch_short_row_contract
    call.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint32] * 2
    call.restype = ctypes.c_uint32

    def run(x, y, debug, config, programs):
        rc = call(torch.npu.current_stream().npu_stream, x.data_ptr(), y.data_ptr(), debug.data_ptr(),
                  config.data_ptr(), programs, x.element_size())
        assert rc == 0, rc
        torch.npu.synchronize()

    return run


@pytest.mark.parametrize('dtype', [torch.int16, torch.int32])
@pytest.mark.parametrize('programs', [1, 41])
@pytest.mark.parametrize('rows', [0, 1, 7, 63, 64, 65, 127, 128, 129, 1024])
@pytest.mark.parametrize('cols,ss,cs,so,padding,offset,tail', [
    (4, 4, 1, 0, 0, 0, 0),
    (4, 5, 1, 1, 3, 1, 1),
    (4, 8, 1, 0, 0, 4, 0),
    (4, 8, 1, 1, 7, 7, 0),
    (4, 12, 2, 1, 13, 1, 1),
    (7, 9, 1, 1, 3, 7, 1),
])
def test_short_row_store_physical_layouts(launch_store, dtype, programs, rows, cols, ss, cs, so, padding, offset, tail):
    input_size, work_size = 16384, 1056
    cpu = (torch.arange(programs * input_size, dtype=torch.int32) * 977 - 31001).to(dtype)
    source = cpu.npu()
    stride, start = programs * cols - tail + padding, 32 + offset
    storage = torch.full((start + max(1, rows) * stride + 32, ), -91, dtype=dtype, device='npu')
    debug = torch.full((programs * 2 * work_size, ), -999, dtype=torch.int32, device='npu')
    config = torch.tensor([rows, cols, ss, cs, so, stride, start, programs, tail, 8, 16], dtype=torch.int32).npu()
    expected = torch.full_like(storage.cpu(), -91)
    for program in range(programs):
        active = cols - (tail if program == programs - 1 else 0)
        if rows:
            ri, ci = torch.arange(rows)[:, None], torch.arange(active)[None, :]
            expected[start + ri * stride + program * cols + ci] = cpu[program * input_size + so + ri * ss + ci * cs]
    for _ in range(3):
        storage.fill_(-91)
        launch_store(source, storage, debug, config, programs)
        assert torch.equal(storage.cpu(), expected)
        physical = debug.cpu().reshape(programs, 2, work_size)
        for program in range(programs):
            active = cols - (tail if program == programs - 1 else 0)
            n = min(rows * (32 // cpu.element_size()), 1024) if active else 0
            # The aligned/dense fast path writes neither workspace.
            native = (so * cpu.element_size()) % 32 == 0 and ((ss * cpu.element_size()) % 32 == 0 or
                                                              (ss == active and stride == active)) and cs == 1
            if native:
                n = 0
            for which, begin in enumerate([8, 16]):
                assert torch.all(physical[program, which, :begin] == -177)
                assert torch.all(physical[program, which, begin + n:] == -177)
    assert torch.equal(source.cpu(), cpu)


@pytest.fixture(scope='module')
def launch_load():
    torch.npu.set_device(0)
    library = ctypes.CDLL(str(Path(os.environ['FLAGTREE_UB_CONTRACT_LIBRARY']).resolve()))
    call = library.launch_short_row_load_contract
    call.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint32] * 2
    call.restype = ctypes.c_uint32

    def run(x, y, debug, config, programs):
        rc = call(torch.npu.current_stream().npu_stream, x.data_ptr(), y.data_ptr(), debug.data_ptr(),
                  config.data_ptr(), programs, x.element_size())
        assert rc == 0, rc
        torch.npu.synchronize()

    return run


@pytest.mark.parametrize('dtype', [torch.int16, torch.int32])
@pytest.mark.parametrize('programs', [1, 41])
@pytest.mark.parametrize('rows', [0, 2, 7, 63, 64, 65, 127, 128, 129, 511])
@pytest.mark.parametrize('cols,pitch,padding,offset,tail', [
    (4, 0, 0, 0, 0),
    (4, 1, 3, 1, 1),
    (7, 2, 7, 7, 1),
    (1, 0, 7, 4, 1),
])
def test_short_row_load_physical_layouts(launch_load, dtype, programs, rows, cols, pitch, padding, offset, tail):
    input_size, work_size, block = 16384, 1056, 32 // torch.empty((), dtype=dtype).element_size()
    stride, start = programs * cols - tail + padding, 32 + offset
    cpu = (torch.arange(start + max(rows, 1) * stride + 32, dtype=torch.int32) * 977 - 31001).to(dtype)
    source = cpu.npu()
    output = torch.full((programs * input_size, ), -91, dtype=dtype, device='npu')
    debug = torch.full((programs * 2 * work_size, ), -999, dtype=torch.int32, device='npu')
    dst_stride, dst_offset = (cols if pitch == 0 else pitch * block), block
    config = torch.tensor([rows, cols, dst_stride, 1, dst_offset, stride, start, programs, tail, 8, 16],
                          dtype=torch.int32).npu()
    expected = torch.full((programs, input_size), -91, dtype=dtype)
    for program in range(programs):
        active = cols - (tail if program == programs - 1 else 0)
        if rows and active:
            ri, ci = torch.arange(rows)[:, None], torch.arange(active)[None, :]
            if dst_stride >= block and dst_stride % block == 0:
                # DMA owns the allocated row padding, which is explicitly zeroed.
                expected[program, dst_offset + ri * dst_stride + torch.arange(block)] = 0
            expected[program, dst_offset + ri * dst_stride + ci] = cpu[start + ri * stride + program * cols + ci]
    for _ in range(3):
        output.fill_(-91)
        launch_load(source, output, debug, config, programs)
        assert torch.equal(output.cpu().reshape_as(expected), expected)
        physical = debug.cpu().reshape(programs, 2, work_size)
        assert torch.all(physical[:, 0] == -177)  # the unused address workspace
        assert torch.all(physical[:, 1, :16] == -177)
        assert torch.all(physical[:, 1, 1040:] == -177)
    assert torch.equal(source.cpu(), cpu)
