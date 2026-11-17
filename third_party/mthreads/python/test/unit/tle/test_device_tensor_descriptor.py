"""Device-resident TME descriptors use the ordinary pointer launch ABI."""

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from test_tle_utils import compile_musa, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _copy(desc_ptr, out, BAD_RANK: tl.constexpr):
    strides: tl.constexpr = [128] if BAD_RANK else [128, 1]
    desc = tle.gpu.mthreads.device_tensor_descriptor(desc_ptr, [128, 128], strides, [128, 128], tl.int8)
    buffer = tle.gpu.alloc((1, 128, 128), tl.int8)
    pipe = tle.pipe(capacity=1, tile=buffer)
    writer, reader = pipe.writer(), pipe.reader()
    slot = writer.acquire(0)
    tle.gpu.copy(desc, slot.tile, (128, 128), (0, 0))
    writer.commit(0)
    ready = reader.wait(0)
    result = ready.slot.tile.load()
    reader.release(0)
    i, j = tl.arange(0, 128), tl.arange(0, 128)
    tl.store(out + i[:, None] * 128 + j[None, :], result)


def test_device_descriptor_has_no_byval_or_descriptor_allocation():
    k = compile_musa(_copy, {"desc_ptr": "*i8", "out": "*i8", "BAD_RANK": "constexpr"}, {"BAD_RANK": False},
                     options={"num_warps": 4})
    assert "byval" not in k.asm["llir"]
    assert "@llvm.musa.tme.ld.tile.2d" in k.asm["llir"]
    assert k.metadata.global_scratch_size == 0


@pytest.mark.parametrize("pointer,bad_rank,match", [("*fp32", False, "scalar global byte pointer"),
                                                    ("*i8", True, "rank must agree")])
def test_device_descriptor_rejects_invalid_metadata(pointer, bad_rank, match):
    with pytest.raises(Exception, match=match):
        compile_musa(
            _copy,
            {"desc_ptr": pointer, "out": "*i8", "BAD_RANK": "constexpr"},
            {"BAD_RANK": bad_rank},
            options={"num_warps": 4},
        )


@triton.jit
def _invalid_descriptor(address, CASE: tl.constexpr):
    if CASE == 0:
        tle.gpu.mthreads.device_tensor_descriptor(address, [], [], [], tl.int8)
    elif CASE == 1:
        tle.gpu.mthreads.device_tensor_descriptor(address, [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
                                                  tl.int8)
    elif CASE == 2:
        tle.gpu.mthreads.device_tensor_descriptor(address, [128], [1], [0], tl.int8)
    elif CASE == 3:
        shared_address = address.to(tl.pointer_type(tl.int8, 3))
        tle.gpu.mthreads.device_tensor_descriptor(shared_address, [128], [1], [128], tl.int8)
    else:
        addresses = address + tl.arange(0, 4)
        tle.gpu.mthreads.device_tensor_descriptor(addresses, [128], [1], [128], tl.int8)


@pytest.mark.parametrize(
    "case,diagnostic",
    [
        (0, "rank must agree"),
        (1, "rank must agree"),
        (2, "block dimensions must be positive"),
        (3, "scalar global byte pointer"),
        (4, "scalar global byte pointer"),
    ],
)
def test_invalid_descriptor_rank_extent_and_pointer_shape(case, diagnostic):
    with pytest.raises(Exception, match=diagnostic):
        compile_musa(_invalid_descriptor, {"address": "*i8", "CASE": "constexpr"}, {"CASE": case},
                     options={"num_warps": 4})


@triton.jit
def _copy_device_tile(desc_ptr, out, row, col_block):
    desc = tle.gpu.mthreads.device_tensor_descriptor(desc_ptr, [256, 384], [384, 1], [128, 128], tl.int8)
    buffer = tle.gpu.alloc((1, 128, 128), tl.int8)
    pipe = tle.pipe(capacity=1, tile=buffer)
    writer, reader = pipe.writer(), pipe.reader()
    slot = writer.acquire(0)
    tle.gpu.copy(desc, slot.tile, (128, 128), (row, col_block * 32))
    writer.commit(0)
    ready = reader.wait(0)
    result = ready.slot.tile.load()
    reader.release(0)
    i, j = tl.arange(0, 128), tl.arange(0, 128)
    tl.store(out + i[:, None] * 128 + j[None, :], result)


def test_device_descriptor_tme_switches_payloads():
    import ctypes

    import torch
    from triton.backends.mthreads.driver import _musa_tme_dtype_enum

    torch_musa = pytest.importorskip("torch_musa")
    if not torch_musa.is_available():
        pytest.skip("MUSA device is not available")

    generator = torch.Generator().manual_seed(1607)
    shape, block_shape = (256, 384), (128, 128)
    host_payloads = [torch.randint(-128, 128, shape, dtype=torch.int8, generator=generator) for _ in range(2)]
    payloads = [p.to("musa") for p in host_payloads]
    encode = triton.runtime.driver.active.utils.fill_tme_descriptor
    encoded = [
        encode(p.data_ptr(), shape, block_shape, p.element_size(), _musa_tme_dtype_enum("i8"), 0) for p in payloads
    ]
    descriptor_bytes = 64
    # Snapshot the vendor-encoded descriptor while the host object is alive.
    host_descriptors = [
        torch.frombuffer(bytearray(ctypes.string_at(d.tme_desc_cpu_ptr(), descriptor_bytes)), dtype=torch.uint8)
        for d in encoded
    ]
    arena = torch.empty(2 * descriptor_bytes, dtype=torch.uint8, device="musa")
    slots = arena[:descriptor_bytes], arena[descriptor_bytes:]
    assert all(p.data_ptr() % descriptor_bytes == 0 for p in slots)

    # Distinct descriptor address/payload, then update slot 0 only after its
    # prior consumer on this stream. No descriptor update is concurrent with
    # a consuming launch; payloads, descriptors and staging bytes stay alive.
    cases = [(0, 0, 32, 64), (1, 1, 96, 192), (0, 1, 32, 64), (1, 1, 0, 0)]
    expected = [host_payloads[src][row:row + 128, col:col + 128] for _, src, row, col in cases]
    assert not torch.equal(expected[0], expected[2])
    assert not torch.equal(expected[1], expected[3])
    outputs = [torch.empty(block_shape, dtype=torch.int8, device="musa") for _ in cases]
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    kernels = []
    try:
        with torch.musa.stream(stream):
            slots[0].copy_(host_descriptors[0], non_blocking=True)
            slots[1].copy_(host_descriptors[1], non_blocking=True)
            for step, ((slot_index, src, row, col), out) in enumerate(zip(cases, outputs)):
                if step == 2:
                    slots[slot_index].copy_(host_descriptors[src], non_blocking=True)
                kernels.append(_copy_device_tile[(1, )](slots[slot_index], out, row, col // 32, num_warps=4))
    finally:
        stream.synchronize()

    assert all("@llvm.musa.tme.ld.tile.2d" in kernel.asm["llir"] for kernel in kernels)
    for out, reference in zip(outputs, expected):
        torch.testing.assert_close(out.cpu(), reference, atol=0, rtol=0)
