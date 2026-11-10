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
    k = compile_musa(
        _copy, {"desc_ptr": "*i8", "out": "*i8", "BAD_RANK": "constexpr"}, {"BAD_RANK": False}, options={"num_warps": 4}
    )
    assert "byval" not in k.asm["llir"]
    assert "@llvm.musa.tme.ld.tile.2d" in k.asm["llir"]
    assert k.metadata.global_scratch_size == 0


@pytest.mark.parametrize(
    "pointer,bad_rank,match", [("*fp32", False, "scalar global byte pointer"), ("*i8", True, "rank must agree")]
)
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
        tle.gpu.mthreads.device_tensor_descriptor(
            address, [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], tl.int8
        )
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
        compile_musa(
            _invalid_descriptor, {"address": "*i8", "CASE": "constexpr"}, {"CASE": case}, options={"num_warps": 4}
        )
