# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Per-copy transaction contributions preserve one shared full epoch."""
import inspect
import re
from types import SimpleNamespace

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._filecheck import run_parser
from triton.backends.compiler import GPUTarget
from triton.experimental.tle.language.gpu import core, types
from triton._flagtree_backend import FLAGTREE_BACKEND

pytestmark = pytest.mark.skipif(
    bool(FLAGTREE_BACKEND) or core.mthreads_common.enabled() or core.iluvatar_copy.enabled()
    or core.tle_semantic.COMMON_IR_ENABLED, reason="NVIDIA per-copy TMA completion API")


@triton.jit
def _joint_full(RAW: tl.constexpr, EXPLICIT_W: tl.constexpr):
    raw = tl.full((), 0, tl.uint64).to(tl.pointer_type(tl.uint8))
    base = raw.to(tl.pointer_type(tl.bfloat16))
    w = tle.gpu.alloc([64, 256], tl.bfloat16, nv_mma_shared_layout=False, alignment_bytes=128)
    x = tle.gpu.alloc([1, 64], tl.bfloat16, nv_mma_shared_layout=False, alignment_bytes=128)
    if RAW:
        wd = tle.gpu.reinterpret_tensor_map(raw, w)
        xd = tle.gpu.reinterpret_tensor_map(raw, x)
    else:
        wd = tl.make_tensor_descriptor(base, [2048, 12288], [12288, 1], [64, 256])
        xd = tl.make_tensor_descriptor(base, [1, 2048], [2048, 1], [1, 64])
    full = tle.gpu.alloc_barrier(arrive_count=2, expect_bytes=32768)
    if EXPLICIT_W:
        tle.gpu.copy(wd, w, [64, 256], [0, 0], barrier=full, expect_bytes=32768)
    else:
        tle.gpu.copy(wd, w, [64, 256], [0, 0], barrier=full)
    tle.gpu.copy(xd, x, [1, 64], [0, 0], barrier=full, expect_bytes=128)
    tle.gpu.barrier_wait(full, phaseIdx=0)


@triton.jit
def _legacy_default(EXPLICIT_NONE: tl.constexpr):
    raw = tl.full((), 0, tl.uint64).to(tl.pointer_type(tl.uint8))
    dst = tle.gpu.alloc([1, 64], tl.bfloat16, nv_mma_shared_layout=False, alignment_bytes=128)
    desc = tle.gpu.reinterpret_tensor_map(raw, dst)
    full = tle.gpu.alloc_barrier(expect_bytes=128)
    if EXPLICIT_NONE:
        tle.gpu.copy(desc, dst, [1, 64], [0, 0], barrier=full, expect_bytes=None)
    else:
        tle.gpu.copy(desc, dst, [1, 64], [0, 0], barrier=full)


@pytest.mark.skip_if_tle_unsupported("gpu.reinterpret_tensor_map")
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("explicit_w", [False, True])
def test_joint_full_has_two_exact_contributions(raw, explicit_w):
    text = run_parser(_joint_full, kwargs={"RAW": raw, "EXPLICIT_W": explicit_w, "num_warps": 8},
                      target=GPUTarget("cuda", 90, 32)).str_nodebug()
    assert text.count("tle.barrier.alloc") == 1
    assert "arrive_count = 2 : i32" in text
    copies = [line for line in text.splitlines() if "ttg.tma_copy " in line]
    assert len(copies) == 2
    assert "expect_bytes = 32768 : i32" in copies[0]
    assert "expect_bytes = 128 : i32" in copies[1]
    roots = set(re.findall(r"ttg.memdesc_index (%[\w]+)\[", text))
    assert len(roots) == 1, "Both contributions and the single wait must use the same allocated full"
    assert text.count("tle.barrier.wait") == 1


@pytest.mark.skip_if_tle_unsupported("gpu.reinterpret_tensor_map")
def test_none_preserves_legacy_default_and_positional_order():
    outputs = [run_parser(_legacy_default, kwargs={"EXPLICIT_NONE": value, "num_warps": 8},
                          target=GPUTarget("cuda", 90, 32)).str_nodebug() for value in [False, True]]
    assert outputs[0] == outputs[1]
    names = list(inspect.signature(core.copy).parameters)
    assert names[:11] == ["src", "dst", "shape", "offsets", "barrier", "mask", "cache_policy",
                          "other", "is_async", "eviction_policy", "input_stride"]
    assert names[11:] == ["expect_bytes", "_semantic"]
    assert inspect.signature(core.copy).parameters["expect_bytes"].default is None


def _guard_values(dtype=tl.bfloat16, dst_dtype=None, storage=types.smem):
    # Complete typed Python values suffice for errors that must precede any IR mutation.
    desc = object.__new__(tl.tensor_descriptor)
    desc.type = SimpleNamespace(block_type=tl.block_type(dtype, [1, 64]))
    dst = object.__new__(types.buffered_tensor)
    dst.shape = [1, 64]
    dst.dtype = dtype if dst_dtype is None else dst_dtype
    dst.type = SimpleNamespace(storage=storage)
    def forbidden(*args, **kwargs):
        raise AssertionError("Rejected overrides must not reach the builder")
    semantic = SimpleNamespace(builder=SimpleNamespace(create_tma_copy=forbidden))
    return desc, dst, semantic


@pytest.mark.parametrize("value", [False, True, 0, -1, 128.0, 2147483648, tl.constexpr(False),
                                  tl.constexpr(0), object.__new__(tl.tensor)])
def test_invalid_compile_time_byte_count_is_rejected(value):
    desc, dst, semantic = _guard_values()
    with pytest.raises(ValueError, match="positive compile-time int32"):
        core.copy(desc, dst, [1, 64], [0, 0], barrier=object(), expect_bytes=value, _semantic=semantic)


@pytest.mark.parametrize("shape, value, message", [([1, 32], 64, "exactly match"),
                                                  ([64], 128, "exactly match"),
                                                  ([1, False], 128, "positive static"),
                                                  ([1, -64], 128, "positive static"),
                                                  ([1, 64], 32768, "complete typed tile payload")])
def test_override_requires_complete_exact_tile(shape, value, message):
    desc, dst, semantic = _guard_values()
    with pytest.raises(ValueError, match=message):
        core.copy(desc, dst, shape, [0, 0], barrier=object(), expect_bytes=value, _semantic=semantic)


@pytest.mark.parametrize("dtype, dst_dtype", [(tl.float32, tl.bfloat16), (tl.int1, tl.int1),
                                              (tl.pointer_type(tl.int8), tl.pointer_type(tl.int8))])
def test_override_requires_matching_byte_addressable_scalars(dtype, dst_dtype):
    desc, dst, semantic = _guard_values(dtype, dst_dtype)
    with pytest.raises(ValueError, match="matching byte-addressable"):
        core.copy(desc, dst, [1, 64], [0, 0], barrier=object(), expect_bytes=128, _semantic=semantic)


@pytest.mark.parametrize("route", ["pointer", "store", "no_barrier", "nonshared", "logical", "async", "stride"])
def test_override_rejects_unsupported_transport_before_builder(route):
    desc, dst, semantic = _guard_values()
    src, destination, barrier = desc, dst, object()
    extra = {}
    if route == "pointer":
        src = object.__new__(tl.tensor)
    elif route == "store":
        src, destination = dst, desc
    elif route == "no_barrier":
        barrier = None
    elif route == "nonshared":
        dst.type.storage = types.tmem
    elif route == "logical":
        src = object.__new__(types._logical_tensor_descriptor)
    elif route == "async":
        extra["is_async"] = True
    else:
        extra["input_stride"] = 1
    with pytest.raises(ValueError, match="explicit barrier"):
        core.copy(src, destination, [1, 64], [0, 0], barrier=barrier, expect_bytes=128,
                  _semantic=semantic, **extra)


@pytest.mark.parametrize("backend", ["mthreads", "iluvatar", "common_ir", "other"])
def test_override_rejects_backend_that_cannot_honor_contribution(monkeypatch, backend):
    import triton._flagtree_backend as backend_module
    monkeypatch.setattr(core.mthreads_common, "enabled", lambda: backend == "mthreads")
    monkeypatch.setattr(core.iluvatar_copy, "enabled", lambda: backend == "iluvatar")
    monkeypatch.setattr(core.tle_semantic, "COMMON_IR_ENABLED", backend == "common_ir")
    monkeypatch.setattr(backend_module, "FLAGTREE_BACKEND", "amd" if backend == "other" else "")
    desc, dst, semantic = _guard_values()
    with pytest.raises(ValueError, match="only by NVIDIA TMA"):
        core.copy(desc, dst, [1, 64], [0, 0], barrier=object(), expect_bytes=128, _semantic=semantic)
