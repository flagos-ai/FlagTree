# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
"""Native 910_95 observability regressions for layout/memory compatibility.

These are deliberately *not* 910B4 smoke tests.  The four migrated legacy
optimizations keep their original compile-on-910_95 / pure-SIMT scheduling
slots, so value-only tests on a B4 cannot prove that the gate-on path ran.

The observer below wraps only test-process call sites.  It records the module
immediately before compiler metadata export and the launcher source generated
for the real JIT launch.  Production compiler and launcher behavior is left
unchanged.  Every test then launches on real hardware and checks both numerical
results and the pass-to-metadata-to-launcher handoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

try:
    import triton
    from triton.backends.ascend.utils import is_compile_on_910_95
    _target_arch = triton.runtime.driver.active.get_current_target().arch
except Exception:
    _target_arch = ""

    def is_compile_on_910_95(_arch):
        return False


# Do this before importing torch_npu, Triton, or defining JIT kernels.  In
# particular, an Ascend 910B4 must collect this module as skipped rather than
# execute a value-only fallback and be mistaken for a 910_95 gate-on result.
if not is_compile_on_910_95(_target_arch):
    pytest.skip(
        "requires a detected Ascend 950 toolchain; 910B4 is not a "
        "native gate-on substitute",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

import triton.language as tl
from triton.backends.ascend import compiler as ascend_compiler
from triton.backends.ascend import driver as ascend_driver

pytestmark = pytest.mark.backend("torch_npu")


@dataclass
class _NativePipelineObserver:
    """Capture only the native JIT handoff points needed by these regressions."""

    pre_export_ir: list[str] = field(default_factory=list)
    metadata_after_export: list[dict[str, Any]] = field(default_factory=list)
    launcher_sources: list[str] = field(default_factory=list)

    @classmethod
    def install(cls, monkeypatch: pytest.MonkeyPatch) -> "_NativePipelineObserver":
        observer = cls()
        export_metadata = ascend_compiler._export_coalesce_metadata
        make_launcher = ascend_driver.make_launcher

        def observe_export(module, metadata, **kwargs):
            # This is after Row or T2L's Axis/Chunk/SLS sequence but before
            # hacc.coalesce_* is intentionally removed for the vendor compiler.
            observer.pre_export_ir.append(str(module))
            result = export_metadata(module, metadata, **kwargs)
            observer.metadata_after_export.append(dict(metadata))
            return result

        def observe_launcher(*args, **kwargs):
            source = make_launcher(*args, **kwargs)
            observer.launcher_sources.append(source)
            return source

        monkeypatch.setattr(ascend_compiler, "_export_coalesce_metadata", observe_export)
        monkeypatch.setattr(ascend_driver, "make_launcher", observe_launcher)
        return observer

    def exported_ir_with(self, needle: str) -> str:
        for ir_text in reversed(self.pre_export_ir):
            if needle in ir_text:
                return ir_text
        raise AssertionError(f"did not observe {needle!r} before metadata export; captured "
                             f"{len(self.pre_export_ir)} module(s)")

    def launcher_with(self, needle: str) -> str:
        for source in reversed(self.launcher_sources):
            if needle in source:
                return source
        raise AssertionError(f"did not observe launcher fragment {needle!r}; generated "
                             f"{len(self.launcher_sources)} launcher(s)")


def _launch_with_observer(monkeypatch, kernel, grid, *args, **compile_options):
    """Force one native compile + launch and retain the associated observables."""
    # ``force_simt_template`` is an existing T2L precondition and may be set by
    # a test below.  Never accept a compile_on_910_95 override here: that bit
    # must continue to come exclusively from real device/toolchain detection.
    assert "compile_on_910_95" not in compile_options
    # JITFunction owns an in-process cache in addition to Triton's file cache.
    # Clearing only this test kernel prevents a prior invocation from bypassing
    # the observer, while TRITON_ALWAYS_COMPILE forces the compiler pipeline.
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    kernel.device_caches.clear()
    observer = _NativePipelineObserver.install(monkeypatch)
    compiled = kernel[grid](*args, **compile_options)
    torch.npu.synchronize()
    assert observer.pre_export_ir, "the native compilation never exported metadata"
    assert observer.launcher_sources, "the native launch never generated a launcher"
    return compiled, observer


def _assert_real_91095_gate(compiled, *, pure_simt: bool) -> None:
    """Do not let a manual mode override masquerade as device gate-on."""
    assert compiled.metadata.compile_on_910_95 is True
    assert compiled.metadata.is_pure_simt is pure_simt
    if not pure_simt:
        assert compiled.metadata.compile_mode == "simd_simt_template"


@triton.jit
def _row_tail_copy(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n:
        return
    offsets = pid + tl.arange(0, BLOCK)
    mask = offsets < n
    value = tl.load(src + offsets, mask=mask, other=0.0)
    tl.store(dst + offsets, value, mask=mask)


@triton.jit
def _chunk_axis1_copy(src, dst, N: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    base = batch * N
    value = tl.load(src + base + offsets, mask=mask, other=0.0)
    tl.store(dst + base + offsets, value, mask=mask)


@triton.jit
def _chunk_axis1_extra_data_predicate(src, dst, N: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    base = batch * N
    value = tl.load(src + base + offsets, mask=mask, other=0.0)
    # A second pid-dependent compare that selects data instead of only feeding
    # an assert.  N - 1 is not a whole number of tiles, so it also cannot be
    # mistaken for the seed mask.
    value = tl.where(offsets < N - 1, value, 0.0)
    tl.store(dst + base + offsets, value, mask=mask)


@triton.jit
def _chunk_axis2_copy(src, dst, N: tl.constexpr, BLOCK: tl.constexpr):
    # Chunk chooses its seed from the greatest program-id axis.  Keeping axes
    # 0 and 1 at one makes this a real grid-Z case, rather than merely testing
    # the driver's metadata-to-string branch in isolation.
    tile = tl.program_id(2)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    value = tl.load(src + offsets, mask=mask, other=0.0)
    tl.store(dst + offsets, value, mask=mask)


@triton.jit
def _sls_masked_stride4_gather(src, dst, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Static power-of-two stride plus the one-tile mask is specifically the
    # SLS indirect-load route; without the mask this stays on strided DMA.
    value = tl.load(src + offsets * 4, mask=mask, other=0.0)
    tl.store(dst + offsets, value, mask=mask)


def test_row_91095_native_metadata_launcher_and_ir(monkeypatch):
    """Row remains pure-SIMT and carries ceil-div grid metadata to launch."""
    n = 19  # H=8 leaves a tail, so ceil-div is observable at the launcher.
    src = torch.arange(n, dtype=torch.float32).npu()
    dst = torch.full_like(src, -1)

    compiled, observer = _launch_with_observer(
        monkeypatch,
        _row_tail_copy,
        (n, ),
        src,
        dst,
        n,
        BLOCK=16,
        compile_mode="simt_only",
    )

    assert torch.equal(dst.cpu(), src.cpu())
    _assert_real_91095_gate(compiled, pure_simt=True)
    assert compiled.metadata.coalesce_factor == 8
    assert compiled.metadata.coalesce_axis == 0
    assert compiled.metadata.coalesce_grid_ceil_div is True
    assert compiled.metadata.row_coalescing_applied is True

    row_ir = observer.exported_ir_with("hacc.coalesce_factor = 8 : i32")
    assert "hacc.coalesce_axis = 0 : i32" in row_ir
    assert "hacc.coalesce_grid_ceil_div = 1 : i32" in row_ir

    launcher = observer.launcher_with("gridX = (gridX + 8 - 1) / 8;")
    assert launcher.count("gridX = (gridX + 8 - 1) / 8;") == 2
    assert "ChunkCoalescing: grid[0] not divisible" not in launcher


def test_chunk_91095_native_metadata_launcher_and_ir(monkeypatch):
    """Chunk's factor/axis metadata is consumed by exact grid division."""
    batch, block, num_tiles = 2, 16, 32
    n = block * num_tiles
    src = torch.arange(batch * n, dtype=torch.float32).reshape(batch, n).npu()
    dst = torch.empty_like(src)

    compiled, observer = _launch_with_observer(
        monkeypatch,
        _chunk_axis1_copy,
        (batch, num_tiles),
        src,
        dst,
        N=n,
        BLOCK=block,
        # This selects the legacy T2L template precondition only.  It does
        # not set compile_on_910_95; _assert_real_91095_gate below proves the
        # device-derived gate remained true for the actual compiled kernel.
        force_simt_template=True,
        # Keep covering the uninstrumented form; the default instrumented one
        # is covered by test_chunk_coalesces_through_overflow_sanitizer_91095.
        sanitize_overflow=False,
    )

    assert torch.equal(dst.cpu(), src.cpu())
    _assert_real_91095_gate(compiled, pure_simt=False)
    assert compiled.metadata.coalesce_factor == 16
    assert compiled.metadata.coalesce_axis == 1
    assert compiled.metadata.coalesce_grid_ceil_div is False
    assert compiled.metadata.row_coalescing_applied is True

    chunk_ir = observer.exported_ir_with("hacc.coalesce_factor = 16 : i32")
    assert "hacc.coalesce_axis = 1 : i32" in chunk_ir
    assert "hacc.coalesce_grid_ceil_div" not in chunk_ir

    launcher = observer.launcher_with("gridY = gridY / 16;")
    assert launcher.count("gridY = gridY / 16;") == 2
    assert launcher.count("ChunkCoalescing: grid[1] not divisible by coalesce_factor 16") == 2


def test_chunk_axis2_91095_native_metadata_launcher_and_ir(monkeypatch):
    """Chunk's real max-axis seed shrinks grid-Z with the floor-div ABI."""
    block, num_tiles = 16, 32
    n = block * num_tiles
    src = torch.arange(n, dtype=torch.float32).npu()
    dst = torch.empty_like(src)

    compiled, observer = _launch_with_observer(
        monkeypatch,
        _chunk_axis2_copy,
        (1, 1, num_tiles),
        src,
        dst,
        N=n,
        BLOCK=block,
        # Preserve the original T2L template prerequisite only; the hardware
        # detection assertion below proves this is not a forged 910_95 result.
        force_simt_template=True,
        sanitize_overflow=False,
    )

    assert torch.equal(dst.cpu(), src.cpu())
    _assert_real_91095_gate(compiled, pure_simt=False)
    assert compiled.metadata.coalesce_factor == 16
    assert compiled.metadata.coalesce_axis == 2
    assert compiled.metadata.coalesce_grid_ceil_div is False
    assert compiled.metadata.row_coalescing_applied is True

    chunk_ir = observer.exported_ir_with("hacc.coalesce_factor = 16 : i32")
    assert "hacc.coalesce_axis = 2 : i32" in chunk_ir
    assert "hacc.coalesce_grid_ceil_div" not in chunk_ir

    launcher = observer.launcher_with("gridZ = gridZ / 16;")
    # make_launcher emits both the stable ABI and local C++ packing paths.
    assert launcher.count("gridZ = gridZ / 16;") == 2
    assert launcher.count("ChunkCoalescing: grid[2] not divisible by coalesce_factor 16") == 2


def test_chunk_coalesces_through_overflow_sanitizer_91095(monkeypatch):
    """The default overflow instrumentation no longer suppresses Chunk."""
    batch, block, num_tiles = 2, 16, 32
    n = block * num_tiles
    src = torch.arange(batch * n, dtype=torch.float32).reshape(batch, n).npu()
    dst = torch.empty_like(src)

    compiled, observer = _launch_with_observer(
        monkeypatch,
        _chunk_axis1_copy,
        (batch, num_tiles),
        src,
        dst,
        N=n,
        BLOCK=block,
        force_simt_template=True,
        # Do not pass sanitize_overflow: the JIT default instruments every
        # pid-derived index with compares that can only trap.
    )

    assert torch.equal(dst.cpu(), src.cpu())
    _assert_real_91095_gate(compiled, pure_simt=False)
    assert compiled.metadata.sanitize_overflow is True
    assert compiled.metadata.coalesce_factor == 16
    assert compiled.metadata.coalesce_axis == 1
    assert compiled.metadata.coalesce_grid_ceil_div is False
    assert compiled.metadata.row_coalescing_applied is True

    chunk_ir = observer.exported_ir_with("hacc.coalesce_factor = 16 : i32")
    assert "hacc.coalesce_axis = 1 : i32" in chunk_ir
    assert "hacc.coalesce_grid_ceil_div" not in chunk_ir

    launcher = observer.launcher_with("gridY = gridY / 16;")
    assert launcher.count("gridY = gridY / 16;") == 2
    assert launcher.count("ChunkCoalescing: grid[1] not divisible by coalesce_factor 16") == 2


def test_chunk_rejects_data_reaching_pid_predicate_91095(monkeypatch):
    """Only trap-only compares are waived; one that selects data must bail."""
    batch, block, num_tiles = 2, 16, 32
    n = block * num_tiles
    src = torch.arange(batch * n, dtype=torch.float32).reshape(batch, n).npu()
    dst = torch.empty_like(src)

    compiled, observer = _launch_with_observer(
        monkeypatch,
        _chunk_axis1_extra_data_predicate,
        (batch, num_tiles),
        src,
        dst,
        N=n,
        BLOCK=block,
        force_simt_template=True,
    )

    expected = src.clone()
    expected[:, n - 1] = 0.0
    assert torch.equal(dst.cpu(), expected.cpu())
    _assert_real_91095_gate(compiled, pure_simt=False)
    assert compiled.metadata.sanitize_overflow is True
    assert compiled.metadata.coalesce_factor == 1
    assert compiled.metadata.coalesce_axis == -1
    assert compiled.metadata.coalesce_grid_ceil_div is False
    assert compiled.metadata.row_coalescing_applied is False
    assert all("hacc.coalesce_factor" not in ir_text for ir_text in observer.pre_export_ir)
    assert all("gridY = gridY / 16;" not in launcher for launcher in observer.launcher_sources)


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_sls_91095_native_ir_metadata_and_mixed_simt_launcher(monkeypatch):
    """SLS emits the masked indirect-load path and preserves its launch ABI."""
    n = 1024
    src = torch.arange(n * 4, dtype=torch.float16).npu()
    dst = torch.empty(n, dtype=torch.float16).npu()

    compiled, observer = _launch_with_observer(
        monkeypatch,
        _sls_masked_stride4_gather,
        (1, ),
        src,
        dst,
        N=n,
        BLOCK=n,
        # Same as Chunk: retain the original template condition but never
        # override the device-derived compile_on_910_95 gate.
        force_simt_template=True,
    )

    assert torch.equal(dst.cpu(), src.cpu()[::4])
    _assert_real_91095_gate(compiled, pure_simt=False)
    assert compiled.metadata.parallel_mode == "mix_simd_simt"
    # This is the historical non-pure-SIMT branch in NPUOptions.__post_init__:
    # 895 and the migrated source both set it to 221184.  Read the real JIT
    # metadata instead of supplying a test-local launcher metadata object.
    assert compiled.metadata.shared_mem_dynamic_size == 221184
    assert compiled.metadata.coalesce_factor == 1
    assert compiled.metadata.coalesce_axis == -1
    assert compiled.metadata.row_coalescing_applied is False

    sls_ir = observer.exported_ir_with("triton_indirect_load")
    assert "parallel_mode = \"mix_simd_simt\"" in sls_ir

    launcher = observer.launcher_with("rtKernelLaunchWithFlagV2")
    # cfg setup is shared via the cann shim (once); both launch paths call it.
    assert launcher.count("rtKernelLaunchWithFlagV2") == 1
    assert launcher.count("rtArgsEx_t argsInfo") == 1
    # The same historical value must reach cfgInfo via both generated launcher
    # paths; the target is not allowed to silently pick a different ABI value.
    assert launcher.count("cann_get_launch_kernel_cfg(221184)") == 2
