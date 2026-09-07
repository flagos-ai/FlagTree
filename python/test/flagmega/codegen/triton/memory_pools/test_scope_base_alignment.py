# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Device-code UT for the generated pool ABI, without a model or pipeline."""

import importlib.util
import sys

import pytest

from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry


_COPY_KERNEL = '''
@triton.jit(noinline=True)
def _copy_scope(pool, output):
    row = tl.arange(0, 1)[:, None]
    col = tl.arange(0, 256)[None, :]
    offsets = row * 256 + col
    stage = tle.gpu.alloc(
        [1, 256], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    tle.gpu.copy(pool.to(tl.pointer_type(tl.bfloat16)) + offsets,
                 stage, [1, 256], is_async=True)
    tle.gpu.async_commit_group()
    tle.gpu.async_wait_group(0)
    values = tl.load(tle.gpu.local_ptr(stage, (row, col), shape=(1, 256)))
    tl.store(output + tl.program_id(0) * 256 + offsets, values)
'''


def _load_scope_module(tmp_path):
    source = TritonTemplateRegistry().render(
        "module.py.jinja",
        {
            "renderer_version": "scope-abi-unit-test",
            "entry_template": "entrypoints/call_graph.py.jinja",
            "use_tle": True,
            "grid_mesh": False,
            "shared_silu": False,
            "kernel_templates": (),
            "replicated_block_runtime_pool": True,
            "symbol": "scope_copy",
            "signature": "pool, output, SCOPE_NBYTES: tl.constexpr",
            "pipeline_schedule": None,
            "entry_events": ({
                "kind": "tir.kernel_call",
                "call": "copy",
                "family": "test",
                "variant": "copy",
                "execution_kind": "local_shard",
                "symbol": "_copy_scope",
                "arguments": "_flagmega_block_scope_base(pool, SCOPE_NBYTES), output",
                "barrier_before": False,
            },),
        },
    ) + _COPY_KERNEL
    path = tmp_path / "scope_abi.py"
    path.write_text(source)
    name = f"_flagmega_scope_abi_{tmp_path.name}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("SM90 shared-layout fixture")
    return torch


@pytest.mark.parametrize("scope_nbytes", [4096, 6656])
def test_scoped_pool_keeps_alignment_across_noinline_boundary(tmp_path, scope_nbytes):
    torch = _torch_cuda()
    module = _load_scope_module(tmp_path)
    pool = torch.zeros(2 * scope_nbytes, dtype=torch.uint8, device="cuda")
    expected = torch.arange(512, dtype=torch.bfloat16, device="cuda").reshape(2, 256)
    for scope in range(2):
        pool[scope * scope_nbytes:scope * scope_nbytes + 512].view(torch.bfloat16).copy_(expected[scope])
    output = torch.empty_like(expected)
    compiled = module.scope_copy[(2,)](pool, output, scope_nbytes, num_warps=8)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert "cp.async" in compiled.asm["ptx"]
    assert "tle.required_async_copy" in compiled.asm["ttgir"]


def test_scope_stride_does_not_invent_pointer_alignment(tmp_path):
    torch = _torch_cuda()
    module = _load_scope_module(tmp_path)
    # A 2-byte-aligned scope cannot promise the 4-byte minimum cp.async width.
    # The generated offset proof must preserve this limitation, not assert a
    # convenient fixed alignment on the returned pointer.
    pool = torch.empty(2 * 4098, dtype=torch.uint8, device="cuda")
    output = torch.empty((2, 256), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        module.scope_copy.warmup(pool, output, 4098, grid=(2,), num_warps=8)
