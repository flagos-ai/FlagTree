# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Native AIU descriptor copies: data, padding, per-copy kp and completion."""
import itertools
import math
from pathlib import Path
import subprocess

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental.tle import language as tle
from triton.experimental.tle.language.gpu.ppu import AIUCachePolicy, CachePolicy
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _copy_tile(desc, B, BLOCK: tl.constexpr, OFFSET: tl.constexpr, POLICY: tl.constexpr):
    tile = tle.gpu.alloc(BLOCK, desc.dtype, nv_mma_shared_layout=False)
    if len(BLOCK) == 2:
        row = tl.program_id(0) * BLOCK[0] + OFFSET[0]
        col = tl.program_id(1) * BLOCK[1] + OFFSET[1]
        tle.gpu.copy(desc, tile, BLOCK, (row, col), cache_policy=POLICY)
        program = tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)
        output = B + program * BLOCK[0] * BLOCK[1]
        output = output + tl.arange(0, BLOCK[0])[:, None] * BLOCK[1] + tl.arange(0, BLOCK[1])[None, :]
    else:
        h = tl.program_id(0) * BLOCK[0] + OFFSET[0]
        row = tl.program_id(1) * BLOCK[1] + OFFSET[1]
        col = tl.program_id(2) * BLOCK[2] + OFFSET[2]
        tle.gpu.copy(desc, tile, BLOCK, (h, row, col), cache_policy=POLICY)
        program = (tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)) * tl.num_programs(2) + tl.program_id(2)
        output = B + program * BLOCK[0] * BLOCK[1] * BLOCK[2]
        output = output + tl.arange(0, BLOCK[0])[:, None, None] * BLOCK[1] * BLOCK[2]
        output = output + tl.arange(0, BLOCK[1])[None, :, None] * BLOCK[2] + tl.arange(0, BLOCK[2])[None, None, :]
    tle.gpu.copy(tile, output, BLOCK)


@triton.jit
def _device(A, B, H, M, N, BLOCK: tl.constexpr, OFFSET: tl.constexpr, POLICY: tl.constexpr):
    if len(BLOCK) == 2:
        desc = tl.make_tensor_descriptor(A, [M, N], [N, 1], BLOCK)
    else:
        desc = tl.make_tensor_descriptor(A, [H, M, N], [M * N, N, 1], BLOCK)
    _copy_tile(desc, B, BLOCK, OFFSET, POLICY)


@triton.jit
def _host(desc, B, BLOCK: tl.constexpr, OFFSET: tl.constexpr, POLICY: tl.constexpr):
    _copy_tile(desc, B, BLOCK, OFFSET, POLICY)


def _check_native(kernel, modes, tmp_path):
    from triton.backends.ppu.compiler import get_ppu_llc
    binary = tmp_path / "copy.hgbin"
    binary.write_bytes(kernel.asm["hgbin"])
    isa = subprocess.check_output([str(Path(get_ppu_llc()).parent / "hgobjdump"), "--dump-isa", str(binary)], text=True)
    (tmp_path / "copy.isa").write_text(isa)
    for suffix in ("ttir", "ttgir", "llir"):
        (tmp_path / ("copy." + suffix)).write_text(kernel.asm[suffix])
    aiu = [line for line in isa.splitlines() if "vmem.aiu.ld.tsm" in line]
    expected = {2 if mode == 3 else mode or 1 for mode in modes}
    actual = {mode for mode in range(1, 8) if any(f".kp{mode}" in line for line in aiu)}
    assert actual == expected
    assert "__flagtree_ppu_aiu_copy" not in isa
    assert "vmem.acp.commit.grp" in isa and "commit_group(0)" in isa
    assert "ttg.tma_copy" not in kernel.asm["ttir"]


def _expected_tiles(source, block, offset, grid):
    output = torch.zeros((math.prod(grid), *block), dtype=source.dtype)
    for program, ids in enumerate(itertools.product(*(range(n) for n in grid))):
        starts = [i * b + o for i, b, o in zip(ids, block, offset)]
        source_slices, dest_slices = [], []
        for start, width, size in zip(starts, block, source.shape):
            lower, upper = max(0, start), min(size, start + width)
            if lower >= upper:
                break
            source_slices.append(slice(lower, upper))
            dest_slices.append(slice(lower - start, upper - start))
        else:
            output[(program, *dest_slices)] = source[tuple(source_slices)]
    return output


@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("kp", range(7))
@pytest.mark.parametrize("shape,block,offset,grid,dtype", [
    ((93, 160), (32, 32), (0, 0), (3, 5), "float32"),
    ((79, 128), (16, 32), (16, 32), (5, 4), "float16"),
    ((33, 160), (32, 64), (0, 0), (2, 3), "int8"),
    ((64, 1024), (32, 512), (0, 0), (2, 2), "bfloat16"),
    ((3, 61, 96), (2, 16, 32), (0, 0, 0), (2, 4, 3), "int32"),
    ((3, 64, 128), (2, 32, 32), (-1, 16, 32), (3, 2, 4), "float16"),
    ((5, 48, 64), (4, 16, 32), (0, 0, 0), (2, 3, 2), "float32"),
    ((2, 32, 64), (1, 32, 32), (0, 0, 0), (2, 1, 2), "float16"),
])
def test_descriptor_tiles(host, kp, shape, block, offset, grid, dtype, tmp_path):
    dtype = getattr(torch, dtype)
    a = torch.empty(shape, dtype=dtype, device="cuda")
    b = torch.empty((math.prod(grid), *block), dtype=dtype, device="cuda")
    policy = None if kp == 0 else AIUCachePolicy(kp=kp)
    # Reuse the allocation with different GPU-produced values; never read A
    # back before AIU. This detects stale-cache failures hidden by CPU copies.
    for seed in [kp * 7, kp * 7 + 31]:
        a.copy_(((torch.arange(math.prod(shape), device="cuda", dtype=torch.int32) + seed) % 97 - 48).reshape(shape))
        reference = ((torch.arange(math.prod(shape), dtype=torch.int32) + seed) % 97 - 48).to(dtype).reshape(shape)
        expected = _expected_tiles(reference, block, offset, grid)
        b.fill_(-111)
        if host:
            kernel = _host[grid](TensorDescriptor.from_tensor(a, list(block)), b, block, offset, policy)
        else:
            h, m, n = (1, *shape) if len(shape) == 2 else shape
            kernel = _device[grid](a, b, h, m, n, block, offset, policy)
        torch.cuda.synchronize()
        assert torch.equal(b.cpu(), expected)
    _check_native(kernel, [kp], tmp_path)


@triton.jit
def _loop(desc, B, STAGES: tl.constexpr, POLICY: tl.constexpr):
    tile = tle.gpu.alloc((32, 32), desc.dtype, nv_mma_shared_layout=False)
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    for index in tl.range(0, 4, num_stages=STAGES):
        tle.gpu.copy(desc, tile, (32, 32), (index * 32, 32), cache_policy=POLICY)
        tle.gpu.copy(tile, B + index * 1024 + r * 32 + c, (32, 32))


@pytest.mark.parametrize("kp", range(1, 7))
@pytest.mark.parametrize("stages", [1, 3])
def test_descriptor_loop(kp, stages, tmp_path):
    a = (torch.arange(128 * 96, dtype=torch.int32, device="cuda") + kp).reshape(128, 96)
    expected = (torch.arange(128 * 96, dtype=torch.int32) + kp).reshape(128, 96)[:, 32:64].reshape(4, 32, 32)
    b = torch.empty((4, 32, 32), dtype=torch.int32, device="cuda")
    kernel = _loop[(1, )](TensorDescriptor.from_tensor(a, [32, 32]), b, stages, AIUCachePolicy(kp=kp))
    torch.cuda.synchronize()
    assert torch.equal(b.cpu(), expected)
    _check_native(kernel, [kp], tmp_path)


@triton.jit
def _two_copies(a, b, C, P: tl.constexpr, Q: tl.constexpr):
    left = tle.gpu.alloc((32, 32), tl.float32, nv_mma_shared_layout=False)
    right = tle.gpu.alloc((32, 32), tl.float32, nv_mma_shared_layout=False)
    tle.gpu.copy(a, left, (32, 32), (0, 0), cache_policy=P)
    tle.gpu.copy(b, right, (32, 32), (0, 0), cache_policy=Q)
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    r, c = tl.broadcast(r, c)
    x = tl.load(tle.gpu.local_ptr(left, (r, c)))
    y = tl.load(tle.gpu.local_ptr(right, (r, c)))
    tl.store(C + r * 32 + c, x + y)


@pytest.mark.parametrize("p,q", [(0, 4), (1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 4)])
def test_independent_per_copy_policy(p, q, tmp_path):
    a = torch.full((32, 32), 10 * p, dtype=torch.float32, device="cuda")
    b = torch.full((32, 32), q, dtype=torch.float32, device="cuda")
    c = torch.empty_like(a)
    kernel = _two_copies[(1, )](TensorDescriptor.from_tensor(a, [32, 32]), TensorDescriptor.from_tensor(b, [32, 32]), c,
                                None if p == 0 else AIUCachePolicy(kp=p), AIUCachePolicy(kp=q))
    torch.cuda.synchronize()
    assert torch.equal(c.cpu(), torch.full((32, 32), 10 * p + q, dtype=torch.float32))
    _check_native(kernel, [p, q], tmp_path)


@triton.jit(noinline=True)
def _descriptor_helper(desc, B, POLICY: tl.constexpr):
    _copy_tile(desc, B, (32, 32), (0, 0), POLICY)


@triton.jit
def _choose_descriptor(a, b, B, choose, POLICY: tl.constexpr):
    if choose:
        desc = a
    else:
        desc = b
    _descriptor_helper(desc, B, POLICY)


@pytest.mark.parametrize("kp", [1, 4, 6])
def test_descriptor_control_flow_and_noinline(kp, tmp_path):
    a = torch.full((32, 32), 7 + kp, dtype=torch.float32, device="cuda")
    b = torch.full((64, 32), 19 + kp, dtype=torch.float32, device="cuda")
    output = torch.empty((32, 32), dtype=torch.float32, device="cuda")
    for choice, value in [(1, 7 + kp), (0, 19 + kp)]:
        kernel = _choose_descriptor[(1, )](TensorDescriptor.from_tensor(a, [32, 32]),
                                           TensorDescriptor.from_tensor(b, [32, 32]), output, choice,
                                           AIUCachePolicy(kp=kp))
        torch.cuda.synchronize()
        assert torch.equal(output.cpu(), torch.full((32, 32), value, dtype=torch.float32))
    _check_native(kernel, [kp], tmp_path)


def test_host_descriptor_runtime_bindings(tmp_path):
    inputs = [
        torch.full(shape, value, dtype=torch.float32, device="cuda")
        for shape, value in [((64, 96), 11), ((80, 160), 29), ((128, 192), 53)]
    ]
    output = torch.empty((32, 32), dtype=torch.float32, device="cuda")
    first = None
    for a, value in zip(inputs, [11, 29, 53]):
        kernel = _host[(1, 1)](TensorDescriptor.from_tensor(a, [32, 32]), output, (32, 32), (0, 0),
                               AIUCachePolicy(kp=4))
        torch.cuda.synchronize()
        assert torch.equal(output.cpu(), torch.full((32, 32), value, dtype=torch.float32))
        if first is None:
            first = kernel
        else:
            assert kernel is first
    _check_native(kernel, [4], tmp_path)
    assert "contiguous block-pointer strides" in kernel.asm["ttgir"]


@triton.jit
def _ordinary(desc, B):
    x = desc.load([tl.program_id(0) * 32, tl.program_id(1) * 32])
    program = tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    tl.store(B + program * 1024 + r * 32 + c, x)


def test_unmarked_descriptor_load_keeps_existing_path():
    a = torch.arange(93 * 160, dtype=torch.int32, device="cuda").reshape(93, 160)
    b = torch.empty((15, 32, 32), dtype=torch.int32, device="cuda")
    kernel = _ordinary[(3, 5)](TensorDescriptor.from_tensor(a, [32, 32]), b)
    torch.cuda.synchronize()
    reference = torch.arange(93 * 160, dtype=torch.int32).reshape(93, 160)
    assert torch.equal(b.cpu(), _expected_tiles(reference, (32, 32), (0, 0), (3, 5)))
    assert "async_aiu_copy" not in kernel.asm["ttgir"]


def test_backend_rejects_kp7_without_frontend(tmp_path, capfd):
    kernel = triton.compile(
        ASTSource(_device, {"A": "*fp32", "B": "*fp32"},
                  dict(H=1, M=128, N=128, BLOCK=(32, 32), OFFSET=(0, 0), POLICY=AIUCachePolicy(kp=6))),
        target=GPUTarget("cuda", 80, 32), options={"num_warps": 4})
    text = kernel.asm["ttgir"]
    assert "cacheModifier = ppu_aiu_kp6" in text
    bad = tmp_path / "kp7.ttgir"
    bad.write_text(text.replace("ppu_aiu_kp6", "ppu_aiu_kp7"))
    with pytest.raises(Exception) as error:
        triton.compile(str(bad), target=GPUTarget("cuda", 80, 32), options={"num_warps": 4})
    diagnostic = str(error.value) + capfd.readouterr().err
    assert "enum values for attribute 'cache'" in diagnostic


@triton.jit
def _invalid(A, B, STRIDE: tl.constexpr, BLOCK: tl.constexpr, SHARED: tl.constexpr, PADDING: tl.constexpr,
             POLICY: tl.constexpr, BARRIER: tl.constexpr, STORE: tl.constexpr, OFFSET: tl.constexpr):
    desc = tl.make_tensor_descriptor(A, [128, 128], [STRIDE, 1], BLOCK, padding_option=PADDING)
    tile = tle.gpu.alloc(SHARED, A.dtype.element_ty, nv_mma_shared_layout=False)
    if STORE:
        tle.gpu.copy(tile, desc, BLOCK, OFFSET, cache_policy=POLICY)
    else:
        tle.gpu.copy(desc, tile, BLOCK, OFFSET, barrier=BARRIER, cache_policy=POLICY)
        r = tl.arange(0, SHARED[0])[:, None]
        c = tl.arange(0, SHARED[1])[None, :]
        tle.gpu.copy(tile, B + r * SHARED[1] + c, SHARED)


@pytest.mark.parametrize("changes,diagnostic", [
    ({"STRIDE": 256}, "contiguous block-pointer strides"),
    ({"SHARED": (16, 32)}, "matching rank-2 or rank-3"),
    ({"PADDING": "nan"}, "requires zero padding"),
    ({"POLICY": AIUCachePolicy(kp=7)}, "GPU-produced input can be stale"),
    ({"POLICY": CachePolicy(".ca")}, "expected AIUCachePolicy"),
    ({"BARRIER": 0}, "explicit TMA barriers are unsupported"),
    ({"STORE": True}, "global-to-shared loads only"),
    ({"BLOCK": (8, 32), "SHARED": (8, 32)}, "16..2048 rows"),
    ({"BLOCK": (32, 4), "SHARED": (32, 4)}, "32 contiguous bytes"),
    ({"OFFSET": (-1, 0)}, "requires nonnegative X/Z offsets"),
    ({"OFFSET": (0, -32)}, "requires nonnegative X/Z offsets"),
])
def test_invalid_descriptor_copy(changes, diagnostic, capfd):
    constants = dict(STRIDE=128, BLOCK=(32, 32), SHARED=(32, 32), PADDING="zero", POLICY=AIUCachePolicy(kp=4),
                     BARRIER=None, STORE=False, OFFSET=(0, 0))
    constants.update(changes)
    with pytest.raises(Exception) as error:
        triton.compile(ASTSource(_invalid, {"A": "*fp32", "B": "*fp32"}, constants), target=GPUTarget("cuda", 80, 32),
                       options={"num_warps": 4})
    assert diagnostic in str(error.value) + capfd.readouterr().err
