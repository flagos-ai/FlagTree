# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A shared layout transfer must select one physical writer per logical value."""

import os
from pathlib import Path
import re

import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def _reduced_layout_copy(source, output):
    layout: tl.constexpr = tle.gpu.BlockEncoding([1], [32], [8], [0])
    offsets = tle.gpu.set_layout(tl.arange(0, 2048), layout)
    values = tl.load(source + offsets)
    squares = values * values
    partials = tl.sum(tl.reshape(squares, (128, 16)), axis=1)
    moved = tle.gpu.set_layout(partials, layout)
    result_offsets = tle.gpu.set_layout(tl.arange(0, 128), layout)
    tl.store(output + result_offsets, moved)


def compile_reduced_layout_copy():
    return triton.compile(
        ASTSource(_reduced_layout_copy, {'source': '*fp32', 'output': '*fp32'},
                  attrs={(0,): [('tt.divisibility', 16)], (1,): [('tt.divisibility', 16)]}),
        target=GPUTarget('cuda', 90, 32),
        options={'num_warps': 8, 'enable_fp_fusion': True},
    )


def test_replicated_shared_conversion_selects_writers():
    # CPU-only normal compilation. This is the actual reduction->layout
    # conversion that can give replicated lanes different last bits after FMA.
    compiled = compile_reduced_layout_copy()
    directory = os.environ.get('TLE_SHARED_OWNER_EVIDENCE')
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        for kind, value in compiled.asm.items():
            destination = path / ('reduction.' + kind)
            destination.write_bytes(value) if isinstance(value, bytes) else destination.write_text(value)
    ptx = compiled.asm['ptx']
    stores = [line.strip() for line in ptx.splitlines() if re.search(r'\bst\.shared\.', line)]
    assert stores and any('st.shared.v2.b32' in line for line in stores)
    assert 'fma.rn.f32' in ptx, 'Keep the original FMA optimization enabled'
    assert all(line.startswith('@%p') for line in stores), (
        'Replicated shared layout stores have no canonical-writer predicate', stores)
    mask = re.search(r'and\.b32\s+(%r\d+),\s*%r\d+,\s*15;', ptx)
    assert mask, 'The low four lane bits duplicate the same shared address'
    owner = re.search(r'setp\.eq\.b32\s+(%p\d+),\s*' + re.escape(mask[1]) + r',\s*0;', ptx)
    assert owner and all(line.startswith('@' + owner[1] + ' ') for line in stores)


@triton.jit
def _small_reduced_layout_copy(source, output):
    layout: tl.constexpr = tle.gpu.BlockEncoding([1], [32], [8], [0])
    offsets = tle.gpu.set_layout(tl.arange(0, 128), layout)
    values = tl.load(source + offsets)
    partials = tl.sum(tl.reshape(values * values, (8, 16)), axis=1)
    moved = tle.gpu.set_layout(partials, layout)
    result_offsets = tle.gpu.set_layout(tl.arange(0, 8), layout)
    tl.store(output + result_offsets, moved)


@triton.jit
def _nonbroadcast_layout_copy(source, output):
    first: tl.constexpr = tle.gpu.BlockEncoding([1], [32], [8], [0])
    second: tl.constexpr = tle.gpu.BlockEncoding([4], [32], [8], [0])
    index = tle.gpu.set_layout(tl.arange(0, 256), first)
    values = tl.load(source + index)
    moved = tle.gpu.set_layout(values, second)
    index = tle.gpu.set_layout(tl.arange(0, 256), second)
    tl.store(output + index, moved)


def test_shared_transfer_bitwise_values():
    import ctypes
    import numpy as np
    import pytest
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip('requires a CUDA SM90 or newer device')
    # Independent CPU execution of the original local16 lane-zero expression:
    # the first own square contracts into FMA; its shuffled peer is rounded.
    fmaf = ctypes.CDLL('libm.so.6').fmaf
    fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    fmaf.restype = ctypes.c_float
    rng = np.random.default_rng(73021)

    def bf16(values):
        bits = values.view(np.uint32)
        return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)

    raw = bf16(rng.standard_normal((28, 2048)).astype(np.float32))
    raw += bf16(rng.standard_normal((28, 2048)).astype(np.float32))
    groups = raw.reshape(28, 128, 16)
    peers = np.multiply(groups, groups, dtype=np.float32)[..., np.arange(16) ^ 8]
    values = np.array([fmaf(float(x), float(x), float(y)) for x, y in zip(groups.flat, peers.flat)],
                      dtype=np.float32).reshape(groups.shape)
    for distance in (4, 2, 1):
        values = np.add(values, values[..., np.arange(16) ^ distance], dtype=np.float32)
    assert np.any(values[..., 0].view(np.uint32) != values[..., 8].view(np.uint32))
    expected = torch.from_numpy(values[..., 0].copy()).to('cuda')
    source = torch.from_numpy(raw).to('cuda')
    output = torch.empty(128, dtype=torch.float32, device='cuda')
    evidence = os.environ.get('TLE_SHARED_OWNER_DEVICE_EVIDENCE')
    if evidence:
        evidence = Path(evidence)
        evidence.mkdir(parents=True, exist_ok=False)
        np.save(evidence / 'inputs.npy', raw)
        np.save(evidence / 'expected.npy', values[..., 0])
    for row in (0, 2, 3, 11, 15, 25):
        for repeat in range(5):
            output.fill_(float('nan'))
            _reduced_layout_copy[(1,)](source[row], output, num_warps=8, enable_fp_fusion=True)
            if evidence:
                np.save(evidence / f'full-{row}-{repeat}.npy', output.cpu().numpy())
            assert torch.equal(output.view(torch.int32), expected[row].view(torch.int32))
            _small_reduced_layout_copy[(1,)](source[row], output, num_warps=8, enable_fp_fusion=True)
            if evidence:
                np.save(evidence / f'small-{row}-{repeat}.npy', output[:8].cpu().numpy())
            assert torch.equal(output[:8].view(torch.int32), expected[row, :8].view(torch.int32))
    copied = torch.empty(256, dtype=torch.float32, device='cuda')
    _nonbroadcast_layout_copy[(1,)](source[0], copied, num_warps=8)
    assert torch.equal(copied.view(torch.int32), source[0, :256].view(torch.int32))
