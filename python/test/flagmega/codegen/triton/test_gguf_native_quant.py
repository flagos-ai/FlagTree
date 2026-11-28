# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Differential GPU tests with known packed GGML blocks, including masked rows."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from jinja2 import Environment, FileSystemLoader

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _packed_fixture(encoding, rows=5, k=512):
    """Construct codes and expected values together; never convert a checkpoint."""
    rng = np.random.default_rng(42)
    blocks = rows * k // 256
    codes = rng.integers(0, 16, (blocks, 8, 32), dtype=np.uint8)
    scales = rng.integers(0, 64, (blocks, 8), dtype=np.uint8)
    d = rng.uniform(0.0001, 0.001, (blocks, 1)).astype(np.float16)
    if encoding == "IQ4_XS":
        high = sum((scales[:, i].astype(np.uint16) >> 4) << (2 * i) for i in range(8))
        low = (scales[:, ::2] & 15) | ((scales[:, 1::2] & 15) << 4)
        qs = codes[:, :, :16] | (codes[:, :, 16:] << 4)
        packed = np.concatenate((d.view(np.uint8), high[:, None].view(np.uint8), low, qs.reshape(blocks, -1)), axis=1)
        values = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)
        expected = d.astype(np.float32)[:, :, None] * (scales.astype(np.float32) - 32)[:, :, None] * values[codes]
    else:
        mins = rng.integers(0, 64, (blocks, 8), dtype=np.uint8)
        dmin = rng.uniform(0.0001, 0.001, (blocks, 1)).astype(np.float16)
        metadata = np.concatenate((
            scales[:, :4] | ((scales[:, 4:] >> 4) << 6),
            mins[:, :4] | ((mins[:, 4:] >> 4) << 6),
            (scales[:, 4:] & 15) | ((mins[:, 4:] & 15) << 4),
        ), axis=1)
        qs = codes[:, ::2, :] | (codes[:, 1::2, :] << 4)
        packed = np.concatenate((d.view(np.uint8), dmin.view(np.uint8), metadata, qs.reshape(blocks, -1)), axis=1)
        expected = (d.astype(np.float32)[:, :, None] * scales[:, :, None] * codes -
                    dmin.astype(np.float32)[:, :, None] * mins[:, :, None])
    return packed.reshape(rows, -1), expected.reshape(rows, k)


@pytest.fixture
def kernels(tmp_path):
    root = Path(__file__).resolve().parents[4] / "triton/flagmega/codegen/triton"
    env = Environment(loader=FileSystemLoader(root))
    source = "import triton\nimport triton.language as tl\n"
    for template in ("block_fp8/simt", "matmul_glu/simt", "embedding/decode", "gdn_recurrent/_helpers"):
        source += env.get_template(f"kernels/{template}.py.jinja").render(tn=8, mesh_size=1) + "\n"
    source += '''
@triton.jit
def test_embedding(weight, output, K: tl.constexpr, ENCODING: tl.constexpr):
    feature = tl.arange(0, 1024)
    mask = feature < K
    if ENCODING == "IQ4_XS":
        value = _flagmega_iq4_xs_embedding(weight, 3, feature, mask, K)
    else:
        value = _flagmega_q4_k_embedding(weight, 3, feature, mask, K)
    tl.store(output + feature, value, mask=mask)

@triton.jit
def test_projection(source, weight, output, N: tl.constexpr, K: tl.constexpr,
                    ENCODING: tl.constexpr, FAMILY: tl.constexpr, UNIFORM: tl.constexpr):
    row = tl.arange(0, 8)
    mask = row < N
    if ENCODING == "IQ4_XS":
        block_bytes: tl.constexpr = 136
    else:
        block_bytes: tl.constexpr = 144
    offsets = row * (K // 256) * block_bytes
    if FAMILY == "linear":
        if ENCODING == "IQ4_XS":
            _flagmega_iq4_xs_linear_local_tile(source, weight, output, offsets, row, mask, K)
        else:
            _flagmega_q4_k_linear_local_tile(source, weight, output, offsets, row, mask, K)
    elif FAMILY == "glu":
        if ENCODING == "IQ4_XS":
            result = _flagmega_iq4_xs_dot(source, weight, offsets, mask, K)
        else:
            result = _flagmega_q4_k_dot(source, weight, offsets, mask, K)
        tl.store(output + row, result.to(tl.bfloat16), mask=mask)
    else:
        if UNIFORM:
            row = tl.full((8,), 3, tl.int32)
        result = _flagmega_dense_rows(source, weight, row, mask, K, 128, UNIFORM, ENCODING)
        tl.store(output + tl.arange(0, 8), result.to(tl.bfloat16), mask=mask)

@triton.jit
def test_recurrent(qkv, state, core, TILED: tl.constexpr):
    offsets = tl.arange(0, 32)
    beta = tl.full((), 0.5, tl.float32)
    decay = tl.full((), 0.9, tl.float32)
    _flagmega_gdn_recurrent_local_core_tile(
        qkv, beta, decay, state, core, offsets, offsets, offsets < 24,
        6, 2, 4, 4, 3, 4, False, 1e-12, False, False,
        TILED_VALUE_HEADS=TILED,
    )
'''
    path = tmp_path / "native_quant_kernels.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("native_quant_kernels", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("encoding", ["IQ4_XS", "Q4_K"])
def test_native_quantized_embedding(kernels, encoding):
    packed, expected = _packed_fixture(encoding)
    weight = torch.from_numpy(packed).cuda()
    output = torch.full((544, ), 123, dtype=torch.bfloat16, device="cuda")
    kernels.test_embedding[(1, )](weight, output, 512, encoding)
    torch.testing.assert_close(output[:512], torch.from_numpy(expected[3]).cuda().bfloat16(), rtol=0, atol=0)
    assert torch.all(output[512:] == 123)


@pytest.mark.parametrize("encoding", ["IQ4_XS", "Q4_K"])
@pytest.mark.parametrize("family,uniform", [("linear", False), ("glu", False), ("gdn", False), ("gdn", True)])
def test_native_quantized_projection(kernels, encoding, family, uniform):
    packed, expected = _packed_fixture(encoding)
    torch.manual_seed(42)
    value = torch.randn(512, dtype=torch.bfloat16, device="cuda")
    output = torch.full((8, ), 123, dtype=torch.bfloat16, device="cuda")
    kernels.test_projection[(1, )](value, torch.from_numpy(packed).cuda(), output, 5, 512, encoding, family, uniform)
    reference = torch.from_numpy(expected).cuda() @ value.float()
    if uniform:
        reference = reference[3].expand(5)
    torch.testing.assert_close(output[:5], reference[:5].bfloat16(), rtol=0.01, atol=0.005)
    assert torch.all(output[5:] == 123)


@pytest.mark.parametrize("tiled", [False, True])
def test_recurrent_preserves_gguf_value_head_order(kernels, tiled):
    torch.manual_seed(123)
    qkv = torch.randn(40, dtype=torch.bfloat16, device="cuda")
    state = torch.randn((6, 4, 4), dtype=torch.float32, device="cuda")
    old_state = state.clone()
    core = torch.full((32, ), 123, dtype=torch.float32, device="cuda")
    kernels.test_recurrent[(1, )](qkv, state, core, tiled)
    mapping = torch.arange(6, device="cuda")
    mapping = mapping % 2 if tiled else mapping // 3
    query = torch.nn.functional.normalize(qkv[:8].float().reshape(2, 4), dim=-1)[mapping] / 2
    key = torch.nn.functional.normalize(qkv[8:16].float().reshape(2, 4), dim=-1)[mapping]
    value = qkv[16:].float().reshape(6, 4)
    recalled = torch.sum(old_state * 0.9 * key[:, None, :], dim=-1)
    updated = old_state * 0.9 + ((value - recalled) * 0.5)[:, :, None] * key[:, None, :]
    expected = torch.sum(updated * query[:, None, :], dim=-1)
    torch.testing.assert_close(state, updated, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(core[:24], expected.flatten(), rtol=1e-5, atol=1e-5)
    assert torch.all(core[24:] == 123)


@pytest.mark.parametrize('encoding', ['IQ4_XS', 'Q4_K'])
def test_native_quantized_output_head_preserves_float32(kernels, encoding):
    packed, expected = _packed_fixture(encoding)
    torch.manual_seed(42)
    value = torch.randn(512, dtype=torch.bfloat16, device='cuda')
    output = torch.full((8, ), 123, dtype=torch.float32, device='cuda')
    kernels.test_projection[(1, )](value, torch.from_numpy(packed).cuda(), output, 5, 512, encoding, 'linear', False)
    reference = torch.from_numpy(expected).cuda() @ value.float()
    torch.testing.assert_close(output[:5], reference[:5], rtol=1e-5, atol=1e-5)
    assert torch.all(output[5:] == 123)
    assert torch.any(output[:5] != output[:5].bfloat16().float())
