# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.ops.math._quantized import decode_native_rows, matrix_width


def q6_rows(rows=3, blocks=2):
    rng = np.random.default_rng(621)
    codes = rng.integers(-32, 32, (rows, blocks, 256), dtype=np.int8)
    scales = rng.integers(-128, 128, (rows, blocks, 16), dtype=np.int8)
    scales[..., :4] = [-128, -1, 0, 127]
    deltas = np.resize(np.array([0.00031, -0.00073, 0.0, -0.0, 2**-24, 0.5], np.float16), (rows, blocks))
    packed = np.zeros((rows, blocks, 210), np.uint8)
    for row in range(rows):
        for block in range(blocks):
            for index, code in enumerate(codes[row, block]):
                half, offset = divmod(index, 128)
                part, lane = divmod(offset, 32)
                value = int(code) + 32
                packed[row, block, half * 64 + (part % 2) * 32 + lane] |= (value & 15) << (4 * (part // 2))
                packed[row, block, 128 + half * 32 + lane] |= (value >> 4) << (2 * part)
            packed[row, block, 192:208] = scales[row, block].view(np.uint8)
            packed[row, block, 208:210] = np.frombuffer(deltas[row, block].tobytes(), np.uint8)
    expected = deltas.astype(np.float32)[..., None] * np.repeat(scales.astype(np.float32), 16, axis=-1)
    expected = expected * codes.astype(np.float32)
    return packed.reshape(rows, -1), expected.reshape(rows, -1)


@pytest.mark.parametrize("blocks", [1, 2, 4])
def test_q6_k_decodes_both_halves_signed_scales_and_trailing_delta(blocks):
    packed, expected = q6_rows(blocks=blocks)
    actual = decode_native_rows(torch.from_numpy(packed), "Q6_K", logical_k=blocks * 256)
    np.testing.assert_array_equal(actual.numpy().view(np.uint32), expected.view(np.uint32))
    assert matrix_width(fm.tensor_type("uint8", packed.shape), "Q6_K").fixed_value == blocks * 256


def test_q6_k_noncontiguous_rows_preserve_input_bytes():
    packed, expected = q6_rows(rows=6)
    storage = torch.from_numpy(packed)
    before = storage.clone()
    actual = decode_native_rows(storage[::2], "Q6_K")
    assert torch.equal(storage, before)
    np.testing.assert_array_equal(actual.numpy().view(np.uint32), expected[::2].view(np.uint32))


def test_q6_k_rejects_incomplete_blocks_wrong_dtype_and_logical_width():
    for weight in (torch.zeros((2, 209), dtype=torch.uint8), torch.zeros((2, 210), dtype=torch.int8)):
        with pytest.raises(EvaluationError, match="complete UINT8 rows"):
            decode_native_rows(weight, "Q6_K")
    with pytest.raises(EvaluationError, match="logical K mismatch"):
        decode_native_rows(torch.zeros((2, 210), dtype=torch.uint8), "Q6_K", logical_k=512)
    with pytest.raises(IRSchemaError, match="whole 256-element"):
        matrix_width(fm.tensor_type("uint8", (2, 209)), "Q6_K")
