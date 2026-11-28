# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Independent IQ4 test data packing."""

import numpy as np

LUT = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)


def packed_rows(encoding, rows=4, blocks=2):
    """Independent packing oracle with distinct rows, scales and both nibbles."""
    rng = np.random.default_rng(613)
    size = 136 if encoding == 'IQ4_XS' else 144
    packed = np.zeros((rows, blocks, size), np.uint8)
    decoded = np.empty((rows, blocks, 256), np.float32)
    for row in range(rows):
        for block in range(blocks):
            data = packed[row, block]
            d = np.float16((row + block + 1) / 1024)
            data[:2] = np.frombuffer(d.tobytes(), np.uint8)
            codes = rng.integers(0, 16, (8, 32), dtype=np.uint8)
            if encoding == 'IQ4_XS':
                scales = np.array([-32, -17, -1, 0, 1, 15, 30, 31], np.int32)
                scales = np.roll(scales, row + block)
                unsigned = scales + 32
                high = sum(int(unsigned[g] >> 4) << (2 * g) for g in range(8))
                data[2:4] = np.frombuffer(np.uint16(high).tobytes(), np.uint8)
                for g in range(8):
                    data[4 + g // 2] |= np.uint8((unsigned[g] & 15) << (4 * (g % 2)))
                    data[8 + g * 16:8 + (g + 1) * 16] = codes[g, :16] | (codes[g, 16:] << 4)
                decoded[row, block] = (np.float32(d) * scales[:, None] * LUT[codes]).reshape(-1)
            else:
                dmin = np.float16((row + 1) / 2048)
                data[2:4] = np.frombuffer(dmin.tobytes(), np.uint8)
                scales = rng.integers(0, 64, 8, dtype=np.uint8)
                mins = rng.integers(0, 64, 8, dtype=np.uint8)
                for g in range(4):
                    data[4 + g] = scales[g] | ((scales[g + 4] >> 4) << 6)
                    data[8 + g] = mins[g] | ((mins[g + 4] >> 4) << 6)
                    data[12 + g] = (scales[g + 4] & 15) | ((mins[g + 4] & 15) << 4)
                for g in range(4):
                    data[16 + g * 32:16 + (g + 1) * 32] = codes[2 * g] | (codes[2 * g + 1] << 4)
                decoded[row, block] = (np.float32(d) * scales[:, None] * codes -
                                       np.float32(dmin) * mins[:, None]).reshape(-1)
    return packed.reshape(rows, -1), decoded.reshape(rows, -1)
