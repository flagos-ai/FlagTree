// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once
#include <stdint.h>

// OpenAI change: Switched from `extern "C"` to `namespace f2reduce`.
namespace f2reduce {

/**
 * Converts a matrix over F_2 into row-reduced echelon form.
 *
 * The matrix should be in row-major format. The stride parameter specifies
 * the offset (in 64-bit words, *not* bytes!) between successive rows of the
 * matrix, and should obey the inequality:
 *
 *     64 |stride| >= cols
 *
 * i.e. that the rows occupy disjoint regions of memory. For best performance
 * the stride should be divisible by 16 words (128 bytes).
 *
 * We adopt 'little-endian' semantics: the element in row i and column j+64*k
 * of the matrix (zero-indexed) is given by (matrix[i * stride + k] >> j) & 1.
 *
 * The matrix is overwritten in place with its row-reduced echelon form.
 */
void inplace_rref_strided(uint64_t *matrix, uint64_t rows, uint64_t cols, uint64_t stride);

uint64_t get_recommended_stride(uint64_t cols);

}  // namespace f2reduce
