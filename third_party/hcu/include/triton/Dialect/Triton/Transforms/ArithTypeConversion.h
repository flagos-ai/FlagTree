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

#ifndef TRITON_DIALECT_TRITON_TRANSFORMS_ARITH_TYPE_CONVERSION_H_
#define TRITON_DIALECT_TRITON_TRANSFORMS_ARITH_TYPE_CONVERSION_H_
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton {

/**
 * @brief Provides helper patterns for converting arith operations using a type
 * converter.
 *
 * Note at of the time of writing this isn't provided in upstream mlir.
 */
void populateArithTypeConversions(const TypeConverter &converter,
                                  RewritePatternSet &patterns);

} // namespace mlir::triton

#endif // TRITON_DIALECT_TRITON_TRANSFORMS_ARITH_TYPE_CONVERSION_H_
