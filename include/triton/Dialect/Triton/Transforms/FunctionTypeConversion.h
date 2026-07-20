/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_DIALECT_TRITON_TRANSFORMS_FUNCTION_TYPE_CONVERSION_H_
#define TRITON_DIALECT_TRITON_TRANSFORMS_FUNCTION_TYPE_CONVERSION_H_
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton {

/**
 * @brief Provides helper patterns for converting triton function operations
 * using a type converter.
 *
 * Note we cannot use upstream passes for this because they are unaware of
 * tt.call and tt.return.
 */
void populateFunctionTypeConversions(const TypeConverter &converter,
                                     RewritePatternSet &patterns);

} // namespace mlir::triton

#endif // TRITON_DIALECT_TRITON_TRANSFORMS_FUNCTION_TYPE_CONVERSION_H_
