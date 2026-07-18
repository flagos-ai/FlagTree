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

#ifndef TRITON_THIRD_PARTY_ILUVATAR_TLE_DIALECT_H_
#define TRITON_THIRD_PARTY_ILUVATAR_TLE_DIALECT_H_

#include "IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton::iluvatar_tle {

inline void registerDialects(DialectRegistry &registry) {
  registry.insert<mlir::triton::iluvatar_tle::IluvatarTleDialect>();
}

inline void addIllegalDialects(ConversionTarget &target) {
  target.addIllegalDialect<mlir::triton::iluvatar_tle::IluvatarTleDialect>();
}

} // namespace mlir::triton::iluvatar_tle

#endif // TRITON_THIRD_PARTY_ILUVATAR_TLE_DIALECT_H_
