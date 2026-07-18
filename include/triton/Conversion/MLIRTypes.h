// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITON_CONVERSION_MLIR_TYPES_H
#define TRITON_CONVERSION_MLIR_TYPES_H

#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

// This file redefines some common MLIR types for easy usage.
namespace mlir {
namespace triton {
namespace type {

// Integer types
inline Type i32Ty(MLIRContext *ctx) { return IntegerType::get(ctx, 32); }
inline Type i16Ty(MLIRContext *ctx) { return IntegerType::get(ctx, 16); }
inline Type i8Ty(MLIRContext *ctx) { return IntegerType::get(ctx, 8); }
inline Type u32Ty(MLIRContext *ctx) {
  return IntegerType::get(ctx, 32, IntegerType::Unsigned);
}
inline Type u1Ty(MLIRContext *ctx) {
  return IntegerType::get(ctx, 1, IntegerType::Unsigned);
}

// Float types
inline Type f16Ty(MLIRContext *ctx) { return Float16Type::get(ctx); }
inline Type f32Ty(MLIRContext *ctx) { return Float32Type::get(ctx); }
inline Type f64Ty(MLIRContext *ctx) { return Float64Type::get(ctx); }
inline Type bf16Ty(MLIRContext *ctx) { return BFloat16Type::get(ctx); }

inline bool isFloat8(Type type) {
  return isa<Float8E4M3B11FNUZType, Float8E4M3FNType, Float8E4M3FNUZType,
             Float8E5M2Type, Float8E5M2FNUZType>(type);
}

inline bool isFloat(Type type) {
  return type.isF32() || type.isF64() || type.isF16() || type.isF128() ||
         type.isBF16() || llvm::isa<Float8E4M3B11FNUZType>(type) ||
         isFloat8(type);
}

inline bool isInt(Type type) { return type.isIntOrFloat() && !isFloat(type); }

} // namespace type
} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_MLIR_TYPES_H
