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
#if LLVM_VERSION_MAJOR < 21
inline Type f16Ty(MLIRContext *ctx) { return FloatType::getF16(ctx); }
inline Type f32Ty(MLIRContext *ctx) { return FloatType::getF32(ctx); }
inline Type f64Ty(MLIRContext *ctx) { return FloatType::getF64(ctx); }
inline Type bf16Ty(MLIRContext *ctx) { return FloatType::getBF16(ctx); }
#else // triton_v3.3.x
inline Type f16Ty(MLIRContext *ctx) { return Float16Type::get(ctx); }
inline Type f32Ty(MLIRContext *ctx) { return Float32Type::get(ctx); }
inline Type f64Ty(MLIRContext *ctx) { return Float64Type::get(ctx); }
inline Type bf16Ty(MLIRContext *ctx) { return BFloat16Type::get(ctx); }
#endif

#if LLVM_VERSION_MAJOR < 21

inline bool isFloat(Type type) {
  return type.isF32() || type.isF64() || type.isF16() || type.isF128() ||
         type.isBF16() || type.isFloat8E4M3B11FNUZ() || type.isFloat8E4M3FN() ||
         type.isFloat8E4M3FNUZ() || type.isFloat8E5M2() ||
         type.isFloat8E5M2FNUZ();
}

inline bool isFloat8(Type type) {
  return type.isFloat8E4M3B11FNUZ() || type.isFloat8E4M3FN() ||
         type.isFloat8E4M3FNUZ() || type.isFloat8E5M2() ||
         type.isFloat8E5M2FNUZ();
}

#else // triton_v3.3.x

inline bool isFloat8(Type type) {
  return isa<Float8E4M3B11FNUZType, Float8E4M3FNType, Float8E4M3FNUZType,
             Float8E5M2Type, Float8E5M2FNUZType>(type);
}

inline bool isFloat(Type type) {
  return type.isF32() || type.isF64() || type.isF16() || type.isF128() ||
         type.isBF16() || llvm::isa<Float8E4M3B11FNUZType>(type) ||
         isFloat8(type);
}

#endif

inline bool isInt(Type type) { return type.isIntOrFloat() && !isFloat(type); }

} // namespace type
} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_MLIR_TYPES_H
