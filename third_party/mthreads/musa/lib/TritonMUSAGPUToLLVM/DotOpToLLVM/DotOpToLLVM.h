#ifndef TRITONMUSAGPU_CONVERSION_DOTOP_TO_LLVM_H
#define TRITONMUSAGPU_CONVERSION_DOTOP_TO_LLVM_H

#include "Dialect/MTGPU/IR/Dialect.h"
#include "Dialect/MUSA/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace MUSA {

LogicalResult convertWMMADot(triton::musa::WmmaDotOp op,
                             triton::musa::WmmaDotOp::Adaptor adaptor,
                             const LLVMTypeConverter *typeConverter,
                             ConversionPatternRewriter &rewriter);

LogicalResult convertSQMMADot(triton::musa::SquadDotOp op,
                              triton::musa::SquadDotOp::Adaptor adaptor,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter,
                              Value threadId);

LogicalResult convertSQMMADot(triton::mtgpu::SqmmaOp op,
                              triton::mtgpu::SqmmaOp::Adaptor adaptor,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter,
                              Value threadId);

} // namespace MUSA
} // namespace triton
} // namespace mlir

#endif // TRITONMUSAGPU_CONVERSION_DOTOP_TO_LLVM_H
