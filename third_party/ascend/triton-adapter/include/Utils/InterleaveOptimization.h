#pragma once

#include "TritonToLinalg/BlockPtrAnalysis.h"
#include "TritonToLinalg/MaskAnalysis.h"
#include "TritonToLinalg/UseAnalysis.h"
#include "Utils/Utils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <numeric>
#include <type_traits>

namespace mlir {
namespace triton {

enum class IndexMode : int { EVEN_MODE = 0, ODD_MODE = 1 };

MemRefType expandInterleaveMemRefType(MemRefType originType);

std::pair<OpFoldResult, IndexMode>
recountReinterpretCastOffset(OpFoldResult originOffset, Builder &builder);

LogicalResult
DeinterleaveStatusOptimization(triton::LoadOp op,
                               triton::LoadOp::Adaptor adaptor,
                               ConversionPatternRewriter &rewriter);

LogicalResult DeinterleaveStatusWithMaskOptimization(
    triton::LoadOp op, triton::LoadOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter, MaskState &mstate,
    memref::AllocOp originAllocOp);

LogicalResult
InterleaveStatusOptimization(SmallVector<Operation *> materializeVec);

LogicalResult
InterleaveStatusWithMaskOptimization(SmallVector<Operation *> materializeVec);

} // namespace triton
} // namespace mlir