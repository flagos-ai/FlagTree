/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITONNPU_UTILS_UTILS_H
#define TRITONNPU_UTILS_UTILS_H

#include "TritonMemoryAccess/OpFoldResultUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/LogicalResult.h"

#include <functional>
#include <optional>
#include <string>

namespace mlir {
namespace ConverterUtils {

const std::string GeneratedByMakeTensorPtrTAG = "GeneratedByMakeTensorPtr";
const std::string discreteMaskAttrName = "DiscreteMask";
const std::string mixCompileDiscreteMaskAttrName = "MixCompileDiscreteMask";
const std::string explicitRouteToSIMDAttrName = "ExplicitRouteToSIMD";
// The access must consume the actual loop mask; full-load/select is unsafe.
const std::string runtimeLoopMaskAttrName = "RuntimeLoopMask";
const std::string discreteAttrName = "DiscreteMemAccess";
const std::string continuousAttrName = "ContinuousMemAccess";
const std::string customSrcPtrIndexAttrName = "SrcPtrIndex";

bool isaPermutedMemRefType(MemRefType);

std::optional<int64_t>
getLastStrideOfReinterpretCastOp(memref::ReinterpretCastOp op);

Value getTransposedValue(Value source, const Location loc,
                         ConversionPatternRewriter &rewriter,
                         llvm::ArrayRef<int> order);

SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n);

Value getScalarValue(Value operand, Location loc,
                     ConversionPatternRewriter &rewriter);

memref::SubViewOp makeSubViewOp(Value src,
                                const llvm::SmallVector<OpFoldResult> &offsets,
                                const llvm::SmallVector<OpFoldResult> &sizes,
                                const Location &loc,
                                ConversionPatternRewriter &rewriter);

tensor::ExtractSliceOp
makeExtractSliceOp(Value src, const llvm::SmallVector<OpFoldResult> &offsets,
                   const llvm::SmallVector<OpFoldResult> &sizes,
                   const Location &loc, ConversionPatternRewriter &rewriter);

std::optional<Operation *> getFullShapeOp(Value val,
                                          ConversionPatternRewriter &rewriter);

SmallVector<OpFoldResult>
getBoundarySizes(llvm::ArrayRef<int32_t> boundaryCheck, Value ptr,
                 const Location &loc, ConversionPatternRewriter &rewriter);

bool requiresVolatileIndirectLoad(Value srcPtr, Operation *loadOp);

SmallVector<int64_t> getBroadcastDims(RankedTensorType src,
                                      RankedTensorType dst);

SmallVector<int64_t> getUnbroadcastDims(RankedTensorType src,
                                        RankedTensorType dst);

} // namespace ConverterUtils

class ConversionPatternRewriter;

namespace triton {

namespace ascend {

// Keep the C++ pass contract aligned with the public Python selector.  The
// Python boundary performs the target validation and canonicalization; pass
// parsing remains strict so direct pass-pipeline users cannot silently turn an
// unknown string into SIMD.
enum class CompileMode {
  Simd,
  SimtTemplate,
  SimtOnly,
};

inline std::optional<CompileMode> parseCompileMode(llvm::StringRef mode) {
  if (mode == "simd")
    return CompileMode::Simd;
  if (mode == "simd_simt_template" || mode == "unstructured_in_simt")
    return CompileMode::SimtTemplate;
  if (mode == "simt_only")
    return CompileMode::SimtOnly;
  return std::nullopt;
}

inline bool isSimtTemplateMode(CompileMode mode) {
  return mode == CompileMode::SimtTemplate;
}

inline bool isPureSimtMode(CompileMode mode) {
  return mode == CompileMode::SimtOnly;
}

} // namespace ascend

enum class IndirectLoadInterfaceOpType { Undefined = 0, Load = 1, Calc = 2 };

// Traceback from rootOp to find the targetOp with the specified condition
mlir::Operation *
findFirstMatchingOperandDef(mlir::Operation *rootOp,
                            const std::function<bool(Operation *)> &condFn);

void traverseBackwardUpdateOperandChainIf(
    Operation *op, std::function<bool(Operation *)> conditionFn,
    std::function<bool(Operation *)> stopFn,
    std::function<void(OpBuilder &, Operation *)> actionFn, OpBuilder &builder,
    DenseSet<Operation *> &handledOperation);

void traverseBackwardUpdateOperandChainIf(
    Operation *rootOp, std::function<bool(Operation *)> conditionFn,
    std::function<bool(Operation *)> stopFn,
    std::function<void(OpBuilder &, Operation *)> actionFn);

void traverseForwardUpdateUserChainIf(
    Operation *op, std::function<bool(Operation *)> conditionFn,
    std::function<bool(Operation *)> stopFn,
    std::function<void(OpBuilder &, Operation *)> actionFn, OpBuilder &builder,
    llvm::SmallPtrSet<Operation *, 16> &stopOps);

void traverseForwardUpdateUserChainIf(
    Operation *rootOp, std::function<bool(Operation *)> conditionFn,
    std::function<bool(Operation *)> stopFn,
    std::function<void(OpBuilder &, Operation *)> actionFn,
    llvm::SmallPtrSet<Operation *, 16> &stopOps);

// UseAnalysis will tag operations whose results are used only as meta-data
// with "MetaUse" tag.
bool isMetaUse(Operation *op);

bool isMixUse(Operation *op);

IndirectLoadInterfaceOpType getIndirectLoadInterfaceOpType(Operation *op);

bool opIsIndirectLoad(Operation *op);

bool opIsIndirectCalc(Operation *op);

/// Maximum expected rank for loop tiling in tensor operations.
static constexpr int kMaxTiledRank = 4;

/// This function generates a series of `scf.for` loops for the given dimensions
/// in `loopDims`. Although the loops are created sequentially, nesting is
/// simulated by adjusting the insertion point to the body of the last created
/// loop. This allows the `bodyFunc` to be inserted into the innermost scope.
///
/// \param rewriter The MLIR OpBuilder used to create operations.
/// \param loc The source location information for debuggability.
/// \param target The memref value whose dimensions are being looped over.
/// \param loopDims An array of dimension indices to create loops for.
/// \param bodyFunc A callable that defines the operations to insert in the
/// innermost loop.
///                 It takes a SmallVector of induction variables (one per
///                 loop).
///
template <typename Func>
void createSimpleNestedLoops(OpBuilder &rewriter, Location loc, Value target,
                             ArrayRef<int> loopDims, Func bodyFunc) {
  MemRefType type = cast<MemRefType>(target.getType());
  int rank = type.getRank();

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  llvm::SmallVector<scf::ForOp, kMaxTiledRank> loops;
  llvm::SmallVector<Value, kMaxTiledRank> ivs;

  for (int dim : loopDims) {
    Value ub;
    if (type.isDynamicDim(dim)) {
      ub = rewriter.create<memref::DimOp>(loc, target, dim).getResult();
    } else {
      ub = rewriter.create<arith::ConstantIndexOp>(loc, type.getDimSize(dim));
    }

    auto forOp = rewriter.create<scf::ForOp>(loc, zero, ub, one);
    rewriter.setInsertionPointToStart(forOp.getBody());
    loops.push_back(forOp);
    ivs.push_back(forOp.getInductionVar());
  }

  bodyFunc(ivs);

  if (!loops.empty()) {
    rewriter.setInsertionPointAfter(loops.front());
  }
}

scf::ForOp createNestedLoops(
    OpBuilder &builder, Location loc, unsigned currentDim, unsigned totalDims,
    ValueRange LBs, ValueRange UBs, ValueRange steps, SmallVector<Value> &ivs,
    ValueRange initArgs,
    function_ref<void(OpBuilder &, Location, SmallVector<Value> &, ValueRange)>
        bodyBuilder);

ModuleOp getModuleOpFromOperation(Operation *op);

bool isTensorPtrType(Type type);

namespace ascend {

// Fractal (zN) block geometry shared by ascend.dot's type inference/verifier
// (DotOp::inferReturnTypes) and its lowering (DotConverter). The inner block is
// [kFractalBlock, b] with b = kBytesPerFractalCol / elemBytes, i.e. f16/bf16 ->
// 16, f32 -> 8, int8 -> 32; fractal_c is the L0C accumulator fractal, always
// [kFractalBlock, kFractalBlock].
inline constexpr int64_t kFractalBlock = 16;
inline constexpr int64_t kBytesPerFractalCol = 32;
inline constexpr unsigned kBitsPerByte = 8;
// Cube accumulator width for integer inputs (int8 -> i32).
inline constexpr unsigned kDotAccIntWidth = 32;

} // namespace ascend

} // namespace triton

class OpBuilder;

/// Returns true when both ranges have identical ordered type signatures.
bool haveSameTypes(TypeRange lhs, TypeRange rhs);

enum class ReduceWithIndexType { MAX, MIN, None };
enum class TieBreakType { LEFT, RIGHT, None };

struct ReduceWithIndexParams {
  ReduceWithIndexType withIndexType = ReduceWithIndexType::None;
  TieBreakType tieBreakType = TieBreakType::None;
  bool isUnsignedSrc;
};

llvm::FailureOr<ReduceWithIndexParams>
getReduceWithIndexParams(triton::ReduceOp op);

void addReduceWithIndexAttr(ReduceWithIndexParams params,
                            ConversionPatternRewriter &rewriter,
                            linalg::ReduceOp reduceOp);

enum class TypelessValue { Undefined = 0, Zero = 1, Min = 2, Max = 3 };

FailureOr<TypedAttr> specializeTypelessValueToAttr(TypelessValue, Type,
                                                   OpBuilder &);

FailureOr<Value> specializeTypelessValueToConstant(TypelessValue, Type,
                                                   Location, OpBuilder &);

bool checkStructureAnnotated(Operation *op, RewriterBase &rewriter);

bool isDistributedTypeCustomOp(Operation *op);
} // namespace mlir

#endif // TRITONNPU_UTILS_UTILS_H
