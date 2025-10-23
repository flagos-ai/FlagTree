#ifndef TRITONNPU_UTILS_UTILS_H
#define TRITONNPU_UTILS_UTILS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"

#include <functional>
#include <optional>

namespace mlir {

namespace ConverterUtils {

const std::string GeneratedByMakeTensorPtrTAG = "GeneratedByMakeTensorPtr";
const std::string discreteMaskAttrName = "DiscreteMask";
const std::string discreteAttrName = "DiscreteMemAccess";

bool isaPermutedMemRefType(MemRefType);

std::optional<int64_t> getLastStrideOfReinterpretCastOp(memref::ReinterpretCastOp op);

Value getTransposedValue(Value source, const Location loc,
                         ConversionPatternRewriter &rewriter,
                         llvm::ArrayRef<int> order);

SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n);

Value getScalarValue(Value operand, Location loc,
                     ConversionPatternRewriter &rewriter);

memref::SubViewOp makeSubViewOp(Value src,
                                const llvm::SmallVector<OpFoldResult> &sizes,
                                const Location &loc,
                                ConversionPatternRewriter &rewriter);

tensor::ExtractSliceOp makeExtractSliceOp(Value src,
                                          const llvm::SmallVector<OpFoldResult> &sizes,
                                          const Location &loc,
                                          ConversionPatternRewriter &rewriter);

std::optional<Operation *> getFullShapeOp(Value val,
                                          ConversionPatternRewriter &rewriter);

SmallVector<OpFoldResult>
getBoundarySizes(llvm::ArrayRef<int32_t> boundaryCheck, Value ptr,
                 const Location &loc, ConversionPatternRewriter &rewriter);

SmallVector<int64_t> getBroadcastDims(RankedTensorType src,
                                      RankedTensorType dst);

SmallVector<int64_t> getUnbroadcastDims(RankedTensorType src,
                                        RankedTensorType dst);

} // namespace ConverterUtils

class ConversionPatternRewriter;

namespace triton {

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

} // namespace triton

class OpBuilder;

OpFoldResult addOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b);

OpFoldResult subOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b);

OpFoldResult mulOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b);

OpFoldResult divOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b);

OpFoldResult remOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b);

OpFoldResult minOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b);

OpFoldResult maxOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b);

LogicalResult
addReduceWithIndexAttrIfNeeded(ConversionPatternRewriter &rewriter,
                               linalg::ReduceOp reduceOp);

OpFoldResult getOpFoldResultOfLayoutInfo(Value value, OpBuilder &builder);

} // namespace mlir

#endif // TRITONNPU_UTILS_UTILS_H