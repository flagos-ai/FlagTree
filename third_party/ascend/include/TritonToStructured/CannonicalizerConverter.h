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
#ifndef TRITON_ADAPTER_CANNONICALIZERCONVERTER_H
#define TRITON_ADAPTER_CANNONICALIZERCONVERTER_H

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace CannonicalizerConverter {

using namespace mlir;
using namespace triton;

class CmpConverter : public OpRewritePattern<arith::CmpIOp> {
public:
  explicit CmpConverter(MLIRContext *context)
      : OpRewritePattern<arith::CmpIOp>(context) {}

  using OpRewritePattern<arith::CmpIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp cmpOp,
                                PatternRewriter &rewriter) const override;
};

class SplatCmpConverter : public OpRewritePattern<arith::CmpIOp> {
public:
  explicit SplatCmpConverter(MLIRContext *context)
      : OpRewritePattern<arith::CmpIOp>(context) {}
  using OpRewritePattern<arith::CmpIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp op,
                                PatternRewriter &rewriter) const override;
};

class AddPtrSplatConverter : public OpRewritePattern<triton::AddPtrOp> {
public:
  explicit AddPtrSplatConverter(MLIRContext *context)
      : OpRewritePattern<triton::AddPtrOp>(context) {}

  LogicalResult matchAndRewrite(triton::AddPtrOp op,
                                PatternRewriter &rewriter) const override;
};

class LoadBroadcastConverter : public OpRewritePattern<triton::LoadOp> {
public:
  explicit LoadBroadcastConverter(MLIRContext *context)
      : OpRewritePattern<triton::LoadOp>(context) {}

  LogicalResult matchAndRewrite(triton::LoadOp op,
                                PatternRewriter &rewriter) const override;
};

class StoreBroadcastConverter : public OpRewritePattern<triton::StoreOp> {
public:
  explicit StoreBroadcastConverter(MLIRContext *context)
      : OpRewritePattern<triton::StoreOp>(context) {}

  LogicalResult matchAndRewrite(triton::StoreOp op,
                                PatternRewriter &rewriter) const override;
};

// Rewrites `tt.load %bptr` whose `%bptr = tt.make_tensor_ptr ...` has all
// statically zero strides. The zero strides are the IR-level signature of a
// broadcasted tensor: every element of the loaded block refers to the same
// memory location `base`. Lower the load into a single scalar `tt.load %base`
// followed by `tt.broadcast` back to the original block shape, so downstream
// passes see only standard ops.
class ZeroStrideMakeTensorPtrConverter
    : public OpRewritePattern<triton::MakeTensorPtrOp> {
public:
  explicit ZeroStrideMakeTensorPtrConverter(MLIRContext *context)
      : OpRewritePattern<triton::MakeTensorPtrOp>(context) {}

  LogicalResult matchAndRewrite(triton::MakeTensorPtrOp op,
                                PatternRewriter &rewriter) const override;
};

class IfYieldAddHoistConverter : public OpRewritePattern<scf::IfOp> {
public:
  explicit IfYieldAddHoistConverter(MLIRContext *context)
      : OpRewritePattern<scf::IfOp>(context) {}

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override;

private:
  bool isSupportedTensorResultType(Type type) const;

  bool isDefinedOutsideIf(Value value, scf::IfOp ifOp) const;

  bool extractAddendFromAddExpr(Value maybeAddExpr, Value baseValue,
                                Value &addendOut) const;

  Value buildZeroTensorLikeType(Type laneType, Location loc,
                                PatternRewriter &rewriter) const;

  bool tryRewriteSingleLane(unsigned laneIdx, Value baseBranchYield,
                            Value addExprBranchYield, bool baseInThenBranch,
                            Type laneType, scf::IfOp ifOp,
                            PatternRewriter &rewriter,
                            SmallVectorImpl<Value> &updatedThenYieldOperands,
                            SmallVectorImpl<Value> &updatedElseYieldOperands,
                            SmallVectorImpl<Value> &hoistedBasePerLane,
                            SmallVectorImpl<bool> &laneRewrittenFlags) const;
};

class PromotePointerIterArgsPattern : public OpRewritePattern<scf::ForOp> {
public:
  explicit PromotePointerIterArgsPattern(MLIRContext *context)
      : OpRewritePattern<scf::ForOp>(context) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override;

private:
  // Information about a pointer iteration argument to be promoted
  struct PointerArgInfo {
    unsigned oldIndex; // Original index in the iteration arguments
    Value basePointer; // Base pointer value passed as init arg
    Value offsetValue; // Offset value used in addptr operation
    Value newIterArg;  // New integer iteration argument
    Value addPtrValue; // The addptr operation result that updates the pointer
    // Offset value used in advancePtr operation (with explicit inlined
    // capacity)
    SmallVector<Value> offsetValues;
    SmallVector<Value> newInitArgs;
    SmallVector<Type> newIterArgTypes;
  };

  LogicalResult matchAndRewriteForAddPtr(scf::ForOp forOp,
                                         PatternRewriter &rewriter) const;

  // Check if the loop meets basic transformation conditions
  LogicalResult matchLoop(scf::ForOp forOp) const;

  // Collect all pointer iteration arguments that match the promotion pattern
  SmallVector<PointerArgInfo> collectPointerIterArgs(scf::ForOp forOp) const;

  // Check if a value has pointer tensor type
  bool isPointerIterArg(Value iterArg) const;

  // Analyze a pointer iteration argument to determine if it matches the
  // promotion pattern
  std::optional<PointerArgInfo> analyzePointerIterArg(Value iterArg,
                                                      Block &loopBody) const;

  // Check if an index corresponds to a pointer argument being promoted
  bool isPointerArgIndex(ArrayRef<PointerArgInfo> pointerArgs,
                         unsigned idx) const;

  // Get pointer argument information for a specific index
  const PointerArgInfo *getPointerArgInfo(ArrayRef<PointerArgInfo> pointerArgs,
                                          unsigned idx) const;

  // Create a new for loop with updated iteration argument types
  scf::ForOp createNewForLoop(scf::ForOp forOp, ArrayRef<Value> newInitArgs,
                              ArrayRef<Type> newIterArgTypes,
                              PatternRewriter &rewriter) const;

  // Rewrite the loop body to use integer iteration arguments instead of
  // pointers
  LogicalResult rewriteLoopBody(scf::ForOp oldForOp, scf::ForOp newForOp,
                                SmallVector<PointerArgInfo> &pointerArgs,
                                DenseMap<unsigned, unsigned> &indexMap,
                                PatternRewriter &rewriter) const;

  // Create new iteration arguments by replacing pointers with integer offsets
  std::tuple<SmallVector<Value>, SmallVector<Type>,
             DenseMap<unsigned, unsigned>>
  createNewIterArgs(scf::ForOp forOp, ArrayRef<PointerArgInfo> pointerArgs,
                    PatternRewriter &rewriter) const;

  // Create IR mapping for cloning operations, rebuilding pointers from integer
  // offsets
  IRMapping createIRMapping(scf::ForOp oldForOp, scf::ForOp newForOp,
                            SmallVector<PointerArgInfo> &pointerArgs,
                            DenseMap<unsigned, unsigned> &indexMap,
                            PatternRewriter &rewriter) const;

  // Reconstruct a pointer value from base pointer and integer offset
  Value rebuildPointer(scf::ForOp forOp, ArrayRef<PointerArgInfo> pointerArgs,
                       unsigned idx, PatternRewriter &rewriter) const;

  // Clone instructions from old loop body to new loop body, skipping
  // transformed addptr ops
  LogicalResult cloneInstructions(Block &oldBody, Block &newBody,
                                  ArrayRef<PointerArgInfo> pointerArgs,
                                  DenseMap<unsigned, unsigned> &indexMap,
                                  IRMapping &mapping,
                                  PatternRewriter &rewriter) const;

  // Clone and transform the yield operation, converting pointer updates to
  // integer additions
  LogicalResult cloneYieldOp(scf::YieldOp yieldOp,
                             ArrayRef<PointerArgInfo> pointerArgs,
                             DenseMap<unsigned, unsigned> &indexMap,
                             IRMapping &mapping,
                             PatternRewriter &rewriter) const;

  // Create integer addition for pointer offset updates in the yield operation
  Value createIntegerAdd(unsigned idx, ArrayRef<PointerArgInfo> pointerArgs,
                         DenseMap<unsigned, unsigned> &indexMap,
                         PatternRewriter &rewriter) const;

  // Extract constant integer value from offset (handles both scalar and tensor
  // constants)
  std::optional<int64_t> extractConstantOffset(Value offsetValue) const;

  // Replace the original loop results with reconstructed pointers from integer
  // results
  LogicalResult replaceResults(scf::ForOp oldForOp, scf::ForOp newForOp,
                               ArrayRef<PointerArgInfo> pointerArgs,
                               DenseMap<unsigned, unsigned> &indexMap,
                               PatternRewriter &rewriter) const;

  // Reconstruct final pointer from integer result after the loop
  Value reconstructPointer(scf::ForOp forOp, unsigned idx, Value intResult,
                           ArrayRef<PointerArgInfo> pointerArgs,
                           PatternRewriter &rewriter) const;

  LogicalResult matchAndRewriteAdvancePtr(scf::ForOp forOp,
                                          PatternRewriter &rewriter) const;

  SmallVector<PointerArgInfo>
  collectPointerIterArgsForAdvancePtr(scf::ForOp forOp) const;

  std::optional<PointerArgInfo>
  analyzePointerIterArgForAdvancePtr(Value iterArg, Block &loopBody) const;

  std::tuple<SmallVector<Value>, SmallVector<Type>,
             DenseMap<unsigned, unsigned>>
  createNewIterArgsForAdvancePtr(scf::ForOp forOp,
                                 SmallVector<PointerArgInfo> &pointerArgs,
                                 PatternRewriter &rewriter) const;

  IRMapping
  createIRMappingForAdvancePtr(scf::ForOp oldForOp, scf::ForOp newForOp,
                               SmallVector<PointerArgInfo> &pointerArgs,
                               DenseMap<unsigned, unsigned> &indexMap,
                               PatternRewriter &rewriter) const;

  Value rebuildPointerForAdvancePtr(scf::ForOp forOp,
                                    ArrayRef<PointerArgInfo> pointerArgs,
                                    unsigned idx,
                                    PatternRewriter &rewriter) const;

  SmallVector<Value>
  createOffsetsForAdvancePtr(unsigned idx, ArrayRef<PointerArgInfo> pointerArgs,
                             DenseMap<unsigned, unsigned> &indexMap,
                             PatternRewriter &rewriter) const;

  LogicalResult cloneInstructionsForAdvancePtr(
      Block &oldBody, Block &newBody, ArrayRef<PointerArgInfo> pointerArgs,
      DenseMap<unsigned, unsigned> &indexMap, IRMapping &mapping,
      PatternRewriter &rewriter) const;

  LogicalResult
  rewriteLoopBodyForAdvancePtr(scf::ForOp oldForOp, scf::ForOp newForOp,
                               SmallVector<PointerArgInfo> &pointerArgs,
                               DenseMap<unsigned, unsigned> &indexMap,
                               PatternRewriter &rewriter) const;

  LogicalResult cloneYieldOpForAdvancePtr(
      scf::YieldOp yieldOp, ArrayRef<PointerArgInfo> pointerArgs,
      DenseMap<unsigned, unsigned> &indexMap, IRMapping &mapping,
      PatternRewriter &rewriter) const;

  SmallVector<Value>
  reconstructPointerForAdvance(scf::ForOp forOp, unsigned idx, Value intResult,
                               ArrayRef<PointerArgInfo> pointerArgs,
                               PatternRewriter &rewriter) const;
};

class SimplifyTensorIterArgsPattern : public OpRewritePattern<scf::ForOp> {
public:
  explicit SimplifyTensorIterArgsPattern(MLIRContext *context)
      : OpRewritePattern<scf::ForOp>(context) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override;

private:
  static constexpr llvm::StringLiteral kSimplifiedAttr =
      "tts.simplify_tensor_iter_args.done";
  static constexpr llvm::StringLiteral kFailedAttr =
      "tts.simplify_tensor_iter_args.failed";
  static constexpr llvm::StringLiteral kIncompleteAttr =
      "tts.simplify_tensor_iter_args.incomplete";
  struct ShapeChainInfo {
    Value base;
    SmallVector<Operation *> chain; // from base -> iter shape
    void dump() const;
  };

  struct RelayMapM1 {
    unsigned innerIdx;
    unsigned outerInitIdx;  // from inner.initArg block-arg mapping
    unsigned outerYieldIdx; // from outer.yield operand position of inner result
  };

  struct CandidateInfo {
    unsigned idx;
    ShapeChainInfo shapeInfo;
    SmallVector<Operation *> arithOps; // in execution order
    std::optional<RelayMapM1> relayMap;
    void dump() const;
  };

  bool isBlockArgumentFromAnotherForLoop(Value v) const;
  std::optional<RelayMapM1> getRelayMapM1(scf::ForOp innerFor,
                                          scf::ForOp outerFor,
                                          unsigned innerIdx) const;
  void splitCandidatesByRelay(SmallVector<CandidateInfo> all,
                              SmallVector<CandidateInfo> &locals,
                              SmallVector<CandidateInfo> &relays) const;

  Value cloneShapeChain(Location loc, Value base, ArrayRef<Operation *> chain,
                        PatternRewriter &rewriter) const;
  Value normalizeInitArgForShapePeel(Value v) const;
  std::optional<ShapeChainInfo> peelShapeChain(Value v) const;
  bool isArithWithConst(Operation *op, Value curVal, Value &nextVal,
                        Value &constVal) const;
  Value getNewConstLikeOperand(Value cst, Type targetTy,
                               PatternRewriter &rewriter) const;
  bool canBuildConstLikeOperand(Value cst, Type targetTy) const;
  LogicalResult collectReverseLinearYieldPath(
      Value yielded, Value iterArg,
      SmallVectorImpl<Operation *> &opsInExecOrder) const;

  bool extractBinaryArithOperands(Operation *op, Value &lhs, Value &rhs) const;
  Value createSameBinaryArithOp(Operation *oldOp, Location loc, Value lhs,
                                Value rhs, PatternRewriter &rewriter) const;
  bool
  isSafeToRewriteLanesByResultUses(scf::ForOp forOp,
                                   ArrayRef<CandidateInfo> candidates) const;
  FailureOr<scf::ForOp> rewriteForWithLocalCandidates(
      scf::ForOp forOp, ArrayRef<CandidateInfo> candidates,
      const IRMapping *outerCaptureMap, PatternRewriter &rewriter) const;
  LogicalResult precheckRelayCandidates(scf::ForOp innerFor,
                                        ArrayRef<CandidateInfo> relayCandidates,
                                        scf::ForOp &outerForOut) const;
  FailureOr<scf::ForOp> rewriteInnerForWithRelayCandidates(
      scf::ForOp innerFor, ArrayRef<CandidateInfo> relayCandidates,
      const IRMapping *outerCaptureMap, PatternRewriter &rewriter) const;
  FailureOr<scf::ForOp> rewriteOuterForWithRelayCandidates(
      scf::ForOp innerFor, scf::ForOp oldInnerFor, scf::ForOp outerFor,
      ArrayRef<CandidateInfo> relayCandidates, PatternRewriter &rewriter) const;
  LogicalResult
  rewriteForWithRelayCandidates(scf::ForOp newfor, scf::ForOp oldFor,
                                ArrayRef<CandidateInfo> relayCandidates,
                                PatternRewriter &rewriter) const;
};
} // namespace CannonicalizerConverter

#endif
