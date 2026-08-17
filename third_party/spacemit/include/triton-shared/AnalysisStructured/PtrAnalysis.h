//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ANALYSISSTRUCTURED_PTRANALYSIS_H
#define TRITON_ANALYSISSTRUCTURED_PTRANALYSIS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include <cstddef>
#include <set>

namespace mlir {

class OpBuilder;

namespace tts {

const extern std::string ptrAnalysisAttr;

// Data structure used to decode pointer arithmetics. offsets, sizes, and
// strides are in unit of elements in a linearly laid-out memory, which is the
// same as pointer arithmetic operations in Triton language. scalar is a
// shortcut used when the entire state describes a single scalar value. source
// is the base pointer. If order is present, PtrState describes block pointer;
// otherwise it describes non-block pointers. When it describes block pointer,
// shape field means the same field as tt.make_tensor_ptr; when it describes a
// non-block pointer, shape field indicates how address wraps around (i.e.,
// modulo); a constant 0 indicates no modulo for the dimension.
// Multi-dimension PtrState, which has one unstructured dimension, is supported
// for gather/scatter access. The unstructured dimension is marked by a tensor
// type offset. The tensor offset for the unstructured dimension must be
// expanded from a 1D tensor. The analysis will fail for multi-dimension
// unstructured offsets. Later, when using the tensor offset to calculate the
// address, it will be collapsed to 1D. To support gather/scatter access, treat
// the unstructured offset as a whole offset instead of decoding the pointer
// arithmetic on it except scalar mul.
// The stride is set to 1 when there's no scalar mul so it still matches the
// offset * stride formula. When there're scalar muls, the stride is set to the
// multiplication of all the scalar strides.
struct PtrState {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> origiOffsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  SmallVector<OpFoldResult> shape;
  SmallVector<int32_t> order;

  Value source;
  Value scalar;

  int32_t getRank() const;

  bool isEmpty() const;

  bool hasModulo() const;

  bool dimHasModulo(uint32_t dim) const;

  bool dimIsStructured(uint32_t dim) const;
  int32_t getNonStructuredDim() const;
  // Verify that all dimensions are not structured.
  bool noStructuredDimExists() const;

  bool isStructured() const;

  bool isBlockPtr() const;

  void dump() const;

  // For unsupported op, save the op to the state.
  LogicalResult rebuildAsUnsupportedOp(Value op);

  // When merge with other state which is not structured, set the nonContinuous
  // dimension offset as op.
  // Fail if the operation already mixes different dimensions.
  // For case
  // clang-format off
  //    %14 = tt.expand_dims %11 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
  //    %dim0_value = tt.broadcast %14 : tensor<64x1xi1> -> tensor<64x64xi1>
  //    %16 = tt.expand_dims %13 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
  //    %dim1_value = tt.broadcast %16 : tensor<1x64xi1> -> tensor<64x64xi1>
  //    add  %dim0_value, %dim1_value
  // clang-format on
  //    the add will have size > 1 for both dim0 and dim1.
  //    It will fail for mix of different dims.
  //
  // Fail if the operation does not contribute to nonContinuousDim.
  // For case
  // clang-format off
  //    %14 = tt.expand_dims %11 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
  //    %dim0_value = tt.broadcast %14 : tensor<64x1xi1> -> tensor<64x64xi1>
  //    %16 = tt.expand_dims %13 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
  //    %dim0_value2 = tt.broadcast %16 : tensor<64x1xi1> -> tensor<64x64xi1>
  //    add  %dim0_value, %dim0_value2
  // clang-format on
  //    the add only have size > 1 for dim0 which doesn't mix of different
  //    dims.
  // But if call rebuildAsGatherScatter on the add with nonContinuousDim = 1 it
  // will fail because it only have dim0.
  LogicalResult rebuildAsGatherScatter(Value op, int nonContinuousDim);

  // Process addition of two PtrStates.
  LogicalResult addState(const PtrState &lhsState, const PtrState &rhsState,
                         bool isAnalysisingUnstructured, Operation *op,
                         OpBuilder &builder);

  // Process multiplication of two PtrStates
  LogicalResult mulState(const PtrState &lhsState, const PtrState &rhsState,
                         bool isAnalysisingUnstructured, Operation *op,
                         OpBuilder &builder);

  LogicalResult mergeUnstructuredState(const PtrState &other, Operation *op);

  tts::MakeTensorPtrOp createTTSMakeTensorPtrOp(
      OpBuilder &builder, SmallVector<OpFoldResult> origiOffsets, Location loc);
  tts::MakeGatherScatterTensorPtrOp
  createTTSMakeGatherScatterTensorPtrOp(OpBuilder &builder, Location loc);
};

class PtrAnalysis {
  // This function is internally used by getLoopIterArgPtrState and
  // getLoopResultPtrState to get the correct PtrState for either an iter-arg or
  // a loop's result.
  //
  // A PtrState of an scf.for's iter-arg is the same as its corresponding
  // init-arg, except that the strides and offsets have to point to the loop's
  // iter-args that were created to carry the offsets and strides.
  //
  // For instance, for a pointer with index i and rank 2, 4 additional args
  // starting at index i + 1 are created. The PtrState's strides and offsets
  // value of the pointer's iter-arg must point to these 4 additionally created
  // iter-args.
  //
  // A similar process is used for getting the PtrState of the loop's i'th
  // result: its strides and offsets have to point to the corresponding stride
  // and offset values returned by the loop.
  PtrState reconcileLoopPtrState(
      scf::ForOp forOp, size_t ptrArgIndex, const PtrState &state,
      llvm::function_ref<Value(scf::ForOp op, size_t)> getReplacementVal);

  DenseSet<Value> maybeStructuredArgs;
  const bool enableMakeGatherScatterTensorPtr;
  // If false, PtrAnalysis will analysis structured ptr while only identify
  // unstructured ptr.
  // If true, PtrAnalysis will caclulate strides and offsets for
  // unstructured pointers. This is used to support gather/scatter access.
  // The default mode is false. Only set to true when caclulating
  // unstructured pointers for gather/scatter access.
  // The reason to have different mode is to support case like:
  //
  // ptr + (row_offsets[:,None] % mod_offset + some_number) +
  //    row_indices[:None]
  //
  // (row_offsets[:,None] % mod_offset + some_number) is structured and
  // has modulo.
  // row_indices[:, None] is unstructured.
  // When visiting the add operation, we need to apply the modulo to
  //   (row_offsets[:,None] % mod_offset + some_number).
  // But we don't have the information about how to apply the modulo.
  //
  // To simplify the analysis, we do the work in two modes:
  // 1. Analyze to identify the unstructured pointers.
  // 2. Analyze to calculate the strides and offsets for unstructured pointers.
  // In mode 1, isAnalysisingUnstructured is set to false, so we only
  //    identify the unstructured pointers and do not calculate the strides and
  //    offsets.
  // When visit the operand again to calculate the offsets and strides for the
  // unstructured state, we'll set isAnalysisingUnstructured to true.
  // This means that we switched to mode 2 now and are analyzing the
  // unstructured pointers and calculating the strides and offsets for them. In
  // mode 2, we know that the pointer is unstructured, so we can just use the
  // value of arith::RemSIOp as offset directly. Once the analysis is done,
  // we'll switch back to mode 1.
  //
  // Note that this is might be a temporary solution, and we may need to
  // revisit this in the future to support more complex cases.
  bool isAnalysisingUnstructured = false;

public:
  PtrAnalysis(bool enableMakeGatherScatterTensorPtr)
      : enableMakeGatherScatterTensorPtr(enableMakeGatherScatterTensorPtr) {}
  void initializeMaybeStructuredArgs(Operation *op);

  llvm::SmallDenseMap<Value, PtrState> knownPtrs;

  IRMapping ptrMap;

  // Recursively parse a Value; call the corresponding
  // function based on the defining operation and argument type.
  LogicalResult visitOperand(Value operand, PtrState &state, const Location loc,
                             OpBuilder &builder);

  // Operand is a result of an scf.for. Such cases occur when there are multiple
  // levels of nested loops where the results of the inner scf.for (pointer) are
  // yielded by the outer loop.
  LogicalResult visitOperandForOp(scf::ForOp forOp, Value operand,
                                  PtrState &state, const Location loc,
                                  OpBuilder &builder);

  // Operand is the result of arith.addi. Process both arguments and insert any
  // arith.addi instruction as needed.
  // Main assumptions:
  //  Only one of lhsState and rhsState has source field set
  //  Current PtrState should be empty
  // Expected result:
  //  source = lhsState.source ? lhsState.source : rhsState.source
  //  sizes[i] = lhsState.sizes[i] (which should match rhsState.sizes[i])
  //  offsets[i] = lhsState.offsets[i] + rhsState.offsets[i]
  //  strides[i] = lhsState.strides[i] + rhsState.strides[i]
  LogicalResult visitOperandAdd(arith::AddIOp addOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  // Operand is the result of arith.muli. Process both arguments and insert any
  // arith.muli instruction as needed.
  // Main assumptions:
  //  Neither lhsState nor rhsState has source field set
  //  Current PtrState should be empty
  //  Currently only support one of the operand is a scalar index
  // Expected result (scalar and tensorState represent the two operands):
  //  source = null
  //  sizes[i] = tensorState.sizes[i]
  //  offsets[i] = tensorState.offsets[i] * scalar
  //  strides[i] = tensorState.strides[i] * scalar
  LogicalResult visitOperandMul(arith::MulIOp mulOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  LogicalResult visitOperandRem(arith::RemSIOp mulOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  // Operand is the result of make_range.
  // Main assumptions:
  //  start, end, and shape are all statically known
  //  The output of make_range is 1-dimensional
  //  Does not check validity of inputs (e.g., stride > 0)
  // Expected result:
  //  source = null
  //  sizes[0] = shape[0]
  //  offset[0] = start
  //  strides[0] = ceiling( (end - start) / shape[0] )
  LogicalResult visitOperandMakeRange(triton::MakeRangeOp rangeOp,
                                      PtrState &state, Location loc,
                                      OpBuilder &builder);

  // Operand is the result of expand_dims
  // Main assumptions:
  //  Only 1 dimension changes for each invocation of reshape
  //  The changed dimension must have size of 1
  // Expected result:
  //  Insert a dimension of size 1, stride 0, and offset 0
  LogicalResult visitOperandExpandDims(triton::ExpandDimsOp expandDimsOp,
                                       PtrState &state, const Location loc,
                                       OpBuilder &builder);

  // Operand is the result of broadcast
  // Main assumptions:
  //  Rank of soure and result is the same
  // Expected result:
  //  Update sizes[i] only, no changes to other fields
  LogicalResult visitOperandBroadcast(triton::BroadcastOp broadcastOp,
                                      PtrState &state, const Location loc,
                                      OpBuilder &builder);

  // Operand is the result of splat
  // Main assumptions:
  //  Source is a scalar value (i.e., an integer or a pointer, not a tensor)
  // Expected result:
  //  sizes[i] reflect the shape of the result, strides[i] = 0,  offsets[i] = 0
  //  if source is an integer, offset[0] = scalar = source
  LogicalResult visitOperandSplat(triton::SplatOp splatOp, PtrState &state,
                                  const Location loc, OpBuilder &builder);

  // Operand is the result of arith.constant that is a splat
  // Main assumptions:
  //  Source is a constant op that produces a constant dense tensor where all
  //  elements are the same (i.e.: a constant that is splatted)
  // Expected result:
  //  sizes[i] reflect the shape of the result, strides[i] = 0,  offsets[i] =
  //  splat value if i == 0, otherwise 0
  LogicalResult visitOperandConstSplat(arith::ConstantOp op, PtrState &state,
                                       const Location loc, OpBuilder &builder);

  LogicalResult visitOperandExtSI(arith::ExtSIOp, PtrState &state,
                                  const Location loc, OpBuilder &builder);

  LogicalResult visitOperandExtUI(arith::ExtUIOp, PtrState &state,
                                  const Location loc, OpBuilder &builder);

  LogicalResult visitOperandCmp(arith::CmpIOp cmpOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  // Operand is the result of addptr.
  // Main assumptions:
  //  The ptr field should populate the source field
  //  ptr and offset fields should result in same rank
  // Expected result:
  //  The resulting state for ptr and offset wil be added
  LogicalResult visitOperandAddptr(triton::AddPtrOp addptrOp, PtrState &state,
                                   const Location loc, OpBuilder &builder);

  // Operand is the result of tts.make_tptr.
  // Main assumptions:
  //  This function is only called when rewriting a loop
  // Expected result:
  //  Directly grab all corresponding fields from tts.make_tptr.
  LogicalResult visitOperandMakeTPtr(tts::MakeTensorPtrOp makeTPtrOp,
                                     PtrState &state, const Location loc,
                                     OpBuilder &builder);

  // Operand is the result of tt.make_tensor_ptr.
  // Expected result:
  //  Parse source pointer and grab results
  LogicalResult visitOperandMakeTensorPtr(triton::MakeTensorPtrOp makeTPtrOp,
                                          PtrState &state, const Location loc,
                                          OpBuilder &builder);

  // Operand is the result of tt.int_to_ptr.
  // Expected result:
  //  Directly grab op result
  LogicalResult visitOperandIntToPtr(triton::IntToPtrOp intToPtrOp,
                                     PtrState &state, const Location loc,
                                     OpBuilder &builder);

  // Operand is the result of tt.bitcast.
  // Expected result:
  //  For pointer-preserving bitcasts (scalar or tensor of triton pointers),
  //  continue visiting the source so pointer-sequence reconstruction can see
  //  through no-op pointee reinterpretation such as !tt.ptr<i1> -> !tt.ptr<i8>.
  //  For non-pointer bitcasts, fall back to the bitcast result itself.
  LogicalResult visitOperandBitcast(triton::BitcastOp bitcastOp,
                                    PtrState &state, const Location loc,
                                    OpBuilder &builder);

  LogicalResult visitOperandAlloc(xsmt::AllocOp allocOp, PtrState &state,
                                  const Location loc, OpBuilder &builder);

  LogicalResult visitOperandSubview(xsmt::SubviewOp subviewOp, PtrState &state,
                                    const Location loc, OpBuilder &builder);

  LogicalResult visitOperandSubviewPack(xsmt::SubviewPackOp subviewPackOp,
                                        PtrState &state, const Location loc,
                                        OpBuilder &builder);

  LogicalResult
  visitOperandBufferTensorViewOp(xsmt::BufferTensorViewOp BufferTensorViewOp,
                                 PtrState &state, const Location loc,
                                 OpBuilder &builder);

  // Get the computed PtrState for the forOp's init-arg at the provided index.
  FailureOr<PtrState> getLoopInitArgPtrState(scf::ForOp forOp, size_t index);

  // Get the computed PtrState for the forOp's iter-arg at the provided index.
  FailureOr<PtrState> getLoopIterArgPtrState(scf::ForOp forOp, size_t index);

  // Get the computed PtrState for the forOp's result at the provided index.
  FailureOr<PtrState> getLoopResultPtrState(scf::ForOp forOp, size_t index);

  // After PtrAnalysis finishes, rewrite the GetStructuredStateOp by creating
  // the correct initialization ops for offsets and strides and passing them to
  // any loop's init-args.
  LogicalResult rewriteGetStructuredStateOp(tts::GetStructuredStateOp op);

  // Parse the state of AddPtrOp, insert any instruction needed to
  // calculate strides and offsets, build PtrState for this operand, and record
  // PtrState for knownPtrs.
  LogicalResult rewriteAddptrOp(triton::AddPtrOp op);

  LogicalResult rewriteMakeTensorPtrOp(
      triton::MakeTensorPtrOp op,
      llvm::DenseMap<Value, SmallVector<OpFoldResult>> offsetMap);

  LogicalResult rewriteAdvanceOp(triton::AdvanceOp op);

  // Parse the state of YieldOp, insert any instruction needed to calculate
  // strides and offsets, build PtrState for this operand, and record PtrState
  // in knownPtrs.
  LogicalResult
  rewriteYieldOp(scf::YieldOp op,
                 llvm::SmallDenseMap<int, PtrState> &knownPtrsFor);

  // Rewrite eligible tt.addptr in loop init args so loop can update the such
  // pointers over iterations. Insert any instruction needed to calculate
  // strides, offsets, and modulos.
  LogicalResult rewriteForOp(scf::ForOp op);

  LogicalResult rewriteLoadOp(triton::LoadOp op, bool useUnsafeMask = false);

  LogicalResult rewriteStoreOp(triton::StoreOp op, bool useUnsafeMask = false);

  LogicalResult rewriteOp(Operation *op, bool useUnsafeMask = false);
};

} // namespace tts

} // namespace mlir

#endif
