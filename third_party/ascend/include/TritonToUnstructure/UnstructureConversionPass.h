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

#ifndef TRITON_ADAPTER_UNSTRUCTURECONVERSION_H
#define TRITON_ADAPTER_UNSTRUCTURECONVERSION_H

#include "ascend/include/TritonToUnstructure/OffsetAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "ascend/include/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"

#define GEN_PASS_DECL_TRITONTOUNSTRUCTURE
#include "ascend/include/TritonToUnstructure/Passes.h.inc"

#define GEN_PASS_DEF_TRITONTOUNSTRUCTURE
#include "ascend/include/TritonToUnstructure/Passes.h.inc"

extern bool compileOn91095Flag;

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createTritonToUnstructurePass(const TritonToUnstructureOptions &options = {});

} // namespace triton
} // namespace mlir

namespace {

using namespace mlir;
using namespace triton;

//  For example, in unstructured load case
//  %0 = tt.load %structured : tensor<128x128x!tt.ptr<i32>>
//  %ptr_2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
//  %1 = tt.addptr %ptr_2, %0 : tensor<128x128x!tt.ptr<f32>>,
//  tensor<128x128xi32> %2 = tt.load %1 : tensor<128x128x!tt.ptr<f32>> tt.store
//  %output %2 : tensor<128x128x!tt.ptr<f32>>
//
//
//  In this case, this will be converted to
//
//  %0 = tt.load %structured : tensor<128x128x!tt.ptr<i32>>
//  %1 = tensor.empty() : tensor<128x128xf32>
//  %2 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %1) ->
//  (tensor<128x128xf32>) {
//    %4 = scf.for %arg4 = %c0 to %c128 step %c1 iter_args(%arg5 = %arg3) ->
//    (tensor<128x128xf32>) {
//      %extracted = tensor.extract %10[%arg3, %arg5] {DiscreteMemAccess} :
//      tensor<128x128xi32> %5 = arith.extsi %extracted : i32 to i64 %6 =
//      tt.addptr %arg1, %5 : !tt.ptr<f32>, i64 %7 = tt.load %6
//      {DiscreteMemAccess}  : tt.ptr<f32> %inserted_slice = tensor.insert_slice
//      %7 into %arg5[%arg2, %arg4] [1, 1] [128, 1] {DiscreteMemAccess} :
//      tensor<1x1xf32> into tensor<128x128xf32> scf.yield %inserted_slice :
//      tensor<128x128xf32>
//    }
//    scf.yield %4 : tensor<128x128xf32>
//  }
//  tt.store %output %2 : tensor<128x128x!tt.ptr<f32>>
template <typename MemAccOpTy>
class UnstructuredMemAccessConverter : public OpRewritePattern<MemAccOpTy> {
  static_assert(std::is_same_v<MemAccOpTy, triton::LoadOp> ||
                std::is_same_v<MemAccOpTy, triton::StoreOp> ||
                std::is_same_v<MemAccOpTy, triton::AtomicRMWOp> ||
                std::is_same_v<MemAccOpTy, triton::AtomicCASOp>);

public:
  using OpRewritePattern<MemAccOpTy>::OpRewritePattern;

  explicit UnstructuredMemAccessConverter(
      MLIRContext *context, bool forceScalarizeMode,
      const llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap,
      const llvm::SmallDenseMap<Value, bool> &fromTensorArg);
  LogicalResult matchAndRewrite(MemAccOpTy op,
                                PatternRewriter &rewriter) const override;

private:
  bool checkUnstructureAnnotated(MemAccOpTy op,
                                 PatternRewriter &rewriter) const;
  Value createExtractOp(Location loc, Value value, PatternRewriter &rewriter,
                        ArrayRef<OpFoldResult> iterIdx) const;
  Value createExtractOp(Location loc, Value value, PatternRewriter &rewriter,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        ArrayRef<OpFoldResult> strides) const;
  template <typename U = MemAccOpTy>
  typename std::enable_if<std::is_same_v<U, triton::LoadOp>,
                          LogicalResult>::type
  splatAndLoadScenario(MemAccOpTy op, const PtrOffsetInfo &ptrOffsetInfo,
                       PatternRewriter &rewriter) const;

  template <typename... Args>
  MemAccOpTy createMemAccOp(MemAccOpTy op, Value ptrToAccess, Location loc,
                            PatternRewriter &rewriter, Args &&...args) const;

  const llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap;
  const llvm::SmallDenseMap<Value, bool> &fromTensorArg;
  bool forceScalarizeMode;
};

class TritonToUnstructurePass
    : public ::impl::TritonToUnstructureBase<TritonToUnstructurePass> {
public:
  explicit TritonToUnstructurePass(const TritonToUnstructureOptions &options);
  void getDependentDialects(DialectRegistry &registry) const override;

  void runOnOperation() override;

private:
  void runPreparse(LoopLikeOpInterface op);
  template <typename MemAccOpTy,
            typename = std::enable_if_t<
                std::is_same_v<MemAccOpTy, triton::LoadOp> ||
                std::is_same_v<MemAccOpTy, triton::StoreOp> ||
                std::is_same_v<MemAccOpTy, triton::AtomicRMWOp> ||
                std::is_same_v<MemAccOpTy, triton::AtomicCASOp>>>
  void runParse(MemAccOpTy op);
  llvm::DenseMap<Value, PtrOffsetInfo> offsetMap;
  llvm::DenseMap<Value, PtrOffsetInfo> offsetMapForLoopArgs;
  llvm::SmallDenseMap<Value, bool> fromTensorArg;

  LogicalResult processIfYieldAddHoistOperations(ModuleOp moduleOp);
};

} // namespace

void replacePtrArguments(triton::FuncOp funcOp,
                         llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

#endif // TRITON_ADAPTER_UNSTRUCTURECONVERSION_H
