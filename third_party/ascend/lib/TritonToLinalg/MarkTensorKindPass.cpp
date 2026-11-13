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

#include "ascend/include/TritonToLinalg/MarkTensorKindPass.h"
#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mark-tensor-kind"

using namespace mlir;
using namespace triton;
const unsigned INT_BIT_WIDTH = 32;
const unsigned SET_INIT_SIZE = 16;

template <typename T, typename = void> struct has_getPtr : std::false_type {};
template <typename T>
struct has_getPtr<T, std::void_t<decltype(std::declval<T>().getPtr())>>
    : std::true_type {};

template <typename T, typename = void> struct has_getSrc : std::false_type {};
template <typename T>
struct has_getSrc<T, std::void_t<decltype(std::declval<T>().getSrc())>>
    : std::true_type {};

template <typename T, typename = void> struct has_getDst : std::false_type {};
template <typename T>
struct has_getDst<T, std::void_t<decltype(std::declval<T>().getDst())>>
    : std::true_type {};

template <typename T, typename = void> struct has_getBase : std::false_type {};
template <typename T>
struct has_getBase<T, std::void_t<decltype(std::declval<T>().getBase())>>
    : std::true_type {};

template <typename OpTy> static Value extractPointer(OpTy op) {
  if constexpr (has_getPtr<OpTy>::value)
    return op.getPtr();
  else if constexpr (has_getDst<OpTy>::value)
    return op.getDst();
  else if constexpr (has_getSrc<OpTy>::value)
    return op.getSrc();
  else if constexpr (has_getBase<OpTy>::value)
    return op.getBase();
  else {
    Operation *raw = op.getOperation();
    if (!raw || raw->getNumOperands() == 0)
      return Value();
    return raw->getOperand(0);
  }
}

static void setBlockArgumentAttr(BlockArgument blockArg, triton::FuncOp func,
                                 TensorKind tensorKind) {
  unsigned argIdx = blockArg.getArgNumber();
  auto existingAttr =
      func.getArgAttrOfType<IntegerAttr>(argIdx, "tt.tensor_kind");
  TensorKind oldVal = existingAttr
                          ? static_cast<TensorKind>(existingAttr.getInt())
                          : TensorKind::NONE;

  TensorKind finalVal = tensorKind;
  if ((oldVal == TensorKind::INPUT && tensorKind == TensorKind::OUTPUT) ||
      (oldVal == TensorKind::OUTPUT && tensorKind == TensorKind::INPUT)) {
    finalVal = TensorKind::INPUT_OUTPUT;
  } else if (oldVal == TensorKind::INPUT_OUTPUT) {
    finalVal = oldVal;
  }

  LLVM_DEBUG(llvm::dbgs() << "Setting tensor_kind for argument " << argIdx
                          << ": " << finalVal << "\n";);

  func.setArgAttr(
      argIdx, "tt.tensor_kind",
      IntegerAttr::get(IntegerType::get(func.getContext(), INT_BIT_WIDTH),
                       static_cast<int>(finalVal)));
}

template <typename OpTy>
static void addTensorKindToArguments(OpTy op, TensorKind tensorKind) {
  Value ptr = extractPointer(op);
  if (!ptr)
    return;

  LLVM_DEBUG(llvm::dbgs() << "Processing op: " << *op.getOperation() << "\n";);

  Value cur = ptr;
  llvm::SmallPtrSet<Value, SET_INIT_SIZE> visited;
  while (visited.insert(cur).second) {
    if (auto blockArg = dyn_cast<BlockArgument>(cur)) {
      if (auto func = dyn_cast_or_null<triton::FuncOp>(
              blockArg.getOwner()->getParentOp())) {
        if (blockArg.getOwner() == &func.getBody().front() &&
            isa<triton::PointerType>(blockArg.getType())) {
          setBlockArgumentAttr(blockArg, func, tensorKind);
          break;
        }
      }
    }

    Operation *defOp = cur.getDefiningOp();
    if (!defOp || defOp->getNumOperands() == 0)
      break;
    cur = defOp->getOperand(0);
  }
}

template <TensorKind Kind, typename OpTy>
struct MarkTensorKindPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    addTensorKindToArguments(op, Kind);
    return success();
  }
};

void MarkTensorKindPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());

  // INPUT tensors
  patterns.add<
      MarkTensorKindPattern<TensorKind::INPUT, triton::LoadOp>,
      MarkTensorKindPattern<TensorKind::INPUT,
                            triton::ascend::IndexSelectSimdOp>,
      MarkTensorKindPattern<TensorKind::INPUT, triton::ascend::GatherOutToUbOp>,
      MarkTensorKindPattern<TensorKind::INPUT, triton::ascend::IndirectLoadOp>,
      MarkTensorKindPattern<TensorKind::INPUT, triton::ascend::StrideLoadOp>>(
      &getContext());

  // OUTPUT tensors
  patterns.add<
      MarkTensorKindPattern<TensorKind::OUTPUT, triton::StoreOp>,
      MarkTensorKindPattern<TensorKind::OUTPUT, triton::ascend::IndexPutOp>,
      MarkTensorKindPattern<TensorKind::OUTPUT,
                            triton::ascend::ScatterUbToOutOp>,
      MarkTensorKindPattern<TensorKind::OUTPUT, triton::ascend::StrideStoreOp>,
      MarkTensorKindPattern<TensorKind::OUTPUT,
                            triton::ascend::IndirectStoreOp>>(&getContext());

  // INPUT_OUTPUT tensors
  patterns.add<
      MarkTensorKindPattern<TensorKind::INPUT_OUTPUT, triton::AtomicRMWOp>,
      MarkTensorKindPattern<TensorKind::INPUT_OUTPUT, triton::AtomicCASOp>>(
      &getContext());

  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<OperationPass<ModuleOp>> triton::createMarkTensorKindPass() {
  return std::make_unique<MarkTensorKindPass>();
}
