/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "TritonPPUGPUToLLVM/AIUUtility.h"
#include "TritonPPUGPUTransforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {

#define GEN_PASS_DEF_TLEPROMOTEASYNCLOADTOAIUPASS
#include "TritonPPUGPUTransforms/Passes.h.inc"

namespace {

// The AIU bulk copy only moves 1/2/4-byte elements (.b8/.b16/.b32); other
// widths stay ordinary tt.load.
static bool isAIULoadElementType(Type elemTy) {
  if (!elemTy.isIntOrFloat())
    return false;
  unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
  return bitWidth == 8 || bitWidth == 16 || bitWidth == 32;
}

// The make_tensor_ptr a block pointer comes from, through tt.advance and the
// iteration arguments and results of scf.for; null for any other producer
// (triton::getMakeTensorPtrOp asserts on those).
static triton::MakeTensorPtrOp findMakeTensorPtrOp(Value ptr) {
  while (true) {
    if (auto arg = dyn_cast<BlockArgument>(ptr)) {
      auto forOp = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
      if (!forOp || arg.getArgNumber() == 0)
        return nullptr;
      ptr = forOp.getInitArgs()[arg.getArgNumber() - 1];
      continue;
    }
    Operation *def = ptr.getDefiningOp();
    if (auto makeTensorPtr = dyn_cast<triton::MakeTensorPtrOp>(def))
      return makeTensorPtr;
    if (auto advance = dyn_cast<triton::AdvanceOp>(def)) {
      ptr = advance.getPtr();
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(def)) {
      ptr = forOp.getInitArgs()[cast<OpResult>(ptr).getResultNumber()];
      continue;
    }
    return nullptr;
  }
}

// The copy reads a 2-D tile with one unit-stride dimension and fills
// everything out of bounds with zeros. Rows that are a constant stride apart
// must fit the descriptor (see AIUUtility.h).
static bool isAIULoadLayout(triton::LoadOp loadOp) {
  if (cast<RankedTensorType>(loadOp.getType()).getRank() != 2 ||
      loadOp.getPadding() == triton::PaddingOption::PAD_NAN)
    return false;
  auto makeTensorPtr = findMakeTensorPtrOp(loadOp.getPtr());
  if (!makeTensorPtr)
    return false;
  ArrayRef<int32_t> order = makeTensorPtr.getOrder();
  auto strides = makeTensorPtr.getStrides();
  auto shape = makeTensorPtr.getShape();
  if (LLVM::PPU::getAIUConstantInt(strides[order[0]]) != 1)
    return LLVM::PPU::getAIUConstantInt(strides[order[1]]) == 1;
  unsigned rowDim = order[1];
  if (LLVM::PPU::getAIURowLayout(strides, shape, rowDim) !=
      LLVM::PPU::AIURowLayout::Strided)
    return true;
  return LLVM::PPU::isAIUStrideExpressible(
      *LLVM::PPU::getAIUConstantInt(strides[rowDim]),
      *LLVM::PPU::getAIUConstantInt(shape[order[0]]),
      llvm::is_contained(loadOp.getBoundaryCheck(), order[0]));
}

class TlePromoteAsyncLoadToAIUPass
    : public impl::TlePromoteAsyncLoadToAIUPassBase<
          TlePromoteAsyncLoadToAIUPass> {
public:
  using impl::TlePromoteAsyncLoadToAIUPassBase<
      TlePromoteAsyncLoadToAIUPass>::TlePromoteAsyncLoadToAIUPassBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder(m.getContext());

    SmallVector<triton::LoadOp> toPromote;
    m.walk([&](triton::LoadOp loadOp) {
      auto asyncAttr =
          llvm::cast_if_present<BoolAttr>(loadOp->getAttr("tt.load.async"));
      if (!asyncAttr || !asyncAttr.getValue())
        return;
      if (!triton::isTensorPointerType(loadOp.getPtr().getType()))
        return;
      auto resultTy = dyn_cast<RankedTensorType>(loadOp.getType());
      if (!resultTy || !isAIULoadElementType(resultTy.getElementType()) ||
          !isAIULoadLayout(loadOp))
        return;
      toPromote.push_back(loadOp);
    });

    for (auto loadOp : toPromote) {
      builder.setInsertionPoint(loadOp);
      auto aiuLoad = triton::AIULoadOp::create(
          builder, loadOp.getLoc(), loadOp.getType(), loadOp.getPtr(),
          loadOp.getBoundaryCheck(), loadOp.getCache(), loadOp.getEvict());
      loadOp.replaceAllUsesWith(aiuLoad.getResult());
      loadOp.erase();
    }
  }
};

} // namespace
} // namespace mlir
