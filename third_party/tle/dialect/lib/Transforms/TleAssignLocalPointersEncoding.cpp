// MIT License

// Copyright (c) 2025 The FlagOS Contributors

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// flagtree tle

#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLEASSIGNLOCALPOINTERSENCODING
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

// Triton shared-memory pointers use LLVM address space 3 (NVVM shared).
constexpr int kSharedMemoryAddressSpace = 3;
constexpr StringLiteral kBarrierGroupAttr = "tle.barrier_group";

class AssignLocalPointersEncodingPass
    : public impl::TritonTleAssignLocalPointersEncodingBase<
          AssignLocalPointersEncodingPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    module.walk([&](triton::tle::LocalPointersOp op) {
      auto tensorTy = dyn_cast<RankedTensorType>(op.getResult().getType());
      if (!tensorTy)
        return;
      auto ptrTy = dyn_cast<triton::PointerType>(tensorTy.getElementType());
      if (!ptrTy)
        return;
      bool updated = false;
      RankedTensorType updatedTensorTy = tensorTy;
      const auto desiredAddrSpace = kSharedMemoryAddressSpace;
      if (ptrTy.getAddressSpace() != desiredAddrSpace) {
        ptrTy =
            triton::PointerType::get(ptrTy.getPointeeType(), desiredAddrSpace);
        updated = true;
      }

      auto encoding = tensorTy.getEncoding();
      if (!encoding) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(op);
        int numWarps = triton::gpu::maybeLookupNumWarps(op).value_or(1);
        int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(builder);
        int numCTAs = triton::gpu::lookupNumCTAs(builder);
        encoding = triton::gpu::getDefaultBlockedEncoding(
            module.getContext(), tensorTy.getShape(), numWarps, threadsPerWarp,
            numCTAs);
        updated = true;
      }

      if (updated)
        updatedTensorTy =
            RankedTensorType::get(tensorTy.getShape(), ptrTy, encoding);

      if (updated)
        op.getResult().setType(updatedTensorTy);

      if (Value offsets = op.getOffsets()) {
        auto offsetsTy = dyn_cast<RankedTensorType>(offsets.getType());
        if (offsetsTy && encoding && offsetsTy.getEncoding() != encoding) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(op);
          auto castedTy = RankedTensorType::get(
              offsetsTy.getShape(), offsetsTy.getElementType(), encoding);
          Value casted = builder.create<triton::gpu::ConvertLayoutOp>(
              op.getLoc(), castedTy, offsets);
          op.setOperand(1, casted);
        }
      }

      tagDependencyGroup(op, builder);
    });
  }

  void tagDependencyGroup(triton::tle::LocalPointersOp op, OpBuilder &builder) {
    auto alloc = op.getSrc().getDefiningOp<triton::gpu::LocalAllocOp>();
    if (!alloc)
      return;
    auto groupAttr = alloc->getAttrOfType<IntegerAttr>(kBarrierGroupAttr);
    if (!groupAttr) {
      groupAttr = builder.getI64IntegerAttr(nextBarrierGroupId++);
      alloc->setAttr(kBarrierGroupAttr, groupAttr);
    }
    op->setAttr(kBarrierGroupAttr, groupAttr);
  }

  int64_t nextBarrierGroupId = 0;
};

} // namespace
} // namespace mlir::triton::tle
