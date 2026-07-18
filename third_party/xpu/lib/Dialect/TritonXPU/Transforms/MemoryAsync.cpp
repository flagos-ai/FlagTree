// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#define DEBUG_TYPE "tritonxpu-memory-async"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUMEMORYASYNC
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

class TritonXPUMemoryAsyncPass
    : public impl::TritonXPUMemoryAsyncBase<TritonXPUMemoryAsyncPass> {
public:
  using impl::TritonXPUMemoryAsyncBase<
      TritonXPUMemoryAsyncPass>::TritonXPUMemoryAsyncBase;

  TritonXPUMemoryAsyncPass() = default;
  TritonXPUMemoryAsyncPass(bool dumpFlag) { this->dumpFlag = dumpFlag; }

  void loadOpAsyncCheck(triton::xpu::LoadOp loadOp_1,
                        triton::xpu::LoadOp loadOp_2) {
    OpBuilder builder(loadOp_1);
    auto context = loadOp_1.getContext();

    if (loadOp_1->getBlock() == loadOp_2->getBlock() &&
        loadOp_1->isBeforeInBlock(loadOp_2)) {
      auto async = MemorySyncModeAttr::get(context, MemorySyncMode::ASYNC);
      if (auto gm2lmOp_1 = dyn_cast<triton::xpu::GM2LMOp>(
              loadOp_1.getPtr().getDefiningOp())) {
        gm2lmOp_1->setAttr("syncMode", async);
        loadOp_1->moveAfter(loadOp_2);
      } else if (auto gm2lmOp_1 = dyn_cast<triton::xpu::GM2LMMaskOp>(
                     loadOp_1.getPtr().getDefiningOp())) {
        gm2lmOp_1->setAttr("syncMode", async);
        loadOp_1->moveAfter(loadOp_2);
      }

      if (dumpFlag)
        LLVM_DEBUG(llvm::dbgs() << "Memory Async Optimization Hit!\n");
    }
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    mod->walk([&](triton::xpu::GM2LMOp gm2lmOp_1) {
      // Pruning
      if (gm2lmOp_1.getSyncMode() == MemorySyncMode::ASYNC)
        return;

      auto loadOp_1 =
          cast<triton::xpu::LoadOp>(*(gm2lmOp_1->getUsers().begin()));

      auto loadop_user_begin = loadOp_1->user_begin();
      auto loadop_user_end = loadOp_1->user_end();

      if (!loadOp_1->hasOneUse())
        return;

      llvm::TypeSwitch<Operation *>(*loadop_user_begin)
          .Case<XPU_VVECTORIZED_BINARY_OP, XPU_SVECTORIZED_BINARY_OP,
                triton::xpu::VvdivFOp, triton::xpu::VCmpFOp>([&](auto vBinOp) {
            auto lhsOp =
                vBinOp.getLhs().template getDefiningOp<triton::xpu::LoadOp>();
            auto rhsOp =
                vBinOp.getRhs().template getDefiningOp<triton::xpu::LoadOp>();

            if (lhsOp && rhsOp) {
              triton::xpu::LoadOp loadOp_2 = lhsOp == loadOp_1 ? rhsOp : lhsOp;
              loadOpAsyncCheck(loadOp_1, loadOp_2);
            }
          })
          .Case<mlir::arith::AddFOp, mlir::arith::SubFOp, mlir::arith::MulFOp,
                mlir::arith::MaximumFOp, mlir::arith::DivFOp,
                triton::xpu::CmpFOp>([&](auto binOp) {
            auto lhsOp =
                binOp.getLhs().template getDefiningOp<triton::xpu::LoadOp>();
            auto rhsOp =
                binOp.getRhs().template getDefiningOp<triton::xpu::LoadOp>();

            if (lhsOp && rhsOp) {
              triton::xpu::LoadOp loadOp_2 = lhsOp == loadOp_1 ? rhsOp : lhsOp;
              loadOpAsyncCheck(loadOp_1, loadOp_2);
            }
          })
          .Case<mlir::triton::xpu::VSelectOp, mlir::arith::SelectOp>(
              [&](auto selectOp) {
                auto tv = selectOp.getTrueValue()
                              .template getDefiningOp<triton::xpu::LoadOp>();
                auto fv = selectOp.getFalseValue()
                              .template getDefiningOp<triton::xpu::LoadOp>();

                if (tv && fv) {
                  triton::xpu::LoadOp loadOp_2 = tv == loadOp_1 ? fv : tv;
                  loadOpAsyncCheck(loadOp_1, loadOp_2);
                }
              })
          .Default([&](auto &op) {
            if (dumpFlag) {
              LLVM_DEBUG({
                op->dump();
                llvm::dbgs() << "Unsupport Op For Memory Async Optimization\n";
              });
            }
          });
    });

    mod->walk([&](triton::xpu::GM2LMMaskOp gm2lmOp_1) {
      // Pruning
      if (gm2lmOp_1.getSyncMode() == MemorySyncMode::ASYNC)
        return;

      auto loadOp_1 =
          cast<triton::xpu::LoadOp>(*(gm2lmOp_1->getUsers().begin()));

      auto loadop_user_begin = loadOp_1->user_begin();
      auto loadop_user_end = loadOp_1->user_end();

      if (!loadOp_1->hasOneUse())
        return;

      llvm::TypeSwitch<Operation *>(*loadop_user_begin)
          .Case<XPU_VVECTORIZED_BINARY_OP, XPU_SVECTORIZED_BINARY_OP,
                triton::xpu::VvdivFOp, triton::xpu::VCmpFOp>([&](auto vBinOp) {
            auto lhsOp =
                vBinOp.getLhs().template getDefiningOp<triton::xpu::LoadOp>();
            auto rhsOp =
                vBinOp.getRhs().template getDefiningOp<triton::xpu::LoadOp>();

            if (lhsOp && rhsOp) {
              triton::xpu::LoadOp loadOp_2 = lhsOp == loadOp_1 ? rhsOp : lhsOp;
              loadOpAsyncCheck(loadOp_1, loadOp_2);
            }
          })
          .Case<mlir::arith::AddFOp, mlir::arith::SubFOp, mlir::arith::MulFOp,
                mlir::arith::MaximumFOp, mlir::arith::DivFOp,
                triton::xpu::CmpFOp>([&](auto binOp) {
            auto lhsOp =
                binOp.getLhs().template getDefiningOp<triton::xpu::LoadOp>();
            auto rhsOp =
                binOp.getRhs().template getDefiningOp<triton::xpu::LoadOp>();

            if (lhsOp && rhsOp) {
              triton::xpu::LoadOp loadOp_2 = lhsOp == loadOp_1 ? rhsOp : lhsOp;
              loadOpAsyncCheck(loadOp_1, loadOp_2);
            }
          })
          .Case<mlir::triton::xpu::VSelectOp, mlir::arith::SelectOp>(
              [&](auto selectOp) {
                auto tv = selectOp.getTrueValue()
                              .template getDefiningOp<triton::xpu::LoadOp>();
                auto fv = selectOp.getFalseValue()
                              .template getDefiningOp<triton::xpu::LoadOp>();

                if (tv && fv) {
                  triton::xpu::LoadOp loadOp_2 = tv == loadOp_1 ? fv : tv;
                  loadOpAsyncCheck(loadOp_1, loadOp_2);
                }
              })
          .Default([&](auto &op) {
            if (dumpFlag) {
              LLVM_DEBUG({
                op->dump();
                llvm::dbgs() << "Unsupport Op For Memory Async Optimization\n";
              });
            }
          });
    });
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
