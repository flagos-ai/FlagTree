#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/LLVMXPU/IR/Dialect.h"
#include "triton/Dialect/LLVMXPU/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mfence-check"

using namespace mlir;

namespace mlir {

namespace LLVM {
namespace XPU {

#define GEN_PASS_DEF_LLVMXPUMFENCECHECK
#include "triton/Dialect/LLVMXPU/Transforms/Passes.h.inc"

struct LLVMXPUMfenceCheckPass
    : public impl::LLVMXPUMfenceCheckBase<LLVMXPUMfenceCheckPass> {
  using impl::LLVMXPUMfenceCheckBase<
      LLVMXPUMfenceCheckPass>::LLVMXPUMfenceCheckBase;
  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();
    OpBuilder builder(&getContext());

    // Find Next User Of This Buffer And Insert Mfence LM For Removing Conflict
    // Example: %31 And %29 Use Buffer %15, %31 Should Wait %29 Complete
    /*
      %28 = "triton_xpu.gm2lm_mask"(%27, %15)
      %29 = "triton_xpu.load"(%28)
      %30 = tt.addptr %arg3, %26
      %31 = "triton_xpu.gm2lm_mask"(%30, %15)
      %32 = "triton_xpu.load"(%31)
    */

    LLVM_DEBUG(llvm::dbgs() << "//---------- LLVMXPUMfenceCheckPass\n");

    for (auto &block : func.getBlocks()) {
      DenseSet<Value> pendingLoads;
      DenseSet<Value> pendingPtrs;
      // 收集load的指针，然后后续check是否有其他的op使用了这个指针或者使用了load
      for (auto &op : llvm::make_early_inc_range(block)) {

        for (auto opp : op.getOperands()) {
          if (pendingLoads.contains(opp)) {
            auto prevLoad = opp.getDefiningOp<LLVM::LoadOp>();
            if (pendingPtrs.contains(prevLoad.getAddr()))
              pendingPtrs.erase(prevLoad.getAddr());
          }
        }

        if (auto mfenceOp = dyn_cast<LLVM::XPU::MfenceOp>(&op)) {
          auto fenceValue = mfenceOp.getNum();
          pendingLoads.clear();
          pendingPtrs.clear();
          LLVM_DEBUG(llvm::dbgs() << "Clear Record\n" << op << "\n");
        } else if (auto gm2lmOp = dyn_cast<LLVM::XPU::GM2LMOp_v3>(&op)) {
          Value destBuffer = gm2lmOp.getDst();
          if (pendingPtrs.contains(destBuffer)) {
            builder.setInsertionPoint(&op);
            createMfenceOpForLM(builder, op.getLoc());
            pendingPtrs.erase(destBuffer);
            LLVM_DEBUG(llvm::dbgs() << "Insert Mfence For\n"
                                    << gm2lmOp << "\n");
          }
        } else if (auto loadOp = dyn_cast<LLVM::LoadOp>(&op)) {
          Value ptr = loadOp.getAddr();
          pendingLoads.insert(loadOp);
          pendingPtrs.insert(ptr);
          LLVM_DEBUG(llvm::dbgs() << "Record Load\n"
                                  << loadOp << " | " << ptr << "\n");
        }
        // else if (auto phiOp = dyn_cast<LLVM::PHINode>(&op)) {
        // }
      }
    }
  }

private:
  void createMfenceOpForLM(OpBuilder &builder, mlir::Location loc) const {
    // The magic number 1 of MfenceOp means mfencing on LM
    auto i32ty = builder.getIntegerType(32);
    auto one = builder.create<LLVM::ConstantOp>(loc, i32ty,
                                                IntegerAttr::get(i32ty, 5));
    builder.create<mlir::LLVM::XPU::MfenceOp>(loc, one);
  }
};

} // namespace XPU
} // namespace LLVM
} // namespace mlir
