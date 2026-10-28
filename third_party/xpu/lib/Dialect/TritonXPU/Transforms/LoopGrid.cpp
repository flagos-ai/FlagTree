//===----------------------------------------------------------------------===//
// TODO[dyq]: Pass Description
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"

#include <functional>

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPULOOPGRID
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPULoopGrid
    : public impl::TritonXPULoopGridBase<TritonXPULoopGrid> {

  using impl::TritonXPULoopGridBase<TritonXPULoopGrid>::TritonXPULoopGridBase;

  static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT = 3;

  Value ceilDiv(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
    auto c1 = builder.create<arith::ConstantIntOp>(loc, lhs.getType(), 1);
    auto sub = builder.create<arith::SubIOp>(loc, rhs, c1);
    auto add = builder.create<arith::AddIOp>(loc, lhs, sub);
    auto div = builder.create<arith::DivSIOp>(loc, add, rhs);
    return div.getResult();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    m.walk([&](triton::FuncOp func) {
      OpBuilder b(func);

      auto i32Ty = b.getI32Type();
      auto origFuncType = func.getFunctionType();
      auto origInputTypes = origFuncType.getInputs();
      SmallVector<Type> newInputTypes(origInputTypes.begin(),
                                      origInputTypes.end());
      newInputTypes.append(TRITON_PROGRAM_INFO_ARG_COUNT, i32Ty);

      auto newFuncType =
          b.getFunctionType(newInputTypes, origFuncType.getResults());

      func.setType(newFuncType);

      auto &body = func.getBody().front();
      for (unsigned int i = 0; i < TRITON_PROGRAM_INFO_ARG_COUNT; i++) {
        body.addArgument(i32Ty, func.getLoc());
      }

      // Filter operations based on the condition
      SmallVector<Operation *> operations;
      for (auto &op : body.getOperations()) {
        if (!isa<triton::ReturnOp>(op)) {
          operations.push_back(&op);
        }
      }

      b.setInsertionPoint(&body, body.begin());
      auto loc = b.getUnknownLoc();
      auto idxTy = b.getIndexType();
      auto argIdx = func.getNumArguments() - TRITON_PROGRAM_INFO_ARG_COUNT;
      auto idxCluster = b.create<triton::xpu::GetClusterIdOp>(loc, i32Ty);
      auto numCluster = b.create<triton::xpu::GetNumClusterOp>(loc, i32Ty);
      auto gridX = func.getArgument(argIdx + 0);
      auto gridY = func.getArgument(argIdx + 1);
      auto gridZ = func.getArgument(argIdx + 2);
      auto gridXY = b.create<arith::MulIOp>(loc, gridX, gridY);
      auto gridXYZ = b.create<arith::MulIOp>(loc, gridXY, gridZ);
      auto numProgramsPerCluster = ceilDiv(b, loc, gridXYZ, numCluster);
      auto lower = b.create<arith::IndexCastOp>(loc, idxTy, idxCluster);
      auto upper = b.create<arith::IndexCastOp>(loc, idxTy, gridXYZ);
      auto step = b.create<arith::IndexCastOp>(loc, idxTy, numCluster);
      auto loopGrid = b.create<scf::ForOp>(loc, lower, upper, step);

      // Determine which top-level ops must stay OUTSIDE the grid-stride loop.
      // Cluster-shared SM staging (tle_copy_g2l into a scope=smem local_alloc)
      // is loop-invariant: it copies the same GM data (loop-invariant desc,
      // constant offsets) into per-cluster SM, which persists across all
      // grid-stride iterations. If it were moved inside the loop it would
      // re-stage on every iteration -- wasting DMA and (without a leading
      // barrier) racing the previous iteration's SM reads ("sm rdwr conflict").
      // So hoist the staging copy, its smem alloc, and their memory-effect-free
      // operand cone (constants / index math) to before the loop => staged ONCE
      // per cluster. Pure LM kernels have no smem copy => hoistSet stays empty
      // => behaviour is byte-identical to before.
      llvm::SetVector<Operation *> hoistSet;
      // Collect the defining cone of `v` inside the loop body into `cone`.
      // Returns false if any op of that cone cannot be hoisted (it has memory
      // effects), in which case the CALLER MUST ABANDON the whole hoist: moving
      // a copy out while one of its operands stays inside the loop would break
      // dominance and fail the verifier.
      std::function<bool(Value, llvm::SetVector<Operation *> &)> collectCone =
          [&](Value v, llvm::SetVector<Operation *> &cone) -> bool {
        Operation *def = v.getDefiningOp();
        if (!def || def->getBlock() != &body)
          return true; // block arg or not a top-level body op: already
                       // dominates.
        if (hoistSet.contains(def) || cone.contains(def))
          return true;
        if (!isMemoryEffectFree(def))
          return false; // only pure ops are safe to pull out via the cone.
        cone.insert(def);
        for (Value operand : def->getOperands())
          if (!collectCone(operand, cone))
            return false;
        return true;
      };
      for (Operation *op : operations) {
        auto copy = dyn_cast<triton::xpu::TLECopyGlobalToLocalOp>(op);
        if (!copy)
          continue;
        Operation *dstDef = copy.getDstBuffer().getDefiningOp();
        if (!dstDef)
          continue;
        auto scope = dstDef->getAttrOfType<StringAttr>("xpu.mem_scope");
        if (!scope || scope.getValue() != "smem")
          continue;
        // The staging copy and its smem alloc are hoisted by construction;
        // everything they depend on must come from a PURE cone. If anything is
        // impure, keep this copy in the loop: still correct (it re-stages every
        // iteration, and the lowering's leading mfence + barrier makes that
        // race-free), just slower.
        llvm::SetVector<Operation *> cone;
        if (dstDef->getBlock() == &body)
          cone.insert(dstDef);
        bool hoistable = true;
        for (Value operand : copy->getOperands())
          hoistable &= collectCone(operand, cone);
        for (Value operand : dstDef->getOperands())
          hoistable &= collectCone(operand, cone);
        if (!hoistable)
          continue;
        hoistSet.insert(cone.begin(), cone.end());
        hoistSet.insert(copy);
      }

      for (auto op : operations) {
        if (hoistSet.contains(op))
          op->moveBefore(loopGrid); // stays before the loop: staged ONCE.
        else
          op->moveBefore(loopGrid.getBody()->getTerminator());
      }

      b.setInsertionPointToStart(loopGrid.getBody());
      Value index =
          b.create<arith::IndexCastOp>(loc, i32Ty, loopGrid.getInductionVar());
      auto pidZ = b.create<arith::RemSIOp>(loc, index, gridZ);
      index = b.create<arith::DivSIOp>(loc, index, gridZ);
      auto pidY = b.create<arith::RemSIOp>(loc, index, gridY);
      auto pidX = b.create<arith::DivSIOp>(loc, index, gridY);

      SmallVector<Value, 4> programId{pidX, pidY, pidZ};
      func.walk([&](triton::GetProgramIdOp op) {
        op.replaceAllUsesWith(programId[op.getAxisAsInt()]);
      });
      func.walk([&](triton::GetNumProgramsOp op) {
        op.replaceAllUsesWith(func.getArgument(argIdx + op.getAxisAsInt()));
      });
      func.walk([&](XPUPrintOp op) {
        OpBuilder replacer(op);
        Value outerIdx = replacer.create<arith::ExtSIOp>(
            op.getLoc(), replacer.getI64Type(), index);
        op->setOperand(3, outerIdx);
      });
    });
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
