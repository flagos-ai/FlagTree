
//===----------------------------------------------------------------------===//
// TODO[dyq]: Pass Description
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"

namespace mlir {
namespace triton {
namespace xpu {
#define GEN_PASS_DEF_TRITONXPUDTYPECONVERT
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUDtypeConvert
    : public impl::TritonXPUDtypeConvertBase<TritonXPUDtypeConvert> {

  using impl::TritonXPUDtypeConvertBase<
      TritonXPUDtypeConvert>::TritonXPUDtypeConvertBase;

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    llvm::SetVector<Operation *> visitedOps;
    llvm::SmallVector<llvm::SetVector<Operation *>> allOpTrees;

    m.walk([&](triton::xpu::StoreOp currStoreOp) {
      if (!visitedOps.contains(currStoreOp)) {
        // Get the opTree on the storeOp val path
        llvm::SetVector<Operation *> allOpTree;
        getOpTreeBwd(allOpTree, visitedOps,
                     currStoreOp.getValue().getDefiningOp());
        allOpTrees.emplace_back(allOpTree);
      }
    });
    // Collect yieldMap
    llvm::DenseMap<Operation *, llvm::SetVector<int>> yieldMap;
    for (auto allOpTree : allOpTrees) {
      for (auto op : allOpTree) {
        if (auto yieldOp = isa<scf::YieldOp>(op)) {
          for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
            auto operandElemTy = getElementTypeOrSelf(operand.getType());
            if ((xpuArch == 2 && mlir::isa<mlir::Float16Type>(operandElemTy)) ||
                (xpuArch == 3 &&
                 mlir::isa<mlir::BFloat16Type>(operandElemTy))) {
              yieldMap[op].insert(i);
            }
          }
        }
      }
    }
    // fp16tofp32/bf16tofp32
    for (auto allOpTree : allOpTrees) {
      for (auto op : allOpTree) {
        auto builder = mlir::OpBuilder(op);
        auto loc = op->getLoc();
        if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
          auto constResTy = getElementTypeOrSelf(constOp.getType());
          if ((xpuArch == 2 && mlir::isa<mlir::Float16Type>(constResTy)) ||
              (xpuArch == 3 && mlir::isa<mlir::BFloat16Type>(constResTy))) {
            SmallVector<Operation *> constUsers;
            for (auto user : constOp.getResult().getUsers()) {
              constUsers.emplace_back(user);
            }
            Operation *extfOp;
            if (mlir::isa<mlir::TensorType>(constOp.getType())) {
              auto tensorType = RankedTensorType::get(
                  mlir::cast<RankedTensorType>(constOp.getType()).getShape(),
                  builder.getF32Type(),
                  mlir::cast<RankedTensorType>(constOp.getType())
                      .getEncoding());
              extfOp = builder.create<arith::ExtFOp>(loc, tensorType, constOp);
            } else {
              extfOp = builder.create<arith::ExtFOp>(loc, builder.getF32Type(),
                                                     constOp);
            }
            extfOp->moveAfter(constOp);
            for (auto op : constUsers) {
              for (int i = 0; i < op->getOperands().size(); ++i) {
                if (op->getOperands()[i] == constOp.getResult()) {
                  op->setOperand(i, extfOp->getResult(0));
                }
              }
            }
          }
        } else {
          for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
            if (!operand.getDefiningOp()) {
              auto operandElemTy = getElementTypeOrSelf(operand.getType());
              if ((xpuArch == 2 &&
                   mlir::isa<mlir::Float16Type>(operandElemTy)) ||
                  (xpuArch == 3 &&
                   mlir::isa<mlir::BFloat16Type>(operandElemTy))) {
                Operation *extfOp;
                if (mlir::isa<mlir::TensorType>(operand.getType())) {
                  auto tensorType = RankedTensorType::get(
                      mlir::cast<RankedTensorType>(operand.getType())
                          .getShape(),
                      builder.getF32Type(),
                      mlir::cast<RankedTensorType>(operand.getType())
                          .getEncoding());
                  extfOp =
                      builder.create<arith::ExtFOp>(loc, tensorType, operand);
                } else {
                  extfOp = builder.create<arith::ExtFOp>(
                      loc, builder.getF32Type(), operand);
                }
                extfOp->moveBefore(op);
                op->setOperand(i, extfOp->getResult(0));
              }
            }
          }

          if (yieldMap.find(op) != yieldMap.end()) {
            for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
              if (yieldMap[op].contains(i)) {
                auto yieidOp = cast<scf::YieldOp>(op);
                Type truncTy =
                    xpuArch == 3 ? builder.getBF16Type() : builder.getF16Type();
                Operation *truncfOp;
                if (mlir::isa<mlir::TensorType>(operand.getType())) {
                  auto tensorType = RankedTensorType::get(
                      mlir::cast<RankedTensorType>(operand.getType())
                          .getShape(),
                      truncTy,
                      mlir::cast<RankedTensorType>(operand.getType())
                          .getEncoding());
                  truncfOp =
                      builder.create<arith::TruncFOp>(loc, tensorType, operand);
                } else {
                  truncfOp =
                      builder.create<arith::TruncFOp>(loc, truncTy, operand);
                }
                truncfOp->moveBefore(op);
                op->setOperand(i, truncfOp->getResult(0));
              }
            }
          }

          if (!isa<scf::IfOp, scf::ForOp>(op)) {
            for (auto [i, res] : llvm::enumerate(op->getResults())) {
              auto resElemTy = getElementTypeOrSelf(res.getType());
              if ((xpuArch == 2 && mlir::isa<mlir::Float16Type>(resElemTy)) ||
                  (xpuArch == 3 && mlir::isa<mlir::BFloat16Type>(resElemTy))) {
                if (mlir::isa<mlir::TensorType>(res.getType())) {
                  auto tensorType = RankedTensorType::get(
                      mlir::cast<RankedTensorType>(res.getType()).getShape(),
                      builder.getF32Type(),
                      mlir::cast<RankedTensorType>(res.getType())
                          .getEncoding());
                  res.setType(tensorType);
                } else {
                  res.setType(builder.getF32Type());
                }
              }
            }
          }
        }
      }
    }

    m.walk([&](arith::ExtFOp extfOp) {
      auto inTy = extfOp.getIn().getType();
      auto resTy = extfOp.getType();
      if (getElementTypeOrSelf(inTy) == getElementTypeOrSelf(resTy)) {
        extfOp.getOut().replaceAllUsesWith(extfOp.getIn());
        extfOp.erase();
      }
    });

    m.walk([&](arith::TruncFOp truncfOp) {
      auto inTy = truncfOp.getIn().getType();
      auto resTy = truncfOp.getType();
      if (getElementTypeOrSelf(inTy) == getElementTypeOrSelf(resTy)) {
        truncfOp.getOut().replaceAllUsesWith(truncfOp.getIn());
        truncfOp.erase();
      }
    });

    if (this->fp16Fast) {
      m.walk([&](triton::xpu::LoadOp loadOp) {
        auto res = loadOp.getResult();
        auto resElemTy = getElementTypeOrSelf(res.getType());
        if (isa<Float16Type>(resElemTy)) {
          if (res.hasOneUse()) {
            auto userBegin = res.user_begin();
            if (auto extfOp = dyn_cast<arith::ExtFOp>(*userBegin)) {
              extfOp.getOut().replaceAllUsesWith(extfOp.getIn());
            }
          }
        }
      });
    }
  }

private:
  bool fp16Fast = mlir::triton::tools::getBoolEnv("TRITONXPU_FP16_FAST");
};

} // namespace xpu
} // namespace triton
} // namespace mlir
