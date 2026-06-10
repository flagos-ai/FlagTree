#include "epu/memory.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#define GEN_PASS_DEF_SETDEVICEINFO
#include "evas/Transform/Linalg/Passes.h.inc"

namespace mlir::triton::ev {

namespace {
/// A pass to insert deallocations for allocated buffers after theirlast use.
using namespace mlir;
using namespace mlir::ev;
static constexpr llvm::StringRef ADDRESS = "address";
static constexpr llvm::StringRef MEMSCOPE = "mem_scope";
static constexpr llvm::StringRef COREBIND = "core_bind";
struct SetDeviceInfoPass : public ::impl::SetDeviceInfoBase<SetDeviceInfoPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto funcOps = llvm::to_vector(module.getOps<func::FuncOp>());
    if (!funcOps.empty()) {
      auto funcOp = funcOps[0];
      funcOp.setSymName("kernel");
    }
    for (auto func : funcOps) {
      if (func.isDeclaration()) {
        continue;
      }
      OpBuilder b(func);
      // func.walk([&](bufferization::AllocTensorOp op) { SetScopeInfo(op, b);
      // });
      func.walk([&](func::CallOp op) {
        SetScopeInfo(op, b);
      });
      // FixHoistedStoreToSubKernel(b);
    }
  }

private:
  llvm::DenseMap<Value, MemScope> gScopeMap;
  llvm::DenseMap<Value, Operation *> hoistedStoreMap;

  // void SetMemrefScopeInfo(func::FuncOp funcOp) {
  //   funcOp.walk([&](memref::AllocOp op) {
      
  //   });
  // }

  void FixHoistedStoreToSubKernel(OpBuilder b) {
    for (auto &[val, op] : hoistedStoreMap) {
      auto callOp = cast<func::CallOp>(val.getDefiningOp());
      auto storeOp = cast<bufferization::MaterializeInDestinationOp>(op);
      b.setInsertionPointAfter(storeOp);
      auto toTensorOp = b.create<tptr::ToMemrefOp>(
          storeOp->getLoc(), storeOp.getDest().getType(), storeOp.getSource());
      auto dstPtr = storeOp.getDest();
      dstPtr.replaceAllUsesWith(toTensorOp->getResult(0));
      storeOp->erase();
      auto addrAttr = getAddress(dstPtr);
      if (addrAttr) {
        callOp->setAttr(ADDRESS, addrAttr);
      }
    }
  }

  Attribute getAddress(Value memref) {
    auto op = memref.getDefiningOp();
    if (isa<memref::AllocOp>(op)) {
      if (op->hasAttr(mlir::ev::phyAddrName)) {
        return op->getAttr(mlir::ev::phyAddrName);
      }
      return nullptr;
    }
    if (isa<memref::ReinterpretCastOp, memref::SubViewOp>(op))
      return getAddress(op->getOperand(0));
    llvm_unreachable("unexpected ops when traverse up ptr");
    return nullptr;
  }

  MemScope getMemScope(Value val) {
    if (isa<MemRefType>(val.getType())) {
      return mlir::ev::getMemScope(val.getType());
    }
    if (gScopeMap.contains(val))
      return gScopeMap[val];

    auto op = val.getDefiningOp();
    if (!op)
      return mlir::ev::MemScope::DDR;

    if (isa<memref::AllocOp>(op)) {
      MemRefType memType = cast<memref::AllocOp>(op).getType();
      return mlir::ev::getMemScope(memType);
    } else if (isa<bufferization::AllocTensorOp>(op)) {
      auto memSpace = cast<bufferization::AllocTensorOp>(op).getMemorySpace();
      assert(memSpace && isa<IntegerAttr>(*memSpace) &&
             "Memscope is not allowed to be empty");
      return static_cast<MemScope>(cast<IntegerAttr>(*memSpace).getInt());
    } else if (isa<bufferization::ToTensorOp>(op)) {
      return mlir::ev::getMemScope(
          cast<bufferization::ToTensorOp>(op).getOperand());
    } else if (op->getDialect()->getNamespace() ==
               tensor::TensorDialect::getDialectNamespace()) {
      for (auto input : op->getOperands()) {
        auto memScope = getMemScope(input);
        if (memScope != MemScope::UNKNOWN) {
          return memScope;
        }
      }
      return MemScope::UNKNOWN;
    } else if (isa<arith::ConstantOp>(op)) {
      return MemScope::DDR;
    } else {
      return MemScope::UNKNOWN;
    }
  }

  Operation *getOutputStoreCopy(Value val) {
    for (auto useOp : val.getUsers()) {
      if (isa<bufferization::MaterializeInDestinationOp>(useOp)) {
        return useOp;
      } else if (useOp->getDialect()->getNamespace() ==
                 tensor::TensorDialect::getDialectNamespace()) {
        return getOutputStoreCopy(useOp->getResult(0));
      }
    }
    return nullptr;
  }
  bool isFuncArgsPointer(Value ptr) {
    auto op = ptr.getDefiningOp();
    if (!op)
      return true;
    if (isa<memref::ReinterpretCastOp, memref::SubViewOp>(op)) {
      return isFuncArgsPointer(op->getOperand(0));
    }
    return false;
  }

  // void SetScopeInfo(bufferization::AllocTensorOp op, OpBuilder b) {
  //   op.setMemorySpaceAttr(b.getI64IntegerAttr(mlir::ev::MemScope::L2));
  // }

  // void SetScopeInfo(memref::AllocOp op, OpBuilder b) {
  //   auto memrefType = op.getMemref().getType();
  //   if(!memrefType.getMemorySpace()) {
  //     auto newType = MemRefType::get(memrefType.getShape(),
  //     memrefType.getElementType(),  memrefType.getLayout(),
  //     b.getI64IntegerAttr(mlir::ev::MemScope::L2));
  //     op.getMemref().setType(newType);
  //   }
  // }

  void SetScopeInfo(func::CallOp op, OpBuilder b) {
    std::vector<Attribute> memScopeArray;
    for (auto input : op.getOperands()) {
      if (!gScopeMap.contains(input)) {
        auto scope = getMemScope(input);
        gScopeMap[input] = scope;
      }
      memScopeArray.push_back(b.getI64IntegerAttr(gScopeMap[input]));
    }
    // for (auto output : op->getResults()) {
    //   if (!gScopeMap.contains(output)) {
    //     if (auto storeOp = getOutputStoreCopy(output)) {
    //       auto storeMemref = storeOp->getOperand(1);
    //       if (!isFuncArgsPointer(storeMemref)) {
    //         // gScopeMap[output] = mlir::ev::getMemScope(storeMemref);
    //         hoistedStoreMap[output] = storeOp;
    //       } else {
    //         gScopeMap[output] = mlir::ev::MemScope::L2;
    //       }
    //     } else {
    //       gScopeMap[output] = mlir::ev::MemScope::L2;
    //     }
    //   }

    //   memScopeArray.push_back(b.getI64IntegerAttr(gScopeMap[output]));
    // }
    op->setAttr(MEMSCOPE, b.getArrayAttr(memScopeArray));
  }

  ArrayAttr getCoreBind(func::CallOp op, OpBuilder b) {
    SymbolTable symbolTable(op->getParentOfType<ModuleOp>());
    func::FuncOp funcOp = symbolTable.lookup<func::FuncOp>(op.getCallee());
    assert(funcOp);
    SmallVector<Value> outputs;
    funcOp.walk([&](func::ReturnOp returnOp) {
      for (auto operand : returnOp->getOperands()) {
        outputs.push_back(operand);
      }
    });
    SmallVector<Attribute> coreBindArray;
    const size_t core_num = 4;
    for (size_t i = 0; i < core_num; ++i) {
      coreBindArray.push_back(b.getI64IntegerAttr(i));
    }
    auto core_bind = [&](int idx) {
      ArrayAttr ret = b.getArrayAttr(coreBindArray);
      for (int i = 0; i < idx; i++)
        ret = b.getArrayAttr(SmallVector<Attribute>(1, ret));
      return ret;
    };
    if (outputs.size() == 1) {
      auto output = outputs[0];
      if (isa_and_nonnull<linalg::TransposeOp, linalg::TakeOp,
                          linalg::ScatterOp, linalg::EvPadOp>(
              output.getDefiningOp())) {
        return b.getArrayAttr(
            SmallVector<Attribute>(1, b.getI64IntegerAttr(0)));
      }
      if (isa_and_nonnull<linalg::TopPOp, linalg::NormOp,
                          linalg::ReduceMedianOp, linalg::SoftmaxOp,
                          linalg::LogSoftmaxOp, linalg::ReduceOp,
                          linalg::ReduceMeanOp, linalg::TopkOp, linalg::SortOp,
                          linalg::ArgmaxOp, linalg::CumsumOp,
                          linalg::InterpolateOp, linalg::FlipOp>(
              output.getDefiningOp())) {
        auto op = output.getDefiningOp();
        if (auto parallelizableAttr =
                op->getAttrOfType<mlir::BoolAttr>("isParallelizable")) {
          if (!parallelizableAttr.getValue()) {
            return b.getArrayAttr(
                SmallVector<Attribute>(1, b.getI64IntegerAttr(0)));
          }
        }
      }
      auto type = cast<ShapedType>(output.getType());
      assert(type && type.hasRank());
      for (auto [idx, dim] : llvm::enumerate(type.getShape())) {
        if (dim < core_num || dim % core_num != 0) {
          if (idx == type.getRank() - 1) {
            return b.getArrayAttr(
                SmallVector<Attribute>(1, b.getI64IntegerAttr(0)));
          }
        } else if (auto reduceOp = dyn_cast_or_null<linalg::ReduceOp>(
                       output.getDefiningOp())) {
          mlir::detail::DenseArrayAttrImpl<int64_t> dimension =
              reduceOp.getDimensionsAttr();
          // dimension is not equal to the current idx
          auto size = dimension.getSize();
          bool not_equal = true;
          for (int i = 0; i < size; i++) {
            if (dimension[i] == idx) {
              not_equal = false;
              break;
            }
          }
          if (not_equal)
            return core_bind(idx);
        } else if (auto broadcastOp = dyn_cast_or_null<linalg::BroadcastOp>(
                       output.getDefiningOp())) {
          mlir::detail::DenseArrayAttrImpl<int64_t> dimension =
              broadcastOp.getDimensionsAttr();
          // dimension is the last dimension
          if (dimension[0] + 1 == type.getRank()) {
            // Only the dimensions before dimension can be bound
            if (idx < dimension[0])
              return core_bind(idx);
          } else {
            // Dimensions no larger than dimension can be bound
            if (idx <= dimension[0]) {
              return core_bind(idx);
            } else {
              return b.getArrayAttr(
                  SmallVector<Attribute>(1, b.getI64IntegerAttr(0)));
            }
          }
        } else if (auto layernormOp = dyn_cast_or_null<linalg::LayernormOp>(
                          output.getDefiningOp())) {
          auto dimension = layernormOp.getDimensionAttr();
          // dimension is not equal to the work idx
          if (idx != dimension.getInt())
            return core_bind(idx);
        } else {
          // The current dimension is divisible by 4, and we don't have to
          // determine dimension
          return core_bind(idx);
        }
      }
    } else {
      return b.getArrayAttr(SmallVector<Attribute>(1, b.getI64IntegerAttr(0)));
    }
    return b.getArrayAttr(SmallVector<Attribute>(1, b.getI64IntegerAttr(0)));
  }

};
} // namespace
std::unique_ptr<Pass> createSetDeviceInfoPass() {
  return std::make_unique<SetDeviceInfoPass>();
}
} // namespace mlir::triton::ev
