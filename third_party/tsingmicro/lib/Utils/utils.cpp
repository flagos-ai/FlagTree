//===------------------- utils.cpp ----------------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "utils/utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::triton::utils {

bool isPtrTypeLike(Type t) {
  if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
    return isa<triton::PointerType>(tensorType.getElementType());
  }
  return isa<triton::PointerType>(t);
}

Value getScalarValue(Value operand, Location loc, OpBuilder &builder) {
  SmallVector<Operation *> ops;

  auto reconstructScalarValue = [&](Value src) {
    for (auto op = ops.rbegin(); op != ops.rend(); ++op) {
      src = TypeSwitch<Operation *, Value>(*op)
                .Case<arith::SIToFPOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return builder.create<arith::SIToFPOp>(loc, resType, src);
                })
                .Case<arith::TruncFOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return builder.create<arith::TruncFOp>(loc, resType, src);
                })
                .Default([](Operation *op) {
                  llvm_unreachable("unsupported op in generating ");
                  return nullptr;
                });
    }
    return src;
  };

  while (true) {
    if (!dyn_cast<ShapedType>(operand.getType())) {
      return reconstructScalarValue(operand);
    } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<DenseElementsAttr>(op.getValue())) {
        if (!attr.isSplat()) {
          InFlightDiagnostic diag = emitError(loc)
                                    << "other value used in masked load "
                                       "produced by unsupported instruction";
          return nullptr;
        }
        auto elemValue = attr.getSplatValue<Attribute>();
        auto constOp = arith::ConstantOp::materialize(
            builder, elemValue, attr.getElementType(), op.getLoc());
        return reconstructScalarValue(constOp.getResult());
      }
    } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
      operand = op.getSrc();
    } else if (auto op = operand.getDefiningOp<arith::SIToFPOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else if (auto op = operand.getDefiningOp<arith::TruncFOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else {
      InFlightDiagnostic diag = emitError(loc)
                                << "other value used in masked load produced "
                                   "by unsupported instruction";
      return nullptr;
    }
  }
  return nullptr;
}

bool isOperandMemorySpaceSPM(Value operand) {
  Operation *lastOp = operand.getDefiningOp();
  Operation *op = lastOp;
  // May be nested scf::ForOp block arguments
  if (!op && isa<BlockArgument>(operand)) {
    auto argBlock = operand.getParentBlock()->getParentOp();
    if (auto funcOp = dyn_cast<func::FuncOp>(argBlock)) {
      return false;
    }
    auto forOp = dyn_cast<scf::ForOp>(argBlock);
    assert(forOp && "BlockArgument should be in a scf::ForOp");

    auto initArgs = forOp.getInitArgs();
    auto arguments = forOp.getBody()->getArguments();

    auto idx =
        std::distance(arguments.begin(),
                      std::find(arguments.begin(), arguments.end(), operand));
    assert(initArgs.size() + forOp.getNumInductionVars() == arguments.size() &&
           "InitArgs and InductionVars should match the arguments size");

    int initArgIdx = idx - forOp.getNumInductionVars();
    assert(initArgIdx >= 0 && initArgIdx < initArgs.size() &&
           "Index out of bounds for initArgs");
    operand = initArgs[idx - forOp.getNumInductionVars()];
    return isOperandMemorySpaceSPM(operand);
  }

  do {
    if (isa<memref::AllocOp>(op))
      return true;
    else if (isa<memref::GetGlobalOp>(op))
      return false;
    else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Here we assume that yieldResults (inner loop region) and
      // loopResults (outer loop region) correspond one-to-one to obtain the
      // inner loop region definingOp of the outer loop region value.
      // FIXME:  Need reference the standard loop analysis to refactor this.

      auto yieldResults = forOp.getYieldedValues();
      mlir::ResultRange loopResults = forOp.getLoopResults().value();
      assert(yieldResults.size() == loopResults.size());
      auto idx = std::distance(
          loopResults.begin(),
          std::find(loopResults.begin(), loopResults.end(), operand));
      operand = yieldResults[idx];
      if (operand.getDefiningOp() == nullptr) {
        operand = forOp.getInitArgs()[idx];
      }
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      bool thenResult = isOperandMemorySpaceSPM(ifOp.thenYield().getOperand(0));
      bool elseResult = isOperandMemorySpaceSPM(ifOp.elseYield().getOperand(0));
      assert(thenResult == elseResult &&
             "Inconsistent memory space for IfOp results: "
             "one branch uses SPM, another branch does not");
      return thenResult;
    } else if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      // Assuming that the selectOp is used to select between two pointers with
      // same memory space, we can check the memory space of the first operand.
      operand = op->getOperand(1);
    } else {
      operand = op->getOperand(0);
    }
    lastOp = op;
    op = operand.getDefiningOp();
  } while (op);
  return false;
}

// Function to declare Tx81 runtime function
Value declareTx81Function(ModuleOp module, OpBuilder &builder, Location loc,
                          StringRef name, Type resultType,
                          ArrayRef<Type> argumentTypes) {
  // Check if the function already exists
  Operation *funcOp = module.lookupSymbol(name);
  if (funcOp)
    return builder.create<LLVM::AddressOfOp>(
        loc, LLVM::LLVMPointerType::get(builder.getContext()), name);

  // Create function type
  Type funcType = LLVM::LLVMFunctionType::get(resultType, argumentTypes,
                                              /*isVarArg=*/false);

  // Create a function declaration
  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(module.getBody());

  builder.create<LLVM::LLVMFuncOp>(loc, name, funcType,
                                   LLVM::Linkage::External);

  builder.restoreInsertionPoint(ip);

  // Return function pointer
  return builder.create<LLVM::AddressOfOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()), name);
}

} // namespace mlir::triton::utils
