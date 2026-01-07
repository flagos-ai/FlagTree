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
#include "utils/ReduceScanCommon.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::triton {

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

TypedAttr getRedBaseAttr(OpBuilder &builder, Operation *redOp,
                         Type constantType) {
  const int64_t bitWidth = constantType.getIntOrFloatBitWidth();

  auto attr = llvm::TypeSwitch<Operation *, TypedAttr>(redOp)
                  .Case([&](arith::AddFOp) {
                    return builder.getFloatAttr(constantType, 0.f);
                  })
                  .Case([&](arith::AddIOp) {
                    return builder.getIntegerAttr(constantType, 0);
                  })
                  .Case([&](arith::MulFOp) {
                    return builder.getFloatAttr(constantType, 1.f);
                  })
                  .Case([&](arith::MulIOp) {
                    return builder.getIntegerAttr(constantType, 1);
                  })
                  .Case<arith::MaximumFOp, arith::MaxNumFOp>([&](auto) {
                    return builder.getFloatAttr(
                        constantType, -std::numeric_limits<float>::infinity());
                  })
                  .Case<arith::MinimumFOp, arith::MinNumFOp>([&](auto) {
                    return builder.getFloatAttr(
                        constantType, std::numeric_limits<float>::infinity());
                  })
                  .Case([&](arith::MinSIOp) {
                    return builder.getIntegerAttr(constantType,
                                                  llvm::maxIntN(bitWidth));
                  })
                  .Case([&](arith::MinUIOp) {
                    return builder.getIntegerAttr(constantType,
                                                  llvm::maxUIntN(bitWidth));
                  })
                  .Case([&](arith::MaxSIOp) {
                    return builder.getIntegerAttr(constantType,
                                                  llvm::minIntN(bitWidth));
                  })
                  .Case([&](arith::MaxUIOp) {
                    return builder.getIntegerAttr(constantType, 0);
                  })
                  .Case([&](arith::OrIOp) {
                    return builder.getIntegerAttr(constantType, 0);
                  })
                  .Case([&](arith::XOrIOp) {
                    return builder.getIntegerAttr(constantType, 0);
                  })
                  .Case([&](arith::AndIOp) {
                    return builder.getIntegerAttr(constantType,
                                                  llvm::maxUIntN(bitWidth));
                  })
                  .Default([](Operation *op) {
                    op->dump();
                    llvm_unreachable("Reduction op not yet supported");
                    return nullptr;
                  });
  return attr;
}

arith::ConstantOp getRedBaseConstOp(ConversionPatternRewriter &rewriter,
                                    Operation *redOp, Type constantType) {
  auto attr = getRedBaseAttr(rewriter, redOp, constantType);
  return rewriter.create<arith::ConstantOp>(redOp->getLoc(), constantType,
                                            attr);
}

bool isTypeRestrictedTargetSupportedReductionOp(mlir::Operation *redOp) {
  return isa<arith::AddFOp, arith::MaximumFOp, arith::MaxNumFOp,
             arith::MinimumFOp, arith::MinNumFOp>(redOp);
}

bool isTargetSupportedReductionOp(mlir::Operation *redOp) {
  return isTypeRestrictedTargetSupportedReductionOp(redOp) ||
         isa<arith::AddIOp, arith::MaxSIOp, arith::MinSIOp>(redOp);
}

bool isReduceLogicOp(mlir::Operation *redOp) {
  return isa<arith::XOrIOp, arith::AndIOp, arith::OrIOp>(redOp);
}

bool isTypeRestrictedTargetSupportedReduceToElementWiseOp(
    mlir::Operation *redOp) {
  return isa<arith::MulFOp>(redOp) || isReduceLogicOp(redOp);
}

bool isTargetSupportedReduceToElementWiseOp(mlir::Operation *redOp) {
  return isTypeRestrictedTargetSupportedReduceToElementWiseOp(redOp) ||
         isa<arith::MulIOp>(redOp);
}

bool isTritonAllowedReductionOp(Operation *redOp) {
  return isTargetSupportedReductionOp(redOp) ||
         isTargetSupportedReduceToElementWiseOp(redOp) ||
         isa<arith::MinUIOp, arith::MaxUIOp>(redOp);
}

bool isTargetSupportedFloatType(Type elementType) {
  // Check if the operation is a supported type.
  return elementType.isBF16() || elementType.isF16() || elementType.isF32() ||
         elementType.isTF32();
}

bool isTargetSupportedType(Type elementType) {
  // Check if the operation is a supported type.
  return isTargetSupportedFloatType(elementType) || elementType.isInteger(8);
}

bool isReductionOpAndTypeSupportedByTarget(mlir::Operation *redOp,
                                           Type elementType) {
  return isTypeRestrictedTargetSupportedReductionOp(redOp) &&
         isTargetSupportedFloatType(elementType);
}

bool isReduceToElementWiseOpAndTypeSupportedByTarget(mlir::Operation *redOp,
                                                     Type elementType,
                                                     int64_t elemCount,
                                                     int64_t rank) {

  // I1 type need special handle (Memref type i1 need 8bit alignment)
  // NOTE: Here can optimize 64 to 8 element
  return (isa<arith::MulFOp>(redOp) &&
          isTargetSupportedFloatType(elementType)) ||
         (isReduceLogicOp(redOp) &&
          !(elementType.isInteger(1) && (rank > 1 || elemCount <= 64)));
}

} // namespace mlir::triton
