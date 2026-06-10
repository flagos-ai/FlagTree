//===----------------------------------------------------------------------===//
//
// Copyright (c) 2024 The EVAS Intelligence Inc. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//==============================================================================

#include "epu/memory.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"

#define GEN_PASS_DEF_SETMEMREFSCOPE
#include "evas/Transform/Linalg/Passes.h.inc"

namespace mlir::triton::ev {

namespace {

IntegerAttr createMemScopeAttr(MLIRContext *context,
                               ::mlir::ev::MemScope scope) {
  return IntegerAttr::get(IntegerType::get(context, 64),
                          static_cast<int64_t>(scope));
}

IntegerAttr createMemScopeAttr(MLIRContext *context, int64_t scope) {
  return IntegerAttr::get(IntegerType::get(context, 64), scope);
}

MemRefType createMemRefTypeWithScope(MemRefType originalType,
                                     ::mlir::ev::MemScope scope) {
  return MemRefType::get(originalType.getShape(), originalType.getElementType(),
                         originalType.getLayout(),
                         createMemScopeAttr(originalType.getContext(), scope));
}

MemRefType createMemRefTypeWithScope(MemRefType originalType, int64_t scope) {
  return MemRefType::get(originalType.getShape(), originalType.getElementType(),
                         originalType.getLayout(),
                         createMemScopeAttr(originalType.getContext(), scope));
}

class MemRefScopeTypeConverter : public TypeConverter {
public:
  MemRefScopeTypeConverter() {
    addConversion([](Type type) { return type; });

    addConversion([](BaseMemRefType memrefType) -> Type {
      return cast<BaseMemRefType>(
          memrefType
              .clonePtrWith(createMemScopeAttr(memrefType.getContext(),
                                               ::mlir::ev::MemScope::DDR),
                            std::nullopt)
              .value());
    });

    addConversion([this](MemRefType memrefType) -> Type {
      return this->convertMemRefType(memrefType);
    });

    addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });

    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
  }

private:
  MemRefType convertMemRefType(MemRefType memrefType) {
    // if (!memrefType.hasRank() || memrefType.getRank() == 0) {
    //   return MemRefType::get(memrefType.getShape(),
    //                         memrefType.getElementType(),
    //                         memrefType.getLayout(),
    //                         IntegerAttr::get(IntegerType::get(memrefType.getContext(),
    //                         64),
    //                                        static_cast<int64_t>(::mlir::ev::MemScope::DDR)));
    // }

    
    if (memrefType.getMemorySpace() && memrefType.getMemorySpaceAsInt() > 0) {
      return memrefType;
    }

    return createMemRefTypeWithScope(memrefType, ::mlir::ev::MemScope::DDR);
  }
};

struct AllocOpConverter : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memrefType = cast<MemRefType>(op.getType());
    auto newType =
        createMemRefTypeWithScope(memrefType, ::mlir::ev::MemScope::MM);

    auto newAlloc = rewriter.create<memref::AllocOp>(op.getLoc(), newType);
    rewriter.replaceOp(op, newAlloc);
    return success();
  }
};

LogicalResult
inferResultsToMemRefTypes(Operation *op, ArrayRef<Value> operands,
                          SmallVector<Type> &inferredReturnTypes) {
  auto inferTypeInterface = dyn_cast<InferTypeOpInterface>(op);
  if (inferTypeInterface) {
    NamedAttrList attrList(op->getAttrs());
    DictionaryAttr attributes = attrList.getDictionary(op->getContext());
    if (failed(inferTypeInterface.inferReturnTypes(
            op->getContext(), op->getLoc(), operands, attributes,
            op->getPropertiesStorage(), op->getRegions(),
            inferredReturnTypes))) {
      return failure();
    }
    return success();
  }


  int64_t firstOperandScope = ::mlir::ev::MemScope::DDR;
  for (const Value &operand : operands) {
    if (auto memRefType = dyn_cast<MemRefType>(operand.getType())) {
      if (memRefType.getMemorySpace()) {
        firstOperandScope = memRefType.getMemorySpaceAsInt();
      }
      break;
    }
  }

  for (Type resultType : op->getResultTypes()) {
    if (auto memrefType = dyn_cast<MemRefType>(resultType)) {
      Type convertedType =
          createMemRefTypeWithScope(memrefType, firstOperandScope);
      inferredReturnTypes.push_back(convertedType);
    } else {
      inferredReturnTypes.push_back(resultType);
    }
  }

  return success();
}

struct GlobalOpConverter : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern<memref::GlobalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memrefType = cast<MemRefType>(op.getType());
    auto newType =
        createMemRefTypeWithScope(memrefType, ::mlir::ev::MemScope::DDR);
    auto newTypeAttr = TypeAttr::get(newType);
    auto newGlobal = rewriter.create<memref::GlobalOp>(
        op.getLoc(), 
        op.getSymNameAttr(),
        op.getSymVisibilityAttr(),
        newTypeAttr,
        op.getInitialValue().value(),
        op.getConstantAttr(),
        op.getAlignmentAttr()
    );

    rewriter.replaceOp(op, newGlobal);
    return success();
    }
  };

struct InferTypeOpConverter : public ConversionPattern {
  InferTypeOpConverter(MLIRContext *context)
      : ConversionPattern(MatchAnyOpTypeTag(), 0, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> inferredReturnTypes;
    if (failed(inferResultsToMemRefTypes(op, operands, inferredReturnTypes))) {
      return failure();
    }

    OperationState newOpState(op->getLoc(), op->getName());
    newOpState.addOperands(operands);
    newOpState.addTypes(inferredReturnTypes);
    newOpState.addAttributes(op->getAttrs());
    newOpState.addSuccessors(op->getSuccessors());

    for (auto &region : op->getRegions()) {
      newOpState.addRegion();
    }

    Operation *newOp = rewriter.create(newOpState);

    for (auto [oldRegion, newRegion] :
         llvm::zip(op->getRegions(), newOp->getRegions())) {
      rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.end());
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

// bool applyInferMemrefScopeConversion(ModuleOp moduleOp) {
//   ConversionTarget target(*moduleOp.getContext());
//   target.markUnknownOpDynamicallyLegal([](Operation *op) {
//     auto types = llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes());
//     for (auto type : types) {
//       if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
//         if (!memrefType.getMemorySpace() || memrefType.getMemorySpaceAsInt() == 0) {
//           return false;
//         }
//       }
//     }
//     return true;
//     });
//   target.addLegalOp<UnrealizedConversionCastOp>();

//   RewritePatternSet patterns(moduleOp.getContext());
//   patterns.add<InferTypeOpConverter>(moduleOp.getContext());
//   patterns.add<AllocOpConverter>(moduleOp.getContext());
//   if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
//     return false;
//   }
//   return true;
// }

struct SetMemRefScopePass
    : public ::impl::SetMemRefScopeBase<SetMemRefScopePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, memref::MemRefDialect, tptr::TPtrDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    MemRefScopeTypeConverter typeConverter;

    ConversionTarget target(getContext());

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (auto globalOp = dyn_cast<memref::GlobalOp>(op)) {
        return typeConverter.isLegal(globalOp.getType());
      }
      return typeConverter.isLegal(op->getResultTypes()) &&
             typeConverter.isLegal(op->getOperandTypes());
    });
    target.addLegalOp<UnrealizedConversionCastOp>();
    RewritePatternSet patterns(&getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    patterns.add<InferTypeOpConverter>(&getContext());
    patterns.add<AllocOpConverter>(&getContext());
    patterns.add<GlobalOpConverter>(&getContext());
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
    
  }
};

} // namespace

std::unique_ptr<Pass> createSetMemRefScopePass() {
  return std::make_unique<SetMemRefScopePass>();
}

} // namespace mlir::triton::ev
