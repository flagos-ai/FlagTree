/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "Analysis/FirstLastUserAnalysis.h"
#include "Conversion/TritonToGCU/TritonToGCUPass.h"

#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/GCU/IR/Types.h"
#include "Dialect/GCUWS/IR/Dialect.h"
#include "Dialect/MathExt/IR/MathExt.h"
#include "Dialect/MathExt/IR/MathExtTypes.h"
#include "Dialect/MemrefExt/IR/MemrefExt.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"
#include "PatternTritonGPUOpToGCU.h"
#include "TritonGCUToGCU/TritionToGCUBase.h"

#include "Utility.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTRITONTOGCUPASS
#include "Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
#define DEBUG_TYPE "triton-ir-to-gcu-ir"
namespace {
struct ConvertTritonToGCUPass
    : public mlir::impl::ConvertTritonToGCUPassBase<ConvertTritonToGCUPass> {
  using Base::Base;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<triton::TritonDialect, triton::gpu::TritonGPUDialect,
                affine::AffineDialect, arith::ArithDialect,
                memref::MemRefDialect, vector::VectorDialect, scf::SCFDialect,
                func::FuncDialect, math::MathDialect, gpu::GPUDialect,
                gcu::GCUDialect, triton::gcu::TritonGCUDialect,
                memref_ext::MemrefExtDialect, math_ext::MathExtDialect>();
  }
};

} // namespace
namespace {
struct TTFuncOpLowering : SharedConversionPattern<triton::FuncOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp ttFuncOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = ttFuncOp.getLoc();
    int32_t numArg = ttFuncOp.front().getNumArguments();
    // Remap proper input types.
    TypeConverter::SignatureConversion signConversion(numArg);
    TypeConverter::SignatureConversion newSignConversion(numArg + 2);

    // Convert argument types one by one and check for errors.
    for (auto [idx, type] :
         llvm::enumerate(ttFuncOp.getFunctionType().getInputs())) {
      SmallVector<Type, 8> converted;
      converted.push_back(getTypeConverter()->convertType(type));
      signConversion.addInputs(idx, converted);
      newSignConversion.addInputs(idx, converted);
    }
    SmallVector<Type, 4> resultTypes;
    for (auto type : ttFuncOp.getFunctionType().getResults()) {
      resultTypes.push_back(getTypeConverter()->convertType(type));
    }

    func::FuncOp func;
    if (ttFuncOp.isPublic()) {
      auto funcType = FunctionType::get(
          getContext(), signConversion.getConvertedTypes(), resultTypes);
      auto funcName = (ttFuncOp.getName() + "_triton_internal__").str();
      func = rewriter.create<func::FuncOp>(loc, funcName, funcType);
      func.setPublic();
    } else {
      std::vector<mlir::Type> argsTypes =
          signConversion.getConvertedTypes().vec();

      argsTypes.push_back(pTagPool.getPrivateTagsType());
      argsTypes.push_back(pTagPool.getSharedTagsType());

      auto funcType = FunctionType::get(getContext(), argsTypes, resultTypes);
      auto funcName = ttFuncOp.getName().str();
      func = rewriter.create<func::FuncOp>(loc, funcName, funcType);
      func.setPrivate();

      newSignConversion.addInputs(numArg, pTagPool.getPrivateTagsType());
      newSignConversion.addInputs(numArg + 1, pTagPool.getSharedTagsType());

      pTagPool.setPrivateFuncNameMap(func.getOperation(), numArg);
      pTagPool.setSharedFuncNameMap(func.getOperation(), numArg + 1);
    }

    auto internalLinkage = mlir::LLVM::linkage::Linkage::Internal;
    auto linkage = mlir::LLVM::LinkageAttr::get(getContext(), internalLinkage);
    func->setAttr("llvm.linkage", linkage);
    // Move the region to the new function, update the entry block signature.
    rewriter.inlineRegionBefore(ttFuncOp.getBody(), func.getBody(), func.end());
    if (ttFuncOp.isPublic()) {
      if (failed(rewriter.convertRegionTypes(
              &func.getBody(), *getTypeConverter(), &signConversion))) {
        return failure();
      }

      auto funcType = FunctionType::get(
          getContext(), signConversion.getConvertedTypes(), resultTypes);
      auto gpufunc =
          rewriter.create<gpu::GPUFuncOp>(loc, ttFuncOp.getName(), funcType);
      gpufunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       rewriter.getUnitAttr());
      OpBuilder::InsertionGuard guard(rewriter);
      auto entryBlock = &gpufunc.getBody().getBlocks().back();
      rewriter.setInsertionPointToStart(entryBlock);

      auto call =
          rewriter.create<func::CallOp>(loc, func, entryBlock->getArguments());
      rewriter.create<gpu::ReturnOp>(loc, call->getResults());
    } else {
      if (failed(rewriter.convertRegionTypes(
              &func.getBody(), *getTypeConverter(), &newSignConversion))) {
        return failure();
      }
    }
    rewriter.eraseOp(ttFuncOp);
    return success();
  }
};

struct TTReturnOpLowering : SharedConversionPattern<triton::ReturnOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    if (op->getParentOfType<gpu::GPUFuncOp>()) {
      rewriter.replaceOpWithNewOp<gpu::ReturnOp>(op, op.getOperands());
    } else {
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op, op.getOperands());
    }
    return success();
  }
};

struct TTCallOpLowering : SharedConversionPattern<triton::CallOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    SmallVector<Type, 4> resultTypes;
    for (auto ty : op->getResultTypes()) {
      resultTypes.push_back(getTypeConverter()->convertType(ty));
    }
    SmallVector<mlir::Value> operands(adaptor.getOperands().begin(),
                                      adaptor.getOperands().end());
    operands.push_back(pTagPool.getPrivateTagsValue(op.getOperation()));
    operands.push_back(pTagPool.getSharedTagsValue(op.getOperation()));
    rewriter.replaceOpWithNewOp<func::CallOp>(op, op.getCallee(), resultTypes,
                                              operands);
    return success();
  }
};

struct TTSCFForOpLowering : SharedConversionPattern<scf::ForOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    // Remap proper input types.
    TypeConverter::SignatureConversion signatureConversion(
        op.getBody()->getNumArguments());

    // Convert argument types one by one and check for errors.
    for (auto [idx, type] : llvm::enumerate(op.getBody()->getArgumentTypes())) {
      SmallVector<Type, 8> converted;
      converted.push_back(getTypeConverter()->convertType(type));
      signatureConversion.addInputs(idx, converted);
    }
    SmallVector<Type, 4> resultTypes;
    for (auto type : op.getResultTypes()) {
      resultTypes.push_back(getTypeConverter()->convertType(type));
    }

    auto forOp = rewriter.create<scf::ForOp>(
        loc, adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), adaptor.getInitArgs());
    forOp.getRegion().getBlocks().clear();

    rewriter.inlineRegionBefore(op.getRegion(), forOp.getRegion(),
                                forOp.getRegion().end());
    if (failed(rewriter.convertRegionTypes(
            &forOp.getRegion(), *getTypeConverter(), &signatureConversion)))
      return failure();

    replaced2Origin[forOp.getOperation()] = op.getOperation();

    rewriter.replaceOp(op, forOp);
    return success();
  }
};

struct TTSCFIfOpLowering : SharedConversionPattern<scf::IfOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    SmallVector<Type, 4> resultTypes;
    for (auto type : op.getResultTypes()) {
      resultTypes.push_back(getTypeConverter()->convertType(type));
    }

    bool hasElse = op.getNumRegions() > 1;

    auto ifOp = rewriter.create<scf::IfOp>(loc, resultTypes,
                                           adaptor.getCondition(), hasElse);

    ifOp.getThenRegion().getBlocks().clear();
    if (hasElse)
      ifOp.getElseRegion().getBlocks().clear();

    rewriter.inlineRegionBefore(op.getThenRegion(), ifOp.getThenRegion(),
                                ifOp.getThenRegion().end());
    if (hasElse)
      rewriter.inlineRegionBefore(op.getElseRegion(), ifOp.getElseRegion(),
                                  ifOp.getElseRegion().end());

    replaced2Origin[ifOp.getOperation()] = op.getOperation();

    rewriter.replaceOp(op, ifOp);
    return success();
  }
};

struct TTSCFYieldOpLowering : SharedConversionPattern<scf::YieldOp> {
  using SharedConversionPattern::SharedConversionPattern;
  std::map<Operation *, std::map<uint64_t, bool>>
      &TTYeiledOPerandHasMultiUseStage;

  TTSCFYieldOpLowering(
      const TypeConverter &converter, MLIRContext *ctx,
      triton::gcu::FirstLastUserAnalysis &userAnalysis,
      std::map<Operation *, Operation *> &replaced2Origin,
      triton::gcu::PrivateTagPool &pTagPool,
      std::map<Operation *, std::map<uint64_t, bool>> &operendStage)
      : SharedConversionPattern(converter, ctx, userAnalysis, replaced2Origin,
                                pTagPool),
        TTYeiledOPerandHasMultiUseStage(operendStage) {}

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    if (isa<scf::IfOp, scf::IndexSwitchOp>(op.getOperation()->getParentOp())) {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
      return success();
    }
    auto loc = op.getLoc();
    SmallVector<Value> updatedOperands;
    for (uint64_t i = 0; i < adaptor.getOperands().size(); ++i) {
      auto operand = adaptor.getOperands()[i];
      if (isa<MemRefType>(operand.getType())) {
        auto definingOp = operand.getDefiningOp();
        auto parent = op.getOperation()->getParentOp();
        bool isMultiUse = TTYeiledOPerandHasMultiUseStage[op.getOperation()][i];
        if (!isMultiUse) {
          updatedOperands.push_back(operand);
          continue;
        }

        auto tag = pTagPool.getPrivateSyncTagInfo(op);
        auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto shape = dyn_cast<MemRefType>(operand.getType()).getShape();
        auto size = std::accumulate(shape.begin(), shape.end(), 1,
                                    std::multiplies<int64_t>());
        if (isa<scf::ForOp, scf::WhileOp>(parent)) {
          if (replaced2Origin.count(parent) == 0) {
            llvm::report_fatal_error("can't find the parent op of scf.yield");
          }
          auto originParent = replaced2Origin[parent];
          auto lastUser =
              userAnalysis.getLastUser(originParent->getResults()[i]);

          auto newAllocOpPos =
              promoteLastUser(lastUser, userAnalysis, replaced2Origin);

          Value nextLoopTensor;
          auto ip = rewriter.saveInsertionPoint();
          if (newAllocOpPos == nullptr) {
            rewriter.setInsertionPoint(parent);
            nextLoopTensor = rewriter.create<memref::AllocOp>(
                loc, dyn_cast<MemRefType>(operand.getType()));
          } else {
            rewriter.setInsertionPoint(newAllocOpPos);
            nextLoopTensor = rewriter.create<memref::AllocOp>(
                loc, dyn_cast<MemRefType>(operand.getType()));
          }
          rewriter.restoreInsertionPoint(ip);

          addDeallocAfterLastUser(rewriter, lastUser, nextLoopTensor);

          rewriter.create<memref::DmaStartOp>(
              loc, operand, SmallVector<Value, 4>(shape.size(), zero),
              nextLoopTensor, SmallVector<Value, 4>(shape.size(), zero),
              rewriter.create<arith::ConstantIndexOp>(loc, size), tag.getTag(),
              ValueRange{tag.getIdx()});
          rewriter.create<memref::DmaWaitOp>(
              loc, tag.getTag(), ValueRange{tag.getIdx()},
              rewriter.create<arith::ConstantIndexOp>(loc, size));

          if (isa_and_nonnull<memref::AllocOp>(definingOp)) {
            for (auto user : definingOp->getUsers()) {
              if (llvm::isa<memref::DeallocOp>(user)) {
                user->moveAfter(parent);
              }
            }
          }
          updatedOperands.push_back(nextLoopTensor);
        } else {
          auto nextLoopTensor = rewriter.create<memref::AllocOp>(
              loc, dyn_cast<MemRefType>(operand.getType()));
          rewriter.create<memref::DmaStartOp>(
              loc, operand, SmallVector<Value, 4>(shape.size(), zero),
              nextLoopTensor, SmallVector<Value, 4>(shape.size(), zero),
              rewriter.create<arith::ConstantIndexOp>(loc, size), tag.getTag(),
              ValueRange{tag.getIdx()});
          rewriter.create<memref::DmaWaitOp>(
              loc, tag.getTag(), ValueRange{tag.getIdx()},
              rewriter.create<arith::ConstantIndexOp>(loc, size));
          updatedOperands.push_back(nextLoopTensor);
        }
        continue;
      }
      updatedOperands.push_back(operand);
    }

    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, updatedOperands);
    return success();
  }
};

struct TTSCFWhileOpLowering : SharedConversionPattern<scf::WhileOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    // Remap proper input types.
    TypeConverter::SignatureConversion signatureConversionBefore(
        op.getBeforeBody()->getNumArguments());

    // Convert argument types one by one and check for errors.
    for (auto [idx, type] :
         llvm::enumerate(op.getBeforeBody()->getArgumentTypes())) {
      SmallVector<Type, 8> converted;
      converted.push_back(getTypeConverter()->convertType(type));
      signatureConversionBefore.addInputs(idx, converted);
    }

    TypeConverter::SignatureConversion signatureConversionAfter(
        op.getBody()->getNumArguments());

    // Convert argument types one by one and check for errors.
    for (auto [idx, type] :
         llvm::enumerate(op.getAfterBody()->getArgumentTypes())) {
      SmallVector<Type, 8> converted;
      converted.push_back(getTypeConverter()->convertType(type));
      signatureConversionAfter.addInputs(idx, converted);
    }

    SmallVector<Type, 4> resultTypes;
    for (auto type : op.getResultTypes()) {
      resultTypes.push_back(getTypeConverter()->convertType(type));
    }

    auto whileOp =
        rewriter.create<scf::WhileOp>(loc, resultTypes, adaptor.getInits());
    whileOp.getBefore().getBlocks().clear();
    rewriter.inlineRegionBefore(op.getBefore(), whileOp.getBefore(),
                                whileOp.getBefore().end());
    whileOp.getAfter().getBlocks().clear();
    rewriter.inlineRegionBefore(op.getAfter(), whileOp.getAfter(),
                                whileOp.getAfter().end());
    if (failed(rewriter.convertRegionTypes(&whileOp.getBefore(),
                                           *getTypeConverter(),
                                           &signatureConversionBefore)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&whileOp.getAfter(),
                                           *getTypeConverter(),
                                           &signatureConversionAfter)))
      return failure();
    replaced2Origin[whileOp.getOperation()] = op.getOperation();

    rewriter.replaceOp(op, whileOp);
    return success();
  }
};

struct TTSCFConditionLowering : SharedConversionPattern<scf::ConditionOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    // Remap proper input types.
    auto conditionOp = rewriter.create<scf::ConditionOp>(
        loc, adaptor.getCondition(), adaptor.getArgs());
    rewriter.replaceOp(op, conditionOp);
    return success();
  }
};

template <typename FT, typename TT>
struct TTIntrinsicOpLowering : SharedConversionPattern<FT> {
  using SharedConversionPattern<FT>::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(FT op,
                  typename SharedConversionPattern<FT>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (this->pTagPool.isExistInMap(op.getOperation())) {
      this->pTagPool.releaseMap(op.getOperation());
    }
    gpu::Dimension dim = gpu::Dimension::x;
    switch (op.getAxis()) {
    case triton::ProgramIDDim::X:
      dim = gpu::Dimension::x;
      break;
    case triton::ProgramIDDim::Y:
      dim = gpu::Dimension::y;
      break;
    case triton::ProgramIDDim::Z:
      dim = gpu::Dimension::z;
      break;
    default:
      dim = gpu::Dimension::x;
      break;
    }
    auto loc = op.getLoc();
    auto newOp = rewriter.create<arith::IndexCastOp>(
        loc, this->getTypeConverter()->convertType(op.getType()),
        rewriter.create<TT>(loc, dim));
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct TTAssertOpLowering : SharedConversionPattern<triton::AssertOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp assertOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (pTagPool.isExistInMap(assertOp.getOperation())) {
      pTagPool.releaseMap(assertOp.getOperation());
    }
    auto loc = assertOp.getLoc();

    auto message = assertOp.getMessage();

    auto assertSingleElement = [&](Value operand, ValueRange iters) {
      // load single element
      auto value = TypeSwitch<Type, Value>(operand.getType())
                       .Case<gcu::PtrType>([&](auto ty) {
                         return rewriter.create<gcu::PtrToIntOp>(loc, operand);
                       })
                       .Default([&](auto ty) { return operand; });
      // Create gcu.assert op
      rewriter.create<gcu::AssertOp>(
          loc, value, mlir::StringAttr::get(rewriter.getContext(), message), "",
          "", 0);
    };

    auto assertMemrefCondition = [&](Value operand) {
      TypeSwitch<Type>(operand.getType())
          .Case<MemRefType>([&](auto ty) {
            // use loop nest to load all elements in memref
            affine::buildAffineLoopNest(
                rewriter, loc, SmallVector<int64_t, 4>(ty.getRank(), 0),
                ty.getShape(), SmallVector<int64_t, 4>(ty.getRank(), 1),
                [&](OpBuilder &builder, Location loc, ValueRange iters) {
                  auto v = builder.create<memref::LoadOp>(loc, operand, iters);
                  assertSingleElement(v, iters);
                });
          })
          .Default([&](auto ty) { assertSingleElement(operand, {}); });
    };

    // handle memref
    assertMemrefCondition(adaptor.getCondition());

    rewriter.eraseOp(assertOp);
    return success();
  }
};

struct TTPrintOpLowering : SharedConversionPattern<triton::PrintOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PrintOp printOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, printOp.getOperation());
    if (pTagPool.isExistInMap(printOp.getOperation())) {
      pTagPool.releaseMap(printOp.getOperation());
    }
    auto loc = printOp.getLoc();
    auto printOpPrefix = printOp.getPrefix();
    auto hex = printOp.getHex();

    // Simple printf of a string without any tensors.
    if (printOp.getNumOperands() == 0) {
      rewriter.create<gpu::PrintfOp>(loc, (printOpPrefix + "\n").str(),
                                     ValueRange{});
      rewriter.eraseOp(printOp);
      return success();
    }

    auto printSingleElement = [&](Value operand, size_t i, size_t n,
                                  ValueRange iters) {
      std::string formatStr;
      llvm::raw_string_ostream os(formatStr);
      os << printOpPrefix << ": ";
      if (n > 1)
        os << "(operand " << i << ") ";

      // format
      auto msg = TypeSwitch<Type, StringRef>(operand.getType())
                     .Case<gcu::PtrType, IndexType>([&](auto ty) {
                       if (hex) {
                         os << "0x%x ";
                         return "0x%x ";
                       } else {
                         os << "%d ";
                         return "%d ";
                       }
                     })
                     .Case<IntegerType>([&](auto ty) {
                       auto isSigned = ty.isSigned();
                       auto width = ty.getWidth();
                       if (hex) {
                         if (width < 64) {
                           os << "0x%x ";
                           return "0x%x ";
                         } else {
                           os << "0x%llx ";
                           return "0x%llx ";
                         }
                       } else {
                         if (isSigned) {
                           if (width < 64) {
                             os << "%d ";
                             return "%d ";
                           } else {
                             os << "%lld ";
                             return "%lld ";
                           }
                         }
                         if (width < 64) {
                           os << "%u ";
                           return "%u ";
                         } else {
                           os << "%llu ";
                           return "%llu ";
                         }
                       }
                     })
                     .Default([&](auto ty) {
                       os << "%f ";
                       return "%f ";
                     });

      // value
      SmallVector<Value, 4> values;
      auto value = TypeSwitch<Type, Value>(operand.getType())
                       .Case<gcu::PtrType>([&](auto ty) {
                         return rewriter.create<gcu::PtrToIntOp>(loc, operand);
                       })
                       .Default([&](auto ty) { return operand; });
      values.push_back(value);

      if (!iters.empty()) {
        // idx format
        os << "(idx ";
        for (auto iter = iters.begin(); iter != iters.end(); ++iter) {
          if (iter != iters.begin())
            os << ", ";
          os << "%d";
        }
        os << ")";
        // idx value
        values.append(iters.begin(), iters.end());
      }
      os << "\n";

      if (!msg.empty())
        rewriter.create<gpu::PrintfOp>(loc, formatStr, ValueRange{values});
    };

    auto printOperand = [&](Value operand, size_t i, size_t n) {
      TypeSwitch<Type>(operand.getType())
          .Case<MemRefType>([&](auto ty) {
            affine::buildAffineLoopNest(
                rewriter, loc, SmallVector<int64_t, 4>(ty.getRank(), 0),
                ty.getShape(), SmallVector<int64_t, 4>(ty.getRank(), 1),
                [&](OpBuilder &builder, Location loc, ValueRange iters) {
                  auto v = builder.create<memref::LoadOp>(loc, operand, iters);
                  printSingleElement(v, i, n, iters);
                });
          })
          .Default([&](auto ty) { printSingleElement(operand, i, n, {}); });
    };

    // print all operands by order
    for (size_t i = 0; i < adaptor.getOperands().size(); ++i) {
      printOperand(adaptor.getOperands()[i], i, adaptor.getOperands().size());
    }

    rewriter.eraseOp(printOp);
    return success();
  }
};

struct TTSplatOpLowering : SharedConversionPattern<triton::SplatOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp splatOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, splatOp.getOperation());
    if (pTagPool.isExistInMap(splatOp.getOperation())) {
      pTagPool.releaseMap(splatOp.getOperation());
    }
    auto lastUser =
        userAnalysis.getLastUser(splatOp.getOperation()->getResults()[0]);
    auto loc = splatOp.getLoc();
    auto numElems = triton::gcu::getElemsPerThread(splatOp.getType());
    auto resultType = dyn_cast<MemRefType>(
        getTypeConverter()->convertType(splatOp.getType()));
    auto output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                              replaced2Origin, resultType);
    auto v = isa<triton::PointerType>(splatOp.getSrc().getType())
                 ? rewriter.create<gcu::PtrToIntOp>(loc, adaptor.getSrc())
                 : adaptor.getSrc();
    auto totalNumElems = triton::gcu::getTotalElemsPerThread(splatOp.getType());
    auto tag = pTagPool.getPrivateSyncTagInfo(splatOp.getOperation());
    doMemset(rewriter, tag, splatOp, output, v, totalNumElems);
    leaveTritionOp(rewriter, splatOp.getOperation());
    rewriter.replaceOp(splatOp, output);
    return success();
  }
};

struct TTConstantOpLowering : SharedConversionPattern<arith::ConstantOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp constOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, constOp.getOperation());
    if (pTagPool.isExistInMap(constOp.getOperation())) {
      pTagPool.releaseMap(constOp.getOperation());
    }
    auto loc = constOp.getLoc();
    if (!isa<TensorType>(constOp.getType()))
      return failure();
    auto lastUser =
        userAnalysis.getLastUser(constOp.getOperation()->getResults()[0]);
    auto totalNumElems = triton::gcu::getTotalElemsPerThread(constOp.getType());
    auto resultType = dyn_cast<MemRefType>(
        getTypeConverter()->convertType(constOp.getType()));
    auto valueAttr = constOp.getValue();
    auto array = dyn_cast<DenseElementsAttr>(valueAttr);
    auto output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                              replaced2Origin, resultType);

    // only support splat constant
    auto attr = array.getSplatValue<TypedAttr>();
    auto v =
        rewriter.create<arith::ConstantOp>(loc, array.getElementType(), attr);
    auto tag = pTagPool.getPrivateSyncTagInfo(constOp.getOperation());
    doMemset(rewriter, tag, constOp, output, v, totalNumElems);
    leaveTritionOp(rewriter, constOp.getOperation());
    rewriter.replaceOp(constOp, output);
    return success();
  }
};

struct TTAddPtrOpLowering : SharedConversionPattern<triton::AddPtrOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp addPtrOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, addPtrOp.getOperation());
    if (pTagPool.isExistInMap(addPtrOp.getOperation())) {
      pTagPool.releaseMap(addPtrOp.getOperation());
    }
    auto loc = addPtrOp.getLoc();
    // vector
    if (isa<TensorType>(addPtrOp.getType())) {
      auto lastUser =
          userAnalysis.getLastUser(addPtrOp.getOperation()->getResults()[0]);
      auto numElems = triton::gcu::getElemsPerThread(addPtrOp.getType());
      auto resultType = dyn_cast<MemRefType>(
          getTypeConverter()->convertType(addPtrOp.getType()));
      auto output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                replaced2Origin, resultType);
      auto ptrs = adaptor.getPtr();
      auto offsets = adaptor.getOffset();
      affine::buildAffineLoopNest(
          rewriter, loc, SmallVector<int64_t, 4>(numElems.size(), 0),
          SmallVector<int64_t, 4>(numElems.begin(), numElems.end()),
          SmallVector<int64_t, 4>(numElems.size(), 1),
          [&](OpBuilder &builder, Location loc, ValueRange iters) {
            auto ptrType =
                dyn_cast<gcu::PtrType>(getTypeConverter()->convertType(
                    dyn_cast<TensorType>(addPtrOp.getType()).getElementType()));
            auto elemType = ptrType.getElementType();
            auto elemBytes = (elemType.getIntOrFloatBitWidth() + 7) / 8;
            auto lhs = builder.create<memref::LoadOp>(loc, ptrs, iters);
            auto rhs =
                builder.create<memref::LoadOp>(loc, offsets, iters).getResult();
            rhs = builder.create<arith::MulIOp>(
                loc,
                rhs.getType().getIntOrFloatBitWidth() < 64
                    ? builder.create<arith::ExtSIOp>(loc, builder.getI64Type(),
                                                     rhs)
                    : rhs,
                builder.create<arith::ConstantIntOp>(loc, elemBytes, 64));
            auto v = builder.create<arith::AddIOp>(loc, lhs, rhs);
            builder.create<memref::StoreOp>(loc, v, output, iters);
          });
      doMemFence(rewriter, addPtrOp);
      leaveTritionOp(rewriter, addPtrOp.getOperation());
      rewriter.replaceOp(addPtrOp, output);
      return success();
    }

    // scalar
    auto resultType = dyn_cast<gcu::PtrType>(
        getTypeConverter()->convertType(addPtrOp.getType()));
    auto elemType = resultType.getElementType();
    auto elemBytes = (elemType.getIntOrFloatBitWidth() + 7) / 8;
    auto ptr = adaptor.getPtr();
    auto offset = adaptor.getOffset();
    if (offset.getType().getIntOrFloatBitWidth() < 64) {
      offset =
          rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), offset);
    }
    offset = rewriter.create<arith::MulIOp>(
        loc, offset,
        rewriter.create<arith::ConstantIntOp>(
            loc, getElementTypeOrSelf(offset.getType()), elemBytes));
    auto v = rewriter.create<gcu::IntToPtrOp>(
        loc, resultType,
        rewriter.create<arith::AddIOp>(
            loc, rewriter.create<gcu::PtrToIntOp>(loc, ptr), offset));
    rewriter.replaceOp(addPtrOp, v);
    return success();
  }
};

struct TTArithSelectOpLowering
    : public SharedConversionPattern<arith::SelectOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult matchAndRewrite(
      arith::SelectOp op,
      typename SharedConversionPattern<arith::SelectOp>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    auto type = op.getType();
    if (!isa<triton::PointerType>(type)) {
      leaveTritionOp(rewriter, op.getOperation());
      return failure();
    }
    auto ty = this->getTypeConverter()->convertType(type);
    auto newOp = rewriter.create<arith::SelectOp>(
        loc, ty, adaptor.getOperands(), op->getAttrs());
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

template <typename FT, typename TT>
struct TTElementwiseOpLowering : public SharedConversionPattern<FT> {
  using SharedConversionPattern<FT>::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(FT op,
                  typename SharedConversionPattern<FT>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (this->pTagPool.isExistInMap(op.getOperation())) {
      this->pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    auto type = op.getType();
    if (!isa<TensorType>(type)) {
      auto ty = this->getTypeConverter()->convertType(type);
      auto newOp = rewriter.create<TT>(loc, ty, adaptor.getOperands());
      rewriter.replaceOp(op, newOp->getResults());
      leaveTritionOp(rewriter, op.getOperation());
      return success();
    }
    auto lastUser =
        this->userAnalysis.getLastUser(op.getOperation()->getResults()[0]);
    auto numElems = triton::gcu::getElemsPerThread(type);
    auto resultType =
        dyn_cast<MemRefType>(this->getTypeConverter()->convertType(type));
    auto output = syncAllocOp(rewriter, loc, lastUser, this->userAnalysis,
                              this->replaced2Origin, resultType);
    affine::buildAffineLoopNest(
        rewriter, loc, SmallVector<int64_t, 4>(numElems.size(), 0),
        SmallVector<int64_t, 4>(numElems.begin(), numElems.end()),
        SmallVector<int64_t, 4>(numElems.size(), 1),
        [&](OpBuilder &builder, Location loc, ValueRange iters) {
          SmallVector<Value, 4> operands;
          for (auto operand : adaptor.getOperands()) {
            operands.push_back(
                builder.create<memref::LoadOp>(loc, operand, iters));
          }
          auto v = builder.create<TT>(loc, resultType.getElementType(),
                                      operands, op->getAttrs());
          builder.create<memref::StoreOp>(loc, v, output, iters);
        });
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct TTBitcastOpLowering : SharedConversionPattern<triton::BitcastOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    auto type = this->getTypeConverter()->convertType(op.getType());
    if (!isa<TensorType>(op.getType())) {
      // arith.bitcast doesn't support pointers
      if (isa<triton::PointerType>(op.getSrc().getType()) &&
          isa<triton::PointerType>(op.getResult().getType())) {
        auto result = rewriter.create<gcu::IntToPtrOp>(
            loc, type, rewriter.create<gcu::PtrToIntOp>(loc, adaptor.getSrc()));
        rewriter.replaceOp(op, result);
        return success();
      } else {
        rewriter.replaceOpWithNewOp<arith::BitcastOp>(op, type,
                                                      adaptor.getSrc());
        return success();
      }
    }

    auto dstType = dyn_cast<MemRefType>(type);
    auto srcType = dyn_cast<MemRefType>(adaptor.getSrc().getType());

    if (dstType.getNumElements() != srcType.getNumElements())
      return op.emitOpError("src and dst element number mismatch");

    auto totalNumElems = rewriter.create<arith::ConstantIndexOp>(
        loc, triton::gcu::getTotalElemsPerThread(op.getSrc().getType()));
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto srcBuf = rewriter.create<memref::ReinterpretCastOp>(
        loc,
        MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic},
                        srcType.getElementType()),
        adaptor.getSrc(), zero, ArrayRef<Value>{totalNumElems},
        ArrayRef<Value>{one});
    auto srcPtrType = gcu::PtrType::get(getContext(), srcType.getElementType());
    auto srcPtr = rewriter.create<gcu::MemRefToPtrOp>(loc, srcPtrType, srcBuf);
    auto ptrInt = rewriter.create<gcu::PtrToIntOp>(loc, srcPtr);
    auto dstPtrType = gcu::PtrType::get(getContext(), dstType.getElementType());
    auto dstPtr = rewriter.create<gcu::IntToPtrOp>(loc, dstPtrType, ptrInt);
    auto dstBuf = rewriter.create<gcu::PtrToMemRefOp>(
        loc,
        MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic},
                        dstType.getElementType()),
        dstPtr);
    auto [strides, offset] = dstType.getStridesAndOffset();
    auto dst = rewriter.create<memref::ReinterpretCastOp>(
        loc, dstType, dstBuf, offset, dstType.getShape(), strides);
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, dst);
    return success();
  }
};

struct TTReduceReturnOpLowering
    : SharedConversionPattern<triton::ReduceReturnOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, returnOp.getOperation());
    if (pTagPool.isExistInMap(returnOp.getOperation())) {
      pTagPool.releaseMap(returnOp.getOperation());
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(returnOp, returnOp.getOperands());
    return success();
  }
};

struct TTScanReturnOpLowering : SharedConversionPattern<triton::ScanReturnOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ScanReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, returnOp.getOperation());
    if (pTagPool.isExistInMap(returnOp.getOperation())) {
      pTagPool.releaseMap(returnOp.getOperation());
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(returnOp, returnOp.getOperands());
    return success();
  }
};

struct TTUnsplatOpLowering : SharedConversionPattern<triton::UnsplatOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::UnsplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();

    SmallVector<Value> indices;
    indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));

    Value unsplat;
    if (isa<RankedTensorType>(src.getType())) {
      // TODO(hzshao TBD): we doesn't have test case for this, so we don't
      // support it now.
      // unsplat = rewriter.create<tensor::ExtractOp>(loc, src, indices);
      return failure();
    } else if (isa<MemRefType>(src.getType())) {
      unsplat = rewriter.create<memref::LoadOp>(loc, src, indices);
    } else {
      return failure();
    }

    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, unsplat);
    return success();
  }
};

struct TTExternElemwiseOpLowering
    : SharedConversionPattern<triton::ExternElementwiseOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto name = op.getSymbol();
    if (name == "__nv_fmaxf") {
      rewriter.replaceOpWithNewOp<arith::MaximumFOp>(op, adaptor.getOperands());
      return success();
    } else if (name == "__nv_fminf") {
      rewriter.replaceOpWithNewOp<arith::MinimumFOp>(op, adaptor.getOperands());
      return success();
    } else if (name == "__nv_floorf") {
      rewriter.replaceOpWithNewOp<math::FloorOp>(op, adaptor.getOperands());
      return success();
    } else if (name == "__nv_min") {
      rewriter.replaceOpWithNewOp<arith::MinSIOp>(op, adaptor.getOperands());
      return success();
    } else if (name == "__nv_max") {
      rewriter.replaceOpWithNewOp<arith::MaxSIOp>(op, adaptor.getOperands());
      return success();
    } else if (name == "__nv_umin") {
      rewriter.replaceOpWithNewOp<arith::MinUIOp>(op, adaptor.getOperands());
      return success();
    } else if (name == "__nv_umax") {
      rewriter.replaceOpWithNewOp<arith::MaxUIOp>(op, adaptor.getOperands());
      return success();
    } else if (name == "__nv_powf") {
      rewriter.replaceOpWithNewOp<math::PowFOp>(op, adaptor.getOperands());
      return success();
    } else if (name == "__nv_log2f") {
      rewriter.replaceOpWithNewOp<math::Log2Op>(op, adaptor.getOperands());
      return success();
    } else if (name == "__nv_exp2f") {
      rewriter.replaceOpWithNewOp<math::Exp2Op>(op, adaptor.getOperands());
      return success();
    } else if (name == "__nv_rsqrtf") {
      rewriter.replaceOpWithNewOp<math::RsqrtOp>(op, adaptor.getOperands());
      return success();
    } else if (name == "__nv_tanhf") {
      rewriter.replaceOpWithNewOp<math::TanhOp>(op, adaptor.getOperands());
      return success();
    } else if (name == "__gcu_begin_clock") {
      auto newOp = rewriter.create<gcu::BeginClockOp>(op->getLoc(),
                                                      adaptor.getOperands());
      rewriter.replaceOp(op, newOp->getResults());
      return success();
    } else if (name == "__gcu_end_clock") {
      auto newOp =
          rewriter.create<gcu::EndClockOp>(op->getLoc(), adaptor.getOperands());
      rewriter.replaceOp(op, newOp->getResults());
      return success();
    }

    auto libname = op.getLibname();
    if (libname == "libnvshmem_device") {
      auto loc = op->getLoc();
      auto resultType = op.getResult().getType();
      auto newOp = rewriter.create<gcu::ExternDeviceCallOp>(
          loc, resultType, adaptor.getOperands(), op.getLibnameAttr(),
          op.getSymbolAttr(), op.getPureAttr());
      rewriter.replaceOp(op, newOp->getResults());
      return success();
    }

    return failure();
  }
};

struct TTHistogramOpLowering : SharedConversionPattern<triton::HistogramOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::HistogramOp histogramOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, histogramOp.getOperation());
    if (pTagPool.isExistInMap(histogramOp.getOperation())) {
      pTagPool.releaseMap(histogramOp.getOperation());
    }
    auto loc = histogramOp.getLoc();
    auto zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0));
    auto zeroIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto oneIndex = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto lastUser =
        userAnalysis.getLastUser(histogramOp.getOperation()->getResults()[0]);
    auto tag = pTagPool.getPrivateSyncTagInfo(histogramOp);
    auto resultType = histogramOp.getType();
    auto wrapResultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(resultType));
    auto resultMemRefType =
        MemRefType::get(resultType.getShape(), wrapResultType.getElementType());
    auto totalNumElems = triton::gcu::getTotalElemsPerThread(resultType);
    auto encoding = resultType.getEncoding();
    auto warpsPerCTA = triton::gcu::getWarpsPerCTA(encoding);

    // Check if we can use mradix hardware (<= 32 bins, as mradix has 32 bins.)
    auto inputTensorType =
        dyn_cast<RankedTensorType>(histogramOp.getOperand(0).getType());
    int64_t numBins = resultType.getShape()[0];
    bool useMradix =
        (numBins <= 32) && inputTensorType.getElementType().isInteger(32);

    if (useMradix) {
      // Gather all warp inputs to shared memory, then single-warp
      // mradix histogram. This avoids multi-warp merge and uses hardware
      // acceleration.

      // Step 1: All warps store their input shards to shared memory
      auto inputType =
          dyn_cast<RankedTensorType>(histogramOp.getOperand(0).getType());
      auto sharedInputTensorType = RankedTensorType::get(
          inputType.getShape(), inputType.getElementType(), encoding);
      auto sharedInputMem = storeToSharedMem(
          rewriter, tag, sharedInputTensorType, adaptor.getSrc(), false,
          std::make_pair(histogramOp.getOperation(), -1), userAnalysis,
          replaced2Origin);

      // Step 2: Allocate output in SMEM (skip SMEM->private copy for input)
      auto smemResultType = MemRefType::get(
          resultType.getShape(), wrapResultType.getElementType(), AffineMap{},
          rewriter.getI64IntegerAttr(2));
      auto smemResMem = syncAllocOp(
          rewriter, loc, std::make_pair(histogramOp.getOperation(), -1),
          userAnalysis, replaced2Origin, smemResultType);

      // Step 3: Master warp runs mradix histogram directly on SMEM
      auto masterWarpId = getMasterThreadId(histogramOp.getOperation());
      auto isMasterThread = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
          rewriter.create<arith::ConstantIndexOp>(loc, masterWarpId));
      auto useMradixAttr = rewriter.getUnitAttr();
      rewriter.create<scf::IfOp>(
          loc, isMasterThread, [&](OpBuilder &builder, Location loc) {
            doMemset(builder, tag, histogramOp, smemResMem, zero,
                     totalNumElems);
            builder.create<math_ext::HistogramOp>(
                loc, smemResMem, sharedInputMem, useMradixAttr);
            builder.create<scf::YieldOp>(loc);
          });
      rewriter.create<gpu::BarrierOp>(loc);

      // Step 4: Broadcast SMEM output to per-warp private
      auto finalOutput = loadFromSharedMem(
          rewriter, tag, resultType, smemResMem, false, lastUser,
          std::make_pair(nullptr, -1), userAnalysis, replaced2Origin);

      rewriter.create<gpu::BarrierOp>(loc);
      leaveTritionOp(rewriter, histogramOp.getOperation());
      rewriter.replaceOp(histogramOp, finalOutput);
      return success();
    }

    // Fallback: Per-warp histogram + merge path
    auto resCurWarp = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                  replaced2Origin, resultMemRefType);
    doMemset(rewriter, tag, histogramOp, resCurWarp, zero, totalNumElems);
    auto sharedMemTensorType = RankedTensorType::get(
        ArrayRef<int64_t>{resultType.getShape()[0] * warpsPerCTA[0]},
        wrapResultType.getElementType(), encoding);
    rewriter.create<math_ext::HistogramOp>(loc, resCurWarp, adaptor.getSrc(),
                                           UnitAttr());
    /// store res of every warp to shared memory
    auto sharedResMem =
        storeToSharedMem(rewriter, tag, sharedMemTensorType, resCurWarp, false,
                         std::make_pair(histogramOp.getOperation(), -1),
                         userAnalysis, replaced2Origin);
    rewriter.create<memref::DeallocOp>(loc, resCurWarp);
    size_t allResSize = resultType.getShape()[0];
    size_t warpResSize = wrapResultType.getShape()[0];
    auto finalOutput = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                   replaced2Origin, wrapResultType);
    doMemset(rewriter, tag, histogramOp, finalOutput, zero, totalNumElems);
    size_t warpsWalkNum = warpsPerCTA[0];
    // if input can't be divided by warp, do not calculate sum of res of every
    // warp
    if (dyn_cast<TensorType>(histogramOp.getOperand(0).getType())
            .getShape()[0] < warpsPerCTA[0])
      warpsWalkNum = dyn_cast<TensorType>(histogramOp.getOperand(0).getType())
                         .getShape()[0];
    /// Compute the results in shared memory based on the output each warp
    /// should produce
    auto warpIdsOfRes = getWarpIds(rewriter, loc, resultType);
    scf::buildLoopNest(
        rewriter, loc, SmallVector<Value, 4>{zeroIndex},
        SmallVector<Value, 4>{
            rewriter.create<arith::ConstantIndexOp>(loc, warpResSize)},
        SmallVector<Value, 4>{oneIndex},
        [&](OpBuilder &builder, Location loc, ValueRange gramIndex) {
          auto res = builder.create<arith::ConstantOp>(
              loc, rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0));
          SmallVector<Value> iterArgs = {res};
          builder.create<scf::ForOp>(
              loc, zeroIndex,
              builder.create<arith::ConstantIndexOp>(loc, warpsWalkNum),
              oneIndex, iterArgs,
              [&](OpBuilder &builder, Location loc, Value warpId,
                  ValueRange sum) {
                auto baseIndexOfRes = builder.create<arith::MulIOp>(
                    loc, warpIdsOfRes[0],
                    builder.create<arith::ConstantIndexOp>(loc, warpResSize));
                auto index = builder.create<arith::AddIOp>(
                    loc,
                    builder.create<arith::AddIOp>(loc, gramIndex[0],
                                                  baseIndexOfRes),
                    builder.create<arith::MulIOp>(
                        loc, warpId,
                        builder.create<arith::ConstantIndexOp>(loc,
                                                               allResSize)));
                auto warpRes = builder.create<memref::LoadOp>(
                    loc, sharedResMem, SmallVector<Value, 4>{index});
                Value newSum =
                    builder.create<arith::AddIOp>(loc, sum[0], warpRes);
                builder.create<memref::StoreOp>(loc, newSum, finalOutput,
                                                gramIndex[0]);
                builder.create<scf::YieldOp>(loc, ValueRange{newSum});
              });
        });
    rewriter.create<gpu::BarrierOp>(loc);
    leaveTritionOp(rewriter, histogramOp.getOperation());
    rewriter.replaceOp(histogramOp, finalOutput);
    return success();
  }
};

struct GCULoadOpLowering : SharedConversionPattern<triton::gcu::LoadOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcu::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, loadOp.getOperation());
    if (pTagPool.isExistInMap(loadOp.getOperation())) {
      pTagPool.releaseMap(loadOp.getOperation());
    }
    auto loc = loadOp.getLoc();
    auto loadType = loadOp.getType();

    if (!isa<TensorType>(loadType))
      return failure();

    auto originOp = loadOp.getOperation();
    if (replaced2Origin.count(originOp) != 0) {
      originOp = replaced2Origin[originOp];
    }
    auto lastUser = userAnalysis.getLastUser(originOp->getResults()[0]);
    auto firstUser = userAnalysis.getFirstUser(originOp->getResults()[0]);

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto elemType = loadOp.getPtr().getType().getElementType();
    auto resultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(loadType));
    bool IsShareOutput = false; // output is shared layout
    if (auto tType = dyn_cast<RankedTensorType>(loadType))
      if (mlir::isa<triton::gpu::SharedEncodingTrait>(tType.getEncoding()))
        IsShareOutput = true;

    auto output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                              replaced2Origin, resultType);

    triton::gcu::TagInfo tag;
    if (IsShareOutput) {
      tag = pTagPool.getSharedSyncTagInfo(loadOp);
    } else {
      if (firstUser.first != nullptr) {
        tag = pTagPool.tryGetPrivateAsyncTagInfo(loadOp);
      } else {
        tag = pTagPool.getPrivateSyncTagInfo(loadOp);
      }
    }
    if (tag.isAsync()) {
      pTagPool.setMap(firstUser.first, tag);
    }

    auto defaultValue =
        loadOp.getDefaultValue()
            ? adaptor.getDefaultValue()
            : triton::gcu::createConstantZero(rewriter, loc, elemType);

    // workaround for offset > tensor dims
    int64_t rank = resultType.getRank();
    Value shapeCheck = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, adaptor.getShape()[0], zero);
    for (unsigned i = 1; i < rank; ++i) {
      auto dimCheck = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, adaptor.getShape()[i], zero);
      shapeCheck = rewriter.create<arith::AndIOp>(loc, shapeCheck, dimCheck);
    }

    auto total_size =
        rewriter
            .create<scf::IfOp>(
                loc, shapeCheck,
                [&](OpBuilder builder, Location loc) {
                  Value load_size = zero;
                  load_size = ConfigGcuLoad(
                      builder, loc, output, loadOp, resultType,
                      adaptor.getPtr(), adaptor.getStrides(),
                      adaptor.getShape(), defaultValue, tag, IsShareOutput);
                  builder.create<scf::YieldOp>(loc, ValueRange{load_size});
                },
                [&](OpBuilder &builder, Location loc) {
                  auto totalNumElems =
                      triton::gcu::getTotalElemsPerThread(loadType);
                  doMemset(builder, tag, loadOp, output, defaultValue,
                           totalNumElems);
                  if (triton::gcu::get_bool_env("TRITON_GCU_DEBUG")) {
                    std::string locStr = "[warning]: load offset is out of "
                                         "range for tensor. loc";
                    if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(loc)) {
                      llvm::StringRef filename = fileLineColLoc.getFilename();
                      locStr += filename.str();
                      locStr += ":";
                      locStr += std::to_string(fileLineColLoc.getLine());
                    }
                    builder.create<gpu::PrintfOp>(loc, locStr, ValueRange{});
                  }
                  builder.create<scf::YieldOp>(loc, ValueRange{zero});
                })
            .getResult(0);
    if (IsShareOutput) {
      auto masterWarpId = getMasterThreadId(loadOp.getOperation());
      auto isMasterThread = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
          rewriter.create<arith::ConstantIndexOp>(loc, masterWarpId));
      auto isAll =
          rewriter.create<arith::AndIOp>(loc, isMasterThread, shapeCheck);
      rewriter.create<scf::IfOp>(
          loc, isAll, [&](OpBuilder builder, Location loc) {
            WaitGcuLoadStore(builder, loc, tag, total_size);
            builder.create<scf::YieldOp>(loc);
          });
      rewriter.create<gpu::BarrierOp>(loc);
    } else {
      auto ip = rewriter.saveInsertionPoint();
      if (tag.isAsync()) {
        rewriter.setInsertionPoint(firstUser.first);
      }
      rewriter.create<scf::IfOp>(
          loc, shapeCheck, [&](OpBuilder builder, Location loc) {
            WaitGcuLoadStore(builder, loc, tag, total_size);
            builder.create<scf::YieldOp>(loc);
          });
      rewriter.restoreInsertionPoint(ip);
    }

    leaveTritionOp(rewriter, loadOp.getOperation());
    rewriter.replaceOp(loadOp, output);
    return success();
  }
};

struct GCUStoreOpLowering : SharedConversionPattern<triton::gcu::StoreOp> {
  using SharedConversionPattern::SharedConversionPattern;

  static bool needsTransposeForStore(triton::gcu::StoreOp storeOp) {
    int64_t rank = storeOp.getValue().getType().getRank();
    auto hint = storeOp.getOrderHint();
    int64_t hint_size = static_cast<int64_t>(hint.size());

    SmallVector<int32_t> order_hint;
    for (int64_t i = 0; i < rank; ++i)
      order_hint.push_back(hint_size == 0 ? -1 : hint[i]);

    bool bDynamicStride = false;
    for (int64_t i = 0; i < rank; ++i) {
      if (order_hint[i] == -1)
        bDynamicStride = true;
    }

    bool bReshape = true;
    for (int64_t i = 0; i < rank; ++i) {
      if ((order_hint[i] == 0 && !bDynamicStride) ||
          (order_hint[i] == 1 && bDynamicStride)) {
        bReshape = false;
        break;
      }
    }

    if (bReshape) {
      if (bDynamicStride) {
        order_hint.push_back(1);
      } else {
        for (int64_t i = 0; i < rank; ++i)
          order_hint[i]--;
        order_hint.push_back(rank);
      }
      rank += 1;
    }

    if (rank == 2 && bDynamicStride) {
      if (order_hint[1] == 1) {
        order_hint[0] = 0;
        bDynamicStride = false;
      } else if (order_hint[0] == 1) {
        order_hint[1] = 0;
        bDynamicStride = false;
      }
    }

    if (bDynamicStride)
      return true;

    for (int64_t i = 0; i < rank; ++i) {
      if (order_hint[i] != i)
        return true;
    }
    return false;
  }

  LogicalResult
  matchAndRewrite(triton::gcu::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, storeOp.getOperation());
    auto loc = storeOp.getLoc();

    // Matrix store
    Value storeValue = adaptor.getValue();
    if (useMatrixStore(storeOp, storeValue)) {
      bool hasTrans = needsTransposeForStore(storeOp);
      // hasTrans = true;
      Value dstPtr = adaptor.getPtr();
      auto dstPtrElemTy = storeOp.getPtr().getType().getElementType();
      if (hasTrans) {
        // Copy accumulator to a local buffer, then DMA to global
        auto lastUser = userAnalysis.getLastUser(storeOp.getValue());
        auto storeInMemRefType = dyn_cast<MemRefType>(storeValue.getType());
        auto dotOutMemRefType =
            MemRefType::get(storeInMemRefType.getShape(), dstPtrElemTy);
        auto dotOut = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                  replaced2Origin, dotOutMemRefType);
        auto dotOutPtrType = gcu::PtrType::get(
            rewriter.getContext(), dotOutMemRefType.getElementType());
        dstPtr =
            rewriter.create<gcu::MemRefToPtrOp>(loc, dotOutPtrType, dotOut);
        storeValue = dotOut;
      }

      ConfigMatrixStore(rewriter, loc, storeOp, adaptor.getValue(), dstPtr,
                        adaptor.getShape(), adaptor.getStrides(),
                        adaptor.getOffsets(), hasTrans);
      if (!hasTrans) {
        leaveTritionOp(rewriter, storeOp.getOperation());
        rewriter.eraseOp(storeOp);
        return success();
      }
    }

    // DMA only supports same element type transfers
    auto valElemTy =
        dyn_cast<MemRefType>(storeValue.getType()).getElementType();
    auto ptrElemTy = storeOp.getPtr().getType().getElementType();
    if (valElemTy != ptrElemTy) {
      return storeOp.emitOpError()
             << "DMA store requires matching element types, got value="
             << valElemTy << " ptr=" << ptrElemTy;
    }

    // Dma store
    if (pTagPool.isExistInMap(storeOp.getOperation())) {
      pTagPool.releaseMap(storeOp.getOperation());
    }
    bool isLastOp = true;
    mlir::Operation *nextOp = storeOp.getOperation()->getNextNode();
    while (!nextOp->hasTrait<mlir::OpTrait::IsTerminator>()) {
      if (mlir::isa<triton::gcu::StoreOp>(nextOp) ||
          mlir::isa<triton::gcu::LoadOp>(nextOp) ||
          mlir::isa<triton::LoadOp>(nextOp) ||
          mlir::isa<triton::StoreOp>(nextOp)) {
        /// Transmit synchronously data
        /// Store ptrA Value1  *** Can't be a asynchronous
        /// Store ptrA Value2
        isLastOp = true;
        break;
      }
      if (!mlir::isa<memref::DeallocOp>(nextOp)) {
        isLastOp = false;
      }
      nextOp = nextOp->getNextNode();
    }

    auto storeType = storeOp.getValue().getType();
    if (!isa<TensorType>(storeType))
      return failure();

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto storeValueType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(storeType));

    auto tag = isLastOp ? pTagPool.getPrivateSyncTagInfo(storeOp)
                        : pTagPool.tryGetPrivateAsyncTagInfo(storeOp);
    if (tag.isAsync()) {
      auto &lastOp = storeOp.getOperation()->getBlock()->back();
      pTagPool.setMap(&lastOp, tag);
    }
    // workaround for offset > tensor dims
    int64_t rank = storeType.getRank();
    Value shapeCheck = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, adaptor.getShape()[0], zero);
    for (unsigned i = 1; i < rank; ++i) {
      auto dimCheck = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, adaptor.getShape()[i], zero);
      shapeCheck = rewriter.create<arith::AndIOp>(loc, shapeCheck, dimCheck);
    }
    auto total_size =
        rewriter
            .create<scf::IfOp>(
                loc, shapeCheck,
                [&](OpBuilder builder, Location loc) {
                  auto store_size = ConfigGcuStore(
                      rewriter, loc, storeValue, storeOp, storeValueType,
                      adaptor.getPtr(), adaptor.getStrides(),
                      adaptor.getShape(), tag);
                  builder.create<scf::YieldOp>(loc, ValueRange{store_size});
                },
                [&](OpBuilder &builder, Location loc) {
                  if (triton::gcu::get_bool_env("TRITON_GCU_DEBUG")) {
                    std::string locStr = "[warning]: store offset is out of "
                                         "range for tensor. loc:";
                    if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(loc)) {
                      llvm::StringRef filename = fileLineColLoc.getFilename();
                      locStr += filename.str();
                      locStr += ":";
                      locStr += std::to_string(fileLineColLoc.getLine());
                    }
                    builder.create<gpu::PrintfOp>(loc, locStr, ValueRange{});
                  }
                  builder.create<scf::YieldOp>(loc, ValueRange{zero});
                })
            .getResult(0);
    auto isNotZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, total_size, zero);

    if (tag.isAsync()) {
      auto &lastOp = storeOp.getOperation()->getBlock()->back();
      auto ip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPoint(&lastOp);
      auto ifOp = rewriter.create<scf::IfOp>(
          loc, isNotZero, [&](OpBuilder builder, Location loc) {
            WaitGcuLoadStore(builder, loc, tag, total_size);
            builder.create<scf::YieldOp>(loc);
          });
      rewriter.restoreInsertionPoint(ip);
      moveDeallocOp(rewriter, storeValue, ifOp, 0);
    } else {
      rewriter.create<scf::IfOp>(
          loc, isNotZero, [&](OpBuilder builder, Location loc) {
            WaitGcuLoadStore(builder, loc, tag, total_size);
            builder.create<scf::YieldOp>(loc);
          });
    }

    leaveTritionOp(rewriter, storeOp.getOperation());
    rewriter.eraseOp(storeOp);
    return success();
  }
};

struct TTGAssertOpLowering : SharedConversionPattern<triton::gcu::AssertOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcu::AssertOp assertOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (pTagPool.isExistInMap(assertOp.getOperation())) {
      pTagPool.releaseMap(assertOp.getOperation());
    }
    auto loc = assertOp.getLoc();

    auto condition = adaptor.getCondition();
    auto message = assertOp.getMessage();
    auto file = assertOp.getFile();
    auto func = assertOp.getFunc();
    auto line = assertOp.getLine();

    // Create gcu.assert op
    rewriter.create<gcu::AssertOp>(loc, condition, message, file, func, line);
    rewriter.eraseOp(assertOp);
    return success();
  }
};

struct TTBroadcastOpLowering : SharedConversionPattern<triton::BroadcastOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto srcType = op.getSrc().getType();
    auto resultType = op.getType();
    auto rank = srcType.getRank();
    auto wrapSrcType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(srcType));
    auto wrapResultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(resultType));
    auto elementType = wrapResultType.getElementType();

    auto loc = op.getLoc();
    auto tag = pTagPool.getPrivateSyncTagInfo(op);
    auto lastUser =
        userAnalysis.getLastUser(op.getOperation()->getResults()[0]);

    auto srcTy = dyn_cast<RankedTensorType>(srcType);
    auto dstTy = dyn_cast<RankedTensorType>(resultType);
    if ((!srcTy) || (!dstTy)) {
      assert(false && "srcTy or dstTy not a RankedTensorType");
    }
    auto srcLayout = srcTy.getEncoding();
    auto dstLayout = dstTy.getEncoding();

    DenseSet<unsigned> broadcastedAxies;
    for (unsigned i = 0; i < rank; ++i) {
      if (srcType.getDimSize(i) != resultType.getDimSize(i)) {
        if (wrapSrcType.getShape()[i] != wrapResultType.getShape()[i]) {
          broadcastedAxies.insert(i);
        }
      }
    }
    // broadcast per thread
    if (srcLayout == dstLayout) {
      auto broadcastedAxiesNum = broadcastedAxies.size();
      if (broadcastedAxiesNum == 0) {
        leaveTritionOp(rewriter, op.getOperation());
        rewriter.replaceOp(op, adaptor.getSrc());
        return success();
      }
      ArrayRef<int64_t> srcShape = wrapSrcType.getShape();
      auto src_input = adaptor.getSrc();
      auto output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                replaced2Origin, wrapResultType);
      SmallVector<int64_t> broadcastShape(rank, 1);
      for (unsigned i = 0; i < rank; ++i)
        broadcastShape[i] = srcShape[i];
      unsigned idx = 0;
      for (auto dim : broadcastedAxies) {
        auto temp_out = output;
        if (idx != broadcastedAxiesNum - 1) {
          broadcastShape[dim] = wrapResultType.getDimSize(dim);
          auto memrefType = MemRefType::get(broadcastShape, elementType);
          temp_out =
              syncAllocOp(rewriter, loc, std::make_pair(op.getOperation(), -1),
                          userAnalysis, replaced2Origin, memrefType);
        }

        auto src = src_input;
        auto dst = temp_out;
        if (rank > 3) { // reshape to rank 3 to broadcast
          ArrayRef<int64_t> beforeSrcShapes =
              dyn_cast<MemRefType>(src_input.getType()).getShape();
          ArrayRef<int64_t> beforeDstShapes =
              dyn_cast<MemRefType>(temp_out.getType()).getShape();
          SmallVector<int64_t> afterSrcShapes;
          SmallVector<int64_t> afterDstShapes;
          if (dim > 0) {
            int64_t tShape = std::accumulate(beforeSrcShapes.begin(),
                                             beforeSrcShapes.begin() + dim, 1,
                                             std::multiplies<int64_t>());
            afterSrcShapes.push_back(tShape);
          }
          afterSrcShapes.push_back(beforeSrcShapes[dim]);
          int64_t tShape = std::accumulate(beforeSrcShapes.begin() + dim + 1,
                                           beforeSrcShapes.end(), 1,
                                           std::multiplies<int64_t>());
          afterSrcShapes.push_back(tShape);
          if (dim > 0) {
            int64_t tShape = std::accumulate(beforeDstShapes.begin(),
                                             beforeDstShapes.begin() + dim, 1,
                                             std::multiplies<int64_t>());
            afterDstShapes.push_back(tShape);
          }
          afterDstShapes.push_back(beforeDstShapes[dim]);
          tShape = std::accumulate(beforeDstShapes.begin() + dim + 1,
                                   beforeDstShapes.end(), 1,
                                   std::multiplies<int64_t>());
          afterDstShapes.push_back(tShape);

          auto afterSrcMemrefType =
              MemRefType::get(afterSrcShapes, elementType);
          auto afterDstMemrefType =
              MemRefType::get(afterDstShapes, elementType);

          auto [srcStrides, srcOffset] =
              afterSrcMemrefType.getStridesAndOffset();
          src = rewriter.create<memref::ReinterpretCastOp>(
              loc, afterSrcMemrefType, src_input, srcOffset, afterSrcShapes,
              srcStrides);
          auto [dstStrides, dstOffset] =
              afterDstMemrefType.getStridesAndOffset();
          dst = rewriter.create<memref::ReinterpretCastOp>(
              loc, afterDstMemrefType, temp_out, dstOffset, afterDstShapes,
              dstStrides);
        }
        auto totalNumElems = triton::gcu::getTotalElemsPerThread(srcType);
        rewriter.create<memref_ext::BroadcastStartOp>(
            loc, dst, src, tag.getTag(), ValueRange{tag.getIdx()});
        rewriter.create<memref::DmaWaitOp>(
            loc, tag.getTag(), ValueRange{tag.getIdx()},
            rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));

        src_input = temp_out;
        idx++;
      }
      leaveTritionOp(rewriter, op.getOperation());
      rewriter.replaceOp(op, output);
      return success();
    }
    // move source to shared memory
    auto sharedSrc = storeToSharedMem(
        rewriter, tag, srcType, adaptor.getSrc(), false,
        std::make_pair(op.getOperation(), -1), userAnalysis, replaced2Origin);
    auto mergedResultType =
        MemRefType::get(resultType.getShape(), elementType, AffineMap{},
                        rewriter.getI64IntegerAttr(2) /*shared memory*/);
    auto mergedOutput =
        syncAllocOp(rewriter, loc, std::make_pair(op.getOperation(), -1),
                    userAnalysis, replaced2Origin, mergedResultType);
    auto totalNumElems = triton::gcu::getTotalElemsPerThread(srcType);
    // broadcast in thread 0
    auto masterWarpId = getMasterThreadId(op.getOperation());
    auto isMasterThread = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
        rewriter.create<arith::ConstantIndexOp>(loc, masterWarpId));
    ArrayRef<int64_t> srcShape = srcType.getShape();
    auto src_input = sharedSrc;

    SmallVector<int64_t> broadcastShape(rank, 1);
    for (unsigned i = 0; i < rank; ++i)
      broadcastShape[i] = srcShape[i];

    unsigned idx = 0;
    for (auto dim : broadcastedAxies) {
      auto temp_out = mergedOutput;
      if (idx != broadcastedAxies.size() - 1) {
        broadcastShape[dim] = resultType.getDimSize(dim);
        auto tempMemrefType =
            MemRefType::get(broadcastShape, elementType, AffineMap{},
                            rewriter.getI64IntegerAttr(2) /*shared memory*/);
        temp_out =
            syncAllocOp(rewriter, loc, std::make_pair(op.getOperation(), -1),
                        userAnalysis, replaced2Origin, tempMemrefType);
      }

      auto src = src_input;
      auto dst = temp_out;
      if (rank > 3) { // reshape to rank 3 to broadcast
        ArrayRef<int64_t> beforeSrcShapes =
            dyn_cast<MemRefType>(src_input.getType()).getShape();
        ArrayRef<int64_t> beforeDstShapes =
            dyn_cast<MemRefType>(temp_out.getType()).getShape();
        SmallVector<int64_t> afterSrcShapes;
        SmallVector<int64_t> afterDstShapes;

        int64_t tShape = std::accumulate(beforeSrcShapes.begin(),
                                         beforeSrcShapes.begin() + dim, 1,
                                         std::multiplies<int64_t>());
        afterSrcShapes.push_back(tShape);
        afterSrcShapes.push_back(beforeSrcShapes[dim]);
        tShape = std::accumulate(beforeSrcShapes.begin() + dim + 1,
                                 beforeSrcShapes.end(), 1,
                                 std::multiplies<int64_t>());
        afterSrcShapes.push_back(tShape);

        tShape = std::accumulate(beforeDstShapes.begin(),
                                 beforeDstShapes.begin() + dim, 1,
                                 std::multiplies<int64_t>());
        afterDstShapes.push_back(tShape);
        afterDstShapes.push_back(beforeDstShapes[dim]);
        tShape = std::accumulate(beforeDstShapes.begin() + dim + 1,
                                 beforeDstShapes.end(), 1,
                                 std::multiplies<int64_t>());
        afterDstShapes.push_back(tShape);

        auto afterSrcMemrefType =
            MemRefType::get(afterSrcShapes, elementType, AffineMap{},
                            rewriter.getI64IntegerAttr(2) /*shared memory*/);
        auto afterDstMemrefType =
            MemRefType::get(afterDstShapes, elementType, AffineMap{},
                            rewriter.getI64IntegerAttr(2) /*shared memory*/);

        auto [srcStrides, srcOffset] = afterSrcMemrefType.getStridesAndOffset();
        src = rewriter.create<memref::ReinterpretCastOp>(
            loc, afterSrcMemrefType, src_input, srcOffset, afterSrcShapes,
            srcStrides);
        auto [dstStrides, dstOffset] = afterDstMemrefType.getStridesAndOffset();
        dst = rewriter.create<memref::ReinterpretCastOp>(
            loc, afterDstMemrefType, temp_out, dstOffset, afterDstShapes,
            dstStrides);
      }

      rewriter.create<scf::IfOp>(
          loc, isMasterThread, [&](OpBuilder &rewriter, Location loc) {
            rewriter.create<memref_ext::BroadcastStartOp>(
                loc, dst, src, tag.getTag(), ValueRange{tag.getIdx()});
            rewriter.create<memref::DmaWaitOp>(
                loc, tag.getTag(), ValueRange{tag.getIdx()},
                rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));
            rewriter.create<scf::YieldOp>(loc);
          });
      src_input = temp_out;
      idx++;
    }
    rewriter.create<gpu::BarrierOp>(loc);
    // read back
    auto output = loadFromSharedMem(
        rewriter, tag, resultType, mergedOutput, false, lastUser,
        std::make_pair(nullptr, -1), userAnalysis, replaced2Origin);
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct TTExpandDimsOpLowering : SharedConversionPattern<triton::ExpandDimsOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    auto resultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    auto srcNumElems = triton::gcu::getElemsPerThread(op.getSrc().getType());
    auto dstNumElems = triton::gcu::getElemsPerThread(op.getType());

    srcNumElems.insert(srcNumElems.begin() + op.getAxis(), 1);

    // noop expand dims
    if (srcNumElems == dstNumElems) {
      auto [strides, offset] = resultType.getStridesAndOffset();
      auto output = rewriter.create<memref::ReinterpretCastOp>(
          loc, resultType, adaptor.getSrc(), offset, resultType.getShape(),
          strides);
      leaveTritionOp(rewriter, op.getOperation());
      rewriter.replaceOp(op, output);
      return success();
    }
    auto type = op.getType();
    auto lastUser =
        userAnalysis.getLastUser(op.getOperation()->getResults()[0]);
    auto tag = pTagPool.getPrivateSyncTagInfo(op);
    auto srcType = dyn_cast<TensorType>(op.getSrc().getType());
    auto resMemType =
        MemRefType::get(type.getShape(), resultType.getElementType(),
                        AffineMap{}, rewriter.getI64IntegerAttr(2));
    // move source to shared memory
    auto sharedSrc = storeToSharedMem(
        rewriter, tag, srcType, adaptor.getSrc(), false,
        std::make_pair(op.getOperation(), -1), userAnalysis, replaced2Origin);
    auto [strides, offset] = resMemType.getStridesAndOffset();
    auto result = rewriter.create<memref::ReinterpretCastOp>(
        loc, resMemType, sharedSrc, offset, type.getShape(), strides);
    // copy back outputs
    Value output = loadFromSharedMem(rewriter, tag, op.getType(), result, false,
                                     lastUser, std::make_pair(nullptr, -1),
                                     userAnalysis, replaced2Origin);
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct TTReshapeOpLowering : SharedConversionPattern<triton::ReshapeOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    auto resultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));

    if (!triton::gcu::isExpensiveView(op.getSrc().getType(), op.getType())) {
      auto [strides, offset] = resultType.getStridesAndOffset();
      auto output = rewriter.create<memref::ReinterpretCastOp>(
          loc, resultType, adaptor.getSrc(), offset, resultType.getShape(),
          strides);
      leaveTritionOp(rewriter, op.getOperation());
      rewriter.replaceOp(op, output);
      return success();
    }

    auto type = op.getType();

    auto tag = pTagPool.getPrivateSyncTagInfo(op);
    auto srcType = dyn_cast<TensorType>(op.getSrc().getType());
    // move source to shared memory
    auto lastUser =
        userAnalysis.getLastUser(op.getOperation()->getResults()[0]);
    auto sharedSrc = storeToSharedMem(
        rewriter, tag, srcType, adaptor.getSrc(), false,
        std::make_pair(op.getOperation(), -1), userAnalysis, replaced2Origin);
    auto resMemType =
        MemRefType::get(type.getShape(), resultType.getElementType(),
                        AffineMap{}, rewriter.getI64IntegerAttr(2));
    auto [strides, offset] = resMemType.getStridesAndOffset();
    auto result = rewriter.create<memref::ReinterpretCastOp>(
        loc, resMemType, sharedSrc, offset, type.getShape(), strides);
    // copy back outputs
    Value output = loadFromSharedMem(rewriter, tag, op.getType(), result, false,
                                     lastUser, std::make_pair(nullptr, -1),
                                     userAnalysis, replaced2Origin);
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct TTSplitOpLowering : SharedConversionPattern<triton::SplitOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    auto srcType = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto srcShape = srcType.getShape();
    auto srcRank = srcType.getRank();
    if (srcRank <= 0)
      return op.emitOpError("the rank must be greater than 0.");
    if (srcShape[srcRank - 1] != 2)
      return op.emitOpError("the last dim must have size 2.");

    auto outType = dyn_cast<RankedTensorType>(op.getOutLHS().getType());
    auto outMemrefType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(outType));

    auto lastUser =
        userAnalysis.getLastUser(op.getOperation()->getResults()[0]);
    auto lhs = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                           replaced2Origin, outMemrefType);
    auto rhs = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                           replaced2Origin, outMemrefType);

    auto outMemrefShape = outMemrefType.getShape();
    SmallVector<int64_t> sliceShape(outMemrefShape.size() + 1, 1);
    for (long unsigned int i = 0; i < outMemrefShape.size(); i++) {
      sliceShape[i] = outMemrefShape[i];
    }
    SmallVector<int64_t> sliceStride(sliceShape.size(), 1);
    for (int i = sliceShape.size() - 2; i >= 0; --i) {
      sliceStride[i] = sliceStride[i + 1] * sliceShape[i + 1];
    }

    auto sliceType =
        MemRefType::get(sliceShape, outMemrefType.getElementType());

    auto sliceLHS = rewriter.create<memref::ReinterpretCastOp>(
        loc, sliceType, lhs, 0, sliceShape, sliceStride);
    auto sliceRHS = rewriter.create<memref::ReinterpretCastOp>(
        loc, sliceType, rhs, 0, sliceShape, sliceStride);

    auto tag = pTagPool.getPrivateSyncTagInfo(op);

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    SmallVector<Value, 4> offsets;
    for (int i = 0; i < outType.getRank(); ++i) {
      offsets.push_back(rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), zero));
    }
    SmallVector<Value, 4> offsetsLHS = offsets;
    SmallVector<Value, 4> offsetsRHS = offsets;
    offsetsLHS.push_back(
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), zero));
    offsetsRHS.push_back(
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), one));

    auto totalNumElems = triton::gcu::getTotalElemsPerThread(outType);
    auto defaultValue = triton::gcu::createConstantZero(
        rewriter, loc, outMemrefType.getElementType());

    rewriter.create<memref_ext::SliceStartOp>(
        loc, sliceLHS, adaptor.getSrc(), offsetsLHS, defaultValue, tag.getTag(),
        ValueRange{tag.getIdx()});
    rewriter.create<memref::DmaWaitOp>(
        loc, tag.getTag(), ValueRange{tag.getIdx()},
        rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));

    rewriter.create<memref_ext::SliceStartOp>(
        loc, sliceRHS, adaptor.getSrc(), offsetsRHS, defaultValue, tag.getTag(),
        ValueRange{tag.getIdx()});
    rewriter.create<memref::DmaWaitOp>(
        loc, tag.getTag(), ValueRange{tag.getIdx()},
        rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));

    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, {lhs, rhs});
    return success();
  }
};

struct TTJoinOpLowering : SharedConversionPattern<triton::JoinOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();

    auto lhsType = dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = dyn_cast<RankedTensorType>(op.getRhs().getType());
    if (lhsType != rhsType)
      return op.emitOpError("the lhs and rhs type must be the same.");

    auto lhsMemrefType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(lhsType));

    auto outType = dyn_cast<RankedTensorType>(op.getResult().getType());
    auto outMemrefType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(outType));

    auto lastUser =
        userAnalysis.getLastUser(op.getOperation()->getResults()[0]);
    auto result = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                              replaced2Origin, outMemrefType);

    auto lhsShape = lhsMemrefType.getShape();
    SmallVector<int64_t> desliceShape(lhsShape.size() + 1, 1);
    for (size_t i = 0; i < lhsShape.size(); i++) {
      desliceShape[i] = lhsShape[i];
    }
    SmallVector<int64_t> desliceStride(desliceShape.size(), 1);
    for (int i = desliceShape.size() - 2; i >= 0; --i) {
      desliceStride[i] = desliceStride[i + 1] * desliceShape[i + 1];
    }

    auto desliceType =
        MemRefType::get(desliceShape, lhsMemrefType.getElementType());
    auto desliceLHS = rewriter.create<memref::ReinterpretCastOp>(
        loc, desliceType, adaptor.getLhs(), 0, desliceShape, desliceStride);
    auto desliceRHS = rewriter.create<memref::ReinterpretCastOp>(
        loc, desliceType, adaptor.getRhs(), 0, desliceShape, desliceStride);

    auto tag = pTagPool.getPrivateSyncTagInfo(op);

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    SmallVector<Value, 4> offsets;
    for (int i = 0; i < lhsType.getRank(); ++i) {
      offsets.push_back(rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), zero));
    }
    SmallVector<Value, 4> offsetsLHS = offsets;
    SmallVector<Value, 4> offsetsRHS = offsets;
    offsetsLHS.push_back(
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), zero));
    offsetsRHS.push_back(
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), one));

    auto totalNumElems = triton::gcu::getTotalElemsPerThread(lhsType);

    rewriter.create<memref_ext::DesliceStartOp>(loc, result, desliceLHS,
                                                offsetsLHS, tag.getTag(),
                                                ValueRange{tag.getIdx()});
    rewriter.create<memref::DmaWaitOp>(
        loc, tag.getTag(), ValueRange{tag.getIdx()},
        rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));

    rewriter.create<memref_ext::DesliceStartOp>(loc, result, desliceRHS,
                                                offsetsRHS, tag.getTag(),
                                                ValueRange{tag.getIdx()});
    rewriter.create<memref::DmaWaitOp>(
        loc, tag.getTag(), ValueRange{tag.getIdx()},
        rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));

    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, {result});
    return success();
  }
};

struct TTCatOpLowering : SharedConversionPattern<triton::CatOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto type = op.getType();
    auto loc = op.getLoc();
    auto resultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(type));

    auto tag = pTagPool.getPrivateSyncTagInfo(op);
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto lastUser =
        userAnalysis.getLastUser(op.getOperation()->getResults()[0]);
    auto lhsSlicedAxies = getSlicedAxies(op.getLhs().getType());
    auto rhsSlicedAxies = getSlicedAxies(op.getRhs().getType());
    auto outputSlicedAxies = getSlicedAxies(op.getType());
    if (!lhsSlicedAxies.count(0) && !rhsSlicedAxies.count(0) &&
        !outputSlicedAxies.count(0)) {
      auto totalNumElems = triton::gcu::getTotalElemsPerThread(type);

      auto output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                replaced2Origin, resultType);
      SmallVector<Value, 4> offsets;
      for (unsigned i = 0; i < resultType.getRank(); ++i) {
        offsets.push_back(rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getI32Type(), zero));
      }
      rewriter.create<memref_ext::DesliceStartOp>(loc, output, adaptor.getLhs(),
                                                  offsets, tag.getTag(),
                                                  ValueRange{tag.getIdx()});
      rewriter.create<memref::DmaWaitOp>(
          loc, tag.getTag(), ValueRange{tag.getIdx()},
          rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));

      offsets[0] = rewriter.create<arith::ConstantIntOp>(
          loc, dyn_cast<MemRefType>(adaptor.getLhs().getType()).getDimSize(0),
          32);
      rewriter.create<memref_ext::DesliceStartOp>(loc, output, adaptor.getRhs(),
                                                  offsets, tag.getTag(),
                                                  ValueRange{tag.getIdx()});
      rewriter.create<memref::DmaWaitOp>(
          loc, tag.getTag(), ValueRange{tag.getIdx()},
          rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));
      rewriter.replaceOp(op, output);
      return success();
    }
    auto mergedResultType =
        MemRefType::get(type.getShape(), type.getElementType(), AffineMap{},
                        rewriter.getI64IntegerAttr(2) /*shared memory*/);
    auto mergedOutput =
        syncAllocOp(rewriter, loc, std::make_pair(op.getOperation(), -1),
                    userAnalysis, replaced2Origin, mergedResultType);
    auto lhsTy = op.getLhs().getType();
    auto [lhsStrides, lhsOffset] =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(lhsTy))
            .getStridesAndOffset();
    storeToSharedMem(
        rewriter, tag, op.getLhs().getType(),
        rewriter.create<memref::ReinterpretCastOp>(
            loc,
            MemRefType::get(lhsTy.getShape(), lhsTy.getElementType(),
                            AffineMap{}, rewriter.getI64IntegerAttr(2)),
            mergedOutput, 0, lhsTy.getShape(), lhsStrides),
        adaptor.getLhs(), false);
    (void)lhsOffset;

    auto rhsTy = op.getRhs().getType();
    auto [rhsStrides, rhsOffset] =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(rhsTy))
            .getStridesAndOffset();
    storeToSharedMem(
        rewriter, tag, op.getRhs().getType(),
        rewriter.create<memref::ReinterpretCastOp>(
            loc,
            MemRefType::get(rhsTy.getShape(), rhsTy.getElementType(),
                            makeStridedLinearLayoutMap(rhsStrides,
                                                       rhsTy.getNumElements(),
                                                       rewriter.getContext()),
                            rewriter.getI64IntegerAttr(2)),
            mergedOutput, rhsTy.getNumElements(), rhsTy.getShape(), rhsStrides),
        adaptor.getRhs(), false);
    (void)rhsOffset;
    // read back
    auto output = loadFromSharedMem(
        rewriter, tag, op.getType(), mergedOutput, false, lastUser,
        std::make_pair(nullptr, -1), userAnalysis, replaced2Origin);
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct TTTransOpLowering : SharedConversionPattern<triton::TransOp> {
  using SharedConversionPattern::SharedConversionPattern;

  void applyTranspose(OpBuilder &rewriter, Location loc, Value src,
                      Value output, triton::gcu::TagInfo tag,
                      ArrayRef<int32_t> order, unsigned totalSize) const {
    auto totalNumElems =
        rewriter.create<arith::ConstantIndexOp>(loc, totalSize);

    SmallVector<Value, 4> layout;
    for (auto i : order) {
      layout.push_back(rewriter.create<arith::ConstantIntOp>(loc, i, 32));
    }
    rewriter.create<memref_ext::TransposeStartOp>(
        loc, output, src, layout, tag.getTag(), ValueRange{tag.getIdx()});
    rewriter.create<memref::DmaWaitOp>(loc, tag.getTag(),
                                       ValueRange{tag.getIdx()}, totalNumElems);
  }

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = dyn_cast<RankedTensorType>(op.getType());
    if ((!srcTy) || (!dstTy)) {
      assert(false && "srcTy or dstTy not a RankedTensorType");
    }
    auto srcLayout = srcTy.getEncoding();
    auto dstLayout = dstTy.getEncoding();
    auto resultType = dyn_cast<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    auto totalNumElems =
        triton::gcu::getTotalElemsPerThread(op.getSrc().getType());
    auto lastUser =
        userAnalysis.getLastUser(op.getOperation()->getResults()[0]);
    // gcu400 only one private dte
    if (mlir::isa<triton::gpu::SharedEncodingTrait>(srcLayout) &&
        mlir::isa<triton::gpu::SharedEncodingTrait>(dstLayout)) {
      // allocate output buffers in shared memory
      auto firstUser =
          userAnalysis.getFirstUser(op.getOperation()->getResults()[0]);

      triton::gcu::TagInfo tag;
      if (firstUser.first != nullptr) {
        tag = pTagPool.tryGetPrivateAsyncTagInfo(op);
      } else {
        tag = pTagPool.getPrivateSyncTagInfo(op);
      }
      if (tag.isAsync()) {
        pTagPool.setMap(firstUser.first, tag);
      }

      auto sharedOutputType = MemRefType::get(
          op.getResult().getType().getShape(), resultType.getElementType(),
          AffineMap{}, rewriter.getI64IntegerAttr(2));
      auto sharedOutput = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                      replaced2Origin, sharedOutputType);
      // split by thread 0
      auto totalNumElemsValue =
          rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems);

      SmallVector<Value, 4> layout;
      for (auto i : op.getOrder()) {
        layout.push_back(rewriter.create<arith::ConstantIntOp>(loc, i, 32));
      }
      auto masterWarpId = getMasterThreadId(op.getOperation());
      auto isMasterThread = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
          rewriter.create<arith::ConstantIndexOp>(loc, masterWarpId));
      rewriter.create<scf::IfOp>(
          loc, isMasterThread, [&](OpBuilder &builder, Location loc) {
            rewriter.create<memref_ext::TransposeStartOp>(
                loc, sharedOutput, adaptor.getSrc(), layout, tag.getTag(),
                ValueRange{tag.getIdx()});
            builder.create<scf::YieldOp>(loc);
          });
      if (tag.isAsync()) {
        auto ip = rewriter.saveInsertionPoint();
        rewriter.setInsertionPoint(firstUser.first);
        rewriter.create<scf::IfOp>(
            loc, isMasterThread, [&](OpBuilder &builder, Location loc) {
              builder.create<memref::DmaWaitOp>(loc, tag.getTag(),
                                                ValueRange{tag.getIdx()},
                                                totalNumElemsValue);
              builder.create<scf::YieldOp>(loc);
            });
        rewriter.create<gpu::BarrierOp>(loc);
        rewriter.restoreInsertionPoint(ip);
      } else {
        rewriter.create<scf::IfOp>(
            loc, isMasterThread, [&](OpBuilder &builder, Location loc) {
              builder.create<memref::DmaWaitOp>(loc, tag.getTag(),
                                                ValueRange{tag.getIdx()},
                                                totalNumElemsValue);
              builder.create<scf::YieldOp>(loc);
            });
        rewriter.create<gpu::BarrierOp>(loc);
      }
      leaveTritionOp(rewriter, op.getOperation());
      rewriter.replaceOp(op, sharedOutput);
      return success();
    } else if ((isa<triton::gpu::BlockedEncodingAttr>(srcLayout) &&
                isa<triton::gpu::BlockedEncodingAttr>(dstLayout)) ||
               (isa<triton::gpu::SliceEncodingAttr>(srcLayout) &&
                isa<triton::gpu::LinearEncodingAttr>(dstLayout)) ||
               (isa<triton::gpu::LinearEncodingAttr>(srcLayout) &&
                isa<triton::gpu::LinearEncodingAttr>(dstLayout))) {
      // move source to shared memory
      auto tag = pTagPool.getPrivateSyncTagInfo(op);
      auto lastUser =
          userAnalysis.getLastUser(op.getOperation()->getResults()[0]);
      auto sharedSrc = storeToSharedMem(
          rewriter, tag, dyn_cast<TensorType>(op.getSrc().getType()),
          adaptor.getSrc(), false, std::make_pair(op.getOperation(), -1),
          userAnalysis, replaced2Origin);

      // allocate output buffers in shared memory
      auto sharedOutputType = MemRefType::get(
          op.getResult().getType().getShape(), resultType.getElementType(),
          AffineMap{}, rewriter.getI64IntegerAttr(2));
      auto sharedOutput =
          syncAllocOp(rewriter, loc, std::make_pair(op.getOperation(), -1),
                      userAnalysis, replaced2Origin, sharedOutputType);

      // split by thread 0
      auto masterWarpId = getMasterThreadId(op.getOperation());
      auto isMasterThread = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
          rewriter.create<arith::ConstantIndexOp>(loc, masterWarpId));
      rewriter.create<scf::IfOp>(
          loc, isMasterThread, [&](OpBuilder &builder, Location loc) {
            applyTranspose(builder, loc, sharedSrc, sharedOutput, tag,
                           op.getOrder(), totalNumElems);
            builder.create<scf::YieldOp>(loc);
          });
      rewriter.create<gpu::BarrierOp>(loc);
      // copy back outputs
      Value output = loadFromSharedMem(
          rewriter, tag, op.getResult().getType(), sharedOutput, false,
          lastUser, std::make_pair(nullptr, -1), userAnalysis, replaced2Origin);
      leaveTritionOp(rewriter, op.getOperation());
      rewriter.replaceOp(op, output);
      return success();
    } else {
      op.dump();
      assert(false && "please check layout of this transop \n");
      return failure();
    }
  }
};

struct TTGConvertLayoutOpLowering
    : SharedConversionPattern<triton::gpu::ConvertLayoutOp> {
  using SharedConversionPattern::SharedConversionPattern;

  Value loadFromSharedMemForDotOperand(OpBuilder builder, Type type,
                                       Value sharedBuffer) const {
    auto loc = sharedBuffer.getLoc();
    auto srcType = dyn_cast<MemRefType>(sharedBuffer.getType());
    auto numElems = triton::gcu::getElemsPerThread(type);
    auto warpIds = getWarpIds(builder, loc, type);
    unsigned rank = srcType.getRank();

    SmallVector<OpFoldResult> offsets, sizes, strides;
    for (unsigned i = 0; i < rank; ++i) {
      Value offset = builder.create<arith::MulIOp>(
          loc, builder.create<arith::ConstantIndexOp>(loc, numElems[i]),
          warpIds[i]);
      offsets.push_back(offset);
      sizes.push_back(builder.getIndexAttr(numElems[i]));
      strides.push_back(builder.getIndexAttr(1));
    }

    auto resultType = getMemRefTypeFromSharedMem(cast<RankedTensorType>(type),
                                                 srcType.getElementType());
    auto subview = builder.create<memref::SubViewOp>(
        loc, resultType, sharedBuffer, offsets, sizes, strides);
    return subview.getResult();
  }

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto srcNumElems = triton::gcu::getElemsPerThread(op.getSrc().getType());
    auto dstNumElems = triton::gcu::getElemsPerThread(op.getType());
    // noop convert
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = dyn_cast<RankedTensorType>(op.getType());
    if ((!srcTy) || (!dstTy)) {
      assert(false && "srcTy or dstTy not a RankedTensorType");
    }
    auto srcLayout = srcTy.getEncoding();
    auto dstLayout = dstTy.getEncoding();
    if (srcLayout == dstLayout) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }
    auto lastUser =
        userAnalysis.getLastUser(op.getOperation()->getResults()[0]);
    auto firstUser =
        userAnalysis.getFirstUser(op.getOperation()->getResults()[0]);

    if (srcNumElems == dstNumElems &&
        op.getSrc().getType().getShape() == op.getType().getShape()) {
      if (isa<triton::gpu::SliceEncodingAttr>(srcLayout) &&
          isa<triton::gpu::SliceEncodingAttr>(dstLayout)) {
        if (cast<triton::gpu::SliceEncodingAttr>(srcLayout).getDim() ==
                cast<triton::gpu::SliceEncodingAttr>(dstLayout).getDim() &&
            triton::gcu::getWarpsPerCTA(srcLayout) ==
                triton::gcu::getWarpsPerCTA(dstLayout)) {
          rewriter.replaceOp(op, adaptor.getSrc());
          return success();
        }
      } else if (isa<triton::gpu::SharedEncodingTrait>(srcLayout) &&
                 (isa<triton::gpu::SliceEncodingAttr>(dstLayout) ||
                  isa<triton::gpu::BlockedEncodingAttr>(dstLayout) ||
                  isa<triton::gpu::DotOperandEncodingAttr>(dstLayout))) {
        auto srcMemRef = dyn_cast<MemRefType>(adaptor.getSrc().getType());
        auto noMemSpaceType =
            MemRefType::get(srcMemRef.getShape(), srcMemRef.getElementType(),
                            srcMemRef.getLayout());
        auto castOp = rewriter.create<memref::MemorySpaceCastOp>(
            op.getLoc(), noMemSpaceType, adaptor.getSrc());
        rewriter.replaceOp(op, castOp.getResult());
        return success();
      } else if (isa<triton::gpu::SliceEncodingAttr>(srcLayout) &&
                 isa<triton::gpu::BlockedEncodingAttr>(dstLayout)) {
        if (triton::gcu::getWarpsPerCTA(srcLayout) ==
            triton::gcu::getWarpsPerCTA(dstLayout)) {
          rewriter.replaceOp(op, adaptor.getSrc());
          return success();
        }
      } else if (isa<triton::gpu::BlockedEncodingAttr>(srcLayout) &&
                 isa<triton::gpu::SliceEncodingAttr>(dstLayout)) {
        if (triton::gcu::getWarpsPerCTA(srcLayout) ==
            triton::gcu::getWarpsPerCTA(dstLayout)) {
          rewriter.replaceOp(op, adaptor.getSrc());
          return success();
        }
      } else {
        rewriter.replaceOp(op, adaptor.getSrc());
        return success();
      }
    }

    triton::gcu::TagInfo tag;
    if (firstUser.first != nullptr) {
      tag = pTagPool.tryGetPrivateAsyncTagInfo(op.getOperation());
    } else {
      tag = pTagPool.getPrivateSyncTagInfo(op.getOperation());
    }
    if (tag.isAsync()) {
      pTagPool.setMap(firstUser.first, tag);
    }
    // share to Distributed
    if (isa<triton::gpu::SharedEncodingTrait>(srcLayout) &&
        isa<triton::gpu::BlockedEncodingAttr>(dstLayout)) {
      // copy to local
      auto output = loadFromSharedMem(rewriter, tag, op.getResult().getType(),
                                      adaptor.getSrc(), false, lastUser,
                                      firstUser, userAnalysis, replaced2Origin);
      leaveTritionOp(rewriter, op.getOperation());
      rewriter.replaceOp(op, output);
      return success();
    } else if (isa<triton::gpu::BlockedEncodingAttr>(srcLayout) &&
               isa<triton::gpu::DotOperandEncodingAttr>(dstLayout)) {
      // Distributed to dot operand via shared memory
      auto sharedSrc = storeToSharedMem(
          rewriter, tag, dyn_cast<TensorType>(op.getSrc().getType()),
          adaptor.getSrc(), false, lastUser, userAnalysis, replaced2Origin);
      auto output = loadFromSharedMemForDotOperand(
          rewriter, op.getResult().getType(), sharedSrc);
      leaveTritionOp(rewriter, op.getOperation());
      rewriter.replaceOp(op, output);
      return success();
    } else if (isa<triton::gpu::SharedEncodingTrait>(srcLayout) &&
               isa<triton::gpu::DotOperandEncodingAttr>(dstLayout)) {
      // Distributed to dot operand
      // to dot a or b
      auto output = loadFromSharedMemForDotOperand(
          rewriter, op.getResult().getType(), adaptor.getSrc());
      leaveTritionOp(rewriter, op.getOperation());
      rewriter.replaceOp(op, output);
      return success();
    } else {
      // move source to shared memory
      // Maybe async dte, so shared buffer lives until the lastUser
      auto sharedSrc = storeToSharedMem(
          rewriter, tag, dyn_cast<TensorType>(op.getSrc().getType()),
          adaptor.getSrc(), false, lastUser, userAnalysis, replaced2Origin);
      // copy back outputs
      auto output = loadFromSharedMem(rewriter, tag, op.getResult().getType(),
                                      sharedSrc, false, lastUser, firstUser,
                                      userAnalysis, replaced2Origin);
      leaveTritionOp(rewriter, op.getOperation());
      rewriter.replaceOp(op, output);
    }
    return success();
  }
};

struct GCUMatmulLowering : SharedConversionPattern<triton::gcu::MatmulOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcu::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    if (!isa<RankedTensorType>(op.getA().getType()) ||
        !isa<RankedTensorType>(op.getB().getType()))
      return failure();
    auto lastUser =
        userAnalysis.getLastUser(op.getOperation()->getResults()[0]);
    auto resultMemRefType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    auto output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                              replaced2Origin, resultMemRefType);
    if (op.getType().getRank() == 2) {
      rewriter.create<gcu::MatMulOp>(loc, output, adaptor.getA(),
                                     adaptor.getB(), Value(), Value());
    }
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, output);
    return success();
  }
};

static Value stripValueFromUnrealizedConversionCastOp(Value v) {
  if (auto ucc =
          dyn_cast_or_null<UnrealizedConversionCastOp>(v.getDefiningOp())) {
    if (ucc.getNumOperands() == 1 && ucc.getNumResults() == 1) {
      return ucc.getOperand(0);
    }
  }
  return v;
}

// Create a stack allocation for the OACC accumulator buffer.
// The hardware requires the physical buffer to be at least
// GEMM_MIN_M (32) rows and OACC_F32_LENGTH (128) columns.
// When the original shape is smaller, allocate a padded buffer and
// reinterpret_cast back to the original type.
static Value createOaccAlloca(OpBuilder &rewriter, memref::AllocOp allocOp) {
  auto origAllocType = cast<MemRefType>(allocOp.getType());
  auto origShape = origAllocType.getShape();
  bool needsPadM = origAllocType.getRank() >= 2 &&
                   origShape[origShape.size() - 2] < GEMM_MIN_M;
  bool needsPadN =
      origAllocType.getRank() >= 2 && origShape.back() < OACC_F32_LENGTH;
  bool needsOaccPad = needsPadM || needsPadN;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(allocOp);
  if (needsOaccPad) {
    SmallVector<int64_t> paddedShape(origShape);
    if (needsPadM)
      paddedShape[paddedShape.size() - 2] = GEMM_MIN_M;
    if (needsPadN)
      paddedShape.back() = OACC_F32_LENGTH;
    auto paddedType =
        MemRefType::get(paddedShape, origAllocType.getElementType());
    auto allocaOp =
        rewriter.create<memref::AllocaOp>(allocOp.getLoc(), paddedType);
    allocaOp.setAlignment(512);

    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
    for (auto s : origShape)
      sizes.push_back(rewriter.getIndexAttr(s));
    int64_t stride = 1;
    SmallVector<int64_t> strideVals(origShape.size());
    for (int i = origShape.size() - 1; i >= 0; --i) {
      strideVals[i] = stride;
      stride *= origShape[i];
    }
    for (auto sv : strideVals)
      strides.push_back(rewriter.getIndexAttr(sv));
    return rewriter.create<memref::ReinterpretCastOp>(
        allocOp.getLoc(), origAllocType, allocaOp.getResult(),
        rewriter.getIndexAttr(0), sizes, strides);
  }
  auto allocaOp =
      rewriter.create<memref::AllocaOp>(allocOp.getLoc(), origAllocType);
  allocaOp.setAlignment(512);
  return allocaOp.getResult();
}

struct TTDotOpLowering : SharedConversionPattern<triton::DotOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    auto ctx = op.getContext();
    if (!isa<RankedTensorType>(op.getA().getType()) ||
        !isa<RankedTensorType>(op.getB().getType()))
      return failure();
    auto lastUser =
        userAnalysis.getLastUser(op.getOperation()->getResults()[0]);
    auto resultMemRefType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));

    Value output;
    Value useBiasVal;
    bool reuseAcc = op->hasAttr(kAccReuseCandidate);
    if (reuseAcc) {
      output = adaptor.getC();
      auto blockArg = cast<BlockArgument>(adaptor.getC());
      auto forOp = cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      Value iv = forOp.getInductionVar();
      Value lb = forOp.getLowerBound();
      useBiasVal =
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, iv, lb);

      unsigned iterArgIdx = blockArg.getArgNumber() - 1;
      Value initArg = forOp.getInitArgs()[iterArgIdx];
      auto allocOp = initArg.getDefiningOp<memref::AllocOp>();
      assert(allocOp && "allocOp is null");

      auto forOpOrig = cast<scf::ForOp>(replaced2Origin[forOp.getOperation()]);
      Value initArgOrig = forOpOrig.getInitArgs()[iterArgIdx];
      bool initArgHasOneUse = initArgOrig.hasOneUse();
      if (initArgHasOneUse)
        removeRedundantZeroFill(rewriter, allocOp);

      auto accReuseAttr = dyn_cast<StringAttr>(op->getAttr(kAccReuseCandidate));
      assert(accReuseAttr && "acc_reuse_candidate attr is not a StringAttr");
      if (accReuseAttr.getValue() == kAccReuseOacc) {
        Value allocaVal = createOaccAlloca(rewriter, allocOp);
        if (initArgHasOneUse) {
          for (auto *user :
               llvm::make_early_inc_range(allocOp.getResult().getUsers())) {
            if (isa<memref::DeallocOp>(user))
              rewriter.eraseOp(user);
          }
          rewriter.replaceOp(allocOp, allocaVal);
        } else {
          forOp.getInitArgsMutable()[iterArgIdx].set(allocaVal);
        }

        // Store to local memory
        if (auto accStore = op->getAttr(kAccStore)) {
          StringRef accStoreVal = mlir::cast<StringAttr>(accStore).getValue();
          if (accStoreVal == kAccStoreLocal ||
              accStoreVal == kAccStoreCvtLocal) {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointAfter(forOp);

            auto dotOutMemRefType = resultMemRefType;
            Operation *cvtOp = nullptr;
            if (accStoreVal == kAccStoreCvtLocal) {
              cvtOp = *op->getResult(0).getUsers().begin();
              if (isa<arith::TruncFOp, arith::TruncIOp>(cvtOp)) {
                dotOutMemRefType = MemRefType::get(
                    resultMemRefType.getShape(),
                    dyn_cast<MemRefType>(cvtOp->getResult(0).getType())
                        .getElementType());
              } else {
                assert(false && "only support truncf or trunci for dot cvt op");
              }
            }

            auto dotOut = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                      replaced2Origin, dotOutMemRefType);
            auto dotOutPtrType =
                gcu::PtrType::get(ctx, dotOutMemRefType.getElementType());
            auto dotOutPtr =
                rewriter.create<gcu::MemRefToPtrOp>(loc, dotOutPtrType, dotOut);
            SmallVector<Value, 2> memDims;
            auto dotOutShape = dotOutMemRefType.getShape();
            memDims.push_back(
                rewriter.create<arith::ConstantIndexOp>(loc, dotOutShape[0]));
            memDims.push_back(
                rewriter.create<arith::ConstantIndexOp>(loc, dotOutShape[1]));
            SmallVector<Value, 2> realDims;
            realDims.push_back(rewriter.create<memref::DimOp>(loc, dotOut, 0));
            realDims.push_back(rewriter.create<memref::DimOp>(loc, dotOut, 1));

            Value forResult = forOp->getResult(iterArgIdx);
            auto matStoreOp = rewriter.create<gcu::MatrixStoreOp>(
                loc, forResult, dotOutPtr, memDims, realDims);
            if (accStoreVal == kAccStoreCvtLocal) {
              rewriter.replaceOp(cvtOp, dotOut);
            } else {
              rewriter.replaceAllUsesExcept(forResult, dotOut, matStoreOp);
            }
          }
        }
      }
      LLVM_DEBUG(llvm::dbgs() << "TTDotOpLowering: reusing accumulator buffer "
                                 "for in-place matmul\n");
    } else {
      output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                           replaced2Origin, resultMemRefType);
    }

    auto aElemTy = op.getA().getType();
    auto bElemTy = op.getB().getType();
    if (op.getType().getRank() == 2) {
      auto lhsVal = stripValueFromUnrealizedConversionCastOp(adaptor.getA());
      auto rhsVal = stripValueFromUnrealizedConversionCastOp(adaptor.getB());
      auto biasVal = adaptor.getC();
      auto matmulOp = rewriter.create<gcu::MatMulOp>(
          loc, output, lhsVal, rhsVal, biasVal, useBiasVal);
      if (op->getAttr("inputPrecision") && aElemTy.isF32() && bElemTy.isF32())
        matmulOp->setAttr("inputPrecision", op->getAttr("inputPrecision"));
      if (op->getAttr(kAccReuseCandidate))
        matmulOp->setAttr(kAccReuseCandidate, op->getAttr(kAccReuseCandidate));
      if (op->getAttr(kAccStore))
        matmulOp->setAttr(kAccStore, op->getAttr(kAccStore));
    } else {
      auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto lhsMemRef = stripValueFromUnrealizedConversionCastOp(adaptor.getA());
      auto lhsMemRefType = dyn_cast<MemRefType>(lhsMemRef.getType());
      auto rhsMemRef = stripValueFromUnrealizedConversionCastOp(adaptor.getB());
      auto rhsMemRefType = dyn_cast<MemRefType>(rhsMemRef.getType());
      auto biasMemRef = adaptor.getC();
      int64_t batchNum = lhsMemRefType.getShape()[0];

      auto rank3HasStaticStrides = [](MemRefType t) -> bool {
        if (t.getRank() != 3)
          return false;
        SmallVector<int64_t> strides;
        int64_t off = 0;
        if (failed(t.getStridesAndOffset(strides, off)) || strides.size() != 3)
          return false;
        for (int64_t stride : strides)
          if (ShapedType::isDynamic(stride))
            return false;
        (void)off;
        return true;
      };
      if (!rank3HasStaticStrides(lhsMemRefType) ||
          !rank3HasStaticStrides(rhsMemRefType) ||
          !rank3HasStaticStrides(resultMemRefType))
        return op.emitOpError(
            "batched dot expects rank-3 memrefs with static strides");
      if (biasMemRef &&
          !rank3HasStaticStrides(cast<MemRefType>(biasMemRef.getType())))
        return op.emitOpError(
            "batched dot expects rank-3 bias memref with static strides");

      auto sliceBatchToRank2 = [&](OpBuilder &builder, Location loc, Value src,
                                   MemRefType srcTy, Value batchIdx) -> Value {
        auto stridesAndOffset = srcTy.getStridesAndOffset();
        auto strides = stridesAndOffset.first;
        auto meta = builder.create<memref::ExtractStridedMetadataOp>(loc, src);
        Value curOffset = meta.getOffset();
        Value batchStride =
            builder.create<arith::ConstantIndexOp>(loc, strides[0]);
        Value batchOffset =
            builder.create<arith::MulIOp>(loc, batchIdx, batchStride);
        Value totalOffset =
            builder.create<arith::AddIOp>(loc, curOffset, batchOffset);

        int64_t d1 = srcTy.getDimSize(1);
        int64_t d2 = srcTy.getDimSize(2);
        MemRefType out2DType = MemRefType::get(
            {d1, d2}, srcTy.getElementType(),
            StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic,
                                   {strides[1], strides[2]}),
            srcTy.getMemorySpace());
        return builder
            .create<memref::ReinterpretCastOp>(
                loc, out2DType, src, totalOffset,
                ArrayRef<OpFoldResult>{builder.getIndexAttr(d1),
                                       builder.getIndexAttr(d2)},
                ArrayRef<OpFoldResult>{builder.getIndexAttr(strides[1]),
                                       builder.getIndexAttr(strides[2])})
            .getResult();
      };

      scf::buildLoopNest(
          rewriter, loc, ValueRange{zero},
          ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, batchNum)},
          ValueRange{one}, [&](OpBuilder &builder, Location loc, ValueRange m) {
            Value newLhs =
                sliceBatchToRank2(builder, loc, lhsMemRef, lhsMemRefType, m[0]);
            Value newRhs =
                sliceBatchToRank2(builder, loc, rhsMemRef, rhsMemRefType, m[0]);
            Value newBias;
            Value newOut =
                sliceBatchToRank2(builder, loc, output, resultMemRefType, m[0]);
            gcu::MatMulOp matmulOp;
            if (biasMemRef) {
              auto biasMemRefType = dyn_cast<MemRefType>(biasMemRef.getType());
              newBias = sliceBatchToRank2(builder, loc, biasMemRef,
                                          biasMemRefType, m[0]);
            }
            matmulOp = builder.create<gcu::MatMulOp>(
                loc, newOut, newLhs, newRhs, newBias, useBiasVal);
            if (op->getAttr("inputPrecision") && aElemTy.isF32() &&
                bElemTy.isF32())
              matmulOp->setAttr("inputPrecision",
                                op->getAttr("inputPrecision"));
            if (op->getAttr(kAccReuseCandidate))
              matmulOp->setAttr(kAccReuseCandidate,
                                op->getAttr(kAccReuseCandidate));
            if (op->getAttr(kAccStore))
              matmulOp->setAttr(kAccStore, op->getAttr(kAccStore));
          });
    }
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, output);
    return success();
  }
};

mlir::gcu::RMWOp getGcuAtomicOp(mlir::triton::RMWOp rmw_op) {
  if (rmw_op == mlir::triton::RMWOp::AND) {
    return mlir::gcu::RMWOp::AND;
  } else if (rmw_op == mlir::triton::RMWOp::OR) {
    return mlir::gcu::RMWOp::OR;
  } else if (rmw_op == mlir::triton::RMWOp::XOR) {
    return mlir::gcu::RMWOp::XOR;
  } else if (rmw_op == mlir::triton::RMWOp::ADD) {
    return mlir::gcu::RMWOp::ADD;
  } else if (rmw_op == mlir::triton::RMWOp::FADD) {
    return mlir::gcu::RMWOp::ADD;
  } else if (rmw_op == mlir::triton::RMWOp::MAX) {
    return mlir::gcu::RMWOp::MAX;
  } else if (rmw_op == mlir::triton::RMWOp::MIN) {
    return mlir::gcu::RMWOp::MIN;
  } else if (rmw_op == mlir::triton::RMWOp::XCHG) {
    return mlir::gcu::RMWOp::XCHG;
  } else if (rmw_op == mlir::triton::RMWOp::UMAX) {
    return mlir::gcu::RMWOp::UMAX;
  } else {
    return mlir::gcu::RMWOp::UMIN;
  }
}

mlir::gcu::MemSemantic
getGcuAtomicMemSemantic(mlir::triton::MemSemantic mem_semantic) {
  if (mem_semantic == mlir::triton::MemSemantic::ACQUIRE) {
    return mlir::gcu::MemSemantic::ACQUIRE;
  } else if (mem_semantic == mlir::triton::MemSemantic::RELEASE) {
    return mlir::gcu::MemSemantic::RELEASE;
  } else if (mem_semantic == mlir::triton::MemSemantic::ACQUIRE_RELEASE) {
    return mlir::gcu::MemSemantic::ACQUIRE_RELEASE;
  } else {
    return mlir::gcu::MemSemantic::RELAXED;
  }
}

mlir::gcu::MemSyncScope
getGcuAtomicMemSyncScope(mlir::triton::MemSyncScope mem_sync) {
  if (mem_sync == mlir::triton::MemSyncScope::SYSTEM) {
    return mlir::gcu::MemSyncScope::SYSTEM;
  } else if (mem_sync == mlir::triton::MemSyncScope::CTA) {
    return mlir::gcu::MemSyncScope::CTA;
  } else {
    return mlir::gcu::MemSyncScope::GCU;
  }
}

// Build the cluster+CTA serialized atomic loop.
// Ensures at most one thread across the cluster executes at a time:
//
//   for cta = 0 .. totalCTAs:       // cluster-level serialization
//     if (flatCtaId == cta):
//       for warp = 0 .. numWarps:   // CTA-level serialization
//         if (threadId == warp):
//           elementLoop: for each element:
//             elemFn(builder, loc, iters)
//         barrier                   // intra-CTA barrier
//     cluster_barrier               // inter-CTA barrier within cluster
//
static void buildCTASerializedLoop(
    ConversionPatternRewriter &rewriter, Location loc, Operation *op,
    Value zero, Value one, SmallVector<unsigned> numElems,
    SmallVector<Value, 4> numElemValues,
    llvm::function_ref<void(OpBuilder &, Location, ValueRange)> elemFn) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  int numWarps = triton::gcu::getNumWarps(module);

  // --- Read cluster dims from module attributes ---
  auto getClusterDim = [&](StringRef name) -> int {
    if (auto attr = module->getAttrOfType<IntegerAttr>(name))
      return attr.getInt();
    return 1;
  };
  int clusterDimX = getClusterDim("ttg.cluster-dims-x");
  int clusterDimY = getClusterDim("ttg.cluster-dims-y");
  int clusterDimZ = getClusterDim("ttg.cluster-dims-z");
  int totalCTAs = clusterDimX * clusterDimY * clusterDimZ;

  // --- Compute flat CTA ID within the cluster ---
  // flatCtaId = ctaIdX + ctaIdY * clusterDimX   (clusterDimZ == 1)
  auto ctaIdX = rewriter.create<gcu::CTAIdOp>(loc, rewriter.getIndexType(),
                                              rewriter.getI32IntegerAttr(0));
  auto ctaIdY = rewriter.create<gcu::CTAIdOp>(loc, rewriter.getIndexType(),
                                              rewriter.getI32IntegerAttr(1));
  auto dimXVal = rewriter.create<arith::ConstantIndexOp>(loc, clusterDimX);
  auto flatCtaId = rewriter.create<arith::AddIOp>(
      loc, ctaIdX, rewriter.create<arith::MulIOp>(loc, ctaIdY, dimXVal));

  auto totalCTAsVal = rewriter.create<arith::ConstantIndexOp>(loc, totalCTAs);

  // --- Outer loop: CTAs in the cluster take turns ---
  auto ctaLoop = rewriter.create<scf::ForOp>(loc, zero, totalCTAsVal, one);
  {
    OpBuilder::InsertionGuard ctaGuard(rewriter);
    rewriter.setInsertionPointToStart(ctaLoop.getBody());
    Value ctaIv = ctaLoop.getInductionVar();
    auto isMyCTA = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  flatCtaId, ctaIv);

    rewriter.create<scf::IfOp>(
        loc, isMyCTA, [&](OpBuilder ctaBuilder, Location loc) {
          if (numWarps <= 1) {
            scf::buildLoopNest(
                ctaBuilder, loc, SmallVector<Value, 4>(numElems.size(), zero),
                numElemValues, SmallVector<Value, 4>(numElems.size(), one),
                elemFn);
          } else {
            // --- Inner loop: warps take turns within this CTA ---
            auto numWarpsVal =
                ctaBuilder.create<arith::ConstantIndexOp>(loc, numWarps);
            auto threadId =
                ctaBuilder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);

            auto warpLoop =
                ctaBuilder.create<scf::ForOp>(loc, zero, numWarpsVal, one);
            {
              OpBuilder::InsertionGuard warpGuard(ctaBuilder);
              ctaBuilder.setInsertionPointToStart(warpLoop.getBody());
              Value warpIv = warpLoop.getInductionVar();
              auto isMyWarp = ctaBuilder.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq, threadId, warpIv);

              ctaBuilder.create<scf::IfOp>(
                  loc, isMyWarp, [&](OpBuilder warpBuilder, Location loc) {
                    scf::buildLoopNest(
                        warpBuilder, loc,
                        SmallVector<Value, 4>(numElems.size(), zero),
                        numElemValues,
                        SmallVector<Value, 4>(numElems.size(), one), elemFn);
                    warpBuilder.create<scf::YieldOp>(loc);
                  });
              ctaBuilder.create<gpu::BarrierOp>(loc);
            }
          }
          ctaBuilder.create<scf::YieldOp>(loc);
        });
    rewriter.create<gcu::ClusterBarrierOp>(loc);
  }
}

// Emit a software RMW:
//   old = load(memref[idx]);
//   new = op(old, val);
//   store(memref[idx], new);
//   return old.
static Value emitSoftwareRMW(OpBuilder &builder, Location loc,
                             mlir::triton::RMWOp rmwOp, Value memref, Value idx,
                             Value val, Type elemType) {
  auto old =
      builder.create<memref::LoadOp>(loc, elemType, memref, ValueRange{idx});
  Value newVal;
  bool isFloat = mlir::isa<FloatType>(elemType);
  if (rmwOp == mlir::triton::RMWOp::ADD || rmwOp == mlir::triton::RMWOp::FADD) {
    newVal = isFloat ? builder.create<arith::AddFOp>(loc, old, val).getResult()
                     : builder.create<arith::AddIOp>(loc, old, val).getResult();
  } else if (rmwOp == mlir::triton::RMWOp::AND) {
    newVal = builder.create<arith::AndIOp>(loc, old, val);
  } else if (rmwOp == mlir::triton::RMWOp::OR) {
    newVal = builder.create<arith::OrIOp>(loc, old, val);
  } else if (rmwOp == mlir::triton::RMWOp::XOR) {
    newVal = builder.create<arith::XOrIOp>(loc, old, val);
  } else if (rmwOp == mlir::triton::RMWOp::MAX) {
    newVal = isFloat
                 ? builder.create<arith::MaximumFOp>(loc, old, val).getResult()
                 : builder.create<arith::MaxSIOp>(loc, old, val).getResult();
  } else if (rmwOp == mlir::triton::RMWOp::MIN) {
    newVal = isFloat
                 ? builder.create<arith::MinimumFOp>(loc, old, val).getResult()
                 : builder.create<arith::MinSIOp>(loc, old, val).getResult();
  } else if (rmwOp == mlir::triton::RMWOp::UMAX) {
    newVal = builder.create<arith::MaxUIOp>(loc, old, val);
  } else if (rmwOp == mlir::triton::RMWOp::UMIN) {
    newVal = builder.create<arith::MinUIOp>(loc, old, val);
  } else if (rmwOp == mlir::triton::RMWOp::XCHG) {
    newVal = val;
  } else {
    llvm::report_fatal_error("unsupported RMW op for CTA software emulation");
  }
  builder.create<memref::StoreOp>(loc, newVal, memref, ValueRange{idx});
  return old;
}

// Emit a software CAS:
//   old = load(memref[idx]);
//   if (old == cmp):
//     store(memref[idx], val);
//   return old.
static Value emitSoftwareCAS(OpBuilder &builder, Location loc, Value memref,
                             Value idx, Value cmp, Value val, Type elemType) {
  auto old =
      builder.create<memref::LoadOp>(loc, elemType, memref, ValueRange{idx});
  Value isEqual;
  if (mlir::isa<FloatType>(elemType))
    isEqual =
        builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, old, cmp);
  else
    isEqual =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, old, cmp);
  builder.create<scf::IfOp>(
      loc, isEqual, [&](OpBuilder innerBuilder, Location loc) {
        innerBuilder.create<memref::StoreOp>(loc, val, memref, ValueRange{idx});
        innerBuilder.create<scf::YieldOp>(loc);
      });
  return old;
}

static bool isAtomicOnSharedMemory(mlir::Operation *op) {
  Type ptrType;
  if (auto rmwOp = dyn_cast<triton::AtomicRMWOp>(op)) {
    ptrType = rmwOp.getPtr().getType();
  } else if (auto casOp = dyn_cast<triton::AtomicCASOp>(op)) {
    ptrType = casOp.getPtr().getType();
  }

  int addressSpace = 1;
  if (auto tensorTy = dyn_cast_or_null<TensorType>(ptrType)) {
    if (auto ptrElemTy =
            dyn_cast<triton::PointerType>(tensorTy.getElementType()))
      addressSpace = ptrElemTy.getAddressSpace();
  } else if (auto ptrTy = dyn_cast<triton::PointerType>(ptrType)) {
    addressSpace = ptrTy.getAddressSpace();
  }
  return addressSpace == 3 || addressSpace == 2;
}

struct TTAtomicRMWOpLowering : SharedConversionPattern<triton::AtomicRMWOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    auto op_attr = getGcuAtomicOp(op.getAtomicRmwOp());
    auto mem_semantic = getGcuAtomicMemSemantic(op.getSem());
    auto mem_sync_scope = getGcuAtomicMemSyncScope(op.getScope());
    bool isCtaScope = mem_sync_scope == mlir::gcu::MemSyncScope::CTA;
    bool isSharedMem = isAtomicOnSharedMemory(op.getOperation());

    if (isCtaScope && !isSharedMem) {
      mem_sync_scope = mlir::gcu::MemSyncScope::GCU;
      isCtaScope = false;
    }

    if (op.getVal().getType().isIntOrFloat()) {
      mlir::Value ptr = adaptor.getPtr();
      mlir::Value val = adaptor.getVal();
      auto mask =
          adaptor.getMask()
              ? adaptor.getMask()
              : rewriter.create<arith::ConstantIntOp>(loc, 1, 1).getResult();
      auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto masterWarpId = getMasterThreadId(op.getOperation());
      auto isMasterThread =
          rewriter
              .create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq,
                  rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
                  rewriter.create<arith::ConstantIndexOp>(loc, masterWarpId))
              .getResult();
      auto thread_mask =
          rewriter.create<arith::AndIOp>(loc, mask, isMasterThread).getResult();

      auto elemType = op.getResult().getType();
      auto memrefType =
          MemRefType::get(SmallVector<int64_t>{1}, elemType, AffineMap{},
                          rewriter.getI64IntegerAttr(2) /*shared memory*/);
      auto lastUser =
          userAnalysis.getLastUser(op.getOperation()->getResults()[0]);
      auto localMem = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                  replaced2Origin, memrefType);

      if (isCtaScope) {
        ModuleOp module = op->getParentOfType<ModuleOp>();
        auto getClusterDim = [&](StringRef name) -> int {
          if (auto attr = module->getAttrOfType<IntegerAttr>(name))
            return attr.getInt();
          return 1;
        };
        int clusterDimX = getClusterDim("ttg.cluster-dims-x");
        int clusterDimY = getClusterDim("ttg.cluster-dims-y");
        int clusterDimZ = getClusterDim("ttg.cluster-dims-z");
        int totalCTAs = clusterDimX * clusterDimY * clusterDimZ;

        auto ctaIdX = rewriter.create<gcu::CTAIdOp>(
            loc, rewriter.getIndexType(), rewriter.getI32IntegerAttr(0));
        auto ctaIdY = rewriter.create<gcu::CTAIdOp>(
            loc, rewriter.getIndexType(), rewriter.getI32IntegerAttr(1));
        auto dimXVal =
            rewriter.create<arith::ConstantIndexOp>(loc, clusterDimX);
        auto flatCtaId = rewriter.create<arith::AddIOp>(
            loc, ctaIdX, rewriter.create<arith::MulIOp>(loc, ctaIdY, dimXVal));
        auto totalCTAsVal =
            rewriter.create<arith::ConstantIndexOp>(loc, totalCTAs);
        auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        auto ctaLoop =
            rewriter.create<scf::ForOp>(loc, zero, totalCTAsVal, one);
        {
          OpBuilder::InsertionGuard ctaGuard(rewriter);
          rewriter.setInsertionPointToStart(ctaLoop.getBody());
          Value ctaIv = ctaLoop.getInductionVar();
          auto isMyCTA = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, flatCtaId, ctaIv);

          rewriter.create<scf::IfOp>(
              loc, isMyCTA, [&](OpBuilder ctaBuilder, Location loc) {
                ctaBuilder.create<scf::IfOp>(
                    loc, thread_mask,
                    [&](OpBuilder innerBuilder, Location loc) {
                      auto dynMemType = MemRefType::get(
                          ArrayRef<int64_t>{ShapedType::kDynamic}, elemType);
                      auto buffer = innerBuilder.create<gcu::PtrToMemRefOp>(
                          loc, dynMemType, ptr);
                      auto oldVal = emitSoftwareRMW(innerBuilder, loc,
                                                    op.getAtomicRmwOp(), buffer,
                                                    zero, val, elemType);
                      innerBuilder.create<memref::StoreOp>(
                          loc, oldVal, localMem, ValueRange{zero});
                      innerBuilder.create<scf::YieldOp>(loc);
                    });
                ctaBuilder.create<scf::YieldOp>(loc);
              });
          rewriter.create<gcu::ClusterBarrierOp>(loc);
        }
      } else {
        rewriter.create<scf::IfOp>(
            loc, thread_mask, [&](OpBuilder builder, Location loc) {
              if (op.getSem() == mlir::triton::MemSemantic::RELEASE ||
                  op.getSem() == mlir::triton::MemSemantic::ACQUIRE_RELEASE)
                builder.create<gcu::MFenceOp>(loc, gcu::MFenceType::Device);

              auto atomicRMWOp = rewriter.create<mlir::gcu::AtomicRMWOp>(
                  loc, elemType, op_attr, ptr, val, mem_semantic,
                  mem_sync_scope);
              builder.create<memref::StoreOp>(loc, atomicRMWOp.getResult(),
                                              localMem, ValueRange{zero});

              if (op.getSem() == mlir::triton::MemSemantic::ACQUIRE ||
                  op.getSem() == mlir::triton::MemSemantic::ACQUIRE_RELEASE)
                builder.create<gcu::MFenceOp>(loc, gcu::MFenceType::Device);
              builder.create<scf::YieldOp>(loc);
            });
      }

      rewriter.create<gpu::BarrierOp>(loc);
      rewriter.replaceOpWithNewOp<memref::LoadOp>(op, elemType, localMem,
                                                  ValueRange{zero});
      return success();
    } else if (mlir::isa<mlir::TensorType>(op.getVal().getType())) {
      mlir::Value ptrs = adaptor.getPtr();
      mlir::Value vals = adaptor.getVal();
      mlir::Value masks = adaptor.getMask();
      auto tensor_type = dyn_cast<mlir::TensorType>(op.getResult().getType());
      auto elemType = tensor_type.getElementType();
      auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto true_bool = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
      auto numElems = triton::gcu::getElemsPerThread(op.getType());
      auto numElemValues = getElemsPerThread(rewriter, loc, op.getType());

      bool hasUsers = !op.getResult().use_empty();
      Value output;
      if (hasUsers) {
        auto resultType = dyn_cast<MemRefType>(
            getTypeConverter()->convertType(op.getResult().getType()));
        auto lastUser =
            userAnalysis.getLastUser(op.getOperation()->getResults()[0]);
        output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                             replaced2Origin, resultType);

        // zero-init output so masked-off elements don't produce garbage values.
        Value zeroVal;
        if (mlir::isa<FloatType>(elemType))
          zeroVal = rewriter.create<arith::ConstantFloatOp>(
              loc, cast<FloatType>(elemType),
              APFloat::getZero(cast<FloatType>(elemType).getFloatSemantics()));
        else
          zeroVal = rewriter.create<arith::ConstantIntOp>(loc, elemType, 0);
        auto totalNumElems = triton::gcu::getTotalElemsPerThread(op.getType());
        auto tag = pTagPool.getPrivateSyncTagInfo(op.getOperation());
        doMemset(rewriter, tag, op.getOperation(), output, zeroVal,
                 totalNumElems);
      }

      if (isCtaScope) {
        auto freeWarpMask = triton::gcu::getFreeWarpMask(op.getType());
        bool hasWarpRedundancy =
            llvm::any_of(freeWarpMask, [](bool b) { return !b; });
        int32_t warpBitmask = 0;
        if (hasWarpRedundancy) {
          for (unsigned i = 0; i < freeWarpMask.size(); ++i)
            if (freeWarpMask[i])
              warpBitmask |= (1 << i);
        }

        buildCTASerializedLoop(
            rewriter, loc, op.getOperation(), zero, one, numElems,
            numElemValues,
            [&](OpBuilder &elemBuilder, Location loc, ValueRange iters) {
              auto ptr_int =
                  elemBuilder.create<memref::LoadOp>(loc, ptrs, iters)
                      .getResult();
              auto ptr = elemBuilder.create<gcu::IntToPtrOp>(
                  loc, gcu::PtrType::get(getContext(), elemType), ptr_int);
              auto val = elemBuilder.create<memref::LoadOp>(loc, vals, iters)
                             .getResult();
              auto mask =
                  adaptor.getMask()
                      ? elemBuilder.create<memref::LoadOp>(loc, masks, iters)
                            .getResult()
                      : elemBuilder.create<arith::ConstantIntOp>(loc, 1, 1)
                            .getResult();
              Value thread_select =
                  elemBuilder
                      .create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                             mask, true_bool)
                      .getResult();

              if (hasWarpRedundancy) {
                auto threadId =
                    elemBuilder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
                auto tidI32 = elemBuilder.create<arith::IndexCastOp>(
                    loc, elemBuilder.getI32Type(), threadId);
                auto bitmaskVal = elemBuilder.create<arith::ConstantIntOp>(
                    loc, warpBitmask, 32);
                auto shifted =
                    elemBuilder.create<arith::ShRUIOp>(loc, bitmaskVal, tidI32);
                auto oneI32 =
                    elemBuilder.create<arith::ConstantIntOp>(loc, 1, 32);
                auto bit =
                    elemBuilder.create<arith::AndIOp>(loc, shifted, oneI32);
                auto zeroI32 =
                    elemBuilder.create<arith::ConstantIntOp>(loc, 0, 32);
                auto isNonRedundant = elemBuilder.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::ne, bit, zeroI32);
                thread_select =
                    elemBuilder
                        .create<arith::AndIOp>(loc, thread_select,
                                               isNonRedundant.getResult())
                        .getResult();
              }

              elemBuilder.create<scf::IfOp>(
                  loc, thread_select,
                  [&](OpBuilder innerBuilder, Location loc) {
                    auto dynMemType = MemRefType::get(
                        ArrayRef<int64_t>{ShapedType::kDynamic}, elemType);
                    auto buffer = innerBuilder.create<gcu::PtrToMemRefOp>(
                        loc, dynMemType, ptr);
                    auto oldVal =
                        emitSoftwareRMW(innerBuilder, loc, op.getAtomicRmwOp(),
                                        buffer, zero, val, elemType);
                    if (hasUsers) {
                      innerBuilder.create<memref::StoreOp>(loc, oldVal, output,
                                                           iters);
                    }
                    innerBuilder.create<scf::YieldOp>(loc);
                  });
            });
      } else {
        scf::buildLoopNest(
            rewriter, loc, SmallVector<Value, 4>(numElems.size(), zero),
            numElemValues, SmallVector<Value, 4>(numElems.size(), one),
            [&](OpBuilder &builder, Location loc, ValueRange iters) {
              auto ptr_int =
                  builder.create<memref::LoadOp>(loc, ptrs, iters).getResult();
              auto ptr = builder.create<gcu::IntToPtrOp>(
                  loc, gcu::PtrType::get(getContext(), elemType), ptr_int);
              auto val =
                  builder.create<memref::LoadOp>(loc, vals, iters).getResult();
              auto mask =
                  adaptor.getMask()
                      ? builder.create<memref::LoadOp>(loc, masks, iters)
                            .getResult()
                      : builder.create<arith::ConstantIntOp>(loc, 1, 1)
                            .getResult();
              auto thread_select = builder.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq, mask, true_bool);

              builder.create<scf::IfOp>(
                  loc, thread_select, [&](OpBuilder builder, Location loc) {
                    if (op.getSem() == mlir::triton::MemSemantic::RELEASE ||
                        op.getSem() ==
                            mlir::triton::MemSemantic::ACQUIRE_RELEASE)
                      builder.create<gcu::MFenceOp>(loc,
                                                    gcu::MFenceType::Device);

                    auto atomicRMWOp = builder.create<mlir::gcu::AtomicRMWOp>(
                        loc, elemType, op_attr, ptr, val, mem_semantic,
                        mem_sync_scope);
                    if (hasUsers) {
                      builder.create<memref::StoreOp>(
                          loc, atomicRMWOp.getResult(), output, iters);
                    }

                    if (op.getSem() == mlir::triton::MemSemantic::ACQUIRE ||
                        op.getSem() ==
                            mlir::triton::MemSemantic::ACQUIRE_RELEASE)
                      builder.create<gcu::MFenceOp>(loc,
                                                    gcu::MFenceType::Device);
                    builder.create<scf::YieldOp>(loc);
                  });
            });
        rewriter.create<gpu::BarrierOp>(loc);
      }
      leaveTritionOp(rewriter, op.getOperation());
      if (hasUsers) {
        rewriter.replaceOp(op, output);
      } else {
        rewriter.eraseOp(op);
      }
      return success();
    }
    return failure();
  }
};

struct TTAtomicCASOpLowering : SharedConversionPattern<triton::AtomicCASOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    auto mem_semantic = getGcuAtomicMemSemantic(op.getSem());
    auto mem_sync_scope = getGcuAtomicMemSyncScope(op.getScope());
    bool isCtaScope = mem_sync_scope == mlir::gcu::MemSyncScope::CTA;
    bool isSharedMem = isAtomicOnSharedMemory(op.getOperation());

    if (isCtaScope && !isSharedMem) {
      mem_sync_scope = mlir::gcu::MemSyncScope::GCU;
      isCtaScope = false;
    }

    if (op.getVal().getType().isIntOrFloat()) {
      mlir::Value ptr = adaptor.getPtr();
      mlir::Value cmp = adaptor.getCmp();
      mlir::Value val = adaptor.getVal();
      if (isCtaScope) {
        auto elemType = op.getResult().getType();
        auto dynMemType =
            MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic}, elemType);
        auto buffer = rewriter.create<gcu::PtrToMemRefOp>(loc, dynMemType, ptr);
        auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto oldVal = emitSoftwareCAS(rewriter, loc, buffer, zero, cmp, val,
                                      op.getResult().getType());
        rewriter.replaceOp(op, oldVal);
      } else {
        auto resTy = op.getResult().getType();
        auto casTy = resTy;
        bool needsBitcast = resTy.isF32();
        if (needsBitcast) {
          auto i32Ty = rewriter.getI32Type();
          casTy = i32Ty;
          ptr = rewriter.create<gcu::IntToPtrOp>(
              loc, gcu::PtrType::get(getContext(), i32Ty),
              rewriter.create<gcu::PtrToIntOp>(loc, ptr));
          cmp = rewriter.create<arith::BitcastOp>(loc, i32Ty, cmp);
          val = rewriter.create<arith::BitcastOp>(loc, i32Ty, val);
        }
        auto new_op = rewriter.create<mlir::gcu::AtomicCASOp>(
            loc, casTy, ptr, cmp, val, mem_semantic, mem_sync_scope);
        mlir::Value result = new_op.getResult();
        if (needsBitcast)
          result = rewriter.create<arith::BitcastOp>(loc, resTy, result);
        rewriter.replaceOp(op, result);
      }
      return success();
    } else if (mlir::isa<mlir::TensorType>(op.getVal().getType())) {
      mlir::Value ptrs = adaptor.getPtr();
      mlir::Value cmps = adaptor.getCmp();
      mlir::Value vals = adaptor.getVal();
      auto tensor_type = dyn_cast<mlir::TensorType>(op.getResult().getType());
      auto elemType = tensor_type.getElementType();
      auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto numElems = triton::gcu::getElemsPerThread(op.getType());
      auto numElemValues = getElemsPerThread(rewriter, loc, op.getType());

      bool hasUsers = !op.getResult().use_empty();
      Value output;
      if (hasUsers) {
        auto resultType = dyn_cast<MemRefType>(
            getTypeConverter()->convertType(op.getResult().getType()));
        auto lastUser =
            userAnalysis.getLastUser(op.getOperation()->getResults()[0]);
        output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                             replaced2Origin, resultType);

        // zero-init output so masked-off elements don't produce garbage values.
        Value zeroVal;
        if (mlir::isa<FloatType>(elemType))
          zeroVal = rewriter.create<arith::ConstantFloatOp>(
              loc, cast<FloatType>(elemType),
              APFloat::getZero(cast<FloatType>(elemType).getFloatSemantics()));
        else
          zeroVal = rewriter.create<arith::ConstantIntOp>(loc, elemType, 0);
        auto totalNumElems = triton::gcu::getTotalElemsPerThread(op.getType());
        auto tag = pTagPool.getPrivateSyncTagInfo(op.getOperation());
        doMemset(rewriter, tag, op.getOperation(), output, zeroVal,
                 totalNumElems);
      }

      if (isCtaScope) {
        // CTA scope: serialize across subthreads for correctness.
        buildCTASerializedLoop(
            rewriter, loc, op.getOperation(), zero, one, numElems,
            numElemValues,
            [&](OpBuilder &elemBuilder, Location loc, ValueRange iters) {
              auto ptr_int =
                  elemBuilder.create<memref::LoadOp>(loc, ptrs, iters)
                      .getResult();
              auto ptr = elemBuilder.create<gcu::IntToPtrOp>(
                  loc, gcu::PtrType::get(getContext(), elemType), ptr_int);
              auto cmp = elemBuilder.create<memref::LoadOp>(loc, cmps, iters)
                             .getResult();
              auto val = elemBuilder.create<memref::LoadOp>(loc, vals, iters)
                             .getResult();

              auto dynMemType = MemRefType::get(
                  ArrayRef<int64_t>{ShapedType::kDynamic}, elemType);
              auto buffer =
                  elemBuilder.create<gcu::PtrToMemRefOp>(loc, dynMemType, ptr);
              auto oldVal = emitSoftwareCAS(elemBuilder, loc, buffer, zero, cmp,
                                            val, elemType);
              if (hasUsers) {
                elemBuilder.create<memref::StoreOp>(loc, oldVal, output, iters);
              }
            });
      } else {
        bool needsBitcast = elemType.isF32();
        auto casElemTy = needsBitcast ? (Type)rewriter.getI32Type() : elemType;

        scf::buildLoopNest(
            rewriter, loc, SmallVector<Value, 4>(numElems.size(), zero),
            numElemValues, SmallVector<Value, 4>(numElems.size(), one),
            [&](OpBuilder &builder, Location loc, ValueRange iters) {
              auto ptr_int =
                  builder.create<memref::LoadOp>(loc, ptrs, iters).getResult();
              auto ptr = builder.create<gcu::IntToPtrOp>(
                  loc, gcu::PtrType::get(getContext(), casElemTy), ptr_int);
              auto cmp =
                  builder.create<memref::LoadOp>(loc, cmps, iters).getResult();
              auto val =
                  builder.create<memref::LoadOp>(loc, vals, iters).getResult();

              if (needsBitcast) {
                cmp = builder.create<arith::BitcastOp>(loc, casElemTy, cmp);
                val = builder.create<arith::BitcastOp>(loc, casElemTy, val);
              }

              if (op.getSem() == mlir::triton::MemSemantic::RELEASE ||
                  op.getSem() == mlir::triton::MemSemantic::ACQUIRE_RELEASE) {
                builder.create<gcu::MFenceOp>(loc, gcu::MFenceType::Device);
              }

              auto atomicCasOp = builder.create<mlir::gcu::AtomicCASOp>(
                  loc, casElemTy, ptr, cmp, val, mem_semantic, mem_sync_scope);
              if (hasUsers) {
                Value result = atomicCasOp.getResult();
                if (needsBitcast)
                  result =
                      builder.create<arith::BitcastOp>(loc, elemType, result);
                builder.create<memref::StoreOp>(loc, result, output, iters);
              }

              if (op.getSem() == mlir::triton::MemSemantic::ACQUIRE ||
                  op.getSem() == mlir::triton::MemSemantic::ACQUIRE_RELEASE) {
                builder.create<gcu::MFenceOp>(loc, gcu::MFenceType::Device);
              }
            });
      }
      leaveTritionOp(rewriter, op.getOperation());
      if (hasUsers) {
        rewriter.replaceOp(op, output);
      } else {
        rewriter.eraseOp(op);
      }
      return success();
    }
    return failure();
  }
};

struct TTInitBarrierOpLowering
    : public SharedConversionPattern<triton::gcu::InitBarrierOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcu::InitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    // rewriter.setInsertionPoint(op.getOperation());
    auto loc = op.getLoc();
    auto barrierType =
        getTypeConverter()->convertType(op.getBarrier().getType());
    auto newOp = rewriter.create<gcu::AllocBarrierOp>(loc, barrierType);
    auto barrier = newOp.getResult();

    auto masterWarpId = getMasterThreadId(op.getOperation());
    auto threadId = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    auto isMasterThread = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, threadId,
        rewriter.create<arith::ConstantIndexOp>(loc, masterWarpId));
    Value count = rewriter.create<arith::ConstantIntOp>(loc, op.getCount(), 32);
    rewriter.create<scf::IfOp>(
        loc, isMasterThread, [&](OpBuilder &builder, Location loc) {
          builder.create<gcu::InitBarrierOp>(loc, barrier, count);
          builder.create<scf::YieldOp>(loc);
        });
    rewriter.create<gpu::BarrierOp>(loc);
    leaveTritionOp(rewriter, op);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct TTInitPipelineOpLowering
    : public SharedConversionPattern<triton::gcuws::InitPipelineOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcuws::InitPipelineOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    auto pipelineType =
        getTypeConverter()->convertType(op.getPipeline().getType());
    auto newOp = rewriter.create<gcu::AllocPipelineOp>(loc, pipelineType);
    auto pipeline = newOp.getResult();

    auto masterWarpId = getMasterThreadId(op.getOperation());
    auto threadId = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    auto isMasterThread = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, threadId,
        rewriter.create<arith::ConstantIndexOp>(loc, masterWarpId));
    rewriter.create<scf::IfOp>(
        loc, isMasterThread, [&](OpBuilder &builder, Location loc) {
          builder.create<gcu::InitPipelineOp>(loc, pipeline);
          builder.create<scf::YieldOp>(loc);
        });
    rewriter.create<gpu::BarrierOp>(loc);
    leaveTritionOp(rewriter, op);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

template <typename SrcOp, typename DstOp>
struct TTBarrierPipelineOpLowering : public SharedConversionPattern<SrcOp> {
  using SharedConversionPattern<SrcOp>::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op,
                  typename SharedConversionPattern<SrcOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (this->pTagPool.isExistInMap(op.getOperation())) {
      this->pTagPool.releaseMap(op.getOperation());
    }
    assert(adaptor.getOperands().size() == 1 &&
           "TTBarrierPipelineOpLowering expects a single operand");
    auto newOp =
        rewriter.create<DstOp>(op.getLoc(), adaptor.getOperands().front());
    leaveTritionOp(rewriter, op);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct TTGWarpSpecializeOpLowering
    : public SharedConversionPattern<triton::gpu::WarpSpecializeOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::WarpSpecializeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation()))
      pTagPool.releaseMap(op.getOperation());

    SmallVector<Type> resultTypes;
    for (auto t : op.getResultTypes())
      resultTypes.push_back(getTypeConverter()->convertType(t));

    const int32_t defaultNumWarps = 1;
    int32_t defaultStartId = 0;
    if (std::optional<ArrayRef<int32_t>> startIds = op.getWarpGroupStartIds()) {
      defaultStartId = startIds->front() + op.getTotalPartitionWarps();
    }
    auto newOp = rewriter.create<gcu::WarpSpecializeOp>(
        op.getLoc(), resultTypes, adaptor.getExplicitCaptures(),
        rewriter.getI32IntegerAttr(defaultNumWarps),
        op.getPartitionNumWarpsAttr(),
        rewriter.getI32IntegerAttr(defaultStartId),
        op.getWarpGroupStartIdsAttr(), op.getActualRegistersAttr());

    newOp.getDefaultRegion().getBlocks().clear();
    rewriter.inlineRegionBefore(op.getDefaultRegion(), newOp.getDefaultRegion(),
                                newOp.getDefaultRegion().end());

    newOp.getPartitionOpHolder().getBlocks().clear();
    rewriter.inlineRegionBefore(op.getPartitionOpHolder(),
                                newOp.getPartitionOpHolder(),
                                newOp.getPartitionOpHolder().end());

    replaced2Origin[newOp.getOperation()] = op.getOperation();
    leaveTritionOp(rewriter, op);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct TTGWarpSpecializePartitionsOpLowering
    : public SharedConversionPattern<triton::gpu::WarpSpecializePartitionsOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::WarpSpecializePartitionsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (pTagPool.isExistInMap(op.getOperation()))
      pTagPool.releaseMap(op.getOperation());

    unsigned numPartitions = op.getPartitionRegions().size();
    auto newOp = rewriter.create<gcu::WarpSpecializePartitionsOp>(
        op.getLoc(), numPartitions);

    for (unsigned i = 0; i < numPartitions; ++i) {
      // Remap default region input types.
      auto &block = op.getPartitionRegions()[i].front();
      TypeConverter::SignatureConversion signatureConversion(
          block.getNumArguments());

      for (auto [idx, type] : llvm::enumerate(block.getArgumentTypes())) {
        SmallVector<Type, 8> converted;
        converted.push_back(getTypeConverter()->convertType(type));
        signatureConversion.addInputs(idx, converted);
      }

      newOp.getPartitionRegions()[i].getBlocks().clear();
      rewriter.inlineRegionBefore(op.getPartitionRegions()[i],
                                  newOp.getPartitionRegions()[i],
                                  newOp.getPartitionRegions()[i].end());
      if (failed(rewriter.convertRegionTypes(&newOp.getPartitionRegions()[i],
                                             *getTypeConverter(),
                                             &signatureConversion)))
        return failure();
    }

    replaced2Origin[newOp.getOperation()] = op.getOperation();
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct TTGWarpYieldOpLowering
    : public SharedConversionPattern<triton::gpu::WarpYieldOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::WarpYieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation()))
      pTagPool.releaseMap(op.getOperation());
    auto newOp =
        rewriter.create<gcu::WarpYieldOp>(op.getLoc(), adaptor.getValues());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct TTGWarpReturnOpLowering
    : public SharedConversionPattern<triton::gpu::WarpReturnOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::WarpReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation()))
      pTagPool.releaseMap(op.getOperation());
    auto newOp = rewriter.create<gcu::WarpReturnOp>(op.getLoc());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

} // namespace

void ConvertTritonToGCUPass::runOnOperation() {
  auto *ctx = &getContext();
  auto moduleOp = getOperation();

  // Get entry function
  Operation *entryFunc = nullptr;
  moduleOp.walk([&](triton::FuncOp funcOp) {
    if (funcOp.isPublic()) {
      entryFunc = funcOp.getOperation();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  assert(entryFunc && "entry function not found");

  // Scan triton IR for triton_gcu.remote_memdesc to detect cross-CTA sharing.
  // (tle.remote_pointers has already been lowered to remote_memdesc by
  // TleToTritonGCUPass which runs before this pass.)
  bool hasCrossCTAShare = false;
  moduleOp.walk([&](triton::gcu::RemoteMemDescOp) {
    hasCrossCTAShare = true;
    return WalkResult::interrupt();
  });

  // Private tag pool
  int numWarps = triton::gcu::getTotalNumWarps(moduleOp);
  bool useAsyncSharedTag = false;
  bool useAllTags = false;
  moduleOp.walk([&](triton::gcu::CopyGlobalToLocalOp copyOp) {
    auto asyncAttr =
        llvm::cast_if_present<BoolAttr>(copyOp->getAttr(kLoadAsync));
    if (asyncAttr && asyncAttr.getValue()) {
      useAsyncSharedTag = true;
      if (copyOp->getParentOfType<gcu::WarpSpecializeOp>()) {
        useAllTags = true;
        return WalkResult::interrupt();
        // } else if (numWarps >= 4) {
        //   for (Operation *user : copyOp.getDstMem().getUsers()) {
        //     if (user == copyOp.getOperation())
        //       continue;
        //     if (isa<triton::DotOp>(user)) {
        //       useAllTags = true;
        //       return WalkResult::interrupt();
        //     } else if (auto localLoadOp =
        //                    dyn_cast<triton::gpu::LocalLoadOp>(user)) {
        //       if (localLoadOp->hasOneUse() &&
        //           isa<triton::DotOp>(*localLoadOp.getResult().user_begin()))
        //           {
        //         useAllTags = true;
        //         return WalkResult::interrupt();
        //       }
        //     }
        //   }
        //   return WalkResult::advance();
      }
    }
    return WalkResult::advance();
  });
  triton::gcu::PrivateTagPool pTagPool(entryFunc, numWarps, useAsyncSharedTag,
                                       useAllTags);

  // pre analysis base triton ir
  triton::gcu::FirstLastUserAnalysis &userAnalysis =
      getAnalysis<triton::gcu::FirstLastUserAnalysis>();

  std::map<Operation *, Operation *> replaced2Origin;
  replaced2Origin.clear();

  std::map<Operation *, std::map<uint64_t, bool>>
      TTYeiledOPerandHasMultiUseStage;
  AnalysisYieldOperendUseStage(moduleOp, userAnalysis,
                               TTYeiledOPerandHasMultiUseStage);

  RewritePatternSet patterns(ctx);
  // define converter
  TypeConverter converter;
  // default
  converter.addConversion([](Type type) { return type; });
  converter.addConversion([](mlir::triton::gcu::PtrType type) {
    return gcu::PtrType::get(type.getContext(), type.getElementType());
  });
  // // pointer type
  // converter.addConversion([](triton::PointerType ptrType) -> Type {
  //   if (auto ty = dyn_cast<RankedTensorType>(ptrType.getPointeeType()))
  //     return mlir::triton::gcu::TileDescType::get(ty.getContext(), ty);
  //   return mlir::triton::gcu::PtrType::get(ptrType.getContext(),
  //                                          ptrType.getPointeeType());
  // });
  // pointer type
  converter.addConversion([](triton::PointerType ptrType) -> Type {
    if (auto ty = dyn_cast<RankedTensorType>(ptrType.getPointeeType()))
      return mlir::gcu::TileDescType::get(ty.getContext(), ty);
    return gcu::PtrType::get(ptrType.getContext(), ptrType.getPointeeType());
  });
  // tensor type
  converter.addConversion([&](TensorType tensorType) {
    auto numElems = triton::gcu::getElemsPerThread(tensorType);
    SmallVector<int64_t, 4> shape(numElems.begin(), numElems.end());
    auto elemType = converter.convertType(tensorType.getElementType());
    // todo_AT weird ptr
    if (isa<mlir::triton::gcu::PtrType>(elemType) ||
        isa<gcu::PtrType>(elemType))
      // use i64 for pointer type
      elemType = IntegerType::get(tensorType.getContext(), 64);
    if (auto tType = dyn_cast<RankedTensorType>(tensorType)) {
      auto encoding = tType.getEncoding();
      if (mlir::isa<triton::gpu::SharedEncodingTrait>(encoding)) {
        return MemRefType::get(
            shape, elemType, AffineMap{},
            IntegerAttr::get(IntegerType::get(tensorType.getContext(), 64), 2));
      } else if (isa<triton::gpu::DotOperandEncodingAttr>(encoding)) {
        return mlir::getMemRefTypeFromSharedMem(tType, elemType);
      }
    }
    return MemRefType::get(shape, elemType);
  });

  converter.addConversion([&](triton::gpu::MemDescType bufferType) {
    auto elemType = converter.convertType(bufferType.getElementType());
    return MemRefType::get(
        bufferType.getShape(), elemType, AffineMap{},
        IntegerAttr::get(IntegerType::get(bufferType.getContext(), 64), 2));
  });
  converter.addConversion([&](triton::gpu::AsyncTokenType tokenType) {
    return IntegerType::get(tokenType.getContext(), 32);
  });
  converter.addConversion([&](triton::gcu::BarrierType barrierType) {
    return gcu::BarrierType::get(
        barrierType.getContext(),
        gcu::AddressSpaceAttr::get(ctx, gcu::AddressSpace::Workgroup));
  });
  converter.addConversion([&](triton::gcuws::PipelineType pipelineType) {
    return gcu::PipelineType::get(
        pipelineType.getContext(), pipelineType.getStageCount(),
        pipelineType.getProducerCount(), pipelineType.getConsumerCount(),
        gcu::AddressSpaceAttr::get(ctx, gcu::AddressSpace::Workgroup),
        pipelineType.getInnerBarrier());
  });
  ConversionTarget target(getContext());

  mlir::triton::populateLoadStoreOpToGCUPatterns(
      converter, patterns, userAnalysis, replaced2Origin, pTagPool);
  mlir::triton::populateReduceOpToGCUPatterns(converter, patterns, userAnalysis,
                                              replaced2Origin, pTagPool);
  mlir::triton::populateScanOpToGCUPatterns(converter, patterns, userAnalysis,
                                            replaced2Origin, pTagPool);
  mlir::triton::populateElementwiseFusionOpToGCUPatterns(
      converter, patterns, userAnalysis, replaced2Origin, pTagPool);
  mlir::triton::populateMakeRangeOpToGCUPatterns(
      converter, patterns, userAnalysis, replaced2Origin, pTagPool);
  mlir::triton::populateTTSmemOpToGCUPatterns(converter, patterns, userAnalysis,
                                              replaced2Origin, pTagPool);
#ifdef ENABLE_TRITON_DISTRIBUTED
  mlir::triton::populateDistributedOpToGCUPatterns(
      converter, patterns, userAnalysis, replaced2Origin, pTagPool);
#endif
  mlir::triton::populateTleOpToGCUPatterns(
      converter, patterns, target, userAnalysis, replaced2Origin, pTagPool);

  patterns.add<
      TTFuncOpLowering, TTReturnOpLowering, TTCallOpLowering,
      TTSCFForOpLowering, TTSCFIfOpLowering, TTSCFWhileOpLowering,
      TTSCFConditionLowering,
      TTIntrinsicOpLowering<triton::GetNumProgramsOp, gpu::GridDimOp>,
      TTIntrinsicOpLowering<triton::GetProgramIdOp, gpu::BlockIdOp>,
      TTPrintOpLowering, TTAssertOpLowering, TTSplatOpLowering,
      TTAddPtrOpLowering, TTConstantOpLowering, TTReduceReturnOpLowering,
      TTScanReturnOpLowering, TTExternElemwiseOpLowering,
      TTElementwiseOpLowering<triton::PtrToIntOp, gcu::PtrToIntOp>,
      TTElementwiseOpLowering<triton::IntToPtrOp, gcu::IntToPtrOp>,
      TTElementwiseOpLowering<triton::gcu::PtrToIntOp, gcu::PtrToIntOp>,
      TTElementwiseOpLowering<triton::gcu::IntToPtrOp, gcu::IntToPtrOp>,
      TTElementwiseOpLowering<triton::MulhiUIOp, math_ext::UmulhiOp>,
      TTArithSelectOpLowering, TTBitcastOpLowering, TTBroadcastOpLowering,
      TTCatOpLowering, TTHistogramOpLowering, TTExpandDimsOpLowering,
      TTReshapeOpLowering, TTSplitOpLowering, TTJoinOpLowering,
      GCUMatmulLowering, TTUnsplatOpLowering, TTGAssertOpLowering,
      TTTransOpLowering, TTGConvertLayoutOpLowering, GCULoadOpLowering,
      GCUStoreOpLowering, TTDotOpLowering, TTAtomicRMWOpLowering,
      TTAtomicCASOpLowering, TTInitBarrierOpLowering, TTInitPipelineOpLowering,
      TTBarrierPipelineOpLowering<triton::gcu::WaitBarrierOp,
                                  gcu::WaitBarrierOp>,
      TTBarrierPipelineOpLowering<triton::gcu::ArriveBarrierOp,
                                  gcu::ArriveBarrierOp>,
      TTBarrierPipelineOpLowering<triton::gcuws::ProducerAcquireOp,
                                  gcu::ProducerAcquireOp>,
      TTBarrierPipelineOpLowering<triton::gcuws::ProducerCommitOp,
                                  gcu::ProducerCommitOp>,
      TTBarrierPipelineOpLowering<triton::gcuws::ConsumerWaitOp,
                                  gcu::ConsumerWaitOp>,
      TTBarrierPipelineOpLowering<triton::gcuws::ConsumerReleaseOp,
                                  gcu::ConsumerReleaseOp>,
      TTGWarpSpecializeOpLowering, TTGWarpSpecializePartitionsOpLowering,
      TTGWarpYieldOpLowering, TTGWarpReturnOpLowering>(
      converter, ctx, userAnalysis, replaced2Origin, pTagPool);

  patterns.add<TTSCFYieldOpLowering>(converter, ctx, userAnalysis,
                                     replaced2Origin, pTagPool,
                                     TTYeiledOPerandHasMultiUseStage);
  target.addLegalDialect<
      gpu::GPUDialect, gcu::GCUDialect, arith::ArithDialect,
      affine::AffineDialect, func::FuncDialect, scf::SCFDialect,
      math::MathDialect, vector::VectorDialect, memref::MemRefDialect,
      memref_ext::MemrefExtDialect, math_ext::MathExtDialect>();
  target.addIllegalDialect<triton::TritonDialect, triton::gpu::TritonGPUDialect,
                           triton::gcuws::GCUWSDialect>();
  target.addIllegalOp<
      mlir::triton::gcu::ElementwiseFusionRegionOp, mlir::triton::gcu::YieldOp,
      mlir::triton::gcu::LoadOp, mlir::triton::gcu::StoreOp,
      mlir::triton::gcu::CopyGlobalToLocalOp,
      mlir::triton::gcu::GatherGlobalToLocalOp,
      mlir::triton::gcu::InitBarrierOp, mlir::triton::gcu::WaitBarrierOp,
      mlir::triton::gcu::ArriveBarrierOp, mlir::triton::gcu::SliceFromLocalOp,
      mlir::triton::gcu::DesliceToLocalOp>();
  target.addDynamicallyLegalOp(OperationName("tle.extract_tile", &getContext()),
                               [](Operation *) { return false; });
  target.addDynamicallyLegalOp(OperationName("tle.insert_tile", &getContext()),
                               [](Operation *) { return false; });
  target.addDynamicallyLegalOp(
      OperationName("tle.exclusive_cumsum", &getContext()),
      [](Operation *) { return false; });
  target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect,
                                    scf::SCFDialect>([](Operation *op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](auto t) {
                           return isa<TensorType, triton::PointerType,
                                      triton::gpu::MemDescType,
                                      triton::gpu::AsyncTokenType>(t);
                         }) &&
           llvm::none_of(op->getResultTypes(), [](auto t) {
             return isa<TensorType, triton::PointerType,
                        triton::gpu::MemDescType, triton::gpu::AsyncTokenType>(
                 t);
           });
  });

  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
    signalPassFailure();

  if (hasCrossCTAShare) {
    moduleOp->setAttr("gcu.UseCrossCTAShare", UnitAttr::get(ctx));
  }

  // Post-conversion fixup: redirect uses of OACC for-loop results to their
  // local buffer copies. During dialect conversion, replaceAllUsesExcept in
  // TTDotOpLowering cannot update the framework's value mapping, so ops
  // converted after the dot (e.g. ElementwiseFusionRegionOp) still reference
  // the raw OACC for-result. Now that all ops are materialized, we can
  // reliably replace those uses with the matrix_store destination buffer.
  moduleOp.walk([](gcu::MatrixStoreOp matStoreOp) {
    Value src = matStoreOp.getValue();
    auto opResult = dyn_cast<OpResult>(src);
    if (!opResult || !isa<scf::ForOp>(opResult.getOwner()))
      return;
    auto memref2ptr = matStoreOp.getPtr().getDefiningOp<gcu::MemRefToPtrOp>();
    if (!memref2ptr)
      return;
    Value dotOut = memref2ptr.getMemref();
    if (src == dotOut)
      return;
    src.replaceAllUsesExcept(dotOut, matStoreOp);
  });
}
