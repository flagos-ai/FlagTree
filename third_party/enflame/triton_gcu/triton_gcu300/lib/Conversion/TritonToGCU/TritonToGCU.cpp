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

#include "Analysis/FirstLastUserAnalysis.h"
#include "Conversion/TritonToGCU/TritonToGCUPass.h"

#include "PatternTritonGPUOpToGCU.h"
#include "Utils.h"

#include "ConstantUtil.h"
#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/GCU/IR/Types.h"
#include "Dialect/MathExt/IR/MathExt.h"
#include "Dialect/MathExt/IR/MathExtTypes.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"
#include "TritonGCUToGCU/TritionToGCUBase.h"
#include "TritonGCUToGCU/TritonGCUAsyncOpToGCU.h"
#include "TritonGCUToGCU/TritonGCUToGCUUtils.h"

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
    // Remap proper input types.
    TypeConverter::SignatureConversion signatureConversion(
        ttFuncOp.front().getNumArguments());

    // Convert argument types one by one and check for errors.
    for (auto [idx, type] :
         llvm::enumerate(ttFuncOp.getFunctionType().getInputs())) {
      SmallVector<Type, 8> converted;
      converted.push_back(getTypeConverter()->convertType(type));
      signatureConversion.addInputs(idx, converted);
    }
    SmallVector<Type, 4> resultTypes;
    for (auto type : ttFuncOp.getFunctionType().getResults()) {
      resultTypes.push_back(getTypeConverter()->convertType(type));
    }

    auto funcType = FunctionType::get(
        getContext(), signatureConversion.getConvertedTypes(), resultTypes);
    auto funcName = ttFuncOp.isPublic()
                        ? (ttFuncOp.getName() + "_triton_internal__").str()
                        : ttFuncOp.getName().str();
    auto func = rewriter.create<func::FuncOp>(loc, funcName, funcType);
    func.getBody().getBlocks().clear();
    func.setPrivate();
    auto internalLinkage = mlir::LLVM::linkage::Linkage::Internal;
    auto linkage = mlir::LLVM::LinkageAttr::get(getContext(), internalLinkage);
    func->setAttr("llvm.linkage", linkage);
    // Move the region to the new function, update the entry block signature.
    rewriter.inlineRegionBefore(ttFuncOp.getBody(), func.getBody(), func.end());
    if (failed(rewriter.convertRegionTypes(&func.getBody(), *getTypeConverter(),
                                           &signatureConversion)))
      return failure();

    if (ttFuncOp.isPublic()) {
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
    }

    rewriter.eraseOp(ttFuncOp);
    return success();
  }
};

struct TTReturnOpLowering : SharedConversionPattern<triton::ReturnOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (returnOp->getParentOfType<gpu::GPUFuncOp>()) {
      rewriter.replaceOpWithNewOp<gpu::ReturnOp>(returnOp,
                                                 returnOp.getOperands());
    } else {
      rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp,
                                                  returnOp.getOperands());
    }
    return success();
  }
};

struct TTCallOpLowering : SharedConversionPattern<triton::CallOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 4> resultTypes;
    for (auto ty : callOp->getResultTypes()) {
      resultTypes.push_back(getTypeConverter()->convertType(ty));
    }
    rewriter.replaceOpWithNewOp<func::CallOp>(
        callOp, callOp.getCallee(), resultTypes, adaptor.getOperands());
    return success();
  }
};

struct TTSCFForOpLowering : SharedConversionPattern<scf::ForOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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
      std::map<Operation *, std::map<uint64_t, bool>> &operendStage)
      : SharedConversionPattern(converter, ctx, userAnalysis, replaced2Origin),
        TTYeiledOPerandHasMultiUseStage(operendStage) {}

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
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

          auto originParent = replaced2Origin[parent];
          auto lastUser = userAnalysis.getLastUserOp(originParent);
          auto newAllocOpPos =
              promoteLastUser(lastUser, userAnalysis, replaced2Origin);

          Operation *allocOp = definingOp;
          while (allocOp && !mlir::isa<memref::AllocOp>(allocOp)) {
            mlir::TypeSwitch<mlir::Operation *>(allocOp)
                .Case<memref::ReinterpretCastOp, memref::MemorySpaceCastOp>(
                    [&](auto castOp) {
                      allocOp = castOp.getSource().getDefiningOp();
                    })
                .Default([&](auto op) { allocOp = nullptr; });
          }
          if (!allocOp) {
            LLVM_DEBUG({
              llvm::dbgs() << "can't find allocOp in the same region\n";
              allocOp->dump();
            });
            continue;
          }
          if (newAllocOpPos == nullptr) {
            allocOp->moveBefore(parent);
          } else {
            allocOp->moveBefore(newAllocOpPos);
          }
          addDeallocAfterLastUser(rewriter, lastUser, allocOp->getResult(0));

          continue;
        }

        auto tag = getPrivateDTETag(rewriter, op);
        auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto shape = dyn_cast<MemRefType>(operand.getType()).getShape();
        auto size = std::accumulate(shape.begin(), shape.end(), 1,
                                    std::multiplies<int64_t>());
        if (isa<scf::ForOp, scf::WhileOp>(parent)) {
          if (replaced2Origin.count(parent) == 0) {
            llvm_unreachable("can't find the origin op");
          }
          auto originParent = replaced2Origin[parent];
          auto lastUser = userAnalysis.getLastUserOp(originParent);

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
              rewriter.create<arith::ConstantIndexOp>(loc, size), tag,
              ValueRange{zero});
          rewriter.create<memref::DmaWaitOp>(
              loc, tag, ValueRange{zero},
              rewriter.create<arith::ConstantIndexOp>(loc, size));

          if (isa_and_nonnull<memref::AllocOp>(definingOp)) {
            rewriter.create<memref::DeallocOp>(loc, definingOp->getResult(0));
          }
          updatedOperands.push_back(nextLoopTensor);
        } else {
          auto nextLoopTensor = rewriter.create<memref::AllocOp>(
              loc, dyn_cast<MemRefType>(operand.getType()));
          rewriter.create<memref::DmaStartOp>(
              loc, operand, SmallVector<Value, 4>(shape.size(), zero),
              nextLoopTensor, SmallVector<Value, 4>(shape.size(), zero),
              rewriter.create<arith::ConstantIndexOp>(loc, size), tag,
              ValueRange{zero});
          rewriter.create<memref::DmaWaitOp>(
              loc, tag, ValueRange{zero},
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
                       if (hex) {
                         os << "0x%x ";
                         return "0x%x ";
                       } else {
                         if (isSigned) {
                           os << "%d ";
                           return "%d ";
                         }
                         os << "%u ";
                         return "%u ";
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

struct TTMakeRangeOpLowering : SharedConversionPattern<triton::MakeRangeOp> {
  unsigned vectorLengthInByte;
  TTMakeRangeOpLowering(const TypeConverter &converter, MLIRContext *ctx,
                        triton::gcu::FirstLastUserAnalysis &userAnalysis,
                        std::map<Operation *, Operation *> &replaced2Origin,
                        unsigned vectorLength, unsigned vectorizationMaxLength)
      : SharedConversionPattern(converter, ctx, userAnalysis, replaced2Origin),
        vectorLengthInByte(vectorLength) {}

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp arangeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, arangeOp.getOperation());
    auto loc = arangeOp.getLoc();
    auto lastUser = userAnalysis.getLastUserOp(arangeOp.getOperation());
    auto warpIds = getWarpIds(rewriter, loc, arangeOp.getType());
    auto slicedAxies = getSlicedAxies(arangeOp.getType());
    auto numElems = triton::gcu::getTotalElemsPerThread(arangeOp.getType());
    auto start = arangeOp.getStart();
    auto resultType = dyn_cast<MemRefType>(
        getTypeConverter()->convertType(arangeOp.getType()));
    auto output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                              replaced2Origin, resultType);
    auto startOffset =
        slicedAxies.empty()
            ? rewriter
                  .create<arith::ConstantIntOp>(loc, start,
                                                resultType.getElementType())
                  .getResult()
            : rewriter.create<arith::IndexCastOp>(
                  loc, resultType.getElementType(),
                  rewriter.create<arith::AddIOp>(
                      loc,
                      rewriter.create<arith::MulIOp>(
                          loc, warpIds.front(),
                          rewriter.create<arith::ConstantIndexOp>(loc,
                                                                  numElems)),
                      rewriter.create<arith::ConstantIndexOp>(loc, start)));

    auto vectorLength =
        vectorLengthInByte / triton::gcu::getBpe(resultType.getElementType());

    auto vectorType = VectorType::get(ArrayRef<int64_t>{vectorLength},
                                      resultType.getElementType());
    auto arangeV =
        rewriter
            .create<gcu::VectorConvertOp>(
                loc, vectorType,
                rewriter
                    .create<vector::StepOp>(
                        loc, VectorType::get(ArrayRef<int64_t>{vectorLength},
                                             rewriter.getIndexType()))
                    .getResult())
            .getResult(0);

    Value vec = rewriter.create<arith::AddIOp>(
        loc, arangeV,
        rewriter.create<vector::BroadcastOp>(loc, vectorType, startOffset));
    Value step = rewriter.create<vector::BroadcastOp>(
        loc, vectorType,
        rewriter.create<arith::ConstantIntOp>(loc, vectorLength,
                                              resultType.getElementType()));
    rewriter.create<scf::ForOp>(
        loc, rewriter.create<arith::ConstantIndexOp>(loc, 0),
        rewriter.create<arith::ConstantIndexOp>(loc, numElems),
        rewriter.create<arith::ConstantIndexOp>(loc, vectorLength),
        ValueRange{vec},
        [&](OpBuilder &builder, Location loc, Value iters,
            ValueRange iterArgs) {
          builder.create<vector::StoreOp>(loc, iterArgs[0], output, iters);
          builder.create<scf::YieldOp>(
              loc, ValueRange{
                       builder.create<arith::AddIOp>(loc, iterArgs[0], step)});
        });
    leaveTritionOp(rewriter, arangeOp.getOperation());
    rewriter.replaceOp(arangeOp, output);
    return success();
  }
};

struct TTSplatOpLowering : SharedConversionPattern<triton::SplatOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp splatOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, splatOp.getOperation());
    auto lastUser = userAnalysis.getLastUserOp(splatOp.getOperation());
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
    doMemset(rewriter, splatOp, output, v, totalNumElems);
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
    auto loc = constOp.getLoc();
    if (!isa<TensorType>(constOp.getType()))
      return failure();
    auto lastUser = userAnalysis.getLastUserOp(constOp.getOperation());
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
    doMemset(rewriter, constOp, output, v, totalNumElems);
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
    auto loc = addPtrOp.getLoc();
    enterTritionOp(rewriter, addPtrOp.getOperation());
    // vector
    if (isa<TensorType>(addPtrOp.getType())) {
      auto lastUser = userAnalysis.getLastUserOp(addPtrOp.getOperation());
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
                loc, rhs,
                builder.create<arith::ConstantIntOp>(loc, elemBytes,
                                                     rhs.getType()));
            auto v = builder.create<arith::AddIOp>(
                loc, lhs,
                rhs.getType().getIntOrFloatBitWidth() < 64
                    ? builder.create<arith::ExtSIOp>(loc, builder.getI64Type(),
                                                     rhs)
                    : rhs);
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
    offset =
        rewriter.create<arith::MulIOp>(loc, offset,
                                       rewriter.create<arith::ConstantIntOp>(
                                           loc, elemBytes, offset.getType()));
    auto v = rewriter.create<gcu::IntToPtrOp>(
        loc, resultType,
        rewriter.create<arith::AddIOp>(
            loc, rewriter.create<gcu::PtrToIntOp>(loc, ptr),
            offset.getType().getIntOrFloatBitWidth() < 64
                ? rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(),
                                                  offset)
                : offset));
    rewriter.replaceOp(addPtrOp, v);
    return success();
  }
};

struct TTLoadOpLowering : SharedConversionPattern<triton::LoadOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult loadSingleElement(triton::LoadOp loadOp, OpBuilder &builder,
                                  Value ptr, Value output, Value offset,
                                  Value tag, Value mask, Value other) const {
    auto loc = loadOp.getLoc();

    auto elemType = dyn_cast<gcu::PtrType>(ptr.getType()).getElementType();

    auto memType1D =
        MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic}, elemType);
    auto buffer = builder.create<gcu::PtrToMemRefOp>(loc, memType1D, ptr);
    auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);

    auto from = builder.create<memref::ReinterpretCastOp>(
        loc, MemRefType::get(ArrayRef<int64_t>{1}, elemType), buffer, 0,
        ArrayRef<int64_t>{1}, ArrayRef<int64_t>{1});

    auto to = builder.create<memref::ReinterpretCastOp>(
        loc, MemRefType::get(ArrayRef<int64_t>{1}, elemType), output, offset,
        ValueRange{one}, ValueRange{one});
    auto result = success();
    builder.create<scf::IfOp>(
        loc, mask,
        [&](OpBuilder &builder, Location loc) {
          builder.create<memref::DmaStartOp>(loc, from, ValueRange{zero}, to,
                                             ValueRange{zero}, one, tag,
                                             ValueRange{zero});
          builder.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero}, one);
          builder.create<scf::YieldOp>(loc);
        },
        [&](OpBuilder &builder, Location loc) {
          builder.create<memref::StoreOp>(loc, other, to, ValueRange{offset});
          doMemFence(builder, loadOp);
          builder.create<scf::YieldOp>(loc);
        });
    return result;
  }

  LogicalResult
  matchAndRewrite(triton::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, loadOp.getOperation());
    auto loc = loadOp.getLoc();
    assert(!(isa<triton::PointerType>(loadOp.getPtr().getType()) &&
             isa<RankedTensorType>(
                 dyn_cast<triton::PointerType>(loadOp.getPtr().getType())
                     .getPointeeType())));

    // tensor
    if (isa<TensorType>(loadOp.getType())) {
      auto lastUser = userAnalysis.getLastUserOp(loadOp.getOperation());
      auto numElems = triton::gcu::getElemsPerThread(loadOp.getType());
      auto numElemValues = getElemsPerThread(rewriter, loc, loadOp.getType());

      auto resultType = dyn_cast<MemRefType>(
          getTypeConverter()->convertType(loadOp.getType()));
      auto output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                replaced2Origin, resultType);
      auto offsets = syncAllocOp(
          rewriter, loc, loadOp.getOperation(), userAnalysis, replaced2Origin,
          MemRefType::get(resultType.getShape(), rewriter.getI32Type()));
      auto masks = syncAllocOp(
          rewriter, loc, loadOp.getOperation(), userAnalysis, replaced2Origin,
          MemRefType::get(resultType.getShape(), rewriter.getI1Type()));
      auto others = syncAllocOp(
          rewriter, loc, loadOp.getOperation(), userAnalysis, replaced2Origin,
          MemRefType::get(resultType.getShape(), resultType.getElementType()));

      auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto firstIndex = SmallVector<Value, 4>(numElems.size(), zero);
      auto firstAddr =
          rewriter.create<memref::LoadOp>(loc, adaptor.getPtr(), firstIndex);
      scf::buildLoopNest(
          rewriter, loc, SmallVector<Value, 4>(numElems.size(), zero),
          numElemValues, SmallVector<Value, 4>(numElems.size(), one),
          [&](OpBuilder &builder, Location loc, ValueRange iters) {
            auto addr =
                builder.create<memref::LoadOp>(loc, adaptor.getPtr(), iters);
            auto offset = builder.create<arith::SubIOp>(loc, addr, firstAddr);
            builder.create<memref::StoreOp>(
                loc,
                builder.create<arith::TruncIOp>(loc, builder.getI32Type(),
                                                offset),
                offsets, iters);

            auto mask =
                adaptor.getMask()
                    ? builder
                          .create<memref::LoadOp>(loc, adaptor.getMask(), iters)
                          .getResult()
                    : builder
                          .create<arith::ConstantIntOp>(loc, 1,
                                                        builder.getI1Type())
                          .getResult();
            builder.create<memref::StoreOp>(loc, mask, masks, iters);

            auto other = adaptor.getOther()
                             ? rewriter
                                   .create<memref::LoadOp>(
                                       loc, adaptor.getOther(), iters)
                                   .getResult()
                             : triton::gcu::createConstantZero(
                                   rewriter, loc, resultType.getElementType());
            builder.create<memref::StoreOp>(loc, other, others, iters);
          });

      auto totalNumElems =
          rewriter.create<arith::ConstantIndexOp>(loc, 1).getResult();
      for (unsigned i = 0; i < numElemValues.size(); ++i) {
        totalNumElems = rewriter.create<arith::MulIOp>(loc, totalNumElems,
                                                       numElemValues[i]);
      }

      auto output1D = castToMemref1D(rewriter, loc, output, totalNumElems);
      auto offsets1D = castToMemref1D(rewriter, loc, offsets, totalNumElems);
      auto masks1D = castToMemref1D(rewriter, loc, masks, totalNumElems);
      auto others1D = castToMemref1D(rewriter, loc, others, totalNumElems);
      rewriter.create<gcu::GatherLoadOp>(
          loc,
          rewriter.create<gcu::MemRefToPtrOp>(
              loc, gcu::PtrType::get(getContext(), resultType.getElementType()),
              output1D),
          rewriter.create<gcu::IntToPtrOp>(
              loc, gcu::PtrType::get(getContext(), resultType.getElementType()),
              firstAddr),
          rewriter.create<gcu::MemRefToPtrOp>(
              loc, gcu::PtrType::get(getContext(), rewriter.getI32Type()),
              offsets1D),
          rewriter.create<gcu::MemRefToPtrOp>(
              loc, gcu::PtrType::get(getContext(), rewriter.getI1Type()),
              masks1D),
          rewriter.create<gcu::MemRefToPtrOp>(
              loc, gcu::PtrType::get(getContext(), resultType.getElementType()),
              others1D),
          rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                              totalNumElems));
      leaveTritionOp(rewriter, loadOp.getOperation());
      rewriter.replaceOp(loadOp, output);
      return success();
    }

    // scalar
    auto tag = getPrivateDTETag(rewriter, loadOp);
    auto output = rewriter.create<memref::AllocOp>(
        loc,
        MemRefType::get(ArrayRef<int64_t>{1},
                        getTypeConverter()->convertType(loadOp.getType())));
    auto offset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto mask =
        adaptor.getMask()
            ? adaptor.getMask()
            : rewriter
                  .create<arith::ConstantIntOp>(loc, 1, rewriter.getI1Type())
                  .getResult();
    auto other =
        adaptor.getOther()
            ? adaptor.getOther()
            : triton::gcu::createConstantZero(rewriter, loc, loadOp.getType());
    if (failed(loadSingleElement(loadOp, rewriter, adaptor.getPtr(), output,
                                 offset, tag, mask, other)))
      return failure();
    auto v = rewriter.create<memref::LoadOp>(loc, output, ValueRange{offset});
    leaveTritionOp(rewriter, loadOp.getOperation());
    rewriter.replaceOp(loadOp, v);
    return success();
  }
};

struct TTStoreOpLowering : SharedConversionPattern<triton::StoreOp> {
  using SharedConversionPattern::SharedConversionPattern;

  void storeSingleElement(triton::StoreOp storeOp, OpBuilder &builder,
                          Value ptr, Value values, Value offset, Value tag,
                          Value mask) const {
    auto elemType = dyn_cast<gcu::PtrType>(ptr.getType()).getElementType();
    auto loc = storeOp.getLoc();

    auto memType1D =
        MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic}, elemType);
    auto buffer = builder.create<gcu::PtrToMemRefOp>(loc, memType1D, ptr);
    auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);

    auto from = builder.create<memref::ReinterpretCastOp>(
        loc, MemRefType::get(ArrayRef<int64_t>{1}, elemType), values, offset,
        ValueRange{one}, ValueRange{one});
    auto to = builder.create<memref::ReinterpretCastOp>(
        loc, MemRefType::get(ArrayRef<int64_t>{1}, elemType), buffer, 0,
        ArrayRef<int64_t>{1}, ArrayRef<int64_t>{1});
    builder.create<scf::IfOp>(loc, mask, [&](OpBuilder &builder, Location loc) {
      builder.create<memref::DmaStartOp>(loc, from, ValueRange{zero}, to,
                                         ValueRange{zero}, one, tag,
                                         ValueRange{zero});
      builder.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero}, one);
      builder.create<scf::YieldOp>(loc);
    });
  }

  LogicalResult
  matchAndRewrite(triton::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = storeOp.getLoc();
    enterTritionOp(rewriter, storeOp.getOperation());
    assert(!(isa<triton::PointerType>(storeOp.getPtr().getType()) &&
             isa<RankedTensorType>(
                 dyn_cast<triton::PointerType>(storeOp.getPtr().getType())
                     .getPointeeType())));

    // tensor
    if (isa<TensorType>(storeOp.getPtr().getType())) {
      auto numElems =
          triton::gcu::getElemsPerThread(storeOp.getPtr().getType());
      auto numElemValues =
          getElemsPerThread(rewriter, loc, storeOp.getPtr().getType());
      auto values = adaptor.getValue();
      auto valueType = dyn_cast<MemRefType>(values.getType());

      auto offsets = syncAllocOp(
          rewriter, loc, storeOp.getOperation(), userAnalysis, replaced2Origin,
          MemRefType::get(valueType.getShape(), rewriter.getI32Type()));
      auto masks = syncAllocOp(
          rewriter, loc, storeOp.getOperation(), userAnalysis, replaced2Origin,
          MemRefType::get(valueType.getShape(), rewriter.getI1Type()));

      auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto firstIndex = SmallVector<Value, 4>(numElems.size(), zero);
      auto firstAddr =
          rewriter.create<memref::LoadOp>(loc, adaptor.getPtr(), firstIndex);

      scf::buildLoopNest(
          rewriter, loc, SmallVector<Value, 4>(numElems.size(), zero),
          numElemValues, SmallVector<Value, 4>(numElems.size(), one),
          [&](OpBuilder &builder, Location loc, ValueRange iters) {
            auto addr =
                builder.create<memref::LoadOp>(loc, adaptor.getPtr(), iters);
            auto offset = builder.create<arith::SubIOp>(loc, addr, firstAddr);
            builder.create<memref::StoreOp>(
                loc,
                builder.create<arith::TruncIOp>(loc, builder.getI32Type(),
                                                offset),
                offsets, iters);

            auto mask =
                adaptor.getMask()
                    ? builder
                          .create<memref::LoadOp>(loc, adaptor.getMask(), iters)
                          .getResult()
                    : builder
                          .create<arith::ConstantIntOp>(loc, 1,
                                                        builder.getI1Type())
                          .getResult();
            builder.create<memref::StoreOp>(loc, mask, masks, iters);
          });

      Value totalNumElems =
          rewriter.create<arith::ConstantIndexOp>(loc, 1).getResult();
      for (unsigned i = 0; i < numElemValues.size(); ++i) {
        totalNumElems = rewriter.create<arith::MulIOp>(loc, totalNumElems,
                                                       numElemValues[i]);
      }

      auto values1D = castToMemref1D(rewriter, loc, values, totalNumElems);
      auto offsets1D = castToMemref1D(rewriter, loc, offsets, totalNumElems);
      auto masks1D = castToMemref1D(rewriter, loc, masks, totalNumElems);

      rewriter.replaceOpWithNewOp<gcu::ScatterStoreOp>(
          storeOp,
          rewriter.create<gcu::IntToPtrOp>(
              loc, gcu::PtrType::get(getContext(), valueType.getElementType()),
              firstAddr),
          rewriter.create<gcu::MemRefToPtrOp>(
              loc, gcu::PtrType::get(getContext(), valueType.getElementType()),
              values1D),
          rewriter.create<gcu::MemRefToPtrOp>(
              loc, gcu::PtrType::get(getContext(), rewriter.getI32Type()),
              offsets1D),
          rewriter.create<gcu::MemRefToPtrOp>(
              loc, gcu::PtrType::get(getContext(), rewriter.getI1Type()),
              masks1D),
          rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                              totalNumElems));
      return success();
    }

    // scalar
    auto tag = getPrivateDTETag(rewriter, storeOp);
    auto output = rewriter.create<memref::AllocOp>(
        loc,
        MemRefType::get(ArrayRef<int64_t>{1}, adaptor.getValue().getType()));
    auto offset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    rewriter.create<memref::StoreOp>(loc, adaptor.getValue(), output,
                                     ValueRange{offset});

    // If the tensor is not ranked, then it is a scalar and only thread 0 can
    // write
    auto oneMask =
        rewriter.create<arith::ConstantIntOp>(loc, 1, rewriter.getI1Type())
            .getResult();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto isThread0 = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
    auto mask = adaptor.getMask()
                    ? adaptor.getMask()
                    : rewriter.create<arith::AndIOp>(loc, oneMask, isThread0);
    doMemFence(rewriter, storeOp);
    storeSingleElement(storeOp, rewriter, adaptor.getPtr(), output, offset, tag,
                       mask);
    rewriter.create<memref::DeallocOp>(loc, output);
    leaveTritionOp(rewriter, storeOp.getOperation());
    rewriter.eraseOp(storeOp);
    return success();
  }
};

struct TTArithSelectOpLowering
    : public SharedConversionPattern<arith::SelectOp> {
  TTArithSelectOpLowering(const TypeConverter &converter, MLIRContext *ctx,
                          triton::gcu::FirstLastUserAnalysis &userAnalysis,
                          std::map<Operation *, Operation *> &replaced2Origin)
      : SharedConversionPattern<arith::SelectOp>(converter, ctx, userAnalysis,
                                                 replaced2Origin) {}

  LogicalResult matchAndRewrite(
      arith::SelectOp op,
      typename SharedConversionPattern<arith::SelectOp>::OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
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
  TTElementwiseOpLowering(const TypeConverter &converter, MLIRContext *ctx,
                          triton::gcu::FirstLastUserAnalysis &userAnalysis,
                          std::map<Operation *, Operation *> &replaced2Origin)
      : SharedConversionPattern<FT>(converter, ctx, userAnalysis,
                                    replaced2Origin) {}

  LogicalResult
  matchAndRewrite(FT op,
                  typename SharedConversionPattern<FT>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    auto loc = op.getLoc();
    auto type = op.getType();
    if (!isa<TensorType>(type)) {
      auto ty = this->getTypeConverter()->convertType(type);
      rewriter.replaceOpWithNewOp<TT>(op, ty, adaptor.getOperands());
      leaveTritionOp(rewriter, op.getOperation());
      return success();
    }
    auto lastUser = this->userAnalysis.getLastUserOp(op.getOperation());
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
    auto loc = op.getLoc();
    enterTritionOp(rewriter, op.getOperation());
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

struct TTScanOpLowering : SharedConversionPattern<triton::ScanOp> {
  unsigned vectorSizeInBytes;
  TTScanOpLowering(const TypeConverter &converter, MLIRContext *ctx,
                   triton::gcu::FirstLastUserAnalysis &userAnalysis,
                   std::map<Operation *, Operation *> &replaced2Origin,
                   unsigned vectorSizeInBytes)
      : SharedConversionPattern(converter, ctx, userAnalysis, replaced2Origin),
        vectorSizeInBytes(vectorSizeInBytes) {}

  void applyScan(triton::ScanOp op, OpBuilder &rewriter,
                 ArrayRef<Value> outputs, ArrayRef<Value> inputs, Type type,
                 bool reverse) const {
    auto axis = op.getAxis();
    auto loc = op.getLoc();
    auto numElems = triton::gcu::getElemsPerThread(type);
    auto numOutput = outputs.size();
    auto totalNumElems = triton::gcu::getTotalElemsPerThread(type);
    auto tag = getPrivateDTETag(rewriter, op);
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // initialize outputs by inputs
    for (unsigned i = 0; i < numOutput; ++i) {
      rewriter.create<memref::DmaStartOp>(
          loc, inputs[i], SmallVector<Value, 4>(numElems.size(), zero),
          outputs[i], SmallVector<Value, 4>(numElems.size(), zero),
          rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems), tag,
          ValueRange{zero});
      rewriter.create<memref::DmaWaitOp>(
          loc, tag, ValueRange{zero},
          rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));
    }

    std::array<int64_t, 3> scanInOutDims = {1, 1, 1};
    int64_t scanAxis = 2;
    for (int i = numElems.size() - 1, j = 2; i >= 0; i--) {
      if (static_cast<unsigned>(i) == axis) {
        if (scanInOutDims[j] == 1) {
          scanInOutDims[j] = numElems[i];
        } else {
          scanInOutDims[--j] = numElems[i];
        }
        scanAxis = j;
        --j;
      } else {
        scanInOutDims[j] *= numElems[i];
      }
    }
    SmallVector<Value, 4> outs;
    llvm::transform(outputs, std::back_inserter(outs), [&](auto output) {
      return rewriter.create<memref::ReinterpretCastOp>(
          loc,
          MemRefType::get(scanInOutDims,
                          cast<MemRefType>(output.getType()).getElementType()),
          output, ValueRange{}, ValueRange{}, ValueRange{},
          ArrayRef<int64_t>{0},
          ArrayRef<int64_t>{scanInOutDims[0], scanInOutDims[1],
                            scanInOutDims[2]},
          ArrayRef<int64_t>{scanInOutDims[1] * scanInOutDims[2],
                            scanInOutDims[2], 1});
    });
    if (succeeded(applyGeneralScan(op, rewriter, outs, scanInOutDims, scanAxis,
                                   reverse))) {
      return;
    }
    return applyScanFallback(op, rewriter, outs, scanInOutDims, scanAxis,
                             reverse);
  }

  LogicalResult applyGeneralScan(triton::ScanOp op, OpBuilder &rewriter,
                                 ArrayRef<Value> outputs,
                                 const std::array<int64_t, 3> &scanInOutDims,
                                 int64_t scanAxis, bool reverse) const {
    auto loc = op.getLoc();
    int64_t vectorizeAxis;
    if (scanAxis == 2) {
      assert(scanInOutDims[0] == 1);
      vectorizeAxis = 1;
    } else {
      assert(scanAxis == 1);
      vectorizeAxis = scanInOutDims[0] > scanInOutDims[2] ? 0 : 2;
    }
    unsigned bpe = 4; // gatherscatter offset, i32
    for (auto output : outputs) {
      auto elementTy = cast<MemRefType>(output.getType()).getElementType();
      auto bytes = triton::gcu::getBpe(elementTy);
      bpe = bytes > bpe ? bytes : bpe;
    }
    auto vectorLength = vectorSizeInBytes / bpe;
    if (scanInOutDims[vectorizeAxis] < vectorLength) {
      return failure();
    }
    auto numOutput = outputs.size();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<VectorType, 4> vectorTypes;
    llvm::transform(
        outputs, std::back_inserter(vectorTypes), [vectorLength](auto output) {
          auto elementTy = cast<MemRefType>(output.getType()).getElementType();
          return VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);
        });

    SmallVector<Value, 4> lbs(scanInOutDims.size(), zero);
    lbs[scanAxis] = one;
    std::array<int64_t, 3> loopcnt = scanInOutDims;
    if (loopcnt[vectorizeAxis] % vectorLength != 0) {
      llvm_unreachable("invalid datalayout");
    }
    loopcnt[vectorizeAxis] /= vectorLength;
    SmallVector<Value, 4> ubs{
        rewriter.create<arith::ConstantIndexOp>(loc, loopcnt[0]),
        rewriter.create<arith::ConstantIndexOp>(loc, loopcnt[1]),
        rewriter.create<arith::ConstantIndexOp>(loc, loopcnt[2])};
    SmallVector<Value, 4> step(scanInOutDims.size(), one);

    auto maskType =
        VectorType::get(ArrayRef<int64_t>{vectorLength}, rewriter.getI1Type());
    Value mask = rewriter.create<vector::ConstantMaskOp>(
        loc, maskType,
        DenseI64ArrayAttr::get(rewriter.getContext(),
                               ArrayRef<int64_t>{vectorLength}));
    unsigned strideOnVectorizeAxis =
        std::accumulate(scanInOutDims.begin() + vectorizeAxis + 1,
                        scanInOutDims.end(), 1, std::multiplies<unsigned>());
    auto vecTy =
        VectorType::get(ArrayRef<int64_t>{vectorLength}, rewriter.getI32Type());
    auto indexVec = rewriter.create<arith::MulIOp>(
        loc,
        rewriter
            .create<gcu::VectorConvertOp>(
                loc, vecTy,
                rewriter
                    .create<vector::StepOp>(
                        loc, VectorType::get(ArrayRef<int64_t>{vectorLength},
                                             rewriter.getIndexType()))
                    .getResult())
            .getResult(0),
        rewriter.create<vector::BroadcastOp>(
            loc, vecTy,
            rewriter.create<arith::ConstantOp>(
                loc, rewriter.getI32Type(),
                rewriter.getI32IntegerAttr(strideOnVectorizeAxis))));

    SmallVector<Value, 4> passThruValues;
    for (unsigned i = 0; i < numOutput; ++i) {
      passThruValues.push_back(rewriter.create<vector::BroadcastOp>(
          loc, vectorTypes[i],
          rewriter.create<arith::ConstantOp>(
              loc, vectorTypes[i].getElementType(),
              rewriter.getZeroAttr(vectorTypes[i].getElementType()))));
    }

    scf::buildLoopNest(
        rewriter, loc,
        ArrayRef<Value>(lbs.begin(), lbs.begin() + vectorizeAxis),
        ArrayRef<Value>(ubs.begin(), ubs.begin() + vectorizeAxis),
        ArrayRef<Value>(step.begin(), step.begin() + vectorizeAxis),
        [&](OpBuilder &builder, Location loc, ValueRange outerIters) {
          scf::buildLoopNest(
              rewriter, loc,
              ArrayRef<Value>(lbs.begin() + vectorizeAxis, lbs.end()),
              ArrayRef<Value>(ubs.begin() + vectorizeAxis, ubs.end()),
              ArrayRef<Value>(step.begin() + vectorizeAxis, step.end()),
              [&](OpBuilder &builder, Location loc, ValueRange innerIters) {
                SmallVector<Value, 4> inputIndices;
                SmallVector<Value, 4> outputIndices;

                SmallVector<Type, 4> resultElemTypes;
                SmallVector<Value, 4> operands;
                SmallVector<Value, 4> ivs;
                for (auto iv : outerIters) {
                  ivs.push_back(iv);
                }
                for (auto iv : innerIters) {
                  ivs.push_back(iv);
                }
                if (reverse) {
                  ivs[scanAxis] = builder.create<arith::SubIOp>(
                      loc,
                      builder.create<arith::ConstantIndexOp>(
                          loc, scanInOutDims[scanAxis] - 1),
                      ivs[scanAxis]);
                }
                for (unsigned i = 0; i < ivs.size(); ++i) {
                  if (i == vectorizeAxis) {
                    outputIndices.push_back(builder.create<arith::MulIOp>(
                        loc, ivs[i],
                        rewriter.create<arith::ConstantIndexOp>(loc,
                                                                vectorLength)));
                  } else {
                    outputIndices.push_back(ivs[i]);
                  }
                  if (i == scanAxis) {
                    if (reverse) {
                      inputIndices.push_back(builder.create<arith::AddIOp>(
                          loc, outputIndices[i], one));
                    } else {
                      inputIndices.push_back(builder.create<arith::SubIOp>(
                          loc, outputIndices[i], one));
                    }
                  } else {
                    inputIndices.push_back(outputIndices[i]);
                  }
                }

                for (unsigned i = 0; i < numOutput; ++i) {
                  operands.push_back(builder.create<vector::GatherOp>(
                      loc, vectorTypes[i], outputs[i], inputIndices, indexVec,
                      mask, passThruValues[i]));
                }
                for (unsigned i = 0; i < numOutput; ++i) {
                  operands.push_back(builder.create<vector::GatherOp>(
                      loc, vectorTypes[i], outputs[i], outputIndices, indexVec,
                      mask, passThruValues[i]));
                  resultElemTypes.push_back(vectorTypes[i]);
                }

                auto executeRegionOp =
                    builder.create<scf::ExecuteRegionOp>(loc, resultElemTypes);
                executeRegionOp.getRegion().emplaceBlock();
                IRMapping map;
                for (auto [arg, operand] :
                     llvm::zip(op.getCombineOp().getArguments(), operands)) {
                  map.map(arg, operand);
                }
                {
                  OpBuilder::InsertionGuard guard(builder);
                  builder.setInsertionPointToStart(
                      &executeRegionOp.getRegion().back());
                  for (auto &o : op.getCombineOp().back()) {
                    for (auto operand : o.getOperands()) {
                      if (auto constantOp =
                              operand.getDefiningOp<arith::ConstantOp>()) {
                        if (!map.lookupOrNull(operand)) {
                          OpBuilder::InsertionGuard guard(builder);
                          builder.setInsertionPointAfter(constantOp);
                          map.map(operand,
                                  builder.create<vector::BroadcastOp>(
                                      loc,
                                      VectorType::get(
                                          ArrayRef<int64_t>{vectorLength},
                                          operand.getType()),
                                      operand));
                        }
                      }
                    }
                    auto newO = builder.clone(o, map);
                    for (auto [result, newResult] :
                         llvm::zip(o.getResults(), newO->getResults())) {
                      auto vectorTy = VectorType::get(
                          ArrayRef<int64_t>{vectorLength}, result.getType());
                      newResult.setType(vectorTy);
                      map.map(result, newResult);
                    }
                  }
                }

                for (unsigned i = 0; i < numOutput; ++i) {
                  builder.create<vector::ScatterOp>(
                      loc, outputs[i], outputIndices, indexVec, mask,
                      executeRegionOp.getResult(i));
                }
              });
        });
    doMemFence(rewriter, op);
    return success();
  }

  void applyScanFallback(triton::ScanOp op, OpBuilder &rewriter,
                         ArrayRef<Value> outputs,
                         const std::array<int64_t, 3> &scanInOutDims,
                         int64_t scanAxis, bool reverse) const {
    auto loc = op.getLoc();
    auto numOutput = outputs.size();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    SmallVector<Value, 4> lbs(scanInOutDims.size(), zero);
    lbs[scanAxis] = one;
    SmallVector<Value, 4> ubs{
        rewriter.create<arith::ConstantIndexOp>(loc, scanInOutDims[0]),
        rewriter.create<arith::ConstantIndexOp>(loc, scanInOutDims[1]),
        rewriter.create<arith::ConstantIndexOp>(loc, scanInOutDims[2])};

    scf::buildLoopNest(
        rewriter, loc, lbs, ubs,
        SmallVector<Value, 4>(scanInOutDims.size(), one),
        [&](OpBuilder &builder, Location loc, ValueRange iters) {
          SmallVector<Value, 4> outputIters(iters.begin(), iters.end());
          if (reverse) {
            outputIters[scanAxis] = builder.create<arith::SubIOp>(
                loc,
                builder.create<arith::ConstantIndexOp>(
                    loc, scanInOutDims[scanAxis] - 1),
                outputIters[scanAxis]);
          }

          SmallVector<Value, 4> operands;
          SmallVector<Type, 4> resultElemTypes;
          SmallVector<Value, 4> inputIters(outputIters.begin(),
                                           outputIters.end());
          if (reverse) {
            inputIters[scanAxis] =
                builder.create<arith::AddIOp>(loc, one, inputIters[scanAxis]);
          } else {
            inputIters[scanAxis] =
                builder.create<arith::SubIOp>(loc, inputIters[scanAxis], one);
          }

          for (unsigned i = 0; i < numOutput; ++i) {
            operands.push_back(
                builder.create<memref::LoadOp>(loc, outputs[i], inputIters));
          }
          for (unsigned i = 0; i < numOutput; ++i) {
            operands.push_back(
                builder.create<memref::LoadOp>(loc, outputs[i], outputIters));
            resultElemTypes.push_back(operands.back().getType());
          }

          auto executeRegion =
              builder.create<scf::ExecuteRegionOp>(loc, resultElemTypes);
          executeRegion.getRegion().emplaceBlock();
          IRMapping map;
          for (auto [arg, operand] :
               llvm::zip(op.getCombineOp().getArguments(), operands)) {
            map.map(arg, operand);
          }
          {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(&executeRegion.getRegion().back());
            for (auto &o : op.getCombineOp().back()) {
              auto newO = builder.clone(o, map);
              for (auto [result, newResult] :
                   llvm::zip(o.getResults(), newO->getResults())) {
                map.map(result, newResult);
              }
            }
          }

          for (unsigned i = 0; i < numOutput; ++i) {
            builder.create<memref::StoreOp>(loc, executeRegion.getResult(i),
                                            outputs[i], outputIters);
          }
        });

    doMemFence(rewriter, op);
  }

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    enterTritionOp(rewriter, op.getOperation());
    auto inputType = dyn_cast<TensorType>(op.getSrcs()[0].getType());

    auto slicedAxies = getSlicedAxies(inputType);
    bool isScanDimSplit = slicedAxies.count(op.getAxis());

    auto numInput = op.getSrcs().size();
    auto numOutput = op.getResults().size();

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto lastUser = userAnalysis.getLastUserOp(op.getOperation());
    // create outputs
    SmallVector<Value, 4> outputs;
    SmallVector<Type, 4> outputElemTypes;
    for (unsigned i = 0; i < numOutput; ++i) {
      auto resultType =
          dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType(i)));
      auto elemType = resultType.getElementType();
      outputElemTypes.push_back(elemType);
      Value output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                 replaced2Origin, resultType);
      outputs.push_back(output);
    }
    auto encodingAttr = dyn_cast<RankedTensorType>(inputType).getEncoding();
    auto warpsPerCTA = triton::gpu::getWarpsPerCTA(encodingAttr);
    auto threadsPerWarp = triton::gpu::getThreadsPerWarp(encodingAttr);
    auto elementsPerThread = triton::gcu::getElemsPerThread(inputType);
    bool isValidBlockEncoding = true;
    for (auto [dim, elems, threads, warps] :
         llvm::zip(inputType.getShape(), elementsPerThread, threadsPerWarp,
                   warpsPerCTA)) {
      if (dim != elems * threads * warps) {
        isValidBlockEncoding = false;
        break;
      }
    }
    if (isScanDimSplit || !isValidBlockEncoding) {
      auto tag = getPrivateDTETag(rewriter, op);

      // move to shared memory
      SmallVector<Value, 4> sharedInputs;
      for (unsigned i = 0; i < numInput; ++i) {
        sharedInputs.push_back(storeToSharedMem(
            rewriter, tag,
            dyn_cast<RankedTensorType>(op.getSrcs()[i].getType()),
            adaptor.getSrcs()[i], false, op.getOperation(), userAnalysis,
            replaced2Origin));
      }

      // load all shared memory to thread 0
      SmallVector<Value, 4> mergedInputs;
      RankedTensorType mergedInputType;
      for (unsigned i = 0; i < numInput; ++i) {
        auto tType = dyn_cast<RankedTensorType>(op.getSrcs()[i].getType());
        auto tensorType =
            RankedTensorType::get(tType.getShape(), tType.getElementType(),
                                  triton::gpu::getDefaultBlockedEncoding(
                                      getContext(), tType.getShape(), 1, 1, 1));
        mergedInputType = tensorType;
        mergedInputs.push_back(loadFromSharedMem(
            rewriter, tag, tensorType, sharedInputs[i], true, op.getOperation(),
            nullptr, userAnalysis, replaced2Origin));
      }

      SmallVector<Value, 4> mergedOutputs;
      for (unsigned i = 0; i < numOutput; ++i) {
        auto tType = dyn_cast<RankedTensorType>(op.getResultTypes()[i]);
        auto tensorType =
            RankedTensorType::get(tType.getShape(), tType.getElementType(),
                                  triton::gpu::getDefaultBlockedEncoding(
                                      getContext(), tType.getShape(), 1, 1, 1));
        auto resultType =
            dyn_cast<MemRefType>(getTypeConverter()->convertType(tensorType));
        mergedOutputs.push_back(syncAllocOp(rewriter, loc, op.getOperation(),
                                            userAnalysis, replaced2Origin,
                                            resultType));
      }

      // computing in thread 0
      auto isThread0 = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
      rewriter.create<scf::IfOp>(
          loc, isThread0, [&](OpBuilder &builder, Location loc) {
            applyScan(op, builder, mergedOutputs, mergedInputs, mergedInputType,
                      op.getReverse());
            builder.create<scf::YieldOp>(loc);
          });

      // save back to shared memory
      SmallVector<Value, 4> mergedSharedOutputs;
      for (unsigned i = 0; i < numOutput; ++i) {
        auto tType = dyn_cast<RankedTensorType>(op.getResultTypes()[i]);
        auto tensorType =
            RankedTensorType::get(tType.getShape(), outputElemTypes[i],
                                  triton::gpu::getDefaultBlockedEncoding(
                                      getContext(), tType.getShape(), 1, 1, 1));
        mergedSharedOutputs.push_back(
            storeToSharedMem(rewriter, tag, tensorType, mergedOutputs[i], true,
                             op.getOperation(), userAnalysis, replaced2Origin));
      }
      // load from shared memory
      for (unsigned i = 0; i < numOutput; ++i) {
        outputs[i] = loadFromSharedMem(rewriter, tag, op.getResultTypes()[i],
                                       mergedSharedOutputs[i], false, lastUser,
                                       nullptr, userAnalysis, replaced2Origin);
      }
    } else {
      applyScan(op, rewriter, outputs,
                SmallVector<Value, 4>(adaptor.getSrcs().begin(),
                                      adaptor.getSrcs().end()),
                inputType, op.getReverse());
    }

    SmallVector<Value, 4> finalOutputs;
    for (unsigned i = 0; i < numOutput; ++i) {
      auto output = outputs[i];
      auto resultType = dyn_cast<MemRefType>(
          getTypeConverter()->convertType(op.getResultTypes()[i]));
      if (resultType.getNumElements() !=
          dyn_cast<MemRefType>(output.getType()).getNumElements()) {
        return op.emitOpError("element number mismatch");
      }
      auto [strides, offset] = resultType.getStridesAndOffset();
      output = rewriter.create<memref::ReinterpretCastOp>(
          loc, resultType, output, offset, resultType.getShape(), strides);
      finalOutputs.push_back(output);
    }
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, finalOutputs);
    return success();
  }
};

struct TTReduceReturnOpLowering
    : SharedConversionPattern<triton::ReduceReturnOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(returnOp, returnOp.getOperands());
    return success();
  }
};

struct TTScanReturnOpLowering : SharedConversionPattern<triton::ScanReturnOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ScanReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(returnOp, returnOp.getOperands());
    return success();
  }
};

struct TTExternElemwiseOpLowering
    : SharedConversionPattern<triton::ExternElementwiseOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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
    } else if (name == "__gcu_begin_clock") {
      rewriter.replaceOpWithNewOp<gcu::BeginClockOp>(op, adaptor.getOperands());
      return success();
    } else if (name == "__gcu_end_clock") {
      rewriter.replaceOpWithNewOp<gcu::EndClockOp>(op, adaptor.getOperands());
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
    auto loc = histogramOp.getLoc();
    auto zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0));
    auto zeroIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto oneIndex = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    enterTritionOp(rewriter, histogramOp.getOperation());
    auto lastUser = userAnalysis.getLastUserOp(histogramOp.getOperation());
    auto tag = getPrivateDTETag(rewriter, histogramOp);
    auto resultType = histogramOp.getType();
    auto wrapResultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(resultType));
    auto resultMemRefType =
        MemRefType::get(resultType.getShape(), wrapResultType.getElementType());
    auto totalNumElems = triton::gcu::getTotalElemsPerThread(resultType);
    auto resCurWarp = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                  replaced2Origin, resultMemRefType);
    doMemset(rewriter, histogramOp, resCurWarp, zero, totalNumElems);
    auto encoding = resultType.getEncoding();
    auto warpsPerCTA = triton::gpu::getWarpsPerCTA(encoding);
    auto sharedMemTensorType = RankedTensorType::get(
        ArrayRef<int64_t>{resultType.getShape()[0] * warpsPerCTA[0]},
        wrapResultType.getElementType(), encoding);
    rewriter.create<math_ext::HistogramOp>(loc, resCurWarp, adaptor.getSrc());
    /// store res of every warp to shared memry
    auto sharedResMem = storeToSharedMem(
        rewriter, tag, sharedMemTensorType, resCurWarp, false,
        histogramOp.getOperation(), userAnalysis, replaced2Origin);
    rewriter.create<memref::DeallocOp>(loc, resCurWarp);
    size_t allResSize = resultType.getShape()[0];
    size_t warpResSize = wrapResultType.getShape()[0];
    auto finalOutput = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                   replaced2Origin, wrapResultType);
    doMemset(rewriter, histogramOp, finalOutput, zero, totalNumElems);
    size_t warpsWalkNum = warpsPerCTA[0];
    // if input can't be divided by warp, do not calculate sum of res of every
    // warp
    if (dyn_cast<TensorType>(histogramOp.getOperand().getType()).getShape()[0] <
        warpsPerCTA[0])
      warpsWalkNum = dyn_cast<TensorType>(histogramOp.getOperand().getType())
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
    auto loc = loadOp.getLoc();
    auto loadType = loadOp.getType();

    if (!isa<TensorType>(loadType))
      return failure();

    auto originOp = loadOp.getOperation();
    if (replaced2Origin.count(originOp) != 0) {
      originOp = replaced2Origin[originOp];
    }
    auto lastUser = userAnalysis.getLastUserOp(originOp);

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
    auto outTransType =
        MemRefType::get(resultType.getShape(), resultType.getElementType());
    auto outTrans = syncAllocOp(rewriter, loc, loadOp.getOperation(),
                                userAnalysis, replaced2Origin, outTransType);

    auto tagShare = getShareDTETag(rewriter, loadOp);
    auto tagPrivate = getPrivateDTETag(rewriter, loadOp);
    auto tagDte = IsShareOutput ? tagShare : tagPrivate;
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
                  auto load_size =
                      ConfigGcuLoad(builder, loc, output, outTrans, loadOp,
                                    resultType, adaptor.getPtr(),
                                    adaptor.getStrides(), adaptor.getShape(),
                                    defaultValue, tagDte, zero, IsShareOutput);
                  builder.create<scf::YieldOp>(loc, ValueRange{load_size});
                },
                [&](OpBuilder &builder, Location loc) {
                  auto totalNumElems =
                      triton::gcu::getTotalElemsPerThread(loadType);
                  doMemset(builder, loadOp, output, defaultValue,
                           totalNumElems);
                  if (triton::gcu::get_bool_env("TRITON_GCU_DEBUG")) {
                    std::string locStr = "[warning]: load offset is out of "
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
    if (IsShareOutput) {
      auto isThread0 = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
      auto isAll = rewriter.create<arith::AndIOp>(loc, isThread0, shapeCheck);
      rewriter.create<scf::IfOp>(
          loc, isAll, [&](OpBuilder builder, Location loc) {
            WaitGcuLoadStore(builder, loc, tagDte, zero, total_size);
            builder.create<scf::YieldOp>(loc);
          });
      rewriter.create<gpu::BarrierOp>(loc);
    } else {
      rewriter.create<scf::IfOp>(
          loc, shapeCheck, [&](OpBuilder builder, Location loc) {
            WaitGcuLoadStore(builder, loc, tagDte, zero, total_size);
            builder.create<scf::YieldOp>(loc);
          });
    }

    leaveTritionOp(rewriter, loadOp.getOperation());
    rewriter.replaceOp(loadOp, output);
    return success();
  }
};

struct GCUStoreOpLowering : SharedConversionPattern<triton::gcu::StoreOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::gcu::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, storeOp.getOperation());

    bool isLastOp = true;
    auto loc = storeOp.getLoc();
    auto storeType = storeOp.getValue().getType();
    if (!isa<TensorType>(storeType))
      return failure();

    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto storeValueType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(storeType));
    auto storeTransType = MemRefType::get(storeValueType.getShape(),
                                          storeValueType.getElementType());
    auto storeTrans = syncAllocOp(rewriter, loc, nullptr, userAnalysis,
                                  replaced2Origin, storeTransType);
    auto tagDte = isLastOp ? getPrivateDTETag(rewriter, storeOp)
                           : createPrivateDTETag(rewriter, storeOp);

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
                      rewriter, loc, adaptor.getValue(), storeTrans, storeOp,
                      storeValueType, adaptor.getPtr(), adaptor.getStrides(),
                      adaptor.getShape(), tagDte, zero);
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
    if (!isLastOp) {
      auto &lastOp = storeOp.getOperation()->getBlock()->back();
      auto ip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPoint(&lastOp);
      auto ifOp = rewriter.create<scf::IfOp>(
          loc, isNotZero, [&](OpBuilder builder, Location loc) {
            WaitGcuLoadStore(builder, loc, tagDte, zero, total_size);
            builder.create<scf::YieldOp>(loc);
          });

      if (!storeTrans.getUsers().empty()) {
        rewriter.create<memref::DeallocOp>(loc, storeTrans);
      } else {
        rewriter.eraseOp(storeTrans.getDefiningOp());
      }

      rewriter.restoreInsertionPoint(ip);
      moveDeallocOp(rewriter, adaptor.getValue(), ifOp, 0);
    } else {
      rewriter.create<scf::IfOp>(
          loc, isNotZero, [&](OpBuilder builder, Location loc) {
            WaitGcuLoadStore(builder, loc, tagDte, zero, total_size);
            builder.create<scf::YieldOp>(loc);
          });

      if (!storeTrans.getUsers().empty()) {
        rewriter.create<memref::DeallocOp>(loc, storeTrans);
      } else {
        rewriter.eraseOp(storeTrans.getDefiningOp());
      }
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
    auto srcType = op.getSrc().getType();
    auto resultType = op.getType();
    auto rank = srcType.getRank();
    auto wrapSrcType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(srcType));
    auto wrapResultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(resultType));
    auto elementType = wrapResultType.getElementType();

    auto loc = op.getLoc();
    auto tag = getPrivateDTETag(rewriter, op);
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto lastUser = userAnalysis.getLastUserOp(op.getOperation());

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
          temp_out = syncAllocOp(rewriter, loc, op.getOperation(), userAnalysis,
                                 replaced2Origin, memrefType);
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
        rewriter.create<memref_ext::BroadcastStartOp>(loc, dst, src, tag,
                                                      ValueRange{zero});
        rewriter.create<memref::DmaWaitOp>(
            loc, tag, ValueRange{zero},
            rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));

        src_input = temp_out;
        idx++;
      }
      leaveTritionOp(rewriter, op.getOperation());
      rewriter.replaceOp(op, output);
      return success();
    }
    // move source to shared memory
    auto sharedSrc =
        storeToSharedMem(rewriter, tag, srcType, adaptor.getSrc(), false,
                         op.getOperation(), userAnalysis, replaced2Origin);
    auto mergedResultType =
        MemRefType::get(resultType.getShape(), elementType, AffineMap{},
                        rewriter.getI64IntegerAttr(2) /*shared memory*/);
    auto mergedOutput =
        syncAllocOp(rewriter, loc, op.getOperation(), userAnalysis,
                    replaced2Origin, mergedResultType);
    auto totalNumElems = triton::gcu::getTotalElemsPerThread(srcType);
    // broadcast in thread 0
    auto isThread0 = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
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
        temp_out = syncAllocOp(rewriter, loc, op.getOperation(), userAnalysis,
                               replaced2Origin, tempMemrefType);
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
          loc, isThread0, [&](OpBuilder &rewriter, Location loc) {
            rewriter.create<memref_ext::BroadcastStartOp>(loc, dst, src, tag,
                                                          ValueRange{zero});
            rewriter.create<memref::DmaWaitOp>(
                loc, tag, ValueRange{zero},
                rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));
            rewriter.create<scf::YieldOp>(loc);
          });
      src_input = temp_out;
      idx++;
    }
    rewriter.create<gpu::BarrierOp>(loc);
    // read back
    auto output =
        loadFromSharedMem(rewriter, tag, resultType, mergedOutput, false,
                          lastUser, nullptr, userAnalysis, replaced2Origin);
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
    auto lastUser = userAnalysis.getLastUserOp(op.getOperation());
    auto tag = getPrivateDTETag(rewriter, op);
    auto srcType = dyn_cast<TensorType>(op.getSrc().getType());
    auto resMemType =
        MemRefType::get(type.getShape(), resultType.getElementType(),
                        AffineMap{}, rewriter.getI64IntegerAttr(2));
    // move source to shared memory
    auto sharedSrc =
        storeToSharedMem(rewriter, tag, srcType, adaptor.getSrc(), false,
                         op.getOperation(), userAnalysis, replaced2Origin);
    auto [strides, offset] = resMemType.getStridesAndOffset();
    auto result = rewriter.create<memref::ReinterpretCastOp>(
        loc, resMemType, sharedSrc, offset, type.getShape(), strides);
    // copy back outputs
    Value output =
        loadFromSharedMem(rewriter, tag, op.getType(), result, false, lastUser,
                          nullptr, userAnalysis, replaced2Origin);
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
    auto loc = op.getLoc();
    auto resultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    auto srcNumElems = triton::gcu::getElemsPerThread(op.getSrc().getType());
    auto dstNumElems = triton::gcu::getElemsPerThread(op.getType());

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

    auto tag = getPrivateDTETag(rewriter, op);
    auto srcType = dyn_cast<TensorType>(op.getSrc().getType());
    // move source to shared memory
    auto lastUser = userAnalysis.getLastUserOp(op.getOperation());
    auto sharedSrc =
        storeToSharedMem(rewriter, tag, srcType, adaptor.getSrc(), false,
                         op.getOperation(), userAnalysis, replaced2Origin);
    auto resMemType =
        MemRefType::get(type.getShape(), resultType.getElementType(),
                        AffineMap{}, rewriter.getI64IntegerAttr(2));
    auto [strides, offset] = resMemType.getStridesAndOffset();
    auto result = rewriter.create<memref::ReinterpretCastOp>(
        loc, resMemType, sharedSrc, offset, type.getShape(), strides);
    // copy back outputs
    Value output =
        loadFromSharedMem(rewriter, tag, op.getType(), result, false, lastUser,
                          nullptr, userAnalysis, replaced2Origin);
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

    auto lastUser = userAnalysis.getLastUserOp(op.getOperation());
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

    auto tag = getPrivateDTETag(rewriter, op);

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

    rewriter.create<memref_ext::SliceStartOp>(loc, sliceLHS, adaptor.getSrc(),
                                              offsetsLHS, defaultValue, tag,
                                              ValueRange{zero});
    rewriter.create<memref::DmaWaitOp>(
        loc, tag, ValueRange{zero},
        rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));

    rewriter.create<memref_ext::SliceStartOp>(loc, sliceRHS, adaptor.getSrc(),
                                              offsetsRHS, defaultValue, tag,
                                              ValueRange{zero});
    rewriter.create<memref::DmaWaitOp>(
        loc, tag, ValueRange{zero},
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

    auto lastUser = userAnalysis.getLastUserOp(op.getOperation());
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

    auto tag = getPrivateDTETag(rewriter, op);

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

    rewriter.create<memref_ext::DesliceStartOp>(
        loc, result, desliceLHS, offsetsLHS, tag, ValueRange{zero});
    rewriter.create<memref::DmaWaitOp>(
        loc, tag, ValueRange{zero},
        rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));

    rewriter.create<memref_ext::DesliceStartOp>(
        loc, result, desliceRHS, offsetsRHS, tag, ValueRange{zero});
    rewriter.create<memref::DmaWaitOp>(
        loc, tag, ValueRange{zero},
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
    auto type = op.getType();
    auto loc = op.getLoc();
    auto resultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(type));

    auto tag = getPrivateDTETag(rewriter, op);
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto lastUser = userAnalysis.getLastUserOp(op.getOperation());
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
      rewriter.create<memref_ext::DesliceStartOp>(
          loc, output, adaptor.getLhs(), offsets, tag, ValueRange{zero});
      rewriter.create<memref::DmaWaitOp>(
          loc, tag, ValueRange{zero},
          rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));

      offsets[0] = rewriter.create<arith::ConstantIntOp>(
          loc, dyn_cast<MemRefType>(adaptor.getLhs().getType()).getDimSize(0),
          rewriter.getI32Type());
      rewriter.create<memref_ext::DesliceStartOp>(
          loc, output, adaptor.getRhs(), offsets, tag, ValueRange{zero});
      rewriter.create<memref::DmaWaitOp>(
          loc, tag, ValueRange{zero},
          rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems));
      rewriter.replaceOp(op, output);
      return success();
    }
    auto mergedResultType =
        MemRefType::get(type.getShape(), type.getElementType(), AffineMap{},
                        rewriter.getI64IntegerAttr(2) /*shared memory*/);
    auto mergedOutput =
        syncAllocOp(rewriter, loc, op.getOperation(), userAnalysis,
                    replaced2Origin, mergedResultType);
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
    auto output =
        loadFromSharedMem(rewriter, tag, op.getType(), mergedOutput, false,
                          lastUser, nullptr, userAnalysis, replaced2Origin);
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct TTTransOpLowering : SharedConversionPattern<triton::TransOp> {
  using SharedConversionPattern::SharedConversionPattern;

  void applyTranspose(OpBuilder &rewriter, Location loc, Value src,
                      Value output, Value tag, ArrayRef<int32_t> order,
                      unsigned totalSize) const {
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto totalNumElems =
        rewriter.create<arith::ConstantIndexOp>(loc, totalSize);

    SmallVector<Value, 4> layout;
    for (auto i : order) {
      layout.push_back(
          rewriter.create<arith::ConstantIntOp>(loc, i, rewriter.getI32Type()));
    }
    rewriter.create<memref_ext::TransposeStartOp>(loc, output, src, layout, tag,
                                                  ValueRange{zero});
    rewriter.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero},
                                       totalNumElems);
  }

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
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
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto totalNumElems =
        triton::gcu::getTotalElemsPerThread(op.getSrc().getType());
    auto lastUser = userAnalysis.getLastUserOp(op.getOperation());
    // gcu400 only one private dte
    if (mlir::isa<triton::gpu::SharedEncodingTrait>(srcLayout) &&
        mlir::isa<triton::gpu::SharedEncodingTrait>(dstLayout)) {
      // allocate output buffers in shared memory
      auto firstUser = nullptr;
      auto tag = (firstUser == nullptr) ? getPrivateDTETag(rewriter, op)
                                        : createPrivateDTETag(rewriter, op);
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
        layout.push_back(rewriter.create<arith::ConstantIntOp>(
            loc, i, rewriter.getI32Type()));
      }
      auto isThread0 = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
      rewriter.create<scf::IfOp>(
          loc, isThread0, [&](OpBuilder &builder, Location loc) {
            rewriter.create<memref_ext::TransposeStartOp>(
                loc, sharedOutput, adaptor.getSrc(), layout, tag,
                ValueRange{zero});
            builder.create<scf::YieldOp>(loc);
          });
      if (firstUser != nullptr) {
        auto ip = rewriter.saveInsertionPoint();
        rewriter.setInsertionPoint(firstUser);
        rewriter.create<scf::IfOp>(
            loc, isThread0, [&](OpBuilder &builder, Location loc) {
              builder.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero},
                                                totalNumElemsValue);
              builder.create<scf::YieldOp>(loc);
            });
        rewriter.create<gpu::BarrierOp>(loc);
        rewriter.restoreInsertionPoint(ip);
      } else {
        rewriter.create<scf::IfOp>(
            loc, isThread0, [&](OpBuilder &builder, Location loc) {
              builder.create<memref::DmaWaitOp>(loc, tag, ValueRange{zero},
                                                totalNumElemsValue);
              builder.create<scf::YieldOp>(loc);
            });
        rewriter.create<gpu::BarrierOp>(loc);
      }
      leaveTritionOp(rewriter, op.getOperation());
      rewriter.replaceOp(op, sharedOutput);
      return success();
    } else if (isa<triton::gpu::BlockedEncodingAttr>(srcLayout) &&
               isa<triton::gpu::BlockedEncodingAttr>(dstLayout)) {
      // move source to shared memory
      auto tag = getPrivateDTETag(rewriter, op);
      auto lastUser = userAnalysis.getLastUserOp(op.getOperation());
      auto sharedSrc = storeToSharedMem(
          rewriter, tag, dyn_cast<TensorType>(op.getSrc().getType()),
          adaptor.getSrc(), false, op.getOperation(), userAnalysis,
          replaced2Origin);

      // allocate output buffers in shared memory
      auto sharedOutputType = MemRefType::get(
          op.getResult().getType().getShape(), resultType.getElementType(),
          AffineMap{}, rewriter.getI64IntegerAttr(2));
      auto sharedOutput =
          syncAllocOp(rewriter, loc, op.getOperation(), userAnalysis,
                      replaced2Origin, sharedOutputType);

      // split by thread 0
      auto isThread0 = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x), zero);
      rewriter.create<scf::IfOp>(
          loc, isThread0, [&](OpBuilder &builder, Location loc) {
            applyTranspose(builder, loc, sharedSrc, sharedOutput, tag,
                           op.getOrder(), totalNumElems);
            builder.create<scf::YieldOp>(loc);
          });
      rewriter.create<gpu::BarrierOp>(loc);
      // copy back outputs
      Value output = loadFromSharedMem(rewriter, tag, op.getResult().getType(),
                                       sharedOutput, false, lastUser, nullptr,
                                       userAnalysis, replaced2Origin);
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

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
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
    auto lastUser = userAnalysis.getLastUserOp(op.getOperation());
    auto firstUser = nullptr;
    auto tag = (firstUser == nullptr) ? getPrivateDTETag(rewriter, op)
                                      : createPrivateDTETag(rewriter, op);
    if (srcNumElems == dstNumElems &&
        op.getSrc().getType().getShape() == op.getType().getShape()) {
      if (mlir::isa<triton::gpu::SharedEncodingTrait>(srcLayout) &&
          isa<triton::gpu::DotOperandEncodingAttr>(dstLayout)) {
        // give up L2 to matmul because 1:acore crash 2:L2 latency is more
        // 100cyle than L1 we don't had enough resource to refine latency
      } else if (isa<triton::gpu::SliceEncodingAttr>(srcLayout) &&
                 isa<triton::gpu::SliceEncodingAttr>(dstLayout)) {
        if (cast<triton::gpu::SliceEncodingAttr>(srcLayout).getDim() ==
            cast<triton::gpu::SliceEncodingAttr>(dstLayout).getDim()) {
          rewriter.replaceOp(op, adaptor.getSrc());
          return success();
        }
      } else {
        if (mlir::isa<triton::gpu::SharedEncodingTrait>(srcLayout)) {
          auto output = CopyFromSharedMem(
              rewriter, tag, op.getResult().getType(), adaptor.getSrc(), false,
              lastUser, firstUser, userAnalysis, replaced2Origin);
          leaveTritionOp(rewriter, op.getOperation());
          rewriter.replaceOp(op, output);
          return success();
        }
        rewriter.replaceOp(op, adaptor.getSrc());
        return success();
      }
    }
    // share to Distributed
    if (mlir::isa<triton::gpu::SharedEncodingTrait>(srcLayout) &&
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
      // Distributed to dot operand
      auto sharedSrc = storeToSharedMem(
          rewriter, tag, dyn_cast<TensorType>(op.getSrc().getType()),
          adaptor.getSrc(), false, op.getOperation(), userAnalysis,
          replaced2Origin);
      // to dot a or b calculate warp idx
      auto output = loadFromSharedMemForDotOperand(
          rewriter, tag, op.getResult().getType(),
          op.getSrc().getType().getShape(), sharedSrc, lastUser, firstUser,
          userAnalysis, replaced2Origin);
      leaveTritionOp(rewriter, op.getOperation());
      rewriter.replaceOp(op, output);
      return success();
    } else if (mlir::isa<triton::gpu::SharedEncodingTrait>(srcLayout) &&
               isa<triton::gpu::DotOperandEncodingAttr>(dstLayout)) {
      // Distributed to dot operand
      // to dot a or b
      auto output = loadFromSharedMemForDotOperand(
          rewriter, tag, op.getResult().getType(),
          op.getSrc().getType().getShape(), adaptor.getSrc(), lastUser,
          firstUser, userAnalysis, replaced2Origin);
      leaveTritionOp(rewriter, op.getOperation());
      rewriter.replaceOp(op, output);
      return success();
    } else {
      // move source to shared memory
      auto sharedSrc = storeToSharedMem(
          rewriter, tag, dyn_cast<TensorType>(op.getSrc().getType()),
          adaptor.getSrc(), false, op.getOperation(), userAnalysis,
          replaced2Origin);
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
    auto loc = op.getLoc();
    enterTritionOp(rewriter, op.getOperation());
    if (!isa<RankedTensorType>(op.getA().getType()) ||
        !isa<RankedTensorType>(op.getB().getType()))
      return failure();
    if (op.getType().getRank() != 2) {
      llvm::report_fatal_error(
          "triton::gcu::MatmulOp no bias not support 3D or more 3D dot \n");
    }
    auto lastUser = userAnalysis.getLastUserOp(op.getOperation());
    auto resultMemRefType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    auto output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                              replaced2Origin, resultMemRefType);
    rewriter.create<gcu::MatMulOp>(loc, output, adaptor.getA(), adaptor.getB(),
                                   Value());
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct TTDotOpLowering : SharedConversionPattern<triton::DotOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    enterTritionOp(rewriter, op.getOperation());
    if (!isa<RankedTensorType>(op.getA().getType()) ||
        !isa<RankedTensorType>(op.getB().getType()))
      return failure();
    auto lastUser = userAnalysis.getLastUserOp(op.getOperation());
    auto resultMemRefType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    auto output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                              replaced2Origin, resultMemRefType);
    if (op.getType().getRank() == 2) {
      rewriter.create<gcu::MatMulOp>(loc, output, adaptor.getA(),
                                     adaptor.getB(), adaptor.getC());
    } else {
      auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto lhsMemRef = adaptor.getA();
      auto lhsMemRefType = dyn_cast<MemRefType>(lhsMemRef.getType());
      auto rhsMemRef = adaptor.getB();
      auto rhsMemRefType = dyn_cast<MemRefType>(rhsMemRef.getType());
      auto biasMemRef = adaptor.getC();
      auto biasMemRefType = dyn_cast<MemRefType>(biasMemRef.getType());
      int64_t batchNum = lhsMemRefType.getShape()[0];

      auto createFlattened1DMemRef = [&](Value memRef, MemRefType memRefType) {
        auto elementType = memRefType.getElementType();
        int64_t size = 1;
        for (int i = 0; i < memRefType.getRank(); i++) {
          size *= memRefType.getShape()[i];
        }
        // Create flattened buffer
        MemRefType flatType = MemRefType::get({size}, elementType);
        Value flatBuffer = rewriter.create<memref::ReinterpretCastOp>(
            loc, flatType, memRef, zero,
            ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, size)},
            ValueRange{one});

        // Convert flattened buffer to 1D MemRef
        auto ptrType = gcu::PtrType::get(getContext(), elementType);
        Value ptr =
            rewriter.create<gcu::MemRefToPtrOp>(loc, ptrType, flatBuffer);
        MemRefType memType1D =
            MemRefType::get({ShapedType::kDynamic}, rewriter.getI8Type());
        return rewriter.create<gcu::PtrToMemRefOp>(loc, memType1D, ptr);
      };

      // Create 1D MemRefs for lhs, rhs, bias, and output
      Value lhsBuffer = createFlattened1DMemRef(lhsMemRef, lhsMemRefType);
      Value rhsBuffer = createFlattened1DMemRef(rhsMemRef, rhsMemRefType);
      Value biasBuffer = createFlattened1DMemRef(biasMemRef, biasMemRefType);
      Value outBuffer = createFlattened1DMemRef(output, resultMemRefType);
      auto bitWidthOfInt8 = rewriter.getI8Type().getIntOrFloatBitWidth();
      scf::buildLoopNest(
          rewriter, loc, ValueRange{zero},
          ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, batchNum)},
          ValueRange{one},
          [&](OpBuilder &rewriter, Location loc, ValueRange m) {
            auto createViewWithOffset = [&](MemRefType memRefType,
                                            Value buffer) {
              int64_t tailIndex = memRefType.getRank() - 1;
              int64_t dim0 = memRefType.getShape()[tailIndex - 1];
              int64_t dim1 = memRefType.getShape()[tailIndex];
              auto elementType = memRefType.getElementType();
              int64_t elementSize =
                  elementType.getIntOrFloatBitWidth() / bitWidthOfInt8;
              Value offset = rewriter.create<arith::MulIOp>(
                  loc, m[0],
                  rewriter.create<arith::ConstantIndexOp>(
                      loc, dim0 * dim1 * elementSize));
              return rewriter.create<memref::ViewOp>(
                  loc, MemRefType::get({dim0, dim1}, elementType), buffer,
                  offset, ValueRange{});
            };

            Value newLhsMemRef = createViewWithOffset(lhsMemRefType, lhsBuffer);
            Value newRhsMemRef = createViewWithOffset(rhsMemRefType, rhsBuffer);
            Value newBiasMemRef =
                createViewWithOffset(biasMemRefType, biasBuffer);
            Value newOutMemRef =
                createViewWithOffset(resultMemRefType, outBuffer);
            rewriter.create<gcu::MatMulOp>(loc, newOutMemRef, newLhsMemRef,
                                           newRhsMemRef, newBiasMemRef);
          });
    }
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, output);
    return success();
  }
};

} // namespace

void ConvertTritonToGCUPass::runOnOperation() {
  auto *ctx = &getContext();
  auto module = getOperation();

  // pre analysis base triton ir
  triton::gcu::FirstLastUserAnalysis &userAnalysis =
      getAnalysis<triton::gcu::FirstLastUserAnalysis>();

  std::map<Operation *, Operation *> replaced2Origin;
  replaced2Origin.clear();

  std::map<Operation *, Operation *> asyncLoad2Tag;
  std::map<Operation *, Operation *> asyncWait2Tag;
  llvm::DenseMap<Operation *, Value> asyncLoad2TagIdex;
  getPipelineAsyncResourceMaping(module, asyncLoad2Tag, asyncLoad2TagIdex,
                                 asyncWait2Tag);
  std::map<Operation *, std::map<uint64_t, bool>>
      TTYeiledOPerandHasMultiUseStage;
  AnalysisYieldOperendUseStage(module, userAnalysis,
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
      if (mlir::isa<triton::gpu::SharedEncodingTrait>(tType.getEncoding())) {
        return MemRefType::get(
            shape, elemType, AffineMap{},
            IntegerAttr::get(IntegerType::get(tensorType.getContext(), 64), 2));
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
  ConversionTarget target(getContext());

  mlir::triton::populateReduceOpToGCUPatterns(converter, patterns, userAnalysis,
                                              replaced2Origin);
  mlir::triton::populateElementwiseFusionOpToGCUPatterns(
      converter, patterns, userAnalysis, replaced2Origin);

  patterns
      .add<TTFuncOpLowering, TTReturnOpLowering, TTCallOpLowering,
           TTSCFForOpLowering, TTSCFIfOpLowering, TTSCFWhileOpLowering,
           TTSCFConditionLowering,
           TTIntrinsicOpLowering<triton::GetNumProgramsOp, gpu::GridDimOp>,
           TTIntrinsicOpLowering<triton::GetProgramIdOp, gpu::BlockIdOp>,
           TTPrintOpLowering, TTAssertOpLowering, TTAddPtrOpLowering,
           TTLoadOpLowering, TTStoreOpLowering, TTConstantOpLowering,
           TTReduceReturnOpLowering, TTScanReturnOpLowering,
           TTExternElemwiseOpLowering,
           TTElementwiseOpLowering<triton::PtrToIntOp, gcu::PtrToIntOp>,
           TTElementwiseOpLowering<triton::IntToPtrOp, gcu::IntToPtrOp>,
           TTElementwiseOpLowering<triton::gcu::PtrToIntOp, gcu::PtrToIntOp>,
           TTElementwiseOpLowering<triton::gcu::IntToPtrOp, gcu::IntToPtrOp>,
           TTElementwiseOpLowering<triton::MulhiUIOp, math_ext::UmulhiOp>,
           TTArithSelectOpLowering, TTBitcastOpLowering, TTBroadcastOpLowering,
           TTCatOpLowering, TTHistogramOpLowering, TTExpandDimsOpLowering,
           TTReshapeOpLowering, TTSplitOpLowering, TTJoinOpLowering,
           GCUMatmulLowering, TTGAssertOpLowering, TTTransOpLowering,
           TTGConvertLayoutOpLowering, GCULoadOpLowering, GCUStoreOpLowering,
           TTDotOpLowering, TTSplatOpLowering>(converter, ctx, userAnalysis,
                                               replaced2Origin);

  patterns.add<TTScanOpLowering>(converter, ctx, userAnalysis, replaced2Origin,
                                 vectorLength);
  patterns.add<TTMakeRangeOpLowering>(converter, ctx, userAnalysis,
                                      replaced2Origin, vectorLength,
                                      vectorizationMaxLength);
  patterns.add<TTSCFYieldOpLowering>(converter, ctx, userAnalysis,
                                     replaced2Origin,
                                     TTYeiledOPerandHasMultiUseStage);

  patterns.add<TTShareAllocOpLowering, TTShareDeallocOpLowering,
               TTMemDescSubviewOpLowering>(converter, ctx);

  patterns.add<TTLocalLoadOpLowering>(converter, ctx, userAnalysis,
                                      replaced2Origin);

  patterns.add<TTAsyncLoadGlobalToShareOpLowering>(
      converter, ctx, asyncLoad2Tag, asyncLoad2TagIdex);
  patterns.add<TTAsyncWaitOpLowering>(converter, ctx, asyncWait2Tag);

  target.addLegalDialect<
      gpu::GPUDialect, gcu::GCUDialect, arith::ArithDialect,
      affine::AffineDialect, func::FuncDialect, scf::SCFDialect,
      math::MathDialect, vector::VectorDialect, memref::MemRefDialect,
      memref_ext::MemrefExtDialect, math_ext::MathExtDialect>();
  target.addIllegalDialect<triton::TritonDialect,
                           triton::gpu::TritonGPUDialect>();
  target.addIllegalOp<mlir::triton::gcu::ElementwiseFusionRegionOp,
                      mlir::triton::gcu::YieldOp, mlir::triton::gcu::LoadOp,
                      mlir::triton::gcu::StoreOp>();
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

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
