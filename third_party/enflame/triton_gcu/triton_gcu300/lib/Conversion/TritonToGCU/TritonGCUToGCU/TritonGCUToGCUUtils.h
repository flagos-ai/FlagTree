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
#ifndef KURAMA_TRITONGCU_TO_GCU_UTILS_H_
#define KURAMA_TRITONGCU_TO_GCU_UTILS_H_

#include <map>

#include "ConstantUtil.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace triton {
namespace gcu {
class FirstLastUserAnalysis;
}
} // namespace triton
} // namespace mlir
using namespace mlir;

Value getPrivateDTETag(OpBuilder &builder, Operation *op);
Value getShareDTETag(OpBuilder &builder, Operation *op);
Value createPrivateDTETag(OpBuilder &builder, Operation *op);
DenseSet<unsigned> getSlicedAxies(Type type);
SmallVector<Value, 4> getWarpIds(OpBuilder &builder, Location loc, Type type);
SmallVector<Value, 4> getElemsPerThread(OpBuilder &builder, Location loc,
                                        Type type);
func::FuncOp getOrDefineFunction(gpu::GPUModuleOp moduleOp, Location loc,
                                 OpBuilder &rewriter, StringRef name,
                                 FunctionType type);
void doMemFence(OpBuilder &rewriter, Operation *op);

void doMemsetConfig(OpBuilder &rewriter, Location loc, Value output, Value v,
                    Value tagDte, Value tagIdx);
void doMemset(OpBuilder &rewriter, Operation *op, Value output, Value v,
              unsigned totalNumElems);

Value castToMemref1D(OpBuilder &rewriter, Location loc, Value v,
                     Value totalNumElems);
bool isMustAliasOp(mlir::Operation *op);

mlir::Operation *
promoteLastUser(mlir::Operation *&lastUser,
                triton::gcu::FirstLastUserAnalysis &userAnalysis,
                std::map<Operation *, Operation *> &replaced2Origin);

void addDeallocAfterLastUser(OpBuilder &builder, mlir::Operation *lastUser,
                             Value alloc);
Value syncAllocOp(OpBuilder &builder, Location &loc, Operation *lastUser,
                  triton::gcu::FirstLastUserAnalysis &userAnalysis,
                  std::map<Operation *, Operation *> &replaced2Origin,
                  MemRefType type);
Value asyncAllocOp(OpBuilder &builder, Operation *ttParent, MemRefType type);

void createPrintfOp(ConversionPatternRewriter &rewriter, Location loc,
                    ::llvm::StringRef printOpPrefix, bool hex, Value value);

void enterTritionOp(ConversionPatternRewriter &rewriter, Operation *ttParent);

void leaveTritionOp(ConversionPatternRewriter &rewriter, Operation *ttParent);

Value loadFromSharedMem(OpBuilder &builder, Value tag, Type type, Value buffer,
                        bool onlyThread0, Operation *lastTTUser,
                        Operation *firstTTUser,
                        triton::gcu::FirstLastUserAnalysis &userAnalysis,
                        std::map<Operation *, Operation *> &replaced2Origin);
Value CopyFromSharedMem(OpBuilder &builder, Value tag, Type type, Value buffer,
                        bool onlyThread0, Operation *lastTTUser,
                        Operation *firstTTUser,
                        triton::gcu::FirstLastUserAnalysis &userAnalysis,
                        std::map<Operation *, Operation *> &replaced2Origin);

Value loadFromSharedMemForDotOperand(
    OpBuilder builder, Value tag, Type type, ArrayRef<int64_t> mnShape,
    Value sharedBuffer, Operation *lastTTUser, Operation *firstTTUser,
    triton::gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin);

void storeToSharedMem(OpBuilder &builder, Value tag, TensorType type,
                      Value sharedBuffer, Value buffer, bool onlyThread0);
Value storeToSharedMem(OpBuilder &builder, Value tag, TensorType type,
                       Value buffer, bool onlyThread0, Operation *lastTTUser,
                       triton::gcu::FirstLastUserAnalysis &userAnalysis,
                       std::map<Operation *, Operation *> &replaced2Origin);
void AnalysisYieldOperendUseStage(
    Operation *module, triton::gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, std::map<uint64_t, bool>>
        &TTYeiledOPerandHasMultiUseStage);

void GetOrderValueByStride(
    OpBuilder &rewriter, Location loc, SmallVector<unsigned> nInitStrideDims,
    SmallVector<Value, 4> &initStride, SmallVector<Value, 4> &initShape,
    SmallVector<Value, 4> &initOffset, SmallVector<Value, 4> &orderStride,
    SmallVector<Value, 4> &orderShape, SmallVector<Value, 4> &orderOffset,
    SmallVector<Value, 4> &vOrder);

void GetOrderSlicefor30(OpBuilder &rewriter, Location loc, int64_t rank,
                        SmallVector<Value, 4> &initStride,
                        SmallVector<Value, 4> &initSliceShape,
                        SmallVector<Value, 4> &orderSliceShape);

Value ConfigGcuLoad(OpBuilder &rewriter, Location loc, Value srcOut,
                    Value transOut, mlir::Operation *op, MemRefType resultType,
                    Value loadPtr, mlir::ValueRange configStrides,
                    mlir::ValueRange configShapes, Value defaultValue,
                    Value tagDte, Value tagIdx, bool IsShareOutput = false);

Value ConfigGcuStore(OpBuilder &rewriter, Location loc, Value storeValue,
                     Value transOut, mlir::Operation *op,
                     MemRefType storeValueType, Value storePtr,
                     mlir::ValueRange configStrides,
                     mlir::ValueRange configShapes, Value tagDte, Value tagIdx);

void WaitGcuLoadStore(OpBuilder &rewriter, Location loc, Value tagDte,
                      Value tagIdx, Value totalSize);

void moveDeallocOp(ConversionPatternRewriter &rewriter, Value v, Operation *pos,
                   size_t depth);

void mergeContinuousDims(OpBuilder &subBuilder, Location loc,
                         Value &sharedMemref, Value &warpMemref,
                         SmallVector<Value, 4> &offsets,
                         SmallVector<Value, 4> &mergedOffsets,
                         MemRefType &sharedMemType, MemRefType &warpMemType,
                         Value &sharedBuffer, Value &warpOutput);
#endif
