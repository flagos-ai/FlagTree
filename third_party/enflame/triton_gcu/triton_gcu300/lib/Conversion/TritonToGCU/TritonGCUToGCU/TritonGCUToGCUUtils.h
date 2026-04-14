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
#include <utility>
#include <vector>

#include "ConstantUtil.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
constexpr int64_t INVALID_ALIGNMENT = -1;
const char *const kAlignment = "alignment";
namespace triton {
namespace gcu {
class FirstLastUserAnalysis;

struct TagInfo {
  mlir::Value tag;
  mlir::Value idx;
  bool isShared;

  TagInfo() = default;
  TagInfo(mlir::Value tag, mlir::Value idx, bool isShared)
      : tag(tag), idx(idx), isShared(isShared) {}
  TagInfo(const TagInfo &other) = default;
  TagInfo(TagInfo &&other) = default;
  TagInfo &operator=(const TagInfo &other) = default;
  TagInfo &operator=(TagInfo &&other) = default;

  bool isSharedTag() const { return isShared; }

  bool isAsync() const {
    auto op = idx.getDefiningOp();
    if (!llvm::isa<arith::ConstantIndexOp>(op)) {
      llvm::report_fatal_error("idx is not a ConstantIndexOp op");
    }
    auto constIndxOp = llvm::cast<arith::ConstantIndexOp>(op);
    return constIndxOp.value() != 0;
  }

  mlir::Value getTag() const { return tag; }

  mlir::Value getIdx() const { return idx; }
  int32_t getIdxInt() const {
    auto op = idx.getDefiningOp();
    if (!llvm::isa<arith::ConstantIndexOp>(op)) {
      llvm::report_fatal_error("idx is not a ConstantIndexOp op");
    }
    auto constIndxOp = llvm::cast<arith::ConstantIndexOp>(op);
    return constIndxOp.value();
  }
};

class PrivateDTETagPool {
public:
  explicit PrivateDTETagPool(mlir::Operation *entryFunc) {
    OpBuilder builder(entryFunc);
    auto func = llvm::dyn_cast<FunctionOpInterface>(entryFunc);
    auto firstOp = &func.getFunctionBody().getBlocks().front().front();

    size = 17;
    peakSize = 1;
    bitset = std::vector<bool>(size, false);
    bitset[0] = true; // for sync tag

    auto tagType =
        MemRefType::get(ArrayRef<int64_t>{size}, builder.getI32Type());
    builder.setInsertionPoint(firstOp);
    tagsAllocOp = builder.create<memref::AllocOp>(entryFunc->getLoc(), tagType);
    tagsAllocOp->setAttr("gcu_private_tag", builder.getUnitAttr());
  }

  ~PrivateDTETagPool() { updateUsedSize(); }

  PrivateDTETagPool(const PrivateDTETagPool &other) = delete;
  PrivateDTETagPool(PrivateDTETagPool &&other) = delete;
  PrivateDTETagPool &operator=(const PrivateDTETagPool &other) = delete;
  PrivateDTETagPool &operator=(PrivateDTETagPool &&other) = delete;

public:
  TagInfo getSyncTagInfo(mlir::Operation *op) {
    auto loc = op->getLoc();
    auto builder = OpBuilder(op);
    auto tags = getTagsValue(op);
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    return TagInfo(tags, zero, false);
  }

  TagInfo trygGetAsyncTagInfo(mlir::Operation *op) {
    int32_t idx = -1;
    for (int32_t i = 1; i < size; i++) {
      if (!bitset[i]) {
        peakSize = peakSize < (i + 1) ? (i + 1) : peakSize;
        idx = i;
        break;
      }
    }

    auto loc = op->getLoc();
    auto builder = OpBuilder(op);
    auto tags = getTagsValue(op);
    if (idx == -1) {
      auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
      return TagInfo(tags, zero, false);
    } else {
      bitset[idx] = true;
      auto idxValue = builder.create<arith::ConstantIndexOp>(loc, idx);
      return TagInfo(tags, idxValue, false);
    }
  }

  void setMap(Operation *op, TagInfo tagInfo) {
    op2TagInfoMap[op].push_back(tagInfo);
  }

  bool isExistInMap(Operation *op) const {
    return op2TagInfoMap.count(op) != 0;
  }

  void releaseMap(Operation *op) {
    if (op2TagInfoMap.count(op) == 0)
      return;

    auto tags = op2TagInfoMap[op];
    for (int i = 0; i < tags.size(); ++i) {
      bitset[tags[i].getIdxInt()] = false;
    }
    op2TagInfoMap.erase(op);
  }

  Type getTagsType() { return tagsAllocOp->getResult(0).getType(); }

  Value getTagsValue(Operation *op) {
    auto func = op->getParentOfType<FunctionOpInterface>();
    assert(func && "can't find func op");
    Value tags;
    if (funcName2TagArgPos.count(func.getName()) != 0) {
      tags = func.getArgument(funcName2TagArgPos[func.getName()]);
    } else {
      tags = tagsAllocOp->getResult(0);
    }
    return tags;
  }

  void updateUsedSize() {
    OpBuilder builder(tagsAllocOp);
    auto loc = tagsAllocOp->getLoc();
    auto tagsType =
        MemRefType::get(ArrayRef<int64_t>{peakSize}, builder.getI32Type());
    auto allocOp = builder.create<memref::AllocOp>(loc, tagsType);
    allocOp->setAttr("gcu_private_tag", builder.getUnitAttr());
    tagsAllocOp->getResult(0).replaceAllUsesWith(allocOp->getResult(0));
    tagsAllocOp->dropAllUses();
    tagsAllocOp->erase();

    mlir::gpu::GPUModuleOp moduleOp =
        allocOp->getParentOfType<mlir::gpu::GPUModuleOp>();
    if (!moduleOp) {
      llvm::report_fatal_error("can't find GPUModuleOp for tags");
    }

    for (auto &[funcName, argPos] : funcName2TagArgPos) {
      func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(funcName);
      if (!funcOp) {
        llvm::report_fatal_error("can't find func op in GPUModuleOp for tags");
      }
      if (argPos >= funcOp.getNumArguments()) {
        llvm::report_fatal_error("arg index out of range for tags");
      }
      FunctionType currentType = funcOp.getFunctionType();
      SmallVector<Type> newInputTypes =
          llvm::to_vector(currentType.getInputs());
      SmallVector<Type> newResultTypes =
          llvm::to_vector(currentType.getResults());
      newInputTypes[argPos] = tagsType;
      FunctionType newFuncType =
          FunctionType::get(funcOp.getContext(), newInputTypes, newResultTypes);
      funcOp.setType(newFuncType);
      Block &entryBlock = funcOp.getBody().front();
      entryBlock.getArgument(argPos).setType(tagsType);
    }
  }

  void setFuncNameMap(Operation *op, int argNum) {
    assert(llvm::isa<FunctionOpInterface>(op) && "not a triton func op");
    auto func = llvm::cast<FunctionOpInterface>(op);
    assert(argNum >= 0 && "arg number is negative");
    assert(argNum < func.getNumArguments() && "arg number is too big");
    funcName2TagArgPos[func.getName()] = argNum;
  }

private:
  int32_t size;
  int32_t peakSize;

  std::vector<bool> bitset;

  llvm::DenseMap<Operation *, std::vector<TagInfo>> op2TagInfoMap;
  llvm::DenseMap<llvm::StringRef, int32_t> funcName2TagArgPos;

  mlir::Operation *tagsAllocOp;
};

} // namespace gcu
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
Value castToMemref1D(OpBuilder &rewriter, Location loc, Value v,
                     Value totalNumElems);

bool isMustAliasOp(mlir::Operation *op);

mlir::Operation *
promoteLastUser(std::pair<mlir::Operation *, int> &lastUser,
                triton::gcu::FirstLastUserAnalysis &userAnalysis,
                std::map<Operation *, Operation *> &replaced2Origin);

void addDeallocAfterLastUser(OpBuilder &builder,
                             std::pair<mlir::Operation *, int> lastUser,
                             Value alloc);
Value syncAllocOp(OpBuilder &builder, Location &loc,
                  std::pair<Operation *, int> lastUser,
                  triton::gcu::FirstLastUserAnalysis &userAnalysis,
                  std::map<Operation *, Operation *> &replaced2Origin,
                  MemRefType type, int64_t memoryAlignment = INVALID_ALIGNMENT);

Value asyncAllocOp(OpBuilder &builder, Operation *ttParent, MemRefType type,
                   int64_t memoryAlignment = INVALID_ALIGNMENT);

void createPrintfOp(ConversionPatternRewriter &rewriter, Location loc,
                    ::llvm::StringRef printOpPrefix, bool hex, Value value);

void enterTritionOp(ConversionPatternRewriter &rewriter, Operation *ttParent);

void leaveTritionOp(ConversionPatternRewriter &rewriter, Operation *ttParent);

void doMemFence(OpBuilder &rewriter, Operation *op);

void doMemsetConfig(OpBuilder &rewriter, Location loc, Value output, Value v,
                    triton::gcu::TagInfo tag);

void doMemset(OpBuilder &rewriter, triton::gcu::PrivateDTETagPool &pTagPool,
              Operation *op, Value output, Value v, int totalNumElems);

Value loadFromSharedMem(OpBuilder &builder, triton::gcu::TagInfo tag, Type type,
                        Value buffer, bool onlyThread0,
                        std::pair<mlir::Operation *, int> lastTTUser,
                        std::pair<mlir::Operation *, int> firstTTUser,
                        triton::gcu::FirstLastUserAnalysis &userAnalysis,
                        std::map<Operation *, Operation *> &replaced2Origin);

Value CopyFromSharedMem(OpBuilder &builder, triton::gcu::TagInfo tag, Type type,
                        Value buffer, bool onlyThread0,
                        std::pair<mlir::Operation *, int> lastTTUser,
                        std::pair<mlir::Operation *, int> firstTTUser,
                        triton::gcu::FirstLastUserAnalysis &userAnalysis,
                        std::map<Operation *, Operation *> &replaced2Origin);

Value loadFromSharedMemForDotOperand(
    OpBuilder builder, triton::gcu::TagInfo tag, Type type,
    ArrayRef<int64_t> mnShape, Value sharedBuffer,
    std::pair<mlir::Operation *, int> lastTTUser,
    std::pair<mlir::Operation *, int> firstTTUser,
    triton::gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin);

void storeToSharedMem(OpBuilder &builder, triton::gcu::TagInfo tag,
                      TensorType type, Value sharedBuffer, Value buffer,
                      bool onlyThread0);

Value storeToSharedMem(OpBuilder &builder, triton::gcu::TagInfo tag,
                       TensorType type, Value buffer, bool onlyThread0,
                       std::pair<mlir::Operation *, int> lastTTUser,
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

Value ConfigGcuLoad(OpBuilder &rewriter, Location loc,
                    triton::gcu::PrivateDTETagPool &pTagPool, Value srcOut,
                    Value transOut, mlir::Operation *op, MemRefType resultType,
                    Value loadPtr, mlir::ValueRange configStrides,
                    mlir::ValueRange configShapes, Value defaultValue,
                    triton::gcu::TagInfo tag, bool IsShareOutput = false);

Value ConfigGcuLoadEx(OpBuilder &rewriter, Location loc,
                      triton::gcu::PrivateDTETagPool &pTagPool, Value srcOut,
                      Value tempBuffer, mlir::Operation *op,
                      MemRefType resultType, Value loadPtr,
                      mlir::ValueRange configStrides,
                      mlir::ValueRange configShapes, Value defaultValue,
                      triton::gcu::TagInfo tag, bool IsShareOutput = false);

Value ConfigGcuStore(OpBuilder &rewriter, Location loc, Value storeValue,
                     Value transOut, mlir::Operation *op,
                     MemRefType storeValueType, Value storePtr,
                     mlir::ValueRange configStrides,
                     mlir::ValueRange configShapes, triton::gcu::TagInfo tag);

void WaitGcuLoadStore(OpBuilder &rewriter, Location loc,
                      triton::gcu::TagInfo tag, Value totalSize);

void moveDeallocOp(ConversionPatternRewriter &rewriter, Value v, Operation *pos,
                   size_t depth);

void mergeContinuousDims(OpBuilder &subBuilder, Location loc,
                         Value &sharedMemref, Value &warpMemref,
                         SmallVector<Value, 4> &offsets,
                         SmallVector<Value, 4> &mergedOffsets,
                         MemRefType &sharedMemType, MemRefType &warpMemType,
                         Value &sharedBuffer, Value &warpOutput);
#endif
