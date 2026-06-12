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

#ifndef KURAMA_TRITONGPU_TO_GCU_UTILITY_H
#define KURAMA_TRITONGPU_TO_GCU_UTILITY_H

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Constants.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

namespace mlir {

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

class PrivateTagPool {
public:
  PrivateTagPool(mlir::Operation *entryFunc, int32_t numWarps,
                 bool useAsyncSharedTag, bool useAllTags = false);

  ~PrivateTagPool() { updateUsedSize(); }

  TagInfo getPrivateSyncTagInfo(mlir::Operation *op);
  TagInfo tryGetPrivateAsyncTagInfo(mlir::Operation *op);
  TagInfo getSharedSyncTagInfo(mlir::Operation *op);
  TagInfo tryGetSharedAsyncTagInfo(mlir::Operation *op);

  void setMap(Operation *op, TagInfo tagInfo);
  bool isExistInMap(Operation *op) const;
  void releaseMap(Operation *op);

  Type getPrivateTagsType();
  Type getSharedTagsType();

  /// Tag memref for DMA sync; must be a value visible in `op`'s region (e.g.
  /// partition block arg for isolated `warp_specialize` partition regions).
  Value getPrivateTagsValue(Operation *op);
  Value getSharedTagsValue(Operation *op);

  void updateUsedSize();

  void setPrivateFuncNameMap(Operation *op, int argNum);
  void setSharedFuncNameMap(Operation *op, int argNum);

private:
  /// tt.load.async attribute is true, use async shared tag.
  /// at the same time, we don't use async private tag.
  bool useAsyncSharedTag;

  /// if sync shared tag is not used, we don't need to allocate shared tag.
  bool usedSyncSharedTag;

  int32_t sTagsSize;
  int32_t pTagsSize;

  int32_t sTagsPeakSize;
  int32_t pTagsPeakSize;

  /// To recognize unused tag in all tags 'tagsAllocOp'
  std::vector<bool> pTagsBitset;
  std::vector<bool> sTagsBitset;

  ///  All tags
  mlir::Operation *sTagsAllocOp;
  mlir::Operation *pTagsAllocOp;

  /// before the operation, we can release corresponding tag.
  llvm::DenseMap<Operation *, std::vector<TagInfo>> op2TagInfoMap;

  /// All tags locate at callee func's argument.
  llvm::DenseMap<llvm::StringRef, int32_t> pTagsArgPosMap;
  llvm::DenseMap<llvm::StringRef, int32_t> sTagsArgPosMap;
};

/// Returns true if tt.reshape needs extra overhead instead of a zero-copy
/// memref.reinterpret_cast
bool isExpensiveView(Type srcTy, Type dstTy);

} // namespace gcu
} // namespace triton

int32_t getMasterThreadId(Operation *op, int32_t defaultWarpId = 0);
int32_t getMasterThreadId(Region *region, int32_t defaultWarpId = 0);
void captureValuesToWarpSpecializeOp(Operation *entryFunc, Value capture);
Value getShareDTETag(OpBuilder &builder, Operation *op);
DenseSet<unsigned> getSlicedAxies(Type type);
SmallVector<Value, 4> getWarpIds(OpBuilder &builder, Location loc, Type type);
SmallVector<Value, 4> getElemsPerThread(OpBuilder &builder, Location loc,
                                        Type type);
bool isMustAliasOp(OpOperand &use);

mlir::Operation *
promoteLastUser(std::pair<Operation *, int> &lastUser,
                triton::gcu::FirstLastUserAnalysis &userAnalysis,
                std::map<Operation *, Operation *> &replaced2Origin);

void addDeallocAfterLastUser(OpBuilder &builder,
                             std::pair<Operation *, int> lastUser, Value alloc);
Value syncAllocOp(OpBuilder &builder, Location &loc,
                  std::pair<Operation *, int> lastUser,
                  triton::gcu::FirstLastUserAnalysis &userAnalysis,
                  std::map<Operation *, Operation *> &replaced2Origin,
                  MemRefType type, int64_t memoryAlignment = INVALID_ALIGNMENT);

void createPrintfOp(OpBuilder &rewriter, Location loc,
                    ::llvm::StringRef printOpPrefix, bool hex, Value value);

void enterTritionOp(ConversionPatternRewriter &rewriter, Operation *ttParent);

void leaveTritionOp(ConversionPatternRewriter &rewriter, Operation *ttParent);

void doMemFence(OpBuilder &rewriter, Operation *op);

void doMemsetConfig(OpBuilder &rewriter, Location loc, Value output, Value v,
                    triton::gcu::TagInfo tag);

void doMemset(OpBuilder &rewriter, triton::gcu::TagInfo tag, Operation *op,
              Value output, Value v, int totalNumElems);

Value loadFromSharedMem(OpBuilder &builder, triton::gcu::TagInfo tag, Type type,
                        Value buffer, bool onlyThread0,
                        std::pair<Operation *, int> lastTTUser,
                        std::pair<Operation *, int> firstTTUser,
                        triton::gcu::FirstLastUserAnalysis &userAnalysis,
                        std::map<Operation *, Operation *> &replaced2Origin);
Value CopyFromSharedMem(OpBuilder &builder, triton::gcu::TagInfo tag, Type type,
                        Value buffer, bool onlyThread0,
                        std::pair<Operation *, int> lastTTUser,
                        std::pair<Operation *, int> firstTTUser,
                        triton::gcu::FirstLastUserAnalysis &userAnalysis,
                        std::map<Operation *, Operation *> &replaced2Origin);

void storeToSharedMem(OpBuilder &builder, triton::gcu::TagInfo tag,
                      TensorType type, Value sharedBuffer, Value buffer,
                      bool onlyThread0);
Value storeToSharedMem(OpBuilder &builder, triton::gcu::TagInfo tag,
                       TensorType type, Value buffer, bool onlyThread0,
                       std::pair<Operation *, int> lastTTUser,
                       triton::gcu::FirstLastUserAnalysis &userAnalysis,
                       std::map<Operation *, Operation *> &replaced2Origin);

MemRefType getMemRefTypeFromSharedMem(RankedTensorType tType, Type elemType);

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

Value ConfigGcuLoad(OpBuilder &rewriter, Location loc, Value srcOut,
                    mlir::Operation *op, MemRefType resultType, Value loadPtr,
                    mlir::ValueRange configStrides,
                    mlir::ValueRange configShapes, Value defaultValue,
                    triton::gcu::TagInfo tag, bool IsShareOutput = false);

Value ConfigGcuStore(OpBuilder &rewriter, Location loc, Value storeValue,
                     mlir::Operation *op, MemRefType storeValueType,
                     Value storePtr, mlir::ValueRange configStrides,
                     mlir::ValueRange configShapes, triton::gcu::TagInfo tag);

void WaitGcuLoadStore(OpBuilder &rewriter, Location loc,
                      triton::gcu::TagInfo tag, Value totalSize);

bool useMatrixStore(triton::gcu::StoreOp storeOp, Value adaptedValue);

void ConfigMatrixStore(OpBuilder &rewriter, Location loc,
                       triton::gcu::StoreOp storeOp, Value value, Value ptr,
                       ValueRange dstShapes, ValueRange dstStrides,
                       ValueRange dstOffsets, bool hasTrans);

void removeRedundantZeroFill(ConversionPatternRewriter &rewriter,
                             memref::AllocOp allocOp);

void moveDeallocOp(ConversionPatternRewriter &rewriter, Value v, Operation *pos,
                   size_t depth);

void mergeContinuousDims(OpBuilder &subBuilder, Location loc,
                         Value &sharedMemref, Value &warpMemref,
                         SmallVector<Value, 4> &offsets,
                         SmallVector<Value, 4> &mergedOffsets,
                         MemRefType &sharedMemType, MemRefType &warpMemType,
                         Value &sharedBuffer, Value &warpOutput);

namespace triton {
namespace gcu {
bool get_bool_env(const char *name);

Value createConstantZero(OpBuilder &builder, Location loc, Type elemType);

SmallVector<unsigned> getWarpsPerCTA(Attribute layout);

/// Returns true if an extract_tile/insert_tile op needs SMEM relay because
/// at least one cut dimension (tileShape[i] < srcShape[i]) has warpsPerCTA > 1,
/// meaning cross-warp communication is required.
bool needsSmemRelay(RankedTensorType srcTy, ArrayRef<int64_t> tileShape);

SmallVector<unsigned> getElemsPerThread(Type type);
unsigned getTotalElemsPerThread(Type type);
unsigned getBpe(Type type);

// For each flat warp index [0, totalWarps), returns true if the warp holds
// unique data for the given tensor layout, false if it is a redundant copy of
// another warp's data.  Used to guard operations like atomics where only
// non-redundant warps should participate.
SmallVector<bool> getFreeWarpMask(Type type);

inline int64_t ceilDiv(int64_t lhs, int64_t rhs) {
  assert(rhs >= 1);
  // C/C++'s integer division rounds towards 0.
  return lhs % rhs > 0 ? lhs / rhs + 1 : lhs / rhs;
}
int getNumWarps(ModuleOp mod);
int getTotalNumWarps(mlir::gpu::GPUModuleOp mod);

class TritonGCUBuilder {
public:
  TritonGCUBuilder(Location loc, OpBuilder &builder)
      : loc(loc), builder(&builder) {}
  Type getTarType();
  Value tarAddr(Value memref);
  Value tarValue(int64_t v);
  Value tarValue(Value v);
  Value tarStride(VectorType type, int64_t stride);
  Value tarLoad(VectorType type, Value &tarAddr, const Value &tarStride);
  void tarStore(Value v, Value &tarAddr, const Value &tarStride);
  Value tarGather(VectorType type, Value &tarAddr, Value num, Value other,
                  Value mask);
  void tarScatter(Value &tarAddr, Value v, Value num, Value mask);
  void tarJump(Value &tarAddr, const Value &tarValue);

private:
  Location loc;
  OpBuilder *builder;
  static constexpr unsigned oaccSizeInBytes = 512;
};

bool isNvLibDeviceSymbol(StringRef symbol);
bool isMixedPrecisionSymbol(StringRef symbol);

enum class LoadKind {
  Load,
  StridedLoad,
  TarLoad,
  MaskedLoad,
  GatherLoad,
};

enum class StoreKind {
  Store,
  StridedStore,
  TarStore,
  MaskedStore,
  ScatterStore,
};

class ReduceGenerator {
public:
  using PrologueArgListType = Block::BlockArgListType;
  using PrologueOpIteratorRange = llvm::iterator_range<Block::iterator>;
  explicit ReduceGenerator(
      triton::ReduceOp op, PrologueArgListType prologueArgList = {},
      PrologueOpIteratorRange prologueOps = llvm::make_range(nullptr, nullptr));
  SmallVector<Value> normalizeInputs(OpBuilder &builder, Location loc,
                                     ValueRange inputs);
  SmallVector<Value> normalizeOutputs(OpBuilder &builder, Location loc,
                                      ValueRange outputs);
  bool hasVectorizeImpl() const;
  void applyReduce(OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
                   ArrayRef<Value> inputs);

private:
  int64_t reduceAxis = 2;
  std::array<int64_t, 3> reduceInputDims = {1, 1, 1};
  std::array<int64_t, 3> reduceOutputDims = {1, 1, 1};
  triton::ReduceOp op;
  PrologueOpIteratorRange prologueOps;
  PrologueArgListType prologueArgList;
  bool partialReduce = false;
  int64_t vectorizeAxis = 3;
  unsigned vectorLength;
  static constexpr unsigned oaccSizeInBytes = 512;
  constexpr static unsigned loopUnrollTime = 16;

private:
  void collectVectorizeInfo();
  void processPrologue(OpBuilder &builder, Location loc, IRMapping &mapper);
  template <unsigned reduceAxis, unsigned vectorizeAxis>
  void applyVectorizeImpl(OpBuilder &builder, Location loc,
                          ArrayRef<Value> outputs, ArrayRef<Value> inputs);
};
} // namespace gcu
} // namespace triton
} // namespace mlir
#endif
