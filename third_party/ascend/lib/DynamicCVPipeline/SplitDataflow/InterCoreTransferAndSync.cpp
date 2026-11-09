/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/SplitDataflow/InterCoreTransferAndSync.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/Utils.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <set>

#include "ascend/include/DynamicCVPipeline/Common/FlagIdManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/DataDependencyAnalysis.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/FlagIdReuse.h"

#include "Utils/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

static constexpr const char *DEBUG_TYPE = "inter-core-transfer-and-sync";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir::triton;
using namespace hivm;

static constexpr int kIntegerBitWidth = 32;
static constexpr int iniNzDimWidth = 16;

static uint64_t getElemBytesForAlign(Type t) {
  static constexpr uint64_t kBitsPerByte = 8;
  if (auto ft = dyn_cast<FloatType>(t)) {
    return (uint64_t)((ft.getWidth() + kBitsPerByte - 1) / kBitsPerByte);
  }
  if (auto it = dyn_cast<IntegerType>(t)) {
    return (uint64_t)((it.getWidth() + kBitsPerByte - 1) / kBitsPerByte);
  }
  if (isa<IndexType>(t)) {
    return 8ULL;
  }
  if (auto ct = dyn_cast<ComplexType>(t)) {
    return 2ULL * getElemBytesForAlign(ct.getElementType());
  }
  return 0ULL;
}

static uint64_t getBlockElemsFor32BAlign(Type elemType) {
  constexpr uint64_t kAlignBytes = 32;
  uint64_t elemBytes = getElemBytesForAlign(elemType);
  if (elemBytes == 0) {
    return 0;
  }
  if (elemBytes < 0 || kAlignBytes % elemBytes != 0) {
    return 0;
  }
  if (elemBytes >= kAlignBytes) {
    return 1;
  }
  return kAlignBytes / elemBytes;
}

std::pair<bool, bool>
InterCoreTransferAndSyncPass::analyzeMatmulOperand(DependencyInfo &dep) {
  bool isMatmulA = true;
  bool isTranspose = false;

  Value startValue = dep.value;

  llvm::DenseSet<Value> visited;
  SmallVector<Value> chain;
  chain.push_back(startValue);
  visited.insert(startValue);

  while (!chain.empty()) {
    Value currentValue = chain.back();
    chain.pop_back();

    for (Operation *userOp : currentValue.getUsers()) {
      if (CVPipeline::getOpBlockId(userOp) != dep.iniConsumerBlockId) {
        continue;
      }
      if (auto matmulOp = dyn_cast<linalg::MatmulOp>(userOp)) {
        auto operands = matmulOp->getOperands();
        if (operands[0] == currentValue) {
          isMatmulA = true;
        } else if (operands[1] == currentValue) {
          isMatmulA = false;
        } else {
          LOG_DEBUG("[error] matmul C is from Vector!");
          CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
        }

        Operation *defOp = currentValue.getDefiningOp();
        if (defOp && isa<linalg::TransposeOp>(defOp)) {
          isTranspose = true;
        }
        return {isMatmulA, isTranspose};
      }

      for (Value result : userOp->getResults()) {
        if (visited.insert(result).second) {
          chain.push_back(result);
        }
      }
    }
  }

  return {isMatmulA, isTranspose};
}

int InterCoreTransferAndSyncPass::getNzDimWidth(DependencyInfo &dep,
                                                int64_t blk) {
  auto [isA, isTransposed] = analyzeMatmulOperand(dep);
  if (blk == 32) {
    if (isA == isTransposed) {
      return iniNzDimWidth * 2;
    }
  }
  return iniNzDimWidth;
}

static void attachCommonTags(Operation *op, int blockId, StringRef coreType) {
  MLIRContext *ctx = op->getContext();
  setOpBlockId(op, blockId);
  setOpCoreType(op, coreType);
}

static void attachTransferTags(Operation *op, int blockId, StringRef coreType,
                               int transferId) {
  MLIRContext *ctx = op->getContext();
  setOpBlockId(op, blockId);
  setOpCoreType(op, coreType);
  op->setAttr(
      CVPipeline::kTransferId,
      IntegerAttr::get(IntegerType::get(ctx, kIntegerBitWidth), transferId));
}

static void attachMemCrossDeps(Operation *op, int tid, int seqId,
                               OpBuilder &builder) {
  op->setAttr(CVPipeline::kMemCrossDeps,
              builder.getArrayAttr({builder.getI32IntegerAttr(tid),
                                    builder.getI32IntegerAttr(seqId)}));
}

static void attachCrossCoreDeps(Operation *op, int tid, int seqId,
                                OpBuilder &builder) {
  op->setAttr(CVPipeline::kCrossCoreDeps,
              builder.getArrayAttr({builder.getI32IntegerAttr(tid),
                                    builder.getI32IntegerAttr(seqId)}));
}

static void attachAnalyzeFlagIdTag(Operation *op) {
  MLIRContext *ctx = op->getContext();
  op->setAttr(CVPipeline::kAnalyzeFlagId, UnitAttr::get(ctx));
}

/// Get the operation just before the terminating yield of the block that
/// contains \p op. Returns nullptr if the block has no yield terminator.
static Operation *getOpBeforeYield(Operation *op) {
  if (!op || !op->getBlock() || !op->getBlock()->getTerminator()) {
    return nullptr;
  }
  Operation *terminator = op->getBlock()->getTerminator();
  if (!isa<scf::YieldOp>(terminator)) {
    return nullptr;
  }
  return terminator->getPrevNode();
}

/// Get the sub-block id of \p op, or std::nullopt if \p op is null or does
/// not carry the sub-block tag.
static std::optional<int> getSubBlockId(Operation *op) {
  if (!op) {
    return std::nullopt;
  }
  auto attr = op->getAttrOfType<IntegerAttr>(CVPipeline::kSubBlock);
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static bool isChannelSplitNeeded(RankedTensorType tensorType) {
  static constexpr int32_t alignM = 16;
  return mlir::utils::getNumPerBlock(tensorType) == alignM / 2;
}

// Check whether a value is a valid C->C transfer candidate: at least one user
// whose block_id matches \p consumerBlockId must be a matmul op consuming the
// value as an A/B input. The value may also be used as init (outs) by other
// users (or the same matmul) — the replacement logic in handleCubeToCube
// ensures only the input operands are replaced.
static bool hasAnyMatmulABInputUser(Value value, int consumerBlockId) {
  for (Operation *user : value.getUsers()) {
    if (CVPipeline::getOpBlockId(user).value_or(-1) != consumerBlockId) {
      continue;
    }
    auto matmulUser = dyn_cast<linalg::MatmulOp>(user);
    if (!matmulUser) {
      continue;
    }
    for (Value input : matmulUser.getDpsInputs()) {
      if (input == value) {
        return true;
      }
    }
  }
  return false;
}

/// Check if \p value is a valid C2C matmul dependency.
/// Valid chains: matmul -> (intermediate_op)* -> matmul.
static bool isValidC2CMatmulDependency(Value value, int consumerBlockId) {
  // Walk back through valid intermediate ops to find the source matmul.
  Operation *defOp = CVPipeline::getSourceThroughCIntermediateOps(value);

  if (!isa_and_nonnull<linalg::MatmulOp>(defOp))
    return false;

  return hasAnyMatmulABInputUser(value, consumerBlockId);
}

static void attachLoopIdTag(Operation *op, int loopId) {
  MLIRContext *ctx = op->getContext();
  op->setAttr(
      CVPipeline::kLoopId,
      IntegerAttr::get(IntegerType::get(ctx, kIntegerBitWidth), loopId));
}

bool hasStartEndSyncWhitelistKernel(ModuleOp module) {
  std::vector<std::string> whitelist{
      "_swa_bwd_dq_kernel",
  };

  bool isWhitelistKernel = false;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (llvm::is_contained(whitelist, func.getName().str())) {
      isWhitelistKernel = true;
      break;
    }
  }
  return isWhitelistKernel;
}

// Block Start/End Operation Retrieval
std::pair<mlir::Operation *, mlir::Operation *>
InterCoreTransferAndSyncPass::getBlockStartEnd(int targetId,
                                               mlir::ModuleOp module) {
  mlir::Operation *knownOpInBlock = nullptr;
  module.walk<WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (knownOpInBlock) {
      return;
    }
    if (CVPipeline::getOpBlockId(op) == targetId) {
      knownOpInBlock = op;
    }
  });

  if (!knownOpInBlock) {
    return {nullptr, nullptr};
  }

  mlir::Block *block = knownOpInBlock->getBlock();
  if (!block)
    return {nullptr, nullptr};

  mlir::Operation *start = nullptr;
  mlir::Operation *end = nullptr;

  // Iterate through all operations in the current block
  for (Operation &op : *block) {
    auto blockIdOpt = CVPipeline::getOpBlockId(&op);
    if (!blockIdOpt) {
      continue;
    }
    int blockId = *blockIdOpt;
    if (!start) {
      if (targetId == blockId) {
        start = &op;
        end = &op;
      }
    } else {
      if (targetId == blockId) {
        end = &op;
      } else {
        break;
      }
    }
  }
  return {start, end};
}

std::pair<mlir::Operation *, mlir::Operation *>
InterCoreTransferAndSyncPass::getSubBlockStartEnd(mlir::Operation *defOp) {
  if (!defOp) {
    return {nullptr, nullptr};
  }
  auto targetSubBlockId = getSubBlockId(defOp);
  if (!targetSubBlockId) {
    return {nullptr, nullptr};
  }
  if (!defOp->getBlock()) {
    return {nullptr, nullptr};
  }
  mlir::Operation *subBlockStart = nullptr;
  mlir::Operation *subBlockEnd = nullptr;
  for (Operation &op : *defOp->getBlock()) {
    auto opSubBlockId = getSubBlockId(&op);
    if (!opSubBlockId) {
      continue;
    }
    if (!subBlockStart) {
      if (*opSubBlockId == *targetSubBlockId) {
        subBlockStart = &op;
        subBlockEnd = &op;
      }
    } else {
      if (*opSubBlockId == *targetSubBlockId) {
        subBlockEnd = &op;
      } else {
        break;
      }
    }
  }
  return {subBlockStart, subBlockEnd};
}

bool InterCoreTransferAndSyncPass::isOuterLayerDependency(
    size_t depIndex, mlir::Operation *currProdEnd,
    mlir::Operation *currConsStart,
    llvm::SmallVector<DependencyInfo> &memDependencies) {
  if (!currProdEnd || !currConsStart) {
    return false;
  }
  mlir::Block *currBlock = currProdEnd->getBlock();
  if (currBlock != currConsStart->getBlock()) {
    return false;
  }
  for (size_t i = 0; i < memDependencies.size(); ++i) {
    if (i == depIndex) {
      continue;
    }
    auto &otherDep = memDependencies[i];

    if (otherDep.type != memDependencies[depIndex].type) {
      continue;
    }

    auto [otherProdStart, otherProdEnd] =
        getBlockStartEnd(otherDep.producerBlockId, module);
    auto [otherConsStart, otherConsEnd] =
        getBlockStartEnd(otherDep.consumerBlockId, module);

    if (!otherProdEnd || !otherConsStart) {
      continue;
    }

    if (otherProdEnd->getBlock() != currBlock ||
        otherConsStart->getBlock() != currBlock) {
      continue;
    }

    // otherProdEnd is before currProdEnd
    // AND currConsStart is before otherConsStart
    bool isOtherInsideCurrent = !otherProdEnd->isBeforeInBlock(currProdEnd) &&
                                !currConsStart->isBeforeInBlock(otherConsStart);

    if (otherProdEnd == currProdEnd && otherConsStart == currConsStart) {
      if (i < depIndex) {
        // if otherDep has smaller index, current dep is outer layer and can be
        // skipped
        return true;
      }
    } else if (isOtherInsideCurrent) {
      return true;
    }
  }

  return false;
}

// Nd2NzNormalizer
SmallVector<int64_t>
InterCoreTransferAndSyncPass::computeExpectedShape(mlir::Value depValue,
                                                   int NzDimWidth) {
  auto tensorTy = dyn_cast<TensorType>(depValue.getType());
  static constexpr int NdShapeLength = 2;
  if (!tensorTy || tensorTy.getRank() != NdShapeLength) {
    LOG_DEBUG("source shape is not 2-dim!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return {};
  }

  int64_t M = tensorTy.getDimSize(0);
  int64_t N = tensorTy.getDimSize(1);

  // Compute bit width & Nwidth
  int64_t nWidth = getBlockElemsFor32BAlign(tensorTy.getElementType());
  if (nWidth == 0) {
    LOG_DEBUG("Unsupported element type for 32B alignment.\n");
    return {M, N};
  }

  int mRound = NzDimWidth;
  int nRound = nWidth;

  // Calculate newM / newN using the formula
  int64_t blM = (M + mRound - 1) / mRound;
  int64_t newM = blM * mRound;

  int64_t blN = (N + nRound - 1) / nRound;
  int64_t newN = blN * nRound;
  LOG_DEBUG("newM" << newM << "\n");
  LOG_DEBUG("newN" << newN << "\n");

  return {newM, newN}; // Return 2D shape
}

bool InterCoreTransferAndSyncPass::isExpectedShape(
    Value value, SmallVector<int64_t> &expectedShape) {
  auto tensorTy = dyn_cast<TensorType>(value.getType());
  if (!tensorTy) {
    return true;
  }
  ArrayRef<int64_t> currShape = tensorTy.getShape();
  bool isEqualedShape = currShape.equals(expectedShape);

  LOG_DEBUG("isEqualedShape" << isEqualedShape << "\n");
  return isEqualedShape;
}

// insert copyop before store to avoid mte3 blocking (store and V->C use the
// same PIPE)
mlir::Operation *InterCoreTransferAndSyncPass::getCopyPointBeforeStore(
    Value depValue, Operation *vectorEndOp, int iniProducerBlockId) {
  Operation *curr = vectorEndOp;
  Operation *firstStoreOpAfterProducer = nullptr;
  while (curr) {
    auto blockIdOpt = CVPipeline::getOpBlockId(curr);
    if (blockIdOpt != iniProducerBlockId) {
      break;
    }
    if (curr == depValue.getDefiningOp()) {
      break;
    }
    if (CVPipeline::isStoreLike(curr)) {
      firstStoreOpAfterProducer = curr->getPrevNode();
      LOG_DEBUG("firstStoreOpAfterProducer: " << *firstStoreOpAfterProducer
                                              << "\n");
    }
    curr = curr->getPrevNode();
  }
  return firstStoreOpAfterProducer;
}

// padding v->c tensor
mlir::Value InterCoreTransferAndSyncPass::alignShapeByInsertSlice(
    OpBuilder &builder, DependencyInfo &dep, Location loc,
    mlir::Value origValue, SmallVector<int64_t> expectedShape,
    int originBlockId) {
  auto origTensorType = dyn_cast<RankedTensorType>(origValue.getType());
  if (!origTensorType) {
    return origValue;
  }

  int64_t iniM = origTensorType.getDimSize(0);
  int64_t iniN = origTensorType.getDimSize(1);
  Type elemType = origTensorType.getElementType();
  if (isa<mlir::BlockArgument>(origValue)) {
    auto [originProdStart, originProdEnd] =
        getBlockStartEnd(originBlockId, module);
    builder.setInsertionPointAfter(originProdEnd);
  } else {
    builder.setInsertionPointAfter(origValue.getDefiningOp());
  }

  TypedAttr zeroAttr;
  if (auto floatElemTy = dyn_cast<FloatType>(elemType)) {
    zeroAttr = FloatAttr::get(
        floatElemTy, APFloat::getZero(floatElemTy.getFloatSemantics()));
  } else if (auto intElemTy = dyn_cast<IntegerType>(elemType)) {
    zeroAttr = IntegerAttr::get(intElemTy, 0);
  } else {
    LOG_DEBUG("Unsupported element type for alignShapeByInsertSlice: "
              << elemType << "\n");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return origValue;
  }
  auto zeroConstOp = builder.create<arith::ConstantOp>(loc, zeroAttr);
  auto tensorEmptyOp =
      builder.create<tensor::EmptyOp>(loc, expectedShape, elemType);
  auto linalgFillOp = builder.create<linalg::FillOp>(
      loc, zeroConstOp.getResult(), tensorEmptyOp.getResult());
  SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(0),
                                       builder.getIndexAttr(0)};
  SmallVector<OpFoldResult> insertsizes = {builder.getIndexAttr(iniM),
                                           builder.getIndexAttr(iniN)};
  SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1),
                                       builder.getIndexAttr(1)};
  auto tensorInsertSliceOp = builder.create<tensor::InsertSliceOp>(
      loc, origValue, linalgFillOp->getResult(0), offsets, insertsizes,
      strides);

  attachCommonTags(zeroConstOp, originBlockId, CVPipeline::kCoreTypeVector);
  attachCommonTags(tensorEmptyOp, originBlockId, CVPipeline::kCoreTypeVector);
  attachCommonTags(linalgFillOp, originBlockId, CVPipeline::kCoreTypeVector);
  attachCommonTags(tensorInsertSliceOp, originBlockId,
                   CVPipeline::kCoreTypeVector);

  return tensorInsertSliceOp.getResult();
}

void InterCoreTransferAndSyncPass::Nd2NzNormalize(OpBuilder &builder,
                                                  DependencyInfo &dep,
                                                  Location loc) {
  Value origValue = dep.value;
  Value newValue = origValue;
  // Step 0: Check if this Value has already been processed
  auto it = ndnzValueMapping.find(origValue);
  if (it != ndnzValueMapping.end()) {
    return;
  }

  Type elemType = cast<RankedTensorType>(origValue.getType()).getElementType();
  int64_t blk = getBlockElemsFor32BAlign(elemType);
  if (blk == 0) {
    LOG_DEBUG("Invalid block size.\n");
    return;
  }

  auto NzDimWidth = getNzDimWidth(dep, blk);

  // Step 1: Compute expected shape
  SmallVector<int64_t> expectedShape =
      computeExpectedShape(origValue, NzDimWidth);
  int originBlockId = dep.iniProducerBlockId;
  // Step 2: If shapes match, return original value
  bool isEqualedShape = isExpectedShape(origValue, expectedShape);
  LOG_DEBUG("newValue" << newValue << "\n");
  if (!isEqualedShape) {
    newValue = alignShapeByInsertSlice(builder, dep, loc, origValue,
                                       expectedShape, originBlockId);
  }

  // Step 3: insert nd2nz
  auto srcTensorType = cast<RankedTensorType>(newValue.getType());
  int64_t M = srcTensorType.getDimSize(0);
  int64_t N = srcTensorType.getDimSize(1);

  SmallVector<int64_t> shape3D = {M, N / blk, blk};
  SmallVector<int64_t> shapeTrans = {N / blk, M, blk};
  SmallVector<int64_t> shapeFinal = {N / blk, M / NzDimWidth, NzDimWidth, blk};

  auto type3D = RankedTensorType::get(shape3D, elemType);
  auto typeTrans = RankedTensorType::get(shapeTrans, elemType);
  auto typeFinal = RankedTensorType::get(shapeFinal, elemType);

  Operation *origDefOp = origValue.getDefiningOp();
  std::optional<int> subBlockId = getSubBlockId(origDefOp);

  auto [newProdStart, newProdEnd] =
      getBlockStartEnd(dep.producerBlockId, module);
  if (dep.iniProducerBlockId == dep.producerBlockId) {
    auto producerPoint =
        getCopyPointBeforeStore(newValue, newProdEnd, dep.iniProducerBlockId);
    if (producerPoint) {
      newProdEnd = producerPoint;
    }
    if (subBlockId) {
      if (Operation *subBlockEnd = getSubBlockStartEnd(origDefOp).second)
        newProdEnd = subBlockEnd;
    }
  }
  builder.setInsertionPointAfter(newProdEnd);

  auto reshape3Dcst =
      builder.create<arith::ConstantOp>(loc, builder.getI64TensorAttr(shape3D));
  auto reshape3DOp =
      builder.create<tensor::ReshapeOp>(loc, type3D, newValue, reshape3Dcst);

  auto emptyTrans = builder.create<tensor::EmptyOp>(loc, shapeTrans, elemType);
  SmallVector<int64_t> transposeOrder = {1, 0, 2};
  auto transposeOp = builder.create<linalg::TransposeOp>(
      loc, reshape3DOp.getResult(), emptyTrans.getResult(), transposeOrder);
  auto reshape4Dcst = builder.create<arith::ConstantOp>(
      loc, builder.getI64TensorAttr(shapeFinal));
  auto reshape4DOp = builder.create<tensor::ReshapeOp>(
      loc, typeFinal, transposeOp->getResult(0), reshape4Dcst);

  attachCommonTags(reshape3Dcst, originBlockId, CVPipeline::kCoreTypeVector);
  attachCommonTags(reshape3DOp, originBlockId, CVPipeline::kCoreTypeVector);
  attachCommonTags(emptyTrans, originBlockId, CVPipeline::kCoreTypeVector);
  attachCommonTags(transposeOp, originBlockId, CVPipeline::kCoreTypeVector);
  attachCommonTags(reshape4Dcst, originBlockId, CVPipeline::kCoreTypeVector);
  attachCommonTags(reshape4DOp, originBlockId, CVPipeline::kCoreTypeVector);
  if (subBlockId) {
    CVPipeline::setSubBlockId(reshape3Dcst, *subBlockId);
    CVPipeline::setSubBlockId(reshape3DOp, *subBlockId);
    CVPipeline::setSubBlockId(emptyTrans, *subBlockId);
    CVPipeline::setSubBlockId(transposeOp, *subBlockId);
    CVPipeline::setSubBlockId(reshape4Dcst, *subBlockId);
    CVPipeline::setSubBlockId(reshape4DOp, *subBlockId);
  }
  LOG_DEBUG("[reshape3DOp]: " << *reshape3DOp << "\n");
  LOG_DEBUG("[transposeOp]: " << *transposeOp << "\n");
  LOG_DEBUG("[reshape4DOp]: " << *reshape4DOp << "\n");
  ndnzValueMapping[origValue] = reshape4DOp.getResult();
}

// mark memref.alloc
mlir::Operation *InterCoreTransferAndSyncPass::annotateTightlyCoupledBuffer(
    OpBuilder &builder, Operation *allocOp, Location loc) {
  builder.setInsertionPointAfter(allocOp);
  auto markAllocOp =
      builder.create<annotation::MarkOp>(loc, allocOp->getResult(0));
  auto writeAttr = builder.getStringAttr("write");
  auto readAttr = builder.getStringAttr("read");
  auto effectsAttr = builder.getArrayAttr({writeAttr, readAttr});
  markAllocOp->setAttr("effects", effectsAttr);
  markAllocOp->setAttr(
      hivm::HIVMTightlyCoupledBufferAttr::name,
      HIVMTightlyCoupledBufferAttr::get(builder.getContext(), markAllocIndex));
  return markAllocOp;
}

// find the insert point for memref.alloc
Operation *
InterCoreTransferAndSyncPass::findMainLoopforTransfer(Operation *endOp,
                                                      Operation *startOp) {
  Operation *lca = endOp->getParentOp();
  Operation *current = lca;
  while (current) {
    if (isa<scf::ForOp, scf::WhileOp>(current)) {
      return current;
    }
    current = current->getParentOp();
  }
  return nullptr;
}

std::pair<Operation *, Operation *>
InterCoreTransferAndSyncPass::createTransferAllocs(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> shape, Type elemType,
    hivm::AddressSpace addrSpace, Operation *prodEndOp, Operation *consStartOp,
    int prodBlockId, int consBlockId, StringRef prodTag, StringRef consTag,
    int transferIndex) {
  auto addressSpaceAttr = builder.getAttr<hivm::AddressSpaceAttr>(addrSpace);
  auto allocType = MemRefType::get(shape, elemType, nullptr, addressSpaceAttr);

  Operation *prodAllocOp = nullptr;
  Operation *consAllocOp = nullptr;

  Operation *mainLoopOp = findMainLoopforTransfer(prodEndOp, consStartOp);

  if (mainLoopOp) {
    builder.setInsertionPoint(mainLoopOp);
    prodAllocOp = builder.create<memref::AllocOp>(loc, allocType);
    auto markProdOp = annotateTightlyCoupledBuffer(builder, prodAllocOp, loc);
    consAllocOp = builder.create<memref::AllocOp>(loc, allocType);
    auto markConsOp = annotateTightlyCoupledBuffer(builder, consAllocOp, loc);

    int loopBlockId = CVPipeline::getOpBlockId(mainLoopOp).value_or(-1);
    attachTransferTags(prodAllocOp, loopBlockId, prodTag, transferIndex);
    attachTransferTags(consAllocOp, loopBlockId, consTag, transferIndex);
    attachTransferTags(markProdOp, loopBlockId, prodTag, transferIndex);
    attachTransferTags(markConsOp, loopBlockId, consTag, transferIndex);

    builder.setInsertionPointAfter(prodEndOp);
  } else {
    builder.setInsertionPointAfter(prodEndOp);
    consAllocOp = builder.create<memref::AllocOp>(loc, allocType);
    auto markConsOp = annotateTightlyCoupledBuffer(builder, consAllocOp, loc);

    prodAllocOp = builder.create<memref::AllocOp>(loc, allocType);
    auto markProdOp = annotateTightlyCoupledBuffer(builder, prodAllocOp, loc);

    attachTransferTags(prodAllocOp, prodBlockId, prodTag, transferIndex);
    attachTransferTags(consAllocOp, prodBlockId, consTag, transferIndex);
    attachTransferTags(markProdOp, prodBlockId, prodTag, transferIndex);
    attachTransferTags(markConsOp, prodBlockId, consTag, transferIndex);
  }
  markAllocIndex++;

  return {prodAllocOp, consAllocOp};
}

mlir::Operation *InterCoreTransferAndSyncPass::analyzeConsumerReadInsertPoint(
    Value srcValue, int iniConsumerId) {
  llvm::DenseSet<mlir::Operation *> consumerOps;
  for (Operation *user : srcValue.getUsers()) {
    auto userBlockIdOpt = CVPipeline::getOpBlockId(user);
    if (userBlockIdOpt && *userBlockIdOpt == iniConsumerId) {
      consumerOps.insert(user);
    }
  }

  mlir::Operation *firstFoundOp = nullptr;

  module->walk<WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (consumerOps.contains(op)) {
      firstFoundOp = op;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  return firstFoundOp;
}

mlir::Operation *
InterCoreTransferAndSyncPass::getConsumerWaitPoint(int transferIndex) {
  mlir::Operation *consumerWaitPoint = nullptr;
  module.walk<WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (consumerWaitPoint) {
      return;
    }
    if (!isa<hivm::ConvertLayoutOp>(op) &&
        !isa<memref::MemorySpaceCastOp>(op) && !isa<memref::LoadOp>(op)) {
      return;
    }
    auto transferIdAttr =
        op->getAttrOfType<IntegerAttr>(CVPipeline::kTransferId);
    if (transferIdAttr && transferIdAttr.getInt() == transferIndex) {
      consumerWaitPoint = op;
    }
  });
  return consumerWaitPoint;
}

mlir::Operation *InterCoreTransferAndSyncPass::getFixpipePointAfterProducer(
    Value srcValue, int producerBlockId) {
  mlir::Operation *fixpipePoint = srcValue.getDefiningOp();
  int blockId = CVPipeline::getOpBlockId(fixpipePoint).value_or(-1);
  if (blockId != producerBlockId) {
    return nullptr;
  }
  return fixpipePoint;
}

Operation *InterCoreTransferAndSyncPass::insertVectorToCubeTransfer(
    OpBuilder &builder, Value srcValue, Value normalizedValue,
    Operation *vectorEndOp, Operation *cubeStartOp, Location loc,
    int transferIndex, DependencyInfo &dep, bool is1DTensor,
    Operation **consumedDataOp) {
  mlir::Operation *sendOp = nullptr;
  mlir::Operation *receiveOp = nullptr;
  Value receiveValue;

  int vecBlockId = CVPipeline::getOpBlockId(vectorEndOp).value_or(-1);
  int cubeBlockId = CVPipeline::getOpBlockId(cubeStartOp).value_or(-1);

  if (isScalarDependency(dep.value)) {
    builder.setInsertionPointAfter(vectorEndOp);
    SmallVector<Operation *> writeOps;
    LOG_DEBUG("before writeToSSBuffer\n");
    auto addrOpt = ssbufferManager.writeToSSBuffer(srcValue, builder, writeOps);
    if (!addrOpt) {
      LOG_DEBUG("[v->c] Failed to write scalar value to SSBuffer\n");
      return nullptr;
    }
    int64_t addr = *addrOpt;
    LOG_DEBUG("after writeToSSBuffer\n");
    Operation *storeOp = nullptr;
    for (Operation *op : writeOps) {
      attachTransferTags(op, vecBlockId, CVPipeline::kCoreTypeVector,
                         transferIndex);
      if (isa<memref::StoreOp>(op)) {
        storeOp = op;
      }
    }
    sendOp = storeOp;
    attachCrossCoreDeps(sendOp, transferIndex, CVPipeline::crossCoreProducerId,
                        builder);
    LOG_DEBUG("before readFromSSBuffer\n");
    builder.setInsertionPoint(cubeStartOp);
    SmallVector<Operation *> readOps;
    auto loadedValueOpt =
        ssbufferManager.readFromSSBuffer(addr, builder, readOps);
    if (!loadedValueOpt) {
      LOG_DEBUG("[v->c] Failed to read scalar value from SSBuffer\n");
      return nullptr;
    }
    receiveValue = *loadedValueOpt;
    LOG_DEBUG("after readFromSSBuffer\n");
    Operation *loadOp = nullptr;
    for (Operation *op : readOps) {
      attachTransferTags(op, cubeBlockId, CVPipeline::kCoreTypeCube,
                         transferIndex);
      if (isa<memref::LoadOp>(op)) {
        loadOp = op;
      }
    }
    receiveOp = loadOp;
    attachCrossCoreDeps(receiveOp, transferIndex,
                        CVPipeline::crossCoreConsumerId, builder);
  } else {
    // Step 1: Get input information (2D tensor: MxN)
    auto srcTensorType = cast<RankedTensorType>(srcValue.getType());
    auto normalizedTensorType =
        cast<RankedTensorType>(normalizedValue.getType());
    Type elemType = srcTensorType.getElementType();

    auto [vecAllocOp, cubeAllocOp] = createTransferAllocs(
        builder, loc, normalizedTensorType.getShape(), elemType,
        hivm::AddressSpace::L1, vectorEndOp, cubeStartOp, vecBlockId,
        cubeBlockId, CVPipeline::kCoreTypeVector, CVPipeline::kCoreTypeCube,
        transferIndex);

    auto copyOp = builder.create<hivm::CopyOp>(
        loc, mlir::TypeRange{}, normalizedValue, vecAllocOp->getResult(0));

    attachTransferTags(copyOp, vecBlockId, CVPipeline::kCoreTypeVector,
                       transferIndex);
    attachCrossCoreDeps(copyOp, transferIndex, CVPipeline::crossCoreProducerId,
                        builder);
    // Propagate the sub-block tag from the src defining op to the copy op.
    if (Operation *srcDefOp = srcValue.getDefiningOp()) {
      if (auto subBlockId = getSubBlockId(srcDefOp)) {
        CVPipeline::setSubBlockId(copyOp, *subBlockId);
      }
    }
    LOG_DEBUG("[copyOp]: " << *copyOp << "\n");

    builder.setInsertionPoint(cubeStartOp);

    Value memValue = cubeAllocOp->getResult(0);
    if (!is1DTensor) {
      auto nzLayout =
          hivm::DataLayoutAttr::get(builder.getContext(), hivm::DataLayout::nZ);
      auto ndLayout =
          hivm::DataLayoutAttr::get(builder.getContext(), hivm::DataLayout::ND);
      auto cbufaddressSpaceAttr =
          builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::L1);
      auto newAllocType = MemRefType::get(srcTensorType.getShape(), elemType,
                                          nullptr, cbufaddressSpaceAttr);
      auto convertLayoutOp =
          builder.create<hivm::ConvertLayoutOp>(loc, newAllocType, memValue,
                                                nzLayout, // srcLayout
                                                ndLayout  // dstLayout
          );
      memValue = convertLayoutOp.getResult();
      attachTransferTags(convertLayoutOp, cubeBlockId,
                         CVPipeline::kCoreTypeCube, transferIndex);
      attachCrossCoreDeps(convertLayoutOp, transferIndex,
                          CVPipeline::crossCoreConsumerId, builder);
    }
    auto plainMemrefType = MemRefType::get(srcTensorType.getShape(), elemType);
    auto memspaceCastOp = builder.create<memref::MemorySpaceCastOp>(
        loc, plainMemrefType, memValue);
    auto toTensorOp = builder.create<bufferization::ToTensorOp>(
        loc, srcTensorType, memspaceCastOp.getResult(), true, true);

    attachTransferTags(memspaceCastOp, cubeBlockId, CVPipeline::kCoreTypeCube,
                       transferIndex);
    if (is1DTensor) {
      attachCrossCoreDeps(memspaceCastOp, transferIndex,
                          CVPipeline::crossCoreConsumerId, builder);
    }
    attachTransferTags(toTensorOp, cubeBlockId, CVPipeline::kCoreTypeCube,
                       transferIndex);
    LOG_DEBUG("[toTensorOp]: " << *toTensorOp << "\n");
    sendOp = copyOp;
    receiveOp = toTensorOp;
    receiveValue = toTensorOp.getResult();
  }

  llvm::SmallVector<Operation *> users(srcValue.getUsers().begin(),
                                       srcValue.getUsers().end());
  if (dep.consumerYieldOp) {
    dep.consumerYieldOp->replaceUsesOfWith(srcValue, receiveValue);
  }
  for (Operation *user : users) {
    LOG_DEBUG("[v->c user]" << *user << "\n");
    auto userBlockIdOpt = CVPipeline::getOpBlockId(user);
    if (userBlockIdOpt && *userBlockIdOpt == dep.iniConsumerBlockId) {
      user->replaceUsesOfWith(srcValue, receiveValue);
    }
  }
  if (consumedDataOp) {
    *consumedDataOp = receiveOp;
  }
  return sendOp;
}

Operation *InterCoreTransferAndSyncPass::insertCubeToVectorTransfer(
    OpBuilder &builder, Value srcValue, Operation *cubeEndOp,
    Operation *vectorStartOp, Location loc, int transferIndex,
    DependencyInfo &dep, Operation **consumedDataOp) {
  LOG_DEBUG("Inserting [Cube->Vector] transfer for value: " << srcValue
                                                            << "\n");
  auto srcTensorType = cast<RankedTensorType>(srcValue.getType());
  int64_t M = srcTensorType.getDimSize(0);
  int64_t N = srcTensorType.getDimSize(1);
  Type elemType = srcTensorType.getElementType();

  int cubeBlockId =
      CVPipeline::getOpBlockId(srcValue.getDefiningOp()).value_or(-1);
  int vecBlockId = CVPipeline::getOpBlockId(vectorStartOp).value_or(-1);

  auto targetShape = dep.isAllTranspoesd ? std::vector<int64_t>{N, M}
                                         : std::vector<int64_t>{M, N};
  auto targetTensorType = RankedTensorType::get(targetShape, elemType);
  auto [cubeAllocOp, vecAllocOp] = createTransferAllocs(
      builder, loc, targetShape, elemType, hivm::AddressSpace::UB, cubeEndOp,
      vectorStartOp, cubeBlockId, vecBlockId, CVPipeline::kCoreTypeCube,
      CVPipeline::kCoreTypeVector, transferIndex);
  auto dmaModeAttr = FixpipeDMAModeAttr::get(
      builder.getContext(),
      dep.isAllTranspoesd ? FixpipeDMAMode::NZ2DN : FixpipeDMAMode::NZ2ND);

  auto realValue = dep.value;
  if (dep.realValue) {
    realValue = dep.realValue;
  }
  auto fixpipeOp = builder.create<hivm::FixpipeOp>(
      loc, mlir::TypeRange{},    // No return value
      realValue,                 // src
      cubeAllocOp->getResult(0), // dst
      mlir::ValueRange{}, dmaModeAttr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, mlir::ArrayAttr{}, nullptr);
  attachTransferTags(fixpipeOp, cubeBlockId, CVPipeline::kCoreTypeCube,
                     transferIndex);
  attachCrossCoreDeps(fixpipeOp, transferIndex, CVPipeline::crossCoreProducerId,
                      builder);
  LOG_DEBUG("[fixpipeOp]: " << *fixpipeOp << "\n");

  // Vector side: memspace_cast + to_tensor
  builder.setInsertionPoint(vectorStartOp);

  auto plainMemrefType = MemRefType::get(targetShape, elemType);
  auto memspaceCastOp = builder.create<memref::MemorySpaceCastOp>(
      loc, plainMemrefType, vecAllocOp->getResult(0));

  auto toTensorOp = builder.create<bufferization::ToTensorOp>(
      loc, targetTensorType, memspaceCastOp.getResult(), true, true);

  attachTransferTags(memspaceCastOp, vecBlockId, CVPipeline::kCoreTypeVector,
                     transferIndex);
  attachCrossCoreDeps(memspaceCastOp, transferIndex,
                      CVPipeline::crossCoreConsumerId, builder);
  attachTransferTags(toTensorOp, vecBlockId, CVPipeline::kCoreTypeVector,
                     transferIndex);
  // Propagate the sub-block tag from vectorStartOp to the vector-side ops.
  if (auto subBlockId = getSubBlockId(vectorStartOp)) {
    CVPipeline::setSubBlockId(memspaceCastOp, *subBlockId);
    CVPipeline::setSubBlockId(toTensorOp, *subBlockId);
  }
  LOG_DEBUG("[toTensorOp]: " << *toTensorOp << "\n");

  if (dep.isAllTranspoesd) {
    for (auto *userOp : srcValue.getUsers()) {
      if (isa<linalg::TransposeOp>(userOp)) {
        srcValue = userOp->getResults()[0];
        break;
      }
    }
  }
  llvm::SmallVector<Operation *> users(srcValue.getUsers().begin(),
                                       srcValue.getUsers().end());
  if (dep.consumerYieldOp) {
    dep.consumerYieldOp->replaceUsesOfWith(srcValue, toTensorOp.getResult());
  }
  for (Operation *user : users) {
    LOG_DEBUG("[c->v user]" << *user << "\n");
    auto userBlockIdOpt = CVPipeline::getOpBlockId(user);
    if (userBlockIdOpt && *userBlockIdOpt == dep.iniConsumerBlockId) {
      user->replaceUsesOfWith(srcValue, toTensorOp.getResult());
    }
  }

  if (consumedDataOp) {
    *consumedDataOp = toTensorOp;
  }
  return fixpipeOp;
}

TransferPipeConfig
InterCoreTransferAndSyncPass::getTransferPipeConfig(Operation *transferOp,
                                                    bool isStoreDirectly) {
  auto cubeCoreAttr =
      hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE);
  auto vecCoreAttr =
      hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR);
  auto pipeFixAttr = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_FIX);
  auto pipeVAttr = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_V);
  auto pipeMte3Attr = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_MTE3);
  auto pipeMte1Attr = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_MTE1);
  auto pipeSAttr = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
  TransferPipeConfig config;
  if (isa<hivm::FixpipeOp>(transferOp)) {
    if (isStoreDirectly) {
      config.forReadTPipe = pipeFixAttr;
      config.forReadPipe = pipeMte3Attr;
      config.forWriteTPipe = pipeMte3Attr;
      config.forWritePipe = pipeFixAttr;
      config.srcCoreAttr = cubeCoreAttr;
      config.dstCoreAttr = vecCoreAttr;
      config.srcCoreType = CVPipeline::kCoreTypeCube;
      config.dstCoreType = CVPipeline::kCoreTypeVector;
    } else {
      config.forReadTPipe = pipeFixAttr;
      config.forReadPipe = pipeVAttr;
      config.forWriteTPipe = pipeVAttr;
      config.forWritePipe = pipeFixAttr;
      config.srcCoreAttr = cubeCoreAttr;
      config.dstCoreAttr = vecCoreAttr;
      config.srcCoreType = CVPipeline::kCoreTypeCube;
      config.dstCoreType = CVPipeline::kCoreTypeVector;
    }
  } else if (isa<hivm::CopyOp>(transferOp)) {
    config.forReadTPipe = pipeMte3Attr;
    config.forReadPipe = pipeMte1Attr;
    config.forWriteTPipe = pipeMte1Attr;
    config.forWritePipe = pipeMte3Attr;
    config.srcCoreAttr = vecCoreAttr;
    config.dstCoreAttr = cubeCoreAttr;
    config.srcCoreType = CVPipeline::kCoreTypeVector;
    config.dstCoreType = CVPipeline::kCoreTypeCube;
  } else if (isa<memref::StoreOp>(transferOp)) {
    // Scalar sync uses PIPE_S to stay isolated from tensor flag space.
    config.forReadTPipe = pipeSAttr;
    config.forReadPipe = pipeSAttr;
    config.forWriteTPipe = pipeSAttr;
    config.forWritePipe = pipeSAttr;
    config.srcCoreAttr = vecCoreAttr;
    config.dstCoreAttr = cubeCoreAttr;
    config.srcCoreType = CVPipeline::kCoreTypeVector;
    config.dstCoreType = CVPipeline::kCoreTypeCube;
  }
  return config;
}

// Check if a value fixpiped to ub is directly connected to a storage op
// Skip ViewLikeOpInterface and ExtractSliceOp during traversal
bool InterCoreTransferAndSyncPass::isStoreDirectlyInUserChain(
    Value toTensorValue) {
  // Traverse user chain starting from toTensorValue
  llvm::SmallVector<Value> workList = {toTensorValue};
  llvm::DenseSet<Value> visited;
  bool hasStore = false;
  while (!workList.empty()) {
    Value currVal = workList.pop_back_val();
    if (visited.count(currVal)) {
      continue;
    }
    visited.insert(currVal);

    for (Operation *user : currVal.getUsers()) {
      // Check if user is a storage op
      if (CVPipeline::isStoreLike(user)) {
        hasStore = true;
        continue;
      }

      // Check if user is in skip range
      if (CVPipeline::isViewLike(user) || CVPipeline::isZeroAdd(user) ||
          user->hasAttr(CVPipeline::kForMayNotExec)) {
        // Continue traversing through skip ops
        for (Value result : user->getResults()) {
          if (!visited.count(result)) {
            workList.push_back(result);
          }
        }
      } else {
        return false;
      }
    }
  }
  return hasStore;
}

void InterCoreTransferAndSyncPass::insertInterCoreSync(
    OpBuilder &builder, Operation *transferOp, Operation *consumerStartOp,
    Operation *consumerEndOp, int flag, Location loc, int transferIndex,
    FlagIdReuseManager &flagIdReuseManager, Operation *consumedDataOp,
    bool isStoreDirectly) {
  LOG_DEBUG("Inserting inter-core synchronization for transferOp: "
            << *transferOp << "\n");

  auto flagId = builder.getIntegerAttr(builder.getI64Type(), flag);

  int producerBlockId = CVPipeline::getOpBlockId(transferOp).value_or(-1);
  int consumerBlockId = CVPipeline::getOpBlockId(consumerStartOp).value_or(-1);

  Operation *mainLoopOp = findMainLoopforTransfer(transferOp, consumerStartOp);

  auto config = getTransferPipeConfig(transferOp, isStoreDirectly);

  builder.setInsertionPointAfter(transferOp);
  auto setOpForRead = builder.create<SyncBlockSetOp>(
      loc, config.srcCoreAttr, config.forReadTPipe, config.forReadPipe, flagId);
  attachTransferTags(setOpForRead, producerBlockId, config.srcCoreType,
                     transferIndex);
  // Propagate the sub-block tag from transferOp to the sync ops.
  if (auto subBlockId = getSubBlockId(transferOp)) {
    CVPipeline::setSubBlockId(setOpForRead, *subBlockId);
  }

  builder.setInsertionPoint(consumerStartOp);
  auto waitOpForRead = builder.create<SyncBlockWaitOp>(
      loc, config.dstCoreAttr, config.forReadTPipe, config.forReadPipe, flagId);
  attachTransferTags(waitOpForRead, consumerBlockId, config.dstCoreType,
                     transferIndex);
  if (auto subBlockId = getSubBlockId(consumerStartOp)) {
    CVPipeline::setSubBlockId(waitOpForRead, *subBlockId);
  }

  if (mainLoopOp) {
    builder.setInsertionPoint(transferOp);
    auto waitOpForWrite = builder.create<SyncBlockWaitOp>(
        loc, config.srcCoreAttr, config.forWriteTPipe, config.forWritePipe,
        flagId);
    attachTransferTags(waitOpForWrite, producerBlockId, config.srcCoreType,
                       transferIndex);

    builder.setInsertionPointAfter(consumerEndOp);
    auto setOpForWrite = builder.create<SyncBlockSetOp>(
        loc, config.dstCoreAttr, config.forWriteTPipe, config.forWritePipe,
        flagId);
    attachTransferTags(setOpForWrite, consumerBlockId, config.dstCoreType,
                       transferIndex);

    builder.setInsertionPoint(mainLoopOp);
    auto setOpForStart = builder.create<SyncBlockSetOp>(
        loc, config.dstCoreAttr, config.forWriteTPipe, config.forWritePipe,
        flagId);
    builder.setInsertionPointAfter(mainLoopOp);
    auto waitOpForEnd = builder.create<SyncBlockWaitOp>(
        loc, config.srcCoreAttr, config.forWriteTPipe, config.forWritePipe,
        flagId);

    int startEndBlockId = CVPipeline::getOpBlockId(mainLoopOp).value_or(-1);
    attachTransferTags(setOpForStart, startEndBlockId, config.dstCoreType,
                       transferIndex);
    attachTransferTags(waitOpForEnd, startEndBlockId, config.srcCoreType,
                       transferIndex);

    if (auto subBlockId = getSubBlockId(transferOp)) {
      CVPipeline::setSubBlockId(waitOpForWrite, *subBlockId);
    }
    if (auto subBlockId = getSubBlockId(consumerEndOp)) {
      CVPipeline::setSubBlockId(setOpForWrite, *subBlockId);
    }

    attachAnalyzeFlagIdTag(setOpForRead);
    attachAnalyzeFlagIdTag(waitOpForRead);
    attachAnalyzeFlagIdTag(waitOpForWrite);
    attachAnalyzeFlagIdTag(setOpForWrite);
    attachAnalyzeFlagIdTag(setOpForStart);
    attachAnalyzeFlagIdTag(waitOpForEnd);

    if (hasStartEndSyncWhitelistKernel(module)) {
      auto loopId =
          mainLoopOp->getAttrOfType<IntegerAttr>(CVPipeline::kLoopId).getInt();
      attachLoopIdTag(setOpForStart, loopId);
      attachLoopIdTag(waitOpForEnd, loopId);
    }

    // E2: register every set->wait pair of this transfer, not just the
    // loop start/end pair. Each pair is the only proof of cross-core
    // ordering for the sync ops it connects.
    flagIdReuseManager.insertRelationBetweenSetAndWait(setOpForRead,
                                                       waitOpForRead);
    flagIdReuseManager.insertRelationBetweenSetAndWait(setOpForWrite,
                                                       waitOpForWrite);
    flagIdReuseManager.insertRelationBetweenSetAndWait(setOpForStart,
                                                       waitOpForEnd);
    // E4: link the read-wait to the consumed data it guards so the sync
    // op is threaded into the downstream dataflow graph.
    flagIdReuseManager.insertRelationBetweenSetAndWait(waitOpForRead,
                                                       consumedDataOp);
    return;
  }
  attachAnalyzeFlagIdTag(setOpForRead);
  attachAnalyzeFlagIdTag(waitOpForRead);
  flagIdReuseManager.insertRelationBetweenSetAndWait(setOpForRead,
                                                     waitOpForRead);
  flagIdReuseManager.insertRelationBetweenSetAndWait(waitOpForRead,
                                                     consumedDataOp);
  return;
}

bool hasMemDepSyncWhitelistKernel(ModuleOp module) {
  std::vector<std::string> whitelist{
      "_hstu_attn_fwd",
      "parallel_path_fwd_kernel",
  };

  // Check if any func name matches the whitelist
  bool hasWhitelistedKernel = false;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (llvm::is_contained(whitelist, func.getName().str())) {
      hasWhitelistedKernel = true;
      break;
    }
  }
  return hasWhitelistedKernel;
}

void InterCoreTransferAndSyncPass::insertMemDepSync(
    OpBuilder &builder, Operation *producerStartOp, Operation *producerEndOp,
    Operation *consumerStartOp, Operation *consumerEndOp, int flag,
    Location loc, bool isCubeToVector, FlagIdReuseManager &flagIdReuseManager) {
  LOG_DEBUG("Inserting Memdep sync: "
            << (isCubeToVector ? "CUBE->VECTOR" : "VECTOR->CUBE")
            << ", flag = " << flag << "\n");

  // CUBE -> VECTOR: srcPipe = PIPE_FIX, srcCoreType = CUBE, dstCoreType =
  // VECTOR VECTOR -> CUBE: srcPipe = PIPE_MTE3, srcCoreType = VECTOR,
  // dstCoreType = CUBE
  hivm::PIPE srcPipe =
      isCubeToVector ? hivm::PIPE::PIPE_FIX : hivm::PIPE::PIPE_MTE3;
  hivm::TCoreType srcCoreType =
      isCubeToVector ? hivm::TCoreType::CUBE : hivm::TCoreType::VECTOR;
  hivm::TCoreType dstCoreType =
      isCubeToVector ? hivm::TCoreType::VECTOR : hivm::TCoreType::CUBE;
  hivm::PIPE dstPipe = hivm::PIPE::PIPE_MTE2;

  auto srcCoreAttr =
      hivm::TCoreTypeAttr::get(builder.getContext(), srcCoreType);
  auto dstCoreAttr =
      hivm::TCoreTypeAttr::get(builder.getContext(), dstCoreType);
  auto srcPipeAttr = PipeAttr::get(builder.getContext(), srcPipe);
  auto dstPipeAttr = PipeAttr::get(builder.getContext(), dstPipe);
  auto flagId = builder.getIntegerAttr(builder.getI64Type(), flag);

  builder.setInsertionPointAfter(producerEndOp);
  auto setOp = builder.create<SyncBlockSetOp>(loc, srcCoreAttr, srcPipeAttr,
                                              dstPipeAttr, flagId);

  builder.setInsertionPoint(consumerStartOp);
  auto waitOp = builder.create<SyncBlockWaitOp>(loc, dstCoreAttr, srcPipeAttr,
                                                dstPipeAttr, flagId);

  auto prodBlockIdOpt = CVPipeline::getOpBlockId(producerEndOp);
  auto consBlockIdOpt = CVPipeline::getOpBlockId(consumerStartOp);
  StringRef prodCoreType =
      isCubeToVector ? CVPipeline::kCoreTypeCube : CVPipeline::kCoreTypeVector;
  StringRef consCoreType =
      isCubeToVector ? CVPipeline::kCoreTypeVector : CVPipeline::kCoreTypeCube;
  if (prodBlockIdOpt) {
    attachCommonTags(setOp, *prodBlockIdOpt, prodCoreType);
  }
  if (consBlockIdOpt) {
    attachCommonTags(waitOp, *consBlockIdOpt, consCoreType);
  }
  attachAnalyzeFlagIdTag(setOp);
  attachAnalyzeFlagIdTag(waitOp);
  flagIdReuseManager.insertRelationBetweenSetAndWait(setOp, waitOp);
  if (hasMemDepSyncWhitelistKernel(module)) {
    Operation *mainLoopOp =
        findMainLoopforTransfer(producerEndOp, consumerStartOp);
    if (mainLoopOp) {
      builder.setInsertionPoint(producerStartOp);
      auto waitOpForWrite = builder.create<SyncBlockWaitOp>(
          loc, srcCoreAttr, dstPipeAttr, srcPipeAttr, flagId);
      attachCommonTags(waitOpForWrite, *prodBlockIdOpt, prodCoreType);

      builder.setInsertionPointAfter(consumerEndOp);
      auto setOpForWrite = builder.create<SyncBlockSetOp>(
          loc, dstCoreAttr, dstPipeAttr, srcPipeAttr, flagId);
      attachCommonTags(setOpForWrite, *consBlockIdOpt, consCoreType);

      builder.setInsertionPoint(mainLoopOp);
      auto setOpForStart = builder.create<SyncBlockSetOp>(
          loc, dstCoreAttr, dstPipeAttr, srcPipeAttr, flagId);
      builder.setInsertionPointAfter(mainLoopOp);
      auto waitOpForEnd = builder.create<SyncBlockWaitOp>(
          loc, srcCoreAttr, dstPipeAttr, srcPipeAttr, flagId);

      int startEndBlockId = CVPipeline::getOpBlockId(mainLoopOp).value_or(-1);
      attachCommonTags(setOpForStart, startEndBlockId, consCoreType);
      attachCommonTags(waitOpForEnd, startEndBlockId, prodCoreType);

      attachAnalyzeFlagIdTag(waitOpForWrite);
      attachAnalyzeFlagIdTag(setOpForWrite);
      attachAnalyzeFlagIdTag(setOpForStart);
      attachAnalyzeFlagIdTag(waitOpForEnd);
      flagIdReuseManager.insertRelationBetweenSetAndWait(setOpForWrite,
                                                         waitOpForWrite);
      flagIdReuseManager.insertRelationBetweenSetAndWait(setOpForStart,
                                                         waitOpForEnd);
    }
  }
  LOG_DEBUG("[PIPE_MTE2 setOp]: " << *setOp << "\n");
  LOG_DEBUG("[PIPE_MTE2 waitOp]: " << *waitOp << "\n");
}

namespace {

// Walk `scfOp`'s regions looking for an `scf::YieldOp` that directly yields a
// VECTOR-side `bufferization.to_tensor` backed by a memref.alloc annotated with
// `hivm.tightly_coupled_buffer<N>`.
std::optional<VectorToTensorInfo> findVectorToTensorInScfOp(Operation *scfOp) {
  std::optional<VectorToTensorInfo> result;
  scfOp->walk([&](scf::YieldOp yield) {
    if (result) {
      return;
    }
    for (Value yielded : yield.getResults()) {
      auto toTensor = yielded.getDefiningOp<bufferization::ToTensorOp>();
      if (!toTensor) {
        continue;
      }
      auto coreAttr =
          toTensor->getAttrOfType<StringAttr>(CVPipeline::kCoreType);
      if (!coreAttr || coreAttr.getValue() != CVPipeline::kCoreTypeVector) {
        continue;
      }
      Value memref = CVPipeline::traceBackToMemrefAlloc(toTensor.getBuffer());
      auto alloc = memref.getDefiningOp<memref::AllocOp>();
      if (!alloc) {
        continue;
      }
      auto id = CVPipeline::getTightlyCoupledBufferId(alloc);
      if (!id.has_value()) {
        continue;
      }
      result = VectorToTensorInfo{toTensor, yield, *id};
      return;
    }
  });
  return result;
}

// Walk downstream from `scfResult` through view-like ops looking for a
// store-like op. Returns the store op or null.
Operation *findStoreLikeAfterScfOp(Value scfResult) {
  SmallVector<Operation *> worklist;
  for (Operation *user : scfResult.getUsers()) {
    if (CVPipeline::isStoreLike(user)) {
      return user;
    }
    if (CVPipeline::isViewLike(user)) {
      worklist.push_back(user);
    }
  }
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    for (Operation *user : current->getUsers()) {
      if (CVPipeline::isStoreLike(user)) {
        return user;
      }
      if (CVPipeline::isViewLike(user)) {
        worklist.push_back(user);
      }
    }
  }
  return nullptr;
}

} // namespace

std::optional<CubeToVectorDirectStoreInfo>
InterCoreTransferAndSyncPass::matchCubeToVectorDirectStorePattern(
    Operation *scfOp) {
  if (!isa<scf::ForOp, scf::WhileOp, scf::IfOp>(scfOp)) {
    return std::nullopt;
  }

  auto anchor = findVectorToTensorInScfOp(scfOp);
  if (!anchor) {
    return std::nullopt;
  }

  Value scfResult;
  for (auto [idx, yielded] : llvm::enumerate(anchor->yield.getResults())) {
    if (yielded == anchor->toTensor.getResult()) {
      if (idx < scfOp->getNumResults()) {
        scfResult = scfOp->getResult(idx);
      }
      break;
    }
  }
  if (!scfResult) {
    return std::nullopt;
  }

  Operation *storeOp = findStoreLikeAfterScfOp(scfResult);
  if (!storeOp) {
    return std::nullopt;
  }

  CubeToVectorDirectStoreInfo info;
  info.tightlyCoupledBufferId = anchor->tightlyCoupledBufferId;
  info.storeOp = storeOp;
  return info;
}

// Match CUBE -> VECTOR direct store: data reaches VECTOR through UB and is
// stored without VECTOR tensor computation. The extra sync guards the MTE3
// store after the SCF region.
// Remove VECTOR add-from-matmul pseudo-ops from SCF yields.
void InterCoreTransferAndSyncPass::removeVectorPseudoOps() {
  LOG_DEBUG("Removing VECTOR pseudo-ops (addf/addi carrying "
            "ssbuffer.add_from_matmul)...\n");

  module.walk([&](Operation *op) {
    if (!isa<arith::AddFOp, arith::AddIOp>(op)) {
      return;
    }
    // (1) Must carry the marker attributes.
    if (!op->hasAttr(CVPipeline::kAddFromMatmul)) {
      return;
    }
    auto coreAttr = op->getAttrOfType<StringAttr>(CVPipeline::kCoreType);
    if (!coreAttr || coreAttr.getValue() != CVPipeline::kCoreTypeVector) {
      return;
    }
    if (op->getNumOperands() != 2) {
      return;
    }
    // (2) One operand must be a zero-filled tensor; the other is the
    // data-flow source we want to keep yielding.
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Value keptOperand = nullptr;
    if (CVPipeline::isZeroFillValue(lhs)) {
      keptOperand = rhs;
    } else if (CVPipeline::isZeroFillValue(rhs)) {
      keptOperand = lhs;
    } else {
      return; // No zero-fill operand: not the pseudo-op.
    }
    LOG_DEBUG("[pseudo-op] removing " << *op << "\n");
    op->getResult(0).replaceAllUsesWith(keptOperand);
    op->erase();
  });
}

void InterCoreTransferAndSyncPass::processCubeToVectorDirectStoreSync(
    OpBuilder &builder, FlagIdManager &flagManager,
    FlagIdReuseManager &flagIdReuseManager) {
  LOG_DEBUG("Processing cube-to-vector-direct-store sync pattern...\n");

  module.walk([&](Operation *op) {
    if (!isa<scf::ForOp, scf::WhileOp, scf::IfOp>(op)) {
      return;
    }
    auto match = matchCubeToVectorDirectStorePattern(op);
    if (!match.has_value()) {
      return;
    }
    auto &info = *match;
    Operation *storeOp = info.storeOp;

    int flagId = flagManager.acquireId();
    Location loc = op->getLoc();
    auto flagIdAttr = builder.getIntegerAttr(builder.getI64Type(), flagId);
    auto cubeAttr = TCoreTypeAttr::get(builder.getContext(), TCoreType::CUBE);
    auto vecAttr = TCoreTypeAttr::get(builder.getContext(), TCoreType::VECTOR);
    auto pipeFixAttr = PipeAttr::get(builder.getContext(), PIPE::PIPE_FIX);
    auto pipeMte3Attr = PipeAttr::get(builder.getContext(), PIPE::PIPE_MTE3);

    // Both set/wait share the same transfer id for this post-region store
    // path; advance the class-level counter so subsequent transfers keep
    // getting unique ids.
    int syncTransferId = transferIndex++;

    // Insert the set after the SCF op and any trailing sync_block_* ops
    // already appended by previous handlers in this pass.
    Operation *setInsertPoint = op;
    while (auto *next = setInsertPoint->getNextNode()) {
      if (isa<SyncBlockWaitOp>(next) || isa<SyncBlockSetOp>(next)) {
        setInsertPoint = next;
      } else {
        break;
      }
    }
    builder.setInsertionPointAfter(setInsertPoint);
    auto setOp = builder.create<SyncBlockSetOp>(loc, cubeAttr, pipeFixAttr,
                                                pipeMte3Attr, flagIdAttr);
    // set: block_id matches the SCF op, core_type = CUBE.
    if (auto opBlockIdOpt = CVPipeline::getOpBlockId(op)) {
      attachTransferTags(setOp, *opBlockIdOpt, CVPipeline::kCoreTypeCube,
                         syncTransferId);
    }
    attachAnalyzeFlagIdTag(setOp);

    // Insert the wait right before the materialize_in_destination.
    builder.setInsertionPoint(storeOp);
    auto waitOp = builder.create<SyncBlockWaitOp>(loc, vecAttr, pipeFixAttr,
                                                  pipeMte3Attr, flagIdAttr);
    // wait: block_id matches the store, core_type = VECTOR.
    if (auto storeBlockIdOpt = CVPipeline::getOpBlockId(storeOp)) {
      attachTransferTags(waitOp, *storeBlockIdOpt, CVPipeline::kCoreTypeVector,
                         syncTransferId);
    }
    attachAnalyzeFlagIdTag(waitOp);

    // Register the set/wait pair so the analyze/flag-reuse pass can reuse
    // the same flag id if ordering permits.
    flagIdReuseManager.insertRelationBetweenSetAndWait(setOp, waitOp);
    LOG_DEBUG("[cube-to-vector-direct-store] inserted set/wait, flag = "
              << flagId << "\n");
  });
}

static bool isConcretePipe(hivm::PIPE pipe) {
  return pipe != hivm::PIPE::PIPE_UNASSIGNED && pipe != hivm::PIPE::PIPE_ALL &&
         pipe != hivm::PIPE::PIPE_NUM;
}

static std::optional<hivm::TCoreType> getAnalyzeCoreType(Operation *op) {
  if (auto coreAttr =
          op->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name)) {
    auto coreType = coreAttr.getTcoretype();
    if (coreType == hivm::TCoreType::CUBE ||
        coreType == hivm::TCoreType::VECTOR) {
      return coreType;
    }
  }

  auto coreStringAttr = op->getAttrOfType<StringAttr>(CVPipeline::kCoreType);
  if (!coreStringAttr) {
    return std::nullopt;
  }
  StringRef coreType = coreStringAttr.getValue();
  if (coreType == CVPipeline::kCoreTypeCube) {
    return hivm::TCoreType::CUBE;
  }
  if (coreType == CVPipeline::kCoreTypeVector) {
    return hivm::TCoreType::VECTOR;
  }
  return std::nullopt;
}

static std::optional<hivm::AddressSpace> getMemRefAddressSpace(Type type) {
  auto memRefType = dyn_cast<MemRefType>(type);
  if (!memRefType) {
    return std::nullopt;
  }
  Attribute memorySpace = memRefType.getMemorySpace();
  if (!memorySpace) {
    return std::nullopt;
  }
  auto addressSpaceAttr = dyn_cast<hivm::AddressSpaceAttr>(memorySpace);
  if (!addressSpaceAttr) {
    return std::nullopt;
  }
  return addressSpaceAttr.getAddressSpace();
}

static std::optional<hivm::PIPE> getCopyPipeForAnalyze(hivm::CopyOp copyOp) {
  if (copyOp.hasPureBufferSemantics()) {
    return copyOp.getPipe();
  }

  auto srcAddressSpace = getMemRefAddressSpace(copyOp.getSrcOperandType());
  auto dstAddressSpace = getMemRefAddressSpace(copyOp.getDstOperandType());
  if (srcAddressSpace && dstAddressSpace) {
    if (*srcAddressSpace == hivm::AddressSpace::UB &&
        *dstAddressSpace == hivm::AddressSpace::UB) {
      return hivm::PIPE::PIPE_V;
    }
    if (*srcAddressSpace == hivm::AddressSpace::L0C &&
        *dstAddressSpace == hivm::AddressSpace::GM) {
      return hivm::PIPE::PIPE_FIX;
    }
    if (*srcAddressSpace == hivm::AddressSpace::GM &&
        *dstAddressSpace == hivm::AddressSpace::L1) {
      return hivm::PIPE::PIPE_MTE2;
    }
    if (*srcAddressSpace == hivm::AddressSpace::UB &&
        *dstAddressSpace == hivm::AddressSpace::L1) {
      return hivm::PIPE::PIPE_MTE3;
    }
  }

  // SplitDataflow inserts tensor-to-L1 copies before full bufferization. The
  // sync pair for this transfer uses MTE3 on the vector side.
  if (dstAddressSpace && *dstAddressSpace == hivm::AddressSpace::L1) {
    return hivm::PIPE::PIPE_MTE3;
  }
  return std::nullopt;
}

/// Returns the MatmulOp that is yielded by the split-if op.
/// Returns nullptr if the block has no yield terminator.
linalg::MatmulOp
InterCoreTransferAndSyncPass::findYieldMatmulInSplitIf(Value splittedIfResult) {
  auto ifOp = splittedIfResult.getDefiningOp<scf::IfOp>();
  if (!ifOp) {
    return nullptr;
  }

  auto result = dyn_cast<OpResult>(splittedIfResult);
  if (!result) {
    return nullptr;
  }
  unsigned resultIndex = result.getResultNumber();

  auto yieldOp = ifOp.thenYield();
  if (!yieldOp || resultIndex >= yieldOp.getNumOperands())
    return nullptr;
  auto definingOp = yieldOp.getOperand(resultIndex).getDefiningOp();

  return dyn_cast<linalg::MatmulOp>(definingOp);
}

void InterCoreTransferAndSyncPass::AnalyzeSplittedIf(DependencyInfo &dep) {
  auto srcOp = dep.value.getDefiningOp();
  if (!srcOp || !srcOp->hasAttr(CVPipeline::kSplittedIf)) {
    return;
  }
  Operation *consIfOp = nullptr;
  for (Operation *nextOp = srcOp->getNextNode(); nextOp;
       nextOp = nextOp->getNextNode()) {
    auto op = dyn_cast<scf::IfOp>(nextOp);
    if (op && CVPipeline::getOpBlockId(op) == dep.consumerBlockId) {
      consIfOp = op;
      break;
    }
  }
  if (!consIfOp || !consIfOp->hasAttr(CVPipeline::kSplittedIf)) {
    return;
  }
  if (srcOp->getAttrOfType<IntegerAttr>(CVPipeline::kSplittedIf) !=
      consIfOp->getAttrOfType<IntegerAttr>(CVPipeline::kSplittedIf)) {
    return;
  }
  if (auto matmulOp = findYieldMatmulInSplitIf(dep.value)) {
    dep.isSplitedIf = true;
    dep.realValue = matmulOp->getResult(0);
  }
}

// V->C Transfer Logic
LogicalResult InterCoreTransferAndSyncPass::handleVectorToCube(
    OpBuilder &builder, DependencyInfo &dep, FlagIdManager &flagManager,
    FlagIdReuseManager &flagIdReuseManager) {
  mlir::Value srcValue = dep.value;
  Location loc = dep.value.getLoc();
  Value normalizedVal = srcValue;

  // Step 1: Shape normalization (automatically insert slice)
  auto it = ndnzValueMapping.find(dep.value);
  if (it != ndnzValueMapping.end()) {
    normalizedVal = it->second;
  }

  // Get start/end operations for V/C blocks
  auto [prodStart, prodEnd] = getBlockStartEnd(dep.producerBlockId, module);
  auto [consStart, consEnd] = getBlockStartEnd(dep.consumerBlockId, module);

  Operation *consumedDataOp = nullptr;
  if (dep.consumerBlockId == dep.iniConsumerBlockId) {
    auto consumerPoint =
        analyzeConsumerReadInsertPoint(srcValue, dep.iniConsumerBlockId);
    if (consumerPoint && consumerPoint->getBlock() == consStart->getBlock()) {
      consStart = consumerPoint;
    }
  }
  if (dep.iniProducerBlockId == dep.producerBlockId) {
    auto producerPoint =
        getCopyPointBeforeStore(normalizedVal, prodEnd, dep.iniProducerBlockId);
    if (producerPoint) {
      prodEnd = producerPoint;
    }
    if (Operation *srcDefOp = srcValue.getDefiningOp();
        srcDefOp && getSubBlockId(srcDefOp)) {
      if (Operation *subBlockEnd = getSubBlockStartEnd(srcDefOp).second)
        prodEnd = subBlockEnd;
    }
  }
  LOG_DEBUG("after analyzeConsumerReadInsertPoint\n");
  Operation *transferOp = insertVectorToCubeTransfer(
      builder, srcValue, normalizedVal, prodEnd, consStart, loc, transferIndex,
      dep, is1DTensorDependency(dep.value), &consumedDataOp);
  if (!transferOp) {
    return failure();
  }
  int flagId = flagManager.acquireId();
  auto [newConsStart, newConsEnd] =
      getBlockStartEnd(dep.consumerBlockId, module);

  if (dep.consumerBlockId == dep.iniConsumerBlockId) {
    auto newconsumerPoint = getConsumerWaitPoint(transferIndex);
    if (newconsumerPoint &&
        newConsStart->getBlock() == newconsumerPoint->getBlock()) {
      newConsStart = newconsumerPoint;
    }
  }

  insertInterCoreSync(builder, transferOp, newConsStart, newConsEnd, flagId,
                      loc, transferIndex, flagIdReuseManager, consumedDataOp);

  transferIndex++;
  LOG_DEBUG("Inserted V->C transfer and sync: block "
            << dep.producerBlockId << " -> block " << dep.consumerBlockId
            << "\n");
  return success();
}

// C->V Transfer Logic
LogicalResult InterCoreTransferAndSyncPass::handleCubeToVector(
    OpBuilder &builder, DependencyInfo &dep, FlagIdManager &flagManager,
    FlagIdReuseManager &flagIdReuseManager) {
  mlir::Value srcValue = dep.value;

  Location loc = srcValue.getLoc();
  auto [prodStart, prodEnd] =
      getBlockStartEnd(dep.producerBlockId, module); // C Block
  auto [consStart, consEnd] =
      getBlockStartEnd(dep.consumerBlockId, module); // V Block
  LOG_DEBUG("[newProdStart]" << *prodStart << "\n");
  LOG_DEBUG("[newProdEnd]" << *prodEnd << "\n");
  LOG_DEBUG("[newConsStart]" << *consStart << "\n");
  LOG_DEBUG("[newConsEnd]" << *consEnd << "\n");

  AnalyzeSplittedIf(dep);

  if (srcValue.getDefiningOp() &&
      !isa<scf::ForOp, scf::WhileOp, scf::IfOp>(srcValue.getDefiningOp())) {
    auto producerPoint =
        getFixpipePointAfterProducer(srcValue, dep.iniProducerBlockId);
    if (producerPoint) {
      prodEnd = producerPoint;
    }
  }

  // uses of srcValue. Only applied when the source op belongs to a
  // sub-block.
  Operation *consumerPoint = nullptr;
  if (dep.consumerBlockId == dep.iniConsumerBlockId) {
    consumerPoint =
        analyzeConsumerReadInsertPoint(srcValue, dep.iniConsumerBlockId);
    if (consumerPoint && getSubBlockId(consumerPoint)) {
      consStart = getSubBlockStartEnd(consumerPoint).first;
    }
  }

  if (dep.isSplitedIf) {
    prodEnd = dep.realValue.getDefiningOp();
    auto consumerPoint =
        analyzeConsumerReadInsertPoint(srcValue, dep.iniConsumerBlockId);
    if (consumerPoint) {
      consStart = consumerPoint;
    }
  }

  Operation *consumedDataOp = nullptr;
  Operation *transferOp =
      insertCubeToVectorTransfer(builder, srcValue, prodEnd, consStart, loc,
                                 transferIndex, dep, &consumedDataOp);

  auto [newProdStart, newProdEnd] =
      getBlockStartEnd(dep.producerBlockId, module); // C Block
  auto [newConsStart, newConsEnd] =
      getBlockStartEnd(dep.consumerBlockId, module); // V Block
  if (Operation *subBlockEnd = getSubBlockStartEnd(consumerPoint).second) {
    newConsEnd = subBlockEnd;
  }
  int flagId = flagManager.acquireId();

  bool isStoreDirectly =
      isStoreDirectlyInUserChain(consumedDataOp->getResult(0));

  if (dep.consumerBlockId == dep.iniConsumerBlockId) {
    auto newconsumerPoint = getConsumerWaitPoint(transferIndex);
    if (newconsumerPoint && getSubBlockId(consumerPoint)) {
      // insert the wait for read at the sub-block start
      newConsStart = getSubBlockStartEnd(consumerPoint).first;
    }
  }
  if (dep.isSplitedIf) {
    auto newconsumerPoint = getConsumerWaitPoint(transferIndex);
    auto yieldBeforeOp = getOpBeforeYield(newconsumerPoint);
    if (newconsumerPoint && yieldBeforeOp) {
      newConsStart = newconsumerPoint;
      newConsEnd = yieldBeforeOp;
    }
  }
  insertInterCoreSync(builder, transferOp, newConsStart, newConsEnd, flagId,
                      loc, transferIndex, flagIdReuseManager, consumedDataOp,
                      isStoreDirectly);

  transferIndex++;
  LOG_DEBUG("Inserted C->V transfer and sync: block "
            << dep.producerBlockId << " -> block " << dep.consumerBlockId
            << "\n");
  return success();
}

// C->C Shared L1 buffer allocation.
// Intra-block C->C: allocates right after the producing matmul (before the
// fixpipe). Cross-block C->C: allocates before the main loop if one encloses
// both producer and consumer, otherwise after the producer block end.
Operation *InterCoreTransferAndSyncPass::createC2CSharedL1Buffer(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> shape, Type elemType,
    int prodBlockId, Operation *prodEnd, Operation *consStart,
    bool isIntraC2C) {
  auto addressSpaceAttr =
      builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::L1);
  auto allocType = MemRefType::get(shape, elemType, nullptr, addressSpaceAttr);

  Operation *allocOp = nullptr;
  Operation *mainLoopOp = findMainLoopforTransfer(prodEnd, consStart);
  if (mainLoopOp && !isIntraC2C) {
    builder.setInsertionPoint(mainLoopOp);
    allocOp = builder.create<memref::AllocOp>(loc, allocType);
    int loopBlockId = CVPipeline::getOpBlockId(mainLoopOp).value_or(-1);
    attachCommonTags(allocOp, loopBlockId, CVPipeline::kCoreTypeCube);
    builder.setInsertionPointAfter(prodEnd);
  } else {
    builder.setInsertionPointAfter(prodEnd);
    allocOp = builder.create<memref::AllocOp>(loc, allocType);
    attachCommonTags(allocOp, prodBlockId, CVPipeline::kCoreTypeCube);
  }
  return allocOp;
}

// C->C Transfer Logic
LogicalResult
InterCoreTransferAndSyncPass::handleCubeToCube(OpBuilder &builder,
                                               DependencyInfo &dep) {
  mlir::Value transferValue = dep.value;
  Location loc = transferValue.getLoc();

  // Check if this is a matmul -> trunc -> matmul pattern.
  // If so, use the matmul result (pre-trunc) as the fixpipe source and
  // fold the type conversion into the fixpipe as pre_quant.
  Operation *truncOp = transferValue.getDefiningOp();
  mlir::Value fixpipeSrcValue = transferValue;
  std::optional<FixpipePreQuantMode> quantMode =
      CVPipeline::getFixpipePreQuantMode(truncOp);
  if (quantMode) {
    fixpipeSrcValue = transferValue.getDefiningOp()->getOperand(0);
  }

  auto [prodStart, prodEnd] =
      getBlockStartEnd(dep.producerBlockId, module); // C Block
  auto [consStart, consEnd] =
      getBlockStartEnd(dep.consumerBlockId, module); // C Block

  // Adjust consStart to the first operation that actually reads transferValue
  // within the consumer block, matching the logic in handleVectorToCube.
  if (dep.consumerBlockId == dep.iniConsumerBlockId) {
    auto consumerPoint =
        analyzeConsumerReadInsertPoint(transferValue, dep.iniConsumerBlockId);
    if (consumerPoint) {
      consStart = consumerPoint;
    }
  }

  auto transferTensorType = cast<RankedTensorType>(transferValue.getType());
  int64_t M = transferTensorType.getDimSize(0);
  int64_t N = transferTensorType.getDimSize(1);
  Type elemType = transferTensorType.getElementType();
  auto shape = std::vector<int64_t>{M, N};

  int prodBlockId =
      CVPipeline::getOpBlockId(fixpipeSrcValue.getDefiningOp()).value_or(-1);
  int consBlockId = CVPipeline::getOpBlockId(consStart).value_or(-1);

  bool isIntraC2C = (dep.iniProducerBlockId == dep.iniConsumerBlockId);

  if (!isa<scf::ForOp, scf::WhileOp, scf::IfOp>(
          transferValue.getDefiningOp())) {
    auto producerPoint =
        getFixpipePointAfterProducer(transferValue, dep.iniProducerBlockId);
    if (producerPoint) {
      prodEnd = producerPoint;
    }
  }

  // Allocate a single shared L1 buffer for producer and consumer.
  Operation *allocOp =
      createC2CSharedL1Buffer(builder, loc, shape, elemType, prodBlockId,
                              prodEnd, consStart, isIntraC2C);

  // Producer side: insert fixpipe to write matmul L0C output to L1 buffer
  auto dmaModeAttr =
      FixpipeDMAModeAttr::get(builder.getContext(), FixpipeDMAMode::NZ2NZ);
  bool channelSplit = isChannelSplitNeeded(transferTensorType);

  // Build quantModeAttr when a trunc is being folded in as pre_quant.
  FixpipePreQuantModeAttr quantModeAttr = nullptr;
  if (quantMode.has_value()) {
    quantModeAttr =
        FixpipePreQuantModeAttr::get(builder.getContext(), quantMode.value());
  }

  auto fixpipeOp = builder.create<hivm::FixpipeOp>(
      loc, mlir::TypeRange{}, fixpipeSrcValue, allocOp->getResult(0),
      mlir::ValueRange{}, dmaModeAttr, nullptr, nullptr, quantModeAttr, nullptr,
      builder.getBoolAttr(channelSplit), nullptr, nullptr, mlir::ArrayAttr{},
      nullptr);
  attachCommonTags(fixpipeOp, prodBlockId, CVPipeline::kCoreTypeCube);
  // Tag C2C fixpipe as kIntraDeps producer when it is a cross-block C2C.
  if (!isIntraC2C) {
    fixpipeOp->setAttr(
        CVPipeline::kIntraDeps,
        builder.getI32ArrayAttr(
            {intraDepsGroupId, CVPipeline::crossCoreProducerId}));
  }
  LOG_DEBUG("[fixpipeOp C->C]: " << *fixpipeOp << "\n");

  // Consumer side: read L1 buffer via MemorySpaceCast + ToTensor
  builder.setInsertionPoint(consStart);
  auto plainMemrefType = MemRefType::get(shape, elemType);
  auto memspaceCastOp = builder.create<memref::MemorySpaceCastOp>(
      loc, plainMemrefType, allocOp->getResult(0));
  auto targetTensorType = RankedTensorType::get(shape, elemType);
  auto toTensorOp = builder.create<bufferization::ToTensorOp>(
      loc, targetTensorType, memspaceCastOp.getResult(), true, true);
  attachCommonTags(memspaceCastOp, consBlockId, CVPipeline::kCoreTypeCube);
  // Tag C2C memspaceCastOp as kIntraDeps consumer when it is a cross-block C2C.
  if (!isIntraC2C) {
    memspaceCastOp->setAttr(
        CVPipeline::kIntraDeps,
        builder.getI32ArrayAttr(
            {intraDepsGroupId, CVPipeline::crossCoreConsumerId}));
  }
  attachCommonTags(toTensorOp, consBlockId, CVPipeline::kCoreTypeCube);

  // Replace uses of transferValue within the consumer block.
  // For matmul ops, only replace input (A/B) operands so that the init (outs)
  // still directly uses the producer matmul result when the same value is
  // used as both input and init.
  llvm::SmallVector<Operation *> users(transferValue.getUsers().begin(),
                                       transferValue.getUsers().end());
  for (Operation *user : users) {
    auto userBlockIdOpt = CVPipeline::getOpBlockId(user);
    if (userBlockIdOpt && *userBlockIdOpt == dep.iniConsumerBlockId) {
      if (user == fixpipeOp) {
        continue;
      }
      if (auto matmulUser = dyn_cast<linalg::MatmulOp>(user)) {
        for (int64_t i = 0; i < matmulUser.getNumDpsInputs(); ++i) {
          if (matmulUser->getOpOperand(i).get() == transferValue) {
            matmulUser->setOperand(i, toTensorOp.getResult());
          }
        }
      } else {
        LOG_DEBUG(
            "warning: c2c transferValue is used by non-matmul op: " << *user);
        user->replaceUsesOfWith(transferValue, toTensorOp.getResult());
      }
    }
  }

  // Erase the trunc op after all uses are replaced (dead code elimination).
  if (truncOp && CVPipeline::getFixpipePreQuantMode(truncOp).has_value()) {
    truncOp->erase();
  }

  intraDepsGroupId++;
  LOG_DEBUG("Inserted C->C fixpipe transfer: block "
            << dep.producerBlockId << " -> block " << dep.consumerBlockId
            << "\n");
  return success();
}

// Memory Dependency
LogicalResult InterCoreTransferAndSyncPass::handleMemoryDependency(
    OpBuilder &builder, DependencyInfo &dep, size_t depIndex,
    llvm::SmallVector<DependencyInfo> memDependencies,
    FlagIdManager &flagManager, FlagIdReuseManager &flagIdReuseManager) {
  LOG_DEBUG("Handling memory dependency...\n");

  // Get producer and consumer block start/end operations
  auto [prodStart, prodEnd] = getBlockStartEnd(dep.producerBlockId, module);
  auto [consStart, consEnd] = getBlockStartEnd(dep.consumerBlockId, module);

  if (!prodStart || !prodEnd || !consStart || !consEnd) {
    LOG_DEBUG("[ERROR] Failed to get block start/end operations.\n");
    return failure();
  }

  if (isOuterLayerDependency(depIndex, prodEnd, consStart, memDependencies)) {
    LOG_DEBUG("[MEMDEP] Skipping outer layer dependency: block "
              << dep.producerBlockId << " -> block " << dep.consumerBlockId
              << "\n");
    return success();
  }

  attachMemCrossDeps(dep.predOp, transferIndex, CVPipeline::crossCoreProducerId,
                     builder);
  attachMemCrossDeps(dep.nextOp, transferIndex, CVPipeline::crossCoreConsumerId,
                     builder);
  attachCrossCoreDeps(dep.predOp, transferIndex,
                      CVPipeline::crossCoreProducerId, builder);
  attachCrossCoreDeps(dep.nextOp, transferIndex,
                      CVPipeline::crossCoreConsumerId, builder);
  // Get flag ID
  int flagId = flagManager.acquireId();

  // Determine sync direction: CUBE->VECTOR or VECTOR->CUBE
  bool isCubeToVector = (dep.type == DependencyType::CubeToVector);

  // Get location info
  Location loc = prodEnd->getLoc();
  if (dep.iniProducerBlockId == dep.producerBlockId &&
      dep.iniConsumerBlockId == dep.consumerBlockId &&
      hasMemDepSyncWhitelistKernel(module)) {
    prodEnd = dep.predOp;
    prodStart = dep.predOp;
  }

  insertMemDepSync(builder, prodStart, prodEnd, consStart, consEnd, flagId, loc,
                   isCubeToVector, flagIdReuseManager);

  transferIndex++;

  LOG_DEBUG("Inserted PIPE_MTE2 sync: block "
            << dep.producerBlockId << " -> block " << dep.consumerBlockId
            << ", flagId = " << flagId << "\n");

  return success();
}

llvm::SmallVector<mlir::Operation *>
InterCoreTransferAndSyncPass::insertAnalyzeFlagRelations(
    mlir::ModuleOp module, FlagIdReuseManager &flagIdReuseManager) {
  using OpVector = llvm::SmallVector<mlir::Operation *>;

  // E1 (per-pipe FIFO) is isolated per MLIR block: ops on one (core, pipe)
  llvm::DenseMap<Block *,
                 llvm::SmallDenseMap<hivm::TCoreType,
                                     llvm::SmallDenseMap<hivm::PIPE, OpVector>>>
      sequenceOpMap;
  llvm::DenseSet<mlir::Operation *> relationOpSet;
  llvm::SmallVector<mlir::Operation *> relationOps;
  llvm::SmallVector<mlir::Operation *> analyzeFlagIdOps;

  auto insertRelation = [&](Operation *before, Operation *after) {
    if (!before || !after || before == after) {
      return;
    }
    flagIdReuseManager.insertRelationBetweenSetAndWait(before, after);
  };

  auto noteRelationOp = [&](Operation *op) {
    if (!relationOpSet.insert(op).second) {
      return;
    }
    relationOps.push_back(op);
  };

  auto notePipeOp = [&](Operation *op, hivm::TCoreType coreType,
                        hivm::PIPE pipe) {
    if (!isConcretePipe(pipe)) {
      return;
    }
    if (Block *block = op->getBlock()) {
      sequenceOpMap[block][coreType][pipe].push_back(op);
    }
    noteRelationOp(op);
  };

  module.walk([&](mlir::Operation *op) {
    if (auto setOp = llvm::dyn_cast<hivm::SyncBlockSetOp>(op)) {
      auto coreAttr = setOp.getTcoreType();
      auto pipeAttr = setOp.getTpipeAttr();
      if (coreAttr && pipeAttr) {
        notePipeOp(op, coreAttr.getTcoretype(), pipeAttr.getPipe());
      }
      if (op->hasAttr(CVPipeline::kAnalyzeFlagId)) {
        analyzeFlagIdOps.push_back(op);
      }
      return;
    }

    if (auto waitOp = llvm::dyn_cast<hivm::SyncBlockWaitOp>(op)) {
      auto coreAttr = waitOp.getTcoreType();
      auto pipeAttr = waitOp.getPipeAttr();
      if (coreAttr && pipeAttr) {
        notePipeOp(op, coreAttr.getTcoretype(), pipeAttr.getPipe());
      }
      if (op->hasAttr(CVPipeline::kAnalyzeFlagId)) {
        analyzeFlagIdOps.push_back(op);
      }
      return;
    }

    auto coreType = getAnalyzeCoreType(op);
    if (!coreType) {
      return;
    }

    noteRelationOp(op);
    if (auto pipeOp = llvm::dyn_cast<hivm::OpPipeInterface>(op)) {
      if (auto copyOp = llvm::dyn_cast<hivm::CopyOp>(op)) {
        if (auto pipe = getCopyPipeForAnalyze(copyOp)) {
          notePipeOp(op, *coreType, *pipe);
        }
        return;
      }
      if (pipeOp.isMacroOp()) {
        notePipeOp(op, *coreType, pipeOp.getInPipe());
        notePipeOp(op, *coreType, pipeOp.getOutPipe());
        return;
      }
      notePipeOp(op, *coreType, pipeOp.getPipe());
    }
  });

  for (Operation *op : relationOps) {
    for (Value operand : op->getOperands()) {
      Operation *definingOp = operand.getDefiningOp();
      if (!definingOp) {
        if (auto blockArgument = dyn_cast<BlockArgument>(operand)) {
          definingOp = blockArgument.getOwner()->getParentOp();
        }
      }
      if (definingOp && relationOpSet.contains(definingOp)) {
        insertRelation(definingOp, op);
      }
    }
  }

  // E1: per-block, per-(core, pipe) FIFO chain.
  for (auto &blockEntry : sequenceOpMap) {
    for (auto &coreEntry : blockEntry.second) {
      for (auto &pipeEntry : coreEntry.second) {
        auto &ops = pipeEntry.second;
        for (size_t i = 0; i + 1 < ops.size(); ++i) {
          insertRelation(ops[i], ops[i + 1]);
        }
      }
    }
  }
  return analyzeFlagIdOps;
}

void InterCoreTransferAndSyncPass::remapInterCoreTransferFlagIds(
    llvm::DenseMap<int, int> &remapResult) {
  module.walk([&](mlir::Operation *op) {
    if (!llvm::isa<hivm::SyncBlockSetOp>(op) &&
        !llvm::isa<hivm::SyncBlockWaitOp>(op)) {
      return;
    }
    bool trackedForReuse = op->hasAttr(CVPipeline::kAnalyzeFlagId);
    op->removeAttr(CVPipeline::kAnalyzeFlagId);
    if (!trackedForReuse || remapResult.empty()) {
      return;
    }
    if (auto intAttr = op->getAttrOfType<mlir::IntegerAttr>("static_flag_id")) {
      int flagId = static_cast<int>(intAttr.getInt());
      auto it = remapResult.find(flagId);
      if (it == remapResult.end()) {
        return;
      }
      auto newFlagAttr = mlir::IntegerAttr::get(intAttr.getType(), it->second);
      op->setAttr("static_flag_id", newFlagAttr);
    }
  });
}

void InterCoreTransferAndSyncPass::sortDependencies(
    llvm::SmallVector<DependencyInfo> &dependencies, mlir::ModuleOp module) {
  if (dependencies.size() <= 1) {
    return;
  }

  // Step 1: Walk the entire module and assign a monotonically increasing order
  //         to each operation, representing its position in the IR.
  llvm::DenseMap<mlir::Operation *, unsigned> opOrder;
  unsigned order = 0;
  module.walk<WalkOrder::PreOrder>(
      [&](mlir::Operation *op) { opOrder[op] = order++; });

  // Step 2: Helper lambda — get the earliest user op of dep.value within the
  //         consumer compute block.
  auto getFirstConsumerOp =
      [&](const DependencyInfo &dep) -> mlir::Operation * {
    mlir::Operation *firstConsumer = nullptr;
    unsigned firstOrder = std::numeric_limits<unsigned>::max();
    for (auto *user : dep.value.getUsers()) {
      auto userBlockIdOpt = CVPipeline::getOpBlockId(user);
      if (userBlockIdOpt && *userBlockIdOpt == dep.consumerBlockId) {
        auto it = opOrder.find(user);
        if (it != opOrder.end() && it->second < firstOrder) {
          firstOrder = it->second;
          firstConsumer = user;
        }
      }
    }
    return firstConsumer;
  };

  // Step 3: Sort
  std::sort(dependencies.begin(), dependencies.end(),
            [&](const DependencyInfo &a, const DependencyInfo &b) {
              // the dependency whose consumer op appears earlier comes first.
              auto *aConsOp = getFirstConsumerOp(a);
              auto *bConsOp = getFirstConsumerOp(b);
              if (aConsOp && bConsOp) {
                unsigned aConsOpOrder = opOrder.lookup(aConsOp);
                unsigned bConsOpOrder = opOrder.lookup(bConsOp);
                if (aConsOpOrder != bConsOpOrder) {
                  return aConsOpOrder < bConsOpOrder;
                }
              }
              return false;
            });
}

void InterCoreTransferAndSyncPass::analyzeLoopInclusion() {
  loopInclusions.clear();
  int maxLoopId = -1;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    auto loopOp = dyn_cast<LoopLikeOpInterface>(op);
    if (!loopOp) {
      return;
    }

    int loopId = 0;
    Operation *parentLoop =
        loopOp.getOperation()->getParentOfType<LoopLikeOpInterface>();
    while (parentLoop) {
      ++loopId;
      parentLoop = parentLoop->getParentOfType<LoopLikeOpInterface>();
    }
    loopOp->setAttr(
        CVPipeline::kLoopId,
        IntegerAttr::get(
            IntegerType::get(loopOp->getContext(), kIntegerBitWidth), loopId));
    loopInclusions[loopId].push_back(loopOp);
    maxLoopId = std::max(maxLoopId, loopId);
  });
  for (int loopId = 0; loopId <= maxLoopId; ++loopId) {
    if (loopInclusions[loopId].size() != 1) {
      break;
    }
    bool allSingle = true;
    for (int outerId = loopId - 1; outerId >= 0; --outerId) {
      if (loopInclusions[outerId].size() != 1) {
        allSingle = false;
        break;
      }
    }
    if (allSingle) {
      singleLoopSet.insert(loopId);
    }
  }
}

void InterCoreTransferAndSyncPass::moveStartEndSync(OpBuilder &builder) {
  llvm::SmallVector<Operation *> startSyncOps;
  llvm::SmallVector<Operation *> endSyncOps;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!op->hasAttr(CVPipeline::kLoopId) ||
        !(isa<hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp>(op))) {
      return;
    }
    auto loopId = op->getAttrOfType<IntegerAttr>(CVPipeline::kLoopId).getInt();
    auto it = singleLoopSet.find(loopId);
    if (it != singleLoopSet.end()) {
      if (isa<hivm::SyncBlockSetOp>(op)) {
        startSyncOps.push_back(op);
      } else if (isa<hivm::SyncBlockWaitOp>(op)) {
        endSyncOps.push_back(op);
      }
    }
  });

  if (loopInclusions.count(0) == 0 || loopInclusions[0].empty()) {
    return;
  }
  Operation *outerLoop = loopInclusions[0][0];
  if (!outerLoop) {
    return;
  }
  // move startSyncOps and endSyncOps to the outer loop
  builder.setInsertionPoint(outerLoop);
  for (auto it = startSyncOps.begin(); it != startSyncOps.end(); ++it) {
    builder.clone(**it);
  }
  builder.setInsertionPointAfter(outerLoop);
  for (auto it = endSyncOps.rbegin(); it != endSyncOps.rend(); ++it) {
    builder.clone(**it);
  }
  for (auto op : startSyncOps) {
    op->erase();
  }
  for (auto op : endSyncOps) {
    op->erase();
  }
}

// Main Processing
LogicalResult InterCoreTransferAndSyncPass::processDependencies(
    FlagIdManager &flagManager, FlagIdReuseManager &flagIdReuseManager) {
  LOG_DEBUG("Starting InterCoreTransferAndSyncPass processDependencies...\n");
  OpBuilder builder(module.getContext());

  auto &info = getAnalysis<DataDependencyInfo>();
  if (!info.isValid()) {
    LOG_DEBUG("Error: Data dependency analysis failed.\n");
    return failure();
  }

  // Analyze loop nesting before inserting inter-core transfer/sync.
  if (hasStartEndSyncWhitelistKernel(module)) {
    analyzeLoopInclusion();
  }

  llvm::SmallVector<DependencyInfo> &V2CDependencies =
      info.getV2CDependencies();
  sortDependencies(V2CDependencies, module);
  LOG_DEBUG("[DEBUG] V2CDependencies size: " << V2CDependencies.size() << "\n");
  for (size_t i = 0; i < V2CDependencies.size(); ++i) {
    auto &dep = V2CDependencies[i];
    LOG_DEBUG("[V2C-" << i << "] producerBlockId = " << dep.producerBlockId
                      << ", consumerBlockId = " << dep.consumerBlockId
                      << ", iniProducerBlockId = " << dep.iniProducerBlockId
                      << ", iniConsumerBlockId = " << dep.iniConsumerBlockId
                      << ", value = " << dep.value << "\n");
  }
  LOG_DEBUG("Step 1: Handle V->C dependencies\n");
  // Step 1: Handle V->C dependencies
  for (auto &dep : V2CDependencies) {
    if (!isScalarDependency(dep.value) && !is1DTensorDependency(dep.value)) {
      Location loc = dep.value.getLoc();
      Nd2NzNormalize(builder, dep, loc);
    }
  }

  for (auto &dep : V2CDependencies) {
    LOG_DEBUG("[V->C] producerBlockId = " << dep.producerBlockId
                                          << ", consumerBlockId = "
                                          << dep.consumerBlockId << "\n");
    if (failed(handleVectorToCube(builder, dep, flagManager,
                                  flagIdReuseManager))) {
      LOG_DEBUG("[ERROR] V->C failed! producerBlockId = "
                << dep.producerBlockId
                << ", consumerBlockId = " << dep.consumerBlockId << "\n");
      return failure();
    }
  }
  LOG_DEBUG("Completed V->C transfers and syncs.\n");

  llvm::SmallVector<DependencyInfo> &C2VDependencies =
      info.getC2VDependencies();
  sortDependencies(C2VDependencies, module);
  LOG_DEBUG("[DEBUG] C2VDependencies size: " << C2VDependencies.size() << "\n");
  // Step 2: Handle C->V dependencies
  for (auto &dep : C2VDependencies) {
    LOG_DEBUG("[C->V] producerBlockId = " << dep.producerBlockId
                                          << ", consumerBlockId = "
                                          << dep.consumerBlockId << "\n");
    if (failed(handleCubeToVector(builder, dep, flagManager,
                                  flagIdReuseManager))) {
      LOG_DEBUG("[ERROR] C->V failed!  producerBlockId = "
                << dep.producerBlockId
                << ", consumerBlockId = " << dep.consumerBlockId << "\n");
      return failure();
    }
  }
  LOG_DEBUG("Completed C->V transfers and syncs.\n");

  // Step 3.1: Handle inter-block C->C dependencies (fixpipe L0C to L1)
  llvm::SmallVector<DependencyInfo> &C2CDependencies =
      info.getC2CDependencies();
  sortDependencies(C2CDependencies, module);
  LOG_DEBUG("[DEBUG] InterC2CDependencies size: " << C2CDependencies.size()
                                                  << "\n");
  for (auto &dep : C2CDependencies) {
    LOG_DEBUG("[C->C] producerBlockId = " << dep.producerBlockId
                                          << ", consumerBlockId = "
                                          << dep.consumerBlockId << "\n");
    // Only handle C->C transfer when both the defining op and the consuming
    // users of dep.value are matmul ops.
    if (!isValidC2CMatmulDependency(dep.value, dep.consumerBlockId)) {
      continue;
    }
    if (failed(handleCubeToCube(builder, dep))) {
      LOG_DEBUG("[ERROR] Inter C->C failed! producerBlockId = "
                << dep.producerBlockId
                << ", consumerBlockId = " << dep.consumerBlockId << "\n");
      return failure();
    }
  }
  // Step 3: Handle intra-block C->C dependencies.
  llvm::SmallVector<DependencyInfo> &intraC2CDependencies =
      info.getIntraC2CDependencies();
  sortDependencies(intraC2CDependencies, module);
  LOG_DEBUG("[DEBUG] IntraC2CDependencies size: " << intraC2CDependencies.size()
                                                  << "\n");
  for (auto &dep : intraC2CDependencies) {
    LOG_DEBUG("[IntraC2C] producer and consumer' BlockId = "
              << dep.producerBlockId << "\n");
    if (!isValidC2CMatmulDependency(dep.value, dep.consumerBlockId)) {
      continue;
    }
    // dep.operand must be an input (A/B) operand of the consumer matmul,
    // not an init (outs) operand.
    if (!dep.operand) {
      continue;
    }
    auto consumerMatmul = dyn_cast<linalg::MatmulOp>(dep.operand->getOwner());
    if (!consumerMatmul ||
        dep.operand->getOperandNumber() >=
            static_cast<unsigned>(consumerMatmul.getNumDpsInputs())) {
      continue;
    }
    if (failed(handleCubeToCube(builder, dep))) {
      LOG_DEBUG("[ERROR] IntraC2C failed! producer and consumer' BlockId = "
                << dep.consumerBlockId << "\n");
      return failure();
    }
  }
  LOG_DEBUG("Completed C->C transfers and syncs.\n");

  llvm::SmallVector<DependencyInfo> &memDependencies =
      info.getMemoryDependencies();
  LOG_DEBUG("[DEBUG] MemoryDependencies size: " << memDependencies.size()
                                                << "\n");

  for (size_t i = 0; i < memDependencies.size(); ++i) {
    auto &dep = memDependencies[i];
    LOG_DEBUG("[MEMDEP] value = "
              << dep.value << " producerBlockId = " << dep.producerBlockId
              << ", consumerBlockId = " << dep.consumerBlockId << "\n");
    if (failed(handleMemoryDependency(builder, dep, i, memDependencies,
                                      flagManager, flagIdReuseManager))) {
      LOG_DEBUG("[ERROR] Memdep failed! producerBlockId = "
                << dep.producerBlockId
                << ", consumerBlockId = " << dep.consumerBlockId << "\n");
      return failure();
    }
  }
  LOG_DEBUG("Completed memory syncs.\n");
  LOG_DEBUG("=====================================================\n");

  if (!flagManager.checkCurrentId()) {
    llvm::SmallVector<mlir::Operation *> analyzeFlagIdOps =
        insertAnalyzeFlagRelations(module, flagIdReuseManager);
    DenseMap<int, int> remapResult =
        flagIdReuseManager.reuseInterCoreTransferFlagIds(analyzeFlagIdOps);
    remapInterCoreTransferFlagIds(remapResult);
  }

  // Remove VECTOR pseudo-ops before inserting the direct-store sync.
  removeVectorPseudoOps();

  // Synchronize CUBE's fixpipe with VECTOR's direct MTE3 store. No VECTOR
  // tensor computation occurs between the transfer and the store.
  processCubeToVectorDirectStoreSync(builder, flagManager, flagIdReuseManager);

  // move start/end sync ops
  if (hasStartEndSyncWhitelistKernel(module)) {
    moveStartEndSync(builder);
  }

  LOG_DEBUG("InterCoreTransferAndSyncPass success!\n");

  return success();
}

// Declare dependent dialects
void InterCoreTransferAndSyncPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<func::FuncDialect, arith::ArithDialect, linalg::LinalgDialect,
                  scf::SCFDialect, tensor::TensorDialect,
                  bufferization::BufferizationDialect, memref::MemRefDialect,
                  hivm::HIVMDialect, annotation::AnnotationDialect>();
}

// Pass Entry Point
void InterCoreTransferAndSyncPass::runOnOperation() {
  LOG_DEBUG("\n--- enter InterCoreTransferAndSyncPass --->\n");
  module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  // Phase 1: Initialize FlagIdManager as local variable
  FlagIdManager flagManager(module);
  FlagIdReuseManager flagIdReuseManager;

  // Phase 2: Execute transfer and sync insertion
  if (failed(processDependencies(flagManager, flagIdReuseManager))) {
    LOG_DEBUG("Error: Inter-core transfer and sync failed.\n");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  LOG_DEBUG("Module after InterCoreTransferAndSyncPass:\n" << module << "\n");

  LOG_DEBUG("--- exit InterCoreTransferAndSyncPass --->\n");
}

// Create the pass
namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createInterCoreTransferAndSyncPass() {
  return std::make_unique<InterCoreTransferAndSyncPass>();
}

void registerInterCoreTransferAndSyncPasses() {
  registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createInterCoreTransferAndSyncPass();
  });
}

} // namespace triton
} // namespace mlir
