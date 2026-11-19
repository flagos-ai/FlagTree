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

#include "ascend/include/DynamicCVPipeline/SplitDataflow/DataDependencyAnalysis.h"
#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/Utils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

static constexpr const char *DEBUG_TYPE = "data-dependency-analysis";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir::triton;
using namespace mlir::CVPipeline;

// Attribute name constants
static constexpr const char *ssbufferCoreTypeCubeAttr = "CUBE";
static constexpr const char *ssbufferCoreTypeVectorAttr = "VECTOR";
static constexpr int ND_SHAPE_LENGTH = 2;
static constexpr int SHAPE_1D_LENGTH = 1;
static constexpr int constantIntType = 32;

// Helper: ssbuffer.core_type
llvm::StringRef getSsbufferCoreType(Operation *op) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>(CVPipeline::kCoreType)) {
    return attr.getValue();
  }
  return "";
}

// Helper: Get CoreType from op and index
llvm::StringRef getCoreTypeWithIndex(Operation *op, int index) {
  llvm::StringRef typeStr = getSsbufferCoreType(op);
  if (typeStr.contains(", ")) {
    llvm::SmallVector<llvm::StringRef> types;
    typeStr.split(types, ", ", -1, false);
    if (index < types.size()) {
      return types[index].trim();
    }
    LOG_DEBUG("Warning: Core type string has multiple types but value is not "
              "an OpResult or index out of range.\n");
    return "";
  }

  return typeStr;
}

void DataDependencyAnalysisPass::updateCoreTypeAtIndex(
    Operation *op, int index, llvm::StringRef newCoreType) {
  auto coreTypeAttr =
      op->getAttrOfType<mlir::StringAttr>(CVPipeline::kCoreType);
  if (!coreTypeAttr) {
    LOG_DEBUG("[warning]: failed to rewrite coretype\n");
    return;
  }

  llvm::StringRef typeStr = coreTypeAttr.getValue();
  if (typeStr.contains(", ")) {
    llvm::SmallVector<llvm::StringRef> types;
    typeStr.split(types, ", ", -1, false);
    if (index < static_cast<int>(types.size())) {
      types[index] = newCoreType;
    } else {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      LOG_DEBUG("[info]: invalid index for updateCoreTypeAtIndex.\n");
      return;
    }
    std::string newTypeStr;
    for (size_t i = 0; i < types.size(); ++i) {
      if (i > 0)
        newTypeStr += ", ";
      newTypeStr += types[i].str();
    }
    op->setAttr(CVPipeline::kCoreType,
                StringAttr::get(op->getContext(), newTypeStr));
  } else {
    if (index != 0) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      LOG_DEBUG("[info]: invalid index for updateCoreTypeAtIndex.\n");
      return;
    }
    op->setAttr(CVPipeline::kCoreType,
                StringAttr::get(op->getContext(), newCoreType));
  }
}

// Helper: Check if operation is control flow
bool DataDependencyAnalysisPass::isControlFlowOp(mlir::Operation *op) {
  if (!op)
    return false;
  return isa<scf::ForOp, scf::WhileOp>(op) || isa<scf::IfOp>(op) ||
         isa<scf::YieldOp, scf::ConditionOp>(op);
}

bool DataDependencyAnalysisPass::isCubeAndVectorOp(mlir::Operation *op) {
  if (isa<tensor::EmptyOp, linalg::FillOp>(op)) {
    return true;
  }
  return false;
}

bool DataDependencyAnalysisPass::isValidShapeForDependency(mlir::Value value) {
  auto tensorTy = dyn_cast<TensorType>(value.getType());
  if (!tensorTy) {
    return false;
  }

  if (tensorTy.getRank() != ND_SHAPE_LENGTH) {
    return false;
  }
  return true;
}

bool DataDependencyAnalysisPass::isValidScalarDependency(mlir::Value value) {
  if (isa<mlir::IntegerType, mlir::FloatType>(value.getType())) {
    auto defOp = value.getDefiningOp();
    if (defOp && isa<tensor::ExtractOp>(defOp)) {
      return true;
    }
  }
  return false;
}

bool DataDependencyAnalysisPass::isValid1DValueForDependency(
    mlir::Value value) {
  auto tensorTy = dyn_cast<TensorType>(value.getType());
  if (tensorTy && tensorTy.getRank() == SHAPE_1D_LENGTH) {
    return true;
  }
  return false;
}

// Check if a value is only used by transpose ops whose users are all vector
// ops. Returns whether the condition holds and, if so, the yield op that uses
// the transposed value (nullptr if there is none).
std::pair<bool, std::optional<mlir::Operation *>>
DataDependencyAnalysisPass::isAllTransposedInVector(mlir::Value value) {
  if (!isa<linalg::MatmulOp>(value.getDefiningOp())) {
    return {false, std::nullopt};
  }
  if (!llvm::hasSingleElement(value.getUsers())) {
    return {false, std::nullopt};
  }
  auto *userOp = *value.getUsers().begin();
  if (!isa<linalg::TransposeOp>(userOp))
    return {false, std::nullopt};
  for (mlir::Operation *transposeOpUser : userOp->getUsers()) {
    if (getSsbufferCoreType(transposeOpUser) != ssbufferCoreTypeVectorAttr)
      return {false, std::nullopt};
  }

  for (mlir::Operation *transposeOpUser : userOp->getUsers()) {
    if (isa<scf::YieldOp>(transposeOpUser)) {
      return {true, transposeOpUser};
    }
  }
  return {true, std::nullopt};
}

// Helper: Check if value is a valid tensor for dependency analysis
// Returns true if value is TensorType and not defined by EmptyOp/FillOp
bool DataDependencyAnalysisPass::isValidValueForDependency(mlir::Value value) {
  if (isValidScalarDependency(value)) {
    return true;
  }
  if (isValid1DValueForDependency(value)) {
    return true;
  }
  if (!isValidShapeForDependency(value)) {
    return false;
  }

  Operation *defOp = value.getDefiningOp();
  // Op that can be processed both by CUBE and VECTOR should not be data
  // dependency
  if (defOp && isCubeAndVectorOp(defOp)) {
    return false;
  }

  return true;
}

// Helper: Check if value is a BlockArgument
bool DataDependencyAnalysisPass::isOuterOpArg(mlir::Value value) {
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *ownerBlock = blockArg.getOwner();
    return true;
  }
  return false;
}

// Resolve nested loop iterArg init values until reaching a non-BlockArgument.
mlir::Value DataDependencyAnalysisPass::resolveNestedIterArgInitValue(
    mlir::Value initValue) {
  llvm::DenseSet<mlir::Value> visited;
  mlir::Value currentValue = initValue;
  while (true) {
    if (!visited.insert(currentValue).second)
      break;
    auto blockArg = dyn_cast<mlir::BlockArgument>(currentValue);
    if (!blockArg)
      break;
    mlir::Operation *parentOp = blockArg.getOwner()->getParentOp();
    LOG_DEBUG("parentOp: " << *parentOp << "\n");

    unsigned argIndex = blockArg.getArgNumber();
    ValueRange initArgs;
    if (auto outerFor = dyn_cast<scf::ForOp>(parentOp)) {
      if (argIndex == 0)
        break;
      --argIndex;
      initArgs = outerFor.getInitArgs();
    } else if (auto outerWhile = dyn_cast<scf::WhileOp>(parentOp)) {
      // Triton keeps while before/after block args aligned 1:1 with init args.
      initArgs = outerWhile.getInits();
    } else {
      break;
    }

    LOG_DEBUG("argIndex: " << argIndex << "\n");
    if (argIndex >= initArgs.size())
      break;
    currentValue = initArgs[argIndex];
  }
  LOG_DEBUG("currentValue: " << currentValue << "\n");
  return currentValue;
}

// Helper: Build and record BlockInfo
void DataDependencyAnalysisPass::collectBlockInfo(
    DataDependencyInfo &info, int blockId,
    llvm::SmallVector<mlir::Operation *> &ops) {
  if (ops.empty()) {
    LOG_DEBUG("Warning: Block ID " << blockId << " has no operations.\n");
    return;
  }

  BlockInfo blockInfo;
  blockInfo.blockId = blockId;
  blockInfo.isCube = false;

  // In cases with one or more core_types
  // as long as there is a cube, it is necessary to check the dataflow.
  StringRef coreType = getSsbufferCoreType(ops[0]);
  if (coreType.contains(ssbufferCoreTypeCubeAttr)) {
    blockInfo.isCube = true;
  }

  blockInfo.isControl = false;
  if (isControlFlowOp(ops[0])) {
    blockInfo.isControl = true;
  }

  llvm::DenseSet<mlir::Operation *> opSet(ops.begin(), ops.end());

  for (auto *op : ops) {
    blockInfo.Operations.push_back(op);
    for (auto operand : op->getOperands()) {
      mlir::Operation *defOp = operand.getDefiningOp();
      // If defOp is not null and defOp is not in current ops set, it's an
      // external input
      if (!defOp || opSet.find(defOp) == opSet.end()) {
        blockInfo.inputs.insert(operand);
      }
    }
    for (auto result : op->getResults()) {
      // If any user is not in the current ops set, it's an external output
      bool hasExternalUser = false;
      for (mlir::Operation *user : result.getUsers()) {
        if (opSet.find(user) == opSet.end()) {
          hasExternalUser = true;
          break;
        }
      }
      if (hasExternalUser) {
        blockInfo.outputs.push_back(result);
      }
    }
  }

  info.getBlockInfoMap()[blockInfo.blockId] = blockInfo;

  LOG_DEBUG("Block_ID=" << blockInfo.blockId << "Processed!\n");
}

// Block Information Collection
void DataDependencyAnalysisPass::createBlockInfoMap(DataDependencyInfo &info) {
  int currentId = -2;
  static constexpr int startCurrId = -2;
  llvm::SmallVector<mlir::Operation *> currentOps;

  module.walk([&](mlir::Operation *op) {
    auto opBlockIdOpt = CVPipeline::getOpBlockId(op);
    if (opBlockIdOpt) {
      int opBlockId = *opBlockIdOpt;
      // When the id changes, the block ends && Exclude the initial state
      if (opBlockId != currentId && currentId != startCurrId) {
        collectBlockInfo(info, currentId, currentOps);
        currentOps.clear();
      }
      currentId = opBlockId;
      currentOps.push_back(op);
    }
  });
  // Process the last group
  if (!currentOps.empty()) {
    collectBlockInfo(info, currentId, currentOps);
  }
}

mlir::Operation *DataDependencyAnalysisPass::createBlockInfoConstOp(
    OpBuilder &builder, Location loc, llvm::StringRef coreType,
    DataDependencyInfo &info) {
  int newId = CVPipeline::getAvailableBlockId(module);
  auto constOp = builder.create<arith::ConstantIntOp>(loc, 0, constantIntType);
  setOpBlockId(constOp, newId);
  setOpCoreType(constOp, coreType);

  BlockInfo blockInfo;
  blockInfo.blockId = newId;
  blockInfo.isCube = (coreType == ssbufferCoreTypeCubeAttr);
  blockInfo.isControl = false;
  blockInfo.Operations.push_back(constOp);
  info.getBlockInfoMap()[newId] = blockInfo;

  return constOp;
}

bool DataDependencyAnalysisPass::collectDepInfo(
    mlir::Value depvalue, DependencyType dependencyType,
    llvm::SmallVector<DependencyInfo> &dependencies, int iniProdId,
    int iniConsId, DataDependencyInfo &info, bool isAllTranspoesd) {
  DependencyInfo depInfo;
  depInfo.type = dependencyType;
  depInfo.value = depvalue;
  LOG_DEBUG("try finding common level block IDs\n");
  depInfo.iniProducerBlockId = iniProdId;
  depInfo.iniConsumerBlockId = iniConsId;
  std::pair<int, int> commonLevelIds =
      findCommonLevelBlockIds(info, iniProdId, iniConsId);
  if (commonLevelIds.first == -1 || commonLevelIds.second == -1) {
    LOG_DEBUG("Could not find common level block IDs for producer and consumer "
              "blocks");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return false;
  }

  depInfo.producerBlockId = commonLevelIds.first;
  depInfo.consumerBlockId = commonLevelIds.second;
  if (isAllTranspoesd) {
    depInfo.isAllTranspoesd = true;
  }
  dependencies.push_back(depInfo);
  return true;
}

// Collects users of iterArg that have a different core type than initCoreType.
llvm::SmallVector<mlir::Operation *>
DataDependencyAnalysisPass::collectDiffCoreTypeUsers(
    mlir::BlockArgument iterArg, llvm::StringRef initCoreType) {
  llvm::SmallVector<mlir::Operation *> diffUsers;

  for (mlir::Operation *user : iterArg.getUsers()) {
    if (isa<scf::YieldOp>(user)) {
      continue;
    }
    if (isControlFlowOp(user)) {
      LOG_DEBUG("cannot process nested iterarg!");
      continue;
    }

    auto userCoreType = getCoreTypeWithIndex(user, 0);
    if (userCoreType != initCoreType && !userCoreType.empty()) {
      diffUsers.push_back(user);
    }
  }

  return diffUsers;
}

// Inserts a producer block at the beginning of the loop body and records
// cross-core-type dependencies for each user in diffUsers.
void DataDependencyAnalysisPass::insertProducerAndRecordDeps(
    mlir::LoopLikeOpInterface loopOp, mlir::BlockArgument loopArg,
    llvm::StringRef initCoreType,
    llvm::SmallVector<mlir::Operation *> &diffUsers, DataDependencyInfo &info) {
  auto &v2cDependencies = info.getV2CDependencies();
  auto &c2vDependencies = info.getC2VDependencies();

  Operation *loopOperation = loopOp.getOperation();
  CVPipeline::MainLoop loop(loopOperation);
  OpBuilder builder(loopOperation);
  builder.setInsertionPointToStart(loop.getBody());
  Location loc = loopOperation->getLoc();
  auto constOp = createBlockInfoConstOp(builder, loc, initCoreType, info);
  int newId = *CVPipeline::getOpBlockId(constOp);

  llvm::DenseSet<int> processedUserBlockIds;
  for (auto &user : diffUsers) {
    auto userBlockIdOpt = CVPipeline::getOpBlockId(user);
    if (!userBlockIdOpt) {
      LOG_DEBUG("Warning: User block ID not found for iterArg user.\n");
      continue;
    }
    int userBlockId = *userBlockIdOpt;
    if (!processedUserBlockIds.insert(userBlockId).second) {
      continue;
    }
    // Determine dependency type based on initCoreType
    DependencyType depType;
    if (initCoreType == ssbufferCoreTypeVectorAttr) {
      depType = DependencyType::VectorToCube;
    } else if (initCoreType == ssbufferCoreTypeCubeAttr) {
      depType = DependencyType::CubeToVector;
    }

    auto &targetDeps = (depType == DependencyType::VectorToCube)
                           ? v2cDependencies
                           : c2vDependencies;
    if (!collectDepInfo(loopArg, depType, targetDeps, newId, userBlockId,
                        info)) {
      continue;
    }

    LOG_DEBUG("Recorded iterArg dependency: "
              << initCoreType << " -> "
              << (depType == DependencyType::VectorToCube
                      ? ssbufferCoreTypeCubeAttr
                      : ssbufferCoreTypeVectorAttr)
              << ", producerBlockId=" << newId
              << ", consumerBlockId=" << userBlockId << "\n");
  }
}

void DataDependencyAnalysisPass::insertConsumerAndRecordDeps(
    mlir::LoopLikeOpInterface loopOp, mlir::Value yieldedValue,
    int iterArgIndex, llvm::StringRef initCoreType, DataDependencyInfo &info) {
  auto &v2cDependencies = info.getV2CDependencies();
  auto &c2vDependencies = info.getC2VDependencies();

  Operation *loopOperation = loopOp.getOperation();
  scf::YieldOp yieldOp = CVPipeline::MainLoop::getLoopYieldOp(loopOperation);
  OpBuilder builder(yieldOp);
  Location loc = yieldOp.getLoc();
  auto constOp = createBlockInfoConstOp(builder, loc, initCoreType, info);
  int newId = *CVPipeline::getOpBlockId(constOp);

  Operation *yieldedDefOp = yieldedValue.getDefiningOp();
  auto yieldedDefBlockIdOpt = CVPipeline::getOpBlockId(yieldedDefOp);
  if (!yieldedDefBlockIdOpt) {
    LOG_DEBUG("Warning: Yielded defining op block ID not found.\n");
    return;
  }
  int yieldedDefBlockId = *yieldedDefBlockIdOpt;

  DependencyType depType;
  if (initCoreType == ssbufferCoreTypeVectorAttr) {
    depType = DependencyType::CubeToVector;
  } else if (initCoreType == ssbufferCoreTypeCubeAttr) {
    depType = DependencyType::VectorToCube;
  }

  auto &targetDeps = (depType == DependencyType::VectorToCube)
                         ? v2cDependencies
                         : c2vDependencies;
  LOG_DEBUG("iniProducerBlockId=" << yieldedDefBlockId
                                  << ", iniConsumerBlockId=" << newId << "\n");
  if (!collectDepInfo(yieldedValue, depType, targetDeps, yieldedDefBlockId,
                      newId, info)) {
    return;
  }
  targetDeps.back().consumerYieldOp = yieldOp;

  updateCoreTypeAtIndex(yieldOp, iterArgIndex, initCoreType);
  updateCoreTypeAtIndex(loopOperation, iterArgIndex, initCoreType);

  LOG_DEBUG("Recorded yield producer dependency: "
            << initCoreType << ", iniProducerBlockId=" << yieldedDefBlockId
            << ", iniConsumerBlockId=" << newId << "\n");
}

void DataDependencyAnalysisPass::recordInitValueDeps(
    mlir::LoopLikeOpInterface loopOp, mlir::Value initValue,
    llvm::StringRef yieldCoreType, DataDependencyInfo &info) {
  auto &v2cDependencies = info.getV2CDependencies();
  auto &c2vDependencies = info.getC2VDependencies();

  Operation *initDefOp = initValue.getDefiningOp();
  auto initDefBlockIdOpt = CVPipeline::getOpBlockId(initDefOp);
  if (!initDefBlockIdOpt) {
    LOG_DEBUG("Warning: Init defining op block ID not found.\n");
    return;
  }
  int initDefBlockId = *initDefBlockIdOpt;

  auto loopBlockIdOpt = CVPipeline::getOpBlockId(loopOp.getOperation());
  if (!loopBlockIdOpt) {
    LOG_DEBUG("Warning: Loop block ID not found.\n");
    return;
  }
  int loopBlockId = *loopBlockIdOpt;

  DependencyType depType;
  if (yieldCoreType == ssbufferCoreTypeVectorAttr) {
    depType = DependencyType::CubeToVector;
  } else if (yieldCoreType == ssbufferCoreTypeCubeAttr) {
    depType = DependencyType::VectorToCube;
  }

  auto &targetDeps = (depType == DependencyType::VectorToCube)
                         ? v2cDependencies
                         : c2vDependencies;
  LOG_DEBUG("iniProducerBlockId=" << initDefBlockId << ", iniConsumerBlockId="
                                  << loopBlockId << "\n");
  if (!collectDepInfo(initValue, depType, targetDeps, initDefBlockId,
                      loopBlockId, info)) {
    return;
  }

  LOG_DEBUG("Recorded init value dependency: "
            << yieldCoreType << ", iniProducerBlockId=" << initDefBlockId
            << ", iniConsumerBlockId=" << loopBlockId << "\n");
}

static bool checkLoopYieldCoreType(scf::YieldOp yieldOp) {
  if (!yieldOp) {
    return false;
  }
  for (unsigned index = 0; index < yieldOp.getNumOperands(); ++index) {
    mlir::Value value = yieldOp.getOperand(index);
    llvm::StringRef yieldCoreType = getCoreTypeWithIndex(yieldOp, index);

    mlir::Operation *definingOp = value.getDefiningOp();
    if (!definingOp || !isa<scf::ForOp, scf::WhileOp>(definingOp)) {
      continue;
    }
    auto defResult = dyn_cast<mlir::OpResult>(value);
    int resultIndex = defResult ? defResult.getResultNumber() : 0;
    llvm::StringRef definingCoreType =
        getCoreTypeWithIndex(definingOp, resultIndex);

    if (yieldCoreType != definingCoreType) {
      return false;
    }
  }
  return true;
}

// Process iterArg dependencies for all supported loop-like operations.
void DataDependencyAnalysisPass::processIterArgDependencies() {
  auto &info = getAnalysis<DataDependencyInfo>();

  llvm::SmallVector<mlir::LoopLikeOpInterface> loopOps;
  module.walk(
      [&](mlir::LoopLikeOpInterface loopOp) { loopOps.push_back(loopOp); });
  LOG_DEBUG("Processing iterArg dependencies, found " << loopOps.size()
                                                      << " loop ops\n");

  for (mlir::LoopLikeOpInterface loopOp : loopOps) {
    Operation *loopOperation = loopOp.getOperation();
    if (!isa<scf::ForOp, scf::WhileOp>(loopOperation)) {
      continue;
    }

    CVPipeline::MainLoop loop(loopOperation);
    scf::YieldOp yieldOp = CVPipeline::MainLoop::getLoopYieldOp(loopOperation);
    if (!checkLoopYieldCoreType(yieldOp)) {
      LOG_DEBUG("[ERROR]: Yield core type mismatch defining op\n");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }

    size_t numIterArgs = loopOp.getInits().size();
    SmallVector<Value> iterArgs = loop.getIterArgs();
    if (iterArgs.size() != numIterArgs) {
      LOG_DEBUG("[ERROR]: Loop iter_arg count does not match init count\n");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }

    for (int iterArgIndex = 0; iterArgIndex < numIterArgs; ++iterArgIndex) {
      mlir::Value initValue = loopOp.getInits()[iterArgIndex];
      mlir::BlockArgument iterArg =
          cast<mlir::BlockArgument>(iterArgs[iterArgIndex]);
      mlir::Value yieldedValue = loopOp.getYieldedValues()[iterArgIndex];
      LOG_DEBUG("initValue" << initValue << "\n");
      LOG_DEBUG("yieldedValue" << yieldedValue << "\n");

      if (!isValid1DValueForDependency(iterArg) &&
          (!isValidShapeForDependency(initValue) ||
           !isValidShapeForDependency(yieldedValue))) {
        LOG_DEBUG("iterarg: " << iterArg
                              << " is not valid tensor for dependency!");
        continue;
      }

      Operation *initDefOp = initValue.getDefiningOp();
      Operation *yieldedDefOp = yieldedValue.getDefiningOp();
      if (!yieldedDefOp) {
        continue;
      }
      auto yieldCoreType = getCoreTypeWithIndex(loopOperation, iterArgIndex);

      if (!initDefOp) {
        auto realInitValue = resolveNestedIterArgInitValue(initValue);
        auto realInitDefOp = realInitValue.getDefiningOp();
        auto realInitDefReuslt = dyn_cast<mlir::OpResult>(realInitValue);
        if (!realInitDefOp || isCubeAndVectorOp(realInitDefOp)) {
          continue;
        }
        if (getCoreTypeWithIndex(realInitDefOp,
                                 realInitDefReuslt
                                     ? realInitDefReuslt.getResultNumber()
                                     : 0) != yieldCoreType) {
          CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
          LOG_DEBUG("[info]: nested conflict iterarg!");
          return;
        }
        initDefOp = realInitDefOp;
        initValue = realInitValue;
      }
      auto initDefReuslt = dyn_cast<mlir::OpResult>(initValue);
      auto initCoreType = getCoreTypeWithIndex(
          initDefOp, initDefReuslt ? initDefReuslt.getResultNumber() : 0);

      LOG_DEBUG("[initDefOp]: " << *initDefOp << "\n");
      if (isCubeAndVectorOp(initDefOp)) {
        if (yieldCoreType == CVPipeline::kCoreTypeVector) {
          auto diffUsers = collectDiffCoreTypeUsers(iterArg, yieldCoreType);
          if (!diffUsers.empty()) {
            insertProducerAndRecordDeps(loopOp, iterArg, yieldCoreType,
                                        diffUsers, info);
          }
        } else if (yieldCoreType == CVPipeline::kCoreTypeCube) {
          auto vectorUsers =
              collectDiffCoreTypeUsers(iterArg, CVPipeline::kCoreTypeCube);
          auto cubeUsers =
              collectDiffCoreTypeUsers(iterArg, CVPipeline::kCoreTypeVector);
          if (!vectorUsers.empty()) {
            insertConsumerAndRecordDeps(loopOp, yieldedValue, iterArgIndex,
                                        CVPipeline::kCoreTypeVector, info);
            if (!cubeUsers.empty()) {
              insertProducerAndRecordDeps(loopOp, iterArg,
                                          CVPipeline::kCoreTypeVector,
                                          cubeUsers, info);
            }
          }
        }
      } else if (initCoreType == yieldCoreType) {
        auto diffUsers = collectDiffCoreTypeUsers(iterArg, yieldCoreType);
        if (!diffUsers.empty()) {
          insertProducerAndRecordDeps(loopOp, iterArg, yieldCoreType, diffUsers,
                                      info);
        }
      } else {
        llvm::SmallVector<mlir::Operation *> initCoreTypeUsers;
        llvm::SmallVector<mlir::Operation *> yieldCoreTypeUsers;

        for (mlir::Operation *user : iterArg.getUsers()) {
          if (isa<scf::YieldOp>(user)) {
            continue;
          }
          if (isControlFlowOp(user)) {
            LOG_DEBUG("cannot process nested iterarg!");
            continue;
          }
          auto userCoreType = getCoreTypeWithIndex(user, 0);
          if (userCoreType == initCoreType) {
            initCoreTypeUsers.push_back(user);
          } else if (userCoreType == yieldCoreType) {
            yieldCoreTypeUsers.push_back(user);
          }
        }
        if (!initCoreTypeUsers.empty() && !yieldCoreTypeUsers.empty()) {
          recordInitValueDeps(loopOp, initValue, yieldCoreType, info);
          insertProducerAndRecordDeps(loopOp, iterArg, yieldCoreType,
                                      yieldCoreTypeUsers, info);
        } else if (!initCoreTypeUsers.empty() && yieldCoreTypeUsers.empty()) {
          insertConsumerAndRecordDeps(loopOp, yieldedValue, iterArgIndex,
                                      initCoreType, info);
        } else if (initCoreTypeUsers.empty() && !yieldCoreTypeUsers.empty()) {
          recordInitValueDeps(loopOp, initValue, yieldCoreType, info);
        } else {
          LOG_DEBUG("no dependencies with: " << iterArg << "\n");
        }
      }
    }
  }
}

// Analyze V->C and C->C
void DataDependencyAnalysisPass::analyzeExternalInputs(
    DataDependencyInfo &info) {
  auto &blockInfoMap = info.getBlockInfoMap();
  auto &v2cDependencies = info.getV2CDependencies();

  LOG_DEBUG("Analyzing external inputs for Cube blocks...\n");
  for (auto &[id, blockInfo] : blockInfoMap) {
    if (!blockInfo.isCube || blockInfo.isControl || blockInfo.inputs.empty())
      continue;
    LOG_DEBUG("Analyzing external inputs for Cube Block ID: " << id << "\n");
    for (mlir::Value input : blockInfo.inputs) {
      // Check if input is a value which can be produced by CUBE
      if (!isValidValueForDependency(input)) {
        LOG_DEBUG("Warning: [v->c] Input value is not a valid tensor for "
                  "dependency analysis.\n");
        continue;
      }
      // Check if input is a blockarg.
      if (isOuterOpArg(input)) {
        LOG_DEBUG("Warning: [v->c] Input value is a function/scf parameter.\n");
        continue;
      }

      Operation *defOp = input.getDefiningOp();
      auto defReuslt = dyn_cast<mlir::OpResult>(input);
      auto coreType = getCoreTypeWithIndex(
          defOp, defReuslt ? defReuslt.getResultNumber() : 0);
      if (coreType == "") {
        LOG_DEBUG("Warning: [v->c] Input value has no core type attribute.\n");
        continue;
      }

      // Case 1: Cube -> C->C dependency
      if (coreType == ssbufferCoreTypeCubeAttr) {
        LOG_DEBUG("Found external input with CUBE core type: " << input
                                                               << "\n");
        auto producerIdOpt = CVPipeline::getOpBlockId(input.getDefiningOp());
        if (!producerIdOpt) {
          LOG_DEBUG("Warning: [c->c] Producer block ID not found for input "
                    "value.\n");
          continue;
        }
        int producerId = *producerIdOpt;
        if (!collectDepInfo(input, DependencyType::CubeToCube,
                            info.getC2CDependencies(), producerId,
                            blockInfo.blockId, info)) {
          continue;
        }
        continue;
      }
      // Case 2: Vector -> V->C dependency
      if (coreType == ssbufferCoreTypeVectorAttr) {
        LOG_DEBUG("Found external input with VECTOR core type: " << input
                                                                 << "\n");
        auto producerIdOpt = CVPipeline::getOpBlockId(input.getDefiningOp());
        if (!producerIdOpt) {
          LOG_DEBUG(
              "Warning: [v->c] Producer block ID not found for input value.\n");
          continue;
        }
        int producerId = *producerIdOpt;
        if (!collectDepInfo(input, DependencyType::VectorToCube,
                            v2cDependencies, producerId, blockInfo.blockId,
                            info)) {
          continue;
        }
      }
    }
  }
  LOG_DEBUG("External input analysis complete.\n");
}

// Analyze C->V
void DataDependencyAnalysisPass::analyzeExternalOutputs(
    DataDependencyInfo &info) {
  auto &blockInfoMap = info.getBlockInfoMap();
  auto &c2vDependencies = info.getC2VDependencies();

  LOG_DEBUG("Analyzing external outputs for Cube blocks...\n");
  for (auto &[id, blockInfo] : blockInfoMap) {
    if (!blockInfo.isCube || blockInfo.outputs.empty())
      continue;

    for (mlir::Value output : blockInfo.outputs) {
      // Check if output is a value which can be produced by CUBE
      if (!isValidValueForDependency(output)) {
        LOG_DEBUG("Warning: [c->v] Output value is not a valid tensor for "
                  "dependency analysis.\n");
        continue;
      }
      if (isa<mlir::IntegerType, mlir::FloatType>(output.getType())) {
        LOG_DEBUG("Warning: [c->v] Output value is a scalar, not a valid "
                  "tensor for dependency analysis.\n");
        continue;
      }

      auto opResult = dyn_cast<OpResult>(output);
      unsigned resultIndex = opResult.getResultNumber();
      StringRef resultCoreType =
          getCoreTypeWithIndex(output.getDefiningOp(), resultIndex);
      if (resultCoreType != ssbufferCoreTypeCubeAttr) {
        continue;
      }

      // Check who is using this output
      llvm::DenseSet<int> handledBlockIds;

      // if c->v value will be transposed and then used by vector op, the value
      // can be transposed within fixpipe
      auto [isAllTranspoesd, transposedYieldOp] =
          isAllTransposedInVector(output);

      for (mlir::Operation *user : output.getUsers()) {
        int outputIndex = 0;
        if (isa<scf::YieldOp>(user)) {
          for (unsigned i = 0; i < user->getNumOperands(); ++i) {
            if (user->getOperand(i) == output) {
              outputIndex = i;
              break;
            }
          }
        }
        auto userCoreType = getCoreTypeWithIndex(user, outputIndex);
        if ((userCoreType == "")) {
          LOG_DEBUG(
              "Warning: [c->v] Input value has no core type attribute.\n");
          continue;
        }
        if (userCoreType == ssbufferCoreTypeVectorAttr) {
          LOG_DEBUG("Found external output used by VECTOR core type: " << output
                                                                       << "\n");
          auto consumerIdOpt = CVPipeline::getOpBlockId(user);
          if (!consumerIdOpt) {
            LOG_DEBUG("Warning: [c->v] Consumer block ID not found for user "
                      "operation.\n");
            continue;
          }
          int consumerId = *consumerIdOpt;
          auto inserted = handledBlockIds.insert(consumerId).second;
          if (inserted) {
            if (!collectDepInfo(output, DependencyType::CubeToVector,
                                c2vDependencies, blockInfo.blockId, consumerId,
                                info, isAllTranspoesd)) {
              continue;
            }
            if (transposedYieldOp.has_value()) {
              c2vDependencies.back().consumerYieldOp = *transposedYieldOp;
            }
          }
        }
        // If user belongs to Cube block, this C->C dependency was handled
        // in the Input analysis phase, so here we only handle C->V.
      }
    }
  }
  LOG_DEBUG("External output analysis complete.\n");
}

// Trace an operand's defining op back to find the source matmul.
static linalg::MatmulOp resolveSameBlockMatmulProducer(mlir::Value operand,
                                                       int consumerBlockId) {
  Operation *defOp = CVPipeline::getSourceThroughCIntermediateOps(operand);

  auto producer = dyn_cast_if_present<linalg::MatmulOp>(defOp);
  if (!producer) {
    return nullptr;
  }
  auto producerBlockIdOpt = CVPipeline::getOpBlockId(producer);
  if (!producerBlockIdOpt || *producerBlockIdOpt != consumerBlockId) {
    return nullptr;
  }
  return producer;
}

// Analyze C->C dependencies between the matmuls inside the same computeBlock.
void DataDependencyAnalysisPass::analyzeInternalDeps(DataDependencyInfo &info) {
  auto &blockInfoMap = info.getBlockInfoMap();
  auto &intraBlockDeps = info.getIntraC2CDependencies();

  LOG_DEBUG("Analyzing intra-block matmul dependencies...\n");
  for (auto &[blockId, blockInfo] : blockInfoMap) {
    if (!blockInfo.isCube) {
      continue;
    }

    llvm::SmallVector<linalg::MatmulOp> matmuls;
    for (mlir::Operation *op : blockInfo.Operations) {
      if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op)) {
        matmuls.push_back(matmulOp);
      }
    }
    if (matmuls.size() < 2) {
      continue;
    }

    for (linalg::MatmulOp consumer : matmuls) {
      for (OpOperand &opOperand : consumer->getOpOperands()) {
        if (!resolveSameBlockMatmulProducer(opOperand.get(), blockId)) {
          continue;
        }
        DependencyInfo depInfo;
        depInfo.type = DependencyType::CubeToCube;
        depInfo.value = opOperand.get();
        depInfo.operand = &opOperand;
        depInfo.producerBlockId = blockId;
        depInfo.consumerBlockId = blockId;
        depInfo.iniProducerBlockId = blockId;
        depInfo.iniConsumerBlockId = blockId;
        intraBlockDeps.push_back(depInfo);
      }
    }
  }
  LOG_DEBUG("Intra-block matmul dependency analysis complete. Found "
            << intraBlockDeps.size() << " dependencies.\n");
}

void DataDependencyAnalysisPass::collectMemDepInfo(
    llvm::StringRef predCoreType, int producerBlockId, int consumerBlockId,
    int predBlockId, int currBlockId,
    llvm::SmallVector<DependencyInfo> &memoryDependencies,
    mlir::Operation *predOp, mlir::Operation *nextOp) {
  DependencyInfo depInfo;

  if (predCoreType == ssbufferCoreTypeCubeAttr) {
    depInfo.type = DependencyType::CubeToVector;
  } else if (predCoreType == ssbufferCoreTypeVectorAttr) {
    depInfo.type = DependencyType::VectorToCube;
  }
  depInfo.producerBlockId = producerBlockId;
  depInfo.consumerBlockId = consumerBlockId;
  depInfo.iniProducerBlockId = predBlockId;
  depInfo.iniConsumerBlockId = currBlockId;

  depInfo.predOp = predOp;
  depInfo.nextOp = nextOp;

  memoryDependencies.push_back(depInfo);
}

void DataDependencyAnalysisPass::analyzeMemoryEffect(DataDependencyInfo &info) {
  auto &memoryDependencies = info.getMemoryDependencies();
  LOG_DEBUG("\n=== start mem dep analysis ===\n");

  auto &aliasAnalysis = getAnalysis<mlir::AliasAnalysis>();
  MemoryDependenceGraph memDepGraph(module, aliasAnalysis);

  auto walkResult = module.walk([&](mlir::Operation *op) -> WalkResult {
    if (op->getNumRegions() > 0) {
      return WalkResult::advance();
    }
    if (isa<annotation::MarkOp, gpu::BarrierOp>(op)) {
      return WalkResult::advance();
    }
    auto currBlockIdOpt = CVPipeline::getOpBlockId(op);
    llvm::StringRef currCoreType = getSsbufferCoreType(op);
    if (!currBlockIdOpt || currCoreType.empty()) {
      return WalkResult::advance();
    }
    int currBlockId = *currBlockIdOpt;

    for (mlir::Operation *predOp : memDepGraph.getExecBefore(op)) {
      if (isa<annotation::MarkOp, gpu::BarrierOp>(predOp)) {
        continue;
      }
      if (predOp->getNumRegions() > 0) {
        auto realdeps = memDepGraph.getRealDependency(predOp, op);
        if (realdeps.empty()) {
          return WalkResult::advance();
        }
        for (mlir::Operation *realPredOp : realdeps) {
          if (isa<annotation::MarkOp, gpu::BarrierOp>(realPredOp)) {
            continue;
          }
          auto realPredBlockIdOpt = CVPipeline::getOpBlockId(realPredOp);
          llvm::StringRef realPredCoreType = getSsbufferCoreType(realPredOp);
          if (!realPredBlockIdOpt || realPredCoreType == currCoreType ||
              realPredCoreType.empty()) {
            continue;
          }
          int realPredBlockId = *realPredBlockIdOpt;
          auto [producerBlockId, consumerBlockId] =
              findCommonLevelBlockIds(info, realPredBlockId, currBlockId);
          if (producerBlockId == -1 || consumerBlockId == -1) {
            LOG_DEBUG("Could not find common level block IDs for producer and "
                      "consumer blocks");
            return WalkResult::interrupt();
          }
          collectMemDepInfo(realPredCoreType, producerBlockId, consumerBlockId,
                            realPredBlockId, currBlockId, memoryDependencies,
                            realPredOp, op);

          LOG_DEBUG("\n=op with region mem dep analysis= "
                    << "\nrealpredcoretype" << realPredCoreType
                    << "\nproducer Block: " << realPredBlockId
                    << "\nproducer Op: " << *realPredOp << "\nconsumer Block: "
                    << currBlockId << "\nconsumer Op: " << *op << "\n");
        }
        continue;
      }
      auto predBlockIdOpt = CVPipeline::getOpBlockId(predOp);
      llvm::StringRef predCoreType = getSsbufferCoreType(predOp);
      if (!predBlockIdOpt || predCoreType == currCoreType ||
          predCoreType.empty()) {
        continue;
      }
      int predBlockId = *predBlockIdOpt;

      auto [producerBlockId, consumerBlockId] =
          findCommonLevelBlockIds(info, predBlockId, currBlockId);
      if (producerBlockId == -1 || consumerBlockId == -1) {
        LOG_DEBUG("Could not find common level block IDs for producer and "
                  "consumer blocks");
        return WalkResult::interrupt();
      }
      if (producerBlockId == consumerBlockId) {
        continue;
      }

      collectMemDepInfo(predCoreType, producerBlockId, consumerBlockId,
                        predBlockId, currBlockId, memoryDependencies, predOp,
                        op);

      LOG_DEBUG("\n=mem dep analysis= " << "\npredcoretype" << predCoreType
                                        << "\nproducer Block: " << predBlockId
                                        << "\nproducer Op: " << *predOp
                                        << "\nconsumer Block: " << currBlockId
                                        << "\nconsumer Op: " << *op << "\n");
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
  }
  LOG_DEBUG("=== mem dep analysis complete ===\n");
}

// Producer/Consumer Hierarchy Analysis
std::pair<int, int> DataDependencyAnalysisPass::findCommonLevelBlockIds(
    DataDependencyInfo &info, int producerBlockId, int consumerBlockId) {
  auto &blockInfoMap = info.getBlockInfoMap();

  LOG_DEBUG("start findCommonLevelBlockIds...\n");

  // Step 1: Get corresponding BlockInfo from Map
  auto pIt = blockInfoMap.find(producerBlockId);
  auto cIt = blockInfoMap.find(consumerBlockId);
  // Defensive programming: if corresponding Block info not found, return
  // original ID or error code
  if (pIt == blockInfoMap.end() || cIt == blockInfoMap.end()) {
    return {producerBlockId, consumerBlockId};
  }

  BlockInfo &pInfo = pIt->second;
  BlockInfo &cInfo = cIt->second;

  // Take the first operation of each Block as representative to check hierarchy
  // (Assumes all operations in a Block are closely related in hierarchy)
  mlir::Operation *producerOp = pInfo.Operations[0];
  mlir::Operation *consumerOp = cInfo.Operations[0];

  mlir::Block *pBlock = producerOp->getBlock();
  mlir::Block *cBlock = consumerOp->getBlock();

  // Case 1: In the same MLIR Block
  if (pBlock == cBlock) {
    return {producerBlockId, consumerBlockId};
  }

  // Case 2: In different Blocks, find Lowest Common Ancestor (LCA)
  // Step 1: Collect producer's ancestor chain
  llvm::SmallVector<mlir::Operation *> pAncestors;
  pAncestors.push_back(producerOp);
  mlir::Operation *current = producerOp->getParentOp();
  while (current) {
    pAncestors.push_back(current);
    current = current->getParentOp();
  }

  // Step 2: Walk up consumer's ancestors, using current and before for rolling
  mlir::Operation *before = consumerOp; // Initialize as consumerOp itself
  current = consumerOp;                 // Initialize as parent

  while (current) {
    // --- Found common ancestor ---
    auto it = std::find(pAncestors.begin(), pAncestors.end(), current);
    if (it != pAncestors.end()) {
      size_t pIndex = std::distance(pAncestors.begin(), it);
      if (pIndex == 0) {
        break;
      }
      mlir::Operation *pPrevOp = pAncestors[pIndex - 1];
      auto pPrevIdOpt = CVPipeline::getOpBlockId(pPrevOp);
      auto cPrevIdOpt = CVPipeline::getOpBlockId(before);
      int pPrevId = pPrevIdOpt ? *pPrevIdOpt : -1;
      int cPrevId = cPrevIdOpt ? *cPrevIdOpt : -1;
      if (!pPrevIdOpt) {
        LOG_DEBUG("Warning: Producer ancestor operation has no block ID "
                  "attribute.\n");
      }
      if (!cPrevIdOpt) {
        LOG_DEBUG("Warning: Consumer ancestor operation has no block ID "
                  "attribute.\n");
      }
      return {pPrevId, cPrevId};
    }

    // Rolling: continue upward
    before = current;
    current = current->getParentOp();
  }
  LOG_DEBUG("Warning: No common ancestor found for producer block "
            << producerBlockId << " and consumer block " << consumerBlockId
            << "\n");
  return {-1, -1};
}

// Deduplicate dependencies: remove duplicates with same value,
// iniConsumerBlockId, and iniProducerBlockId
void DataDependencyAnalysisPass::deduplicateDependencies(
    llvm::SmallVector<DependencyInfo> &dependencies) {
  auto newEnd =
      std::unique(dependencies.begin(), dependencies.end(),
                  [](const DependencyInfo &a, const DependencyInfo &b) {
                    return a.value == b.value &&
                           a.iniConsumerBlockId == b.iniConsumerBlockId &&
                           a.iniProducerBlockId == b.iniProducerBlockId;
                  });

  dependencies.erase(newEnd, dependencies.end());
}

void DataDependencyAnalysisPass::runOnOperation() {
  LOG_DEBUG("\n--- enter DataDependencyAnalysisPass --->\n");
  module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  auto &info = getAnalysis<DataDependencyInfo>();

  // Step 1: Collect block information (populate blockInfoMap)
  createBlockInfoMap(info);

  // Step 2: Analyze iter_args dependencies
  processIterArgDependencies();
  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }
  createBlockInfoMap(info);

  // Step 3: Analyze dependencies (v2c/c2v/c2c).  Scalar V->C deps come from
  // analyzeExternalInputs (VECTOR scalars crossing into CUBE).
  analyzeExternalInputs(info);

  analyzeExternalOutputs(info);

  // Step 4: Analyze intra-block c2c dependencies
  analyzeInternalDeps(info);

  // Step 5: Analyze memory dependencies (memdep sync)
  analyzeMemoryEffect(info);

  // Step 6: Deduplicate dependencies (remove duplicates with same value,
  // iniConsumerBlockId, iniProducerBlockId)
  deduplicateDependencies(info.getV2CDependencies());
  deduplicateDependencies(info.getC2VDependencies());
  deduplicateDependencies(info.getC2CDependencies());
  deduplicateDependencies(info.getMemoryDependencies());

  info.setValid(true);

  LOG_DEBUG("DataDependencyAnalysisPass: Analysis complete.\n");
  LOG_DEBUG("  V->C dependencies: " << info.getV2CDependencies().size()
                                    << "\n");
  LOG_DEBUG("  C->V dependencies: " << info.getC2VDependencies().size()
                                    << "\n");
  LOG_DEBUG("  C->C dependencies: " << info.getC2CDependencies().size()
                                    << "\n");
  LOG_DEBUG("  Memory dependencies: " << info.getMemoryDependencies().size()
                                      << "\n");
  LOG_DEBUG("  Intra-block C->C dependencies: "
            << info.getIntraC2CDependencies().size() << "\n");

  LOG_DEBUG("\n--- exit DataDependencyAnalysisPass --->\n");
}

// Create the pass
namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createDataDependencyAnalysisPass() {
  return std::make_unique<DataDependencyAnalysisPass>();
}

} // namespace triton
} // namespace mlir
