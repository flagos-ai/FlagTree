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

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

static constexpr const char *DEBUG_TYPE = "data-dependency-analysis";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir::triton;
using namespace mlir::CVPipeline;

// Attribute name constants
static constexpr const char *kBlockIdAttr = "ssbuffer.block_id";
static constexpr const char *kCoreTypeAttr = "ssbuffer.core_type";
static constexpr const char *kTransferIdAttr = "ssbuffer.transfer_id";

// Helper: ssbuffer.core_type
llvm::StringRef getSsbufferCoreType(Operation *op)
{
    if (auto attr = op->getAttrOfType<mlir::StringAttr>(kCoreTypeAttr)) {
        return attr.getValue();
    }
    return "";
}

// Helper: Get CoreType from op and index
llvm::StringRef getCoreTypeWithIndex(Operation *op, int index)
{
    llvm::StringRef typeStr = getSsbufferCoreType(op);
    if (typeStr.contains(", ")) {
        llvm::SmallVector<llvm::StringRef> types;
        typeStr.split(types, ", ", -1, false);
        if (index < types.size()) {
            return types[index].trim();
        }
        LOG_DEBUG("Warning: Core type string has multiple types but value is not an OpResult or index out of range.\n");
        return "";
    }

    return typeStr;
}

// Helper: Check if operation is control flow
bool DataDependencyAnalysisPass::isControlFlowOp(mlir::Operation *op)
{
    if (!op)
        return false;
    return isa<scf::ForOp>(op) || isa<scf::IfOp>(op) || isa<scf::WhileOp>(op) || isa<scf::YieldOp>(op);
}

// Helper: Build and record BlockInfo
void DataDependencyAnalysisPass::collectBlockInfo(DataDependencyInfo &info, int blockId,
    llvm::SmallVector<mlir::Operation *> &ops)
{
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
    if (coreType.contains("CUBE")) {
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
            // If defOp is not null and defOp is not in current ops set, it's an external input
            if (!defOp || opSet.find(defOp) == opSet.end()) {
                blockInfo.inputs.push_back(operand);
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
void DataDependencyAnalysisPass::createBlockInfoMap(DataDependencyInfo &info)
{
    int currentId = -2;
    static constexpr int startCurrId = -2;
    llvm::SmallVector<mlir::Operation *> currentOps;

    module.walk([&](mlir::Operation *op) {
        int opBlockId = getSsbufferBlockId(op);
        if (opBlockId != -1) {
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

void DataDependencyAnalysisPass::collectDepInfo(mlir::Value depvalue, DependencyType dependencyType,
    llvm::SmallVector<DependencyInfo> &dependencies, int iniProdId, int iniConsId, DataDependencyInfo &info)
{
    DependencyInfo depInfo;
    depInfo.type = dependencyType;
    depInfo.value = depvalue;
    LOG_DEBUG("try finding common level block IDs\n");
    depInfo.iniProducerBlockId = iniProdId;
    depInfo.iniConsumerBlockId = iniConsId;
    std::pair<int, int> commonLevelIds = findCommonLevelBlockIds(info, iniProdId, iniConsId);
    assert(commonLevelIds.first != -1 && commonLevelIds.second != -1 &&
        "Could not find common level block IDs for producer and consumer blocks");
    depInfo.producerBlockId = commonLevelIds.first;
    depInfo.consumerBlockId = commonLevelIds.second;

    dependencies.push_back(depInfo);
}

// Analyze V->C
void DataDependencyAnalysisPass::analyzeExternalInputs(DataDependencyInfo &info)
{
    auto &blockInfoMap = info.getBlockInfoMap();
    auto &v2cDependencies = info.getV2CDependencies();

    LOG_DEBUG("Analyzing external inputs for Cube blocks...\n");
    for (auto &[id, blockInfo] : blockInfoMap) {
        if (!blockInfo.isCube || blockInfo.isControl || blockInfo.inputs.empty())
            continue;
        LOG_DEBUG("Analyzing external inputs for Cube Block ID: " << id << "\n");
        for (mlir::Value input : blockInfo.inputs) {
            // Check if input is a func.func blockarg.
            if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(input)) {
                LOG_DEBUG("Warning: [v->c] Input value is a function parameter.\n");
                continue;
            }
            // Check if input is a value which can be produced by CUBE
            if (!dyn_cast<mlir::TensorType>(input.getType())) {
                LOG_DEBUG("Warning: [v->c] Input value is not TensorType\n");
                continue;
            }
            if (isa<tensor::EmptyOp, linalg::FillOp>(input.getDefiningOp())) {
                LOG_DEBUG("Warning: [v->c] Input value is defined by tensor::EmptyOp/linalg::FillOp.\n");
                continue;
            }

            Operation *defOp = input.getDefiningOp();
            auto defReuslt = dyn_cast<mlir::OpResult>(input);
            auto coreType = getCoreTypeWithIndex(defOp, defReuslt ? defReuslt.getResultNumber() : 0);
            if (coreType == "") {
                LOG_DEBUG("Warning: [v->c] Input value has no core type attribute.\n");
                continue;
            }

            // Case 1: Cube -> C->C special case
            if (coreType == "CUBE") {
                continue;
            }
            // Case 2: Vector -> V->C dependency
            if (coreType == "VECTOR") {
                LOG_DEBUG("Found external input with VECTOR core type: " << input << "\n");
                auto producerId = getSsbufferBlockId(input.getDefiningOp());
                if (producerId == -1) {
                    LOG_DEBUG("Warning: [v->c] Producer block ID not found for input value.\n");
                    continue;
                }
                collectDepInfo(input, DependencyType::VectorToCube, v2cDependencies, producerId, blockInfo.blockId,
                    info);
            }
        }
    }
    LOG_DEBUG("External input analysis complete.\n");
}

// Analyze C->V
void DataDependencyAnalysisPass::analyzeExternalOutputs(DataDependencyInfo &info)
{
    auto &blockInfoMap = info.getBlockInfoMap();
    auto &c2vDependencies = info.getC2VDependencies();

    LOG_DEBUG("Analyzing external outputs for Cube blocks...\n");
    for (auto &[id, blockInfo] : blockInfoMap) {
        if (!blockInfo.isCube || blockInfo.outputs.empty())
            continue;

        for (mlir::Value output : blockInfo.outputs) {
            // Check if input is a value which can be produced by VECTOR
            if (!dyn_cast<mlir::TensorType>(output.getType())) {
                LOG_DEBUG("Warning: ExternalOutput is not TensorType\n");
                continue;
            }
            if (isa<tensor::EmptyOp, linalg::FillOp>(output.getDefiningOp())) {
                LOG_DEBUG("Warning: [c->v] output value is defined by tensor::EmptyOp/linalg::FillOp.\n");
                continue;
            }

            auto opResult = dyn_cast<OpResult>(output);
            unsigned resultIndex = opResult.getResultNumber();
            StringRef resultCoreType = getCoreTypeWithIndex(output.getDefiningOp(), resultIndex);
            if (resultCoreType != "CUBE") {
                continue;
            }
            // Check who is using this output
            for (mlir::Operation *user : output.getUsers()) {
                int outputIndex = 0;
                if (isControlFlowOp(user)) {
                    for (unsigned i = 0; i < user->getNumOperands(); ++i) {
                        if (user->getOperand(i) == output) {
                            outputIndex = i;
                            break;
                        }
                    }
                }
                auto userCoreType = getCoreTypeWithIndex(user, outputIndex);
                if ((userCoreType == "")) {
                    LOG_DEBUG("Warning: [c->v] Input value has no core type attribute.\n");
                    continue;
                }
                if (userCoreType == "VECTOR") {
                    LOG_DEBUG("Found external output used by VECTOR core type: " << output << "\n");
                    auto consumerId = getSsbufferBlockId(user);
                    if (consumerId == -1) {
                        LOG_DEBUG("Warning: [c->v] Consumer block ID not found for user operation.\n");
                        continue;
                    }
                    collectDepInfo(output, DependencyType::CubeToVector, c2vDependencies, blockInfo.blockId, consumerId,
                        info);
                }
                // If user belongs to Cube block, this C->C dependency was handled
                // in the Input analysis phase, so here we only handle C->V.
            }
        }
    }
    LOG_DEBUG("External output analysis complete.\n");
}

void DataDependencyAnalysisPass::analyzeMemoryEffect(DataDependencyInfo &info)
{
    auto &memoryDependencies = info.getMemoryDependencies();
    LOG_DEBUG("\n=== start mem dep analysis ===\n");

    auto &aliasAnalysis = getAnalysis<mlir::AliasAnalysis>();
    MemoryDependenceGraph memDepGraph(module, aliasAnalysis);

    module.walk([&](mlir::Operation *op) {
        int currBlockId = getSsbufferBlockId(op);
        llvm::StringRef currCoreType = getSsbufferCoreType(op);
        if (currBlockId == -1 || currCoreType.empty()) {
            return;
        }

        for (mlir::Operation *predOp : memDepGraph.getExecBefore(op)) {
            int predBlockId = getSsbufferBlockId(predOp);
            llvm::StringRef predCoreType = getSsbufferCoreType(predOp);
            if (predBlockId == -1 || predCoreType == currCoreType || predCoreType.empty()) {
                continue;
            }

            auto [producerBlockId, consumerBlockId] = findCommonLevelBlockIds(info, predBlockId, currBlockId);
            assert(producerBlockId != -1 && consumerBlockId != -1 &&
                "Could not find common level block IDs for producer and consumer blocks");
            if (producerBlockId == consumerBlockId) {
                continue;
            }

            DependencyInfo depInfo;

            if (predCoreType == "CUBE") {
                depInfo.type = DependencyType::CubeToVector;
            } else if (predCoreType == "VECTOR") {
                depInfo.type = DependencyType::VectorToCube;
            }
            depInfo.producerBlockId = producerBlockId;
            depInfo.consumerBlockId = consumerBlockId;
            depInfo.iniProducerBlockId = predBlockId;
            depInfo.iniConsumerBlockId = currBlockId;

            memoryDependencies.push_back(depInfo);
            LOG_DEBUG("=mem dep analysis= "
                << "producer Block: " << predBlockId << "consumer Block: " << currBlockId << "\n");
        }
    });
    LOG_DEBUG("=== mem dep analysis complete ===\n");
}

// Producer/Consumer Hierarchy Analysis
std::pair<int, int> DataDependencyAnalysisPass::findCommonLevelBlockIds(DataDependencyInfo &info, int producerBlockId,
    int consumerBlockId)
{
    auto &blockInfoMap = info.getBlockInfoMap();

    LOG_DEBUG("start findCommonLevelBlockIds...\n");

    // Step 1: Get corresponding BlockInfo from Map
    auto pIt = blockInfoMap.find(producerBlockId);
    auto cIt = blockInfoMap.find(consumerBlockId);
    // Defensive programming: if corresponding Block info not found, return original ID or error code
    if (pIt == blockInfoMap.end() || cIt == blockInfoMap.end()) {
        return { producerBlockId, consumerBlockId };
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
        return { producerBlockId, consumerBlockId };
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
            int pPrevId = getSsbufferBlockId(pPrevOp);
            int cPrevId = getSsbufferBlockId(before);
            if (pPrevId == -1) {
                LOG_DEBUG("Warning: Producer ancestor operation has no block ID attribute.\n");
            }
            if (cPrevId == -1) {
                LOG_DEBUG("Warning: Consumer ancestor operation has no block ID attribute.\n");
            }
            return { pPrevId, cPrevId };
        }

        // Rolling: continue upward
        before = current;
        current = current->getParentOp();
    }
    LOG_DEBUG("Warning: No common ancestor found for producer block " << producerBlockId << " and consumer block " <<
        consumerBlockId << "\n");
    return { -1, -1 };
}

void DataDependencyAnalysisPass::runOnOperation()
{
    LOG_DEBUG("\n--- enter DataDependencyAnalysisPass --->\n");
    module = getOperation();

    auto &info = getAnalysis<DataDependencyInfo>();

    // Step 1: Collect block information (populate blockInfoMap)
    createBlockInfoMap(info);

    // Step 2: Analyze dependencies (populate v2c, c2v lists)
    analyzeExternalInputs(info);
    analyzeExternalOutputs(info);

    // Step 3: Analyze memory dependencies (PIPE_S sync)
    analyzeMemoryEffect(info);

    info.setValid(true);

    LOG_DEBUG("DataDependencyAnalysisPass: Analysis complete.\n");
    LOG_DEBUG("  V->C dependencies: " << info.getV2CDependencies().size() << "\n");
    LOG_DEBUG("  C->V dependencies: " << info.getC2VDependencies().size() << "\n");
    LOG_DEBUG("  Memory dependencies: " << info.getMemoryDependencies().size() << "\n");

    LOG_DEBUG("\n--- exit DataDependencyAnalysisPass --->\n");
}

// Create the pass
namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createDataDependencyAnalysisPass()
{
    return std::make_unique<DataDependencyAnalysisPass>();
}

// Helper: Get BlockId
int getSsbufferBlockId(Operation *op)
{
    if (auto attr = op->getAttrOfType<IntegerAttr>(kBlockIdAttr)) {
        return attr.getInt();
    }
    return -1;
}
} // namespace triton
} // namespace mlir