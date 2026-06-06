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

#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/OpClassifier.h"

#include "llvm/ADT/DenseSet.h"

#include <queue>

using namespace mlir;
static constexpr const char *DEBUG_TYPE = "op-classifier";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")
using namespace mlir::triton;

namespace {

bool isInsideNestedLinalgRegion(Operation *op)
{
    for (Operation *parent = op ? op->getParentOp() : nullptr; parent; parent = parent->getParentOp()) {
        if (isa<linalg::LinalgDialect>(parent->getDialect())) {
            return true;
        }
    }
    return false;
}

} // namespace

// Helper: describe operation for logging
std::string OpClassifierPass::describeOp(Operation *op) const
{
    if (!op)
        return "<null-op>";
    std::string result;
    llvm::raw_string_ostream os(result);
    os << op->getName().getStringRef();
    return result;
}

// Helper: convert OpCoreType to string for IR attribute
std::string coreTypeToString(OpCoreType ct)
{
    switch (ct) {
        case OP_CUBE_ONLY:
            return "CUBE";
        case OP_VECTOR_ONLY:
            return "VECTOR";
        case OP_CUBE_AND_VECTOR:
            return "CUBE_AND_VECTOR";
        default:
            return "UNDETERMINED";
    }
}

namespace {

// Maximum number of core types in a comma-separated core type string (e.g., "CUBE,VECTOR")
constexpr size_t kMaxCoreTypeParts = 4;

// Minimum number of inputs for linalg.matmul (A and B matrices)
constexpr size_t kMinMatmulInputs = 2;

} // namespace

// Helper: parse OpCoreType from string (handles both "CUBE" and "CUBE,VECTOR" formats)
// For comma-separated multi-value strings, returns the i-th component if index is provided,
// otherwise returns the first component.
OpCoreType parseCoreTypeFromString(const std::string &coreTypeStr, size_t index = 0)
{
    llvm::StringRef ref(coreTypeStr);
    llvm::SmallVector<llvm::StringRef, kMaxCoreTypeParts> parts;
    ref.split(parts, ',');
    if (parts.size() > 1 && index < parts.size()) {
        // Multi-value: pick i-th component
        return parseCoreTypeFromString(parts[index].str());
    }
    StringRef singleValue = parts[0].trim();
    if (singleValue == "CUBE") {
        return OP_CUBE_ONLY;
    } else if (singleValue == "VECTOR") {
        return OP_VECTOR_ONLY;
    } else if (singleValue == "CUBE_AND_VECTOR") {
        return OP_CUBE_AND_VECTOR;
    }
    return OP_UNDETERMINED;
}

// Initialize the pass
void OpClassifierPass::initializePass(ModuleOp module)
{
    // Reset data structures
    opCoreTypes.clear();
    allOps.clear();
    cubeSeeds.clear();

    // Collect all operations
    module.walk([&](Operation *op) {
        if (isa<ModuleOp, func::FuncOp, func::ReturnOp>(op)) {
            return; // Skip module, function, and return ops
        }
        allOps.push_back(op);
    });

    LLVM_DEBUG(DBGS() << "Collected: " << allOps.size() << " ops\n");

    // Initialize all ops to UNDETERMINED
    for (Operation *op : allOps) {
        opCoreTypes[op] = OP_UNDETERMINED;
    }
}

// Mark an operation as CUBE
void OpClassifierPass::markCube(Operation *op)
{
    if (op && opCoreTypes.count(op)) {
        opCoreTypes[op] = static_cast<OpCoreType>(opCoreTypes[op] | OP_CUBE_ONLY);
        LLVM_DEBUG(DBGS() << "CUBE: " << *op << "\n");
    }
}

// ============================================================================
// Pattern: to_tensor → matmul (Upstream)
// ============================================================================
// Matches cases where matmul's input comes from bufferization.to_tensor.
// to_tensor converts a memref to a tensor, typically used for loading matrix data
// from memory.
// IR Example
//   %memref = memref.alloc() : memref<1024x1024xf32>
//   %tensor = bufferization.to_tensor %memref : memref<1024x1024xf32> -> tensor<1024x1024xf32>
//   %result = linalg.matmul ins(%tensor, %tensor_b) outs(%init)
// Matching Logic
//   1. Check if matmul's operand defining op is bufferization.to_tensor
//   2. If matched, mark to_tensor as CUBE and add to cubeSeeds
//   3. Also trace back to_tensor's memref source and mark its defining op as CUBE
// Purpose: Ensure matmul's input data loading executes on CUBE for efficient matrix
// data transfer.
// ============================================================================
void OpClassifierPass::matchToTensorPattern(Operation *def)
{
    auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(def);
    if (!toTensorOp)
        return;

    markCube(toTensorOp);
    cubeSeeds.push_back(toTensorOp);

    // Also mark the memref allocation as CUBE
    Value memref = toTensorOp.getBuffer();
    if (Operation *memrefDef = memref.getDefiningOp()) {
        markCube(memrefDef);
        cubeSeeds.push_back(memrefDef);
    }
}

// ============================================================================
// Pattern: transpose → matmul (Upstream)
// ============================================================================
// Matches cases where matmul's input comes from linalg.transpose.
// transpose performs matrix transposition, commonly used as preprocessing for
// matrix multiplication inputs.
// IR Example
//   %input = tensor.empty() : tensor<1024x1024xf32>
//   %transposed = linalg.transpose ins(%input: tensor<1024x1024xf32>) outs(%out: tensor<1024x1024xf32>)
//                   permutation = [1, 0]
//   %result = linalg.matmul ins(%transposed, %tensor_b) outs(%init)
// Matching Logic
//   1. Check if matmul's operand defining op is linalg.transpose
//   2. Mark transpose itself as CUBE (transpose operation is suitable for CUBE execution)
//   3. Only mark transpose's input as CUBE seed when its operands come from
//      - bufferization dialect ops (e.g., to_tensor, clone), excluding AllocTensorOp
//      - tensor.empty op
//   4. Check both transpose's operands and DpsInits (output initial values), either one
//      satisfying the condition is sufficient
// Special Handling
//   When transpose's input comes from compute ops (e.g., arith.truncf), transpose itself
//   remains CUBE, but its upstream chain stays VECTOR to avoid unnecessary CUBE expansion.
// Purpose: transpose is a common preprocessing op for matmul; executing on CUBE leverages
// matrix transfer capabilities.
// ============================================================================
void OpClassifierPass::matchTransposePattern(Operation *def)
{
    auto transposeOp = dyn_cast<linalg::TransposeOp>(def);
    if (!transposeOp)
        return;

    markCube(transposeOp);

    // Helper lambda to check if an operand's defining op qualifies for CUBE seed
    auto shouldMarkCubeSeed = [](Operation *opDef) -> bool {
        if (!opDef)
            return false;
        return (isa<bufferization::BufferizationDialect>(opDef->getDialect()) &&
                !isa<bufferization::AllocTensorOp>(opDef)) ||
               isa<tensor::EmptyOp>(opDef);
    };

    // Check input tensor
    auto operands = transposeOp->getOperands();
    for (const auto &op : operands) {
        if (shouldMarkCubeSeed(op.getDefiningOp())) {
            markCube(op.getDefiningOp());
            cubeSeeds.push_back(op.getDefiningOp());
            break; // No need to check other operands, one is enough to seed the transpose as CUBE
        }
    }

    // Check outs (DpsInits)
    auto outs = transposeOp.getDpsInits();
    for (const auto &out : outs) {
        if (shouldMarkCubeSeed(out.getDefiningOp())) {
            markCube(out.getDefiningOp());
            cubeSeeds.push_back(out.getDefiningOp());
            break;
        }
    }
}

// ============================================================================
// Pattern: linalg.fill → matmul (Upstream)
// ============================================================================
// Matches cases where matmul's output initial value comes from linalg.fill.
// fill initializes the output matrix (typically to 0).
// IR Example
//   %value = arith.constant 0.0 : f32
//   %out = tensor.empty() : tensor<1024x1024xf32>
//   %init = linalg.fill ins(%value: f32) outs(%out: tensor<1024x1024xf32>)
//   %result = linalg.matmul ins(%a, %b) outs(%init)
// Matching Logic
//   1. Check if matmul's operand defining op is linalg.fill
//   2. If matched, mark fill as CUBE and add to cubeSeeds
// Purpose: fill operation initializes matmul's output buffer; executing on CUBE leverages
// efficient data filling capabilities.
// ============================================================================
void OpClassifierPass::matchFillPattern(Operation *def)
{
    auto fillOp = dyn_cast<linalg::FillOp>(def);
    if (!fillOp)
        return;

    markCube(fillOp);
    cubeSeeds.push_back(fillOp);
}

// ============================================================================
// Pattern: matmul → hivm.hir.store (Downstream)
// ============================================================================
// Matches cases where matmul's output is directly stored to memory.
// hivm.store is the HIVM dialect's store operation, writing tensor to memory.
// IR Example
//   %result = linalg.matmul ins(%a, %b) outs(%init)
//   hivm.store %result, %memref[%offset] : tensor<1024x1024xf32> to memref
// Matching Logic
//   1. Check if matmul result's user is hivm.store
//   2. If matched, mark store as CUBE and add to cubeSeeds
// Purpose: Ensure matmul result's store operation executes on CUBE for efficient
// matrix data write-back.
// ============================================================================
void OpClassifierPass::matchStorePattern(Operation *user)
{
    if (!isa<hivm::StoreOp>(user))
        return;

    markCube(user);
    cubeSeeds.push_back(user);
}

// ============================================================================
// Pattern: matmul → tensor.extract_slice → hivm.store (Downstream)
// ============================================================================
// Matches cases where matmul's output is sliced before being stored.
// extract_slice extracts a sub-region (slice) of a tensor, commonly used after
// tiled matrix computation for result extraction.
// IR Example
//   %result = linalg.matmul ins(%a, %b) outs(%init) : tensor<1024x1024xf32>
//   %slice = tensor.extract_slice %result[0, 0][256, 256][1, 1] : tensor<1024x1024xf32> to tensor<256x256xf32>
//   hivm.store %slice, %memref[%offset] : tensor<256x256xf32> to memref
// Matching Logic
//   1. Check if matmul result's user is tensor.extract_slice
//   2. If matched, mark extract_slice as CUBE and add to cubeSeeds
//   3. Further check all users of extract_slice and mark hivm.store as CUBE
// Purpose: After tiled matrix computation, slice and store operations on CUBE improve
// local data transfer efficiency.
// ============================================================================
void OpClassifierPass::matchExtractSlicePattern(Operation *user)
{
    auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    if (!extractSliceOp)
        return;

    markCube(extractSliceOp);
    cubeSeeds.push_back(extractSliceOp);
    // Also mark downstream hivm.hir.store as CUBE
    for (Operation *sliceUser : extractSliceOp->getUsers()) {
        if (isa<hivm::StoreOp>(sliceUser)) {
            markCube(sliceUser);
            cubeSeeds.push_back(sliceUser);
        }
    }
}

// ============================================================================
// Pattern: matmul → materialize_in_destination (Downstream)
// ============================================================================
// Matches cases where matmul's output is written to a destination buffer via
// materialize_in_destination. This operation materializes tensor results into a
// pre-allocated destination location.
// IR Example
//   %result = linalg.matmul ins(%a, %b) outs(%init) : tensor<1024x1024xf32>
//   %memref = memref.alloc() : memref<1024x1024xf32>
//   bufferization.materialize_in_destination %result in %memref
//     : tensor<1024x1024xf32> to memref<1024x1024xf32>
// Matching Logic
//   1. Check if matmul result's user is bufferization.materialize_in_destination
//   2. If matched, mark it as CUBE and add to cubeSeeds
// Purpose: matmul result materialization on CUBE leverages efficient matrix data
// transfer capabilities.
// ============================================================================
void OpClassifierPass::matchMaterializePattern(Operation *user)
{
    if (!isa<bufferization::MaterializeInDestinationOp>(user))
        return;

    markCube(user);
    cubeSeeds.push_back(user);
}

// Pattern matching for CUBE operations
int OpClassifierPass::patternMatchCUBE()
{
    LDBG("--- Step 1: pattern match --->\n");

    for (Operation *op : allOps) {
        if (!isa<linalg::MatmulOp>(op))
            continue;

        // matmul always CUBE
        opCoreTypes[op] = OP_CUBE_ONLY;
        LLVM_DEBUG(DBGS() << "CUBE (matmul): " << *op << "\n");

        // ---- Upstream pattern matching ----
        for (Value operand : op->getOperands()) {
            Operation *def = operand.getDefiningOp();
            if (!def)
                continue;

            matchToTensorPattern(def);
            matchTransposePattern(def);
            matchFillPattern(def);
        }

        // ---- Downstream pattern matching ----
        for (Value result : op->getResults()) {
            for (Operation *user : result.getUsers()) {
                matchStorePattern(user);
                matchExtractSlicePattern(user);
                matchMaterializePattern(user);
            }
        }
    }

    LLVM_DEBUG(DBGS() << "seeds: " << cubeSeeds.size() << " to_tensor(s)\n");
    for (Operation *seed : cubeSeeds) {
        LLVM_DEBUG(DBGS() << "CUBE: " << *seed << "\n");
    }

    return 0;
}

// Helper function to get upstream operations based on both SSA and memory dependencies
void OpClassifierPass::getUpstreamOpsWithMemoryDeps(Operation *cur, llvm::SmallVectorImpl<Operation *> &upstreamOps)
{
    // Collect SSA dependencies (direct operands)
    for (Value operand : cur->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (def && isa<memref::CopyOp, memref::AllocOp>(def) && opCoreTypes[def] == OP_CUBE_ONLY) {
            continue;
        }
        if (def && def != cur) {
            LLVM_DEBUG(DBGS() << "push op: " << *def << "\n");
            upstreamOps.push_back(def);
        }
    }
    if (!isa<bufferization::ToTensorOp>(cur)) {
        // Only consider memory dependencies for non-to_tensor ops,
        // as to_tensor already has SSA deps to its memref source
        return;
    }
    // Collect memory dependencies
    if (memDepGraph) {
        // Get operations that define memory used by current op
        for (Operation *memDef : memDepGraph->getMemDefs(cur)) {
            LLVM_DEBUG(DBGS() << "memDef: cur " << *cur << " -> memDef " << *memDef << "\n");
            if (isa<memref::CopyOp>(memDef) && opCoreTypes[memDef] == OP_UNDETERMINED) {
                LLVM_DEBUG(DBGS() << "push op: " << *memDef << "\n");
                upstreamOps.push_back(memDef);
            }
        }
    }
}

// Propagate CUBE core type upstream
int OpClassifierPass::propagateCubeUpstream()
{
    LLVM_DEBUG(DBGS() << "--- Step 2: CUBE upstream BFS --->\n");
    llvm::DenseSet<Operation *> cubeVisited;
    std::queue<Operation *> cubeQueue;

    for (Operation *seed : cubeSeeds) {
        cubeVisited.insert(seed);
        cubeQueue.push(seed);
    }

    while (!cubeQueue.empty()) {
        Operation *cur = cubeQueue.front();
        cubeQueue.pop();
        LLVM_DEBUG(DBGS() << "cur: " << *cur << "\n");

        // Get upstream operations considering both SSA and memory dependencies
        llvm::SmallVector<Operation *> upstreamOps;
        getUpstreamOpsWithMemoryDeps(cur, upstreamOps);

        for (Operation *def : upstreamOps) {
            if (!def || cubeVisited.count(def) || isa<linalg::MatmulOp>(def))
                continue;

            // Skip operations inside linalg block (internal values)
            // But don't skip the linalg op itself
            if (isInsideNestedLinalgRegion(def)) {
                LLVM_DEBUG(DBGS() << "skip " << def->getName().getStringRef() << ": inside linalg block\n");
                continue;
            }

            cubeVisited.insert(def);

            LLVM_DEBUG(DBGS() << "\tcolor-cube: " << def->getName().getStringRef() << "\n");
            opCoreTypes[def] = OP_CUBE_ONLY;
            cubeQueue.push(def);
        }
    }

    // Handle fill ops: mark as CUBE only if linalg.fill output buffer is CUBE
    markFillOpsAsCube();

    return 0;
}

// ============================================================================
// Step 3: Mark remaining operations as VECTOR
// ============================================================================
// After Step 2 (CUBE propagation), any operation still marked as UNDETERMINED
// is classified as VECTOR. This establishes the default that non-CUBE operations
// should execute on the VECTOR core.
// ============================================================================
int OpClassifierPass::markRemainingAsVector()
{
    LLVM_DEBUG(DBGS() << "--- Step 3: default VECTOR --->\n");
    for (Operation *op : allOps) {
        if (isa<annotation::MarkOp>(op)) {
            auto markOp = cast<annotation::MarkOp>(op);
            Value src = markOp.getSrc();
            if (Operation *srcDef = src.getDefiningOp()) {
                opCoreTypes[op] = getCoreType(srcDef);
            } else {
                opCoreTypes[op] = OP_VECTOR_ONLY;
            }
            continue;
        }

        if (opCoreTypes[op] == OP_UNDETERMINED && !isa<scf::YieldOp>(op)) {
            opCoreTypes[op] = OP_VECTOR_ONLY;
        }
    }
    return 0;
}

// ============================================================================
// Helper: Mark fill operations as CUBE when their output buffer is CUBE
// ============================================================================
// This function handles a special case where linalg.fill's result feeds into
// a CUBE operation. In such cases, the fill itself should be marked CUBE.
// Case 1: fill's outs operand is defined by a CUBE_ONLY operation (not tensor.empty)
//   IR example
//     %empty = tensor.empty() : tensor<1024x1024xf32>  // CUBE
//     %filled = linalg.fill ins(%value) outs(%empty)   // should be CUBE
//   Here %filled's outs comes from a CUBE op, so fill is CUBE
// Case 2: fill's outs is a BlockArgument (scf.for/scf.if iter_arg) that is CUBE
//   IR example
//     scf.for ... iter_args(%arg = %filled) {  // %arg is CUBE
//       ...
//     }
//   Here the iter_arg is CUBE, so fill is CUBE
// ============================================================================
void OpClassifierPass::markFillOpsAsCube()
{
    LLVM_DEBUG(DBGS() << "--- Marking fill ops as CUBE when outs is CUBE --->\n");

    for (Operation *op : allOps) {
        // Only process fill ops that are still UNDETERMINED
        if (!isa<linalg::FillOp>(op) || opCoreTypes[op] != OP_UNDETERMINED) {
            continue;
        }

        auto fillOp = cast<linalg::FillOp>(op);
        Value outs = fillOp.getDpsInits()[0];
        Operation *outsDef = outs.getDefiningOp();

        bool outsIsCube = false;

        // Case 1: outs is defined by a CUBE_ONLY operation (not tensor.empty)
        // tensor.empty is a special case - it doesn't determine CUBE/VECTOR
        if (outsDef && opCoreTypes[outsDef] == OP_CUBE_ONLY && !isa<tensor::EmptyOp>(outsDef)) {
            outsIsCube = true;
            LLVM_DEBUG(DBGS() << "\tfill outs defined by CUBE op: " << outsDef->getName().getStringRef() << "\n");
        } else if (!outsDef) { // Case 2: outs is a BlockArgument (scf.for/scf.if iter_arg)
            auto blockArg = dyn_cast<BlockArgument>(outs);
            if (blockArg) {
                Operation *parentOp = blockArg.getOwner()->getParentOp();
                // Check if it's an scf.for or scf.if iter_arg that is CUBE
                if ((isa<scf::ForOp>(parentOp) || isa<scf::IfOp>(parentOp)) && opCoreTypes[parentOp] == OP_CUBE_ONLY) {
                    outsIsCube = true;
                    LLVM_DEBUG(DBGS() << "\tfill outs is CUBE iter_arg of: " << parentOp->getName().getStringRef()
                                      << "\n");
                }
            }
        }

        if (outsIsCube) {
            LLVM_DEBUG(DBGS() << "\tfill-cube (outs is CUBE): " << *op << "\n");
            opCoreTypes[op] = OP_CUBE_ONLY;
        }
    }
}

// ============================================================================
// Step 4: Propagate VECTOR core type upstream
// ============================================================================
// After Step 3, propagate VECTOR classification to upstream operations.
// BFS traversal from all VECTOR operations, marking their upstream dependencies
// as VECTOR unless they are CUBE-only operations (e.g., matmul).
// Skips builtin.module and func.func operations.
// ============================================================================
int OpClassifierPass::propagateVectorUpstream()
{
    LLVM_DEBUG(DBGS() << "--- Step 4: VECTOR upstream BFS --->\n");
    llvm::DenseSet<Operation *> vecVisited;
    std::queue<Operation *> vecQueue;

    for (Operation *op : allOps) {
        if (opCoreTypes[op] == OP_VECTOR_ONLY) {
            vecVisited.insert(op);
            vecQueue.push(op);
        }
    }

    while (!vecQueue.empty()) {
        Operation *cur = vecQueue.front();
        vecQueue.pop();
        // Skip builtin.module, func.func, and func.return operations
        if (isa<ModuleOp, func::FuncOp, func::ReturnOp>(cur) ||
            (isa<memref::CopyOp, memref::AllocaOp>(cur) && opCoreTypes[cur] == OP_CUBE_ONLY)) {
            continue;
        }
        LLVM_DEBUG(DBGS() << "vec-def: " << *cur << "\n");

        // Get upstream operations considering both SSA and memory dependencies
        llvm::SmallVector<Operation *> upstreamOps;
        getUpstreamOpsWithMemoryDeps(cur, upstreamOps);

        for (Operation *def : upstreamOps) {
            if (!def || vecVisited.count(def))
                continue;

            // Skip operations that should not be marked VECTOR:
            // - matmul: never mark matmul as vector (CUBE-only operation)
            if (isa<linalg::MatmulOp>(def)) {
                LLVM_DEBUG(DBGS() << "skip " << def->getName().getStringRef() << ": should not be marked VECTOR\n");
                continue;
            }

            vecVisited.insert(def);

            LLVM_DEBUG(DBGS() << "\tcolor-vec: " << def->getName().getStringRef() << "\n");
            opCoreTypes[def] = static_cast<OpCoreType>(opCoreTypes[def] | OP_VECTOR_ONLY);
            vecQueue.push(def);
        }
    }

    return 0;
}

// ============================================================================
// Helper Functions
// ============================================================================

OpCoreType OpClassifierPass::getCoreType(Operation *op) const
{
    auto it = opCoreTypes.find(op);
    if (it != opCoreTypes.end()) {
        return it->second;
    }
    return OP_VECTOR_ONLY;
}

void OpClassifierPass::setCoreType(Operation *op, OpCoreType coreType)
{
    opCoreTypes[op] = coreType;
}

std::string OpClassifierPass::joinCoreTypes(const std::vector<OpCoreType> &coreTypes)
{
    std::string coreTypeStr;
    for (size_t i = 0; i < coreTypes.size(); ++i) {
        if (i > 0) {
            coreTypeStr += ", ";
        }
        coreTypeStr += coreTypeToString(coreTypes[i]);
    }
    return coreTypeStr;
}

// ============================================================================
// Step 5: SCF Yield Results Handling
// ============================================================================

bool OpClassifierPass::shouldStopPropagationForYield(Operation *op, OpCoreType targetCoreType)
{
    // Stop at linalg.matmul
    if (isa<linalg::MatmulOp>(op)) {
        return true;
    }

    // Stop at hivm.hir.store
    if (isa<hivm::StoreOp>(op)) {
        return true;
    }

    // Stop at tensor-type operations that are already VECTOR_ONLY when targeting CUBE
    if (targetCoreType == OP_CUBE_ONLY) {
        if (op->getNumResults() > 0) {
            for (Value result : op->getResults()) {
                if (isa<TensorType>(result.getType())) {
                    OpCoreType currentType = getCoreType(op);
                    if (currentType == OP_VECTOR_ONLY) {
                        return true;
                    }
                }
            }
        }
    }

    // Stop at scf dialect operations
    if (isa<scf::SCFDialect>(op->getDialect())) {
        return true;
    }

    // Stop at func.func or builtin.module
    if (isa<func::FuncOp, ModuleOp>(op)) {
        return true;
    }

    return false;
}

void OpClassifierPass::propagateCoreTypeUpwardForYield(Operation *startOp, OpCoreType targetCoreType)
{
    if (!startOp)
        return;

    llvm::DenseSet<Operation *> visited;
    std::queue<Operation *> queue;
    queue.push(startOp);
    visited.insert(startOp);

    while (!queue.empty()) {
        Operation *cur = queue.front();
        queue.pop();

        if (!cur || !cur->getBlock())
            continue;

        if (shouldStopPropagationForYield(cur, targetCoreType))
            continue;

        opCoreTypes[cur] = targetCoreType;

        // Propagate to operands (upstream)
        for (Value operand : cur->getOperands()) {
            if (Operation *defOp = operand.getDefiningOp()) {
                if (!visited.count(defOp) && defOp != cur) {
                    visited.insert(defOp);
                    queue.push(defOp);
                }
            }
        }

        // Also propagate to memory dependencies
        if (memDepGraph) {
            for (Operation *memDef : memDepGraph->getMemDefs(cur)) {
                if (!visited.count(memDef) && memDef != cur) {
                    visited.insert(memDef);
                    queue.push(memDef);
                }
            }
        }
    }
}

// handleYieldFromElseRegion: Extracts core_type from then region yield for else region processing.
// For scf.if with else region, when processing the else region's yield terminator, we need
// to use the then region's yield's core_type attribute as the authoritative source. If the
// then yield has a comma-separated multi-value type string (e.g., "CUBE,VECTOR"), this function
// extracts the i-th component for the operand at index i.
// Parameters
//   - coreTypes: Vector to append the extracted core type to
//   - operandIndex: Index of the operand in the yield (for extracting multi-value types)
//   - thenYieldForElse: The then region's yield operation (from scf.if)
//   - operand: The current operand value from the else region's yield
// Returns: true if core_type was successfully extracted and propagated, false otherwise.
bool OpClassifierPass::handleYieldFromElseRegion(std::vector<OpCoreType> &coreTypes, unsigned operandIndex,
                                                 Operation *thenYieldForElse, Value &operand)
{
    // Only handle if thenYieldForElse is provided and is a scf.yield
    if (!thenYieldForElse || !isa<scf::YieldOp>(thenYieldForElse)) {
        return false;
    }

    // Get the core_type attribute from then region yield
    auto thenCoreTypeAttr = thenYieldForElse->getAttr("ssbuffer.core_type");
    if (!thenCoreTypeAttr) {
        return false;
    }

    StringAttr thenCoreTypeStrAttr = dyn_cast<StringAttr>(thenCoreTypeAttr);
    if (!thenCoreTypeStrAttr) {
        return false;
    }

    // Extract the i-th component from comma-separated core type string
    std::string thenCoreTypeStr = thenCoreTypeStrAttr.getValue().str();
    OpCoreType coreTypeToUse = parseCoreTypeFromString(thenCoreTypeStr, operandIndex);
    coreTypes.push_back(coreTypeToUse);

    // Propagate the determined core_type upstream to the defining operation
    if (Operation *defOp = operand.getDefiningOp()) {
        propagateCoreTypeUpwardForYield(defOp, coreTypeToUse);
    }

    return true;
}

// ============================================================================
// processYieldOperation: Set core_type on scf.yield and its parent scf op
// ============================================================================
// For each operand of scf.yield:
//   - If thenYieldForElse is provided (else region of scf.if): use then yield's
//     per-operand core_type as reference to determine this operand's type, and
//     propagate that type upstream.
//   - Otherwise (then region of scf.if, or scf.for): look at the operand's
//     defining op's core_type. If it is OP_CUBE_ONLY -> "CUBE", else -> "VECTOR".
//     BlockArguments without a defining op (e.g., iter_arg block argument) default
//     to VECTOR.
// After collecting all operand types, set ssbuffer.core_type on both the yield op
// and its parent scf operation (scf.if or scf.for).
void OpClassifierPass::processYieldOperation(Operation *op, Operation *thenYieldForElse)
{
    std::vector<OpCoreType> coreTypes;

    // Get the parent SCF operation (scf.if or scf.for)
    Operation *parentOp = op->getParentOp();

    // Process each yield operand in order
    if (op->getNumOperands() > 0) {
        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
            Value operand = op->getOperand(i);
            // ---------------------------------------------------------------
            // Case 1: else region yield of scf.if
            // ---------------------------------------------------------------
            // Use then region yield's core_type attribute to determine type.
            // If then yield has a comma-separated multi-value type string (e.g.
            // "CUBE,VECTOR"), extract the i-th component for this operand.
            if (handleYieldFromElseRegion(coreTypes, i, thenYieldForElse, operand)) {
                continue;
            }

            // ---------------------------------------------------------------
            // Case 2: then region yield of scf.if, or scf.for yield
            // ---------------------------------------------------------------
            // Determine type from the operand's defining operation:
            //   - OP_CUBE_ONLY -> yield type is "CUBE"
            //   - anything else (OP_UNDETERMINED, OP_VECTOR_ONLY) -> "VECTOR"
            // If the operand is a BlockArgument without a defining op (e.g., an
            // iter_arg that is passed through without modification), default to
            // VECTOR since no compute op is associated with it.
            if (Operation *def = operand.getDefiningOp()) {
                OpCoreType ct = getCoreType(def);
                if (ct == OP_CUBE_ONLY) {
                    coreTypes.push_back(OP_CUBE_ONLY);
                } else {
                    coreTypes.push_back(OP_VECTOR_ONLY);
                }
            } else {
                // BlockArgument (e.g., iter_arg passed through scf.for) -> VECTOR
                coreTypes.push_back(OP_VECTOR_ONLY);
            }
        }
    }

    // Write ssbuffer.core_type attribute onto the yield op and its parent scf op
    if (!coreTypes.empty()) {
        std::string coreTypeStr = joinCoreTypes(coreTypes);
        op->setAttr("ssbuffer.core_type", StringAttr::get(op->getContext(), coreTypeStr));

        OpCoreType opCt = (coreTypeStr.find("CUBE") != std::string::npos) ? OP_CUBE_ONLY : OP_VECTOR_ONLY;
        opCoreTypes[op] = opCt;

        if (parentOp && llvm::isa<scf::SCFDialect>(parentOp->getDialect())) {
            parentOp->setAttr("ssbuffer.core_type", StringAttr::get(parentOp->getContext(), coreTypeStr));
            opCoreTypes[parentOp] = opCt;
        }
    }
}

// ============================================================================
// Step 5: Process SCF Yield Operations
// ============================================================================
// Entry point for yield processing. Walks all scf ops and sets core_type on
// their yield terminators and on the ops themselves.
// Processing order
//   1. All scf.if ops: process then-region yield first (no thenYieldForElse),
//      then process else-region yield (passing thenYieldForElse for type reference).
//   2. Remaining scf.yield ops not inside scf.if (e.g., scf.for terminators).
// After this step, every scf.yield and its parent scf.if/scf.for carry a
// ssbuffer.core_type attribute reflecting the core_type of their result
// values (derived from the defining ops of each yield operand).
int OpClassifierPass::handleSCFYield()
{
    // Collect all scf.if operations
    llvm::SmallVector<scf::IfOp> ifOps;
    for (Operation *op : allOps) {
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
            ifOps.push_back(ifOp);
        }
    }

    // Process scf.if operations: first then region, then else region
    for (scf::IfOp ifOp : ifOps) {
        Operation *thenYield = nullptr;
        if (!ifOp.getThenRegion().empty()) {
            thenYield = ifOp.getThenRegion().back().getTerminator();
        }

        if (isa<scf::YieldOp>(thenYield)) {
            processYieldOperation(thenYield, nullptr);
        }

        if (!ifOp.getElseRegion().empty()) {
            Operation *elseYield = ifOp.getElseRegion().back().getTerminator();
            if (isa<scf::YieldOp>(elseYield)) {
                processYieldOperation(elseYield, thenYield);
            }
        }
    }

    // Process remaining scf.yield operations (not inside scf.if, e.g. scf.for)
    for (Operation *op : allOps) {
        if (!isa<scf::YieldOp>(op))
            continue;

        Operation *parentOp = op->getParentOp();
        if (dyn_cast_or_null<scf::IfOp>(parentOp))
            continue;

        processYieldOperation(op, nullptr);
    }

    return 0;
}

// ============================================================================
// Step 6: CUBE_AND_VECTOR Operation Handling
// ============================================================================
// Problem: an operation (e.g., linalg.fill) is used by both CUBE users
// (linalg.matmul) and VECTOR users (arith.addf) simultaneously.
// Both cores cannot share the same op instance.
// Solution: clone the op into two versions.
//   - Original op  → reclassified as OP_CUBE_ONLY  → CUBE users keep using it
//   - Cloned op    → classified as OP_VECTOR_ONLY → VECTOR users switched to it
// Example
//       shared_fill (CUBE_AND_VECTOR)
//          ↑              ↑
//      matmul(CUBE)   arith.addf(VECTOR)
//       shared_fill (CUBE_ONLY)    → matmul (CUBE)
//       shared_fill_vector_clone   → arith.addf (VECTOR)
//         (VECTOR_ONLY)
// This step runs after step 4 (VECTOR propagation) and before step 6
// (SCF yield processing). The do-while loop handles cases where splitting
// an op produces new CUBE_AND_VECTOR ops among its operands.
// ============================================================================

void OpClassifierPass::splitOperationForCubeAndVector(Operation *op, llvm::DenseSet<Operation *> &processedOps,
                                                      llvm::DenseMap<Operation *, Operation *> &opToVectorClone)
{
    // Guard against double-processing the same op
    if (processedOps.count(op))
        return;
    processedOps.insert(op);

    // Only split ops explicitly marked CUBE_AND_VECTOR
    if (opCoreTypes[op] != OP_CUBE_AND_VECTOR)
        return;

    // ------------------------------------------------------------------
    // Phase 1: Recursively handle CUBE_AND_VECTOR operands first
    // ------------------------------------------------------------------
    // An operand of this op might itself be CUBE_AND_VECTOR (shared upstream).
    // We must split the operand's chain before we can split this op, because
    // when we build the operand mapping below, we need the VECTOR clone of
    // the operand to be available. Process operands depth-first.
    for (Value operand : op->getOperands()) {
        if (Operation *def = operand.getDefiningOp()) {
            if (opCoreTypes[def] == OP_CUBE_AND_VECTOR) {
                splitOperationForCubeAndVector(def, processedOps, opToVectorClone);
            }
        }
    }

    // ------------------------------------------------------------------
    // Phase 2: Build IRMapping — redirect operands to VECTOR clones
    // ------------------------------------------------------------------
    // When cloning this op for VECTOR users, any operand that already has
    // a VECTOR clone (produced by a recursive call above) must be replaced
    // with that clone so the cloned op uses the VECTOR-side values.
    IRMapping mapping;
    for (Value operand : op->getOperands()) {
        if (Operation *def = operand.getDefiningOp()) {
            auto it = opToVectorClone.find(def);
            if (it != opToVectorClone.end()) {
                // Operand has a VECTOR clone — use it in the cloned op
                Operation *vectorClone = it->second;
                for (unsigned i = 0; i < def->getNumResults(); ++i) {
                    if (def->getResult(i) == operand) {
                        mapping.map(operand, vectorClone->getResult(i));
                        break;
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Phase 3: Clone the op for VECTOR users
    // ------------------------------------------------------------------
    OpBuilder builder(op);
    Operation *vectorOp = builder.clone(*op, mapping);

    // Reclassify: original → CUBE_ONLY, clone → VECTOR_ONLY
    opCoreTypes[op] = OP_CUBE_ONLY;
    opCoreTypes[vectorOp] = OP_VECTOR_ONLY;
    opToVectorClone[op] = vectorOp; // record for callers' operand mapping
    allOps.push_back(vectorOp);     // track new op so it gets core_type stamped

    // ------------------------------------------------------------------
    // Phase 4: Redirect VECTOR-only users to the cloned result
    // ------------------------------------------------------------------
    // VECTOR users of the original result must now use the cloned result.
    // CUBE users still use the original. We iterate through all result uses
    // and switch VECTOR-only users over to vectorOp's corresponding result.
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
        Value originalResult = op->getResult(i);
        Value vectorResult = vectorOp->getResult(i);

        llvm::SmallVector<OpOperand *> usesToUpdate;
        for (OpOperand &use : originalResult.getUses()) {
            Operation *user = use.getOwner();
            if (user && user != vectorOp && getCoreType(user) == OP_VECTOR_ONLY) {
                usesToUpdate.push_back(&use);
            }
        }

        for (OpOperand *use : usesToUpdate) {
            use->set(vectorResult); // redirect to VECTOR clone
        }
    }
}

// ============================================================================
// handleCubeAndVector: entry point for CUBE_AND_VECTOR splitting
// ============================================================================
// Repeatedly scans for CUBE_AND_VECTOR ops and splits them until no more
// remain. The do-while loop is needed because splitOperationForCubeAndVector
// is called recursively on operands; each recursive call may turn a
// CUBE_AND_VECTOR operand into CUBE_ONLY + VECTOR_ONLY (via the same
// splitOperationForCubeAndVector), which can in turn enable more splits at
// the current level.

int OpClassifierPass::handleCubeAndVector()
{
    bool changed = true;
    do {
        changed = false;

        // Collect all ops still marked CUBE_AND_VECTOR
        llvm::SmallVector<Operation *> cubeAndVectorOps;
        for (Operation *op : allOps) {
            if (opCoreTypes[op] == OP_CUBE_AND_VECTOR) {
                cubeAndVectorOps.push_back(op);
            }
        }

        if (cubeAndVectorOps.empty())
            break;

        llvm::DenseMap<Operation *, Operation *> opToVectorClone;
        llvm::DenseSet<Operation *> processedOps;

        for (Operation *op : cubeAndVectorOps) {
            if (processedOps.count(op))
                continue;
            splitOperationForCubeAndVector(op, processedOps, opToVectorClone);
            changed = true;
        }
    } while (changed);

    return 0;
}

// ============================================================================
// Step 7: Stamp Core Type to IR
// ============================================================================

int OpClassifierPass::stampToIR()
{
    for (Operation *op : allOps) {
        if (!op || !op->getBlock())
            continue;

        auto it = opCoreTypes.find(op);
        if (it == opCoreTypes.end())
            continue;
        OpCoreType coreType = it->second;

        // Skip scf dialect operations
        if (llvm::isa<scf::SCFDialect>(op->getDialect()))
            continue;

        // Skip linalg operations' internal block operations
        Operation *parent = op->getParentOp();
        bool isInsideLinalgBlock = false;
        while (parent) {
            if (llvm::isa<linalg::LinalgDialect>(parent->getDialect())) {
                isInsideLinalgBlock = true;
                break;
            }
            parent = parent->getParentOp();
        }
        if (isInsideLinalgBlock)
            continue;

        // Skip builtin.module, func.func, and tt.func operations
        if (isa<ModuleOp, func::FuncOp>(op))
            continue;

        op->setAttr("ssbuffer.core_type", StringAttr::get(op->getContext(), coreTypeToString(coreType)));
    }

    return 0;
}

// ============================================================================
// Pre-legalize matmul: Replace matmul with zero-filled accumulator + add
// ============================================================================
//   matmul(A, B, C) -> matmul(A, B, zero) + add(result, C)
//   - zero = linalg.fill(tensor.empty(const 0), const 0)
//   - add(result, C) adds the original outs value to the matmul result
// This effectively changes a*b+c to result=a*b+0, add(result, c)
// This must be done before initializePass so that the classification pass
// sees the modified IR structure.
// ============================================================================

// Helper: bulk delete operations and clean up tracking structures
void OpClassifierPass::bulkDeleteOps(llvm::SmallVectorImpl<Operation *> &opsToDelete)
{
    llvm::DenseSet<Operation *> deletedOps(opsToDelete.begin(), opsToDelete.end());
    for (Operation *op : opsToDelete) {
        if (!op || !op->getBlock()) {
            continue;
        }
        if (op->use_empty()) {
            auto it = std::find(allOps.begin(), allOps.end(), op);
            if (it != allOps.end()) {
                allOps.erase(it);
            }
            opCoreTypes.erase(op);
            op->erase();
        }
    }

    // Remove any stale references from the classifier map
    for (Operation *op : deletedOps) {
        opCoreTypes.erase(op);
    }
}

int OpClassifierPass::preLegalizeMatmul()
{
    LLVM_DEBUG(DBGS() << "--- Pre-legalizing matmul operations --->\n");

    // Collect all linalg::MatmulOp from the module
    llvm::SmallVector<linalg::MatmulOp> matmulOps;
    getOperation().walk([&](linalg::MatmulOp matmulOp) { matmulOps.push_back(matmulOp); });

    if (matmulOps.empty()) {
        LLVM_DEBUG(DBGS() << "\tNo matmul operations found\n");
        return 0;
    }

    LLVM_DEBUG(DBGS() << "\tFound " << matmulOps.size() << " matmul operations\n");

    // deferredDelete pattern: collect ops to delete, erase after iteration completes
    // avoids iterator invalidation when erasing during the loop
    llvm::SmallVector<Operation *> opsToDelete;

    for (linalg::MatmulOp matmulOp : matmulOps) {
        if (!matmulOp || !matmulOp->getBlock()) {
            continue;
        }

        if (matmulOp.getNumResults() == 0) {
            continue;
        }

        auto outputs = matmulOp.getDpsInits();
        if (outputs.empty()) {
            continue;
        }
        Value outsValue = outputs[0];
        if (!outsValue) {
            continue;
        }

        Value matmulResult = matmulOp.getResult(0);

        // Get element type from result tensor
        auto rankedTensorType = dyn_cast<RankedTensorType>(matmulResult.getType());
        if (!rankedTensorType) {
            LLVM_DEBUG(DBGS() << "matmul result is not a ranked tensor, skipping\n");
            continue;
        }
        Type elemType = rankedTensorType.getElementType();

        mlir::OpBuilder builder(matmulOp);
        Location loc = matmulOp.getLoc();

        // [Step 1] Create tensor.empty for the new accumulator tensor
        // Same shape and type as original matmul output
        SmallVector<Value> dynamicSizes;
        for (int64_t i = 0; i < rankedTensorType.getRank(); ++i) {
            if (rankedTensorType.isDynamicDim(i)) {
                dynamicSizes.push_back(builder.create<tensor::DimOp>(loc, outsValue, i));
            }
        }
        auto emptyOp = builder.create<tensor::EmptyOp>(loc, rankedTensorType.getShape(),
                                                       rankedTensorType.getElementType(), dynamicSizes);

        // [Step 2] Create zero constant based on element type
        // Supports both floating-point (arith.constant float) and integer types
        Value zeroValue;
        if (auto floatType = dyn_cast<FloatType>(elemType)) {
            APFloat zeroAPFloat = APFloat::getZero(floatType.getFloatSemantics());
            zeroValue = builder.create<arith::ConstantFloatOp>(loc, floatType, zeroAPFloat).getResult();
        } else if (auto intType = dyn_cast<IntegerType>(elemType)) {
            zeroValue = builder.create<arith::ConstantIntOp>(loc, intType, 0).getResult();
        } else {
            LLVM_DEBUG(DBGS() << "matmul element type is not float or int, skipping\n");
            continue;
        }

        // [Step 3] Use linalg.fill to populate empty tensor with zero -> zero accumulator
        auto fillOp = builder.create<linalg::FillOp>(loc, ValueRange {zeroValue}, ValueRange {emptyOp.getResult()});

        // [Step 4] Get matmul's two input matrices A and B
        auto inputs = matmulOp.getDpsInputs();
        if (inputs.size() < kMinMatmulInputs) {
            LLVM_DEBUG(DBGS() << "matmul has insufficient inputs, skipping\n");
            continue;
        }
        Value a = inputs[0];
        Value b = inputs[1];

        // [Step 5] Create new matmul using zero-filled tensor as accumulator
        // New matmul runs entirely on CUBE with no VECTOR dependency
        auto newMatmul = builder.create<linalg::MatmulOp>(loc, ValueRange {a, b}, ValueRange {fillOp.getResult(0)});

        // Copy attributes from original matmul to new matmul (skipping internal attrs)
        for (auto attr : matmulOp->getAttrs()) {
            StringRef attrName = attr.getName().getValue();
            if (attrName == "operandSegmentSizes" || attrName == "res_attrs" || attrName == "arg_attrs") {
                continue;
            }
            newMatmul->setAttr(attr.getName(), attr.getValue());
        }

        // [Step 6] Create add: add(new_matmul_result, outs_value)
        // This is the "c" in a*b+c, added after the matmul result
        Operation *addOp;
        if (isa<FloatType>(elemType)) {
            addOp = builder.create<arith::AddFOp>(loc, newMatmul.getResult(0), outsValue).getOperation();
        } else {
            addOp = builder.create<arith::AddIOp>(loc, newMatmul.getResult(0), outsValue).getOperation();
        }

        // Move addOp to right after newMatmul
        addOp->moveAfter(newMatmul.getOperation());

        // [Step 7] Get add result and replace uses of original matmul result
        Value addResult = addOp->getResult(0);

        // Replace all uses of original matmulResult with addResult, EXCEPT addOp's own operand
        for (auto &use : llvm::make_early_inc_range(matmulResult.getUses())) {
            if (use.getOwner() == addOp) {
                continue;
            }
            use.set(addResult);
        }

        // [Step 8] Mark new operations as CUBE_ONLY
        opCoreTypes[newMatmul.getOperation()] = OP_CUBE_ONLY;
        opCoreTypes[emptyOp.getOperation()] = OP_CUBE_ONLY;
        opCoreTypes[zeroValue.getDefiningOp()] = OP_CUBE_ONLY;
        opCoreTypes[fillOp.getOperation()] = OP_CUBE_ONLY;
        opCoreTypes[addOp] = OP_CUBE_ONLY;

        // [Step 9] Mark original matmul for deferred deletion
        opsToDelete.push_back(matmulOp.getOperation());

        LLVM_DEBUG(DBGS() << "[OpClassifier] Transformed matmul to zero-filled accumulator + add\n");
    }

    // Step 10: Delete all marked original matmuls in bulk
    bulkDeleteOps(opsToDelete);

    LLVM_DEBUG(DBGS() << "--- Pre-legalize matmul complete --->\n");
    return 0;
}

// Run the pass
void OpClassifierPass::runOnOperation()
{
    ModuleOp module = getOperation();

    LLVM_DEBUG(DBGS() << "\n--- Before Plan Compute Block  --->\n");
    LLVM_DEBUG(DBGS() << module << "\n");

    LLVM_DEBUG(DBGS() << "\n--- Running OpClassifierPass --->\n");

    // Pre-legalize matmul: transform matmul to zero-filled accumulator + add BEFORE initializePass
    // This transforms: matmul(A, B, C) -> matmul(A, B, zero) + add(result, C)
    // where zero = linalg.fill(tensor.empty(const 0), const 0)
    // This effectively changes a*b+c to result=a*b+0, add(result, c)
    if (preLegalizeMatmul() != 0) {
        signalPassFailure();
        return;
    }

    // Initialize memory dependence graph for tracking memory side effects

    aliasAnalysis = std::make_shared<AliasAnalysis>(module);
    memDepGraph = std::make_shared<CVPipeline::MemoryDependenceGraph>(module, *aliasAnalysis);

    // Initialize the pass
    initializePass(module);

    // Step 1: Pattern match around each linalg.matmul to find CUBE seeds
    if (patternMatchCUBE() != 0) {
        signalPassFailure();
        return;
    }

    // Step 2: CUBE upstream BFS from seed loads
    if (propagateCubeUpstream() != 0) {
        signalPassFailure();
        return;
    }

    // Step 3: Mark remaining operations as VECTOR
    if (markRemainingAsVector() != 0) {
        signalPassFailure();
        return;
    }

    // Step 4: VECTOR upstream BFS
    if (propagateVectorUpstream() != 0) {
        signalPassFailure();
        return;
    }

    // Step 5: Handle CUBE_AND_VECTOR operations
    if (handleCubeAndVector() != 0) {
        signalPassFailure();
        return;
    }

    // Step 6: Process SCF yield results
    if (handleSCFYield() != 0) {
        signalPassFailure();
        return;
    }

    // Step 7: Stamp to IR
    if (stampToIR() != 0) {
        signalPassFailure();
        return;
    }

    LLVM_DEBUG(DBGS() << "\n--- OpClassifierPass: end --->\n");
}

// Create the pass
namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createOpClassifierPass()
{
    return std::make_unique<OpClassifierPass>();
}

void registerAddOpClassifierPasses()
{
    registerPass([]() -> std::unique_ptr<mlir::Pass> { return createOpClassifierPass(); });
}

} // namespace triton
} // namespace mlir
