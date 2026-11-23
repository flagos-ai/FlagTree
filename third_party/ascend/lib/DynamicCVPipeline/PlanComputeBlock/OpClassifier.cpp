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

#include <queue>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/OpClassifier.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/Utils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

using namespace mlir;
using namespace mlir::CVPipeline;
static constexpr const char *DEBUG_TYPE = "op-classifier";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)
using namespace mlir::triton;

namespace {

bool isInsideNestedLinalgRegion(Operation *op) {
  for (Operation *parent = op ? op->getParentOp() : nullptr; parent;
       parent = parent->getParentOp()) {
    if (isa<linalg::LinalgDialect>(parent->getDialect())) {
      return true;
    }
  }
  return false;
}

} // namespace

// Helper: describe operation for logging
std::string OpClassifierPass::describeOp(Operation *op) const {
  if (!op)
    return "<null-op>";
  std::string result;
  llvm::raw_string_ostream os(result);
  os << op->getName().getStringRef();
  return result;
}

// Helper: convert OpCoreType to string for IR attribute
std::string coreTypeToString(OpCoreType ct) {
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

// Maximum number of core types in a comma-separated core type string (e.g.,
// "CUBE,VECTOR")
constexpr size_t kMaxCoreTypeParts = 4;

// Minimum number of inputs for linalg.matmul (A and B matrices)
constexpr size_t kMinMatmulInputs = 2;

} // namespace

// Helper: parse OpCoreType from string (handles both "CUBE" and "CUBE,VECTOR"
// formats) For comma-separated multi-value strings, returns the i-th component
// if index is provided, otherwise returns the first component.
OpCoreType parseCoreTypeFromString(const std::string &coreTypeStr,
                                   size_t index = 0) {
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
void OpClassifierPass::initializePass(ModuleOp module) {
  // Reset data structures
  opCoreTypes.clear();
  allOps.clear();
  cubeSeeds.clear();
  vectorOnlyProducerCache.clear();
  inBroadcastChain.clear();
  CloneOpMap.clear();

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
void OpClassifierPass::markCube(Operation *op) {
  if (op && opCoreTypes.count(op)) {
    opCoreTypes[op] = static_cast<OpCoreType>(opCoreTypes[op] | OP_CUBE_ONLY);
    LLVM_DEBUG(DBGS() << "CUBE: " << *op << "\n");
  }
}

static bool isExtractedLoadStoreRelated(Operation *op) {
  if (!op)
    return false;
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case([](bufferization::ToTensorOp toTensorOp) {
        return isExtractedLoadStoreRelated(
            toTensorOp.getBuffer().getDefiningOp());
      })
      .Case([](memref::AllocOp allocOp) {
        Value memref = allocOp.getMemref();
        for (auto user : memref.getUsers()) {
          auto forOp = user->getParentOfType<scf::ForOp>();
          if (forOp && forOp->hasAttr(hivm::ExtractLoadStoreAttr))
            return true;
        }
        return false;
      })
      .Case([](ViewLikeOpInterface viewOp) {
        return isExtractedLoadStoreRelated(
            viewOp.getViewSource().getDefiningOp());
      })
      .Default([](auto) { return false; });
}

// ============================================================================
// Pattern: to_tensor → matmul (Upstream)
// ============================================================================
// Matches cases where matmul's input comes from bufferization.to_tensor.
// to_tensor converts a memref to a tensor, typically used for loading matrix
// data from memory. IR Example
//   %memref = memref.alloc() : memref<1024x1024xf32>
//   %tensor = bufferization.to_tensor %memref : memref<1024x1024xf32> ->
//   tensor<1024x1024xf32> %result = linalg.matmul ins(%tensor, %tensor_b)
//   outs(%init)
// Matching Logic
//   1. Check if matmul's operand defining op is bufferization.to_tensor
//   2. If matched, mark to_tensor as CUBE and add to cubeSeeds
//   3. Also trace back to_tensor's memref source and mark its defining op as
//   CUBE
// Purpose: Ensure matmul's input data loading executes on CUBE for efficient
// matrix data transfer.
// ============================================================================
void OpClassifierPass::matchToTensorPattern(Operation *def) {
  auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(def);
  constexpr llvm::StringLiteral kMayImplicitTransposeWithLastAxis =
      "MayImplicitTransposeWithLastAxis";

  if (!toTensorOp)
    return;

  // special case: implicit transpose -> remains vector
  if (utils::getAnnotateOpWithAttr(toTensorOp.getResult(),
                                   kMayImplicitTransposeWithLastAxis)) {
    return;
  }

  // special case: ExtractedLoadOrStore -> remains vector
  if (isExtractedLoadStoreRelated(toTensorOp)) {
    return;
  }

  markCube(toTensorOp);
  cubeSeeds.push_back(toTensorOp);

  Value memref = toTensorOp.getBuffer();
  // Also mark the memref allocation as CUBE
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
//   %transposed = linalg.transpose ins(%input: tensor<1024x1024xf32>)
//   outs(%out: tensor<1024x1024xf32>)
//                   permutation = [1, 0]
//   %result = linalg.matmul ins(%transposed, %tensor_b) outs(%init)
// Matching Logic
//   1. Check if matmul's operand defining op is linalg.transpose
//   2. Mark transpose itself as CUBE (transpose operation is suitable for CUBE
//   execution)
//   3. Only mark transpose's input as CUBE seed when its operands come from
//      - bufferization dialect ops (e.g., to_tensor, clone), excluding
//      AllocTensorOp
//      - tensor.empty op
//   4. Check both transpose's operands and DpsInits (output initial values),
//   either one
//      satisfying the condition is sufficient
// Special Handling
//   When transpose's input comes from compute ops (e.g., arith.truncf),
//   transpose itself remains CUBE, but its upstream chain stays VECTOR to avoid
//   unnecessary CUBE expansion.
// Purpose: transpose is a common preprocessing op for matmul; executing on CUBE
// leverages matrix transfer capabilities.
// ============================================================================
void OpClassifierPass::matchTransposePattern(Operation *def) {
  auto transposeOp = dyn_cast<linalg::TransposeOp>(def);
  if (!transposeOp)
    return;

  markCube(transposeOp);

  // Helper lambda to check if an operand's defining op qualifies for CUBE seed
  auto shouldMarkCubeSeed = [](Operation *opDef) -> bool {
    if (!opDef || isExtractedLoadStoreRelated(opDef))
      return false;
    return (isa<bufferization::BufferizationDialect>(opDef->getDialect()) &&
            !isa<bufferization::AllocTensorOp>(opDef)) ||
           isa<tensor::EmptyOp>(opDef);
  };

  // Check input tensor
  auto operands = transposeOp->getOperands();
  for (const auto &op : operands) {
    auto defOp = op.getDefiningOp();
    if (!shouldMarkCubeSeed(defOp) ||
        utils::getAnnotateOpWithAttr(op,
                                     hivm::kMayImplicitTransposeWithLastAxis)) {
      continue;
    }
    if (llvm::isa<bufferization::ToTensorOp>(defOp)) {
      matchToTensorPattern(defOp);
      continue;
    }
    markCube(defOp);
    cubeSeeds.push_back(defOp);
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
// Purpose: fill operation initializes matmul's output buffer; executing on CUBE
// leverages efficient data filling capabilities.
// ============================================================================
void OpClassifierPass::matchFillPattern(Operation *def) {
  auto fillOp = dyn_cast<linalg::FillOp>(def);
  if (!fillOp)
    return;

  markCube(fillOp);
  cubeSeeds.push_back(fillOp);
}

// ============================================================================
// Pattern: tensor.empty → matmul (Upstream)
// ============================================================================
// Matches cases where matmul's output initial value comes from tensor.empty.
// empty initializes the output matrix (typically to 0).
// IR Example
//   %value = arith.constant 0.0 : f32
//   %out = tensor.empty() : tensor<1024x1024xf32>
//   %result = linalg.matmul ins(%a, %b) outs(%out)
// Matching Logic
//   1. Check if matmul's operand defining op is tensor.empty
//   2. If matched, mark empty as CUBE and add to cubeSeeds
// Purpose: empty operation initializes matmul's output buffer;
// ============================================================================
void OpClassifierPass::matchEmptyPattern(Operation *def) {
  auto emptyOp = dyn_cast<tensor::EmptyOp>(def);
  if (!emptyOp)
    return;

  markCube(emptyOp);
  cubeSeeds.push_back(emptyOp);
}

// ============================================================================
// Pattern: tensor.broadcast → matmul (Upstream)
// ============================================================================
// Uses bias table buffer to  optimize A*B+C.
//   %value = arith.constant 0.0 : f32
//   %out = linalg.broadcast ins(%value: f32) outs(%out: tensor<1024x1024xf32>)
//   %result = linalg.matmul ins(%a, %b) outs(%out)
// Matching Logic
//   1. Check if matmul's operand defining op is tensor.broadcast
//   2. If matched, mark broadcast as CUBE and add to cubeSeeds
// Purpose: broadcast operation initializes matmul's output buffer;
// ============================================================================
Value OpClassifierPass::extractMmadBiasFromPotentialUnitDimExpand(Value bias) {
  // It assumes that there only exists expand op in mmad bias defining chain,
  // while other reshape op like collapse op seems unlikely
  if (auto expandShapeOp = bias.getDefiningOp<tensor::ExpandShapeOp>()) {
    auto reassociation = expandShapeOp.getReassociationIndices();
    auto expandedShape = expandShapeOp.getResultType().getShape();
    if (llvm::all_of(reassociation, [&expandedShape](ReassociationIndices cur) {
          uint32_t nonUnitCount =
              llvm::count_if(cur, [&expandedShape](int64_t idx) {
                return expandedShape[idx] != 1;
              });

          return nonUnitCount <= 1;
        })) {
      bias = expandShapeOp.getSrc();
      markCube(expandShapeOp);
      cubeSeeds.push_back(expandShapeOp);
      inBroadcastChain.insert(expandShapeOp);
    }
  }
  return bias;
}
void OpClassifierPass::matchBroadcastPattern(Operation *def) {
  auto broadcastOp = dyn_cast<linalg::BroadcastOp>(def);

  if (!broadcastOp)
    return;
  if (!CVPipeline::allResultHasOneUser(def)) {
    return;
  }
  // Only match small broadcast with correct dimensions (1D->2D, dimensions=[1])
  if (auto btUsage = CVPipeline::getBTSizeFromValidBroadcastOp(broadcastOp)) {
    if (btUsage == -1 || btUsage > CVPipeline::CACHE_TABLE_BUFFER_SIZE) {
      return;
    }
  }

  // check if the broadcast used by matmul's outs
  for (Operation *user : broadcastOp->getUsers()) {
    auto matmulOp = dyn_cast<linalg::MatmulOp>(user);
    if (!matmulOp)
      continue;

    auto outs = matmulOp.getDpsInits();
    for (Value out : outs) {
      if (out.getDefiningOp() != broadcastOp) {
        return;
      }
    }
  }

  markCube(broadcastOp);
  cubeSeeds.push_back(broadcastOp);
  inBroadcastChain.insert(broadcastOp);

  // Maybe need add some extractShape && cast
  Value src = broadcastOp.getDpsInputs()[0];
  if (auto expandShapeOp = src.getDefiningOp<tensor::ExpandShapeOp>()) {
    src = extractMmadBiasFromPotentialUnitDimExpand(src);
  }

  if (auto castOp = src.getDefiningOp<arith::ExtFOp>()) {
    if (getElementTypeOrSelf(castOp.getIn().getType()).isF16() &&
        getElementTypeOrSelf(castOp.getResult().getType()).isF32()) {
      src = castOp.getIn();
      markCube(castOp);
      cubeSeeds.push_back(castOp);
      inBroadcastChain.insert(castOp);
    }
  }
  if (auto expandShapeOp = src.getDefiningOp<tensor::ExpandShapeOp>()) {
    src = extractMmadBiasFromPotentialUnitDimExpand(src);
  }
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
// Purpose: Ensure matmul result's store operation executes on CUBE for
// efficient matrix data write-back.
// ============================================================================
void OpClassifierPass::matchStorePattern(Operation *user) {
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
//   %slice = tensor.extract_slice %result[0, 0][256, 256][1, 1] :
//   tensor<1024x1024xf32> to tensor<256x256xf32> hivm.store %slice,
//   %memref[%offset] : tensor<256x256xf32> to memref
// Matching Logic
//   1. Check if matmul result's user is tensor.extract_slice
//   2. If matched, mark extract_slice as CUBE and add to cubeSeeds
//   3. Further check all users of extract_slice and mark hivm.store as CUBE
// Purpose: After tiled matrix computation, slice and store operations on CUBE
// improve local data transfer efficiency.
// ============================================================================
void OpClassifierPass::matchExtractSlicePattern(Operation *user) {
  auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
  if (!extractSliceOp)
    return;

  markCube(extractSliceOp);
  cubeSeeds.push_back(extractSliceOp);
  // Also mark downstream hivm.hir.store as CUBE
  for (Operation *sliceUser : extractSliceOp->getUsers()) {
    if (isa<hivm::StoreOp>(sliceUser) ||
        isa<bufferization::MaterializeInDestinationOp>(sliceUser)) {
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
//   1. Check if matmul result's user is
//   bufferization.materialize_in_destination
//   2. If matched, mark it as CUBE and add to cubeSeeds
// Purpose: matmul result materialization on CUBE leverages efficient matrix
// data transfer capabilities.
// ============================================================================
void OpClassifierPass::matchMaterializePattern(Operation *user) {
  if (!isa<bufferization::MaterializeInDestinationOp>(user))
    return;

  markCube(user);
  cubeSeeds.push_back(user);
}

// Pattern matching for CUBE operations
int OpClassifierPass::patternMatchCUBE() {
  LOG_DEBUG("--- Step 1: pattern match --->\n");

  for (Operation *op : allOps) {
    if (!isa<linalg::MatmulOp>(op))
      continue;

    // matmul always CUBE
    opCoreTypes[op] = OP_CUBE_ONLY;
    LLVM_DEBUG(DBGS() << "CUBE (matmul): " << *op << "\n");

    // ---- Upstream pattern matching ----
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      // A null defining op means `operand` is a loop iter_arg block argument.
      // Walk back to its init operand (repeating to also cross nested loops).
      for (Value cur = operand; !def;) {
        auto blockArg = dyn_cast<BlockArgument>(cur);
        auto loopLike = blockArg ? dyn_cast_or_null<LoopLikeOpInterface>(
                                       blockArg.getOwner()->getParentOp())
                                 : nullptr;
        OpOperand *init =
            loopLike ? loopLike.getTiedLoopInit(blockArg) : nullptr;
        if (!init) {
          break;
        }
        cur = init->get();
        def = cur.getDefiningOp();
      }
      if (!def)
        continue;

      matchToTensorPattern(def);
      matchTransposePattern(def);
      matchFillPattern(def);
      matchEmptyPattern(def);
      matchBroadcastPattern(def);
    }

    // ---- Downstream pattern matching ----
    for (Value result : op->getResults()) {
      if (!result.hasOneUse()) {
        continue;
      }
      for (Operation *user : result.getUsers()) {
        // If user is scf.yield, follow the chain to find real users
        Operation *curUser = user;
        Value prevResult = result;
        while (curUser) {
          if (!CVPipeline::allResultHasOneUser(curUser)) {
            break;
          }
          if (auto yieldOp = dyn_cast<scf::YieldOp>(curUser)) {
            if (Operation *scfOp = yieldOp->getParentOp()) {
              // Find which operand index the previous result corresponds to in
              // the yield
              unsigned yieldOperandIdx = 0;
              for (unsigned i = 0; i < yieldOp->getNumOperands(); ++i) {
                if (yieldOp->getOperand(i) == prevResult) {
                  yieldOperandIdx = i;
                  break;
                }
              }
              // Get the corresponding scf result
              if (yieldOperandIdx < scfOp->getNumResults()) {
                Value scfResult = scfOp->getResult(yieldOperandIdx);
                prevResult = scfResult;
                // Find the next user (skipping yield)
                curUser = nullptr;
                for (Operation *nextUser : scfResult.getUsers()) {
                  if (!isa<scf::YieldOp>(nextUser)) {
                    curUser = nextUser;
                    break;
                  }
                }
                // If no non-yield user found, continue searching from yield
                if (!curUser) {
                  for (Operation *nextUser : scfResult.getUsers()) {
                    if (isa<scf::YieldOp>(nextUser)) {
                      curUser = nextUser;
                      break;
                    }
                  }
                }
                continue;
              }
            }
            break;
          }
          matchStorePattern(curUser);
          matchExtractSlicePattern(curUser);
          matchMaterializePattern(curUser);
          break;
        }
      }
    }
  }

  LLVM_DEBUG(DBGS() << "seeds: " << cubeSeeds.size() << " to_tensor(s)\n");
  for (Operation *seed : cubeSeeds) {
    LLVM_DEBUG(DBGS() << "CUBE: " << *seed << "\n");
  }

  return 0;
}

// Helper: Find iter_arg initialization op and yield-assigning op for scf.for
// or scf.while loop-carried scalar. When a def comes from an scf.for/scf.while
// iter_arg and is a scalar compute op, we need to:
// 1. Find the iter_arg's initialization op (the op that provides the initial
// value)
// 2. Find the yieldOp, then trace to the op that provides the yielded value
// All found ops are added to upstreamOps
static void
findIterArgUpstreamOps(Value def,
                       llvm::SmallVectorImpl<Operation *> &upstreamOps) {
  // Check if def is a block argument (iter_arg)
  auto blockArg = dyn_cast<BlockArgument>(def);
  if (!blockArg)
    return;

  Operation *parentOp = blockArg.getOwner()->getParentOp();
  if (!isa<scf::ForOp, scf::WhileOp>(parentOp))
    return;

  // Get the iter_arg index from block argument
  unsigned argIdx = blockArg.getArgNumber();
  auto inits = getLoopInitValues(parentOp);
  // For scf.for: argIdx 0 is lb (loop lower bound), not an iter_arg, skip it.
  // For scf.while: argIdx 0 can be a valid iter_arg, don't skip.
  if (argIdx >= inits.size() || (isa<scf::ForOp>(parentOp) && argIdx == 0))
    return;

  Value initValue = inits[argIdx - (isa<scf::ForOp>(parentOp) ? 1 : 0)];
  if (!isScalarType(initValue))
    return;

  Operation *initDef = initValue.getDefiningOp();
  if (initDef && initDef != parentOp) {
    LLVM_DEBUG(DBGS() << "[findIterArgUpstreamOps] init def: " << *initDef
                      << "\n");
    upstreamOps.push_back(initDef);
  }

  // For scf.while, iter args in after region are 0-indexed in both inits and
  // yield For scf.for, iter args are 1-indexed in initArgs (arg 0 is lb) and
  // 1-indexed in yield (arg 0 is lb)
  unsigned yieldOperandIdx = (isa<scf::ForOp>(parentOp)) ? argIdx - 1 : argIdx;

  Operation *yieldOp = getLoopYieldOp(parentOp);
  if (!yieldOp || !isa<scf::YieldOp>(yieldOp) ||
      yieldOperandIdx >= yieldOp->getNumOperands())
    return;

  Value yieldedValue = yieldOp->getOperand(yieldOperandIdx);
  Operation *yieldedDef = yieldedValue.getDefiningOp();
  if (yieldedDef && yieldedDef != parentOp) {
    LLVM_DEBUG(DBGS() << "[findIterArgUpstreamOps] yielded def: " << *yieldedDef
                      << "\n");
    upstreamOps.push_back(yieldedDef);
  }
}

// Helper function to get upstream operations based on both SSA and memory
// dependencies
void OpClassifierPass::getUpstreamOpsWithMemoryDeps(
    Operation *cur, llvm::SmallVectorImpl<Operation *> &upstreamOps) {
  // Collect SSA dependencies (direct operands)
  for (Value operand : cur->getOperands()) {
    Operation *def = operand.getDefiningOp();
    if (def && def != cur) {
      // Check if this def is an scf.for iter_arg that is a scalar
      LLVM_DEBUG(DBGS() << "push op: " << *def << "\n");
      upstreamOps.push_back(def);
    }
    if (isScalarIterArgOp(operand)) {
      findIterArgUpstreamOps(operand, upstreamOps);
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
      LLVM_DEBUG(DBGS() << "memDef: cur " << *cur << " -> memDef " << *memDef
                        << "\n");
      if (isa<memref::CopyOp>(memDef)) {
        LLVM_DEBUG(DBGS() << "push op: " << *memDef << "\n");
        upstreamOps.push_back(memDef);
      }
    }
  }
}

// arith/math op with a tensor result is VECTOR-only (not CUBE).
static bool isTensorArithOrMathOp(Operation *op) {
  if (!isa<arith::ArithDialect, math::MathDialect>(op->getDialect())) {
    return false;
  }
  for (Value result : op->getResults()) {
    if (isa<RankedTensorType>(result.getType())) {
      return true;
    }
  }
  return false;
}

// True if ANY op in the defining chain is VECTOR-only.  Conservative by
// design: once a chain contains a VECTOR-only op the value (and any extract of
// it) cannot be recomputed on CUBE, so mixed VECTOR-only + CUBE producers still
// classify as VECTOR.  Memoized in `cache` so a tensor extracted from multiple
// times is not re-walked.
static bool hasVectorOnlyProducer(Value value,
                                  llvm::DenseMap<Value, bool> &cache) {
  auto it = cache.find(value);
  if (it != cache.end()) {
    return it->second;
  }

  llvm::SmallVector<Value> worklist{value};
  llvm::DenseSet<Operation *> visited;
  bool result = false;
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    Operation *defOp = cur.getDefiningOp();
    if (!defOp || !visited.insert(defOp).second) {
      continue;
    }
    if (CVPipeline::isVectorOnlyOp(defOp)) {
      result = true;
      break;
    }
    for (Value operand : defOp->getOperands()) {
      worklist.push_back(operand);
    }
  }
  cache[value] = result;
  return result;
}

// Shared skip predicates for both CUBE upstream BFS paths.
bool OpClassifierPass::shouldSkipCubeUpstream(Operation *op) {
  if (isTensorArithOrMathOp(op)) {
    LLVM_DEBUG(DBGS() << "skip " << op->getName().getStringRef()
                      << ": arith/math tensor op\n");
    return true;
  }
  if (auto extOp = dyn_cast<tensor::ExtractOp>(op)) {
    if (hasVectorOnlyProducer(extOp.getTensor(), vectorOnlyProducerCache)) {
      LLVM_DEBUG(DBGS() << "skip " << op->getName().getStringRef()
                        << ": extract of vector-only producer\n");
      return true;
    }
  }
  if (isInsideNestedLinalgRegion(op)) {
    LLVM_DEBUG(DBGS() << "skip " << op->getName().getStringRef()
                      << ": inside linalg block\n");
    return true;
  }
  if (isExtractedLoadStoreRelated(op)) {
    LOG_DEBUG("skip " << *op << ": Extracted Load/Store Related");
    return true;
  }
  return false;
}

// Propagate CUBE core type upstream
int OpClassifierPass::propagateCubeUpstream() {
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
    if (inBroadcastChain.contains(cur)) {
      continue;
    }
    // Get upstream operations considering both SSA and memory dependencies
    llvm::SmallVector<Operation *> upstreamOps;
    getUpstreamOpsWithMemoryDeps(cur, upstreamOps);

    for (Operation *def : upstreamOps) {
      if (!def || cubeVisited.count(def) || isa<linalg::MatmulOp>(def))
        continue;

      // Shared skip rules (see shouldSkipCubeUpstream).
      if (shouldSkipCubeUpstream(def))
        continue;

      cubeVisited.insert(def);

      LLVM_DEBUG(DBGS() << "\tcolor-cube: " << def->getName().getStringRef()
                        << "\n");
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
// is classified as VECTOR. This establishes the default that non-CUBE
// operations should execute on the VECTOR core.
// ============================================================================
int OpClassifierPass::markRemainingAsVector() {
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
// Case 1: fill's outs operand is defined by a CUBE_ONLY operation (not
// tensor.empty)
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
void OpClassifierPass::markFillOpsAsCube() {
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
    if (outsDef && opCoreTypes[outsDef] == OP_CUBE_ONLY &&
        !isa<tensor::EmptyOp>(outsDef)) {
      outsIsCube = true;
      LLVM_DEBUG(DBGS() << "\tfill outs defined by CUBE op: "
                        << outsDef->getName().getStringRef() << "\n");
    } else if (!outsDef) { // Case 2: outs is a BlockArgument
                           // (scf.for/scf.if/scf.while iter_arg)
      auto blockArg = dyn_cast<BlockArgument>(outs);
      if (blockArg) {
        Operation *parentOp = blockArg.getOwner()->getParentOp();
        // Check if it's an scf.for, scf.if, or scf.while iter_arg that is CUBE
        if ((isa<scf::ForOp>(parentOp) || isa<scf::IfOp>(parentOp) ||
             isa<scf::WhileOp>(parentOp)) &&
            opCoreTypes[parentOp] == OP_CUBE_ONLY) {
          outsIsCube = true;
          LLVM_DEBUG(DBGS() << "\tfill outs is CUBE iter_arg of: "
                            << parentOp->getName().getStringRef() << "\n");
        }
      }
    }

    if (outsIsCube) {
      LLVM_DEBUG(DBGS() << "\tfill-cube (outs is CUBE): " << *op << "\n");
      opCoreTypes[op] = OP_CUBE_ONLY;
    }

    // Case 3: handle fill op in scf.if with all CUBE ops, Vector ops are
    // automatically marked as vector.
    handleFillInScfIf(op);
  }
}

// Helper: Handle fill op in scf.if - if all ops in scf.if are CUBE, mark scf.if
// and propagate upstream
void OpClassifierPass::handleFillInScfIf(Operation *fillOp) {
  Operation *parentOp = fillOp->getParentOp();
  auto ifOp = dyn_cast<scf::IfOp>(parentOp);
  if (!ifOp)
    return;

  // Only handle scf.if used for conditional branching (no iter_args)
  // If scf.if has results (iter_args), it's used as a loop structure, skip it
  if (!ifOp.getResults().empty())
    return;

  // Check if all ops in scf.if's blocks are CUBE (except yield terminator)
  bool allOpsAreCube = true;
  for (Region &region : ifOp->getRegions()) {
    for (Block &block : region) {
      for (Operation &innerOp : block.getOperations()) {
        if (isa<scf::YieldOp>(innerOp))
          continue;
        // Check if this op is marked as CUBE
        if (opCoreTypes[&innerOp] != OP_CUBE_ONLY) {
          allOpsAreCube = false;
          break;
        }
      }
      if (!allOpsAreCube)
        break;
    }
    if (!allOpsAreCube)
      break;
  }

  if (allOpsAreCube) {
    LLVM_DEBUG(
        DBGS() << "\tfill in scf.if with all CUBE ops, mark scf.if as CUBE: "
               << *ifOp << "\n");
    opCoreTypes[ifOp] = OP_CUBE_ONLY;
    // Propagate CUBE upstream for the scf.if
    propagateCubeUpstreamForOp(ifOp);
  }
}

// Helper: Propagate CUBE core type upstream for a given operation
void OpClassifierPass::propagateCubeUpstreamForOp(Operation *startOp) {
  std::queue<Operation *> cubeQueue;
  llvm::DenseSet<Operation *> cubeVisited;
  cubeVisited.insert(startOp);
  cubeQueue.push(startOp);

  while (!cubeQueue.empty()) {
    Operation *cur = cubeQueue.front();
    cubeQueue.pop();

    // Get upstream operations
    llvm::SmallVector<Operation *> upstreamOps;
    getUpstreamOpsWithMemoryDeps(cur, upstreamOps);

    for (Operation *upstreamOp : upstreamOps) {
      if (!upstreamOp || cubeVisited.count(upstreamOp))
        continue;
      if (isa<linalg::MatmulOp>(upstreamOp))
        continue;
      // Same skip rules as the main CUBE BFS.
      if (shouldSkipCubeUpstream(upstreamOp))
        continue;

      cubeVisited.insert(upstreamOp);
      LLVM_DEBUG(DBGS() << "\t\tcube upstream: "
                        << upstreamOp->getName().getStringRef() << "\n");
      opCoreTypes[upstreamOp] = OP_CUBE_ONLY;
      cubeQueue.push(upstreamOp);
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
int OpClassifierPass::propagateVectorUpstream() {
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
        (isa<memref::CopyOp, memref::AllocaOp>(cur) &&
         opCoreTypes[cur] == OP_CUBE_ONLY)) {
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
      if (isa<linalg::MatmulOp>(def) ||
          isa<scf::SCFDialect>(def->getDialect())) {
        LLVM_DEBUG(DBGS() << "skip " << def->getName().getStringRef()
                          << ": should not be marked VECTOR\n");
        continue;
      }

      vecVisited.insert(def);

      LLVM_DEBUG(DBGS() << "\tcolor-vec: " << def->getName().getStringRef()
                        << "\n");
      opCoreTypes[def] =
          static_cast<OpCoreType>(opCoreTypes[def] | OP_VECTOR_ONLY);
      vecQueue.push(def);
    }
  }

  return 0;
}

namespace {

Value getAliasSource(Value val) {
  auto *op = val.getDefiningOp();
  if (!op) {
    return nullptr;
  }
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case([](ViewLikeOpInterface viewOp) { return viewOp.getViewSource(); })
      .Case([](bufferization::ToTensorOp totensorOp) {
        return totensorOp.getBuffer();
      })
      .Default([](auto) { return nullptr; });
}

Value getViewSource(Value val) {
  while (auto source = getAliasSource(val)) {
    val = source;
  }
  return val;
}

// A loader loop may only contain data-movement and (scalar) index computation.
// Any tensor-producing compute op means the loop does real vector work and must
// not be swallowed into the cube pipe.
bool isDisqualifyingLoaderOp(Operation *op) {
  if (isa<linalg::MatmulOp, linalg::ReduceOp, linalg::BroadcastOp,
          linalg::GenericOp, linalg::MapOp>(op)) {
    return true;
  }
  if (isa<math::MathDialect>(op->getDialect())) {
    return true;
  }
  // Elementwise arith on tensors is vector compute; scalar index math is fine.
  if (isa<arith::ArithDialect>(op->getDialect())) {
    for (Value result : op->getResults()) {
      if (isa<RankedTensorType>(result.getType())) {
        return true;
      }
    }
  }
  if (auto memInt = llvm::dyn_cast<MemoryEffectOpInterface>(op)) {
    // No matmul - must not be store to gm
    llvm::SmallVector<MemoryEffects::EffectInstance> effects;
    memInt.getEffects(effects);
    for (auto effect : effects) {
      Value val = effect.getValue();
      if (!val || !isa<MemoryEffects::Write>(effect.getEffect())) {
        continue;
      }
      val = getViewSource(val);
      if (auto barg = llvm::dyn_cast<BlockArgument>(val)) {
        if (llvm::isa<func::FuncOp>(barg.getOwner()->getParentOp())) {
          return true;
        }
      }
    }
  }
  return false;
}

} // namespace

bool OpClassifierPass::isCubeLoaderForOp(scf::ForOp forOp) {
  if (forOp->hasAttr(hivm::ExtractLoadStoreAttr))
    return false;

  // Every result must be live and used only by CUBE consumers
  if (forOp.getNumResults() == 0) {
    return false;
  }
  for (Value result : forOp.getResults()) {
    for (Operation *user : result.getUsers()) {
      if (isa<linalg::MatmulOp>(user) || getCoreType(user) == OP_CUBE_ONLY)
        continue;
      return false;
    }
  }
  // Body must be pure data movement
  bool disqualified = false;
  forOp.getBody()->walk([&](Operation *op) {
    if (op == forOp.getOperation())
      return WalkResult::advance();
    if (isDisqualifyingLoaderOp(op)) {
      disqualified = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !disqualified;
}

bool OpClassifierPass::isCubeLoaderForWhileOp(scf::WhileOp whileOp) {
  // Every result must be live and used only by CUBE consumers
  for (Value result : whileOp.getResults()) {
    for (Operation *user : result.getUsers()) {
      if (isa<linalg::MatmulOp>(user) || getCoreType(user) == OP_CUBE_ONLY)
        continue;
      return false;
    }
  }
  // After body must be pure data movement
  bool disqualified = false;
  whileOp.getAfterBody()->walk([&](Operation *op) {
    if (op == whileOp.getOperation())
      return WalkResult::advance();
    if (isDisqualifyingLoaderOp(op)) {
      disqualified = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !disqualified;
}

// Helper: mark all ops in a loop body as CUBE_ONLY
static void
markLoopBodyAsCube(Operation *loop, Block *body,
                   llvm::DenseMap<Operation *, OpCoreType> &opCoreTypes) {
  body->walk([&](Operation *op) {
    if (op == loop || isa<scf::SCFDialect>(op->getDialect()))
      return;
    auto it = opCoreTypes.find(op);
    if (it != opCoreTypes.end())
      it->second = OP_CUBE_ONLY;
  });
  opCoreTypes[loop] = OP_CUBE_ONLY;
}

llvm::LogicalResult OpClassifierPass::penetrateCubeIntoForLoops() {
  // Collect first so recoloring earlier loops cannot perturb the scan.
  // Use a combined walk to collect both ForOps and WhileOps
  llvm::SmallVector<Operation *> loaderLoops;
  getOperation().walk([&](scf::ForOp forOp) {
    if (isCubeLoaderForOp(forOp))
      loaderLoops.push_back(forOp);
  });
  getOperation().walk([&](scf::WhileOp whileOp) {
    if (isCubeLoaderForWhileOp(whileOp))
      loaderLoops.push_back(whileOp);
  });

  for (Operation *loop : loaderLoops) {
    Block *body = getLoopBodyBlock(loop);
    if (body)
      markLoopBodyAsCube(loop, body, opCoreTypes);
  }

  return llvm::success();
}

// ============================================================================
// Helper Functions
// ============================================================================

OpCoreType OpClassifierPass::getCoreType(Operation *op) const {
  auto it = opCoreTypes.find(op);
  if (it != opCoreTypes.end()) {
    return it->second;
  }
  return OP_VECTOR_ONLY;
}

void OpClassifierPass::setCoreType(Operation *op, OpCoreType coreType) {
  opCoreTypes[op] = coreType;
}

std::string
OpClassifierPass::joinCoreTypes(const std::vector<OpCoreType> &coreTypes) {
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

bool OpClassifierPass::shouldStopPropagationForYield(
    Operation *op, OpCoreType targetCoreType) {
  // Stop at linalg.matmul
  if (isa<linalg::MatmulOp>(op)) {
    return true;
  }

  // Stop at hivm.hir.store
  if (isa<hivm::StoreOp>(op)) {
    return true;
  }

  // Stop at tensor-type operations that are already VECTOR_ONLY when targeting
  // CUBE
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

void OpClassifierPass::propagateCoreTypeUpwardForYield(
    Operation *startOp, OpCoreType targetCoreType) {
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

// handleYieldFromElseRegion: Extracts core_type from then region yield for else
// region processing. For scf.if with else region, when processing the else
// region's yield terminator, we need to use the then region's yield's core_type
// attribute as the authoritative source. If the then yield has a
// comma-separated multi-value type string (e.g., "CUBE,VECTOR"), this function
// extracts the i-th component for the operand at index i.
// Parameters
//   - coreTypes: Vector to append the extracted core type to
//   - operandIndex: Index of the operand in the yield (for extracting
//   multi-value types)
//   - thenYieldForElse: The then region's yield operation (from scf.if)
//   - operand: The current operand value from the else region's yield
// Returns: true if core_type was successfully extracted and propagated, false
// otherwise.
bool OpClassifierPass::handleYieldFromElseRegion(
    std::vector<OpCoreType> &coreTypes, unsigned operandIndex,
    Operation *thenYieldForElse, Value &operand, Operation *elseYieldOp) {
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
  OpCoreType coreTypeToUse =
      parseCoreTypeFromString(thenCoreTypeStr, operandIndex);
  coreTypes.push_back(coreTypeToUse);

  // Propagate the determined core_type upstream to the defining operation
  if (Operation *defOp = operand.getDefiningOp()) {
    if (CloneOpMap.count(defOp) && opCoreTypes.count(defOp) &&
        opCoreTypes[defOp] != coreTypeToUse) {
      Operation *cloneOp = CloneOpMap[defOp];
      // Replace the operand in else yield with the cloneOp's result
      for (unsigned i = 0; i < defOp->getNumResults(); ++i) {
        if (defOp->getResult(i) == operand) {
          elseYieldOp->setOperand(operandIndex, cloneOp->getResult(i));
          defOp = cloneOp;
          break;
        }
      }
    }
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
//     defining op's core_type. If it is OP_CUBE_ONLY -> "CUBE", else ->
//     "VECTOR". BlockArguments without a defining op (e.g., iter_arg block
//     argument) default to VECTOR.
// After collecting all operand types, set ssbuffer.core_type on both the yield
// op and its parent scf operation (scf.if or scf.for).
void OpClassifierPass::processYieldOperation(Operation *op,
                                             Operation *thenYieldForElse) {
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
      if (handleYieldFromElseRegion(coreTypes, i, thenYieldForElse, operand,
                                    op)) {
        continue;
      }

      if (Operation *defOp = operand.getDefiningOp()) {
        if (CloneOpMap.count(defOp) && opCoreTypes.count(defOp) &&
            opCoreTypes[defOp] == OP_CUBE_ONLY) {
          Operation *cloneOp = CloneOpMap[defOp];
          // Replace the operand in else yield with the cloneOp's result
          for (unsigned j = 0; j < defOp->getNumResults(); ++j) {
            if (defOp->getResult(j) == operand &&
                j < cloneOp->getNumResults()) {
              op->setOperand(i, cloneOp->getResult(j));
              defOp = cloneOp;
              operand = op->getOperand(i);
              break;
            }
          }
        }
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
    op->setAttr("ssbuffer.core_type",
                StringAttr::get(op->getContext(), coreTypeStr));

    OpCoreType opCt = (coreTypeStr.find("CUBE") != std::string::npos)
                          ? OP_CUBE_ONLY
                          : OP_VECTOR_ONLY;
    opCoreTypes[op] = opCt;

    if (parentOp && llvm::isa<scf::SCFDialect>(parentOp->getDialect())) {
      parentOp->setAttr("ssbuffer.core_type",
                        StringAttr::get(parentOp->getContext(), coreTypeStr));
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
//      then process else-region yield (passing thenYieldForElse for type
//      reference).
//   2. Remaining scf.yield ops not inside scf.if (e.g., scf.for terminators).
// After this step, every scf.yield and its parent scf.if/scf.for carry a
// ssbuffer.core_type attribute reflecting the core_type of their result
// values (derived from the defining ops of each yield operand).
int OpClassifierPass::handleSCFYield() {
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

OpCoreType OpClassifierPass::getForInitCoreType(OpOperand *operand) const {
  // Unified handling for scf.for and scf.while using the tied loop interface
  Operation *owner = operand->getOwner();

  if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
    auto iterArg = forOp.getTiedLoopRegionIterArg(operand);
    if (!iterArg)
      return OP_UNDETERMINED;
    auto sourceOperand = forOp.getTiedLoopYieldedValue(iterArg);
    if (!sourceOperand)
      return OP_UNDETERMINED;
    auto defOp = sourceOperand->get().getDefiningOp();
    if (!defOp)
      return OP_UNDETERMINED;
    return getCoreType(defOp);
  }

  if (auto whileOp = dyn_cast<scf::WhileOp>(owner)) {
    auto iterArg = whileOp.getTiedLoopRegionIterArg(operand);
    if (!iterArg)
      return OP_UNDETERMINED;
    auto sourceOperand = whileOp.getTiedLoopYieldedValue(iterArg);
    if (!sourceOperand)
      return OP_UNDETERMINED;
    auto defOp = sourceOperand->get().getDefiningOp();
    if (!defOp)
      return OP_UNDETERMINED;
    return getCoreType(defOp);
  }

  return OP_UNDETERMINED;
}

// ============================================================================
// Step 7: CUBE_AND_VECTOR Operation Handling
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

void OpClassifierPass::splitOperationForCubeAndVector(
    Operation *op, llvm::DenseSet<Operation *> &processedOps,
    llvm::DenseMap<Operation *, Operation *> &opToVectorClone) {
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
  CloneOpMap[op] = vectorOp;      // record for laterClone map
  CloneOpMap[vectorOp] = op;      // record for laterClone map

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
      if (!user || user == vectorOp) {
        continue;
      }
      OpCoreType coreType = getCoreType(user);
      if (llvm::isa<scf::ForOp, scf::WhileOp>(user)) {
        coreType = getForInitCoreType(&use);
      }
      if (coreType == OP_VECTOR_ONLY) {
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

int OpClassifierPass::handleCubeAndVector() {
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
// Step 9: Stamp Core Type to IR
// ============================================================================

int OpClassifierPass::stampToIR() {
  for (Operation *op : allOps) {
    if (!op || !op->getBlock())
      continue;

    auto it = opCoreTypes.find(op);
    if (it == opCoreTypes.end())
      continue;
    OpCoreType coreType = it->second;

    // Skip most scf dialect operations except scf.condition (terminator in
    // while's before region)
    if (llvm::isa<scf::SCFDialect>(op->getDialect()) &&
        !llvm::isa<scf::ConditionOp>(op))
      continue;

    // scf.condition is always VECTOR (it's the condition check in while's
    // before region)
    if (llvm::isa<scf::ConditionOp>(op)) {
      coreType = OP_VECTOR_ONLY;
    }

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

    op->setAttr("ssbuffer.core_type",
                StringAttr::get(op->getContext(), coreTypeToString(coreType)));
  }

  return 0;
}

// ============================================================================
// Step 1: Mark Synchronization Op
// ============================================================================
llvm::LogicalResult OpClassifierPass::markSynchronizationOp() {
  ModuleOp module = getOperation();
  CVPipeline::ComputeBlockIdManager bm(module);

  for (Operation *op : allOps) {
    if (!isSyncOp(op)) {
      continue;
    }
    if (isa<gpu::BarrierOp>(op)) {
      setCoreType(op, OP_CUBE_AND_VECTOR);
      LLVM_DEBUG(DBGS() << "Barrier classified as CUBE_AND_VECTOR: " << *op
                        << "\n");
    }
    // Pre-existing hivm.sync_block_set/wait carry an explicit tcore_type
    // operand.
    else if (isa<hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp>(op)) {
      auto tcoreAttr = op->getAttrOfType<hivm::TCoreTypeAttr>("tcore_type");
      if (!tcoreAttr) {
        continue;
      }
      switch (tcoreAttr.getTcoretype()) {
      case hivm::TCoreType::CUBE:
        setCoreType(op, OP_CUBE_ONLY);
        break;
      case hivm::TCoreType::VECTOR:
        setCoreType(op, OP_VECTOR_ONLY);
        break;
      default:
        // CUBE_OR_VECTOR / CUBE_AND_VECTOR are not produced by the sync
        // frontend; let the default-VECTOR sweep decide.
        break;
      }
      LOG_DEBUG("sync block set/wait classified by tcore_type: " << *op
                                                                 << "\n");
    } else {
      // Pre-existing hivm.sync_block_all carry an explicit SyncBlockMode
      // attribute Its core type is decided by this attribute
      auto syncBlock = dyn_cast<hivm::SyncBlockOp>(op);
      if (!syncBlock) {
        continue;
      }
      switch (syncBlock.getSyncBlockModeAttr().getSyncMode()) {
      case hivm::SyncBlockMode::ALL_CUBE:
        // stamped as "CUBE" by stampToIR
        setCoreType(op, OP_CUBE_ONLY);
        break;
      case hivm::SyncBlockMode::ALL_VECTOR:
        [[fallthrough]];
      case hivm::SyncBlockMode::ALL_SUB_VECTOR:
        // stamped as "VECTOR" by stampToIR
        setCoreType(op, OP_VECTOR_ONLY);
        break;
      case hivm::SyncBlockMode::ALL:
        // handleCubeAndVector splits it into a CUBE copy + a VECTOR copy that
        // keep the same block id; the degrade pass re-pairs them by block id.
        setCoreType(op, OP_CUBE_AND_VECTOR);
        break;
      default:
        // BARRIER_CUBE / BARRIER_VECTOR are not planned by this pipeline.
        break;
      }
    }

    LOG_DEBUG("sync_block_all classified by mode: " << *op << "\n");

    // Give every debug barrier / sync block its own unique block id so the sync
    // ops lowered from it inherit a block id that no other op shares. Sync ops
    // that OpClassifier already stamped (e.g. hivm.sync_block_all) keep their
    // id; markOpBlockId would fail on them because they are already recorded.
    if (llvm::failed(bm.markOpBlockId(op))) {
      return failure();
    }
    LOG_DEBUG("======== Assigned unique block_id for synchronization op "
              << *op << "\n");
    op->setAttr(CVPipeline::kExternalSync,
                IntegerAttr::get(IntegerType::get(op->getContext(), 32), 1));
  }

  return success();
}

// Run the pass
void OpClassifierPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LLVM_DEBUG(DBGS() << "\n--- Before Plan Compute Block  --->\n");
  LLVM_DEBUG(DBGS() << module << "\n");

  LLVM_DEBUG(DBGS() << "\n--- Running OpClassifierPass --->\n");

  // Initialize memory dependence graph for tracking memory side effects
  aliasAnalysis = std::make_shared<AliasAnalysis>(module);
  memDepGraph = std::make_shared<CVPipeline::MemoryDependenceGraph>(
      module, *aliasAnalysis);

  // Initialize the pass
  initializePass(module);

  // Step 1: Mark synchronization ops at the very beginning
  if (failed(markSynchronizationOp())) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Step 2: Pattern match around each linalg.matmul to find CUBE seeds
  if (patternMatchCUBE() != 0) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Step 3: CUBE upstream BFS from seed loads
  if (propagateCubeUpstream() != 0) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Step 4: Penetrate CUBE coloring into pure loader for-loops.
  if (penetrateCubeIntoForLoops().failed()) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Step 5: Mark remaining operations as VECTOR
  if (markRemainingAsVector() != 0) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Step 6: VECTOR upstream BFS
  if (propagateVectorUpstream() != 0) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Step 7: Handle CUBE_AND_VECTOR operations
  if (handleCubeAndVector() != 0) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Step 8: Process SCF yield results
  if (handleSCFYield() != 0) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Step 9: Stamp to IR
  if (stampToIR() != 0) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  LLVM_DEBUG(DBGS() << "\n--- OpClassifierPass: end --->\n");
}

// Create the pass
namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createOpClassifierPass() {
  return std::make_unique<OpClassifierPass>();
}

void registerAddOpClassifierPasses() {
  registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createOpClassifierPass(); });
}

} // namespace triton
} // namespace mlir
