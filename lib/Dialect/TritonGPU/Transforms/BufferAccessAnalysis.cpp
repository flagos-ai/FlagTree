//===- BufferAccessAnalysis.cpp - Buffer Access Pattern Analysis ---------===//
//
// This file implements buffer access analysis for detecting pipelining
// opportunities in Triton GPU kernels.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonGPU/Transforms/BufferAccessAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "buffer-access-analysis"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// BufferAccessAnalysis Implementation
//===----------------------------------------------------------------------===//

void BufferAccessAnalysis::analyze(triton::FuncOp function) {
  clear();

  // Walk the function in pre-order and post-order
  function.walk([&](Operation *op) {
    visitOperation(op);
  });

  LLVM_DEBUG(llvm::dbgs() << "Analyzed " << bufferInfoMap.size()
                          << " buffers\n");
}

void BufferAccessAnalysis::visitOperation(Operation *op) {
  opStack.push_back(op);

  // Handle different operation types
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    loopStack.push_back(forOp);
    visitForLoop(forOp);
  } else if (op->hasAttr("allocation")) {
    visitAllocation(op);
  } else if (isa<triton::LoadOp>(op)) {
    visitLoad(op);
  } else if (isa<triton::StoreOp>(op)) {
    visitStore(op);
  }
  // Enhanced: Detect block pointer patterns (MakeTensorPtrOp)
  else if (isa<triton::MakeTensorPtrOp>(op)) {
    visitMakeTensorPtr(op);
  }
  // Enhanced: Detect shared memory operations
  else if (isa<triton::gpu::LocalAllocOp>(op)) {
    visitLocalAlloc(op);
  } else if (isa<triton::gpu::LocalLoadOp>(op)) {
    visitLocalLoad(op);
  } else if (isa<triton::gpu::LocalStoreOp>(op)) {
    visitLocalStore(op);
  }

  // Clean up stacks on exit
  op->walk<WalkOrder::PostOrder>([&](Operation *nestedOp) {
    if (nestedOp == op) {
      opStack.pop_back();
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (!loopStack.empty() && loopStack.back() == forOp) {
          loopStack.pop_back();
        }
      }
    }
  });
}

void BufferAccessAnalysis::visitAllocation(Operation *allocOp) {
  Value buffer = allocOp->getResult(0);

  auto info = std::make_unique<BufferAccessInfo>();
  info->buffer = buffer;
  info->scope = determineMemoryScope(buffer);
  info->lca = allocOp;
  info->loopContext = loopStack.empty() ? nullptr : loopStack.back();

  // Calculate element count
  if (auto tensorType = mlir::dyn_cast<RankedTensorType>(buffer.getType())) {
    int64_t count = 1;
    for (auto dim : tensorType.getShape()) {
      if (dim != ShapedType::kDynamic) {
        count *= dim;
      }
    }
    info->elementCount = count;
  }

  LLVM_DEBUG(llvm::dbgs() << "Allocated buffer: " << buffer
                          << " scope=" << static_cast<int>(info->scope)
                          << " elements=" << info->elementCount << "\n");

  bufferInfoMap[buffer] = std::move(info);
}

void BufferAccessAnalysis::visitLoad(Operation *loadOp) {
  auto load = cast<triton::LoadOp>(loadOp);
  Value ptr = load.getPtr();
  Value buffer = getBaseBuffer(ptr);

  if (!buffer || bufferInfoMap.find(buffer) == bufferInfoMap.end()) {
    return;
  }

  auto &info = bufferInfoMap[buffer];

  // Update access tracking
  if (!info->firstAccess) {
    info->firstAccess = loadOp;
  }
  info->lastAccess = loadOp;

  // Add as consumer
  if (!llvm::is_contained(info->consumers, loadOp)) {
    info->consumers.push_back(loadOp);
  }

  // Update LCA
  info->lca = findLowestCommonAncestor(info->lca, opStack.back());

  // Analyze access pattern
  analyzeAccessPattern(loadOp, info.get());

  LLVM_DEBUG(llvm::dbgs() << "Load from buffer: " << buffer << "\n");
}

void BufferAccessAnalysis::visitStore(Operation *storeOp) {
  auto store = cast<triton::StoreOp>(storeOp);
  Value ptr = store.getPtr();
  Value buffer = getBaseBuffer(ptr);

  if (!buffer || bufferInfoMap.find(buffer) == bufferInfoMap.end()) {
    return;
  }

  auto &info = bufferInfoMap[buffer];

  // Update producer (should be unique)
  if (!info->producer) {
    info->producer = storeOp;
  } else {
    // Multiple producers - mark as invalid
    LLVM_DEBUG(llvm::dbgs()
               << "Warning: Multiple producers for buffer " << buffer << "\n");
  }

  // Update access tracking
  if (!info->firstAccess) {
    info->firstAccess = storeOp;
  }
  info->lastAccess = storeOp;

  // Update LCA
  info->lca = findLowestCommonAncestor(info->lca, opStack.back());

  // Track predecessor buffer (data source)
  Value value = store.getValue();
  Value predBuffer = getBaseBuffer(value);
  if (predBuffer && predBuffer != buffer) {
    info->predecessorBuffer = predBuffer;
  }

  // Analyze access pattern
  analyzeAccessPattern(storeOp, info.get());

  LLVM_DEBUG(llvm::dbgs() << "Store to buffer: " << buffer << "\n");
}

void BufferAccessAnalysis::visitForLoop(scf::ForOp forOp) {
  // Loop-specific analysis is handled during traversal
  LLVM_DEBUG(llvm::dbgs() << "Entering loop\n");
}

void BufferAccessAnalysis::visitMakeTensorPtr(Operation *op) {
  auto makeTensorPtrOp = cast<triton::MakeTensorPtrOp>(op);
  // Track block pointer creation for pipelining analysis
  Value result = makeTensorPtrOp.getResult();
  Value base = makeTensorPtrOp.getBase();

  auto info = std::make_unique<BufferAccessInfo>();
  info->buffer = result;
  info->scope = MemoryScope::Global;  // Block pointers typically access global memory
  info->lca = op;
  info->loopContext = loopStack.empty() ? nullptr : loopStack.back();
  info->isBlockPtr = true;

  // Extract shape information from tensor pointer
  auto shape = makeTensorPtrOp.getShape();
  int64_t count = 1;
  for (Value dim : shape) {
    if (auto constOp = dim.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = mlir::dyn_cast<IntegerAttr>(constOp.getValue())) {
        count *= intAttr.getInt();
      }
    }
  }
  info->elementCount = count;

  LLVM_DEBUG(llvm::dbgs() << "MakeTensorPtr: " << result
                          << " elements=" << info->elementCount << "\n");

  blockPtrMap[result] = base;
  bufferInfoMap[result] = std::move(info);
}

void BufferAccessAnalysis::visitLocalAlloc(Operation *op) {
  auto localAllocOp = cast<triton::gpu::LocalAllocOp>(op);
  Value buffer = localAllocOp.getResult();

  auto info = std::make_unique<BufferAccessInfo>();
  info->buffer = buffer;
  info->scope = MemoryScope::Shared;  // LocalAlloc creates shared memory
  info->lca = op;
  info->loopContext = loopStack.empty() ? nullptr : loopStack.back();

  // Get element count from memdesc type
  if (auto memDescType = mlir::dyn_cast<triton::MemDescType>(buffer.getType())) {
    auto shape = memDescType.getShape();
    int64_t count = 1;
    for (auto dim : shape) {
      if (dim != ShapedType::kDynamic) {
        count *= dim;
      }
    }
    info->elementCount = count;
    info->elementType = memDescType.getElementType();
  }

  LLVM_DEBUG(llvm::dbgs() << "LocalAlloc (shared memory): " << buffer
                          << " elements=" << info->elementCount << "\n");

  bufferInfoMap[buffer] = std::move(info);
}

void BufferAccessAnalysis::visitLocalLoad(Operation *op) {
  auto localLoadOp = cast<triton::gpu::LocalLoadOp>(op);
  Value src = localLoadOp.getSrc();
  Value baseBuffer = getBaseBuffer(src);

  if (!baseBuffer) {
    // Try to find the base from memdesc subview
    if (auto subviewOp = src.getDefiningOp<triton::gpu::MemDescSubviewOp>()) {
      baseBuffer = subviewOp.getSrc();
    }
  }

  if (!baseBuffer || bufferInfoMap.find(baseBuffer) == bufferInfoMap.end()) {
    LLVM_DEBUG(llvm::dbgs() << "LocalLoad: could not find base buffer\n");
    return;
  }

  auto &info = bufferInfoMap[baseBuffer];

  // Update access tracking
  if (!info->firstAccess) {
    info->firstAccess = op;
  }
  info->lastAccess = op;

  // Add as consumer (Shared→Register load)
  if (!llvm::is_contained(info->consumers, op)) {
    info->consumers.push_back(op);
  }

  info->lca = findLowestCommonAncestor(info->lca, opStack.back());

  LLVM_DEBUG(llvm::dbgs() << "LocalLoad from shared buffer: " << baseBuffer << "\n");
}

void BufferAccessAnalysis::visitLocalStore(Operation *op) {
  auto localStoreOp = cast<triton::gpu::LocalStoreOp>(op);
  Value dst = localStoreOp.getDst();
  Value baseBuffer = getBaseBuffer(dst);

  if (!baseBuffer) {
    // Try to find the base from memdesc subview
    if (auto subviewOp = dst.getDefiningOp<triton::gpu::MemDescSubviewOp>()) {
      baseBuffer = subviewOp.getSrc();
    }
  }

  if (!baseBuffer || bufferInfoMap.find(baseBuffer) == bufferInfoMap.end()) {
    LLVM_DEBUG(llvm::dbgs() << "LocalStore: could not find base buffer\n");
    return;
  }

  auto &info = bufferInfoMap[baseBuffer];

  // Update producer
  if (!info->producer) {
    info->producer = op;
  }

  // Update access tracking
  if (!info->firstAccess) {
    info->firstAccess = op;
  }
  info->lastAccess = op;

  info->lca = findLowestCommonAncestor(info->lca, opStack.back());

  // Track the source of the store (for Global→Shared pipeline)
  Value srcValue = localStoreOp.getSrc();
  if (auto loadOp = srcValue.getDefiningOp<triton::LoadOp>()) {
    // This is a Global→Shared transfer pattern
    info->isGlobalToShared = true;
    LLVM_DEBUG(llvm::dbgs() << "LocalStore: Global→Shared transfer detected\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "LocalStore to shared buffer: " << baseBuffer << "\n");
}

Value BufferAccessAnalysis::getBaseBuffer(Value ptr) {
  // Trace pointer back to allocation
  Value current = ptr;
  int maxDepth = 10; // Prevent infinite loops

  while (current && maxDepth-- > 0) {
    Operation *defOp = current.getDefiningOp();
    if (!defOp) {
      break;
    }

    // Check if this is an allocation
    if (defOp->hasAttr("allocation")) {
      return current;
    }

    // Follow pointer operations
    if (auto splatOp = dyn_cast<triton::SplatOp>(defOp)) {
      current = splatOp.getSrc();
    } else if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(defOp)) {
      current = broadcastOp.getSrc();
    } else if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(defOp)) {
      current = addPtrOp.getPtr();
    } else if (auto convertOp = dyn_cast<triton::gpu::ConvertLayoutOp>(defOp)) {
      current = convertOp.getSrc();
    } else {
      // Can't trace further
      break;
    }
  }

  return nullptr;
}

MemoryScope BufferAccessAnalysis::determineMemoryScope(Value buffer) {
  auto tensorType = mlir::dyn_cast<RankedTensorType>(buffer.getType());
  if (!tensorType) {
    return MemoryScope::Unknown;
  }

  auto encoding = tensorType.getEncoding();
  if (!encoding) {
    return MemoryScope::Global;
  }

  // Check for shared memory encoding
  if (auto sharedEnc = mlir::dyn_cast<triton::gpu::SharedEncodingAttr>(encoding)) {
    return MemoryScope::Shared;
  }

  // Check for register encoding (blocked, slice, etc.)
  if (mlir::isa<triton::gpu::BlockedEncodingAttr>(encoding) ||
      mlir::isa<triton::gpu::SliceEncodingAttr>(encoding) ||
      mlir::isa<triton::gpu::NvidiaMmaEncodingAttr>(encoding)) {
    return MemoryScope::Register;
  }

  return MemoryScope::Global;
}

void BufferAccessAnalysis::analyzeAccessPattern(Operation *memOp,
                                                 BufferAccessInfo *info) {
  // Simple heuristic: if accessed in a loop with induction variable,
  // assume sequential or strided
  if (loopStack.empty()) {
    info->isSequential = false;
    info->isStrided = false;
    return;
  }

  // For now, mark as sequential if in loop
  // Full analysis would examine index expressions
  info->isSequential = true;
  info->isStrided = false;
  info->stride = 1;
}

Operation *BufferAccessAnalysis::findLowestCommonAncestor(Operation *op1,
                                                           Operation *op2) {
  if (!op1) return op2;
  if (!op2) return op1;
  if (op1 == op2) return op1;

  // Build path from op1 to root
  SmallVector<Operation *> path1;
  Operation *current = op1;
  while (current) {
    path1.push_back(current);
    current = current->getParentOp();
  }

  // Traverse from op2 and find first intersection
  current = op2;
  while (current) {
    if (llvm::is_contained(path1, current)) {
      return current;
    }
    current = current->getParentOp();
  }

  return op1->getParentOfType<FuncOp>();
}

bool BufferAccessAnalysis::hasMemoryDependency(BufferAccessInfo *info) {
  // Check for memory dependencies between producer and consumers
  // that would prevent safe pipelining

  if (!info->producer || info->consumers.empty()) {
    return false;
  }

  // Get the loop context for checking cross-iteration dependencies
  scf::ForOp loop = info->loopContext;
  if (!loop) {
    // Not in a loop - no cross-iteration dependencies possible
    return false;
  }

  Value inductionVar = loop.getInductionVar();

  // Helper to extract index expressions from a memory operation
  auto extractIndices = [](Operation *memOp) -> SmallVector<Value> {
    SmallVector<Value> indices;

    if (auto loadOp = dyn_cast<triton::LoadOp>(memOp)) {
      Value ptr = loadOp.getPtr();
      // Trace through addptr operations to find indices
      while (ptr) {
        if (auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>()) {
          indices.push_back(addPtrOp.getOffset());
          ptr = addPtrOp.getPtr();
        } else {
          break;
        }
      }
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(memOp)) {
      Value ptr = storeOp.getPtr();
      while (ptr) {
        if (auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>()) {
          indices.push_back(addPtrOp.getOffset());
          ptr = addPtrOp.getPtr();
        } else {
          break;
        }
      }
    } else if (auto localLoadOp = dyn_cast<triton::gpu::LocalLoadOp>(memOp)) {
      Value src = localLoadOp.getSrc();
      if (auto subviewOp = src.getDefiningOp<triton::gpu::MemDescSubviewOp>()) {
        for (Value offset : subviewOp.getOffsets()) {
          indices.push_back(offset);
        }
      }
    } else if (auto localStoreOp = dyn_cast<triton::gpu::LocalStoreOp>(memOp)) {
      Value dst = localStoreOp.getDst();
      if (auto subviewOp = dst.getDefiningOp<triton::gpu::MemDescSubviewOp>()) {
        for (Value offset : subviewOp.getOffsets()) {
          indices.push_back(offset);
        }
      }
    }

    return indices;
  };

  // Helper to check if an index depends on the loop induction variable
  auto dependsOnInductionVar = [&inductionVar](Value index) -> bool {
    if (!index || !inductionVar) {
      return false;
    }

    // Direct use
    if (index == inductionVar) {
      return true;
    }

    // Check if index is derived from induction variable
    Operation *defOp = index.getDefiningOp();
    if (!defOp) {
      return false;
    }

    // Simple check: walk through arithmetic operations
    SmallVector<Operation *> worklist;
    DenseSet<Operation *> visited;
    worklist.push_back(defOp);

    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (visited.count(op)) {
        continue;
      }
      visited.insert(op);

      for (Value operand : op->getOperands()) {
        if (operand == inductionVar) {
          return true;
        }
        if (auto definingOp = operand.getDefiningOp()) {
          worklist.push_back(definingOp);
        }
      }
    }

    return false;
  };

  // Extract producer indices
  SmallVector<Value> producerIndices = extractIndices(info->producer);

  // Check each consumer for potential dependencies
  for (Operation *consumer : info->consumers) {
    SmallVector<Value> consumerIndices = extractIndices(consumer);

    // Case 1: RAW (Read-After-Write) dependency
    // Consumer reads from location written by producer in same or previous iteration
    // This is the normal producer-consumer pattern we want for pipelining

    // Case 2: WAR (Write-After-Read) dependency within same iteration
    // Producer writes to location that consumer reads
    // Check if producer and consumer access same indices
    bool sameIteration = true;
    if (!producerIndices.empty() && !consumerIndices.empty()) {
      // Check if any producer index depends on induction variable
      for (Value idx : producerIndices) {
        if (dependsOnInductionVar(idx)) {
          sameIteration = false; // Different iterations access different locations
          break;
        }
      }
    }

    // Case 3: Cross-iteration dependency that prevents pipelining
    // If producer writes to a location that consumer reads from a FUTURE iteration
    // this would require the pipeline to wait

    // Check for loop-carried dependency
    if (loop) {
      // If neither producer nor consumer indices depend on induction variable,
      // they access the same location every iteration - potential dependency
      bool producerDepends = false;
      bool consumerDepends = false;

      for (Value idx : producerIndices) {
        if (dependsOnInductionVar(idx)) {
          producerDepends = true;
          break;
        }
      }

      for (Value idx : consumerIndices) {
        if (dependsOnInductionVar(idx)) {
          consumerDepends = true;
          break;
        }
      }

      // If both access patterns depend on induction variable in different ways,
      // need more sophisticated analysis
      if (producerDepends && consumerDepends) {
        // Check if they access the same iteration's data
        // For simple cases: producer[i] and consumer[i] is safe
        // producer[i] and consumer[i-1] requires distance >= numStages

        // Conservative: if patterns look different, assume dependency
        if (producerIndices.size() != consumerIndices.size()) {
          LLVM_DEBUG(llvm::dbgs() << "Memory dependency: different index patterns\n");
          return true;
        }
      }

      // If neither depends on induction variable, they access same location
      // every iteration - this is a dependency
      if (!producerDepends && !consumerDepends &&
          !producerIndices.empty() && !consumerIndices.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "Memory dependency: loop-invariant access pattern\n");
        return true;
      }
    }

    // Check dominance relationship
    // Producer should dominate consumer for safe RAW dependency
    if (info->producer->getBlock() == consumer->getBlock()) {
      if (!info->producer->isBeforeInBlock(consumer)) {
        // Consumer comes before producer in same block - problematic
        LLVM_DEBUG(llvm::dbgs() << "Memory dependency: consumer before producer\n");
        return true;
      }
    }
  }

  // No problematic dependencies found
  LLVM_DEBUG(llvm::dbgs() << "No memory dependency detected\n");
  return false;
}

BufferAccessInfo *BufferAccessAnalysis::getAccessInfo(Value buffer) {
  auto it = bufferInfoMap.find(buffer);
  if (it != bufferInfoMap.end()) {
    return it->second.get();
  }
  return nullptr;
}

SmallVector<Value> BufferAccessAnalysis::getBuffersInLoop(scf::ForOp loop) {
  SmallVector<Value> buffers;
  for (auto &entry : bufferInfoMap) {
    if (entry.second->loopContext == loop) {
      buffers.push_back(entry.first);
    }
  }
  return buffers;
}

bool BufferAccessAnalysis::isPipelinable(Value buffer) {
  auto *info = getAccessInfo(buffer);
  if (!info) {
    return false;
  }

  // Must be accessed within a loop
  if (!info->loopContext) {
    return false;
  }

  // Must have clear producer-consumer relationship
  if (!info->producer || info->consumers.empty()) {
    return false;
  }

  // Must not have conflicting memory dependencies
  if (hasMemoryDependency(info)) {
    return false;
  }

  return true;
}

Operation *BufferAccessAnalysis::computeLCA(Value buffer) {
  auto *info = getAccessInfo(buffer);
  return info ? info->lca : nullptr;
}

void BufferAccessAnalysis::clear() {
  bufferInfoMap.clear();
  blockPtrMap.clear();
  loopStack.clear();
  opStack.clear();
}
