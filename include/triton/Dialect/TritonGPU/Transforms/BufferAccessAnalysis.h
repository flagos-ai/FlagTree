#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_BUFFERACCESSANALYSIS_H
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_BUFFERACCESSANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace mlir {
namespace triton {
namespace gpu {

/// Memory scope classification for buffers
enum class MemoryScope {
  Global,      /// Global memory (HBM/VRAM)
  Shared,      /// Shared memory (SMEM/LDS)
  Register,    /// Register file
  Unknown      /// Cannot determine
};

/// Information about how a buffer is accessed
struct BufferAccessInfo {
  /// The buffer value being accessed
  Value buffer;

  /// Memory scope of the buffer
  MemoryScope scope;

  /// Operation that produces/writes to this buffer (may be null)
  Operation *producer;

  /// Operations that consume/read from this buffer
  SmallVector<Operation *> consumers;

  /// First access in program order
  Operation *firstAccess;

  /// Last access in program order
  Operation *lastAccess;

  /// Enclosing loop context (if accessed within a loop)
  scf::ForOp loopContext;

  /// Lowest common ancestor of all accesses
  Operation *lca;

  /// Access pattern metadata
  bool isSequential;
  bool isStrided;
  int64_t stride;
  int64_t elementCount;

  /// Element type of the buffer
  Type elementType;

  /// Predecessor buffer (for data flow tracking)
  Value predecessorBuffer;

  /// Enhanced: Block pointer tracking
  bool isBlockPtr;

  /// Enhanced: Global→Shared transfer detection
  bool isGlobalToShared;

  BufferAccessInfo()
      : buffer(nullptr), scope(MemoryScope::Unknown), producer(nullptr),
        firstAccess(nullptr), lastAccess(nullptr), loopContext(nullptr),
        lca(nullptr), isSequential(false), isStrided(false), stride(1),
        elementCount(0), elementType(nullptr), predecessorBuffer(nullptr),
        isBlockPtr(false), isGlobalToShared(false) {}
};

/// Analysis pass for tracking buffer accesses and dependencies
class BufferAccessAnalysis {
public:
  BufferAccessAnalysis() = default;

  /// Run analysis on a function
  void analyze(triton::FuncOp function);

  /// Get access information for a buffer
  BufferAccessInfo *getAccessInfo(Value buffer);

  /// Get all buffers accessed within a loop
  SmallVector<Value> getBuffersInLoop(scf::ForOp loop);

  /// Check if a buffer can be pipelined
  bool isPipelinable(Value buffer);

  /// Compute the lowest common ancestor of all buffer accesses
  Operation *computeLCA(Value buffer);

  /// Clear all analysis results
  void clear();

private:
  /// Map from buffer to access information
  DenseMap<Value, std::unique_ptr<BufferAccessInfo>> bufferInfoMap;

  /// Map from block pointer to base pointer (for tracking global memory sources)
  DenseMap<Value, Value> blockPtrMap;

  /// Current loop nesting during traversal
  SmallVector<scf::ForOp> loopStack;

  /// Operation nesting for LCA computation
  SmallVector<Operation *> opStack;

  /// Visitor functions
  void visitOperation(Operation *op);
  void visitAllocation(Operation *allocOp);
  void visitLoad(Operation *loadOp);
  void visitStore(Operation *storeOp);
  void visitForLoop(scf::ForOp forOp);

  /// Enhanced visitor functions for block pointers and shared memory
  void visitMakeTensorPtr(Operation *makeTensorPtrOp);
  void visitLocalAlloc(Operation *localAllocOp);
  void visitLocalLoad(Operation *localLoadOp);
  void visitLocalStore(Operation *localStoreOp);

  /// Helper functions
  Value getBaseBuffer(Value ptr);
  MemoryScope determineMemoryScope(Value buffer);
  void analyzeAccessPattern(Operation *memOp, BufferAccessInfo *info);
  Operation *findLowestCommonAncestor(Operation *op1, Operation *op2);
  bool hasMemoryDependency(BufferAccessInfo *info);
};

} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_BUFFERACCESSANALYSIS_H
