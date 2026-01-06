#include "Analysis/Allocation.h"
#include "Analysis/Alias.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <cassert>
#include <cstdint>

#define DEBUG_TYPE "allocation-shared-memory"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

//===----------------------------------------------------------------------===//
// Shared Memory Allocation Analysis
//===----------------------------------------------------------------------===//
namespace mlir::triton::alloc {

class AllocationAnalysis {
public:
  enum class BufferAccessMode { READ, WRITE, READ_WRITE, UnSupported };

public:
  AllocationAnalysis(Operation *operation,
                     Allocation::FuncAllocMapT *funcAllocMap,
                     Allocation *allocation)
      : operation(operation), funcAllocMap(funcAllocMap),
        allocation(allocation) {
    run();
  }

private:
  using BufferT = Allocation::BufferT;

  /// Value -> Liveness Range
  /// Use MapVector to ensure determinism.
  using BufferRangeMapT = llvm::MapVector<BufferT *, Interval<size_t>>;
  /// Nodes -> Nodes
  using GraphT = DenseMap<BufferT *, DenseSet<BufferT *>>;

  void run() {
    getValuesAndSizes();
    resolveLiveness();
    computeOffsets();
  }

  /// Initializes explicitly defined shared memory values for a given operation.
  void getExplicitValueSize(Operation *op) {
    // FIXME: Support memory hierarchy (Multi-memory allocation. eg: Shared
    // memory && scratch memory)
    auto alloc = dyn_cast<memref::AllocOp>(op);
    if (!alloc)
      return;
    // Bytes could be a different value once we support padding or other
    // allocation policies.
    auto allocType = alloc.getType();
    // FIXME: padding or other alignment
    auto bitWidth = allocType.getElementTypeBitWidth();
    auto elemByte = (bitWidth + 7) / 8;
    int64_t bytes = allocType.getNumElements() * elemByte;

    auto alignment = alloc.getAlignment().value_or(1);

    // WORKAROUND: Reduce op output will write to alignment 256 bytes
    // FIXME: Handle tensors that require more memory than their shape suggests,
    // for example, due to padding or alignment requirements.
    bytes = (bytes + alignment - 1) / alignment * alignment;

    allocation->addBuffer<BufferT::BufferKind::Explicit>(alloc, bytes,
                                                         alignment, 0);
  }

  void getValueAlias(Value value,
                     triton::alias::SharedMemoryAliasAnalysis &analysis) {
    dataflow::Lattice<triton::alias::AliasInfo> *latticeElement =
        analysis.getLatticeElement(value);
    if (!latticeElement)
      return;

    triton::alias::AliasInfo &info = latticeElement->getValue();
    if (info.getAllocs().empty()) {
      LLVM_DEBUG({
        llvm::dbgs() << "\tNo allocs found for value: ";
        value.dump();
      });
      return;
    }

    for (auto alloc : info.getAllocs()) {
      // FIXME: Why this happens? DPS?
      if (value == alloc)
        continue;
      LLVM_DEBUG({
        llvm::dbgs() << "\tAdd alias value: ";
        value.dump();
        llvm::dbgs() << "\t to alloc: ";
        alloc.dump();
      });
      allocation->addAlias(value, alloc);
    }
  }

  /// Extract all shared memory values and their sizes
  void getValuesAndSizes() {
    // Get the alloc values
    operation->walk<WalkOrder::PreOrder>(
        [&](Operation *op) { getExplicitValueSize(op); });

    LDBG("\nGet buffer and size --");
    for (auto valueBufferIter : allocation->valueBuffer) {
      auto *buffer = valueBufferIter.second;
      LLVM_DEBUG(llvm::dbgs()
                     << "-- buffer " << buffer->id << " size: " << buffer->size
                     << " offset: " << buffer->offset << "\n\t";
                 buffer->owner->dump(););
    }
    LLVM_DEBUG({ llvm::dbgs() << "\n\n"; });

    // Get the alias values
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    triton::alias::SharedMemoryAliasAnalysis *aliasAnalysis =
        solver->load<triton::alias::SharedMemoryAliasAnalysis>();
    // Run the analysis rooted at every isolated from above operation, including
    // the top-level function but also any nested regions.
    operation->walk([&](Operation *op) {
      if (op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
          failed(solver->initializeAndRun(op))) {
        // TODO: return error instead of bailing out..
        llvm_unreachable("failed to run SharedMemoryAliasAnalysis");
      }
    });

    LDBG("\n==== Value alias =============");
    operation->walk<WalkOrder::PreOrder>([&](Operation *op) {
      LLVM_DEBUG({
        llvm::dbgs() << "\nValue Alias for op: ";
        if (!op->hasTrait<OpTrait::IsIsolatedFromAbove>())
          op->dump();
        else
          op->getName();
      });

      for (auto operand : op->getOperands()) {
        getValueAlias(operand, *aliasAnalysis);
      }
      for (auto value : op->getResults()) {
        getValueAlias(value, *aliasAnalysis);
      }
    });
    LLVM_DEBUG({ llvm::dbgs() << "\n\n"; });
  }

  /// Traverse the use-def chain to find out the earliest memory access pattern
  /// operation

  /// Check whether we need to update the memory access pattern
  bool needUpdateMemAccessPattern(Operation *last, Operation *current,
                                  DenseMap<Operation *, size_t> &operationId) {
    assert(current && "current op is null");
    if (!last)
      return true;

    auto lastOpParent = last->getParentOp();
    auto currentOpParent = current->getParentOp();
    assert(lastOpParent && currentOpParent && "parent op is null");
    // Same region, compare operation id
    if (lastOpParent == currentOpParent)
      return operationId[last] < operationId[current];

    // Sub-region, always update
    if (lastOpParent->isProperAncestor(currentOpParent))
      return true;

    return false;
  }

  /// Set the access pattern according the live operation
  void setAccessPattern(
      DenseMap<BufferT *, llvm::SmallVector<Operation *, 3>> &BufferAccessMap,
      DenseMap<Operation *, size_t> &operationId, BufferT *buffer,
      Operation *liveOp, BufferAccessMode mode) {
    auto minAccessIDOp = BufferAccessMap[buffer][static_cast<size_t>(mode)];

    BufferAccessMap[buffer][static_cast<size_t>(mode)] =
        needUpdateMemAccessPattern(minAccessIDOp, liveOp, operationId)
            ? liveOp
            : minAccessIDOp;
  }

  bool isPartialWrite(Value operand, BufferT *buffer) {
    auto memrefType = cast<MemRefType>(operand.getType());
    assert(isa<memref::AllocOp>(buffer->owner));
    auto bufferType = cast<MemRefType>(buffer->owner->getResultTypes().front());
    return memrefType.getShape() != bufferType.getShape();
  }

  /// Handle MemoryEffectOpInterface access pattern
  void memoryEffectOpInterfaceAccessPattern(
      DenseMap<BufferT *, llvm::SmallVector<Operation *, 3>> &BufferAccessMap,
      DenseMap<Operation *, size_t> &operationId, BufferT *buffer,
      Operation *liveOp, OpOperand &opOperand) {
    auto memEffectOp = cast<MemoryEffectOpInterface>(liveOp);
    SmallVector<MemoryEffects::EffectInstance, 2> effects;
    memEffectOp.getEffects(effects);

    auto operand = opOperand.get();
    for (const auto &effect : effects) {
      if (effect.getValue() != operand)
        continue;
      if (isa<MemoryEffects::Read>(effect.getEffect())) {
        setAccessPattern(BufferAccessMap, operationId, buffer, liveOp,
                         BufferAccessMode::READ);
        LLVM_DEBUG(llvm::dbgs()
                   << "  -> MemoryEffectOpInterface READ access for buffer "
                   << buffer->id << "\n");
      } else if (isa<MemoryEffects::Write>(effect.getEffect())) {
        if (isPartialWrite(operand, buffer))
          continue;

        setAccessPattern(BufferAccessMap, operationId, buffer, liveOp,
                         BufferAccessMode::WRITE);
        LLVM_DEBUG(llvm::dbgs()
                   << "  -> MemoryEffectOpInterface WRITE access for buffer "
                   << buffer->id << "\n");
      } else {
        assert(false && "unknown memory effect");
      }
    }
  }

  /// Resolve the buffer access pattern for all buffers
  DenseMap<BufferT *, llvm::SmallVector<Operation *, 3>>
  resolveBufferAccessPattern(Liveness &liveness,
                             DenseMap<Operation *, size_t> &operationId) {
    // Each access pattern will store the earliest operation that performs the
    // access.
    DenseMap<BufferT *, llvm::SmallVector<Operation *, 3>>
        InLoopBufferAccessPattern;
    for (auto [K, V] : allocation->valueBuffer) {
      if (V->owner == findOutmostParentLoopOp(V->owner))
        continue;
      InLoopBufferAccessPattern[V] = llvm::SmallVector<Operation *, 3>(
          static_cast<size_t>(BufferAccessMode::UnSupported), nullptr);
    }

    // TODO: More complicated access pattern analysis
    auto getMemoryAccessPattern = [&](Value value, BufferT *buffer) {
      LLVM_DEBUG(llvm::dbgs() << "\nAnalyzing memory access pattern for buffer "
                              << buffer->id << " buffer: ";
                 buffer->owner->dump(); llvm::dbgs() << "\n";);

      auto liveOperations = liveness.resolveLiveness(value);
      std::for_each(
          liveOperations.begin(), liveOperations.end(), [&](Operation *liveOp) {
            for (auto &opOperand : liveOp->getOpOperands()) {
              auto operand = opOperand.get();
              auto bufferIds = allocation->getBufferIds(operand);
              if (bufferIds.empty())
                continue;

              // scf::if may has multiple buffers associated with one
              // operand
              if (!bufferIds.contains(buffer->id))
                continue;

              LLVM_DEBUG(llvm::dbgs() << "Live Operation: \n"; liveOp->dump();
                         llvm::dbgs() << "  Operand: \n\t"; operand.dump();
                         llvm::dbgs() << "\n"; fflush(stderr););

              if (liveOp->mightHaveTrait<OpTrait::IsTerminator>()) {
                // FIXME: Handle terminators properly. scf::if
                InLoopBufferAccessPattern[buffer] =
                    llvm::SmallVector<Operation *, 3>(
                        static_cast<size_t>(BufferAccessMode::UnSupported),
                        nullptr);

                LLVM_DEBUG(llvm::dbgs() << "  -> terminator for buffer "
                                        << buffer->id << "\n");
              } else if (isa<MemoryEffectOpInterface>(liveOp)) {
                // Memory effect ops
                LLVM_DEBUG(llvm::dbgs() << "  -> MemoryEffectOpInterface "
                                           "buffer access pattern!\n";);
                memoryEffectOpInterfaceAccessPattern(InLoopBufferAccessPattern,
                                                     operationId, buffer,
                                                     liveOp, opOperand);

                continue;
              } else {
                LLVM_DEBUG(llvm::dbgs()
                               << "  -> unknown buffer access pattern!\n";);
                assert(allocation->valueBuffer.contains(operand));
                assert(false && "unknown buffer access pattern");
              }
            }
          });
    };

    // Compute access pattern for explicitly defined buffers
    for (auto valueBufferIter : allocation->valueBuffer) {
      auto value = valueBufferIter.first;
      auto *buffer = valueBufferIter.second;
      if (!InLoopBufferAccessPattern.contains(buffer))
        continue;
      getMemoryAccessPattern(value, buffer);
    }

    // Compute access pattern for alias buffers
    for (const auto &[value, buffers] : allocation->aliasBuffer) {
      for (auto *buffer : buffers) {
        if (!InLoopBufferAccessPattern.contains(buffer))
          continue;
        getMemoryAccessPattern(value, buffer);
      }
    }
    return InLoopBufferAccessPattern;
  }

  void updateToLoopLiveness(BufferT *buffer,
                            DenseMap<Operation *, size_t> &operationId) {
    auto parentOp = findOutmostParentLoopOp(buffer->owner);
    assert(bufferRange.count(buffer));

    assert(parentOp->hasTrait<OpTrait::SingleBlock>());
    auto &entryBlock = parentOp->getRegion(0).getBlocks().front();
    auto firstOp = &entryBlock.front();

    auto minId = std::min(operationId[firstOp], bufferRange[buffer].start());
    auto maxId = std::max(operationId[parentOp] + 1, bufferRange[buffer].end());
    bufferRange[buffer] = Interval(minId, maxId);
  }

  void updateInLoopBufferLiveness(
      DenseMap<BufferT *, llvm::SmallVector<Operation *, 3>> &BufferAccessMap,
      DenseMap<Operation *, size_t> &operationId) {

    LLVM_DEBUG(
        llvm::dbgs() << "\n======  InLoopBuffer Access Pattern :  ==========\n";
        for (auto [buffer, access]
             : BufferAccessMap) {
          auto minWOp = access[static_cast<size_t>(BufferAccessMode::WRITE)];
          auto minROp = access[static_cast<size_t>(BufferAccessMode::READ)];
          auto minRWOp =
              access[static_cast<size_t>(BufferAccessMode::READ_WRITE)];
          llvm::dbgs() << "Buffer " << buffer->id << " ";
          buffer->owner->dump();
          llvm::dbgs() << " READ=";
          minROp == nullptr
              ? llvm::dbgs() << std::numeric_limits<size_t>::max() << "\n"
              : llvm::dbgs() << operationId[minROp] << "\n";
          llvm::dbgs() << ", WRITE=";
          minWOp == nullptr
              ? llvm::dbgs() << std::numeric_limits<size_t>::max() << "\n"
              : llvm::dbgs() << operationId[minWOp] << "\n";
          llvm::dbgs() << ", READ_WRITE=";
          minRWOp == nullptr
              ? llvm::dbgs() << std::numeric_limits<size_t>::max() << "\n"
              : llvm::dbgs() << operationId[minRWOp] << "\n";
          llvm::dbgs() << ", \n\n";
          fflush(stderr);
        };);

    for (auto [buffer, access] : BufferAccessMap) {
      auto minWOp = access[static_cast<size_t>(BufferAccessMode::WRITE)];
      auto minROp = access[static_cast<size_t>(BufferAccessMode::READ)];
      auto minRWOp = access[static_cast<size_t>(BufferAccessMode::READ_WRITE)];

      assert(minRWOp == nullptr &&
             "READ_WRITE access pattern is not supported yet");
      if (!minWOp || (minROp && operationId[minROp] < operationId[minWOp]))
        updateToLoopLiveness(buffer, operationId);
    }
  }

  /// Computes the liveness range of the allocated value.
  /// Each buffer is allocated only once.
  void resolveExplicitBufferLiveness(
      function_ref<Interval<size_t>(Value value, BufferT *buffer)>
          getLiveness) {
    for (auto valueBufferIter : allocation->valueBuffer) {
      auto value = valueBufferIter.first;
      auto *buffer = valueBufferIter.second;
      bufferRange[buffer] = getLiveness(value, buffer);
      LLVM_DEBUG({
        llvm::dbgs() << "-- buffer " << buffer->id << "; value: ";
        value.dump();
      });
    }
  }

  /// Extends the liveness range by unionizing the liveness range of the aliased
  /// values because each allocated buffer could be an alias of others, if block
  /// arguments are involved.
  void resolveAliasBufferLiveness(
      function_ref<Interval<size_t>(Value value, BufferT *buffer)>
          getLiveness) {
    for (const auto &[value, buffers] : allocation->aliasBuffer) {
      auto range = getLiveness(value, buffers.front());
      for (auto *buffer : buffers) {
        auto minId = range.start();
        auto maxId = range.end();
        if (bufferRange.count(buffer)) {
          // Extend the allocated buffer's range
          minId = std::min(minId, bufferRange[buffer].start());
          maxId = std::max(maxId, bufferRange[buffer].end());
        }
        bufferRange[buffer] = Interval(minId, maxId);
      }
    }
  }

  Operation *findOutmostParentLoopOp(Operation *op) {
    if (!op) {
      return nullptr;
    }
    auto parentOp = op->getParentOp();
    if (!parentOp) {
      return op;
    }
    if (!isa<scf::ForOp>(parentOp) && !isa<scf::IfOp>(parentOp) &&
        !isa<scf::WhileOp>(parentOp)) {
      return op;
    }
    return findOutmostParentLoopOp(parentOp);
  }

  DenseMap<size_t, Operation *> idToOperation;

  /// Resolves liveness of all values involved under the root operation.
  void resolveLiveness() {
    // Assign an ID to each operation using post-order traversal.
    // To achieve the correct liveness range, the parent operation's ID
    // should be greater than each of its child operation's ID .
    // Example:
    //     ...
    //     %5 = triton.convert_layout %4
    //     %6 = scf.for ... iter_args(%arg0 = %0) -> (i32) {
    //       %2 = triton.convert_layout %5
    //       ...
    //       scf.yield %arg0
    //     }
    // For example, %5 is defined in the parent region and used in
    // the child region, and is not passed as a block argument.
    // %6 should should have an ID greater than its child operations,
    // otherwise %5 liveness range ends before the child operation's liveness
    // range ends.
    DenseMap<Operation *, size_t> operationId;
    LLVM_DEBUG(
        llvm::dbgs()
        << "\n=== Assigning operation IDs using post-order traversal ===\n");
    operation->walk<WalkOrder::PostOrder>([&](Operation *op) {
      LLVM_DEBUG(llvm::dbgs() << "Assigning ID " << operationId.size()
                              << " to operation: ";
                 if (!op->hasTrait<OpTrait::IsIsolatedFromAbove>()) op->dump();
                 else op->getName(););
      operationId[op] = operationId.size();
    });
    LLVM_DEBUG(llvm::dbgs() << "\n\n");

    for (auto [K, V] : operationId)
      idToOperation[V] = K;

    // Analyze liveness of explicit buffers
    Liveness liveness(operation);
    auto getValueLivenessRange = [&](Value value, BufferT *buffer) {
      auto liveOperations = liveness.resolveLiveness(value);
      // TODO: Support async
      // Update regions for buffer.

      auto minId = std::numeric_limits<size_t>::max();
      auto maxId = std::numeric_limits<size_t>::min();
      std::for_each(liveOperations.begin(), liveOperations.end(),
                    [&](Operation *liveOp) {
                      minId = std::min(minId, operationId[liveOp]);
                      // FIXME: Optimize. Since buffer in loop will always has
                      // same address, so assumed they have the same liveness
                      // range with the parent loop operation.
                      auto parentOp = liveOp;
                      maxId = std::max(maxId, operationId[parentOp] + 1);
                    });
      return Interval(minId, maxId);
    };

    resolveExplicitBufferLiveness(getValueLivenessRange);
    resolveAliasBufferLiveness(getValueLivenessRange);

    // Process in-loop buffer access pattern to extend liveness range
    auto BufferAccessMap = resolveBufferAccessPattern(liveness, operationId);
    updateInLoopBufferLiveness(BufferAccessMap, operationId);
  }

  void dumpBuffers() {
    LDBG("\nDump bufferRange: id size offset ---------");
    for (auto bufferIter : bufferRange) {
      LLVM_DEBUG({
        bufferIter.first->owner->dump();
        llvm::dbgs() << "-- " << bufferIter.first->id << "  "
                     << bufferIter.first->size << " "
                     << bufferIter.first->offset << " "
                     << "interval " << bufferIter.second.start() << " "
                     << bufferIter.second.end() << "\n";
        llvm::dbgs() << "\t start: ";
        idToOperation.at(bufferIter.second.start())->dump();
        llvm::dbgs() << "\t end: ";
        idToOperation.at(bufferIter.second.end())->dump();
      });
    }
    llvm::dbgs() << "\n\n";
  }

  void dumpAllocationSize() const {
    LDBG("\nDump shared memory allocation size -----------");
    auto liveBuffers = allocation->getLiveBuffers();
    auto analyzedSize = 0;
    for (auto [op, bufferIds] : liveBuffers) {
      auto size = 0;
      for (auto bufferId : bufferIds) {
        auto bufferSize = allocation->getAllocatedSize(bufferId);
        size += bufferSize;
      }
      analyzedSize = std::max(analyzedSize, size);
    }
    llvm::dbgs() << "Allocated: " << allocation->sharedMemorySize
                 << ", analyzed: " << analyzedSize << "\n";
    llvm::dbgs() << "\n\n";
  }

  void dumpInterferenceGraph(const GraphT &interference) const {
    LDBG("\nDump interference graph: \n");
    for (auto edges : interference) {
      llvm::dbgs() << "-- from " << edges.first->id << " to ";
      for (auto node : edges.second) {
        llvm::dbgs() << node->id << "; ";
      }
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "\n\n";
  }

  /// Computes the shared memory offsets for all related values.
  /// Paper: Algorithms for Compile-Time Memory Optimization
  /// (https://dl.acm.org/doi/pdf/10.5555/314500.315082)
  void computeOffsets() {
    SmallVector<BufferT *> buffers;
    for (auto bufferIter : bufferRange) {
      buffers.emplace_back(bufferIter.first);
    }

    // Sort buffers by size in descending order to reduce the fragmentation
    // on big buffers caused by smaller buffers. Big buffers have a higher
    // chance to overlap with multiple other buffers, and allocating them first
    // (by calculateStarts) ensures a higher chance that they will occupy a
    // standalone smem slot.
    std::sort(buffers.begin(), buffers.end(),
              [&](BufferT *A, BufferT *B) { return A->size > B->size; });

    calculateStarts(buffers);
    LLVM_DEBUG(dumpBuffers());

    // NOTE: The original paper doesn't consider interference between
    // the bumped ranges. Buffers that previously do not interfere with
    // could interfere after offset bumping if their liveness ranges overlap.
    // Therefore, we rerun the interference graph algorithm after bumping so
    // that we regroup the buffers and color them again. Since we always
    // increase the buffer offset and keep reducing conflicts, we will
    // eventually reach a fixed point.
    GraphT interference;
    buildInterferenceGraph(buffers, interference);
    do {
      allocate(buffers, interference);
      buildInterferenceGraph(buffers, interference);
    } while (!interference.empty());

    LLVM_DEBUG(dumpAllocationSize());
    // TODO: What is sharingGroup
    // Update allocation for sharingGroup.
    LLVM_DEBUG(dumpBuffers());
  }

  /// Computes the initial shared memory offsets.
  void calculateStarts(const SmallVector<BufferT *> &buffers) {
    //  v = values in shared memory
    //  t = triplet of (size, start, end)
    //  shared memory space
    //  -
    //  |         *******t4
    //  | /|\ v2 inserts t4, t5, and t6
    //  |  |
    //  | ******t5         ************t6
    //  | ^^^^^v2^^^^^^
    //  |  |      *********************t2
    //  | \|/ v2 erases t1
    //  | ******t1 ^^^^^^^^^v1^^^^^^^^^ ************t3
    //  |---------------------------------------------| liveness range
    //    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 ...
    // If the available triple's range is less than a given buffer range,
    // we won't know if there has been an overlap without using graph coloring.
    // Start -> Liveness Range
    using TripleMapT = std::multimap<size_t, Interval<size_t>>;
    TripleMapT tripleMap;
    tripleMap.insert(std::make_pair(0, Interval<size_t>()));
    SmallVector<BufferT *> xBuffers = buffers;
    while (!xBuffers.empty()) {
      auto tripleIt = tripleMap.begin();
      auto offset = tripleIt->first;
      auto range = tripleIt->second;
      tripleMap.erase(tripleIt);
      auto bufferIt =
          std::find_if(xBuffers.begin(), xBuffers.end(), [&](auto *buffer) {
            auto xRange = bufferRange[buffer];
            bool res = xRange.intersects(range);
            for (const auto &val : tripleMap)
              res = res &&
                    !val.second.intersects(xRange); // only one buffer intersect
            return res;
          });
      if (bufferIt != xBuffers.end()) {
        auto buffer = *bufferIt;
        auto xSize = buffer->size;
        auto xRange = bufferRange.lookup(buffer);
        // TODO(Keren): A buffer's size shouldn't be determined here, have to
        // clean it up
        size_t alignOffset = buffer->setOffsetAligned(offset);
        tripleMap.insert({alignOffset + xSize,
                          Interval{std::max(range.start(), xRange.start()),
                                   std::min(range.end(), xRange.end())}});
        // We could either insert (range.start, xRange.start) or (range.start,
        // xRange.end), both are correct and determine the potential buffer
        // offset, and the graph coloring algorithm will solve the interference,
        // if any
        if (range.start() < xRange.start())
          tripleMap.insert({offset, Interval{range.start(), xRange.end()}});
        if (xRange.end() < range.end())
          tripleMap.insert({offset, Interval{xRange.start(), range.end()}});
        xBuffers.erase(bufferIt);
      }
    }
  }

  /// Builds a graph of all shared memory values. Edges are created between
  /// shared memory values that are overlapping.
  void buildInterferenceGraph(const SmallVector<BufferT *> &buffers,
                              GraphT &interference) {
    // Reset interference graph
    interference.clear();
    for (auto x : buffers) {
      for (auto y : buffers) {
        if (x == y)
          continue;
        auto xStart = x->offset;
        auto yStart = y->offset;
        auto xSize = x->size;
        auto ySize = y->size;
        Interval xSizeRange = {xStart, xStart + xSize};
        Interval ySizeRange = {yStart, yStart + ySize};
        auto xOpRange = bufferRange.lookup(x);
        auto yOpRange = bufferRange.lookup(y);

        // Buffers interfere if their allocation offsets overlap and they are
        // live at the same time.
        if (xOpRange.intersects(yOpRange) &&
            xSizeRange.intersects(ySizeRange)) {
          interference[x].insert(y);
        }

        // TODO: Async
        // Buffers also interfere if their allocation offsets overlap and they
        // exist within regions that may execute simultaneously with respect to
        // each other.
        // if x and y belong to different regions (ignore producer region).
      }
    }

    LLVM_DEBUG(dumpInterferenceGraph(interference));
  }

  /// Finalizes shared memory offsets considering interference.
  void allocate(const SmallVector<BufferT *> &buffers,
                const GraphT &interference) {
    LDBG("\n------------ graph coloring ------------");

    // Reset shared memory size
    allocation->sharedMemorySize = 0;
    // First-fit graph coloring
    // Neighbors are nodes that interfere with each other.
    // We color a node by finding the index of the first available
    // non-neighboring node or the first neighboring node without any color.
    // Nodes with the same color do not interfere with each other.
    DenseMap<BufferT *, int> colors;
    for (auto value : buffers) {
      colors[value] = (value == buffers[0]) ? 0 : -1;
    }
    SmallVector<bool> available(buffers.size());
    for (auto x : buffers) {
      std::fill(available.begin(), available.end(), true);
      for (auto y : interference.lookup(x)) {
        int color = colors[y];
        if (color >= 0) {
          available[color] = false;
        }
      }
      auto it = std::find(available.begin(), available.end(), true);
      colors[x] = std::distance(available.begin(), it);
      LLVM_DEBUG({
        llvm::dbgs() << "-- color " << x->id << " " << colors[x] << "\n";
      });
    }
    LLVM_DEBUG({ llvm::dbgs() << "\n\n"; });

    // Finalize allocation
    // color0: [0, 7), [0, 8), [0, 15) -> [0, 7), [0, 8), [0, 15)
    // color1: [7, 9) -> [0 + 1 * 15, 9 + 1 * 15) -> [15, 24)
    // color2: [8, 12) -> [8 + 2 * 15, 12 + 2 * 15) -> [38, 42)
    // TODO(Keren): We are wasting memory here.
    // Nodes with color2 can actually start with 24.
    for (auto x : buffers) {
      size_t newOffset = 0;
      for (auto y : interference.lookup(x)) {
        newOffset = std::max(newOffset, y->offset + y->size);
      }
      if (colors.lookup(x) != 0)
        x->setOffsetAligned(newOffset);
      allocation->sharedMemorySize =
          std::max(allocation->sharedMemorySize, x->offset + x->size);
    }
    LLVM_DEBUG(dumpBuffers());
  }

private:
  Operation *operation;
  Allocation::FuncAllocMapT *funcAllocMap;
  Allocation *allocation;
  BufferRangeMapT bufferRange;
};

void Allocation::run(FuncAllocMapT &funcAllocMap) {
  triton::alloc::AllocationAnalysis(getOperation(), &funcAllocMap, this);
}

std::map<Operation *, SmallVector<Allocation::BufferId>>
Allocation::getLiveBuffers() {
  std::map<Operation *, SmallVector<BufferId>> liveBuffers;

  Operation *rootOperation = getOperation();
  Liveness liveness(rootOperation);
  auto analyzeOperation = [&](Operation *op) -> void {
    for (auto result : op->getOpResults()) {
      auto bufferId = getBufferId(result);
      if (bufferId == Allocation::InvalidBufferId)
        continue;
      auto liveOperations = liveness.resolveLiveness(result);
      for (auto depOp : liveOperations)
        liveBuffers[depOp].push_back(bufferId);
    }
  };
  rootOperation->walk(analyzeOperation);
  return liveBuffers;
}

} // namespace mlir::triton::alloc