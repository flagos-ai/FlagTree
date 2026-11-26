#ifndef TSINGMICRO_MEMBAR_H
#define TSINGMICRO_MEMBAR_H

#include "Analysis/Allocation.h"
#include "Analysis/Utility.h"

#include <functional>
#include <map>
#include <set>

namespace mlir {
class OpBuilder;
}

namespace mlir::triton::membar {

/// Which dependency shape is being checked between two intersecting ops.
enum class MembarHazardKind {
  WriteRead, ///< writer op (lhs set) vs reader op (rhs set)
  ReadWrite, ///< reader op (lhs set) vs writer op (rhs set)
  WriteWrite ///< two writers
};

/// Return true to suppress a barrier between two ops even if their intervals
/// intersect. Operand roles follow `MembarHazardKind`.
using MembarFilterFn =
    std::function<bool(Operation *, Operation *, MembarHazardKind)>;

/// Ops that only compute addresses / metadata (memref shape, arith indices,
/// scf/cf control flow) and do not constitute a real SPM/DDR data access for
/// barrier insertion. `memref.load` / `memref.store` / `memref.copy` are not
/// included here.
bool isPureAddressOp(Operation *op);

struct BlockInfo {
  using IntervalT = triton::alloc::Interval<size_t>;
  using IntervalMapT = std::map<IntervalT, std::set<Operation *>>;

  IntervalMapT syncReadIntervals;
  IntervalMapT syncWriteIntervals;

  BlockInfo &join(const BlockInfo &other) {
    for (auto &interval : other.syncReadIntervals)
      syncReadIntervals[interval.first].insert(interval.second.begin(),
                                               interval.second.end());
    for (auto &interval : other.syncWriteIntervals)
      syncWriteIntervals[interval.first].insert(interval.second.begin(),
                                                interval.second.end());
    return *this;
  }

  void sync() {
    syncReadIntervals.clear();
    syncWriteIntervals.clear();
  }

  bool isIntersected(const BlockInfo &other, MembarFilterFn filter) const;

  bool operator==(const BlockInfo &other) const {
    return syncReadIntervals == other.syncReadIntervals &&
           syncWriteIntervals == other.syncWriteIntervals;
  }
  bool operator!=(const BlockInfo &other) const { return !(*this == other); }
};

/// Membar-like analysis for Tx81 IR using `mlir::triton::alloc::Allocation`.
/// Inserts `tx::BarrierOp` as needed.
class MembarAnalysis {
  using VirtualBlock = std::pair<Block *, Block::iterator>;

public:
  using FuncBlockInfoMapT = CallGraph<BlockInfo>::FuncDataMapT;

  MembarAnalysis() = default;
  explicit MembarAnalysis(triton::alloc::Allocation *allocation,
                          MembarFilterFn filter = nullptr)
      : allocation(allocation), filter(std::move(filter)) {}

  void run(FuncBlockInfoMapT &funcBlockInfoMap);

private:
  void resolve(FunctionOpInterface funcOp, FuncBlockInfoMapT *funcBlockInfoMap,
               OpBuilder *builder);
  void update(Operation *op, BlockInfo *blockInfo,
              FuncBlockInfoMapT *funcBlockInfoMap, OpBuilder *builder);
  void visitTerminator(Operation *op, SmallVector<VirtualBlock> &successors);
  void insertBarrier(Operation *op, OpBuilder *builder);

private:
  triton::alloc::Allocation *allocation = nullptr;
  MembarFilterFn filter = nullptr;
};

class ModuleMembarAnalysis : public CallGraph<BlockInfo> {
public:
  explicit ModuleMembarAnalysis(triton::alloc::ModuleAllocation *moduleAlloc,
                                MembarFilterFn filter = nullptr)
      : CallGraph<BlockInfo>(moduleAlloc->getModuleOp()),
        moduleAlloc(moduleAlloc), filter(std::move(filter)) {}

  void run() {
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
        [&](FunctionOpInterface funcOp) {
          auto *alloc = moduleAlloc->getFuncData(funcOp);
          auto [it, inserted] = funcMap.try_emplace(funcOp, BlockInfo());
          if (inserted && alloc) {
            MembarAnalysis analysis(alloc, filter);
            analysis.run(funcMap);
          }
        });
  }

private:
  triton::alloc::ModuleAllocation *moduleAlloc;
  MembarFilterFn filter;
};

} // namespace mlir::triton::membar

#endif // TSINGMICRO_MEMBAR_H
