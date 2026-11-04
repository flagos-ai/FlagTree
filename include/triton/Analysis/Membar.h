/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_ANALYSIS_MEMBAR_H
#define TRITON_ANALYSIS_MEMBAR_H

#include "Allocation.h"

#ifdef __TLE__
#include <limits>
#endif
#include <set>

namespace mlir {

class OpBuilder;

/// Callback to allow backend to provide more information on whether a barrier
/// is needed between two operations. Even though two operations access the same
/// shared memory they may not require a barrier in between them.
using MembarFilterFn = std::function<bool(Operation *, Operation *)>;

struct BlockInfo {
  using IntervalMapT = std::map<Interval<size_t>, std::set<Operation *>>;

  IntervalMapT syncReadIntervals;
  IntervalMapT syncWriteIntervals;
#ifdef __TLE__
  /// Some path from the function entry reaches this point without a CTA
  /// barrier. Only MembarAnalysis sets it; it tracks a callee's entry prefix.
  bool entryUnsynced = false;
#endif

  BlockInfo() = default;

  /// Unions two BlockInfo objects.
  BlockInfo &join(const BlockInfo &other) {
    for (auto &interval : other.syncReadIntervals)
      syncReadIntervals[interval.first].insert(interval.second.begin(),
                                               interval.second.end());
    for (auto &interval : other.syncWriteIntervals)
      syncWriteIntervals[interval.first].insert(interval.second.begin(),
                                                interval.second.end());
#ifdef __TLE__
    entryUnsynced |= other.entryUnsynced;
#endif
    return *this;
  }

#ifdef __TLE__
  /// Shifts a callee's intervals into the caller's frame at `offset`.
  BlockInfo translated(size_t offset) const {
    auto shift = [&](const IntervalMapT &src, IntervalMapT &dst) {
      constexpr size_t kMax = std::numeric_limits<size_t>::max();
      for (auto &[interval, ops] : src) {
        // The all-shared-memory interval (e.g. an elected mbarrier arrive)
        // also covers the caller's allocations below the callee frame.
        if (interval == Interval<size_t>()) {
          dst[interval].insert(ops.begin(), ops.end());
          continue;
        }
        size_t end = interval.end() == kMax ? kMax : interval.end() + offset;
        dst[Interval<size_t>(interval.start() + offset, end)].insert(
            ops.begin(), ops.end());
      }
    };
    BlockInfo result;
    shift(syncReadIntervals, result.syncReadIntervals);
    shift(syncWriteIntervals, result.syncWriteIntervals);
    result.entryUnsynced = entryUnsynced;
    return result;
  }
#endif

  void dump() {
    auto &err = llvm::errs();
    err << "Block Interval:\n";
    err << "  Read Intervals:\n";
    for (auto &[interval, ops] : syncReadIntervals) {
      err << "    [" << interval.start() << ", " << interval.end() << "] ";
      for (auto &op : ops)
        err << op->getName() << " ";
      err << "\n";
    }
    err << "  Write Intervals:\n";
    for (auto &[interval, ops] : syncWriteIntervals) {
      err << "    [" << interval.start() << ", " << interval.end() << "] ";
      for (auto &op : ops)
        err << op->getName() << " ";
      err << "\n";
    }
  }

  /// Returns true if intervals in two BlockInfo objects are intersected.
  bool isIntersected(const BlockInfo &other, MembarFilterFn filter) const {
    return /*RAW*/ isIntersected(syncWriteIntervals, other.syncReadIntervals,
                                 filter) ||
           /*WAR*/
           isIntersected(syncReadIntervals, other.syncWriteIntervals, filter) ||
           /*WAW*/
           isIntersected(syncWriteIntervals, other.syncWriteIntervals, filter);
  }

  /// Clears the intervals because a barrier is inserted.
  void sync() {
    syncReadIntervals.clear();
    syncWriteIntervals.clear();
#ifdef __TLE__
    entryUnsynced = false;
#endif
  }

  /// Compares two BlockInfo objects.
  bool operator==(const BlockInfo &other) const {
#ifdef __TLE__
    if (entryUnsynced != other.entryUnsynced)
      return false;
#endif
    return syncReadIntervals == other.syncReadIntervals &&
           syncWriteIntervals == other.syncWriteIntervals;
  }

  bool operator!=(const BlockInfo &other) const { return !(*this == other); }

private:
  bool isIntersected(const IntervalMapT &lhsIntervalSet,
                     const IntervalMapT &rhsIntervalSet,
                     MembarFilterFn filter) const {
    for (auto &lhs : lhsIntervalSet)
      for (auto &rhs : rhsIntervalSet)
        if (lhs.first.intersects(rhs.first))
          for (auto lhsOp : lhs.second)
            for (auto rhsOp : rhs.second)
              if (!filter || !filter(lhsOp, rhsOp))
                return true;

    return false;
  }
};

//===----------------------------------------------------------------------===//
// Shared Memory Barrier Analysis
//===----------------------------------------------------------------------===//

// Common class to analyze membar and fence placement.
class MembarOrFenceAnalysis {
  using VirtualBlock = std::pair<Block *, Block::iterator>;

public:
  using FuncBlockInfoMapT = CallGraph<BlockInfo>::FuncDataMapT;
  /// Creates a new Membar analysis that generates the shared memory barrier
  /// in the following circumstances:
  /// - RAW: If a shared memory write is followed by a shared memory read, and
  /// their addresses are intersected, a barrier is inserted.
  /// - WAR: If a shared memory read is followed by a shared memory write, and
  /// their addresses are intersected, a barrier is inserted.
  /// The following circumstances do not require a barrier:
  /// - WAW: not possible because overlapped memory allocation is not allowed.
  /// - RAR: no write is performed.
  /// Temporary storage of operations such as Reduce are considered as both
  /// a shared memory read. If the temporary storage is written but not read,
  /// it is considered as the problem of the operation itself but not the membar
  /// analysis.
  MembarOrFenceAnalysis() = default;
  explicit MembarOrFenceAnalysis(Allocation *allocation, MembarFilterFn filter)
      : allocation(allocation), filter(filter) {}

  virtual ~MembarOrFenceAnalysis() = default;

  /// Runs the membar analysis to the given operation, inserts a barrier if
  /// necessary.
  void run(FuncBlockInfoMapT &funcBlockInfoMap);

protected:
  /// Applies the barrier analysis based on the SCF dialect, in which each
  /// region has a single basic block only.
  /// Example:
  /// region1
  ///   op1
  ///   op2 (scf.if)
  ///      region2
  ///        op3
  ///        op4
  ///      region3
  ///        op5
  ///        op6
  ///   op7
  /// TODO: Explain why we don't use ForwardAnalysis:
  void resolve(FunctionOpInterface funcOp, FuncBlockInfoMapT *funcBlockInfoMap,
               OpBuilder *builder);

  /// Collects the successors of the terminator
  void visitTerminator(Operation *operation,
                       SmallVector<VirtualBlock> &successors);

  /// Updates the BlockInfo operation based on the operation.
  virtual void update(Operation *operation, BlockInfo *blockInfo,
                      FuncBlockInfoMapT *funcBlockInfoMap,
                      OpBuilder *builder) = 0;

#ifdef __TLE__
  /// Whether the function entry starts with BlockInfo::entryUnsynced set.
  virtual bool tracksEntrySync() const { return false; }
#endif

  Allocation *allocation = nullptr;
  MembarFilterFn filter = nullptr;
};

class MembarAnalysis : public MembarOrFenceAnalysis {
public:
#ifdef __TLE__
  /// Per function: the shared-memory accesses that can run before the first
  /// CTA barrier on some path from the function entry.
  using EntryPrefixMapT = DenseMap<FunctionOpInterface, BlockInfo>;
#endif

  MembarAnalysis() = default;
#ifdef __TLE__
  explicit MembarAnalysis(Allocation *allocation, MembarFilterFn filter,
                          EntryPrefixMapT *entryPrefixMap = nullptr)
      : MembarOrFenceAnalysis(allocation, filter),
        entryPrefixMap(entryPrefixMap) {}
#else
  explicit MembarAnalysis(Allocation *allocation, MembarFilterFn filter)
      : MembarOrFenceAnalysis(allocation, filter) {}
#endif

  ~MembarAnalysis() override = default;

private:
  /// Updates the BlockInfo operation based on the operation.
  virtual void update(Operation *operation, BlockInfo *blockInfo,
                      FuncBlockInfoMapT *funcBlockInfoMap,
                      OpBuilder *builder) override;

#ifdef __TLE__
  bool tracksEntrySync() const override { return entryPrefixMap != nullptr; }

  void updateCall(Operation *op, FunctionOpInterface callee,
                  BlockInfo *blockInfo, FuncBlockInfoMapT *funcBlockInfoMap,
                  MembarFilterFn filter, OpBuilder *builder);

  void addToEntryPrefix(const BlockInfo &blockInfo);
#endif

  void insertBarrier(Operation *operation, OpBuilder *builder);

#ifdef __TLE__
  EntryPrefixMapT *entryPrefixMap = nullptr;
#endif
};

/// Postorder traversal on the callgraph to insert membar instructions
/// of each function.
/// Each function maintains a BlockInfo map that includes all potential buffers
/// after returning. This way users do not have to explicitly insert membars
/// before and after function calls, but might be a bit conservative.
template <typename AnalysisType>
class ModuleMembarOrFenceAnalysis : public CallGraph<BlockInfo> {
public:
  ModuleMembarOrFenceAnalysis(ModuleAllocation *moduleAllocation,
                              MembarFilterFn filter = nullptr)
      : CallGraph<BlockInfo>(moduleAllocation->getModuleOp()),
        moduleAllocation(moduleAllocation), filter(filter) {}

  void run() {
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        // Pre-order walk callback
        [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
        // Post-order walk callback
        [&](FunctionOpInterface funcOp) {
          auto *allocation = moduleAllocation->getFuncData(funcOp);
          auto [it, inserted] = funcMap.try_emplace(funcOp, BlockInfo());
          if (inserted) {
            AnalysisType analysis(allocation, filter);
            analysis.run(funcMap);
          }
        });
  }

private:
  ModuleAllocation *moduleAllocation;
  MembarFilterFn filter;
};

#ifdef __TLE__
/// Like ModuleMembarOrFenceAnalysis<MembarAnalysis>, but also keeps each
/// callee's entry prefix so a call is ordered against what the callee touches
/// before its first barrier, not only against its exit state.
class ModuleMembarAnalysis : public CallGraph<BlockInfo> {
public:
  ModuleMembarAnalysis(ModuleAllocation *moduleAllocation,
                       MembarFilterFn filter = nullptr)
      : CallGraph<BlockInfo>(moduleAllocation->getModuleOp()),
        moduleAllocation(moduleAllocation), filter(filter) {}

  void run() {
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        // Pre-order walk callback
        [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
        // Post-order walk callback
        [&](FunctionOpInterface funcOp) {
          auto *allocation = moduleAllocation->getFuncData(funcOp);
          auto [it, inserted] = funcMap.try_emplace(funcOp, BlockInfo());
          if (inserted) {
            MembarAnalysis analysis(allocation, filter, &entryPrefixMap);
            analysis.run(funcMap);
          }
        });
  }

private:
  ModuleAllocation *moduleAllocation;
  MembarFilterFn filter;
  MembarAnalysis::EntryPrefixMapT entryPrefixMap;
};
#else
typedef ModuleMembarOrFenceAnalysis<MembarAnalysis> ModuleMembarAnalysis;
#endif

} // namespace mlir

#endif // TRITON_ANALYSIS_MEMBAR_H
