// Copyright 2026 FlagOS Contributors

#ifndef TRITON_TLE_ANALYSIS_COMPATIBLE_PROPOSAL_MERGE_H
#define TRITON_TLE_ANALYSIS_COMPATIBLE_PROPOSAL_MERGE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>

namespace mlir::triton::tle {

/// Merge overlapping, compatible partial value-to-encoding assignments in the
/// original lexicographic pair order. Compatibility means that all shared
/// values have equal encodings; merge preserves every existing assignment.
template <typename Proposal, typename Compatible, typename Merge>
llvm::SmallVector<Proposal>
mergeCompatibleProposals(llvm::SmallVector<Proposal> proposals,
                         Compatible compatible, Merge merge) {
  using Assignment = typename Proposal::value_type;
  llvm::DenseMap<Assignment, llvm::SmallVector<size_t>> owners;
  llvm::SmallVector<bool> active(proposals.size(), true);
  for (size_t i = 0; i < proposals.size(); ++i)
    for (auto &entry : proposals[i])
      owners[entry].push_back(i);

  // Compatible proposals must share an equal value-to-encoding assignment.
  // Index assignments instead of values to exclude known encoding conflicts.
  // Equal ownership lists share a frontier, but compatibility still checks
  // every assignment in the proposals.
  // The lists remain immutable: a later active proposal has not reached its
  // own row yet and therefore cannot have gained keys through a merge.
  llvm::DenseMap<llvm::ArrayRef<size_t>, size_t> ownershipGroups;
  llvm::SmallVector<llvm::ArrayRef<size_t>> groups;
  llvm::DenseMap<Assignment, size_t> groupForAssignment;
  for (auto &entry : owners) {
    llvm::ArrayRef<size_t> list(entry.second);
    auto [group, inserted] = ownershipGroups.try_emplace(list, groups.size());
    if (inserted)
      groups.push_back(list);
    groupForAssignment.insert({entry.first, group->second});
  }

  struct Cursor {
    llvm::ArrayRef<size_t> owners;
    size_t position;
  };
  // Row epochs avoid allocating a node for every rejected pair. They also
  // suppress repeated candidates when a merge adds another ownership group.
  llvm::SmallVector<size_t> rejectedInRow(proposals.size(), 0);
  llvm::SmallVector<size_t> groupAddedInRow(groups.size(), 0);
  for (size_t i = 0; i < proposals.size(); ++i) {
    if (!active[i])
      continue;
    size_t row = i + 1;
    llvm::SmallVector<Cursor> cursors;
    auto addGroup = [&](const Assignment &assignment) {
      size_t group = groupForAssignment.find(assignment)->second;
      if (groupAddedInRow[group] == row)
        return;
      groupAddedInRow[group] = row;
      auto list = groups[group];
      size_t position =
          std::upper_bound(list.begin(), list.end(), i) - list.begin();
      if (position < list.size())
        cursors.push_back({list, position});
    };
    for (auto &entry : proposals[i])
      addGroup(entry);

    auto skipConsumed = [&](Cursor &cursor) {
      while (cursor.position < cursor.owners.size()) {
        size_t candidate = cursor.owners[cursor.position];
        if (active[candidate] && rejectedInRow[candidate] != row)
          break;
        ++cursor.position;
      }
    };
    auto mergeCandidate = [&](size_t j) {
      if (!compatible(proposals[i], proposals[j])) {
        // Adding assignments cannot resolve an existing conflict.
        rejectedInRow[j] = row;
        return;
      }
      active[j] = false;
      for (auto &entry : proposals[j])
        if (!proposals[i].count(entry.first))
          addGroup(entry);
      merge(proposals[i], proposals[j]);
    };
    while (!cursors.empty()) {
      if (cursors.size() == 1) {
        // Traverse a single ownership list directly. Re-enter the frontier
        // selection loop only if a union introduces a different list.
        auto list = cursors.front().owners;
        while (cursors.front().position < list.size()) {
          size_t j = list[cursors.front().position++];
          if (active[j] && rejectedInRow[j] != row)
            mergeCandidate(j);
          if (cursors.size() != 1)
            break;
        }
        if (cursors.size() == 1)
          break;
      }
      size_t j = proposals.size();
      for (auto &cursor : cursors) {
        skipConsumed(cursor);
        if (cursor.position < cursor.owners.size())
          j = std::min(j, cursor.owners[cursor.position]);
      }
      if (j == proposals.size())
        break;
      for (auto &cursor : cursors)
        if (cursor.position < cursor.owners.size() &&
            cursor.owners[cursor.position] == j)
          ++cursor.position;
      mergeCandidate(j);
    }
    // An exhausted earlier row cannot gain a compatible partner from a later
    // union: compatibility with that union would imply compatibility with an
    // overlapping constituent already checked in the earlier row.
  }

  llvm::SmallVector<Proposal> result;
  for (size_t i = 0; i < proposals.size(); ++i)
    if (active[i])
      result.push_back(std::move(proposals[i]));
  return result;
}

} // namespace mlir::triton::tle

#endif
