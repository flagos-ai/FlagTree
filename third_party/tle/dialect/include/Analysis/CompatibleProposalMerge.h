// Copyright 2026 FlagOS Contributors

#ifndef TRITON_TLE_ANALYSIS_COMPATIBLE_PROPOSAL_MERGE_H
#define TRITON_TLE_ANALYSIS_COMPATIBLE_PROPOSAL_MERGE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <set>

namespace mlir::triton::tle {

/// Merge overlapping, compatible partial value-to-encoding assignments in the
/// original lexicographic pair order. Compatibility means that all shared
/// values have equal encodings; merge preserves every existing assignment.
template <typename Proposal, typename Compatible, typename Merge>
llvm::SmallVector<Proposal>
mergeCompatibleProposals(llvm::SmallVector<Proposal> proposals,
                         Compatible compatible, Merge merge) {
  using Key = typename Proposal::key_type;
  llvm::DenseMap<Key, llvm::SmallVector<size_t>> owners;
  llvm::SmallVector<bool> active(proposals.size(), true);
  for (size_t i = 0; i < proposals.size(); ++i)
    for (auto &entry : proposals[i])
      owners[entry.first].push_back(i);

  for (size_t i = 0; i < proposals.size(); ++i) {
    if (!active[i])
      continue;
    std::set<size_t> candidates;
    llvm::DenseSet<size_t> rejected;
    auto addCandidates = [&](const Key &key) {
      for (size_t owner : owners.find(key)->second)
        if (owner > i && active[owner] && !rejected.contains(owner))
          candidates.insert(owner);
    };
    for (auto &entry : proposals[i])
      addCandidates(entry.first);

    while (!candidates.empty()) {
      size_t j = *candidates.begin();
      candidates.erase(candidates.begin());
      if (!active[j])
        continue;
      if (!compatible(proposals[i], proposals[j])) {
        // Conflicting assignments cannot become compatible when more
        // assignments are added. No need to compare this pair again.
        rejected.insert(j);
        continue;
      }
      active[j] = false;
      for (auto &entry : proposals[j])
        if (!proposals[i].count(entry.first)) {
          addCandidates(entry.first);
          owners[entry.first].push_back(i);
        }
      merge(proposals[i], proposals[j]);
    }
    // An exhausted earlier row cannot gain a compatible partner from a later
    // union: compatibility with that union would imply compatibility with at
    // least one of its overlapping constituents, already checked in this row.
    // Thus we may advance i while preserving the original first-pair order.
  }

  llvm::SmallVector<Proposal> result;
  for (size_t i = 0; i < proposals.size(); ++i)
    if (active[i])
      result.push_back(std::move(proposals[i]));
  return result;
}

} // namespace mlir::triton::tle

#endif
