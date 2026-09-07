// Copyright 2026 FlagOS Contributors

#include "Analysis/CompatibleProposalMerge.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include <gtest/gtest.h>
#include <initializer_list>

namespace mlir::triton::tle {
namespace {

struct Proposal : llvm::MapVector<unsigned, unsigned> {
  Proposal(std::initializer_list<std::pair<unsigned, unsigned>> entries = {}) {
    for (auto entry : entries)
      insert(entry);
  }
  bool operator==(const Proposal &other) const {
    return getArrayRef() == other.getArrayRef();
  }
};

bool compatible(const Proposal &lhs, const Proposal &rhs) {
  bool overlap = false;
  for (auto [key, value] : lhs) {
    auto found = rhs.find(key);
    if (found == rhs.end())
      continue;
    overlap = true;
    if (found->second != value)
      return false;
  }
  return overlap;
}

void merge(Proposal &dst, const Proposal &src) {
  for (auto entry : src)
    dst.insert(entry);
}

llvm::SmallVector<Proposal> referenceMerge(llvm::SmallVector<Proposal> values) {
  bool merged = true;
  while (merged) {
    merged = false;
    for (size_t i = 0; i < values.size() && !merged; ++i)
      for (size_t j = i + 1; j < values.size(); ++j)
        if (compatible(values[i], values[j])) {
          merge(values[i], values[j]);
          values.erase(values.begin() + j);
          merged = true;
          break;
        }
  }
  return values;
}

void expectSame(const llvm::SmallVector<Proposal> &values) {
  EXPECT_EQ(mergeCompatibleProposals(values, compatible, merge),
            referenceMerge(values));
}

TEST(CompatibleProposalMerge, DisjointProposalsDoNotRequirePairwiseComparison) {
  llvm::SmallVector<Proposal> values;
  for (unsigned i = 0; i < 256; ++i)
    values.push_back(Proposal{{i, 0}});
  unsigned calls = 0;
  auto result = mergeCompatibleProposals(values, [&](const auto &lhs, const auto &rhs) {
    ++calls;
    return compatible(lhs, rhs);
  }, merge);
  EXPECT_EQ(result, values);
  EXPECT_LE(calls, values.size());
}

TEST(CompatibleProposalMerge, LexicographicConflictResolution) {
  // Compatibility is not transitive: a connected-component union is invalid.
  expectSame({Proposal{{0, 0}}, Proposal{{1, 0}},
              Proposal{{1, 0}, {2, 1}}, Proposal{{0, 0}, {2, 2}}});
}

TEST(CompatibleProposalMerge, NewlyOverlappingEarlierCandidate) {
  expectSame({Proposal{{0, 0}}, Proposal{{1, 0}}, Proposal{{2, 0}},
              Proposal{{0, 0}, {1, 0}}, Proposal{{1, 0}, {2, 0}}});
}

TEST(CompatibleProposalMerge, EmptyDuplicateAndConflictingAssignments) {
  expectSame({});
  expectSame({Proposal{}, Proposal{{1, 0}}, Proposal{}, Proposal{{1, 0}},
              Proposal{{1, 1}}, Proposal{{1, 1}, {2, 1}}});
}

TEST(CompatibleProposalMerge, DeterministicOverlappingAssignmentGraphs) {
  unsigned state = 54321;
  for (unsigned trial = 0; trial < 100; ++trial) {
    llvm::SmallVector<Proposal> values;
    for (unsigned i = 0; i < 32; ++i) {
      Proposal value;
      for (unsigned j = 0; j < (i % 5); ++j) {
        state = state * 1664525u + 1013904223u;
        unsigned key = (state >> 8) % 12;
        state = state * 1664525u + 1013904223u;
        value.insert({key, (state >> 8) % 3});
      }
      values.push_back(std::move(value));
    }
    expectSame(values);
  }
}

} // namespace
} // namespace mlir::triton::tle
