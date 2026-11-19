// Copyright 2026 FlagOS Contributors

#include "Analysis/CompatibleProposalMerge.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include <cstdlib>
#include <gtest/gtest.h>
#include <new>

namespace {
thread_local bool trackAllocations = false;
thread_local size_t allocationCount = 0;
} // namespace

// This replacement is confined to the unit-test executable. Count allocations
// only around the merge, excluding input construction and GoogleTest reporting.
void *operator new(std::size_t size) {
  if (trackAllocations)
    ++allocationCount;
  if (void *pointer = std::malloc(size ? size : 1))
    return pointer;
  throw std::bad_alloc();
}
void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept {
  std::free(pointer);
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete[](void *pointer) noexcept { ::operator delete(pointer); }
void operator delete[](void *pointer, std::size_t) noexcept {
  ::operator delete(pointer);
}

namespace mlir::triton::tle {
TEST(CompatibleProposalMerge, DenseConflictsDoNotAllocatePerPair) {
  using Proposal = llvm::MapVector<unsigned, unsigned>;
  llvm::SmallVector<Proposal> values;
  for (unsigned index = 0; index < 256; ++index) {
    Proposal value;
    value.insert({0, index});
    values.push_back(std::move(value));
  }
  auto compatible = [](const Proposal &lhs, const Proposal &rhs) {
    return lhs.lookup(0) == rhs.lookup(0);
  };
  auto merge = [](Proposal &dst, const Proposal &src) {
    for (auto entry : src)
      dst.insert(entry);
  };
  allocationCount = 0;
  trackAllocations = true;
  auto result = mergeCompatibleProposals(values, compatible, merge);
  trackAllocations = false;
  ASSERT_EQ(result.size(), values.size());
  for (size_t index = 0; index < result.size(); ++index)
    EXPECT_EQ(result[index].getArrayRef(), values[index].getArrayRef());
  EXPECT_LE(allocationCount, values.size() * 16);
}
} // namespace mlir::triton::tle
