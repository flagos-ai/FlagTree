#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "gtest/gtest.h"
#include <cstring>
#include <new>

namespace {
struct StatisticOwner
    : mlir::PassWrapper<StatisticOwner, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StatisticOwner)
  void runOnOperation() override {}
};

// The constructor is compiled in MLIRPass, while sizeof is evaluated here.
// Leave enough spare storage to detect a mismatched layout without corrupting
// the host heap. An empty pass manager cannot exercise this ABI boundary.
TEST(LLVMABI, PassStatisticConstructorRespectsConsumerLayout) {
  using Statistic = mlir::Pass::Statistic;
  StatisticOwner owner;
  alignas(Statistic) unsigned char storage[sizeof(Statistic) + 128];
  std::memset(storage, 0xa5, sizeof(storage));
  auto *stat =
      new (storage) Statistic(&owner, "abi-probe", "ABI boundary probe");
  ASSERT_EQ(owner.getStatistics().size(), 1u);
  EXPECT_EQ(owner.getStatistics().front(), stat);
  for (size_t i = sizeof(Statistic); i < sizeof(storage); ++i)
    EXPECT_EQ(storage[i], 0xa5) << "write beyond Statistic at byte " << i;
  *stat = 7;
#if LLVM_ENABLE_STATS
  EXPECT_EQ(stat->getValue(), 7u);
#endif
  stat->~Statistic();
}
} // namespace
