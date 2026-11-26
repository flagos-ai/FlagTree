#include "AscendModel/Analysis/PipelineAnalysis.h"
#include "AscendModel/HardwareConfig.h"
#include "AscendModel/IR/AscendModelDialect.h"

#include <gtest/gtest.h>

#include <cstdint>

using mlir::ascend::DependencyGraph;
using mlir::ascend::HWUnit;
using mlir::ascend::PipelineOp;
using mlir::ascend::PipelineScheduler;

namespace {

PipelineOp makeOp(int64_t id, HWUnit unit, int64_t duration) {
  PipelineOp op;
  op.opId = id;
  op.hwUnit = unit;
  op.duration = duration;
  op.opName = "op" + std::to_string(id);
  return op;
}

} // namespace

TEST(CostModelPipelineSchedulerTest, DifferentHardwareUnitsRunInParallel) {
  PipelineScheduler scheduler;
  scheduler.addOperation(makeOp(/*id=*/1, HWUnit::Cube, /*duration=*/100));
  scheduler.addOperation(makeOp(/*id=*/2, HWUnit::Vector, /*duration=*/80));

  ASSERT_TRUE(scheduler.schedule());
  EXPECT_EQ(scheduler.getTotalCycles(), 100);

  const auto &ops = scheduler.getAllOps();
  ASSERT_EQ(ops.size(), 2u);
  EXPECT_EQ(ops[0].startCycle, 0);
  EXPECT_EQ(ops[1].startCycle, 0);
  EXPECT_EQ(ops[0].endCycle, 100);
  EXPECT_EQ(ops[1].endCycle, 80);
}

TEST(CostModelPipelineSchedulerTest, SameHardwareUnitSerializesOperations) {
  PipelineScheduler scheduler;
  scheduler.addOperation(makeOp(/*id=*/1, HWUnit::Vector, /*duration=*/40));
  scheduler.addOperation(makeOp(/*id=*/2, HWUnit::Vector, /*duration=*/60));

  ASSERT_TRUE(scheduler.schedule());
  EXPECT_EQ(scheduler.getTotalCycles(), 100);

  const auto &ops = scheduler.getAllOps();
  ASSERT_EQ(ops.size(), 2u);
  EXPECT_TRUE((ops[0].endCycle <= ops[1].startCycle) ||
              (ops[1].endCycle <= ops[0].startCycle));
  EXPECT_EQ(ops[0].duration + ops[1].duration, 100);
}

TEST(CostModelPipelineSchedulerTest, DependenciesDelayConsumersAcrossUnits) {
  PipelineScheduler scheduler;
  scheduler.addOperation(makeOp(/*id=*/1, HWUnit::Cube, /*duration=*/100));

  PipelineOp consumer = makeOp(/*id=*/2, HWUnit::Vector, /*duration=*/30);
  consumer.dependsOn.push_back(1);
  scheduler.addOperation(consumer);
  scheduler.addDependency(/*fromId=*/1, /*toId=*/2);

  ASSERT_TRUE(scheduler.schedule());
  EXPECT_EQ(scheduler.getTotalCycles(), 130);

  const auto &ops = scheduler.getAllOps();
  ASSERT_EQ(ops.size(), 2u);
  EXPECT_EQ(ops[0].startCycle, 0);
  EXPECT_EQ(ops[0].endCycle, 100);
  EXPECT_EQ(ops[1].startCycle, 100);
  EXPECT_EQ(ops[1].endCycle, 130);
}

TEST(CostModelPipelineSchedulerTest, RejectsCyclicDependencyGraph) {
  PipelineScheduler scheduler;
  PipelineOp first = makeOp(/*id=*/1, HWUnit::Cube, /*duration=*/10);
  first.dependsOn.push_back(2);
  PipelineOp second = makeOp(/*id=*/2, HWUnit::Vector, /*duration=*/10);
  second.dependsOn.push_back(1);
  scheduler.addOperation(first);
  scheduler.addOperation(second);
  scheduler.addDependency(/*fromId=*/1, /*toId=*/2);
  scheduler.addDependency(/*fromId=*/2, /*toId=*/1);

  EXPECT_FALSE(scheduler.schedule());
}

TEST(CostModelPipelineSchedulerTest, KernelCyclesApplyBarrierScalarAndWaves) {
  auto config = mlir::ascend::HardwareConfig::getDefault910B();
  ASSERT_NE(config, nullptr);

  PipelineScheduler scheduler(config.get());
  scheduler.addOperation(makeOp(/*id=*/1, HWUnit::Vector, /*duration=*/100));
  ASSERT_TRUE(scheduler.schedule());

  const int64_t expectedBarrierCycles =
      2 * config->getPipeBarrierCyclesPerIter();
  const int64_t expectedPerProgramCycles =
      static_cast<int64_t>((100 + expectedBarrierCycles) *
                           (1.0 + config->getAIVScalarOverheadFactor()));
  const int64_t expectedWaves = 3; // ceil(81 / 40)

  EXPECT_EQ(scheduler.getKernelCycles(/*numPrograms=*/81,
                                      /*numParallelUnits=*/40,
                                      /*numInnerIters=*/2),
            expectedPerProgramCycles * expectedWaves);
}

TEST(CostModelDependencyGraphTest, TopologicalOrderAndCycleDetection) {
  DependencyGraph graph;
  graph.addOp(/*opId=*/1, /*op=*/nullptr);
  graph.addOp(/*opId=*/2, /*op=*/nullptr);
  graph.addOp(/*opId=*/3, /*op=*/nullptr);
  graph.addDependency(/*fromId=*/1, /*toId=*/2);
  graph.addDependency(/*fromId=*/2, /*toId=*/3);

  EXPECT_FALSE(graph.hasCycle());
  auto deps = graph.getDependencies(/*opId=*/3);
  ASSERT_EQ(deps.size(), 1u);
  EXPECT_EQ(deps[0], 2);
  EXPECT_EQ(graph.getTopologicalOrder().size(), 3u);

  graph.addDependency(/*fromId=*/3, /*toId=*/1);
  EXPECT_TRUE(graph.hasCycle());
}

TEST(CostModelRooflineAnalyzerTest, ProducesReportAndJsonFromScheduledOps) {
  PipelineScheduler scheduler;
  scheduler.addOperation(makeOp(/*id=*/1, HWUnit::Vector, /*duration=*/40));
  scheduler.addOperation(makeOp(/*id=*/2, HWUnit::VecMTE2, /*duration=*/25));
  scheduler.addOperation(makeOp(/*id=*/3, HWUnit::MTE3, /*duration=*/10));
  ASSERT_TRUE(scheduler.schedule());

  mlir::ascend::RooflineAnalyzer analyzer(scheduler);
  auto report = analyzer.analyze();

  EXPECT_EQ(report.totalCycles, 40);
  EXPECT_EQ(report.unitBusyCycles[HWUnit::Vector], 40);
  EXPECT_EQ(report.unitBusyCycles[HWUnit::VecMTE2], 25);
  EXPECT_EQ(report.unitBusyCycles[HWUnit::MTE3], 10);
  EXPECT_EQ(report.bottleneckUnit, HWUnit::Vector);

  std::string json = report.toJSON();
  EXPECT_NE(json.find("\"total_cycles\""), std::string::npos);
  EXPECT_NE(json.find("\"unit_utilization\""), std::string::npos);
}
