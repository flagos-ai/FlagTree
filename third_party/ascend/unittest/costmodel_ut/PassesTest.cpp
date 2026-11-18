#include "AscendModel/Transforms/Passes.h"
#include "AscendModel/IR/AscendModelDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>

using mlir::ModuleOp;
using mlir::Operation;
using mlir::OwningOpRef;
using mlir::Pass;
using mlir::PassManager;
using mlir::StringAttr;
using mlir::ascend::createAssignOpIDsPass;
using mlir::ascend::createEstimateCyclesPass;
using mlir::ascend::createPerfReportPass;
using mlir::ascend::createPipelineAnalysisPass;
using mlir::ascend::EstimateCyclesPassOptions;

namespace {

constexpr const char *kVectorModule = R"mlir(
module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = ascend.vector_load %arg0 {bytes = 16 : i64} : tensor<4xf32> -> tensor<4xf32>
    %1 = ascend.add %0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    ascend.vector_store %1 {bytes = 16 : i64} : tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
)mlir";

void registerDialects(mlir::MLIRContext &context) {
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::ascend::AscendModelDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
}

OwningOpRef<ModuleOp> parseModule(mlir::MLIRContext &context,
                                  llvm::StringRef source) {
  registerDialects(context);
  return mlir::parseSourceString<ModuleOp>(source, &context);
}

template <typename... PassTs>
bool runPasses(ModuleOp module, PassTs &&...passes) {
  PassManager pm(module.getContext());
  (pm.addPass(std::forward<PassTs>(passes)), ...);
  return mlir::succeeded(pm.run(module));
}

Operation *findFirstOp(ModuleOp module, llvm::StringRef name) {
  Operation *result = nullptr;
  module.walk([&](Operation *op) {
    if (!result && op->getName().getStringRef() == name)
      result = op;
  });
  return result;
}

int64_t getI64Attr(Operation *op, llvm::StringRef name) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
  return attr ? attr.getInt() : -1;
}

} // namespace

TEST(CostModelPassesTest, AssignOpIDsPassAnnotatesAscendOpsOnly) {
  mlir::MLIRContext context;
  auto module = parseModule(context, R"mlir(
module {
  func.func @main(%arg0: i32, %arg1: i32, %arg2: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = arith.addi %arg0, %arg1 : i32
    %0 = ascend.add %arg2, %arg2 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  ASSERT_TRUE(runPasses(*module, createAssignOpIDsPass()));

  auto totalOps = module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
      "ascend.total_ops");
  ASSERT_TRUE(totalOps);
  EXPECT_EQ(totalOps.getInt(), 1);

  Operation *addOp = findFirstOp(*module, "ascend.add");
  ASSERT_NE(addOp, nullptr);
  EXPECT_EQ(getI64Attr(addOp, "op_id"), 0);

  Operation *arithOp = findFirstOp(*module, "arith.addi");
  ASSERT_NE(arithOp, nullptr);
  EXPECT_FALSE(arithOp->hasAttr("op_id"));
}

TEST(CostModelPassesTest, EstimateCyclesAnnotatesComputeAndTransferOps) {
  mlir::MLIRContext context;
  auto module = parseModule(context, kVectorModule);
  ASSERT_TRUE(module);

  ASSERT_TRUE(runPasses(*module, createEstimateCyclesPass()));

  Operation *loadOp = findFirstOp(*module, "ascend.vector_load");
  Operation *addOp = findFirstOp(*module, "ascend.add");
  Operation *storeOp = findFirstOp(*module, "ascend.vector_store");
  ASSERT_NE(loadOp, nullptr);
  ASSERT_NE(addOp, nullptr);
  ASSERT_NE(storeOp, nullptr);

  EXPECT_GT(getI64Attr(loadOp, "estimated_cycles"), 0);
  EXPECT_GT(getI64Attr(addOp, "estimated_cycles"), 0);
  EXPECT_GT(getI64Attr(storeOp, "estimated_cycles"), 0);
  EXPECT_EQ(getI64Attr(loadOp, "bytes"), 16);
  EXPECT_EQ(getI64Attr(storeOp, "bytes"), 16);
  EXPECT_EQ(getI64Attr(addOp, "flops"), 4);
  EXPECT_TRUE(loadOp->getAttrOfType<StringAttr>("hw_unit"));
  EXPECT_TRUE(addOp->getAttrOfType<StringAttr>("hw_unit"));
  EXPECT_TRUE(storeOp->getAttrOfType<StringAttr>("hw_unit"));
}

TEST(CostModelPassesTest, EstimateCyclesReportsInvalidArgBindings) {
  mlir::MLIRContext context;
  auto module = parseModule(context, kVectorModule);
  ASSERT_TRUE(module);

  EstimateCyclesPassOptions options;
  options.argBindingsStr = "arg0";

  EXPECT_FALSE(runPasses(*module, createEstimateCyclesPass(options)));
}

TEST(CostModelPassesTest, PipelineAnalysisSetsCycleSummaryAttrs) {
  mlir::MLIRContext context;
  auto module = parseModule(context, kVectorModule);
  ASSERT_TRUE(module);

  ASSERT_TRUE(runPasses(*module, createAssignOpIDsPass(),
                        createEstimateCyclesPass(),
                        createPipelineAnalysisPass()));

  auto scheduled = module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
      "ascend.scheduled_cycles_one_iter");
  auto roofline = module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
      "ascend.roofline_cycles");
  auto simple = module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
      "ascend.simple_sum_cycles");
  ASSERT_TRUE(scheduled);
  ASSERT_TRUE(roofline);
  ASSERT_TRUE(simple);
  EXPECT_GT(scheduled.getInt(), 0);
  EXPECT_GT(roofline.getInt(), 0);
  EXPECT_GT(simple.getInt(), 0);
}

TEST(CostModelPassesTest, PerfReportPassAcceptsEstimatedPipeline) {
  mlir::MLIRContext context;
  auto module = parseModule(context, kVectorModule);
  ASSERT_TRUE(module);

  EXPECT_TRUE(runPasses(*module, createAssignOpIDsPass(),
                        createEstimateCyclesPass(),
                        createPipelineAnalysisPass(), createPerfReportPass()));
}
