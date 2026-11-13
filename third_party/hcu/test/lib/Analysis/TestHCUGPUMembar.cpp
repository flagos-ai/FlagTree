#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include <algorithm>
#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "TritonHCUGPUToLLVM/MembarUtility.h"
#include "hcu/lib/TritonHCUGPUToLLVM/AsyncUtility.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"

using namespace mlir;

namespace {

struct TestHCUGPUMembarPass
    : public PassWrapper<TestHCUGPUMembarPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestHCUGPUMembarPass);

  StringRef getArgument() const final { return "test-tritonhcugpu-membar"; }
  StringRef getDescription() const final {
    return "print the result of the membar analysis as run in the hcugpu "
           "backend";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    triton::HCU::annotateLocalLoadsSyncedViaAsyncWait(moduleOp);
    // Print all ops after membar pass
    ModuleAllocation allocation(moduleOp);
    ModuleMembarAnalysis membarPass(&allocation, triton::HCU::membarFilter);
    membarPass.run();
  }
};


// Test-only entry directly exercises the private, read-only proof. In
// particular, do not insert a publication barrier here: missing/misplaced
// barriers must be rejected by the helper rather than repaired beforehand.
struct TestHCUAsyncSlotWARProofPass
    : public PassWrapper<TestHCUAsyncSlotWARProofPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestHCUAsyncSlotWARProofPass);

  StringRef getArgument() const final { return "test-hcu-async-slot-war-proof"; }
  StringRef getDescription() const final {
    return "record exact async-slot WAR pairs without mutating the input CFG";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto pairs = triton::HCU::prepareAsyncSlotPrefetchWAR(module);
    module->setAttr("test.hcu.async_slot_war_pairs",
                    IntegerAttr::get(IntegerType::get(&getContext(), 32),
                                     pairs.size()));
  }
};


// Direct entry for the opaque-B/ordinary-A prefix2 proof. In particular, do
// not repair missing publication barriers or run membar before this probe.
// Test-only: compare actual attribute identity across parse/print. Unit TPW
// is omitted from text, so a text-only roundtrip cannot catch a wrong default.
struct TestHCUMfmaParserRoundTripPass
    : public PassWrapper<TestHCUMfmaParserRoundTripPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestHCUMfmaParserRoundTripPass);

  StringRef getArgument() const final { return "test-hcu-mfma-parser-roundtrip"; }
  StringRef getDescription() const final {
    return "check five MFMA attribute identity roundtrips without codegen";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::gpu::TritonGPUDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    unsigned checked = 0, failures = 0;
    auto check = [&](triton::gpu::AMDMfmaEncodingAttr original) {
      ++checked;
      std::string spelling;
      llvm::raw_string_ostream os(spelling);
      os << original;
      auto parsed = dyn_cast_or_null<triton::gpu::AMDMfmaEncodingAttr>(
          parseAttribute(spelling, ctx));
      bool hasField = spelling.find("tilesPerWarp") != std::string::npos;
      bool equal = parsed && parsed == original &&
                   parsed.getTilesPerWarp() == original.getTilesPerWarp() &&
                   hasField == !original.hasUnitTilesPerWarp();
      if (!equal) {
        ++failures;
        getOperation().emitRemark()
            << "MFMA roundtrip case " << checked << ": rank="
            << original.getRank() << ", instrComponents="
            << original.getInstrShape().size() << ", expectedTPWSize="
            << original.getTilesPerWarp().size() << ", actualTPWSize="
            << (parsed ? parsed.getTilesPerWarp().size() : 0);
      }
    };
    for (const auto &warps :
         SmallVector<SmallVector<unsigned>>{{2, 2}, {1, 2, 2}}) {
      for (const auto &instr :
           SmallVector<SmallVector<unsigned>>{{16, 16}, {16, 16, 16}}) {
        auto cta = triton::gpu::CTAEncodingAttr::getDefault(ctx, warps.size());
        check(triton::gpu::AMDMfmaEncodingAttr::get(
            ctx, 3, warps, instr, false, cta));
      }
    }
    auto cta = triton::gpu::CTAEncodingAttr::getDefault(ctx, 2);
    check(triton::gpu::AMDMfmaEncodingAttr::get(
        ctx, 3, {2, 2}, {16, 16, 16}, false, cta, {2, 1}));
    auto i32 = IntegerType::get(ctx, 32);
    getOperation()->setAttr("test.hcu.mfma_roundtrip_checked",
                            IntegerAttr::get(i32, checked));
    getOperation()->setAttr("test.hcu.mfma_roundtrip_failures",
                            IntegerAttr::get(i32, failures));
    if (failures) {
      getOperation().emitError() << failures << "/" << checked
                                << " MFMA identity roundtrips failed";
      signalPassFailure();
    }
  }
};

// Test-only direct expander entry: no duplicate constant/trip classification.
// Pure addi bodies need no memory predication; the callback observes decisions.
struct TestHCUPipelineConstantProbePass
    : public PassWrapper<TestHCUPipelineConstantProbePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestHCUPipelineConstantProbePass);

  StringRef getArgument() const final {
    return "test-hcu-pipeline-constant-probe";
  }
  StringRef getDescription() const final {
    return "observe actual HCU pipelineForLoop predication and verify SSA";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verify(module)))
      return signalPassFailure();
    SmallVector<scf::ForOp> loops;
    module.walk([&](scf::ForOp loop) {
      if (loop->hasAttr("test.hcu.pipeline_probe"))
        loops.push_back(loop);
    });
    if (loops.empty()) {
      module.emitError("pipeline-probe fixture has no marked loop");
      return signalPassFailure();
    }
    unsigned failures = 0;
    SmallVector<Attribute> names;
    SmallVector<int64_t> callCounts;
    SmallVector<int32_t> accepted, modified, maxStages, carried;
    for (scf::ForOp loop : loops) {
      auto name = loop->getAttrOfType<StringAttr>("test.hcu.probe_name");
      auto expectSuccess =
          loop->getAttrOfType<BoolAttr>("test.hcu.expect_success");
      auto expectPredicated =
          loop->getAttrOfType<BoolAttr>("test.hcu.expect_predicated");
      auto supportDynamic =
          loop->getAttrOfType<BoolAttr>("test.hcu.support_dynamic");
      if (!name || !expectSuccess || !expectPredicated || !supportDynamic ||
          loop.getNumResults() != 1 || loop.getRegionIterArgs().size() != 1) {
        loop.emitError("pipeline-probe fixture needs expectations and one carry");
        return signalPassFailure();
      }
      std::vector<std::pair<Operation *, unsigned>> schedule;
      unsigned maxStage = 0;
      for (Operation &op : loop.getBody()->without_terminator()) {
        auto stage = op.getAttrOfType<IntegerAttr>("test.hcu.stage");
        if (!isa<arith::AddIOp>(&op) || op.getNumRegions() || !stage ||
            stage.getInt() < 0 || stage.getInt() > 2) {
          op.emitError("pipeline-probe only accepts addi with stage0..2");
          return signalPassFailure();
        }
        unsigned s = static_cast<unsigned>(stage.getInt());
        maxStage = std::max(maxStage, s);
        schedule.emplace_back(&op, s);
      }
      if (schedule.size() != 2 || maxStage == 0 ||
          schedule.front().second != 0 || schedule.back().second != maxStage) {
        loop.emitError("pipeline-probe requires stage0 and last-stage addi");
        return signalPassFailure();
      }
      int64_t predicateCalls = 0;
      triton::PipeliningOption options;
      options.supportDynamicLoops = supportDynamic.getValue();
      options.peelEpilogue = true;
      options.getScheduleFn =
          [schedule](scf::ForOp,
                     std::vector<std::pair<Operation *, unsigned>> &out) {
            out = schedule;
          };
      options.predicateFn = [&](RewriterBase &, Operation *op, Value pred) {
        ++predicateCalls;
        // The whitelist above admits only total, memory-free integer addi.
        // Keep the cloned operation; real pipeline live-out selects still run.
        return isa<arith::AddIOp>(op) && pred.getType().isInteger(1)
                   ? op
                   : nullptr;
      };
      bool changed = false;
      IRRewriter rewriter(loop);
      auto result = triton::pipelineForLoop(rewriter, loop, options, &changed);
      bool success = succeeded(result);
      bool ssaValid = succeeded(verify(module));
      // With exactly stage0 addi and last-stage addi, dynamic prologue and
      // epilogue each call the callback maxStage times; a static peeled loop
      // never calls it. This tests the real callbacks, not a trip-count clone.
      int64_t expectedCalls = expectPredicated.getValue() ? 2 * maxStage : 0;
      bool matches = success == expectSuccess.getValue() && ssaValid &&
                     (success || !changed) &&
                     (!success || predicateCalls == expectedCalls);
      if (!matches)
        ++failures;
      names.push_back(name);
      callCounts.push_back(predicateCalls);
      accepted.push_back(success);
      modified.push_back(changed);
      maxStages.push_back(maxStage);
      carried.push_back(success ? (*result).getRegionIterArgs().size() : 1);
      module.emitRemark()
          << "pipeline-probe " << name.getValue() << ": success=" << success
          << " expectedSuccess=" << expectSuccess.getValue()
          << " predicateCalls=" << predicateCalls
          << " expectedPredicated=" << expectPredicated.getValue()
          << " expectedCalls=" << expectedCalls
          << " maxStage=" << maxStage << " modified=" << changed
          << " ssa=" << ssaValid << " matches=" << matches;
    }
    auto *ctx = &getContext();
    auto i32 = IntegerType::get(ctx, 32);
    module->setAttr("test.hcu.pipeline_probe_checked",
                    IntegerAttr::get(i32, loops.size()));
    module->setAttr("test.hcu.pipeline_probe_failures",
                    IntegerAttr::get(i32, failures));
    module->setAttr("test.hcu.pipeline_probe_names", ArrayAttr::get(ctx, names));
    module->setAttr("test.hcu.pipeline_probe_calls",
                    DenseI64ArrayAttr::get(ctx, callCounts));
    module->setAttr("test.hcu.pipeline_probe_accepted",
                    DenseI32ArrayAttr::get(ctx, accepted));
    module->setAttr("test.hcu.pipeline_probe_modified",
                    DenseI32ArrayAttr::get(ctx, modified));
    module->setAttr("test.hcu.pipeline_probe_max_stages",
                    DenseI32ArrayAttr::get(ctx, maxStages));
    module->setAttr("test.hcu.pipeline_probe_carried",
                    DenseI32ArrayAttr::get(ctx, carried));
    if (failures) {
      module.emitError() << failures << "/" << loops.size()
                        << " pipeline-probe expectation mismatches";
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::test {
void registerTestHCUGPUMembarPass() {
  PassRegistration<TestHCUGPUMembarPass>();
  PassRegistration<TestHCUPipelineConstantProbePass>();
  PassRegistration<TestHCUMfmaParserRoundTripPass>();
  PassRegistration<TestHCUAsyncSlotWARProofPass>();
}
} // namespace mlir::test
