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

namespace mlir::triton::HCU {
LogicalResult verifyLegacyMlsV0ClosedModule(ModuleOp mod, StringRef arch);
}

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
struct TestHCUPrefix2WARProofPass
    : public PassWrapper<TestHCUPrefix2WARProofPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestHCUPrefix2WARProofPass);
  StringRef getArgument() const final { return "test-hcu-prefix2-war-proof"; }
  StringRef getDescription() const final {
    return "observe exact v0-prefix2 WAR pairs and actual allocation without CFG changes";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto text = [&]() {
      std::string str;
      llvm::raw_string_ostream os(str);
      module.print(os);
      os.flush();
      return str;
    };
    auto before = text();
    ModuleAllocation allocation(module);
    auto b = triton::HCU::getVerifiedLegacyV0Prefix2Queue(module);
    auto pairs = triton::HCU::prepareLegacyV0Prefix2WAR(module, allocation);
    if (before != text()) {
      module.emitError("prefix2 proof/allocation changed input IR");
      return signalPassFailure();
    }
    Builder builder(&getContext());
    SmallVector<Attribute> pairRecords, roots;
    for (auto [read, copy] : pairs) {
      auto id = [&](Operation *op) -> Attribute {
        if (auto name = op->getAttrOfType<StringAttr>("test.hcu.op_id"))
          return name;
        return builder.getStringAttr(op->getName().getStringRef());
      };
      pairRecords.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("read", id(read)),
          builder.getNamedAttr("copy", id(copy))}));
    }
    module.walk([&](triton::gpu::LocalAllocOp root) {
      auto fn = root->getParentOfType<triton::FuncOp>();
      if (!fn) return;
      auto *data = allocation.getFuncData(cast<FunctionOpInterface>(fn.getOperation()));
      auto id = data->getBufferId(root.getResult());
      if (id == Allocation::InvalidBufferId) return;
      auto interval = data->getAllocatedInterval(id);
      auto name = root->getAttrOfType<StringAttr>("test.hcu.op_id");
      auto offset = root->getAttrOfType<IntegerAttr>("allocation.offset");
      roots.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("root", name ? name : builder.getStringAttr("unnamed")),
          builder.getNamedAttr("buffer_id", builder.getI64IntegerAttr(id)),
          builder.getNamedAttr("begin", builder.getI64IntegerAttr(interval.start())),
          builder.getNamedAttr("end", builder.getI64IntegerAttr(interval.end())),
          builder.getNamedAttr("bytes", builder.getI64IntegerAttr(data->getAllocatedSize(id))),
          builder.getNamedAttr("emitted_offset", builder.getI64IntegerAttr(offset ? offset.getInt() : -1))}));
    });
    module->setAttr("test.hcu.prefix2_war_pairs", builder.getI32IntegerAttr(pairs.size()));
    module->setAttr("test.hcu.prefix2_war_exact_pairs", builder.getArrayAttr(pairRecords));
    module->setAttr("test.hcu.prefix2_war_roots", builder.getArrayAttr(roots));
    module->setAttr("test.hcu.prefix2_war_total", builder.getI64IntegerAttr(allocation.getSharedMemorySize()));
    module->setAttr("test.hcu.prefix2_B_verified", builder.getBoolAttr(bool(b)));
    module->setAttr("test.hcu.prefix2_proof_input_unchanged", builder.getBoolAttr(true));
  }
};

// Direct preflight only. Do not call allocation, insert publication barriers,
// run membar, mutate CFG or enter any LLVM conversion. The existing verifier
// selects the separate LP1 proof only under its explicit private capture hint.
struct TestHCULegacyV0LocalPrefetchProofPass
    : public PassWrapper<TestHCULegacyV0LocalPrefetchProofPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestHCULegacyV0LocalPrefetchProofPass);
  StringRef getArgument() const final { return "test-hcu-legacy-v0-local-prefetch-proof"; }
  StringRef getDescription() const final { return "verify the private two-slot LP1 packet queue without lowering"; }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto hint = module->getAttrOfType<IntegerAttr>("ttg.hcu.v0_mls_local_prefetch_capture");
    if (!hint || !hint.getType().isInteger(32) || hint.getInt() != 1) {
      module.emitError("LP1 proof test requires explicit capture hint1");
      return signalPassFailure();
    }
    if (failed(triton::HCU::verifyLegacyMlsV0ClosedModule(module, "gfx936")))
      return signalPassFailure();
  }
};

// Artifact-only private fixture construction. HCUMlsShared's existing parser
// embeds a CTAEncodingAttr which has no standalone textual mnemonic. Build the
// existing attribute in memory, without widening the production parser/API.
// Direct, read-only LP1 exact-RAW getter probe. Never repair publication.
struct TestHCULegacyV0LocalPrefetchRAWPass
    : public PassWrapper<TestHCULegacyV0LocalPrefetchRAWPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestHCULegacyV0LocalPrefetchRAWPass);
  TestHCULegacyV0LocalPrefetchRAWPass() = default;
  TestHCULegacyV0LocalPrefetchRAWPass(const TestHCULegacyV0LocalPrefetchRAWPass &other)
      : PassWrapper(other) {}
  Option<bool> scheduleReadFirst{*this, "schedule-read-first",
      llvm::cl::desc("also exercise the private scheduler with these exact getter pairs"),
      llvm::cl::init(false)};
  Option<bool> observeAWAR{*this, "observe-a-war",
      llvm::cl::desc("observe the independent N4 A WAR getter without changing IR"),
      llvm::cl::init(false)};
  Option<bool> observeIndexedABWAR{*this, "observe-indexed-ab-war",
      llvm::cl::desc("observe exact paired indexed A/B WARs without repair"), llvm::cl::init(false)};
  Option<std::string> indexedABWARMutation{*this, "indexed-ab-war-mutation",
      llvm::cl::desc("test-only none/seed/backcarry/missing-head/late-head/read-after-copy"), llvm::cl::init("none")};
  StringRef getArgument() const final { return "test-hcu-legacy-v0-local-prefetch-raw"; }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto text = [&]() -> std::string {
      std::string s;
      llvm::raw_string_ostream os(s);
      module.print(os);
      os.flush();
      return s;
    };
    auto before = text();
    if (observeIndexedABWAR) {
      bool validBefore = succeeded(verify(module)) &&
          succeeded(triton::HCU::verifyLegacyMlsV0ClosedModule(module, "gfx936"));
      if (!validBefore) return signalPassFailure();
      ModuleAllocation allocation(module);
      auto initial = triton::HCU::prepareLegacyV0IndexedABWAR(module, allocation);
      if (initial.size() != 2 || before != text()) return signalPassFailure();
      auto copy = cast<triton::hcugpu::MatrixLoadToLocalOp>(initial[0].second);
      auto view = copy.getDest().getDefiningOp<triton::gpu::MemDescIndexOp>();
      auto select = view.getIndex().getDefiningOp<arith::SelectOp>();
      auto inc = select.getTrueValue().getDefiningOp<arith::AddIOp>();
      auto counter = cast<BlockArgument>(inc.getLhs());
      auto fn = copy->getParentOfType<triton::FuncOp>();
      auto enter = cast<cf::BranchOp>(fn.getBody().front().getTerminator());
      auto back = cast<cf::BranchOp>(copy->getBlock()->getTerminator());
      auto head = cast<triton::hcugpu::AsyncWaitOp>(&copy->getBlock()->front());
      auto barrier = cast<triton::gpu::LocalBarrierOp>(head->getNextNode());
      OpBuilder builder(module.getContext());
      StringRef mutation = indexedABWARMutation.getValue();
      if (mutation == "seed") {
        builder.setInsertionPoint(enter);
        enter->setOperand(counter.getArgNumber(), builder.create<arith::ConstantIntOp>(enter.getLoc(), 2, 32).getResult());
      } else if (mutation == "backcarry") {
        back->setOperand(counter.getArgNumber(), counter);
      } else if (mutation == "missing-head") {
        barrier.erase();
      } else if (mutation == "late-head") {
        barrier->moveAfter(initial[1].first);
      } else if (mutation == "read-after-copy") {
        initial[0].first->moveAfter(initial[0].second);
      } else if (mutation != "none") {
        module.emitError("unknown indexed-AB WAR test mutation");
        return signalPassFailure();
      }
      bool validAfter = succeeded(verify(module)) &&
          succeeded(triton::HCU::verifyLegacyMlsV0ClosedModule(module, "gfx936"));
      if (!validAfter) return signalPassFailure();
      auto input = text();
      auto result = triton::HCU::prepareLegacyV0IndexedABWAR(module, allocation);
      bool exact = result.size() == (mutation == "none" ? 2 : 0);
      for (auto [read, write] : result)
        exact &= llvm::is_contained(initial, std::make_pair(read, write)) &&
                 isa<triton::gpu::LocalLoadOp>(read) &&
                 isa<triton::hcugpu::MatrixLoadToLocalOp>(write) &&
                 read->getBlock() == write->getBlock() && read->isBeforeInBlock(write);
      bool unchanged = input == text();
      llvm::outs() << "// test.hcu.indexed_ab_war_pairs=" << result.size()
                   << " initial=" << initial.size() << " exact=" << exact
                   << " unchanged=" << unchanged << " valid_before=" << validBefore
                   << " valid_after=" << validAfter << " total=" << allocation.getSharedMemorySize() << "\n";
      for (auto [read, write] : result)
        llvm::outs() << "// test.hcu.indexed_ab_war_pair read->copy opIdx="
                     << cast<triton::gpu::HCUMlsSharedEncodingAttr>(cast<triton::gpu::LocalLoadOp>(read).getSrc().getType().getEncoding()).getOpIdx() << "\n";
      auto *data = allocation.getFuncData(cast<FunctionOpInterface>(fn.getOperation()));
      for (auto root : fn.getBody().front().getOps<triton::gpu::LocalAllocOp>()) {
        auto id = data->getBufferId(root.getResult());
        llvm::outs() << "// test.hcu.indexed_ab_war_frame opIdx="
                     << cast<triton::gpu::HCUMlsSharedEncodingAttr>(root.getType().getEncoding()).getOpIdx()
                     << " bytes=" << data->getAllocatedSize(id) << " base=" << data->getOffset(id) << "\n";
      }
      if (!exact || !unchanged) return signalPassFailure();
      return;
    }
    auto pairs = triton::HCU::prepareLegacyV0LocalPrefetchRAW(module);
    bool exact = pairs.size() <= 1;
    for (auto pair : pairs)
      exact &= isa<triton::hcugpu::MatrixLoadToLocalOp>(pair.first) &&
               isa<triton::gpu::LocalLoadOp>(pair.second) &&
               pair.first->getBlock() == pair.second->getBlock() &&
               pair.first->isBeforeInBlock(pair.second);
    bool unchanged = before == text();
    llvm::outs() << "// test.hcu.lp1_raw_pairs=" << pairs.size()
                 << " unchanged=" << unchanged << " exact=" << exact << "\n";
    if (!unchanged || !exact) {
      module.emitError("LP1 RAW getter mutated IR or returned inexact pairs");
      return signalPassFailure();
    }
    if (observeAWAR) {
      ModuleAllocation allocation(module);
      auto aPairs = triton::HCU::prepareLegacyV0LocalPrefetchAWAR(module, allocation, pairs);
      bool aExact = aPairs.size() <= 1;
      for (auto pair : aPairs)
        aExact &= isa<triton::gpu::LocalLoadOp>(pair.first) &&
                  isa<triton::gpu::LocalStoreOp>(pair.second) &&
                  pair.first->getBlock() == pair.second->getBlock() &&
                  pair.first->isBeforeInBlock(pair.second);
      bool aUnchanged = before == text();
      llvm::outs() << "// test.hcu.lp1_a_war_pairs=" << aPairs.size()
                   << " unchanged=" << aUnchanged << " exact=" << aExact << "\n";
      if (!aUnchanged || !aExact) {
        module.emitError("LP1 A WAR getter mutated IR or returned inexact pairs");
        return signalPassFailure();
      }
    }
    if (!scheduleReadFirst) return;
    // Do not repair publication, run membar or manufacture a witness here.
    // The scheduler receives precisely the read-only getter result above.
    auto requested = module->getAttrOfType<IntegerAttr>("ttg.hcu.v0_lp1_read_first");
    bool wantsMove = requested && requested.getType().isInteger(32) &&
                     requested.getInt() == 1;
    Block *body = pairs.size() == 1 ? pairs.front().first->getBlock() : nullptr;
    SmallVector<Operation *> expected;
    if (body) for (Operation &op : *body) expected.push_back(&op);
    if (body && wantsMove) {
      auto read = llvm::find(expected, pairs.front().second);
      if (read != expected.end()) expected.erase(read);
      auto copy = llvm::find(expected, pairs.front().first);
      if (copy != expected.end()) expected.insert(copy, pairs.front().second);
    }
    auto result = triton::HCU::scheduleLegacyV0LocalPrefetchReadFirst(module, pairs);
    bool schedulerUnchanged = before == text();
    bool expectedOrder = true;
    if (body && succeeded(result) && wantsMove) {
      SmallVector<Operation *> actual;
      for (Operation &op : *body) actual.push_back(&op);
      expectedOrder = actual == expected;
    }
    bool valid = failed(result) ? schedulerUnchanged :
        (wantsMove ? (!schedulerUnchanged && expectedOrder) : schedulerUnchanged);
    llvm::outs() << "// test.hcu.lp1_schedule_ok=" << succeeded(result)
                 << " unchanged=" << schedulerUnchanged
                 << " expected_order=" << expectedOrder
                 << " valid=" << valid << "\n";
    if (!valid) {
      module.emitError("LP1 read-first test observed unexpected mutation/order");
      return signalPassFailure();
    }
    if (failed(result)) return signalPassFailure();
  }
};

struct TestHCUBuildLegacyV0FixturePass
    : public PassWrapper<TestHCUBuildLegacyV0FixturePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestHCUBuildLegacyV0FixturePass);
  StringRef getArgument() const final { return "test-hcu-build-legacy-v0-fixture"; }
  StringRef getDescription() const final {
    return "replace one marked Swizzled fixture frame with opaque MLS version0";
  }
  void runOnOperation() override {
    using namespace triton::gpu;
    ModuleOp module = getOperation();
    SmallVector<LocalAllocOp> roots;
    module.walk([&](LocalAllocOp op) {
      if (op->hasAttr("test.hcu.legacy_v0_packet")) roots.push_back(op);
    });
    if (roots.size() == 2) {
      auto nw = module->getAttrOfType<IntegerAttr>("ttg.num-warps");
      auto hint = module->getAttrOfType<IntegerAttr>("ttg.hcu.v0_indexed_a_packet");
      if (!nw || !nw.getType().isInteger(32) || nw.getInt() != 4 ||
          !hint || !hint.getType().isInteger(32) || hint.getInt() != 1 ||
          roots[0].getType().getEncoding() == roots[1].getType().getEncoding())
        return signalPassFailure();
      unsigned kinds = 0;
      for (auto root : roots) {
        auto t = root.getType();
        auto s = t.getShape();
        int kind = s == ArrayRef<int64_t>({4, 128, 32}) ? 0 :
                   s == ArrayRef<int64_t>({4, 32, 128}) ? 1 : -1;
        if (kind < 0 || (kinds & (1u << kind)) ||
            !isa<SwizzledSharedEncodingAttr>(t.getEncoding())) return signalPassFailure();
        kinds |= 1u << kind;
        auto oldEncoding = t.getEncoding();
        auto cta = CTAEncodingAttr::getDefault(&getContext(), 2);
        SmallVector<unsigned> tile{unsigned(s[1]), unsigned(s[2])};
        auto enc = HCUMlsSharedEncodingAttr::get(&getContext(), unsigned(kind), tile,
            16, 0, 0, ArrayRef<unsigned>({1, 0}), cta);
        auto fixValue = [&](Value value) {
          auto m = dyn_cast<MemDescType>(value.getType());
          if (m && m.getEncoding() == oldEncoding)
            value.setType(MemDescType::get(m.getShape(), m.getElementType(),
                enc, m.getMemorySpace(), m.getMutableMemory()));
        };
        module.walk([&](Operation *op) {
          for (Value v : op->getResults()) fixValue(v);
          for (Region &region : op->getRegions()) for (Block &block : region)
            for (BlockArgument arg : block.getArguments()) fixValue(arg);
        });
        unsigned copies = 0;
        module.walk([&](triton::hcugpu::MatrixLoadToLocalOp copy) {
          if (copy.getDest().getType().getEncoding() != enc) return;
          copy->setAttr(triton::hcugpu::MlsEncodingAttr::getMnemonic(),
              triton::hcugpu::MlsEncodingAttr::get(&getContext(), unsigned(kind), tile,
                  16, 0, 0, ArrayRef<unsigned>({1, 0}), ArrayRef<unsigned>({2, 2})));
          ++copies;
        });
        if (copies != 5) return signalPassFailure();
        root->removeAttr("test.hcu.legacy_v0_packet");
      }
      return;
    }
    if (roots.size() != 1) {
      module.emitError("fixture requires exactly one marked allocation");
      return signalPassFailure();
    }
    auto root = roots.front();
    auto type = root.getType();
    auto shape = type.getShape();
    auto nw = module->getAttrOfType<IntegerAttr>("ttg.num-warps");
    // Reuse the prefix2 bridge for only the explicit captured s25 LP1 N4.
    auto capture = module->getAttrOfType<IntegerAttr>(
        "ttg.hcu.v0_mls_local_prefetch_capture");
    bool fourSlotCapture = shape.size() == 3 && shape[0] == 4 &&
        shape[1] == 32 && shape[2] == 128 && nw && nw.getType().isInteger(32) &&
        nw.getInt() == 4 && capture && capture.getType().isInteger(32) &&
        capture.getInt() == 1;
    // Update SSA results/carried descriptors in memory, never opaque reparse.
    if (shape.size() == 3 && (shape[0] == 2 || fourSlotCapture)) {
      // Counter-only fixtures need unsupported packets/version1 as controls.
      // This test pass only constructs IR; it never authorizes LLVM lowering.
      auto countFixture = module->getAttrOfType<IntegerAttr>(
          "test.hcu.wait_count_fixture");
      bool countOnly = shape[0] == 2 && countFixture &&
                       countFixture.getType().isInteger(32) && countFixture.getInt() == 1;
      unsigned version = 0;
      if (countOnly) {
        auto v = module->getAttrOfType<IntegerAttr>("test.hcu.wait_packet_version");
        if (!v || !v.getType().isInteger(32) || v.getInt() < 0 || v.getInt() > 1 ||
            !nw || !nw.getType().isInteger(32) ||
            (nw.getInt() != 4 && nw.getInt() != 8) ||
            !((shape[1] == 32 && shape[2] == 128) ||
              (shape[1] == 16 && shape[2] == 256) ||
              (shape[1] == 64 && shape[2] == 64))) {
          root.emitError("counter fixture requires explicit version0/1 and a known test shape");
          return signalPassFailure();
        }
        version = v.getInt();
      }
      if (!countOnly && (!nw ||
          !((shape[1] == 32 && shape[2] == 128 && nw.getInt() == 4) ||
            (shape[1] == 16 && shape[2] == 256 && nw.getInt() == 8)))) {
        root.emitError("fixture requires the exact prefix2 s25/s26 packet");
        return signalPassFailure();
      }
      auto oldEncoding = type.getEncoding();
      auto cta = CTAEncodingAttr::getDefault(&getContext(), 2);
      SmallVector<unsigned> tile{unsigned(shape[1]), unsigned(shape[2])};
      auto enc = HCUMlsSharedEncodingAttr::get(&getContext(), 1, tile, 16,
          0, version, ArrayRef<unsigned>({1, 0}), cta);
      auto fixValue = [&](Value value) {
        auto t = dyn_cast<MemDescType>(value.getType());
        if (t && t.getEncoding() == oldEncoding)
          value.setType(MemDescType::get(t.getShape(), t.getElementType(),
              enc, t.getMemorySpace(), t.getMutableMemory()));
      };
      module.walk([&](Operation *op) {
        for (Value value : op->getResults()) fixValue(value);
        for (Region &region : op->getRegions())
          for (Block &block : region)
            for (BlockArgument arg : block.getArguments()) fixValue(arg);
      });
      unsigned copies = 0;
      module.walk([&](triton::hcugpu::MatrixLoadToLocalOp copy) {
        if (copy.getDest().getType().getEncoding() != enc) return;
        copy->setAttr(triton::hcugpu::MlsEncodingAttr::getMnemonic(),
            triton::hcugpu::MlsEncodingAttr::get(&getContext(), 1, tile, 16,
                0, version, ArrayRef<unsigned>({1, 0}),
                ArrayRef<unsigned>({unsigned(nw.getInt() / 2), 2})));
        ++copies;
      });
      auto localPrefetch = module->getAttrOfType<IntegerAttr>(
          "ttg.hcu.v0_mls_local_prefetch_capture");
      unsigned expectedCopies = localPrefetch &&
          localPrefetch.getType().isInteger(32) && localPrefetch.getInt() == 1
              ? unsigned(shape[0]) + 1 : 2;
      if (copies != expectedCopies) return signalPassFailure();
      root->removeAttr("test.hcu.legacy_v0_packet");
      SmallVector<int32_t> order;
      for (unsigned dim : cta.getCTAOrder()) order.push_back(dim);
      module->setAttr("test.hcu.constructed_cta_order",
                     DenseI32ArrayAttr::get(&getContext(), order));
      return;
    }
    auto indexedFixture = module->getAttrOfType<IntegerAttr>("test.hcu.indexed_a_fixture");
    bool indexedA = shape == ArrayRef<int64_t>({8, 128, 32}) && nw &&
        nw.getType().isInteger(32) && nw.getInt() == 4 && indexedFixture &&
        indexedFixture.getType().isInteger(32) && indexedFixture.getInt() == 1;
    if (shape.size() != 3 || (shape[0] != 1 && shape[0] != 8) || !nw ||
        !((shape[1] == 32 && shape[2] == 128 && nw.getInt() == 4) ||
          (shape[1] == 16 && shape[2] == 256 && nw.getInt() == 8) || indexedA) ||
        !root.getResult().hasOneUse()) {
      root.emitError("fixture requires the exact s25/s26 closed frame");
      return signalPassFailure();
    }
    auto view = dyn_cast<MemDescIndexOp>(*root.getResult().getUsers().begin());
    if (!view) return signalPassFailure();
    auto cta = CTAEncodingAttr::getDefault(&getContext(), 2);
    SmallVector<unsigned> tile{unsigned(shape[1]), unsigned(shape[2])};
    auto enc = HCUMlsSharedEncodingAttr::get(&getContext(), indexedA ? 0 : 1, tile, 16, 0, 0,
                                            ArrayRef<unsigned>({1, 0}), cta);
    auto newType = [&](MemDescType t) {
      return MemDescType::get(t.getShape(), t.getElementType(), enc,
                             t.getMemorySpace(), t.getMutableMemory());
    };
    root.getResult().setType(newType(type));
    view.getResult().setType(newType(view.getType()));
    root->removeAttr("test.hcu.legacy_v0_packet");
    unsigned copies = 0;
    for (Operation *user : view.getResult().getUsers()) {
      if (auto copy = dyn_cast<triton::hcugpu::MatrixLoadToLocalOp>(user)) {
        copy->setAttr(triton::hcugpu::MlsEncodingAttr::getMnemonic(),
          triton::hcugpu::MlsEncodingAttr::get(&getContext(), indexedA ? 0 : 1, tile, 16, 0,
            0, ArrayRef<unsigned>({1, 0}),
            ArrayRef<unsigned>({unsigned(nw.getInt() / 2), 2})));
        ++copies;
      }
    }
    if (copies != 1) return signalPassFailure();
    SmallVector<int32_t> order;
    for (unsigned dim : cta.getCTAOrder()) order.push_back(dim);
    module->setAttr("test.hcu.constructed_cta_order",
                    DenseI32ArrayAttr::get(&getContext(), order));
  }
};

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

// Test-only experiment. Run after final LDS reordering and SCF-to-CF, BEFORE
// allocation. It changes one operation position, never offsets or barrier count.
struct TestHCULegacyV0JointPhasePass
    : public PassWrapper<TestHCULegacyV0JointPhasePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestHCULegacyV0JointPhasePass);
  StringRef getArgument() const final { return "test-hcu-legacy-v0-joint-phase"; }
  StringRef getDescription() const final {
    return "move the exact staged A initialization into the B publication phase";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto fail = [&](StringRef reason) {
      module.emitError("joint-phase test: ") << reason;
      signalPassFailure();
    };
    auto isOne = [&](StringRef name) {
      auto attr = module->getAttrOfType<IntegerAttr>(name);
      return attr && attr.getInt() == 1;
    };
    auto target = module->getAttrOfType<StringAttr>("ttg.target");
    if (!target || target.getValue() != "hip:gfx936" ||
        !isOne("ttg.hcu.v0_blockptr") || !isOne("ttg.hcu.v0_blockptr_loop") ||
        module->hasAttr("ttg.hcu.v0_joint_phase") || module->hasAttr("ttg.shared"))
      return fail("needs unallocated existing pure-TL loop input, no joint flag");
    // Full current closed CF/source/token/type/effect proof precedes movement.
    if (failed(triton::HCU::verifyLegacyMlsV0ClosedModule(module, "gfx936")))
      return signalPassFailure();
    SmallVector<triton::gpu::LocalAllocOp> roots;
    SmallVector<triton::gpu::LocalLoadOp> reads;
    SmallVector<triton::hcugpu::MatrixLoadToLocalOp> copies;
    SmallVector<triton::LoadOp> globalLoads;
    module.walk([&](Operation *op) {
      if (auto x = dyn_cast<triton::gpu::LocalAllocOp>(op)) roots.push_back(x);
      if (auto x = dyn_cast<triton::gpu::LocalLoadOp>(op)) reads.push_back(x);
      if (auto x = dyn_cast<triton::hcugpu::MatrixLoadToLocalOp>(op)) copies.push_back(x);
      if (auto x = dyn_cast<triton::LoadOp>(op)) globalLoads.push_back(x);
    });
    if (roots.size() != 2 || reads.size() != 2 || copies.size() != 1 ||
        globalLoads.size() != 1)
      return fail("exact two-root/two-read/one-copy/one-global-load counts");
    auto copy = copies.front();
    triton::gpu::LocalAllocOp aRoot, bRoot;
    for (auto root : roots) {
      if (root->hasAttr("allocation.offset"))
        return fail("allocation attributes must be absent; rerun allocator afterwards");
      if (root.getSrc()) aRoot = root;
      else bRoot = root;
    }
    if (!aRoot || !bRoot || aRoot.getSrc() != globalLoads.front().getResult())
      return fail("unique initialized A and uninitialized B roots");
    auto view = copy.getDest().getDefiningOp<triton::gpu::MemDescIndexOp>();
    if (!view || view.getSrc() != bRoot.getResult())
      return fail("unchanged direct B root/view/copy");
    triton::gpu::LocalLoadOp aRead, bRead;
    for (auto read : reads) {
      if (read.getSrc() == aRoot.getResult()) aRead = read;
      if (read.getSrc() == view.getResult()) bRead = read;
    }
    auto *head = copy->getPrevNode();
    Block *body = copy->getBlock();
    if (!aRead || !bRead || !isa_and_nonnull<triton::gpu::LocalBarrierOp>(head) ||
        aRoot->getBlock() != body || aRead->getBlock() != body ||
        bRead->getBlock() != body || globalLoads.front()->getBlock() != body ||
        !globalLoads.front()->isBeforeInBlock(head) ||
        !head->isBeforeInBlock(bRead.getOperation()) ||
        !bRead->isBeforeInBlock(aRoot.getOperation()) ||
        aRoot->getNextNode() != aRead.getOperation())
      return fail("exact globalA/head/Bcopy/Bread/Ainit/Aread source order");
    // The existing head barrier remains before BOTH shared writes. All local
    // reads remain after the existing wait0+publication barrier. SSA source
    // of A initialization already dominates the insertion point in this block.
    Operation *oldNext = aRoot->getNextNode();
    aRoot->moveBefore(copy.getOperation());
    module->setAttr("ttg.hcu.v0_joint_phase",
                    IntegerAttr::get(IntegerType::get(&getContext(), 32), 1));
    if (failed(triton::HCU::verifyLegacyMlsV0ClosedModule(module, "gfx936"))) {
      aRoot->moveBefore(oldNext);
      module->removeAttr("ttg.hcu.v0_joint_phase");
      return signalPassFailure();
    }
    // No allocator metadata is installed here. The immediately following REAL
    // allocator must independently produce B0/8192, A8192/2048,total10240.
  }
};

} // namespace

namespace mlir::test {
void registerTestHCUGPUMembarPass() {
  PassRegistration<TestHCUGPUMembarPass>();
  PassRegistration<TestHCUPipelineConstantProbePass>();
  PassRegistration<TestHCUMfmaParserRoundTripPass>();
  PassRegistration<TestHCUAsyncSlotWARProofPass>();
  PassRegistration<TestHCUPrefix2WARProofPass>();
  PassRegistration<TestHCULegacyV0LocalPrefetchProofPass>();
  PassRegistration<TestHCULegacyV0LocalPrefetchRAWPass>();
  PassRegistration<TestHCUBuildLegacyV0FixturePass>();
  PassRegistration<TestHCULegacyV0JointPhasePass>();
}
} // namespace mlir::test
