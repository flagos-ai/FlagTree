#include "Gluon/GluonLayoutCandidate.h"

#include "Gluon/Analysis/GluonDotAnalysis.h"
#include "Gluon/Analysis/GluonMemDescAliasAnalysis.h"
#include "Gluon/GluonKernelReplay.h"
#include "Gluon/GluonLayoutPlaceholders.h"
#include "Gluon/Passes.h"
#include "Gluon/Targets/GluonC500LayoutRules.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>

#define DEBUG_TYPE "metax-gluon-layout-candidate"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) (DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gd = ::mlir::triton::gluon;

namespace mlir::triton::gpu::metax::gluon {
namespace {

static std::string sha256(StringRef text) {
  std::array<uint8_t, 32> digest =
      llvm::SHA256::hash(llvm::arrayRefFromStringRef(text));
  return llvm::toHex(ArrayRef<uint8_t>(digest), /*LowerCase=*/true);
}

static void printDotPlan(raw_ostream &os, const DotLayoutPlan &plan) {
  os << "accumulator:";
  plan.mma.print(os);
  os << '\n';
  for (unsigned index : {0u, 1u}) {
    os << "operand:" << index << ':';
    plan.getOperand(index).print(os);
    os << '\n';
  }
}

static std::string computeCandidateDigest(const LayoutCandidate &candidate) {
  std::string canonical;
  llvm::raw_string_ostream os(canonical);
  os << "c500-gluon-dot-candidate-v1\n";
  for (auto [index, dot] : llvm::enumerate(candidate.dots)) {
    os << "dot:" << index << '\n';
    printDotPlan(os, dot);
  }
  os.flush();
  return sha256(canonical);
}

static std::string computeDomainDigest(const CandidateDomain &domain) {
  std::string canonical;
  llvm::raw_string_ostream os(canonical);
  os << "c500-gluon-dot-domain-v1\n"
     << "fallback:" << domain.fallback.digest << '\n';
  for (const LayoutCandidate &candidate : domain.alternatives)
    os << "alternative:" << candidate.digest << '\n';
  os.flush();
  return sha256(canonical);
}

static const DotLayoutPlan *
findMatchingProfile(ArrayRef<DotLayoutPlan> domain, Attribute profile) {
  auto it = llvm::find_if(domain, [&](const DotLayoutPlan &plan) {
    return haveSameC500AccumulatorProfile(plan.mma, profile);
  });
  return it == domain.end() ? nullptr : &*it;
}

static bool sameCandidate(const LayoutCandidate &lhs,
                          const LayoutCandidate &rhs) {
  if (lhs.dots.size() != rhs.dots.size())
    return false;
  return llvm::all_of(llvm::zip(lhs.dots, rhs.dots), [](auto pair) {
    const DotLayoutPlan &left = std::get<0>(pair);
    const DotLayoutPlan &right = std::get<1>(pair);
    return left.mma == right.mma && left.operandA == right.operandA &&
           left.operandB == right.operandB;
  });
}

static FailureOr<tt::FuncOp> getSinglePublicEntry(ModuleOp module) {
  SmallVector<tt::FuncOp, 2> entries;
  llvm::copy_if(module.getOps<tt::FuncOp>(), std::back_inserter(entries),
                [](tt::FuncOp func) { return func.isPublic(); });
  if (entries.size() != 1)
    return module.emitError()
           << "Gluon layout candidate construction requires exactly one "
              "public entry, got "
           << entries.size();
  return entries.front();
}

static void addCandidateAtomicLayoutPipeline(PassManager &pm,
                                             int computeCapability) {
  pm.addPass(createTritonMETAXGPUGluonPropagateLayoutPass());
  pm.addPass(createTritonMETAXGPUGluonResolvePlaceholderLayoutsPass());
  pm.addPass(createTritonMETAXGPUGluonLegalizeC500AsyncCopyLayoutPass());
  pm.addPass(createTritonMETAXGPUGluonLegalizeRegisterSlicesPass());
  pm.addPass(createSCCPPass());
  pm.addPass(tt::createTritonLoopAwareCSE());
  pm.addPass(gd::createGluonCanonicalize());
  ttg::TritonGPUOptimizeDotOperandsOptions dotOptions;
  dotOptions.hoistLayoutConversion = computeCapability >= 80;
  pm.addPass(ttg::createTritonGPUOptimizeDotOperands(dotOptions));
  pm.addPass(ttg::createTritonGPURemoveLayoutConversions());
  pm.addPass(createTritonMETAXGPUGluonLegalizeLocalDotStagingPass());
}

static void addCandidateConcreteFinalizationPipeline(PassManager &pm) {
  pm.addPass(ttg::createTritonGPUReduceDataDuplication());
  pm.addPass(createTritonMETAXGPUGluonReorderInstructionsPass());
  pm.addPass(createCSEPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(ttg::createTritonGPUCombineTensorSelectAndIf());
  pm.addPass(createTritonMETAXGPUGluonSelectC500LayoutTransfersPass());
  pm.addPass(createTritonMETAXGPUGluonInsertGvmArriveBarrierSharedPass());
  pm.addPass(createTritonMETAXGPUGluonVerifySynchronizationPass());
  pm.addPass(createTritonMETAXGPUGluonVerifyLayoutContractsPass());
}

static std::string printModule(ModuleOp module) {
  std::string source;
  llvm::raw_string_ostream os(source);
  OpPrintingFlags flags;
  flags.enableDebugInfo();
  module.print(os, flags);
  os << '\n';
  os.flush();
  return source;
}

static LogicalResult finalizeCandidate(ModuleOp module,
                                       const LayoutCandidate &candidate,
                                       StringRef domainDigest,
                                       StringRef runtimeContract,
                                       StringRef originalEntryName,
                                       int computeCapability) {
  if (failed(insertLayoutRequirements(module, candidate, computeCapability)))
    return failure();

  MLIRContext *context = module.getContext();
  module->setAttr(kLayoutDomainDigestAttr,
                  StringAttr::get(context, domainDigest));
  module->setAttr(kLayoutVariantDigestAttr,
                  StringAttr::get(context, candidate.digest));
  module->setAttr(kLayoutRuntimeContractAttr,
                  StringAttr::get(context, runtimeContract));

  FailureOr<tt::FuncOp> entry = getSinglePublicEntry(module);
  if (failed(entry))
    return failure();
  SymbolTable::setSymbolName(
      *entry, (originalEntryName + "__gluon_layout_" + candidate.digest).str());

  PassManager propagation(context);
  // Propagation may leave a mixed concrete/Auto relation for the immediately
  // following placeholder resolver and local legalizers to close (for example
  // an MMA reduction whose row result joins a loop carrier). Disable the
  // per-pass verifier only across this atomic layout phase, then verify before
  // scheduling, transfer selection, or synchronization.
  propagation.enableVerifier(false);
  addCandidateAtomicLayoutPipeline(propagation, computeCapability);
  if (failed(propagation.run(module))) {
    if (std::getenv("TRITON_GLUON_DUMP_FAILURE"))
      llvm::errs() << printModule(module);
    return failure();
  }
  if (failed(verify(module))) {
    if (std::getenv("TRITON_GLUON_DUMP_FAILURE"))
      llvm::errs() << printModule(module);
    return module.emitError()
           << "Gluon layout propagation and placeholder resolution produced "
              "invalid IR";
  }

  PassManager finalization(context);
  addCandidateConcreteFinalizationPipeline(finalization);
  if (failed(finalization.run(module))) {
    if (std::getenv("TRITON_GLUON_DUMP_FAILURE"))
      llvm::errs() << printModule(module);
    return failure();
  }

  module->setAttr("ttg.gluon.gvm-finalized",
                  IntegerAttr::get(IntegerType::get(context, 32), 1));
  if (failed(verify(module)))
    return module.emitError()
           << "finalized Gluon layout candidate failed IR verification";
  return success();
}

} // namespace

FailureOr<CandidateDomain>
discoverC500LayoutCandidateDomain(ModuleOp module, int computeCapability) {
  if (!module)
    return failure();

  GluonMemDescAliasAnalysis aliases(module);
  if (failed(aliases.initialize()))
    return failure();
  FailureOr<SmallVector<DotPipelineFacts, 4>> facts =
      collectDotPipelineFacts(module, aliases);
  if (failed(facts))
    return failure();
  SmallVector<DotPipelineFacts, 4> tunableFacts;
  llvm::copy_if(*facts, std::back_inserter(tunableFacts),
                isTunableC500Dot);

  CandidateDomain result;
  if (tunableFacts.empty()) {
    result.fallback.digest = computeCandidateDigest(result.fallback);
    LDBG("[candidate-domain] no tunable non-BSM dot; fallback="
         << result.fallback.digest);
    return result;
  }

  tt::ModuleAxisInfoAnalysis axisInfo(module);
  if (failed(inferC500DotMemoryFacts(module, axisInfo, tunableFacts)))
    return failure();

  SmallVector<SmallVector<DotLayoutPlan, 8>, 4> dotDomains;
  for (const DotPipelineFacts &dotFacts : tunableFacts) {
    FailureOr<SmallVector<DotLayoutPlan, 8>> domain =
        inferC500DotLayoutDomain(dotFacts, computeCapability);
    if (failed(domain) || domain->empty())
      return dotFacts.dot->emitError()
             << "C500 produced no legal layout for this dot";
    dotDomains.push_back(std::move(*domain));
  }

  SmallVector<Attribute, 8> commonProfiles;
  for (const DotLayoutPlan &plan : dotDomains.front()) {
    Attribute profile = plan.mma;
    if (llvm::any_of(commonProfiles, [&](Attribute existing) {
          return haveSameC500AccumulatorProfile(existing, profile);
        }))
      continue;
    bool acceptedByAll = true;
    for (ArrayRef<DotLayoutPlan> domain : dotDomains)
      acceptedByAll &= findMatchingProfile(domain, profile) != nullptr;
    if (acceptedByAll)
      commonProfiles.push_back(profile);
  }

  // A mandatory whole-module fallback must be ownership-coherent whenever the
  // existing dot domains have a common profile. Per-dot leaf fallbacks remain
  // the deterministic last resort only when the intersection is empty; using
  // them despite a non-empty intersection can make a shared reduction/store
  // component impossible to type before local conversions are materialized.
  if (commonProfiles.empty()) {
    for (ArrayRef<DotLayoutPlan> domain : dotDomains)
      result.fallback.dots.push_back(domain.front());
  } else {
    for (ArrayRef<DotLayoutPlan> domain : dotDomains) {
      const DotLayoutPlan *plan =
          findMatchingProfile(domain, commonProfiles.front());
      if (!plan)
        return module.emitError()
               << "lost the coherent fallback C500 accumulator profile while "
                  "building the candidate domain";
      result.fallback.dots.push_back(*plan);
    }
  }
  result.fallback.digest = computeCandidateDigest(result.fallback);

  for (Attribute profile : commonProfiles) {
    LayoutCandidate candidate;
    for (ArrayRef<DotLayoutPlan> domain : dotDomains) {
      const DotLayoutPlan *plan = findMatchingProfile(domain, profile);
      if (!plan)
        return module.emitError()
               << "lost a common C500 accumulator profile while building "
                  "the candidate domain";
      candidate.dots.push_back(*plan);
    }
    if (sameCandidate(candidate, result.fallback) ||
        llvm::any_of(result.alternatives,
                     [&](const LayoutCandidate &existing) {
                       return sameCandidate(candidate, existing);
                     }))
      continue;
    candidate.digest = computeCandidateDigest(candidate);
    result.alternatives.push_back(std::move(candidate));
  }

  LDBG("[candidate-domain] dots="
       << facts->size() << ", tunable="
       << tunableFacts.size() << ", common-profiles="
       << commonProfiles.size() << ", alternatives="
       << result.alternatives.size() << ", domain="
       << computeDomainDigest(result));
  return result;
}

FailureOr<CandidateBundle>
buildC500LayoutCandidateBundle(ModuleOp module, int computeCapability) {
  if (!module)
    return failure();

  FailureOr<tt::FuncOp> originalEntry = getSinglePublicEntry(module);
  if (failed(originalEntry))
    return failure();
  const std::string originalEntryName = (*originalEntry).getSymName().str();
  const KernelReplayEffectContract replay =
      analyzeKernelReplayEffectContract(*originalEntry);
  const std::string runtimeContract =
      serializeKernelReplayEffectContract(replay);

  FailureOr<CandidateDomain> domain =
      discoverC500LayoutCandidateDomain(module, computeCapability);
  if (failed(domain))
    return failure();

  CandidateBundle bundle;
  bundle.domainDigest = computeDomainDigest(*domain);
  bundle.fallbackDigest = domain->fallback.digest;
  bundle.runtimeContract = runtimeContract;

  auto materialize = [&](const LayoutCandidate &candidate,
                         bool mandatory) -> LogicalResult {
    OwningOpRef<ModuleOp> experiment =
        cast<ModuleOp>(module->clone());
    std::string diagnostics;
    LogicalResult finalized = failure();
    if (mandatory) {
      finalized = finalizeCandidate(
          *experiment, candidate, bundle.domainDigest, runtimeContract,
          originalEntryName, computeCapability);
    } else {
      ScopedDiagnosticHandler handler(
          module.getContext(), [&](Diagnostic &diagnostic) {
            llvm::raw_string_ostream os(diagnostics);
            os << diagnostic << '\n';
            return success();
          });
      finalized = finalizeCandidate(
          *experiment, candidate, bundle.domainDigest, runtimeContract,
          originalEntryName, computeCapability);
    }

    if (failed(finalized)) {
      if (!mandatory) {
        LDBG("[candidate-eliminated] digest="
             << candidate.digest << ", diagnostics=" << diagnostics);
        (*originalEntry).emitRemark()
            << "eliminated experimental layout candidate "
            << candidate.digest << ": " << diagnostics;
        return success();
      }
      return module.emitError()
             << "mandatory C500 Gluon layout fallback failed finalization";
    }

    auto contract = (*experiment)
                        ->getAttrOfType<StringAttr>(
                            kLayoutRuntimeContractAttr);
    auto digest = (*experiment)
                      ->getAttrOfType<StringAttr>(kLayoutVariantDigestAttr);
    if (!contract || contract.getValue() != runtimeContract || !digest ||
        digest.getValue() != candidate.digest)
      return module.emitError()
             << "finalized layout candidate changed its identity or runtime "
                "effect contract";

    bundle.variants.push_back(
        CandidateVariant{candidate.digest, printModule(*experiment)});
    return success();
  };

  if (failed(materialize(domain->fallback, /*mandatory=*/true)))
    return failure();
  for (const LayoutCandidate &candidate : domain->alternatives)
    if (failed(materialize(candidate, /*mandatory=*/false)))
      return failure();

  if (bundle.variants.empty() ||
      bundle.variants.front().digest != bundle.fallbackDigest)
    return module.emitError()
           << "C500 layout candidate bundle lost its mandatory fallback";

  LDBG("[candidate-bundle] domain="
       << bundle.domainDigest << ", planned="
       << 1 + domain->alternatives.size() << ", retained="
       << bundle.variants.size() << ", fallback=" << bundle.fallbackDigest);
  return bundle;
}

} // namespace mlir::triton::gpu::metax::gluon

#undef LDBG
#undef DBGS
#undef DEBUG_TYPE
