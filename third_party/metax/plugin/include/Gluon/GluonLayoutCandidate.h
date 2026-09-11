#ifndef TRITON_METAX_GLUON_LAYOUT_CANDIDATE_H
#define TRITON_METAX_GLUON_LAYOUT_CANDIDATE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace mlir::triton::gpu::metax::gluon {

inline constexpr StringLiteral kLayoutCandidateManifestAttr =
    "ttg.gluon.layout-candidate-manifest";
inline constexpr StringLiteral kLayoutRuntimeContractAttr =
    "ttg.gluon.layout-runtime-contract";
inline constexpr StringLiteral kLayoutVariantDigestAttr =
    "ttg.gluon.layout-variant-digest";
inline constexpr StringLiteral kLayoutDomainDigestAttr =
    "ttg.gluon.layout-domain-digest";

/// One complete C500 dot contract. The three physical encodings are generated
/// and verified together by the target rule; generic code must not recombine
/// their fields independently.
struct DotLayoutPlan {
  Attribute mma;
  Attribute operandA;
  Attribute operandB;

  Attribute getOperand(unsigned index) const {
    assert(index < 2 && "dot operand index must be zero or one");
    return index == 0 ? operandA : operandB;
  }
};

/// A whole-module candidate. `dots` follows the deterministic preorder of
/// tunable `tt.dot` operations in the normalized module. Fixed-layout and BSM
/// dots are deliberately absent. This ordering is consumed only inside one
/// C++ bundle-construction call; it is never serialized as a replay selector.
struct LayoutCandidate {
  SmallVector<DotLayoutPlan, 4> dots;
  std::string digest;
};

struct CandidateDomain {
  LayoutCandidate fallback;
  SmallVector<LayoutCandidate, 8> alternatives;
};

struct CandidateVariant {
  std::string digest;
  std::string source;
};

/// Finalized Gluon-layout variants. Python persists these opaque module
/// sources and owns only compilation, benchmarking, and winner caching.
struct CandidateBundle {
  std::string domainDigest;
  std::string fallbackDigest;
  std::string runtimeContract;
  SmallVector<CandidateVariant, 8> variants;
};

FailureOr<CandidateDomain>
discoverC500LayoutCandidateDomain(ModuleOp module, int computeCapability);

LogicalResult insertLayoutRequirements(ModuleOp module,
                                       const LayoutCandidate &candidate,
                                       int computeCapability);

FailureOr<CandidateBundle>
buildC500LayoutCandidateBundle(ModuleOp module, int computeCapability);

} // namespace mlir::triton::gpu::metax::gluon

#endif // TRITON_METAX_GLUON_LAYOUT_CANDIDATE_H
