#ifndef TRITON_METAX_GLUON_MEMDESC_ALIAS_ANALYSIS_H
#define TRITON_METAX_GLUON_MEMDESC_ALIAS_ANALYSIS_H

#include "Gluon/Analysis/GluonRegionBranchAnalysis.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"

#include <optional>

namespace mlir::triton::gpu::metax::gluon {

inline constexpr StringLiteral kDefaultSharedLayoutAttrName =
    "ttg.gluon.default-shared-layout";

/// A frontend marker delegates one logical shared-layout family to the
/// compiler. Both local_alloc and memdesc_reinterpret may start such a family;
/// reinterpret remains a root, so no requirement crosses back to its source.
inline bool isCompilerManagedSharedFamilyRoot(Operation *root) {
  return root &&
         isa<triton::gpu::LocalAllocOp,
             triton::gpu::MemDescReinterpretOp>(root) &&
         root->hasAttr(kDefaultSharedLayoutAttrName);
}

inline bool isCompilerManagedSharedFamilyRoot(Value root) {
  return isCompilerManagedSharedFamilyRoot(root.getDefiningOp());
}

/// Stable identity for one logical memdesc layout family. Physical storage
/// aliases are deliberately not represented here.
struct LayoutFamilyId {
  Value contractRoot;

  explicit operator bool() const { return static_cast<bool>(contractRoot); }
  bool operator==(const LayoutFamilyId &other) const {
    return contractRoot == other.contractRoot;
  }
};

/// A supported logical view path in root-to-value order. Index and subslice
/// operations remain in the path so clients can inspect their coverage.
/// Encoding projection treats them as identity only when the declared source
/// and result encodings are equal.
struct MemDescAliasPath {
  Value value;
  Value root;
  SmallVector<Operation *, 4> views;
};

enum class MemDescAliasKnowledge { Bottom, Known, Unknown };

/// Return the source of a supported logical memdesc view. This is a structural
/// query only: callers must still apply each op's own layout semantics.
Value getMemDescLayoutViewSource(Operation *op);

/// Computes logical memdesc families and coordinate paths without consulting
/// late shared-memory allocation. Every concrete reinterpret result starts an
/// independent family: a frontend-marked result is compiler-managed, otherwise
/// it is fixed. An Auto reinterpret has no inference contract and becomes
/// Unknown. RegionBranch carriers retain a path only on consensus.
class GluonMemDescAliasAnalysis {
public:
  explicit GluonMemDescAliasAnalysis(Operation *root)
      : root(root), carriers(root) {}

  LogicalResult initialize();

  FailureOr<MemDescAliasPath> getPath(Value value) const;
  FailureOr<LayoutFamilyId> getFamily(Value value) const;

private:
  struct State {
    MemDescAliasKnowledge knowledge = MemDescAliasKnowledge::Bottom;
    MemDescAliasPath path;
  };

  State transfer(Value value) const;
  const State &lookup(Value value) const;

  Operation *root;
  GluonRegionBranchAnalysis carriers;
  DenseMap<Value, State> states;
  State unknownState{MemDescAliasKnowledge::Unknown, {}};
};

/// Project a view-coordinate encoding into the path's family-root coordinate
/// system. This helper emits a diagnostic on unsupported inference but never
/// mutates IR.
FailureOr<Attribute> projectEncodingToRoot(const MemDescAliasPath &path,
                                           Attribute viewEncoding,
                                           Operation *diagnosticOp);

/// Project a family-root encoding into the path's view coordinate system.
/// This helper emits a diagnostic on unsupported inference but never mutates
/// IR.
FailureOr<Attribute> projectEncodingFromRoot(const MemDescAliasPath &path,
                                             Attribute rootEncoding,
                                             Operation *diagnosticOp);

/// Project one logical tensor dimension through a supported memdesc alias
/// path. Unlike encoding projection, a shape-changing reshape has no unique
/// axis map and therefore returns std::nullopt instead of guessing.
std::optional<unsigned> projectDimensionToRoot(const MemDescAliasPath &path,
                                               unsigned viewDimension);
std::optional<unsigned> projectDimensionFromRoot(const MemDescAliasPath &path,
                                                 unsigned rootDimension);

} // namespace mlir::triton::gpu::metax::gluon

namespace llvm {

template <>
struct DenseMapInfo<
    mlir::triton::gpu::metax::gluon::LayoutFamilyId> {
  using LayoutFamilyId =
      mlir::triton::gpu::metax::gluon::LayoutFamilyId;

  static inline LayoutFamilyId getEmptyKey() {
    return {DenseMapInfo<mlir::Value>::getEmptyKey()};
  }
  static inline LayoutFamilyId getTombstoneKey() {
    return {DenseMapInfo<mlir::Value>::getTombstoneKey()};
  }
  static unsigned getHashValue(const LayoutFamilyId &family) {
    return DenseMapInfo<mlir::Value>::getHashValue(family.contractRoot);
  }
  static bool isEqual(const LayoutFamilyId &lhs,
                      const LayoutFamilyId &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

#endif // TRITON_METAX_GLUON_MEMDESC_ALIAS_ANALYSIS_H
