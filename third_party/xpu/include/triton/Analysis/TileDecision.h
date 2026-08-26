#ifndef TRITONXPU_ANALYSIS_TILEDECISION_H
#define TRITONXPU_ANALYSIS_TILEDECISION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <optional>
#include <string>

//===----------------------------------------------------------------------===//
// Tile decision: picks `iterNum` out of the trip counts UnrollControl can
// express, by narrowing the candidate set one tier at a time.
//
// The order is a dictionary order, not a weighted sum: a lower tier is never
// traded against a higher one, so no unit conversion between "registers" and
// "instructions" has to be invented. Tier 1 and 2 are hard -- they filter --
// and tier 3 is soft: it ranks whatever survived. Adding a consideration means
// pushing one more object into a tier, or adding a tier; nothing registers
// itself and there is no weight to tune.
//
// Every criterion has to end up in the remark (see `Decision::perTierTrace`).
// A criterion that cannot be observed cannot be verified, and the one time this
// pass shipped a silent fallback -- a mis-set budget making
// `budget-unreachable` the normal case -- the symptom was read as the design
// for weeks while every probe ran 8..11% slow.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace xpu {

// One expressible trip count. The boundary set and the segment id land here
// once M4/M5 exist; until then the candidate *is* the trip count.
struct TileCandidate {
  int64_t iterNum = 1;
};

// What the criteria are allowed to read. Filled once per decision point so the
// criteria stay free of pass state and are testable on their own.
struct TileContext {
  int64_t numCol = 1;
  int64_t widthPerCore = 1;
  // Peak simultaneously-live vector registers over the block the tile loop
  // will live in, and the widest vector row in it. Both come from
  // getRegPressure/getBlockRegPressure.
  int64_t peakVRegs = 0;
  int64_t maxVecWidth = 1;
  int64_t scalarPeak = -1; // reported only, never drives a tier
  // Narrowest vector row in slots, or 0 when the segment holds no vector
  // value. A legality bound, so it is already folded into the candidate set;
  // it is carried here only to name which ceiling a fallback hit.
  int64_t vecRow = 0;
  int64_t vrfBudget = 0;
  // iter_args the tile loop will carry: 0 at a pointwise store segment
  // (UnrollControl.cpp:1263 passes an empty range), one per reduce data
  // operand at a reduce segment (:1908).
  int64_t loopResults = 0;
};

// What one criterion said about the candidate set, for the remark.
struct CriterionTrace {
  llvm::StringRef name;
  unsigned tier = 0;
  int64_t candidatesIn = 0;
  int64_t candidatesOut = 0;
  std::optional<int64_t> chosenCost; // set by soft criteria only
  std::string why;                   // non-empty when the tier vetoed
};

struct Decision {
  int64_t iterNum = 1;
  std::string why;
  llvm::SmallVector<CriterionTrace> perTierTrace;
};

class TileCriterion {
public:
  virtual ~TileCriterion() = default;
  virtual llvm::StringRef name() const = 0;
  // 1 = hard (VRF pressure proxy), 2 = hard (LM capacity),
  // 3 = soft (tie-break).
  virtual unsigned tier() const = 0;

  // Hard criteria implement this; fill `why` when rejecting.
  virtual bool isFeasible(const TileCandidate &, const TileContext &,
                          std::string &why) const {
    return true;
  }
  // Soft criteria implement this. Smaller is better; std::nullopt means the
  // criterion has no opinion on this candidate.
  virtual std::optional<int64_t> cost(const TileCandidate &,
                                      const TileContext &) const {
    return std::nullopt;
  }
};

// The trip count a tree of the widest row needs so that the block's vector
// pressure lands inside the calibrated budget.
//
// NOT a zero-spill criterion, and it must not be described as one: the budget
// is a pressure ceiling calibrated against measured time, and on the welford
// reduce segment the time-optimal point spills 14 accumulators while the
// spill-free point is 15.6% slower. Once the pack/unpack unit price is
// calibrated this whole criterion moves to tier 3, so nothing about it may
// assume it sits on a hard tier.
int64_t vrfBudgetTarget(const TileContext &ctx);

// Assembles the first-version criteria and runs the narrowing. `candidates`
// must already be the expressible trip counts, ascending.
class TileDecider {
public:
  TileDecider();
  Decision decide(llvm::ArrayRef<int64_t> candidates,
                  const TileContext &ctx) const;

private:
  llvm::SmallVector<std::unique_ptr<TileCriterion>> criteria;
};

} // namespace xpu
} // namespace triton
} // namespace mlir

#endif // TRITONXPU_ANALYSIS_TILEDECISION_H
