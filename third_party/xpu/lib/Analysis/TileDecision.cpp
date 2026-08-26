#include "triton/Analysis/TileDecision.h"

#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace triton {
namespace xpu {

int64_t vrfBudgetTarget(const TileContext &ctx) {
  // Trip count a tree of the widest row would need to fit the budget, ...
  int64_t blockTarget =
      ctx.vrfBudget > 0 ? llvm::divideCeil(ctx.peakVRegs, ctx.vrfBudget) : 1;
  // ... converted to this tree's row: what is shared between the trees in one
  // block is the per-iteration width, not the number of iterations.
  int64_t target = llvm::divideCeil(blockTarget * ctx.widthPerCore,
                                    std::max<int64_t>(ctx.maxVecWidth, 1));
  return std::max<int64_t>(target, 1);
}

namespace {

class VRFBudgetCriterion : public TileCriterion {
public:
  llvm::StringRef name() const override { return "vrf-budget"; }
  unsigned tier() const override { return 1; }

  bool isFeasible(const TileCandidate &cand, const TileContext &ctx,
                  std::string &why) const override {
    // The smallest surviving trip count is the largest tile that fits the
    // budget, so admitting everything at or above the target never over-tiles:
    // tier 3 picks the smallest one back.
    if (cand.iterNum >= vrfBudgetTarget(ctx))
      return true;
    why = "budget-unreachable";
    return false;
  }
};

// Placeholder for M3. Deciding nothing is deliberate: there is no calibrated
// capacity number yet (§3.3.1 -- legalize does not enforce an alloca ceiling
// either), and a criterion with a made-up bound would silently move decisions.
// It stays on the tier so the remark has a slot for the occupancy figure the
// moment M3 can produce one.
class LMCapacityCriterion : public TileCriterion {
public:
  llvm::StringRef name() const override { return "lm-capacity"; }
  unsigned tier() const override { return 2; }

  bool isFeasible(const TileCandidate &, const TileContext &,
                  std::string &) const override {
    return true;
  }
};

// Loop overhead, analytically: the index arithmetic plus a copy in and out for
// every value the loop carries, charged once per iteration. No calibration and
// no new measurement -- and being monotone in `iterNum`, its argmin is the
// smallest surviving trip count, which is exactly what this pass picked before
// the tiers existed. That is the point: the framework lands as a refactor with
// no observable change, and criteria that *do* move decisions get added one at
// a time behind their own evidence.
class LoopOverheadCriterion : public TileCriterion {
public:
  llvm::StringRef name() const override { return "loop-overhead"; }
  unsigned tier() const override { return 3; }

  std::optional<int64_t> cost(const TileCandidate &cand,
                              const TileContext &ctx) const override {
    return cand.iterNum * (1 + 2 * std::max<int64_t>(ctx.loopResults, 0));
  }
};

} // namespace

TileDecider::TileDecider() {
  criteria.emplace_back(std::make_unique<VRFBudgetCriterion>());
  criteria.emplace_back(std::make_unique<LMCapacityCriterion>());
  criteria.emplace_back(std::make_unique<LoopOverheadCriterion>());
}

Decision TileDecider::decide(llvm::ArrayRef<int64_t> candidates,
                             const TileContext &ctx) const {
  Decision d;
  if (candidates.empty()) {
    d.iterNum = 1;
    d.why = "no-legal-trip-count";
    return d;
  }
  // The largest expressible trip count is the fallback: when a tier admits
  // nothing, the most tiled legal point is the closest thing to satisfying it.
  int64_t maxLegal = candidates.back();

  llvm::SmallVector<int64_t> live(candidates.begin(), candidates.end());
  unsigned maxTier = 0;
  for (auto &c : criteria)
    maxTier = std::max(maxTier, c->tier());

  for (unsigned tier = 1; tier <= maxTier; ++tier) {
    for (auto &c : criteria) {
      if (c->tier() != tier)
        continue;
      CriterionTrace trace;
      trace.name = c->name();
      trace.tier = tier;
      trace.candidatesIn = live.size();

      llvm::SmallVector<int64_t> kept;
      std::string why;
      for (int64_t k : live) {
        std::string thisWhy;
        if (c->isFeasible(TileCandidate{k}, ctx, thisWhy))
          kept.emplace_back(k);
        else if (why.empty())
          why = thisWhy;
      }
      if (kept.empty()) {
        // Never silently: the fallback is reported, and it is the fallback the
        // caller sees in `iterNum`.
        trace.candidatesOut = 0;
        // Which ceiling was hit is the useful part. The vector row is a
        // correctness bound the model cannot trade away; the scalar row is one
        // that tiling could in principle widen.
        trace.why = ctx.vecRow && maxLegal == ctx.vecRow
                        ? "vector-row-bound"
                        : (why.empty() ? "infeasible" : why);
        d.why = trace.why;
        d.iterNum = maxLegal;
        d.perTierTrace.emplace_back(std::move(trace));
        return d;
      }
      live = std::move(kept);
      trace.candidatesOut = live.size();

      // Soft side: rank what survived. Costs of one tier add up, and a
      // criterion with no opinion contributes nothing.
      std::optional<int64_t> best;
      int64_t bestK = live.front();
      bool anyCost = false;
      for (int64_t k : live) {
        auto c0 = c->cost(TileCandidate{k}, ctx);
        if (!c0)
          continue;
        anyCost = true;
        if (!best || *c0 < *best) {
          best = c0;
          bestK = k;
        }
      }
      if (anyCost) {
        trace.chosenCost = best;
        live.assign(1, bestK);
        trace.candidatesOut = 1;
      }
      d.perTierTrace.emplace_back(std::move(trace));
    }
  }
  d.iterNum = live.front();
  if (d.why.empty())
    d.why = "budget";
  return d;
}

} // namespace xpu
} // namespace triton
} // namespace mlir
