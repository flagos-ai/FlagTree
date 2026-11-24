#ifndef TRITON_XPU_ANALYSIS_STRUCTUREDIDANALYSIS_H
#define TRITON_XPU_ANALYSIS_STRUCTUREDIDANALYSIS_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::triton::sdnn {

/// The three coefficients of `dst[i][j] = base + stepRow*i + stepCol*j`, as
/// integer scalar SSA values.
///
/// The three coefficients form the whole content of an `sdnn.ew
/// structured_ids` tile: `base` is the value at (0, 0), `stepCol` advances it
/// along the innermost axis and `stepRow` along the outermost one.  They are
/// integer values here -- the consumer converts them to the f32 the EW
/// coefficient registers take.
struct StructuredId {
  Value base, stepRow, stepCol;
};

/// Prove that an integer tile is a 2-D affine function of index ramps.
///
/// Why: arch5 has no i32-to-float conversion path at all (an i32 DMA source is
/// encoded as f32 and `dmaCfgDataType5` rejects a converting transfer), so a
/// `arith.sitofp` over an index tile can only be *regenerated* on the EW
/// engine, and only the affine part of a tile can be regenerated.  The proof
/// therefore has to be total: a partial decode would emit a tile that looks
/// right but silently drops one additive term.  Anything outside the
/// whitelisted ops -- `divsi`, `remsi`, `tt.load`, `arith.select`, a block
/// argument, an unrecognised op -- fails the whole tile.
///
/// How:
///   - decode the tile into `constant + sum_d steps[d] * index_d` over the
///     whitelisted ops (`tt.make_range`, `tt.splat`, `tt.expand_dims`,
///     `tt.broadcast`, `tensor.expand_shape`, `arith.addi` / `subi` /
///     `muli` with one constant side, `arith.constant`);
///   - map dim 0 of a rank-2 tile onto `stepRow` and dim 1 onto `stepCol`;
///   - treat a rank-1 tile as a single row, so its only step is `stepCol` and
///     `stepRow` comes back as a literal zero.
///
/// The coefficients are materialised as `arith.constant` i32 ops, so a caller
/// inside a rewrite pattern should build its op from them immediately; they
/// live immediately before the tile's root op and are folded by CSE when they
/// are literals.  See the note on the pass for why the frozen signature leaves
/// no builder parameter.
///
/// \param tile the integer tile feeding an `arith.sitofp`.
/// \param ids receives the three coefficients on success.
/// \returns failure when any part of the tile is not a proven affine form.
LogicalResult analyze(Value tile, StructuredId &ids);

/// Ask the same question as `analyze` without touching the IR.
///
/// Why: a caller that has to decide *before* rewriting -- splitting an
/// `arith.sitofp` across the `arith.select` that feeds it only pays off when
/// both branches can be generated -- would otherwise have to run `analyze` and
/// then hunt down the coefficients it materialised for the branch that did
/// decode.  This probe runs the same decoder and erases everything it built,
/// so a declined tile leaves the function byte for byte.
///
/// \param tile the integer tile a caller is about to hand to `analyze`.
/// \returns true when `analyze` would succeed on it.
bool isAffineIdTile(Value tile);

} // namespace mlir::triton::sdnn

#endif // TRITON_XPU_ANALYSIS_STRUCTUREDIDANALYSIS_H
