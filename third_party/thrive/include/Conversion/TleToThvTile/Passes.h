#ifndef THRIVE_CONVERSION_TLETOTHVTILE_PASSES_H_
#define THRIVE_CONVERSION_TLETOTHVTILE_PASSES_H_

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::thvtile {

std::unique_ptr<mlir::Pass> createConvertTleToThvTilePass();

#define GEN_PASS_DECL
#include "Conversion/TleToThvTile/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "Conversion/TleToThvTile/Passes.h.inc"

} // namespace mlir::thvtile

#endif // THRIVE_CONVERSION_TLETOTHVTILE_PASSES_H_
