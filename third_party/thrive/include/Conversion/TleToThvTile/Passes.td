#ifndef THRIVE_CONVERSION_TLETOTHVTILE_PASSES_TD
#define THRIVE_CONVERSION_TLETOTHVTILE_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertTleToThvTile : Pass<"convert-tle-to-thvtile", "mlir::ModuleOp"> {
  let summary = "Convert TLE dialect ops to ThvTile dialect ops";
  let description = [{
    Lowers TLE (Triton Language Extension) operations to ThvTile operations.
    Supports converting ExtractTileOp, InsertTileOp, and ExclusiveCumsumOp
    to thvtile.extract_slice / thvtile.insert_slice / thvtile.scan, where the
    exclusive cumulative sum is emitted as a scan with `exclusive = true` so
    the result is produced directly.
  }];

  let constructor = "mlir::thvtile::createConvertTleToThvTilePass()";
  let dependentDialects = [
    "mlir::triton::tle::TleDialect",
    "mlir::thvtile::ThvTileDialect",
    "mlir::arith::ArithDialect",
  ];
}

#endif // THRIVE_CONVERSION_TLETOTHVTILE_PASSES_TD
