#ifndef THRIVE_CONVERSION_TLETOTHVTILE_TLETOTHVTILEPATTERNS_H_
#define THRIVE_CONVERSION_TLETOTHVTILE_TLETOTHVTILEPATTERNS_H_

#include "Conversion/TleToThvTile/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::thvtile {

void populateSliceOpsToThvTilePatterns(TleToThvTileTypeConverter &typeConverter,
                                       RewritePatternSet &patterns);

void populateCumsumOpToThvTilePatterns(TleToThvTileTypeConverter &typeConverter,
                                       RewritePatternSet &patterns);

} // namespace mlir::thvtile

#endif // THRIVE_CONVERSION_TLETOTHVTILE_TLETOTHVTILEPATTERNS_H_
