#ifndef THRIVE_CONVERSION_TLETOTHVTILE_TYPECONVERTER_H_
#define THRIVE_CONVERSION_TLETOTHVTILE_TYPECONVERTER_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::thvtile {

class TleToThvTileTypeConverter : public TypeConverter {
public:
  explicit TleToThvTileTypeConverter() {
    // Pass through all types unchanged by default.
    addConversion([](Type type) { return type; });
  }
};

} // namespace mlir::thvtile

#endif // THRIVE_CONVERSION_TLETOTHVTILE_TYPECONVERTER_H_
