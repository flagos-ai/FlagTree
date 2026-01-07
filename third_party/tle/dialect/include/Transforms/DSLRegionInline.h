#ifndef TRITON_TLE_DSLREGIONINLINE_H
#define TRITON_TLE_DSLREGIONINLINE_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::tle {
void populateDSLRegionInlinePatterns(RewritePatternSet &patterns);
}

#endif
