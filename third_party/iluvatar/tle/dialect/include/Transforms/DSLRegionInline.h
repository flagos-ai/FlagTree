#ifndef ILUVATAR_TLE_DSLREGIONINLINE_H
#define ILUVATAR_TLE_DSLREGIONINLINE_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::iluvatar_tle {
void populateDSLRegionInlinePatterns(RewritePatternSet &patterns);
}

#endif
