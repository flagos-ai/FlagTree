#ifndef TRITON_DIALECT_FLAGTREE_TRANSFORMS_DSLREGIONINLINE_H
#define TRITON_DIALECT_FLAGTREE_TRANSFORMS_DSLREGIONINLINE_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::flagtree {
void populateDSLRegionInlinePatterns(RewritePatternSet &patterns);
}

#endif
