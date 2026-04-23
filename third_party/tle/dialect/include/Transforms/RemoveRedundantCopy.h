#ifndef TRITON_TLE_REMOVEREDUNDANTCOPY_H
#define TRITON_TLE_REMOVEREDUNDANTCOPY_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::tle {
void populateRemoveRedundantCopyPatterns(RewritePatternSet &patterns);
}

#endif
