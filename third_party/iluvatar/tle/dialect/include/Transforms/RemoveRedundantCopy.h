#ifndef ILUVATAR_TLE_REMOVEREDUNDANTCOPY_H
#define ILUVATAR_TLE_REMOVEREDUNDANTCOPY_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::iluvatar_tle {
void populateRemoveRedundantCopyPatterns(RewritePatternSet &patterns);
}

#endif
