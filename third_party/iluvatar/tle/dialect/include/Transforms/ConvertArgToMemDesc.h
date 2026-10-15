#ifndef ILUVATAR_TLE_CONVERTARGTOMEMDESC_H
#define ILUVATAR_TLE_CONVERTARGTOMEMDESC_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::iluvatar_tle {
void populateConvertArgToMemDescPatterns(RewritePatternSet &patterns);
}

#endif
