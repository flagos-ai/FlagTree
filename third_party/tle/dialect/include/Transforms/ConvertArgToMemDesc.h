#ifndef TRITON_TLE_CONVERTARGTOMEMDESC_H
#define TRITON_TLE_CONVERTARGTOMEMDESC_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::tle {
void populateConvertArgToMemDescPatterns(RewritePatternSet &patterns);
}

#endif
