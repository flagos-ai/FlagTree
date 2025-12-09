#ifndef TRITON_DIALECT_FLAGTREE_TRANSFORMS_CONVERTARGTOMEMDESC_H
#define TRITON_DIALECT_FLAGTREE_TRANSFORMS_CONVERTARGTOMEMDESC_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::flagtree {
void populateConvertArgToMemDescPatterns(RewritePatternSet &patterns);
}

#endif
