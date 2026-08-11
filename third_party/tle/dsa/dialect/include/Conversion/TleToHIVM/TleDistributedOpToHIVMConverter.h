// Copyright 2026- Xcoresigma Technology Co., Ltd

#ifndef TRITON_TLE_CONVERSION_TLE_DISTRIBUTED_OP_TO_HIVM_CONVERTER_H_
#define TRITON_TLE_CONVERSION_TLE_DISTRIBUTED_OP_TO_HIVM_CONVERTER_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton::tle {
void populateTleDistributedOpToHIVMConversionPatterns(
    mlir::RewritePatternSet &patterns, PatternBenefit benefit = 1,
    bool existDot = false);
}
#endif
