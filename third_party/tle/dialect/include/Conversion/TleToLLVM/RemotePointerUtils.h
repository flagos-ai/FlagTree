#ifndef TLE_RAW_CONVERSION_TLETOLLVM_REMOTEPOINTERUTILS_H
#define TLE_RAW_CONVERSION_TLETOLLVM_REMOTEPOINTERUTILS_H

#include "mlir/IR/Value.h"

#include <cstdint>
#include <optional>

namespace mlir {
class ConversionPatternRewriter;
}

namespace mlir::triton {
class ModuleAxisInfoAnalysis;
}

namespace mlir::triton::tle {

struct RemotePointerInfo {
  std::optional<int32_t> constCTAId;
  Value dynamicCTAId;
  Value basePtr;
  Value vectorHintPtr;
  bool stripShardOffsetFromPtr = false;

  bool hasRemoteCTAId() const { return constCTAId || dynamicCTAId; }
};

bool isTlePointerValue(Value ptr);

RemotePointerInfo
getRemotePointerInfoFromValue(Value ptr, ConversionPatternRewriter &rewriter);

unsigned inferTlePointerVectorSize(Value ptr,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass);

unsigned inferTlePointerLayoutVectorHint(Value ptr);

} // namespace mlir::triton::tle

#endif
