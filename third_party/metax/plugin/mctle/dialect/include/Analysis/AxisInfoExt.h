#ifndef TLE_RAW_ANALYSIS_AXIS_INFO_EXT_H
#define TLE_RAW_ANALYSIS_AXIS_INFO_EXT_H

#include "triton/Analysis/AxisInfo.h"

namespace mlir::triton::mctle {

struct AxisInfoExt {
  static void addVisitors(mlir::triton::AxisInfoVisitorList &visitors);
};

class ModuleAxisInfoAnalysis : public mlir::triton::ModuleAxisInfoAnalysis {
public:
  explicit ModuleAxisInfoAnalysis(ModuleOp moduleOp)
      : mlir::triton::ModuleAxisInfoAnalysis(moduleOp,
                                             AxisInfoExt::addVisitors) {}
};

} // namespace mlir::triton::mctle

#endif
