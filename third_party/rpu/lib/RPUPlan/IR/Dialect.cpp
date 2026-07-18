#include "RPUPlan/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h"

#include "RPUPlan/IR/RPUPlanDialect.cpp.inc"

using namespace mlir;
using namespace mlir::rpu::plan;

void RPUPlanDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "RPUPlan/IR/RPUPlanOps.cpp.inc"
      >();
}
