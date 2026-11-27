#ifndef NV_TLE_RAW_PASSES
#define NV_TLE_RAW_PASSES

include "mlir/Pass/PassBase.td"

def NvidiaMaterializeDeferredRaw
    : Pass<"tle-materialize-deferred-raw", "mlir::ModuleOp"> {
  let summary = "Materialize deferred TLE raw calls from compiled CUDA sources";
  let description = [{
    Imports each compiled CUDA source once and binds deferred raw calls to
    the corresponding source-local wrapper definitions.
  }];
  let dependentDialects = ["mlir::triton::tle::TleDialect"];
}

#endif // NV_TLE_RAW_PASSES
