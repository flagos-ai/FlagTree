#ifndef NV_TLE_RAW_PASSES
#define NV_TLE_RAW_PASSES

include "mlir/Pass/PassBase.td"

def NvidiaMaterializeDeferredRaw
    : Pass<"tle-materialize-deferred-raw", "mlir::ModuleOp"> {
  let summary = "Materialize deferred TLE raw DSL regions from pending CUDA sources";
  let description = [{
    Fills the bodies of ``tle.dsl_region`` ops that carry a ``tle_raw.source_id``
    attribute by looking up registered pending CUDA sources at compile time.
  }];
  let dependentDialects = ["mlir::triton::tle::TleDialect"];
}

#endif // NV_TLE_RAW_PASSES
