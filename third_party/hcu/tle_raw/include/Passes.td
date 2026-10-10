#ifndef HCU_TLE_RAW_PASSES
#define HCU_TLE_RAW_PASSES

include "mlir/Pass/PassBase.td"

def HcuMaterializeDeferredRaw
    : Pass<"hcu-tle-materialize-deferred-raw", "mlir::ModuleOp"> {
  let summary = "Materialize deferred TLE raw DSL regions from pending HCU sources";
  let description = [{
    Fills the bodies of ``tle.dsl_region`` ops that carry a ``tle_raw.source_id``
    attribute by looking up registered pending HCU sources at compile time.
  }];
  let dependentDialects = ["mlir::triton::tle::TleDialect"];
}

#endif // HCU_TLE_RAW_PASSES
