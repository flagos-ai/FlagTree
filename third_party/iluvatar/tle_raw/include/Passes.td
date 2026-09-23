#ifndef ILUVATAR_TLE_RAW_PASSES
#define ILUVATAR_TLE_RAW_PASSES

include "mlir/Pass/PassBase.td"

def IluvatarMaterializeDeferredRaw
    : Pass<"tle-materialize-deferred-raw", "mlir::ModuleOp"> {
  let summary = "Materialize deferred TLE raw DSL regions from pending CUDA sources";
  let description = [{
    Fills the bodies of ``iluvatar_tle.dsl_region`` ops that carry a
    ``tle_raw.source_id`` attribute by looking up registered pending CUDA
    sources at compile time.
  }];
  let dependentDialects = ["mlir::triton::iluvatar_tle::IluvatarTleDialect"];
}

#endif // ILUVATAR_TLE_RAW_PASSES
