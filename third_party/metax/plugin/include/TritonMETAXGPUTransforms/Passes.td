#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonMETAXGPUAccelerateMatmul : Pass<"tritonmetaxgpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
    (e.g., METAX matrix cores)
  }];

  let constructor = "mlir::createTritonMETAXGPUAccelerateMatmulPass()";

  let dependentDialects = [];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"2",
           "number of pipeline stages">,
    Option<"disablePrefetch", "disable-prefetch",
           "bool", /*default*/"false",
           "disablePrefetch">,
    Option<"storeCoalesce", "store-coalesce",
           "bool", /*default*/"false",
           "storeCoalesce">,
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"80",
           "computeCapability">
    ];
}


def TritonMETAXGPUPipelineMACA : Pass<"tritonmetaxgpu-pipeline-maca", "mlir::ModuleOp"> {
  let summary = "pipeline-maca";

  let description = [{
  }];

  let constructor = "mlir::createTritonMETAXGPUPipelineMACAPass()";

  let dependentDialects = [];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"2",
           "number of pipeline stages">,
    Option<"pipelineLoadNum", "pipeline-load-num",
           "int32_t", /*default*/"-1",
           "number of pipeline load">,
    Option<"isFullStage", "is-full-stage",
           "bool", /*default*/"false",
           "enable full stages">,
    Option<"isSingleShm", "is-single-shm",
           "bool", /*default*/"false",
           "enable single shared memory">
  ];
}

def TritonMETAXGPUPipelineAsyncNT : Pass<"tritonmetaxgpu-pipeline-async-nt", "mlir::ModuleOp"> {
  let summary = "pipeline-async-nt";

  let description = [{
    The pipeline which supports InsertSliceAsyncOp, recommanded order of A, B input matrix is NT.
  }];

  let constructor = "mlir::createTritonMETAXGPUPipelineAsyncBasePass()";

  let dependentDialects = [];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"2",
           "number of pipeline stages">,
    Option<"isFullStage", "is-full-stage",
           "bool", /*default*/"false",
           "enable full stages">,
    Option<"mixed", "mixed",
           "bool", /*default*/"false",
           "do reg pipeline">,
  ];
}

def TritonMETAXGPUPipelineAsyncTN : Pass<"tritonmetaxgpu-pipeline-async-tn", "mlir::ModuleOp"> {
  let summary = "pipeline-async-tn";

  let description = [{
    The pipeline which supports InsertSliceAsyncOp, this pipeline requires that the order of
    A, B input matrix is TN.
  }];

  let constructor = "mlir::createTritonMETAXGPUPipelineAsyncTNPass()";

  let dependentDialects = [];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"2",
           "number of pipeline stages">,
    Option<"innerStageM", "inner-stage-M",
           "int32_t", /*default*/"0",
           "number of pipeline inner stage of M">,
    Option<"innerStageN", "inner-stage-N",
           "int32_t", /*default*/"0",
           "number of pipeline inner stage of N">
  ];
}

def TritonMETAXGPUPipelineAsyncTT : Pass<"tritonmetaxgpu-pipeline-async-tt", "mlir::ModuleOp"> {
  let summary = "pipeline-async-tt";

  let description = [{
    The pipeline which supports InsertSliceAsyncOp, this pipeline requires that the order of
    A, B input matrix is TT.
  }];

  let constructor = "mlir::createTritonMETAXGPUPipelineAsyncTTPass()";

  let dependentDialects = [];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"2",
           "number of pipeline stages">,
  ];
}

def TritonMETAXGPUChangeLayoutFromRepNToElemN: Pass<"tritonmetaxgpu-change-layout-from-elemn-to-repn", "mlir::ModuleOp"> {
  let summary = "change layout from repN to elemN";

  let description = "Change Blockedlayout from repN to elemN";

  let constructor = "mlir::createTritonMETAXGPUChangeLayoutFromRepNToElemNPass()";

  let dependentDialects = [];
}

def TritonMETAXGPUChangeLayoutForConstancyLoad: Pass<"tritonmetaxgpu-change-layout-for-constancy-load", "mlir::ModuleOp"> {
  let summary = "find and change to globally optimal layout for loadOp with constancy along contiguous dimension";

  let description = "find and change to globally optimal layout for loadOp with constancy along contiguous dimension";

  let constructor = "mlir::createTritonMETAXGPUChangeLayoutForConstancyLoadPass()";

  let dependentDialects = [];
}

def TritonMETAXGPUOptimizeCStorePass: Pass<"tritonmetaxgpu-optimize-cstore-pass", "mlir::ModuleOp"> {
  let summary = "eliminate convertlayout for optimize cstore";

  let description = "Eliminate convertlayout for optimize cstore";

  let constructor = "mlir::createTritonMETAXGPUOptimizeCStorePass()";

  let dependentDialects = [];
  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"2",
           "number of pipeline stages">,
  ];
}

def TritonMETAXGPUOptimizeSmemUsage: Pass<"tritonmetaxgpu-optimize-smem-usage", "mlir::ModuleOp"> {
  let summary = "Reduce convertlayout swizzle smem cost";

  let description = "Reduce convertlayout swizzle smem cost";

  let constructor = "mlir::createTritonMETAXGPUOptimizeSmemUsage()";

  let dependentDialects = [];
  let options = [
    Option<"forceNoVectorize", "force-no-vectorize",
           "bool", /*default*/"false",
           "force lds/sts vec size to be 1 in specific convertLayout case">,
  ];
}

def TritonMETAXGPUChangeLayoutForInt8: Pass<"tritonmetaxgpu-change-layout-for-int8", "mlir::ModuleOp"> {
  let summary = "change convertlayout for int8";

  let description = "Change convertlayout for int8";

  let constructor = "mlir::createTritonMETAXGPUChangeLayoutForInt8Pass()";

  let dependentDialects = [];
  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"2",
           "number of pipeline stages">,
    Option<"pipeline", "pipeline",
           "std::string", /*default=*/"std::string{}",
           "name of pipeline.">,
  ];
}

def TritonMETAXGPUAddPtrOptPass: Pass<"tritonmetaxgpu-addptr-opt-pass", "mlir::ModuleOp"> {
  let summary = "optimize addptr for pipeline";

  let description = "Optimize addptr for pipeline";

  let constructor = "mlir::createTritonMETAXGPUAddPtrOptPass()";

  let dependentDialects = [];
  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"2",
           "number of pipeline stages">,
    Option<"isFullStage", "is-full-stage",
           "bool", /*default*/"false",
           "enable full stages">,
    Option<"mixed", "mixed",
           "bool", /*default*/"false",
           "do reg pipeline">,
  ];
}

def TritonMETAXGPUChangeTransOpGraph: Pass<"tritonmetaxgpu-change-transop-graph", "mlir::ModuleOp"> {
  let summary = "Change TransOp graph";

  let description = "Change TransOp graph";

  let constructor = "mlir::createTritonMETAXGPUChangeTransOpGraphPass()";

  let dependentDialects = [];
}

#endif
