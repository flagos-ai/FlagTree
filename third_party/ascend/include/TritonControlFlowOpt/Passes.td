#ifndef TRITON_CONTROL_FLOW_OPT_PASSES
#define TRITON_CONTROL_FLOW_OPT_PASSES

include "mlir/Pass/PassBase.td"

def TritonControlFlowOpt : Pass<"triton-control-flow-opt", "mlir::ModuleOp"> {
  let summary = "Normalize Triton control flow before pointer analysis";
  let constructor = "triton::createTritonControlFlowOptPass()";
}

#endif // TRITON_CONTROL_FLOW_OPT_PASSES
