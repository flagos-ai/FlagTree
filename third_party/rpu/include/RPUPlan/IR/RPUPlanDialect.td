#ifndef RPU_PLAN_DIALECT
#define RPU_PLAN_DIALECT

include "mlir/IR/OpBase.td"

def RPUPlan_Dialect : Dialect {
  let name = "rpu_plan";
  let cppNamespace = "::mlir::rpu::plan";

  let summary = "Internal RPU kernel plan dialect";

  let description = [{
    RPUPlan is an internal compiler dialect. It records the structured plan
    produced by RPU TTIR recognition before the plan is exported as stable
    rpuplan JSON or consumed by later RPU DSL emission passes.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect"
  ];
}

#endif
