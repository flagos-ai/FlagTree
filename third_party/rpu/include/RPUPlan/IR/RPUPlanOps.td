#ifndef RPU_PLAN_OPS
#define RPU_PLAN_OPS

include "RPUPlan/IR/RPUPlanDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class RPUPlan_Op<string mnemonic, list<Trait> traits = []> :
    Op<RPUPlan_Dialect, mnemonic, traits>;

def RPUPlan_KernelOp : RPUPlan_Op<"kernel", [Symbol]> {
  let summary = "Internal RPU kernel plan";

  let description = [{
    `rpu_plan.kernel` is a regionless symbol op that describes one recognized
    RPU kernel plan. It is not executable IR and does not model tile, load,
    store, memory-scope, or async-pipe behavior.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$kernel_name,
    FlatSymbolRefAttr:$source_func,
    I32Attr:$version,
    StrAttr:$pattern,
    DictionaryAttr:$signature,
    DictionaryAttr:$shape,
    DictionaryAttr:$args,
    DictionaryAttr:$layout,
    DictionaryAttr:$mask,
    ArrayAttr:$required_dsl_features,
    DictionaryAttr:$emission
  );

  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

#endif
