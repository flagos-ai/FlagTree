#ifndef TRITON_ARITH_TO_LINALG_CONVERSION_PASSES
#define TRITON_ARITH_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonArithToLinalg : Pass<"triton-arith-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert Triton arithmetic operations into linalg";
  let options = [
      Option<"pidsToFuncArgs", "pids-to-func-args", "bool", /*default*/"true",
             "Convert tt.get_program_id and tt.get_num_programs to reference to function arguments">,
      Option<"ttToFuncFunc", "tt-to-func-func", "bool", /*default*/"true",
             "Convert tt.func to func.func">,
      Option<"addptrToLinalg", "addptr-to-linalg", "bool", /*default*/"true",
             "Convert tt.addptr on tensors to linalg">,
      Option<"assertToCf", "assert-to-cf", "bool", /*default*/"true",
             "Convert tt.assert to cf.assert">,
      Option<"tensorPtrToLinalg", "tensor-ptr-to-linalg", "bool", /*default*/"false",
             "Convert triton ops on tensor of pointers to linalg.generic">,
  ];
}

#endif
