#ifndef EV_LINALG_PASSES
#define EV_LINALG_PASSES

include "mlir/Pass/PassBase.td"

def SplitComputationalOp:  Pass<"ev-split-computational-op", "mlir::ModuleOp"> {
  let summary = "split computational operations as functions";
  let description = [{
    split computational operations as functions to distinct with memory load/store related operations.
  }];
  let constructor = "mlir::triton::ev::createSplitComputationalOpPass()";
  let dependentDialects = ["mlir::tensor::TensorDialect", "mlir::linalg::LinalgDialect"];
}

def InsertDeallocOp:  Pass<"ev-insert-dealloc-op", "mlir::ModuleOp"> {
  let summary = "insert memref.dealloc for memref.alloc based on the liveness analysis";
  let description = [{
    insert memref.dealloc for memref.alloc based on the liveness analysis.
  }];
  let constructor = "mlir::triton::ev::createInsertDeallocOpPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def SetDeviceInfo:  Pass<"ev-set-device-info", "mlir::ModuleOp"> {
  let summary = "set device info like memscope and corebind to subkernel";
  let description = [{
    set device info like memscope and corebind to subkernel.
  }];
  let constructor = "mlir::triton::ev::createSetDeviceInfoPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def MemoryAlloc : Pass<"ev-memory-alloc", "mlir::ModuleOp"> {
  let summary = "Pass to perform memory allocation for the module";
  let constructor = "mlir::triton::ev::createMemoryAllocPass()";
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::func::FuncDialect",
    "mlir::bufferization::BufferizationDialect"
  ];

  let options = [
    Option<"memScope", "mem-scope", "size_t", /*default=*/"0",
           "Memory scope considered to allocate. 0 indicates all spaces">,
    Option<"alignment", "alignment", "size_t", /*default=*/"128",
           "Memory address alignment, must be power of 2">,
    Option<"preview", "preview", "bool", /*default=*/"false",
           "Whether to preview allocation result.">,
    Option<"bankopt", "bankopt", "bool", /*default=*/"false",
           "Whether to enable bank alone optimization.">,
  ];
}

def DoubleBuffer :  Pass<"ev-double-buffer", "mlir::ModuleOp"> {
  let summary = "double buffer optimization";
  let description = [{
    perform double buffer on the specific value in for loops.
  }];
  let constructor = "mlir::triton::ev::createDoubleBufferPass()";
  let dependentDialects = ["mlir::tensor::TensorDialect", "mlir::linalg::LinalgDialect"];
}

def MaterializeAnnotation : Pass<"ev-materialize-annotation", "mlir::ModuleOp"> {
  let summary = "materialize annotation to device info";
  let description = [{
    materialize annotation to device info like memscope and corebind.
  }];
  let constructor = "mlir::triton::ev::createMaterializeAnnotationPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def RewriteFuncOpArgsType : Pass<"ev-rewrite-func-op-args-type", "mlir::ModuleOp"> {
  let summary = "rewrite func op args type to memref";
  let description = [{
    rewrite func op args type to memref.
  }];
  let constructor = "mlir::triton::ev::createRewriteFuncOpArgsTypePass()";
}

def RemoveLoopIterArgsWithMemrefType : Pass<"ev-linalg-inplace-optimize", "mlir::ModuleOp"> {
  let summary = "inplace optimize";
  let description = [{
    inplace optimize.
  }];
  let constructor = "mlir::triton::ev::createRemoveLoopIterArgsWithMemrefTypePass()";
}

def RemoveScalar : Pass<"remove-scalar", "mlir::ModuleOp"> {
  let summary = "Convert the scalar in the ir";
  let constructor = "mlir::triton::ev::createRemoveScalarPass()";
  let description = [{
    Scalar reading is prone to cache consistency issues, so it is necessary to reduce or avoid the occurrence of scalars.
  }];
}

def Bufferize :  Pass<"ev-bufferize", "mlir::ModuleOp"> {
  let summary = "bufferize pass adapted for ev backend";
  let description = [{
    bufferize pass adapted for ev backend
  }];
  let constructor = "mlir::triton::ev::createBufferizePass()";
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def EncapsulateLinalgOp : Pass<"ev-encapsulate-linalg-op", "mlir::ModuleOp"> {
  let summary = "Encapsulate Linalg operations into sub-functions";
  let description = [{
    This pass extracts Linalg operations (such as linalg.generic, linalg.matmul, etc.)
    into separate sub-functions (FuncOp) and inserts a call (CallOp) at the original location,
    enabling further optimization and analysis.
  }];
  let constructor = "mlir::triton::ev::createEncapsulateLinalgOpPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect"
  ];
}

def SetMemRefScope : Pass<"ev-set-memref-scope", "mlir::ModuleOp"> {
  let summary = "Set memref scope attributes using type converter in dialect conversion";
  let description = [{
    This pass utilizes a type converter in dialect conversion to set the scope attribute
    of all memref types following these rules:
    1. If the memref type is scalar, set to ev::MemScope::DDR
    2. If memref.alloc op is met, set the result type to ev::MemScope::MM
    3. Set function signature input arguments memref type to DDR
  }];
  let constructor = "mlir::triton::ev::createSetMemRefScopePass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def MemoryPromotion : Pass<"ev-memory-promotion", "mlir::ModuleOp"> {
  let summary = "Promote memory scope of linalg.matmul operations from MM to AM(5) and add memref.copy back to MM";
  let description = [{
    This pass promote AM/MM and DDR/L2 pair.
  }];
  let constructor = "mlir::triton::ev::createMemoryPromotionPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def RemoveRedundencyCopy : Pass<"ev-remove-redundency-copy", "mlir::ModuleOp"> {
  let summary = "Optimize redundant memref.copy operations";
  let description = [{
    This pass eliminates redundant memref.copy operations through several optimizations:
    1. Eliminate self-copy: copy(A, A) -> no-op
    2. Chain copy elimination: copy(A, B) + copy(B, C) -> copy(A, C)
    3. Eliminate redundant copy: remove duplicate copy(A, B) operations
  }];
  let constructor = "mlir::triton::ev::createRemoveRedundencyCopyPass()";
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::func::FuncDialect"
  ];
}

def RewriteDataType : Pass<"ev-rewrite-data-type", "mlir::ModuleOp"> {
  let summary = "Rewrite illegal data types using pattern rewriting";
  let description = [{
    Rewrite illegal data types like i1 to i8 and i64 to i32 using pattern rewriting
  }];
  let constructor = "mlir::triton::ev::createRewriteDataTypePass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::func::FuncDialect"];
}
#endif
