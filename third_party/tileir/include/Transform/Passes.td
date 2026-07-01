#ifndef TRITON_TILEIR_TRANSFORM_PASSES
#define TRITON_TILEIR_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def RewriteAssumeWithCudaTile : Pass</*cli-arg*/"rewrite-assume-with-cuda-tile", /*Op*/"mlir::ModuleOp"> {
  let summary = "Rewrite llvm.intr.assume operations into cuda_tile.assume operations";
  let description = [{
    This pass rewrites patterns like:
    ```
    %0 = constant dense<16> : tile<i64>
    %1 = constant dense<0> : tile<i64>
    %38 = bitcast %arg0 : tile<ptr<f16>> -> tile<i64>
    %39 = remi %38, %0 : tile<i64>
    %40 = cmpi eq, %39, %1 : tile<i64>
    %41 = builtin.unrealized_conversion_cast %40 : tile<i1> to i1
    llvm.intr.assume %41 : i1
    ```
    into:
    ```
    assume div_by<16 : i64>, %arg0: tile<ptr<f16>>
    ```

    It also supports integer types (i32 and i64) and rewrites patterns like:
    ```
    %6 = constant dense<8> : tile<i32> loc(#loc1)
    %10 = constant dense<0> : tile<i32> loc(#loc1)
    %54 = remi %46, %6  : tile<i32> loc(#loc38)
    %55 = cmpi eq, %54, %10 : tile<i32> loc(#loc39)
    %56 = builtin.unrealized_conversion_cast %55 : tile<i1> to i1 loc(#loc39)
    llvm.intr.assume %56 : i1 loc(#loc40)
    ```
    into:
    ```
    assume div_by<8 : i64>, %46 : tile<i32>
    ```

    There may be more patterns in the future.   
    If there are no patterns matched, the llvm.intr.assume will be removed without any new op.
    
    This transformation allows the compiler to better understand alignment assumptions
    and potentially generate more efficient code.
  }];

  let constructor = "mlir::triton::createRewriteAssumeWithCudaTilePass()";

  let dependentDialects = ["mlir::triton::TritonDialect", "::mlir::cuda_tile::CudaTileDialect", "mlir::LLVM::LLVMDialect"];
}

def LiftTTCFToSCF : Pass</*cli-arg*/"lift-tt-cf-to-scf", /*Op*/"mlir::ModuleOp"> {
  let summary = "Lift ControlFlow dialect (cf) to SCF dialect inside tt.func";
  let description = [{
    This pass applies MLIR's ControlFlowToSCF transformation to regions nested under
    Triton `tt.func`. It structurizes `cf` control flow (e.g., `cf.cond_br`, `cf.switch`)
    into `scf` constructs so downstream conversions (to cuda_tile) can rely on SCF.
  }];
  let constructor = "mlir::triton::createLiftTTCFToSCFPass()";
  let dependentDialects = ["mlir::triton::TritonDialect", "mlir::cf::ControlFlowDialect", "mlir::scf::SCFDialect", "mlir::ub::UBDialect"];
}

def AutoGenMemoryToken : Pass</*cli-arg*/"auto-gen-memory-token", /*Op*/"mlir::ModuleOp"> {
  let summary = "Automatically generate memory tokens for debug_barrier and cuda_tile memory operations";
  let description = [{
    This pass automatically generates memory tokens for debug_barrier in a serialized manner.
    It also generates memory tokens for cuda_tile memory operations that have alias memory access patterns to ensure their access order, kernels
    which already has user-added memory tokens will be ignored by this pass.

    A simple example looks like this:
    ```
    %1, %token_1 = load_ptr_tko weak %ptr : tile<ptr<i32>> -> tile<i32>, token
    %token2 = store_ptr_tko weak %ptr, %data : tile<ptr<i32>>, tile<i32> -> token
    ```
    will be modified into:
    ```
    %0 = make_token : token
    %1, %token_1 = load_ptr_tko weak %ptr token=%0 : tile<ptr<i32>> -> tile<i32>, token
    %token2 = store_ptr_tko weak %ptr, %data token=%token_1 : tile<ptr<i32>>, tile<i32> -> token
    ```

    For more examples, refer to the test cases in `test/FileCheck/op-conversion-auto-memtoken.mlir`.
  }];

  let constructor = "mlir::triton::createAutoGenMemoryTokenPass()";

  let dependentDialects = ["::mlir::cuda_tile::CudaTileDialect"];
  let options = [
    Option<"enable_autogen_alias_mem_token", "autogen-alias-memtoken", "bool",
           /*default=*/"true",
           "Automatically generate memory token for memory ops with alias memory access.">
    ];
}

#endif
