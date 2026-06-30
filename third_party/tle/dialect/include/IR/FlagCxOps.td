include "mlir/Dialect/LLVMIR/LLVMTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "tle/dialect/include/IR/TleDialect.td"
include "tle/dialect/include/IR/TleAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"

class Tle_Op<string mnemonic, list<Trait> traits = []>
    : Op<Tle_Dialect, mnemonic,
         !listconcat(traits, [VerifyTensorLayoutsTrait])> {}

def Tle_LocalPointerResultType : AnyTypeOf<[TT_Tensor, TT_Ptr, TT_Int]>;

def Tle_GetLocalRankOp : Tle_Op<"get local rank"> {
  let summary = "Get local rank";
  let arguments = (ins
    Tle_LocalPointerResultType:$src
  );
  let results = (outs I32:$result);
  let hasVerifier = 1;
}
