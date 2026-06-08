#ifndef MUSA_TLE_OPS
#define MUSA_TLE_OPS

#ifdef __TLE__
include "mlir/Interfaces/SideEffectInterfaces.td"
include "MUSATLEDialect.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"

class MUSATLE_Op<string mnemonic, list<Trait> traits = []>
    : Op<MUSATLE_Dialect, mnemonic,
         !listconcat(traits, [VerifyTensorLayoutsTrait])> {}

def MUSATLE_LocalPointerResultType : AnyTypeOf<[TT_Tensor, TT_Ptr]>;
def MUSATLE_LocalPointerIndexType : AnyTypeOf<[TT_Tensor, TT_Int]>;

def MUSATLE_LocalPointersOp : MUSATLE_Op<"local_pointers", [Pure]> {
  let arguments = (ins TTG_MemDescType:$src,
      Variadic<MUSATLE_LocalPointerIndexType>:$indices);
  let results = (outs MUSATLE_LocalPointerResultType:$result);
  let hasVerifier = 1;
}
#endif // __TLE__

#endif // MUSA_TLE_OPS
