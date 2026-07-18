#ifndef TRITONXPU_INTERFACES
#define TRITONXPU_INTERFACES


include "mlir/Interfaces/SideEffectInterfaces.td" // MemoryEffects
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/ControlFlowInterfaces.td" // ReturnLike

include "triton/Dialect/Triton/IR/TritonInterfaces.td" // SameOperandsEncoding, SameOperandsAndResultEncoding

#endif // TRITONXPU_INTERFACES
