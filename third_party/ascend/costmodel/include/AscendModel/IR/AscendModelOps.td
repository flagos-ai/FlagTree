//===- AscendModelOps.td - AscendModel dialect operations -----------------===//
//
// Defines operations for the AscendModel dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MODEL_OPS_TD
#define ASCEND_MODEL_OPS_TD

include "AscendModelBase.td"
include "AscendModelInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

//===----------------------------------------------------------------------===//
// Cube Core Operations (矩阵计算)
//===----------------------------------------------------------------------===//

def Ascend_MatmulOp : AscendModel_Op<"matmul", [Pure, DeclareOpInterfaceMethods<EstimateCyclesOpInterface, ["getFlops"]>]> {
  let summary = "Matrix multiplication on Cube Core";
  let description = [{
    Performs matrix multiplication C = A @ B on the Cube Core.
    This operation is the primary compute-bound operation on Ascend 910B.
    The Cube Core executes in 16x16x16 blocks for optimal performance.
  }];

  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs,
    I64Attr:$M,
    I64Attr:$N,
    I64Attr:$K,
    OptionalAttr<I64Attr>:$estimated_cycles,
    OptionalAttr<I64Attr>:$op_id
  );

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Memory Transfer Operations
//===----------------------------------------------------------------------===//

def Ascend_CubeLoadOp : AscendModel_Op<"cube_load", [DeclareOpInterfaceMethods<EstimateCyclesOpInterface, ["getTransferBytes", "getSrcMemSpace", "getDstMemSpace"]>]> {
  let summary = "Load data for Cube Core via MTE2";
  let description = [{
    Loads tensor data from memory to Cube Core's local buffer via MTE2.
    ``src_space``/``dst_space`` select the per-(src,dst,corenum) bandwidth
    (tilesim migration). Defaults: hbm -> l1.
  }];

  let arguments = (ins
    AnyType:$source,
    I64Attr:$bytes,
    OptionalAttr<StrAttr>:$src_space,
    OptionalAttr<StrAttr>:$dst_space,
    OptionalAttr<I64Attr>:$estimated_cycles,
    OptionalAttr<I64Attr>:$op_id
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];
}

def Ascend_CubeStoreOp : AscendModel_Op<"cube_store", [DeclareOpInterfaceMethods<EstimateCyclesOpInterface, ["getTransferBytes", "getSrcMemSpace", "getDstMemSpace"]>]> {
  let summary = "Store Cube Core result via FixPipe";
  let description = [{
    Stores Cube Core computation results to memory via FixPipe.
    ``src_space``/``dst_space`` select the bandwidth (tilesim migration).
    Defaults: l0c -> hbm.
  }];

  let arguments = (ins
    AnyType:$data,
    I64Attr:$bytes,
    OptionalAttr<StrAttr>:$src_space,
    OptionalAttr<StrAttr>:$dst_space,
    OptionalAttr<I64Attr>:$estimated_cycles,
    OptionalAttr<I64Attr>:$op_id
  );

  let results = (outs);

  let assemblyFormat = [{
    $data attr-dict `:` type($data)
  }];
}

def Ascend_VectorLoadOp : AscendModel_Op<"vector_load", [DeclareOpInterfaceMethods<EstimateCyclesOpInterface, ["getTransferBytes", "getSrcMemSpace", "getDstMemSpace"]>]> {
  let summary = "Load data for Vector Core via VecMTE2";
  let description = [{
    Loads tensor data for Vector Core operations via dedicated MTE2.
    ``src_space``/``dst_space`` select the per-(src,dst,corenum) bandwidth
    (tilesim migration). Defaults: hbm -> ub.
  }];

  let arguments = (ins
    AnyType:$source,
    I64Attr:$bytes,
    OptionalAttr<StrAttr>:$src_space,
    OptionalAttr<StrAttr>:$dst_space,
    OptionalAttr<I64Attr>:$estimated_cycles,
    OptionalAttr<I64Attr>:$op_id
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];
}

def Ascend_VectorStoreOp : AscendModel_Op<"vector_store", [DeclareOpInterfaceMethods<EstimateCyclesOpInterface, ["getTransferBytes", "getSrcMemSpace", "getDstMemSpace"]>]> {
  let summary = "Store Vector Core result via MTE3";
  let description = [{
    Stores Vector Core computation results to memory via MTE3.
    ``src_space``/``dst_space`` select the bandwidth (tilesim migration).
    Defaults: ub -> hbm.
  }];

  let arguments = (ins
    AnyType:$data,
    I64Attr:$bytes,
    OptionalAttr<StrAttr>:$src_space,
    OptionalAttr<StrAttr>:$dst_space,
    OptionalAttr<I64Attr>:$estimated_cycles,
    OptionalAttr<I64Attr>:$op_id
  );

  let results = (outs);

  let assemblyFormat = [{
    $data attr-dict `:` type($data)
  }];
}

//===----------------------------------------------------------------------===//
// Vector Core Binary Operations (向量二元运算)
//===----------------------------------------------------------------------===//

// Base class for simple vector binary ops (1 cycle per vector op)
class Ascend_VectorBinarySimple<string mnemonic>
    : AscendModel_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<EstimateCyclesOpInterface, ["getFlops"]>]> {
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs,
    OptionalAttr<I64Attr>:$estimated_cycles,
    OptionalAttr<I64Attr>:$op_id
  );

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
}

// Base class for complex vector binary ops (multiple cycles per vector op)
class Ascend_VectorBinaryComplex<string mnemonic, int cycles_per_op>
    : AscendModel_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<EstimateCyclesOpInterface, ["getCyclesPerVectorOp", "getFlops"]>]> {
  int cyclesPerOp = cycles_per_op;

  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs,
    OptionalAttr<I64Attr>:$estimated_cycles,
    OptionalAttr<I64Attr>:$op_id
  );

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
}

// Simple binary ops (1 cycle)
def Ascend_AddOp : Ascend_VectorBinarySimple<"add"> {
  let summary = "Element-wise addition on Vector Core";
}

def Ascend_SubOp : Ascend_VectorBinarySimple<"sub"> {
  let summary = "Element-wise subtraction on Vector Core";
}

def Ascend_MulOp : Ascend_VectorBinarySimple<"mul"> {
  let summary = "Element-wise multiplication on Vector Core";
}

def Ascend_MaxOp : Ascend_VectorBinarySimple<"max"> {
  let summary = "Element-wise maximum on Vector Core";
}

def Ascend_MinOp : Ascend_VectorBinarySimple<"min"> {
  let summary = "Element-wise minimum on Vector Core";
}

// Complex binary ops (4 cycles - division is expensive)
def Ascend_DivOp : Ascend_VectorBinaryComplex<"div", 4> {
  let summary = "Element-wise division on Vector Core";
}

//===----------------------------------------------------------------------===//
// Vector Core Comparison Operations (向量比较运算)
//===----------------------------------------------------------------------===//

// Base class for comparison ops (result is i1 tensor, 1 cycle per vector op)
class Ascend_VectorCmpOp<string mnemonic>
    : AscendModel_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<EstimateCyclesOpInterface, ["getFlops"]>]> {
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs,
    OptionalAttr<I64Attr>:$estimated_cycles,
    OptionalAttr<I64Attr>:$op_id
  );

  let results = (outs AnyTensor:$result);  // i1 tensor

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
}

def Ascend_CmpEqOp : Ascend_VectorCmpOp<"cmp_eq"> {
  let summary = "Element-wise equal comparison on Vector Core";
}

def Ascend_CmpNeOp : Ascend_VectorCmpOp<"cmp_ne"> {
  let summary = "Element-wise not-equal comparison on Vector Core";
}

def Ascend_CmpLtOp : Ascend_VectorCmpOp<"cmp_lt"> {
  let summary = "Element-wise less-than comparison on Vector Core";
}

def Ascend_CmpLeOp : Ascend_VectorCmpOp<"cmp_le"> {
  let summary = "Element-wise less-or-equal comparison on Vector Core";
}

def Ascend_CmpGtOp : Ascend_VectorCmpOp<"cmp_gt"> {
  let summary = "Element-wise greater-than comparison on Vector Core";
}

def Ascend_CmpGeOp : Ascend_VectorCmpOp<"cmp_ge"> {
  let summary = "Element-wise greater-or-equal comparison on Vector Core";
}

//===----------------------------------------------------------------------===//
// Vector Core Unary Operations (向量一元运算)
//===----------------------------------------------------------------------===//

// Base class for simple vector unary ops (1 cycle)
class Ascend_VectorUnarySimple<string mnemonic>
    : AscendModel_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<EstimateCyclesOpInterface, ["getFlops"]>]> {
  let arguments = (ins
    AnyTensor:$input,
    OptionalAttr<I64Attr>:$estimated_cycles,
    OptionalAttr<I64Attr>:$op_id
  );

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

// Base class for complex vector unary ops (multiple cycles)
class Ascend_VectorUnaryComplex<string mnemonic, int cycles_per_op>
    : AscendModel_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<EstimateCyclesOpInterface, ["getCyclesPerVectorOp", "getFlops"]>]> {
  int cyclesPerOp = cycles_per_op;

  let arguments = (ins
    AnyTensor:$input,
    OptionalAttr<I64Attr>:$estimated_cycles,
    OptionalAttr<I64Attr>:$op_id
  );

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

// Simple unary ops (1 cycle)
def Ascend_NegOp : Ascend_VectorUnarySimple<"neg"> {
  let summary = "Element-wise negation on Vector Core";
}

def Ascend_AbsOp : Ascend_VectorUnarySimple<"abs"> {
  let summary = "Element-wise absolute value on Vector Core";
}

def Ascend_ReluOp : Ascend_VectorUnarySimple<"relu"> {
  let summary = "Element-wise ReLU activation on Vector Core";
}

def Ascend_CastOp : Ascend_VectorUnarySimple<"cast"> {
  let summary = "Element-wise type cast on Vector Core (includes extsi/extui/trunci/sitofp/uitofp/fptosi/fptoui/extf/truncf/bitcast)";
}

// Medium complexity unary ops (2 cycles)
def Ascend_SqrtOp : Ascend_VectorUnaryComplex<"sqrt", 2> {
  let summary = "Element-wise square root on Vector Core";
}

def Ascend_RsqrtOp : Ascend_VectorUnaryComplex<"rsqrt", 2> {
  let summary = "Element-wise reciprocal square root on Vector Core";
}

// Complex unary ops — differentiated cycle costs based on 910B Vector unit
// exp: 3 cycles (hardware-accelerated exponential)
// log: 4 cycles (iterative approximation)
// tanh: 6 cycles (computed via multiple exp operations internally)
// sigmoid: 5 cycles (computed as 1/(1+exp(-x)), reuses exp hardware)
def Ascend_ExpOp : Ascend_VectorUnaryComplex<"exp", 3> {
  let summary = "Element-wise exponential on Vector Core";
}

def Ascend_LogOp : Ascend_VectorUnaryComplex<"log", 4> {
  let summary = "Element-wise natural logarithm on Vector Core";
}

def Ascend_TanhOp : Ascend_VectorUnaryComplex<"tanh", 6> {
  let summary = "Element-wise hyperbolic tangent on Vector Core";
}

def Ascend_SigmoidOp : Ascend_VectorUnaryComplex<"sigmoid", 5> {
  let summary = "Element-wise sigmoid activation on Vector Core";
}

//===----------------------------------------------------------------------===//
// Vector Core Reduce Operations (向量归约运算)
//===----------------------------------------------------------------------===//

class Ascend_ReduceOp<string mnemonic>
    : AscendModel_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<EstimateCyclesOpInterface, ["getFlops"]>]> {
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$axis,
    OptionalAttr<I64Attr>:$estimated_cycles,
    OptionalAttr<I64Attr>:$op_id
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $input `axis` $axis attr-dict `:` type($input) `->` type($result)
  }];
}

def Ascend_ReduceSumOp : Ascend_ReduceOp<"reduce_sum"> {
  let summary = "Sum reduction on Vector Core";
}

def Ascend_ReduceMaxOp : Ascend_ReduceOp<"reduce_max"> {
  let summary = "Max reduction on Vector Core";
}

def Ascend_ReduceMinOp : Ascend_ReduceOp<"reduce_min"> {
  let summary = "Min reduction on Vector Core";
}

def Ascend_ReduceProdOp : Ascend_ReduceOp<"reduce_prod"> {
  let summary = "Product reduction on Vector Core";
}

//===----------------------------------------------------------------------===//
// Vector Core Other Operations
//===----------------------------------------------------------------------===//

def Ascend_BroadcastOp : AscendModel_Op<"broadcast", [Pure, DeclareOpInterfaceMethods<EstimateCyclesOpInterface>]> {
  let summary = "Broadcast tensor on Vector Core";
  let description = [{
    Broadcasts a tensor to a larger shape.
  }];

  let arguments = (ins
    AnyTensor:$input,
    DenseI64ArrayAttr:$shape,
    OptionalAttr<I64Attr>:$estimated_cycles,
    OptionalAttr<I64Attr>:$op_id
  );

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $input `to` $shape attr-dict `:` type($input) `->` type($result)
  }];
}

def Ascend_SelectOp : AscendModel_Op<"select", [Pure, DeclareOpInterfaceMethods<EstimateCyclesOpInterface, ["getFlops"]>]> {
  let summary = "Conditional select on Vector Core";
  let description = [{
    Selects elements from two tensors based on a condition tensor.
  }];

  let arguments = (ins
    AnyTensor:$condition,
    AnyTensor:$true_value,
    AnyTensor:$false_value,
    OptionalAttr<I64Attr>:$estimated_cycles,
    OptionalAttr<I64Attr>:$op_id
  );

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $condition `,` $true_value `,` $false_value attr-dict `:`
    type($condition) `,` type($true_value) `,` type($false_value) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Synchronization Operations
//===----------------------------------------------------------------------===//

def Ascend_SyncOp : AscendModel_Op<"sync", []> {
  let summary = "Synchronization barrier";
  let description = [{
    Synchronization point between different hardware units.
  }];

  let arguments = (ins
    StrAttr:$sync_type
  );

  let results = (outs);

  let assemblyFormat = [{
    $sync_type attr-dict
  }];
}

#endif // ASCEND_MODEL_OPS_TD
