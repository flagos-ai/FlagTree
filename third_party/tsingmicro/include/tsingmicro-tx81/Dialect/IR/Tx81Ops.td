
//===---------------------- Tx81Ops.td ------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Definition of TsingMicro's Tx81 ML accelerator operations.
//
// Data format supported by Tx81 ML accelerator are:
//  f16,fp16,tf32,fp32
//
// For Tx81 accelerator unsupported data type, we can either convert it by
// using `TsmConvert`, or lower the operations to run on RISC-V controller
// instead.
//
// NOTE: CHANGING THE ARGUMENTS AND RETURNS OF ANY OPS RESULT IN THE CHANGE OF
// THEIR RUNTIME INTERFACE AND IMPLEMENTATION IN crt/Target/Tx81.
//
//===----------------------------------------------------------------------===//

#ifndef TSINGMICRO_TX81_OPS
#define TSINGMICRO_TX81_OPS

include "tsingmicro-tx81/Dialect/IR/Tx81AttrDefs.td"
include "tsingmicro-tx81/Dialect/IR/Tx81Types.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/IR/OpBase.td"

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;

class Tx81Op<string mnemonic, list<Trait> traits = []> :
  Op<Tx81Dialect, mnemonic,
     !listconcat(traits, [/*TensorSizeTrait, VerifyTensorLayoutsTrait*/])> {
}

def MemRefOrInt
  : AnyTypeOf<[AnyMemRef, AnySignlessIntegerOrIndex],
              "MemRef or Int as address type.", "::mlir::Type">;

// =============================================================================
// 4.8/4.9 DDR and SPM transfer ops
// =============================================================================

def RdmaOp : Tx81Op<"rdma", [
    AttrSizedOperandSegments
      ]> {

  let summary = "Copy data from global memory DDR(dram) to per thread local SPM(sram)";

  let description = [{
    Copy data from global memory DDR(dram) to per thread local SPM(sram).
  }];

  let arguments = (
    ins
    MemRefOrInt:$source,             // The source address in DDR
    MemRefOrInt:$target,             // The target address in SPM
    Variadic<Index>:$src_shape,      // src shape
    Variadic<Index>:$src_strides,    // src strides
    Variadic<Index>:$dst_shape,      // dst shape
    Variadic<Index>:$dst_strides,    // dst strides
    I32Attr:$rank,                   // rank
    I32Attr:$elem_bytes,             // elem bytes
    I32Attr:$fmt                     // elem fmt
  );

  let results = (outs I64:$dst); // The dest address in SPM
}

def WdmaOp : Tx81Op<"wdma", [
    AttrSizedOperandSegments
      ]> {
  let summary = "Copy data from per thread local SPM(sram) to global memory DDR(dram)";

  let description = [{
    Copy data from per thread local SPM(sram) to global memory DDR(dram).
  }];

  let arguments = (
    ins
    MemRefOrInt:$source,             // The source address in DDR
    MemRefOrInt:$target,             // The target address in SPM
    Variadic<Index>:$src_shape,      // src shape
    Variadic<Index>:$src_strides,    // src strides
    Variadic<Index>:$dst_shape,      // dst shape
    Variadic<Index>:$dst_strides,    // dst strides
    I32Attr:$rank,                   // rank
    I32Attr:$elem_bytes,             // elem bytes
    I32Attr:$fmt                     // elem fmt
  );

  let results = (outs I64:$dst); // The dest address in DDR
}

def MemCopyOp : Tx81Op<"memcpy", [
    MemoryEffects<[MemRead, MemWrite]>
]> {

  let summary = "Copy data from local SPM(sram) to local SPM(sram)";

  let description = [{
    Copy data from local SPM(sram) to local SPM(sram).
  }];

  let arguments = (
    ins
    MemRefOrInt:$input,             // The source address in DDR
    MemRefOrInt:$out,               // Out vector address
    MemRefOrInt:$elem_count,        // elem count
    I32Attr:$fmt                    // elem fmt
  );

  let results = (outs I64:$dst); // The dest address in SPM
}

// =============================================================================
// 4.4~6 TsmConv, TsmDepthwiseConv, TsmBackwardConv
// =============================================================================

def ConvOp : Tx81Op<"conv", [Pure]> {
  let summary = "Convolution engine intrinsic runtime API";

  let description = [{
    A common convolution op for TsmConv, TsmDepthwiseConv, TsmBackwardConv.
    This TsmConv is not a 1 to 1 map to TsingMicro's TsmConv intrinsic, it is
    the wrap of all APIs related to TsmConv. This op wraps the following APIs:
      TsmNewConv, TsmDeleteConv, AddInput, AddWeight, AddBias, AddOutput,
      SetOpType, SetNegativeAxisScale, SetPositiveAxisScale, SetSparse, SetPsum,
      SetPads, SetUnPads, SetKernelStrides, SetDilations, EnableRelu,
      EnableLeakyRelu, DisableRelu, DisableLeakyRelu, SetQuant.
  }];

  let arguments = (
    ins
    I64Attr:$op_type,         // 0: conv, 1: depthwise conv, 2: backward conv,
                              // 3: gemm
    MemRefOrInt:$src_activation,      // Input activation addr in SPM
    I32ArrayAttr:$src_dims,   // dims of src activation in NHWC format
    MemRefOrInt:$weight,              // Input weight addr in SPM
    I16Attr:$weight_dims,     // dims of weight(conv kernel) in Kx, Ky, Sx, Sy
                              // Where K and S is short for size(K) and step(S)
    BoolAttr:$en_bias,        // Enable bias add
    MemRefOrInt:$src_bias,           // The address of bias in SPM
    BoolAttr:$en_neg_scale,   // Enable negative axis scale
    MemRefOrInt:$src_neg_scale,      // The address of negative scale data in SPM
    BoolAttr:$en_pos_scale,   // Enable positive axis scale
    MemRefOrInt:$src_pos_scale,      // The address of positive scale data in SPM
    BoolAttr:$en_sparse,      // Enable sparse
    MemRefOrInt:$src_sparse,         // The sparse matrix addr in SPM
    BoolAttr:$en_psum,        // Enable psum? TODO: Production sum?
    MemRefOrInt:$src_psum,           // psum addr in SPM?
    I32ArrayAttr:$pads,      // Pad in top, bottom, left, right order
    I32ArrayAttr:$unpads,    // Unpad in top, bottom, left, right order
    I32ArrayAttr:$strides,   // Kernel strids in Kx, Ky, Sx, Sy
    I32ArrayAttr:$dilations, // dialation d0, d1 for conv/backwardconv
    BoolAttr:$en_leaky_relu,  // Enable LeakyRelu or normal Relu
    I32ArrayAttr:$out_dims,  // dims of output in NHWC format
    I64Attr:$src_fmt,         // Data format of src activation
    I64Attr:$weight_fmt,      // Data format of weight
    I64Attr:$out_fmt          // Data format of output
    // The param of SetQuant() is unused
  );

  // Output matrix C addr in SPM
  let results = (outs I64:$dst);
}

// =============================================================================
// 4.7. TsmGemm
// =============================================================================

def GemmOp : Tx81Op<"gemm", []> {
  let summary = "Gemm engine intrinsic runtime API";

  let description = [{
    This TsmGemm is not a 1 to 1 map to TsingMicro's TsmGemm intrinsic, it is
    the wrap of all APIs related to TsmGemm. This op wraps the following APIs:
      TsmNewGemm, TsmDeleteGemm, AddInput, ConfigMKN, AddOutput, SetPsum,
      SetTransflag, SetQuant, ConfigBatch, EnableRelu, EnableLeakyRelu,
      DisableRelu, DisableLeakyRelu, AddBias, SetNegativeAxisScale,
      SetPositiveAxisScale.
  }];

  let arguments = (
    ins
    MemRefOrInt:$src_a,          // Input matrix A addr in SPM
    MemRefOrInt:$src_b,          // Input matrix B addr in SPM
    MemRefOrInt:$src_bias,       // The address of bias in SPM
    // Output and initial zeroes buffer
    // FIXME: Whether need add side effect to source operands?
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$dst,
    I32ArrayAttr:$dims,   // The dimensions of M, K, N
    BoolAttr:$en_psum,        // Enable psum. Used as accumulate buffer
    MemRefOrInt:$psum_addr,   // The address of psum in SPM, Always same to output
    BoolAttr:$trans_src_a,  // Should matrix A be transposed
    BoolAttr:$trans_src_b,  // Should matrix B be transposed
    I32Attr:$batch_src_a,  // The batch of matrix A
    I32Attr:$batch_src_b,  // The batch of matrix B
    I32Attr:$relu_mode,    // Enable LeakyRelu or normal Relu or none
    BoolAttr:$en_bias,      // Enable bias add. Only support per channel(C dim), and int8 type
    BoolAttr:$en_neg_scale, // Enable negative axis scale
    MemRefOrInt:$src_neg_scale,    // The address of negative scale data in SPM
    BoolAttr:$en_pos_scale, // Enable positive axis scale
    MemRefOrInt:$src_pos_scale,    // The address of positive scale data in SPM
    I32Attr:$src_fmt,       // Input matrix data format
    I32Attr:$dst_fmt        // Output matrix data format
    // The param of SetQuant() is unused
  );

  // Output matrix C addr in SPM
  let results = (outs Variadic<I64>:$output);
}


// =============================================================================
// Tsm crt ChannelNorm/Dechannelnorm
// =============================================================================

def ChannelNormOp : Tx81Op<"channel_norm", []> {
  let summary = "Align channel dim.";

  let description = [{
    Align (N,H,W,C) to (N,cx,H,W,64) + (N,cx,H,W,c0),
    which align_base = 64, cx = C/align_base
    c0 = C%align_base
    c0_align = get_c0_align(c0)
  }];

  let arguments = (
    ins
    MemRefOrInt:$src,           // Input tensor address in SPM
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$dst,           // Output tensor address in SPM
    DenseI64ArrayAttr:$shape,  // The shape info of src
    // I16Attr:$cx,
    I16Attr:$c0_align,
    I16Attr:$dtype_size
  );

  // Output matrix C addr in SPM
  let results = (outs Variadic<I64>:$output);
}

def DechannelNormOp : Tx81Op<"dechannel_norm", []> {
  let summary = "Inverse operation of channelnorm.";

  let description = [{
    Trans (N,cx,H,W,64) + (N,cx,H,W,c0) back to (N,H,W,C).
  }];

  let arguments = (
    ins
    MemRefOrInt:$src,           // Input tensor address in SPM
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$dst,           // Output tensor address in SPM
    DenseI64ArrayAttr:$shape,  // The shape info of src
    // I16Attr:$cx,
    I16Attr:$c0_align,
    I16Attr:$dtype_size
  );

  // Output matrix C addr in SPM
  let results = (outs Variadic<I64>:$output);
}

// =============================================================================
// 4.10. TsmArith
// =============================================================================

class UnaryOp<string mnemonic, list<Trait> traits = []> :
    Tx81Op<mnemonic, traits # [Elementwise]> {
  let arguments = (ins
    MemRefOrInt:$input,              // Input vector address
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$out,  // Out vector address
    MemRefOrInt:$elem_count,      // Number of input elements
    I16Attr:$fmt              // The data format of src & dst
  );
  let results = (outs Variadic<I64>:$dst);
}

def AbsVVOp : UnaryOp<"absvv"> {
  let summary = "Absolute value of input vector";
}
def SqrtVVOp : UnaryOp<"sqrtvv", [Pure, Elementwise]> {}
def RsqrtVVOp : UnaryOp<"rsqrtvv", [Pure, Elementwise]> {}
def NegVVOp : UnaryOp<"negvv", [Pure, Elementwise]> {}
def RecipVVOp : UnaryOp<"recipvv", [Pure, Elementwise]> {}
def SquareVVOp : UnaryOp<"squarevv", [Pure, Elementwise]> {}

class BinaryVVOp<string mnemonic, list<Trait> traits = []> :
    Tx81Op<mnemonic, traits # [Elementwise]> {
  let arguments = (ins
    MemRefOrInt:$input0,              // First input vector address
    MemRefOrInt:$input1,              // Second vector address
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$out,  // Out vector address
    MemRefOrInt:$elem_count,      // Number of input elements
    I16Attr:$rnd_mode,        // round mode
    I16Attr:$fmt              // The data format of src & dst
  );
  let results = (outs Variadic<I64>:$dst);
}

def AddVVOp : BinaryVVOp<"addvv"> {
  let summary = "Add two vectors element-wise";
}
def SubVVOp : BinaryVVOp<"subvv">;
def MulVVOp : BinaryVVOp<"mulvv">;
def DivVVOp : BinaryVVOp<"divvv">;
def MaxVVOp : BinaryVVOp<"maxvv">;
def MinVVOp : BinaryVVOp<"minvv">;

class BinaryVSOp<string mnemonic, list<Trait> traits = []> :
    Tx81Op<mnemonic, traits # [Elementwise]> {
  let arguments = (ins
    MemRefOrInt:$input0,              // First input vector address
    I32:$value,                       // Const value
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$out,  // Out vector address
    MemRefOrInt:$elem_count,  // Number of input elements
    I16Attr:$rnd_mode,        // round mode
    I16Attr:$fmt              // The data format of src & dst
  );
  let results = (outs Variadic<I64>:$dst);
}

def AddVSOp : BinaryVSOp<"addvs"> {
  let summary = "Add input vector and constant value";
}
def SubVSOp : BinaryVSOp<"subvs">;
def MulVSOp : BinaryVSOp<"mulvs">;
def DivVSOp : BinaryVSOp<"divvs">;

// ...

// =============================================================================
// 4.11. TsmRelation
// =============================================================================

class RelationVVOp<string mnemonic, list<Trait> traits = []> :
    Tx81Op<mnemonic, traits # [Elementwise]> {
  let arguments = (ins
    MemRefOrInt:$input0,              // First input vector address
    MemRefOrInt:$input1,              // Second vector address
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$out,  // Out vector address
    MemRefOrInt:$elem_count,      // Number of input elements
    I16Attr:$fmt              // The data format of src & dst
  );
  let results = (outs Variadic<I64>:$dst);
}

def BoolEqualVV : RelationVVOp<"boolequalvv"> {
  let summary = "compare two input value, if equal, return true";
}

def BoolUnEqualVV : RelationVVOp<"boolunequalvv"> {
  let summary = "compare two input value, if unequal, return true";
}

def BoolGreaterEqualVV : RelationVVOp<"boolgreatrequalvv"> {
  let summary = "compare two input value, if src0 >= src1, return true";
}

def BoolGreaterVV : RelationVVOp<"boolgreatervv"> {
  let summary = "compare two input value, if src0 > src1, return true";
}

def BoolLessEqualVV : RelationVVOp<"boollessequalvv"> {
  let summary = "compare two input value, if src0 <= src1, return true";
}

def BoolLessThenVV : RelationVVOp<"boollessthenvv"> {
  let summary = "compare two input value, if src0 < src1, return true";
}

def EqualVV : RelationVVOp<"equalvv"> {
  let summary = "compare two input value, if equal, return 1.0";
}

def UnEqualVV : RelationVVOp<"unequalvv"> {
  let summary = "compare two input value, if unequal, return 1.0";
}

def GreaterEqualVV : RelationVVOp<"greatrequalvv"> {
  let summary = "compare two input value, if src0 >= src1, return 1.0";
}

def GreaterVV : RelationVVOp<"greatervv"> {
  let summary = "compare two input value, if src0 > src1, return 1.0";
}

def LessEqualVV : RelationVVOp<"lessequalvv"> {
  let summary = "compare two input value, if src0 <= src1, return 1.0";
}

def LessThenVV : RelationVVOp<"lessthenvv"> {
  let summary = "compare two input value, if src0 < src1, return 1.0";
}

class RelationVSOp<string mnemonic, list<Trait> traits = []> :
    Tx81Op<mnemonic, traits # [Elementwise]> {
  let arguments = (ins
    MemRefOrInt:$input0,              // First input vector address
    I32:$value,                       // Const value
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$out,  // Out vector address
    MemRefOrInt:$elem_count,      // Number of input elements
    I16Attr:$fmt              // The data format of src & dst
  );
  let results = (outs Variadic<I64>:$dst);
}

def BoolEqualVS : RelationVSOp<"boolequalvs"> {
  let summary = "compare input value with ConstantOp, if equal, return true";
}

def BoolUnEqualVS : RelationVSOp<"boolunequalvs"> {
  let summary = "compare input value with ConstantOp, if unequal, return true";
}

def BoolGreaterEqualVS : RelationVSOp<"boolgreatrequalvs"> {
  let summary = "compare input value with ConstantOp, if src0 >= src1, return true";
}

def BoolGreaterVS : RelationVSOp<"boolgreatervs"> {
  let summary = "compare input value with ConstantOp, if src0 > src1, return true";
}

def BoolLessEqualVS : RelationVSOp<"boollessequalvs"> {
  let summary = "compare input value with ConstantOp, if src0 <= src1, return true";
}

def BoolLessThenVS : RelationVSOp<"boollessthenvs"> {
  let summary = "compare input value with ConstantOp, if src0 < src1, return true";
}

def EqualVS : RelationVSOp<"equalvs"> {
  let summary = "compare input value with ConstantOp, if equal, return 1.0";
}

def UnEqualVS : RelationVSOp<"unequalvs"> {
  let summary = "compare input value with ConstantOp, if unequal, return 1.0";
}

def GreaterEqualVS : RelationVSOp<"greatrequalvs"> {
  let summary = "compare input value with ConstantOp, if src0 >= src1, return 1.0";
}

def GreaterVS : RelationVSOp<"greatervs"> {
  let summary = "compare input value with ConstantOp, if src0 > src1, return 1.0";
}

def LessEqualVS : RelationVSOp<"lessequalvs"> {
  let summary = "compare input value with ConstantOp, if src0 <= src1, return 1.0";
}

def LessThenVS : RelationVSOp<"lessthenvs"> {
  let summary = "compare input value with ConstantOp, if src0 < src1, return 1.0";
}


// ...
// =============================================================================
// 4.12. TsmLogic
// =============================================================================

class BinaryLogicVVOp<string mnemonic, list<Trait> traits = []> :
    Tx81Op<mnemonic, traits # [Elementwise]> {
  let arguments = (ins
    MemRefOrInt:$input0,              // First input vector address
    MemRefOrInt:$input1,              // Second vector address
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$out,  // Out vector address
    MemRefOrInt:$elem_count,      // Number of input elements
    I16Attr:$fmt              // The data format of src & dst
  );
  let results = (outs Variadic<I64>:$dst);
}

def AndVV : BinaryLogicVVOp<"andvv"> {
  let summary = "And operation on elements at the same position. If the element is not 0, it is represented as 1.";
}

def OrVV : BinaryLogicVVOp<"orvv"> {
  let summary = "OR operation on elements at the same position. If the element is not 0, it is represented as 1.";
}

def XorVV : BinaryLogicVVOp<"xorvv"> {
  let summary = "XOR operation on elements at the same position. If the element is not 0, it is represented as 1.";
}

class BoolUnaryLogicVVOp<string mnemonic, list<Trait> traits = []> :
    Tx81Op<mnemonic, traits # [Elementwise]> {
  let arguments = (ins
    MemRefOrInt:$input,              // input vector address
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$out,  // Out vector address
    MemRefOrInt:$elem_count      // Number of input elements
  );
  let results = (outs Variadic<I64>:$dst);
}

def BoolNotV : BoolUnaryLogicVVOp<"boolnotvv"> {
  let summary = "Not operation on elements at each bit position. src_elem_num must be an integer multiple of 8";
}

class BoolBinaryLogicVVOp<string mnemonic, list<Trait> traits = []> :
    Tx81Op<mnemonic, traits # [Elementwise]> {
  let arguments = (ins
    MemRefOrInt:$input0,              // First input vector address
    MemRefOrInt:$input1,              // Second vector address
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$out,  // Out vector address
    MemRefOrInt:$elem_count      // Number of input elements
  );
  let results = (outs Variadic<I64>:$dst);
}

def BoolAndV : BoolBinaryLogicVVOp<"boolandvv"> {
  let summary = "And operation on elements at each bit position. src_elem_num must be an integer multiple of 8";
}

def BoolXorV : BoolBinaryLogicVVOp<"boolxorvv"> {
  let summary = "Xor operation on elements at each bit position. src_elem_num must be an integer multiple of 8";
}

def BoolOrV : BoolBinaryLogicVVOp<"boolorvv"> {
  let summary = "Or operation on elements at each bit position. src_elem_num must be an integer multiple of 8";
}


// =============================================================================
// 4.13. TsmTranscendental
// =============================================================================

def Log2Op : UnaryOp<"log2", []> {
  let summary = "Logarithm based 2";
}
def LnOp : UnaryOp<"ln", []> {
  let summary = "Logarithm based e";
}
def Pow2Op : UnaryOp<"pow2", []> {
  let summary = "2 ** x";
}
def ExpOp : UnaryOp<"exp", []> {
  let summary = "Exponential with high precision";
}
def ExplpOp : UnaryOp<"explp", []> {
  let summary = "Exponential with low precision";
}
def SinOp : UnaryOp<"sin", []> {
  let summary = "Sine";
}
def CosOp : UnaryOp<"cos", []> {
  let summary = "Cosine";
}

// =============================================================================
// 4.13. TsmActivation
// =============================================================================

class ActivationOp<string mnemonic, list<Trait> traits> :
    Tx81Op<mnemonic, traits> {
  let arguments = (ins
    MemRefOrInt:$input,              // Input vector address
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$out,  // Out vector address
    AnySignlessIntegerOrIndex:$elem_count,   // Number of input elements
    I16Attr:$fmt           // The data format of src & dst
  );

  let results = (outs Variadic<I64>:$dst);
}

def Tanh : ActivationOp<"tanh", []> {
  let summary = "Hyperbolic tangent";
}
def Sigmoid : ActivationOp<"sigmoid", []> {
  let summary = "Logistic sigmoid";
}
def Relu : ActivationOp<"relu", []> {
  let summary = "Rectified linear unit";
}
def Satrelu : ActivationOp<"satrelu", []> {
  let summary = "Saturated ReLU";
}
def Leakyrelu : ActivationOp<"leakyrelu", []> {
  let summary = "Leaky rectified linear unit";
}
def Softplus : ActivationOp<"softplus", []> {
  let summary = "Smooth approximation of ReLU";
}

// =============================================================================
// 4.15. TsmReduce
// =============================================================================

class Reduce<string mnemonic> : Tx81Op<mnemonic, [Pure]> {
  let summary = "Reduction engine intrinsic runtime API";

  let description = [{
    Includes ReduceSum, ReduceAvg, ReduceMin and ReduceMax interfaces.
    Mapping between `dim` and NCHW:
      Reduction on C: dim=0
      Reduction on W: dim=1
      Reduction on H: dim=2
      Reduction on HW: dim=4
  }];

  let arguments = (
    ins
    MemRefOrInt:$src,           // Input tensor address in SPM
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$dst,           // Output tensor address in SPM
    UI32Attr:$dim,      // Which dimension to be reduced
    I64ArrayAttr:$shape,  // The shape info of src
    I16Attr:$fmt        // The data format of src & dst
  );

  // Output tensor address in SPM
  let results = (outs Variadic<MemRefOrInt>);
}

def ReduceSumOp : Reduce<"reduce_sum">;
def ReduceAvgOp : Reduce<"reduce_avg">;
def ReduceMaxOp : Reduce<"reduce_max">;
def ReduceMinOp : Reduce<"reduce_min">;

// =============================================================================
// 4.15. TsmMaskDataMove
// =============================================================================

def MaskMoveOp : Tx81Op<"mask_move", []> {
  let summary = "Mask data move engine intrinsic runtime API";

  let description = [{ When mask is 1, extract the data from src and write it to dst.
When mask=0, the corresponding elements of dst remain unchanged.
  }];

  let arguments = (
    ins
    MemRefOrInt:$source,            // The source address in SPM
    // The target address in SPM
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$target,
    AnySignlessIntegerOrIndex:$elem_count,    // Number of elements to be copied
    MemRefOrInt:$mask,
    I32Attr:$fmt
  );

  // The dst address is not used, use target in arguments instead.
  let results = (outs Variadic<I64>:$dst);
}

// =============================================================================
// 4.19. TsmConvert instructions
// =============================================================================

class ZeroPointConvertOp<string mnemonic, list<Trait> traits> :
    Tx81Op<mnemonic, traits> {
  let arguments = (ins
    MemRefOrInt:$src,
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$dst,
    UI32Attr:$zero_point,
    UI32Attr:$elem_count
  );
}

def INT8ToFP16Op : ZeroPointConvertOp<"int8_fp16", []> {
  let summary = "Data format from int8 to fp16";
}
def INT8ToBF16Op : ZeroPointConvertOp<"int8_bf16", []> {
  let summary = "Data format from int8 to bf16";
}
def INT8ToFP32Op : ZeroPointConvertOp<"int8_fp32", []> {
  let summary = "Data format from int8 to fp32";
}
def INT8ToTF32Op : ZeroPointConvertOp<"int8_tf32", []> {
  let summary = "Data format from int8 to tf32";
}

class RoundConvertOp<string mnemonic, list<Trait> traits> :
    Tx81Op<mnemonic, traits> {
  let arguments = (ins
    MemRefOrInt:$input,
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$output,
    AnySignlessIntegerOrIndex:$elem_count,
    I16Attr:$rnd_mode
  );
  let results = (outs I64:$dst);
}

def INT16ToBF16Op : RoundConvertOp<"int16_bf16", []> {
  let summary = "Data format from int16 to bf16";
}
def INT16ToFP32Op : RoundConvertOp<"int16_fp32", []> {
  let summary = "Data format from int16 to fp32";
}
def INT16ToTF32Op : RoundConvertOp<"int16_tf32", []> {
  let summary = "Data format from int16 to tf32";
}
def INT32ToFP16Op : RoundConvertOp<"int32_fp16", []> {
  let summary = "Data format from int32 to fp16";
}
def INT32ToBF16Op : RoundConvertOp<"int32_bf16", []> {
  let summary = "Data format from int32 to bf16";
}
def INT32ToFP32Op : RoundConvertOp<"int32_fp32", []> {
  let summary = "Data format from int32 to fp32";
}
def INT32ToTF32Op : RoundConvertOp<"int32_tf32", []> {
  let summary = "Data format from int32 to tf32";
}
def BF16ToINT16Op : RoundConvertOp<"bf16_int16", []> {
  let summary = "Data format from bf16 to int16";
}
def BF16ToINT32Op : RoundConvertOp<"bf16_int32", []> {
  let summary = "Data format from bf16 to int32";
}
def FP16ToINT8Op : RoundConvertOp<"fp16_int8", []> {
  let summary = "Data format from fp16 to int8";
}
def FP16ToINT16Op : RoundConvertOp<"fp16_int16", []> {
  let summary = "Data format from fp16 to int16";
}
def FP16ToINT32Op : RoundConvertOp<"fp16_int32", []> {
  let summary = "Data format from fp16 to int32";
}
def FP16ToBF16Op : RoundConvertOp<"fp16_bf16", []> {
  let summary = "Data format from fp16 to bf16";
}
def FP32ToINT8Op : RoundConvertOp<"fp32_int8", []> {
  let summary = "Data format from fp32 to int8";
}
def FP32ToINT16Op : RoundConvertOp<"fp32_int16", []> {
  let summary = "Data format from fp32 to int16";
}
def FP32ToINT32Op : RoundConvertOp<"fp32_int32", []> {
  let summary = "Data format from fp32 to int32";
}
def FP32ToFP16Op : RoundConvertOp<"fp32_fp16", []> {
  let summary = "Data format from fp32 to fp16";
}
def FP32ToBF16Op : RoundConvertOp<"fp32_bf16", []> {
  let summary = "Data format from fp32 to bf16";
}
def FP32ToTF32Op : RoundConvertOp<"fp32_tf32", []> {
  let summary = "Data format from fp32 to tf32";
}
def TF32ToINT8Op : RoundConvertOp<"tf32_int8", []> {
  let summary = "Data format from tf32 to int8";
}
def TF32ToINT16Op : RoundConvertOp<"tf32_int16", []> {
  let summary = "Data format from tf32 to int16";
}
def TF32ToINT32Op : RoundConvertOp<"tf32_int32", []> {
  let summary = "Data format from tf32 to int32";
}
def TF32ToBF16Op : RoundConvertOp<"tf32_bf16", []> {
  let summary = "Data format from tf32 to bf16";
}

class NormalConvertOp<string mnemonic, list<Trait> traits> :
    Tx81Op<mnemonic, traits> {
  let arguments = (ins
    MemRefOrInt:$input,
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$output,
    AnySignlessIntegerOrIndex:$elem_count
  );
  let results = (outs I64:$dst);
}

def INT16ToFP16Op : NormalConvertOp<"int16_fp16", []> {
  let summary = "Data format from int16 to fp16";
}
def BF16ToINT8Op : NormalConvertOp<"bf16_int8", []> {
  let summary = "Data format from bf16 to int8";
}
def BF16ToFP16Op : NormalConvertOp<"bf16_fp16", []> {
  let summary = "Data format from bf16 to fp16";
}
def BF16ToFP32Op : NormalConvertOp<"bf16_fp32", []> {
  let summary = "Data format from bf16 to fp32";
}
def BF16ToTF32Op : NormalConvertOp<"bf16_tf32", []> {
  let summary = "Data format from bf16 to tf32";
}
def FP16ToFP32Op : NormalConvertOp<"fp16_fp32", []> {
  let summary = "Data format from fp16 to fp32";
}
def FP16ToTF32Op : NormalConvertOp<"fp16_tf32", []> {
  let summary = "Data format from fp16 to tf32";
}
def TF32ToFP16Op : NormalConvertOp<"tf32_fp16", []> {
  let summary = "Data format from tf32 to fp16";
}
def TF32ToFP32Op : NormalConvertOp<"tf32_fp32", []> {
  let summary = "Data format from tf32 to fp16";
}

// Micro scaling format conversion operations
def FP8E4M3ToBF16Op : NormalConvertOp<"fp8E4M3_bf16", []> {
  let summary = "Convert data format from FP8 (4 exponent bits, 3 mantissa bits) to BF16";
}
def FP8E5M2ToBF16Op : NormalConvertOp<"fp8E5M2_bf16", []> {
  let summary = "Convert data format from FP8 (5 exponent bits, 2 mantissa bits) to BF16";
}
def FP4E2M1ToBF16Op : NormalConvertOp<"fp4E2M1_bf16", []> {
  let summary = "Convert data format from FP4 (2 exponent bits, 1 mantissa bit) to BF16";
}

def MXFPScaleBF16Op : Tx81Op<"MXFP_scale_bf16", []> {
  let summary = "Scaling up bf16 tensor with e8m0 scale.";

  let description = [{
    Performs conversion of MXFP numbers to BF16/FP16 format according to the
    Open Compute Project (OCP) microscaling formats specification v1.0.
    See: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
  }];

  let arguments = (
    ins
    MemRefOrInt:$src,          // Source tensor address in SPM
    MemRefOrInt:$scale,        // Scale factor tensor address in SPM
    MemRefOrInt:$dst,          // Destination tensor address in SPM
    I32Attr:$elemCount         // Number of elements
  );

  let results = (outs Variadic<I64>:$output);  // Output tensor address in SPM
}

// =============================================================================
// 4.20. TsmPeripheral instructions
// =============================================================================

def CountOp : Tx81Op<"count", [Pure]> {
  let summary = "Count the non-zero elements from given tensor";

  let arguments = (
    ins
    MemRefOrInt:$src,               // Input tensor address in SPM
    I32Attr:$elem_count,     // TODO: Ask TsingMicro for explain.
    //I64Attr:$p_wb_data0,   // TODO: Ask TsingMicro for explain.
    //I64Attr:$p_wb_data1,   // TODO: Ask TsingMicro for explain.
    I16Attr:$fmt
  );

  // The output tensor address in SPM
  let results = (outs MemRefOrInt:$dst);
}

def MemsetOp : Tx81Op<"memset", [
  AttrSizedOperandSegments
]> {
  let summary = "Write given `value` to range of address on SPM(sram)";

  let arguments = (
    ins
    MemRefOrInt:$target,            // SPM address to be memset
    I32:$value,                  // Value to be written
    Variadic<Index>:$dst_shape,      // src shape
    Variadic<Index>:$dst_strides,    // src strides
    I32Attr:$rank,                   // rank
    I16Attr:$fmt
  );

  // The address updated by memset in SPM
  let results = (outs MemRefOrInt:$dst);
}

def Bit2FpOp : Tx81Op<"bit2fp", []> {
  let summary = "Convert a vector of the bitwise into the fp vector";

  let arguments = (ins
    MemRefOrInt:$src,             // Input tensor
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$target,
    AnySignlessIntegerOrIndex:$elem_count,   // Number of input elements
    I16Attr:$fmt           // The data format of src & dst
  );
  let results = (outs I64:$dst);
}

def ArgMaxOp : Tx81Op<"argmax", []> {
  let summary = "Return a max value inner a vector and its corresponding index";

  let arguments = (ins
    MemRefOrInt:$src,              // First input vector address
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$value,  // Address
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$index,  // Address
    I32Attr:$elem_count,   // Number of input elements
    I16Attr:$fmt           // The data format of src & dst
  );

  let results = (outs Variadic<I64>:$dst);
}

def ArgMinOp : Tx81Op<"argmin", []> {
  let summary = "Return a min value inner a vector and its corresponding index";

  let arguments = (ins
    MemRefOrInt:$src,              // First input vector address
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$value,  // Address
    Arg<MemRefOrInt, "the target memref", [MemWrite]>:$index,  // Address
    I32Attr:$elem_count,   // Number of input elements
    I16Attr:$fmt           // The data format of src & dst
  );

  let results = (outs Variadic<I64>:$dst);
}

def BilinearOp : Tx81Op<"bilinear", []> {
  let summary = "Bilinear interpolation";

  let arguments = (ins
    UI64:$src,                 // Input tensor with the NHWC format
    I32ArrayAttr:$src_shape,   // Input tensor shape
    I32ArrayAttr:$dst_shape,   // Output tensor shape
    F32:$scale_w,              // Input tensor "w" divided by output tensor "w"
    F32:$scale_h,              // Input tensor "h" divided by output tensor "h"
    I16Attr:$fmt               // The data format of src & dst
  );
  let results = (outs UI64:$dst);
}

def Lut16Op : Tx81Op<"lut16", []> {
  let summary = "16-bit lookup table";

  let arguments = (ins
    MemRefOrInt:$src,            // Vector offset with respect to LUT
    UI64:$lut16,
    I32Attr:$src_elem_count,   // Number of elements in vector offset
    I32Attr:$lut_elem_count    // Number of elements in LUT
  );

  let results = (outs UI64:$dst);
}

def Lut32Op : Tx81Op<"lut32", []> {
  let summary = "32-bit lookup table";

  let arguments = (ins
    MemRefOrInt:$src,            // Vector offset with respect to LUT
    UI64:$lut32,
    I32Attr:$src_elem_count,   // Number of elements in vector offset
    I32Attr:$lut_elem_count    // Number of elements in LUT
  );

  let results = (outs UI64:$dst);
}

def RandGenOp : Tx81Op<"randgen", []> {
  let summary = "Generate random numbers using two 64-bit seeds";

  let arguments = (ins
    UI64:$src0,          // The first random seed
    UI64:$src1,          // The second random seed
    UI64:$dst0,          // Store the first random seed
    UI64:$dst1,          // Store the second random seed
    UI64:$dst2,          // Random value
    I32Attr:$elem_num,   // Number of random values
    I16Attr:$fmt         // The date format of random value
  );
}


//
// 4.21. TsmDataMove
//

class TransformOp<string mnemonic, list<Trait> traits = []> :
    Tx81Op<mnemonic, traits> {
  let arguments = (ins
    MemRefOrInt:$source,       // Input tensor
    MemRefOrInt:$target,       // Output tensor
    DenseI32ArrayAttr:$src_shape,   // Input shape
    DenseI32ArrayAttr:$dst_shape,   // Output shape
    I16Attr:$fmt               // The data format of src & dst
  );
  let results = (outs I64:$dst);
}

def Mirror : TransformOp<"mirror", []> {
  let summary = "Horizontal mirror to a matrix";
}
def Transpose : TransformOp<"transpose", []> {
  let summary = "Transpose a matrix";
}
def Rotate90 : TransformOp<"rotate90", []> {
  let summary = "Rotate a matrix 90 degree clockwise";
}
def Rotate180 : TransformOp<"rotate180", []> {
  let summary = "Rotate a matrix 180 degree clockwise";
}
def Rotate270 : TransformOp<"rotate270", []> {
  let summary = "Rotate a matrix 270 degree clockwise";
}
def Nchw2nhwc : TransformOp<"nchw2nhwc", []> {
  let summary = "Tranform a tensor from nchw to nhwc";
}
def Nhwc2nchw : TransformOp<"nhwc2nchw", []> {
  let summary = "Tranform a tensor from nhwc to nchw";
}
def TensorNorm : TransformOp<"tensornorm", []> {
  let summary = "Make continuous tensor align std format in ch direction";
}

def Concat : Tx81Op<"concat", []> {
  let summary = "Concatenation based on the dim";

  let arguments = (ins
    UI64:$src1,                 // The first input tensor
    I32ArrayAttr:$src1_shape,   // The first input tensor shape
    UI64:$src2,                 // The second input tensor
    I32ArrayAttr:$src2_shape,   // The second input tensor shape
    I32ArrayAttr:$dst_shape,    // Ouput tensor shape
    I16Attr:$dim,               // Represent the concat direction, such as:
                                //   0 is channel, 1 is width, and 2 is height
    I16Attr:$fmt                // The data format of input & output tensor
  );
  let results = (outs UI64:$dst);
}

def Pad : Tx81Op<"pad", []> {
  let summary = "Tensor padding";

  let arguments = (ins
    UI64:$src,                 // Input tensor
    I32ArrayAttr:$src_shape,   // Input tensor shape
    I32ArrayAttr:$dst_shape,   // Output tensor shape
    I16Attr:$pad,              // Padding mode: top, bottom, left, and right
    I16Attr:$fmt               // The data format of src & dst
  );
  let results = (outs UI64:$dst);
}

def Img2col : Tx81Op<"img2col", []> {
  let summary = "Transform a feature-map tensor into a matrix";

  let arguments = (ins
    UI64:$src,                 // Input tensor
    I32ArrayAttr:$src_shape,   // Input tensor shape
    I32ArrayAttr:$dst_shape,   // Output tensor shape
    I32Attr:$src_elem_num,     // Number of elements in input tensor
    I32Attr:$dst_elem_num,     // Number of elements in output tensor
    I32ArrayAttr:$swr,         // Horizontal stride of convolution
    I32ArrayAttr:$pdr,         // Vertical stride of convolution
    I16Attr:$fmt               // The data format of src & dst
  );
  let results = (outs UI64:$dst);
}

def GatherScatter : Tx81Op<"gatherscatter", []> {
  let summary = "Transfer data in strides and iterations";

  let arguments = (ins
    MemRefOrInt:$source,                 // The source
    MemRefOrInt:$target,                 // The target
    I32Attr:$bytes,                  // Inner loop data size in bytes
    I32Attr:$src_strideN,
    I32Attr:$src_strideH,
    I32Attr:$src_strideW,
    I32Attr:$src_iterN,
    I32Attr:$src_iterH,
    I32Attr:$src_iterW,
    I32Attr:$dst_strideN,
    I32Attr:$dst_strideH,
    I32Attr:$dst_strideW,
    I32Attr:$dst_iterN,
    I32Attr:$dst_iterH,
    I32Attr:$dst_iterW
  );
  let results = (outs I64:$dst);
}

def BarrierOp : Tx81Op<"barrier"> {
  let summary = "Synchronizes all work items";
  let description = [{
    The "barrier" op synchronizes all work items.
  }];
  let assemblyFormat = "attr-dict";
}

#endif // TSINGMICRO_TX81_OPS
