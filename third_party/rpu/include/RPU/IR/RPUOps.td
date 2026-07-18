#ifndef RPU_EXEC_OPS
#define RPU_EXEC_OPS

include "RPU/IR/RPUDialect.td"
include "RPU/IR/RPUTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class RPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<RPU_Dialect, mnemonic, traits>;

def RPU_KernelOp : RPU_Op<"kernel", [Symbol, IsolatedFromAbove]> {
  let summary = "Internal executable RPU kernel";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$kind);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def RPU_LoadContigOp : RPU_Op<"load_contig"> {
  let summary = "Load a contiguous f16 tile";
  let arguments = (ins AnyType:$ptr, I32Attr:$n);
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "$ptr attr-dict `:` type($ptr) `->` type($result)";
  let hasVerifier = 1;
}

def RPU_AddOp : RPU_Op<"add", [SameOperandsAndResultType]> {
  let summary = "Elementwise add two RPU tiles";
  let arguments = (ins RPU_Tile:$lhs, RPU_Tile:$rhs);
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def RPU_MulOp : RPU_Op<"mul", [SameOperandsAndResultType]> {
  let summary = "Elementwise multiply two RPU tiles";
  let arguments = (ins RPU_Tile:$lhs, RPU_Tile:$rhs);
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def RPU_Elementwise16ValueMapOp : RPU_Op<"elementwise16_value_map"> {
  let summary = "Canonical unmasked 16-wide elementwise value-map chain";
  let arguments = (ins
    Variadic<RPU_Tile>:$inputs,
    DenseI64ArrayAttr:$opcodes,
    DenseI64ArrayAttr:$lhs_slots,
    DenseI64ArrayAttr:$rhs_slots
  );
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($result)";
  let hasVerifier = 1;
}

def RPU_CompactElementwise1DOp : RPU_Op<"compact_elementwise1d"> {
  let summary = "High-level compact 1D Add/Mul expression graph";
  let arguments = (ins
    Variadic<RPU_Tile>:$inputs,
    DenseI64ArrayAttr:$opcodes,
    DenseI64ArrayAttr:$lhs_slots,
    DenseI64ArrayAttr:$rhs_slots
  );
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($result)";
  let hasVerifier = 1;
}

def RPU_SoftmaxOp : RPU_Op<"softmax"> {
  let summary = "High-level softmax over a 1D f16 tile";
  let arguments = (ins RPU_Tile:$input);
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def RPU_MaxOp : RPU_Op<"max", [SameOperandsAndResultType]> {
  let summary = "Elementwise maximum of two RPU tiles";
  let arguments = (ins RPU_Tile:$lhs, RPU_Tile:$rhs);
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def RPU_StoreContigOp : RPU_Op<"store_contig"> {
  let summary = "Store a contiguous f16 tile";
  let arguments = (ins AnyType:$ptr, RPU_Tile:$value, I32Attr:$n);
  let assemblyFormat = "$ptr `,` $value attr-dict `:` type($ptr) `,` type($value)";
  let hasVerifier = 1;
}

def RPU_LoadMatrixOp : RPU_Op<"load_matrix"> {
  let summary = "Load a contiguous f16 matrix tile";
  let arguments = (ins AnyType:$ptr, I32Attr:$rows, I32Attr:$cols);
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "$ptr attr-dict `:` type($ptr) `->` type($result)";
  let hasVerifier = 1;
}

def RPU_ZeroOp : RPU_Op<"zero"> {
  let summary = "Create a zero-initialized f16 matrix tile";
  let arguments = (ins I32Attr:$rows, I32Attr:$cols);
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def RPU_MmaOp : RPU_Op<"mma"> {
  let summary = "Matrix multiply accumulate for f16 matrix tiles";
  let arguments = (ins RPU_Tile:$lhs, RPU_Tile:$rhs, RPU_Tile:$acc);
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs) `,` type($rhs) `,` type($acc) `->` type($result)";
  let hasVerifier = 1;
}

def RPU_DotOp : RPU_Op<"dot"> {
  let summary = "High-level zero-accumulated matrix dot";
  let arguments = (ins RPU_Tile:$lhs, RPU_Tile:$rhs);
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

def RPU_StoreMatrixOp : RPU_Op<"store_matrix"> {
  let summary = "Store a contiguous f16 matrix tile";
  let arguments = (ins AnyType:$ptr, RPU_Tile:$value, I32Attr:$rows, I32Attr:$cols);
  let assemblyFormat = "$ptr `,` $value attr-dict `:` type($ptr) `,` type($value)";
  let hasVerifier = 1;
}

def RPU_ReduceMaxAllOp : RPU_Op<"reduce_max_all"> {
  let summary = "Reduce a f16 tile to its maximum scalar";
  let arguments = (ins RPU_Tile:$input);
  let results = (outs F16:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def RPU_SubScalarOp : RPU_Op<"sub_scalar"> {
  let summary = "Subtract a f16 scalar from each f16 tile element";
  let arguments = (ins RPU_Tile:$lhs, F16:$rhs);
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

def RPU_ExpOp : RPU_Op<"exp"> {
  let summary = "Apply elementwise exponential to a f16 tile";
  let arguments = (ins RPU_Tile:$input);
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def RPU_SqrtOp : RPU_Op<"sqrt"> {
  let summary = "Apply elementwise square root to a f16 tile";
  let arguments = (ins RPU_Tile:$input);
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def RPU_ReluOp : RPU_Op<"relu"> {
  let summary = "Apply elementwise relu (max with zero) to a f16 tile";
  let arguments = (ins RPU_Tile:$input);
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def RPU_ReduceSumAxisOp : RPU_Op<"reduce_sum_axis"> {
  let summary = "Reduce a rank-2 f16 tile along a single axis to a rank-1 tile";
  let arguments = (ins RPU_Tile:$input, I64Attr:$axis);
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def RPU_BroadcastAddOp : RPU_Op<"broadcast_add"> {
  let summary = "Add a rank-1 f16 tile (broadcasted along columns) to a rank-2 f16 tile";
  let arguments = (ins RPU_Tile:$lhs, RPU_Tile:$rhs);
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

def RPU_FullOp : RPU_Op<"full"> {
  let summary = "Broadcast a f16 scalar across every lane of a tile";
  let arguments = (ins F16:$value, I32Attr:$n);
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
  let hasVerifier = 1;
}

def RPU_ReduceSumAllOp : RPU_Op<"reduce_sum_all"> {
  let summary = "Reduce a f16 tile to its sum scalar";
  let arguments = (ins RPU_Tile:$input);
  let results = (outs F16:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def RPU_ReciprocalOp : RPU_Op<"reciprocal"> {
  let summary = "Compute reciprocal of a f16 scalar";
  let arguments = (ins F16:$input);
  let results = (outs F16:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def RPU_MulScalarOp : RPU_Op<"mul_scalar"> {
  let summary = "Multiply each f16 tile element by a f16 scalar";
  let arguments = (ins RPU_Tile:$lhs, F16:$rhs);
  let results = (outs RPU_Tile:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

def RPU_ReturnOp : RPU_Op<"return", [Terminator]> {
  let summary = "Return from an executable RPU kernel";
  let assemblyFormat = "attr-dict";
}

#endif
