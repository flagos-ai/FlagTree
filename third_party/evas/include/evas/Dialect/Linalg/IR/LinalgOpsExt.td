//===- LinalgOpsExt.td - EVAS Linalg dialect ops ----------*- tablegen -*-===//

#ifndef EVAS_LINALGEXT_OPS
#define EVAS_LINALGEXT_OPS

include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "evas/Dialect/Linalg/IR/LinalgEnumsExt.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"

class Evas_LinalgOp<string mnemonic, list<Trait> traits = []> :
    Op<Linalg_Dialect, mnemonic, traits>;

def Linalg_CastOp : Evas_LinalgOp<"cast",
  [DestinationStyleOpInterface,
   DeclareOpInterfaceMethods<BufferizableOpInterface,
     ["bufferizesToMemoryRead",
      "bufferizesToMemoryWrite",
      "getAliasingValues",
      "bufferize"]>,
   DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
   DeclareOpInterfaceMethods<AggregatedOpInterface>,
   DeclareOpInterfaceMethods<TilingInterface,
     ["getIterationDomain",
      "getLoopIteratorTypes",
      "getResultTilePosition",
      "getTiledImplementation",
      "generateResultTileValue"]>]> {
  let summary = "Cast operator";
  let description = [{
    Casts elements from the input shaped value into the output shaped value.
  }];

  let arguments = (ins AnyShaped:$input,
                       AnyShaped:$output,
                       DefaultValuedAttr<EVAS_LINALG_EvaMindCastRndModeAttr,
                                         "EvaMindCastRndMode::RMM">:$rounding_mode);

  let results = (outs Variadic<AnyRankedTensor>:$result);

  let assemblyFormat = [{
      `ins` `(` $input `:` type($input) `)`
      `outs` `(` $output `:` type($output) `)`
      `{` `rounding_mode` `:` $rounding_mode `}`
      attr-dict
      (`->` type($result)^)?
  }];

  let hasVerifier = 1;
  let hasFolder = 1;

  let extraClassDeclaration = [{
    ::mlir::Operation::operand_range getInputs() { return getODSOperands(0); }
    ::mlir::Operation::operand_range getOutputs() { return getODSOperands(1); }

    static void build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState, ValueRange inputs,
                      ValueRange outputs,
                      ArrayRef<NamedAttribute> attributes = {}) {
      assert(inputs.size() + outputs.size() == 2u &&
             "mismatched number of parameters");
      odsState.addOperands(inputs[0]);
      odsState.addOperands(outputs[0]);
      odsState.addAttributes(attributes);
    }

    static void build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState,
                      TypeRange resultTensorTypes, ValueRange inputs,
                      ValueRange outputs,
                      ArrayRef<NamedAttribute> attributes = {}) {
      assert(inputs.size() + outputs.size() == 2u &&
             "mismatched number of parameters");
      odsState.addOperands(inputs);
      odsState.addOperands(outputs);
      odsState.addAttributes(attributes);
      odsState.addTypes(resultTensorTypes);
    }

    ShapedType getInputOperandType() {
      return cast<ShapedType>(getInput().getType());
    }
    ShapedType getOutputOperandType() {
      return cast<ShapedType>(getOutput().getType());
    }
    int64_t getInputOperandRank() { return getInputOperandType().getRank(); }
    int64_t getOutputOperandRank() { return getOutputOperandType().getRank(); }
    MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
  }];
}

#endif // EVAS_LINALGEXT_OPS
