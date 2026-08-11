//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TILING_EXT_BASE
#define TRITON_TILING_EXT_BASE

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"

include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtInterfaces.td"


//===----------------------------------------------------------------------===//
// TritonTilingExt dialect definition
//===----------------------------------------------------------------------===//

def TritonTilingExt_Dialect : Dialect {
  let name = "ttx";
  let cppNamespace = "::mlir::ttx";
}

//===----------------------------------------------------------------------===//
// TritonTilingExt op definitions
//===----------------------------------------------------------------------===//

// Base class for TritonTilingExt dialect ops.
class TritonTilingExt_Op<string mnemonic, list<Trait> traits = []>
    : Op<TritonTilingExt_Dialect, mnemonic, traits> {
}

class TritonTilingExt_TilingOp<string mnemonic> : Op<TritonTilingExt_Dialect, mnemonic, [
  // All TritonTilingExtOps have to implement the tiling interface.
  DeclareOpInterfaceMethods<TilingInterface, [
    "getLoopIteratorTypes",
    "getIterationDomain",
    "getTiledImplementation",
    "getResultTilePosition",
    "generateResultTileValue"
  ]>,
  // All TritonTilingExtOps implement TritonTilingExtInterface, which provides a standardized
  // way of providing indexing maps for input and output operands.
  DeclareOpInterfaceMethods<TritonTilingExtInterface>,

  // MemoryEffectsOpInterface provides analysis passes such as DCE to determine
  // whether an operation has no memory side effects and therefore is safe to
  // be deleted. This interface is important during tile and fuse where we
  // create copies of TilingInterface ops with smaller tile sizes but leave the
  // original ops intact.
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,

  // DestinationStyleOpInterface describes ops that have similar semantics to
  // linalg ops, with a separate ins (input) and outs (output) operand groups.
  // Implementing this op gives us access to a wide variety of useful methods
  // to query the inputs and outputs of an op.
  DestinationStyleOpInterface,

  // AttrSizedOperandSegments supports having multiple groups of operands.
  // For example, linalg ops (as well as TritonTilingExtOps) all look like this:
  // ttx.some_op ins(%1) outs(%2) -> resultType
  AttrSizedOperandSegments
]>
{
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let hasCustomAssemblyFormat = 1;

  code baseClassDecls = [{
    // Implemented as part of DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }
  }];

  // Custom print() and parse() methods to make the TritonTilingExt ops have similar looks
  // to the linalg ops.
  // Borrowed from llvm-project/mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp
  let extraClassDefinition = [{
    void $cppClass::print(OpAsmPrinter &p) {
      p.printOptionalAttrDict(this->getOperation()->getAttrs(),
                              /*elidedAttrs=*/{"operand_segment_sizes"});

      if (!getInputs().empty())
        p << " ins(" << getInputs() << " : " << getInputs().getTypes() << ")";
      if (!getOutputs().empty())
        p << " outs(" << getOutputs() << " : " << getOutputs().getTypes() << ")";

      if (!getResultTypes().empty())
        p.printOptionalArrowTypeList(getResultTypes());
    }

    ParseResult $cppClass::parse(OpAsmParser &parser,
                                              OperationState &result) {
      SmallVector<Type> inputTypes;
      SmallVector<Type> outputTypes;
      SMLoc inputsOperandsLoc, outputsOperandsLoc;
      SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands,
          outputsOperands;
      if (parser.parseOptionalAttrDict(result.attributes))
        return failure();

      if (succeeded(parser.parseOptionalKeyword("ins"))) {
        if (parser.parseLParen())
          return failure();

        inputsOperandsLoc = parser.getCurrentLocation();
        if (parser.parseOperandList(inputsOperands) ||
            parser.parseColonTypeList(inputTypes) || parser.parseRParen())
          return failure();
      }

      if (succeeded(parser.parseOptionalKeyword("outs"))) {
        outputsOperandsLoc = parser.getCurrentLocation();
        if (parser.parseLParen() || parser.parseOperandList(outputsOperands) ||
            parser.parseColonTypeList(outputTypes) || parser.parseRParen())
          return failure();
      }

      if (parser.resolveOperands(inputsOperands, inputTypes, inputsOperandsLoc,
                                result.operands) ||
          parser.resolveOperands(outputsOperands, outputTypes,
                                outputsOperandsLoc, result.operands))
        return failure();

      result.addAttribute("operand_segment_sizes",
                          parser.getBuilder().getDenseI32ArrayAttr(
                              {static_cast<int32_t>(inputsOperands.size()),
                              static_cast<int32_t>(outputsOperands.size())}));

      SmallVector<Type, 1> resultTypes;
      if (parser.parseOptionalArrowTypeList(resultTypes))
        return failure();
      result.addTypes(resultTypes);

      return success();
    }

    AffineMap $cppClass::getIndexingMap(MLIRContext *context,
                             unsigned int index,
                             ArrayRef<OpFoldResult> sizes) {
      assert(index < this->getNumOperands());
      if (index < getNumDpsInputs()) {
        return getInputIndexingMap(context, index, sizes);
      }
      return getOutputIndexingMap(context, index - getNumDpsInputs(), sizes);
    }

    // Forward each of the implementation to the shared implementation
    FailureOr<TilingResult> $cppClass::getTiledImplementation(
      OpBuilder &b,
      ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes
    ) {
      return mlir::ttx::getTiledImplementation<$cppClass>(
        *this, b, offsets, sizes
      );
    }

    // Forward each of the implementation to the shared implementation
    LogicalResult $cppClass::getResultTilePosition(
      OpBuilder &b,
      unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes,
      SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes
    ) {
      return mlir::ttx::getResultTilePosition<$cppClass>(
        *this, b, resultNumber, offsets, sizes, resultOffsets, resultSizes
      );
    }

    // Forward each of the implementation to the shared implementation
    FailureOr<TilingResult> $cppClass::generateResultTileValue(
      OpBuilder &b,
      unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets,
      ArrayRef<OpFoldResult> sizes
    ) {
      return mlir::ttx::generateResultTileValue<$cppClass>(
        *this, b, resultNumber, offsets, sizes
      );
    }

    // Implemented as part of MemoryEffectsOpInterface
    void $cppClass::getEffects(
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects
    ) {
      return mlir::ttx::getEffects<$cppClass>(*this, effects);
    }
  }];
}

def TritonTilingExt_CumSumOp : TritonTilingExt_TilingOp<"cumsum"> {
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outputs,
    UI32Attr:$axis
  );

  let hasVerifier = 1;

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins
      "Value":$input,
      "IntegerAttr":$axis,
      "Value":$output,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes
    )>
  ];

  let extraClassDeclaration = baseClassDecls # [{
    int64_t getRank() {
      return cast<ShapedType>(getInput().getType()).getRank();
    }

    Value getInput() {
      return getInputs()[0];
    }

    Value getOutput() {
      return getOutputs()[0];
    }

    static StringRef getAxisAttrStrName() { return "axis"; }
  }];
}

#endif // TRITON_TILING_EXT_BASE
