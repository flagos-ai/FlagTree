//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRITON_TILING_EXT_DIALECT_INTERFACES
#define MLIR_TRITON_TILING_EXT_DIALECT_INTERFACES

include "mlir/IR/OpBase.td"

//
// Linalg operators require providing affine maps that define how input / output
// buffers are accessed together with a region that defines how each output
// element is computed; this requirement doesn't work well for operations such as
// `scan`.
//
// Fortunately, the introduction of the TilingInterface allows us to add tiling
// and fusion support to operations that don't fit into the linalg dialect.
// This fits our purpose perfectly: our `scan` operators can be treated as an
// "opaque" / "completely abstract" operation that can be tiled on the batch
// dimensions -- we don't need to provide any associated body together with it.
//
// However, this doesn't mean that we entirely forgo the "indexing map" concept.
// For example, consider the following:
//
//  - ttx.scan ins(%1 : tensor<128x768xbf16>)
//                 outs(%2 : tensor<128x768xbf16>) -> tensor<128x768xbf16>
//
// Tiling the batch dimension gives us:
//
//  for (i = 0 to 128) {
//    %sliceIn = extract slice from input: tensor<1x768xbf16>
//    %sliceOut = extract slice from output: tensor<1x768xbf16>
//    %res = ttx.scan ins(slice : tensor<1x768xbf16>)
//                 outs(%2 : tensor<1x768xbf16>) -> tensor<1x768xbf16>
//    insert %res into output
//  }
//
// Now our `scan` op has the semantic of running `scan` on a rank-1 tensor and
// can be lowered further to other hardware-specific ops or external library
// calls.
//
// This tiling pattern is essentially the same as tiling a linalg.generic op
// with an identity map. The only difference is we don't need a body associated
// with our `scan` op.
//
// With this idea in mind, the TritonTilingExtInterface exposes methods
// that will be implemented individually by each TritonTilingExtOp, providing
// the indexing map for each input / output that can then be used to generate
// the correct slices during tiling and fusion.
//
// There might be other ops in the future that won't fit in this "indexing map"
// approach; we will consider making TritonTilingExtInterface an optional
// interface for such ops.
//

def TritonTilingExtInterface : OpInterface<"TritonTilingExtInterface"> {
  let cppNamespace = "::mlir::ttx";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing map for the input operand with the given `index`.
        The `tileSizes` input indicates the requested tile size during tiling
        in case the indexing map for the operator is dependent on it.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getInputIndexingMap",
      /*args=*/(ins "MLIRContext*":$context,
                    "unsigned int":$index,
                    "ArrayRef<OpFoldResult>":$tileSizes)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing map for the output operand with the given `index`.
        The `tileSizes` input indicates the requested tile size during tiling
        in case the indexing map for the operator is dependent on it.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getOutputIndexingMap",
      /*args=*/(ins "MLIRContext*":$context,
                    "unsigned int":$index,
                    "ArrayRef<OpFoldResult>":$tileSizes)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing map for the operand with the given `index`.
        This method returns the operand in order of inputs followed by outputs.
        The `tileSizes` input indicates the requested tile size during tiling
        in case the indexing map for the operator is dependent on it.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getIndexingMap",
      /*args=*/(ins "MLIRContext*":$context,
                    "unsigned int":$index,
                    "ArrayRef<OpFoldResult>":$tileSizes)
    >
  ];
}

#endif
