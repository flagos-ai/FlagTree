/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITONGPU_OP_INTERFACES
#define TRITONGPU_OP_INTERFACES

include "mlir/IR/OpBase.td"

def UpcastFpOpInterface : OpInterface<"UpcastFpOpInterface"> {
    let description = [{
        This interface is for operations that upcast floating-point numbers.
    }];

    let cppNamespace = "::mlir::triton::gpu";

    let methods = [
        InterfaceMethod<
            /*desc=*/"Infer destination encoding",
            /*retType=*/"mlir::Attribute",
            /*methodName=*/"inferDstEncoding",
            /*args=*/(ins "unsigned":$opIdx, "mlir::Attribute":$srcEnc)
        >,
        InterfaceMethod<
            /*desc=*/"Infer operand encoding from dst encoding",
            /*retType=*/"mlir::Attribute",
            /*methodName=*/"inferSrcEncoding",
            /*args=*/(ins "unsigned":$opIdx, "mlir::Attribute":$dstEnc)
        >
    ];
}

#define TRITON_GPU_WARP_SLICE_INTERFACE

def WarpSliceOpInterface : OpInterface<"WarpSliceOpInterface"> {
    let description = [{
        A register view whose result is held by a contiguous subset of the
        source's warps. The view preserves lane and CTA ownership. Its register
        mapping and warp range are proven from the layouts and logical offsets,
        independently of the operation that will dispatch those warps.
    }];
    let cppNamespace = "::mlir::triton::gpu";
    let methods = [
        InterfaceMethod<"Source of the register view", "mlir::Value",
                        "getWarpSliceSource">,
        InterfaceMethod<"Result of the register view", "mlir::Value",
                        "getWarpSliceResult">,
        InterfaceMethod<"Prove the warp range and per-thread register mapping",
                        "std::optional<::mlir::triton::gpu::WarpSlice>",
                        "getWarpSlice">
    ];
}

#endif // TRITONGPU_OP_INTERFACES
