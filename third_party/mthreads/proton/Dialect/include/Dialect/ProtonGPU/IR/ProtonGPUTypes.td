// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef PROTONGPU_TYPES
#define PROTONGPU_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "proton/Dialect/include/Dialect/ProtonGPU/IR/ProtonGPUDialect.td"
include "proton/Dialect/include/Dialect/ProtonGPU/IR/ProtonGPUAttrDefs.td"

class PTG_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<ProtonGPU_Dialect, name, traits> {
    let mnemonic = _mnemonic;
}

def PTG_SegmentType : PTG_TypeDef<"Segment", "segment", []> {
  let summary = "A segment in the internal buffer";
  let description = [{
    The `proton_gpu.segment` type represents a segment returned by `PTG_SegmentOp`.

    Each segment is private to a profiling unit as defined by the `granularity` attribute.
    The selected segments, specified by the `selectIds` attribute, collectively total `nBytes` bytes.

    When lowered to LLVM, a segment becomes a struct containing:
    - `base`: pointer to the start of the internal buffer
    - `segmentBase`: pointer to each segment's start in the internal buffer
    - `indexPtr`: pointer to the current index within the segment

    The segment can reside in global memory or shared memory depending on the `memorySpace` attribute.
  }];

  let parameters = (ins
    "int32_t":$nBytes,
    "Attribute":$memorySpace,
    EnumParameter<GranularityAttr>:$granularity,
    OptionalArrayRefParameter<"int32_t">:$selectIds
  );

  let assemblyFormat = [{
    `<` $nBytes `,` $memorySpace `,` $granularity (`,` `[` $selectIds^ `]`)?  `>`
  }];
}

#endif
