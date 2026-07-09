/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TRITON_HCUGPU_ATTRDEFS
#define TRITON_HCUGPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "TritonHCUGPUDialect.td"
include "mlir/IR/EnumAttr.td"

class TritonHCUGPU_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<TritonHCUGPU_Dialect, name, traits, baseCppClass> {
}

def SetFP8Clamping : TritonHCUGPU_Attr<"SetFP8Clamping"> {
  let mnemonic = "amdgcn.set.fp8.clamping";
}

class TritonHCUGPU_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::hcugpu";
}

class TritonHCUGPU_I32EnumAttr<string mnemonic, TritonHCUGPU_I32Enum enumInfo> :
    EnumAttr<TritonHCUGPU_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
  let cppNamespace = "::mlir::triton::hcugpu";
}

def SchedHintCaseNone : I32EnumAttrCase<"none", 0>;
def SchedHintCaseAttention : I32EnumAttrCase<"attention", 2>;

def TritonHCUGPU_SchedHintsEnum : TritonHCUGPU_I32Enum<
  "SchedHint", "Instruction Scheduling Hints for HCU GPUs", [
    SchedHintCaseNone,
    SchedHintCaseAttention,
  ]>;

def TritonHCUGPU_SchedHintVariantAttr :
  TritonHCUGPU_I32EnumAttr<"SchedHintVariant", TritonHCUGPU_SchedHintsEnum>;

def TritonHCUGPU_MlsEncoding : TritonHCUGPU_Attr<"MlsEncoding"> {
  let cppNamespace = "::mlir::triton::hcugpu";
  let mnemonic = "MlsEncoding";
  let summary = "mls encoding attribute.";
  let description = [{
    The attribute holds some information about mls.matrix_load operation.
    it include two parts: mls info(to restore matrix_load info)  and warpsPerCTA info.
      opIdx       :  the operand index for dot operation.
      mlsTile     :  the mls tile shape. opIdx == 0 ? [nonKTile, kTile] : [kTile, nonKTile]
      elemBitWidth:  the element bit width of the mls encoding attribute.
      alt2Kind    :  the alt2 kind of the mls encoding attribute.
      version     :  the version of the mls encoding attribute.
      order       :  the order of the tensor, the fastest-changing axis first.
      warpsPerCTA :  the number of warps per CTA. warpOrder follow the order field to fit triton framework.
  }];

  let parameters = (ins "unsigned":$opIdx,  // or nonKTileIdx
                        ArrayRefParameter<"unsigned">:$mlsTile,
                        "unsigned":$elemBitWidth,
                        "unsigned":$alt2Kind,
                        "unsigned":$version,
                        ArrayRefParameter<"unsigned">:$order,
                        ArrayRefParameter<"unsigned">:$warpsPerCTA);
  let assemblyFormat = "`opIdx` `=` $opIdx `,` "
                       "`mlsTile` `=` `[` $mlsTile `]` `,` "
                       "`elemBitWidth` `=` $elemBitWidth `,` "
                       "`alt2Kind` `=` $alt2Kind `,` "
                       "`version` `=` $version `,` "
                       "`order` `=` `[` $order `]` `,` "
                       "`warpsPerCTA` `=` `[` $warpsPerCTA `]`";
}

#endif
