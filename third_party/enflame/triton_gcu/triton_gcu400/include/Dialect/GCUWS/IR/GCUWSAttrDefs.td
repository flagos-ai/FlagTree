/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GCUWS_ATTRDEFS
#define GCUWS_ATTRDEFS

include "GCUWSDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class GCUWS_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<GCUWS_Dialect, name, traits, baseCppClass> {
}

def GCUWS_TypeArray : ArrayOfAttr<GCUWS_Dialect, "TypeArray", "type_array", "Type"> {}
def GCUWS_IntArray : ArrayOfAttr<GCUWS_Dialect, "IntArray", "int_array", "int"> {}

/*
// Type for synchronization tokens.
def GCUWS_TokenLoadTypeAttr : I32EnumAttr<
    "TokenLoadType", "",
    [
      I32EnumAttrCase<"None", 0, "none">,
      I32EnumAttrCase<"AsyncLoadOp", 1, "asyncLoadOp">,
      I32EnumAttrCase<"TMALoadOp", 2, "tmaLoadOp">,
      I32EnumAttrCase<"LocalStoreOp", 3, "localStoreOp">,
    ]>{
  let cppNamespace = "::mlir::triton::gcuws";
}

def GCUWS_AsyncOpAttr: I32EnumAttr<
  "AsyncOp", "",
  [
    I32EnumAttrCase<"NONE", 0, "none">,
    I32EnumAttrCase<"TMALoad", 1, "tma_load">,
    I32EnumAttrCase<"CpAsync", 2, "cp_async">,
  ]> {
  let cppNamespace = "::mlir::triton::gcuws";
  let genSpecializedAttr = 0;
}

def GCUWS_AsyncOpEnum : EnumAttr<GCUWS_Dialect, GCUWS_AsyncOpAttr, "async_op"> {
  let assemblyFormat = "`<` $value `>`";
}

def GCUWS_AsyncOpArrayAttr : TypedArrayAttrBase<GCUWS_AsyncOpEnum, "array of async op attributes">;
*/

#endif
