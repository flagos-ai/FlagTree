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
#ifndef TRITON_GCU_TYPES
#define TRITON_GCU_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Dialect/TritonGCU/IR/TritonGCUDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//
// Types
//
class TTGCUTypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<TritonGCU_Dialect, name, traits> {
    // Used by dte cte
    let mnemonic = _mnemonic;
}

// Pointer Type
def TTGCU_PtrType : TTGCUTypeDef<"Ptr", "ptr", [MemRefElementTypeInterface]> {
    let summary = "Pointer type (`::mlir::triton::gcu::PtrType`) in GCU IR type system";
    let description = [{
        Pointer type in GCU IR type system.
    }];
    let parameters = (ins "Type":$elementType);
    let assemblyFormat = [{ `<` $elementType `>` }];
}

#endif //TRITON_GCU_TYPES
