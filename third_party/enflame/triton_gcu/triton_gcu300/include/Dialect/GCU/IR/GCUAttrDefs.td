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
#ifndef GCU_ATTR_DEFS
#define GCU_ATTR_DEFS

include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "Dialect/GCU/IR/GCUDialect.td"

class GCU_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<GCU_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// GCU address space attribute.
//===----------------------------------------------------------------------===//

class GCU_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::gcu";
}
class GCU_I32EnumAttr<string mnemonic, GCU_I32Enum enumInfo> :
    EnumAttr<GCU_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def GCU_AddressSpaceGlobal : I32EnumAttrCase<"Global", 1, "global">;
def GCU_AddressSpaceWorkgroup : I32EnumAttrCase<"Workgroup", 2, "workgroup">;
def GCU_AddressSpacePrivate : I32EnumAttrCase<"Private", 3, "private">;
def GCU_AddressSpaceLocal : I32EnumAttrCase<"Local", 4, "local">;
def GCU_AddressSpaceEnum : GCU_I32Enum<
  "AddressSpace", "GCU address space", [
    GCU_AddressSpaceGlobal,
    GCU_AddressSpaceWorkgroup,
    GCU_AddressSpacePrivate,
    GCU_AddressSpaceLocal
  ]>;

def GCU_AddressSpaceAttr :
  GCU_I32EnumAttr<"address_space", GCU_AddressSpaceEnum> {
}

def GCU_MemoryFenceEnum : GCU_I32Enum<
  "MFenceType", "GCU memory fence type", [
    I32EnumAttrCase<"Memory", 0, "memory">,
    I32EnumAttrCase<"Local", 1, "local">,
    I32EnumAttrCase<"Share", 2, "share">,
    I32EnumAttrCase<"Device", 3, "device">
  ]>;

def GCU_MemoryFenceAttr :
  GCU_I32EnumAttr<"mfence_type", GCU_MemoryFenceEnum> {
}



//===----------------------------------------------------------------------===//
// GCU architecture attribute.
//===----------------------------------------------------------------------===//

def GCU_Architecture300 : I32EnumAttrCase<"GCU300", 1, "gcu300">;
def GCU_Architecture400 : I32EnumAttrCase<"GCU400", 2, "gcu400">;
def GCU_Architecture410 : I32EnumAttrCase<"GCU410", 3, "gcu410">;
def GCU_ArchitectureEnum : GCU_I32Enum<
  "Architecture", "GCU Architecture", [
    GCU_Architecture300,
    GCU_Architecture400,
    GCU_Architecture410,
  ]>;

def GCU_ArchAttr : GCU_I32EnumAttr<"gcu_architecture", GCU_ArchitectureEnum> {
}


//===----------------------------------------------------------------------===//
// GCU target attribute.
//===----------------------------------------------------------------------===//

def GCU_TargettAttr :
    GCU_Attr<"GCUTarget", "target"> {
  let description = [{
    GCU target attribute for controlling compilation of GCU targets. All
    parameters decay into default values if not present.

    Examples:

    1. Target with default values.
    ```
      gpu.module @mymodule [#gcu.target] attributes {...} {
        ...
      }
    ```

    2. Target with `gcu300` chip and fast math.
    ```
      gpu.module @mymodule [#gcu.target<arch = "gcu300", flags = {}>] {
        ...
      }
    ```
  }];
  let parameters = (ins
    DefaultValuedParameter<"int", "3", "Optimization level to apply.">:$O,
    StringRefParameter<"Target triple.", "\"amdgcn-amd-amdhsa\"">:$triple,
    StringRefParameter<"Target chip.", "\"gfx900\"">:$chip,
    StringRefParameter<"Target arch.", "\"gcu300\"">:$arch,
    StringRefParameter<"Target chip features.", "\"\"">:$features,
    StringRefParameter<"ABI version.", "\"1\"">:$abi,
    OptionalParameter<"DictionaryAttr", "Target specific flags.">:$flags,
    OptionalParameter<"ArrayAttr", "Files to link to the LLVM module.">:$link
  );
  let assemblyFormat = [{
    (`<` struct($O, $triple, $chip, $arch, $features, $abi, $flags, $link)^ `>`)?
  }];
  let builders = [
    AttrBuilder<(ins CArg<"int", "3">:$optLevel,
                     CArg<"StringRef", "\"amdgcn-amd-amdhsa\"">:$triple,
                     CArg<"StringRef", "\"gfx900\"">:$chip,
                     CArg<"StringRef", "\"gcu300\"">:$arch,
                     CArg<"StringRef", "\"\"">:$features,
                     CArg<"StringRef", "\"1\"">:$abiVersion,
                     CArg<"DictionaryAttr", "nullptr">:$targetFlags,
                     CArg<"ArrayAttr", "nullptr">:$linkFiles), [{
      return Base::get($_ctxt, optLevel, triple, chip, arch, features, abiVersion,
                       targetFlags, linkFiles);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    bool hasFlag(StringRef flag) const;
  }];
  let extraClassDefinition = [{
    bool $cppClass::hasFlag(StringRef flag) const {
      if (DictionaryAttr flags = getFlags())
        return flags.get(flag) != nullptr;
      return false;
    }
  }];
}

def GCU_ReduceOperationAttr : I64EnumAttr<
    "ReduceOperation", "",
    [
      I64EnumAttrCase<"SUM", 0, "sum">,
      I64EnumAttrCase<"MAXSI", 1, "maxsi">,
      I64EnumAttrCase<"MAXUI", 2, "maxui">,
      I64EnumAttrCase<"MAXF", 3, "maxf">,
      I64EnumAttrCase<"MINSI", 4, "minsi">,
      I64EnumAttrCase<"MINUI", 5, "minui">,
      I64EnumAttrCase<"MINF", 6, "minf">,
    ]> {
  let cppNamespace = "::mlir::gcu";
}

def GCU_VectorMovSftModeAttr : I64EnumAttr<
    "VectorMovSftMode", "",
    [
      I64EnumAttrCase<"SHFLB", 0, "shift_left_with_byte">,
      I64EnumAttrCase<"SHFRB", 1, "shift_right_with_byte">,
      I64EnumAttrCase<"SHFLQW", 2, "shift_left_with_qw">,
      I64EnumAttrCase<"SHFRQW", 3, "shift_right_with_qw">
    ]> {
  let cppNamespace = "::mlir::gcu";
}

// InputPrecision
def GCU_InputPrecisionAttr : I32EnumAttr<
    "InputPrecision", "",
    [
      I32EnumAttrCase<"TF32", 0, "tf32">,
      I32EnumAttrCase<"TF32x3", 1, "tf32x3">,
      I32EnumAttrCase<"IEEE", 2, "ieee">
    ]>{
  let cppNamespace = "::mlir::gcu";
}

// atomic
def GCU_AtomicRMWAttr : I32EnumAttr<
    "RMWOp", "",
    [
        I32EnumAttrCase<"AND", 1, "and">,
        I32EnumAttrCase<"OR", 2, "or">,
        I32EnumAttrCase<"XOR", 3, "xor">,
        I32EnumAttrCase<"ADD", 4, "add">,
        I32EnumAttrCase<"MAX", 5, "max">,
        I32EnumAttrCase<"MIN", 6, "min">,
        I32EnumAttrCase<"UMAX", 7, "umax">,
        I32EnumAttrCase<"UMIN", 8, "umin">,
        I32EnumAttrCase<"XCHG", 9, "exch">
    ]> {
    let cppNamespace = "::mlir::gcu";
}

def GCU_MemSemanticAttr : I32EnumAttr<
    "MemSemantic", "",
    [
      I32EnumAttrCase<"RELAXED", 1, "relaxed">,
      I32EnumAttrCase<"ACQUIRE", 2, "acquire">,
      I32EnumAttrCase<"RELEASE", 3, "release">,
      I32EnumAttrCase<"ACQUIRE_RELEASE", 4, "acq_rel">,
    ]> {
    let cppNamespace = "::mlir::gcu";
}

def GCU_MemSyncScopeAttr : I32EnumAttr<
    "MemSyncScope", "",
    [
      I32EnumAttrCase<"GCU", 1, "gcu">,
      I32EnumAttrCase<"CTA", 2, "cta">,
      I32EnumAttrCase<"SYSTEM", 3, "sys">,
    ]> {
    let cppNamespace = "::mlir::gcu";
}

#endif // GCU_ATTR_DEFS
