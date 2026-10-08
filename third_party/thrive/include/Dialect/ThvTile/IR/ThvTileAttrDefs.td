#ifndef THVTILE_ATTRDEFS
#define THVTILE_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "ThvTileDialect.td"

//===----------------------------------------------------------------------===//
// Attribute base class
//===----------------------------------------------------------------------===//

class ThvTile_Attr<string name, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
  : AttrDef<ThvTile_Dialect, name, traits, baseCppClass> {
}

//===----------------------------------------------------------------------===//
// Enum helper classes
//===----------------------------------------------------------------------===//

class ThvTile_I32Enum<string name, string description,
                      list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::thvtile";
}

class ThvTile_I32EnumAttr<string mnemonic, ThvTile_I32Enum enumInfo>
    : EnumAttr<ThvTile_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Block Dimension
//===----------------------------------------------------------------------===//

def ThvTile_BlockDim_X : I32EnumAttrCase<"X", 0, "x">;
def ThvTile_BlockDim_Y : I32EnumAttrCase<"Y", 1, "y">;
def ThvTile_BlockDim_Z : I32EnumAttrCase<"Z", 2, "z">;

def ThvTile_BlockDimEnum : ThvTile_I32Enum<
  "BlockDim", "Block grid dimension", [
    ThvTile_BlockDim_X,
    ThvTile_BlockDim_Y,
    ThvTile_BlockDim_Z
  ]>;

def ThvTile_BlockDimAttr : ThvTile_I32EnumAttr<"block_dim", ThvTile_BlockDimEnum>;

//===----------------------------------------------------------------------===//
// Atomic RMW Kind
// Floating-point atomic operations are not supported.
//===----------------------------------------------------------------------===//

def ThvTile_RMW_ADD  : I32EnumAttrCase<"ADD", 0, "add">;
def ThvTile_RMW_MAX  : I32EnumAttrCase<"MAX", 1, "max">;
def ThvTile_RMW_MIN  : I32EnumAttrCase<"MIN", 2, "min">;
def ThvTile_RMW_AND  : I32EnumAttrCase<"AND", 3, "and">;
def ThvTile_RMW_OR   : I32EnumAttrCase<"OR", 4, "or">;
def ThvTile_RMW_XOR  : I32EnumAttrCase<"XOR", 5, "xor">;
def ThvTile_RMW_XCHG : I32EnumAttrCase<"XCHG", 6, "xchg">;

def ThvTile_AtomicRMWKindEnum : ThvTile_I32Enum<
  "AtomicRMWKind", "Atomic read-modify-write operation kind", [
    ThvTile_RMW_ADD, ThvTile_RMW_MAX, ThvTile_RMW_MIN,
    ThvTile_RMW_AND, ThvTile_RMW_OR, ThvTile_RMW_XOR,
    ThvTile_RMW_XCHG
  ]>;

def ThvTile_AtomicRMWKindAttr : ThvTile_I32EnumAttr<"atomic_rmw_kind",
                                                     ThvTile_AtomicRMWKindEnum>;

//===----------------------------------------------------------------------===//
// Memory Semantics (for atomics)
//===----------------------------------------------------------------------===//

def ThvTile_MemSem_RELAXED : I32EnumAttrCase<"RELAXED", 0, "relaxed">;
def ThvTile_MemSem_ACQUIRE : I32EnumAttrCase<"ACQUIRE", 1, "acquire">;
def ThvTile_MemSem_RELEASE : I32EnumAttrCase<"RELEASE", 2, "release">;
def ThvTile_MemSem_ACQ_REL : I32EnumAttrCase<"ACQ_REL", 3, "acq_rel">;

def ThvTile_MemSemanticEnum : ThvTile_I32Enum<
  "MemSemantic", "Memory ordering semantic", [
    ThvTile_MemSem_RELAXED, ThvTile_MemSem_ACQUIRE,
    ThvTile_MemSem_RELEASE, ThvTile_MemSem_ACQ_REL
  ]>;

def ThvTile_MemSemanticAttr : ThvTile_I32EnumAttr<"mem_semantic",
                                                   ThvTile_MemSemanticEnum>;

//===----------------------------------------------------------------------===//
// Memory Scope (for atomics)
// Atomic ops currently only accept block scope; device is reserved.
//===----------------------------------------------------------------------===//

def ThvTile_MemScope_BLOCK  : I32EnumAttrCase<"BLOCK", 0, "block">;
def ThvTile_MemScope_DEVICE : I32EnumAttrCase<"DEVICE", 1, "device">;

def ThvTile_MemScopeEnum : ThvTile_I32Enum<
  "MemScope", "Memory synchronization scope", [
    ThvTile_MemScope_BLOCK, ThvTile_MemScope_DEVICE
  ]>;

def ThvTile_MemScopeAttr : ThvTile_I32EnumAttr<"mem_scope",
                                                ThvTile_MemScopeEnum>;

//===----------------------------------------------------------------------===//
// Matmul Quantization Format
//===----------------------------------------------------------------------===//

def ThvTile_QFmt_NONE       : I32EnumAttrCase<"NONE", 0, "none">;

def ThvTile_QuantFormatEnum : ThvTile_I32Enum<
  "QuantFormat", "Quantization format for scaled matmul", [
    ThvTile_QFmt_NONE
  ]>;

def ThvTile_QuantFormatAttr : ThvTile_I32EnumAttr<"quant_format",
                                                   ThvTile_QuantFormatEnum>;

//===----------------------------------------------------------------------===//
// Rounding Mode
//===----------------------------------------------------------------------===//

def ThvTile_Round_RTNE : I32EnumAttrCase<"RTNE", 0, "rtne">;
def ThvTile_Round_RTZ  : I32EnumAttrCase<"RTZ", 1, "rtz">;
def ThvTile_Round_RDN  : I32EnumAttrCase<"RDN", 2, "rdn">;
def ThvTile_Round_RUP  : I32EnumAttrCase<"RUP", 3, "rup">;

def ThvTile_RoundingModeEnum : ThvTile_I32Enum<
  "RoundingMode", "Rounding mode for cast operations", [
    ThvTile_Round_RTNE,
    ThvTile_Round_RTZ,
    ThvTile_Round_RDN,
    ThvTile_Round_RUP
  ]>;

def ThvTile_RoundingModeAttr : ThvTile_I32EnumAttr<"rounding_mode",
                                                    ThvTile_RoundingModeEnum>;

//===----------------------------------------------------------------------===//
// Mask Pattern Kind
//===----------------------------------------------------------------------===//

def ThvTile_Mask_NONE      : I32EnumAttrCase<"NONE", 0, "none">;
def ThvTile_Mask_BOUNDARY  : I32EnumAttrCase<"BOUNDARY", 1, "boundary">;
def ThvTile_Mask_STRIDED   : I32EnumAttrCase<"STRIDED", 2, "strided">;
def ThvTile_Mask_ARBITRARY : I32EnumAttrCase<"ARBITRARY", 3, "arbitrary">;

def ThvTile_MaskKindEnum : ThvTile_I32Enum<
  "MaskKind", "Structured mask pattern kind", [
    ThvTile_Mask_NONE,
    ThvTile_Mask_BOUNDARY,
    ThvTile_Mask_STRIDED,
    ThvTile_Mask_ARBITRARY
  ]>;

def ThvTile_MaskKindAttr : ThvTile_I32EnumAttr<"mask_kind", ThvTile_MaskKindEnum>;

#endif
