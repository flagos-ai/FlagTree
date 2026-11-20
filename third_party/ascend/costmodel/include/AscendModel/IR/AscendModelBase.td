//===- AscendModelBase.td - Base definitions for AscendModel dialect ------===//
//
// Defines the AscendModel dialect for Ascend 910B hardware modeling.
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MODEL_BASE_TD
#define ASCEND_MODEL_BASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// AscendModel Dialect Definition
//===----------------------------------------------------------------------===//

def AscendModel_Dialect : Dialect {
  let name = "ascend";
  let summary = "Ascend 910B Hardware Modeling Dialect";
  let description = [{
    This dialect provides operations for modeling Ascend 910B NPU hardware
    behavior and performance characteristics. It supports:

    - Cube Core operations (matrix multiply)
    - Vector Core operations (elementwise, reduction, etc.)
    - Memory transfer operations (MTE2, MTE3, FixPipe)
    - Pipeline modeling and scheduling

    The dialect is designed for performance analysis and roofline modeling.
  }];

  let cppNamespace = "::mlir::ascend";
}

//===----------------------------------------------------------------------===//
// Hardware Unit Enum
//===----------------------------------------------------------------------===//

def HWUnit_Cube     : I32EnumAttrCase<"Cube", 0, "cube">;
def HWUnit_CubeMTE2 : I32EnumAttrCase<"CubeMTE2", 1, "cube_mte2">;
def HWUnit_FixPipe  : I32EnumAttrCase<"FixPipe", 2, "fixpipe">;
def HWUnit_Vector   : I32EnumAttrCase<"Vector", 3, "vector">;
def HWUnit_VecMTE2  : I32EnumAttrCase<"VecMTE2", 4, "vec_mte2">;
def HWUnit_MTE3     : I32EnumAttrCase<"MTE3", 5, "mte3">;
def HWUnit_Scalar   : I32EnumAttrCase<"Scalar", 6, "scalar">;

def HWUnitAttr : I32EnumAttr<"HWUnit", "Hardware execution unit", [
  HWUnit_Cube, HWUnit_CubeMTE2, HWUnit_FixPipe,
  HWUnit_Vector, HWUnit_VecMTE2, HWUnit_MTE3, HWUnit_Scalar
]> {
  let cppNamespace = "::mlir::ascend";
}

//===----------------------------------------------------------------------===//
// Memory Space Enum
//===----------------------------------------------------------------------===//

def MemSpace_HBM : I32EnumAttrCase<"HBM", 0, "hbm">;
def MemSpace_L2  : I32EnumAttrCase<"L2", 1, "l2">;
def MemSpace_L1  : I32EnumAttrCase<"L1", 2, "l1">;
def MemSpace_UB  : I32EnumAttrCase<"UB", 3, "ub">;  // Unified Buffer

def MemSpaceAttr : I32EnumAttr<"MemSpace", "Memory space", [
  MemSpace_HBM, MemSpace_L2, MemSpace_L1, MemSpace_UB
]> {
  let cppNamespace = "::mlir::ascend";
}

//===----------------------------------------------------------------------===//
// Data Type for Operations
//===----------------------------------------------------------------------===//

def DataType_FP32 : I32EnumAttrCase<"FP32", 0, "fp32">;
def DataType_FP16 : I32EnumAttrCase<"FP16", 1, "fp16">;
def DataType_BF16 : I32EnumAttrCase<"BF16", 2, "bf16">;
def DataType_INT8 : I32EnumAttrCase<"INT8", 3, "int8">;
def DataType_INT32 : I32EnumAttrCase<"INT32", 4, "int32">;

def DataTypeAttr : I32EnumAttr<"DataType", "Data type for operations", [
  DataType_FP32, DataType_FP16, DataType_BF16, DataType_INT8, DataType_INT32
]> {
  let cppNamespace = "::mlir::ascend";
}

//===----------------------------------------------------------------------===//
// Vector Operation Kind
//===----------------------------------------------------------------------===//

def VecOpKind_Add    : I32EnumAttrCase<"Add", 0, "add">;
def VecOpKind_Sub    : I32EnumAttrCase<"Sub", 1, "sub">;
def VecOpKind_Mul    : I32EnumAttrCase<"Mul", 2, "mul">;
def VecOpKind_Div    : I32EnumAttrCase<"Div", 3, "div">;
def VecOpKind_Exp    : I32EnumAttrCase<"Exp", 4, "exp">;
def VecOpKind_Log    : I32EnumAttrCase<"Log", 5, "log">;
def VecOpKind_Max    : I32EnumAttrCase<"Max", 6, "max">;
def VecOpKind_Min    : I32EnumAttrCase<"Min", 7, "min">;
def VecOpKind_Select : I32EnumAttrCase<"Select", 8, "select">;
def VecOpKind_Cast   : I32EnumAttrCase<"Cast", 9, "cast">;

def VecOpKindAttr : I32EnumAttr<"VecOpKind", "Vector operation kind", [
  VecOpKind_Add, VecOpKind_Sub, VecOpKind_Mul, VecOpKind_Div,
  VecOpKind_Exp, VecOpKind_Log, VecOpKind_Max, VecOpKind_Min,
  VecOpKind_Select, VecOpKind_Cast
]> {
  let cppNamespace = "::mlir::ascend";
}

//===----------------------------------------------------------------------===//
// Reduction Kind
//===----------------------------------------------------------------------===//

def ReduceKind_Sum : I32EnumAttrCase<"Sum", 0, "sum">;
def ReduceKind_Max : I32EnumAttrCase<"Max", 1, "max">;
def ReduceKind_Min : I32EnumAttrCase<"Min", 2, "min">;
def ReduceKind_Prod : I32EnumAttrCase<"Prod", 3, "prod">;

def ReduceKindAttr : I32EnumAttr<"ReduceKind", "Reduction kind", [
  ReduceKind_Sum, ReduceKind_Max, ReduceKind_Min, ReduceKind_Prod
]> {
  let cppNamespace = "::mlir::ascend";
}

//===----------------------------------------------------------------------===//
// Base Operation Class
//===----------------------------------------------------------------------===//

class AscendModel_Op<string mnemonic, list<Trait> traits = []> :
    Op<AscendModel_Dialect, mnemonic, traits>;

#endif // ASCEND_MODEL_BASE_TD
