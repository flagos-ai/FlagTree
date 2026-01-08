// TODO: When upgrading to Triton 3.4.0, remove this file and use the upstream Triton file.
#ifndef TRITON_OP_INTERFACES
#define TRITON_OP_INTERFACES

include "mlir/IR/OpBase.td"


def TT_DescriptorOpInterface : OpInterface<"DescriptorOpInterface"> {
  let description = [{
    Common interface to get the descriptor argument from an operation on tensor descriptors.
  }];

  let cppNamespace = "::mlir::triton";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the descriptor",
      /*retType=*/"::mlir::TypedValue<mlir::triton::TensorDescType>",
      /*methodName=*/"getDesc",
      /*args=*/(ins)>,
  ];
}

def TT_DescriptorStoreLikeOpInterface : OpInterface<"DescriptorStoreLikeOpInterface", [TT_DescriptorOpInterface]> {
  let cppNamespace = "::mlir::triton";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get Source tensor",
      /*retType=*/"::mlir::TypedValue<mlir::RankedTensorType>",
      /*methodName=*/"getSrc",
      /*args=*/(ins)>,
  ];
}


#endif // TRITON_OP_INTERFACES
