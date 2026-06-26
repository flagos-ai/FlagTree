set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(TritonCombine.inc -gen-rewriters)
add_public_tablegen_target(TritonCombineIncGen)

get_flagtree_backend_lib("TritonTransforms" _EXTRA_LINK_LIBS)

add_triton_library(TritonTransforms
  Combine.cpp
  LoopAwareCSE.cpp
  LoopInvariantCodeMotion.cpp
  LoopPeeling.cpp
  LoopUnroll.cpp
  ReorderBroadcast.cpp
  RewriteTensorPointer.cpp
  RewriteTensorDescriptorToPointer.cpp
  ArithTypeConversion.cpp
  FunctionTypeConversion.cpp

  DEPENDS
  TritonTransformsIncGen
  TritonCombineIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  MLIRTransforms
  MLIRSCFToControlFlow
  TritonIR
  ${_EXTRA_LINK_LIBS}
)
