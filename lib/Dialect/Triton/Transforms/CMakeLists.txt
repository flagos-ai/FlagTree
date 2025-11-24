set_flagtree_backend_td(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(TritonCombine.inc -gen-rewriters)
add_public_tablegen_target(TritonCombineIncGen)

if(IS_COMPILE_TritonNvidiaGPU)
  set(_EXPRESSION_LINK_CPP "ExpressionRestructing.cpp")
else()
  set(_EXPRESSION_LINK_CPP "")
endif()

get_flagtree_backend_lib("TritonTransforms" _EXTRA_LINK_LIBS)

add_triton_library(TritonTransforms
  Combine.cpp
  ReorderBroadcast.cpp
  RewriteTensorPointer.cpp
  ${_EXPRESSION_LINK_CPP}

  DEPENDS
  TritonTransformsIncGen
  TritonCombineIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  TritonIR
  ${_EXTRA_LINK_LIBS}
)
