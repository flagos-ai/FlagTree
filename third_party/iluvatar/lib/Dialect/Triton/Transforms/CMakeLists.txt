set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(TritonCombine.inc -gen-rewriters)
add_public_tablegen_target(TritonCombineIncGen)

if(NOT FLAGTREE_BACKEND STREQUAL "iluvatar")
  set(_EXTRA_LINK_LIBS ExpressionRestructing.cpp)
else()
  set(_EXTRA_LINK_LIBS "")
endif()

add_triton_library(TritonTransforms
  Combine.cpp
  ReorderBroadcast.cpp
  RewriteTensorPointer.cpp
  ${_EXTRA_LINK_LIBS}

  DEPENDS
  TritonTransformsIncGen
  TritonCombineIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  TritonIR
)
