set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(TritonCombine.inc -gen-rewriters)
add_public_tablegen_target(TritonCombineIncGen)

set(TritonTransformsTleSources)
# MetaX CommonIR uses the slim frontend and does not generate TleTableGen.
if(FLAGTREE_TLE AND NOT (FLAGTREE_BACKEND STREQUAL "metax" AND FLAGTREE_COMMON_IR_ENABLED))
  list(APPEND TritonTransformsTleSources
    ${CMAKE_SOURCE_DIR}/third_party/tle/dialect/lib/Transforms/ReorderLoopLoads.cpp
  )
endif()

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
  ${TritonTransformsTleSources}

  DEPENDS
  TritonTransformsIncGen
  TritonCombineIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  MLIRTransforms
  MLIRSCFToControlFlow
  TritonIR
)
