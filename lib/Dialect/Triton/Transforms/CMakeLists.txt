set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(TritonCombine.inc -gen-rewriters)
add_public_tablegen_target(TritonCombineIncGen)

# flagtree tle: reorder-loop-loads is a TLE-owned extension. Its implementation
# lives under third_party/tle and is compiled into TritonTransforms only when
# TLE is enabled; the upstream LoopUnroll.cpp calls it behind the
# __FLAGTREE_REORDER_LOOP_LOADS__ build guard.
set(TritonTransformsTleSources)
if(FLAGTREE_TLE)
  list(APPEND TritonTransformsTleSources
    ${CMAKE_SOURCE_DIR}/third_party/tle/dialect/lib/Transforms/ReorderLoopLoads.cpp
  )
endif()

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
  ${_EXTRA_LINK_LIBS}
)
