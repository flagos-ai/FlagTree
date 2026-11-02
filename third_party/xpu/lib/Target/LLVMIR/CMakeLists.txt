add_triton_library(TritonIntrinsicAttrTable
  IntrinsicAttrTable.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSupport
  )

# XTDK vs public-LLVM feature probe — same rule as the sync source's root
# CMakeLists: the XTDK-private XTDKDL MLIR dialect header exists only in XTDK
# trees (llvm_trust here), and IntrinsicAttrTable.cpp branches on
# TRITON_HAVE_XTDKDL for the MemoryEffectsAttr arity (3 vs 6 memory locations).
if(EXISTS "${LLVM_INCLUDE_DIRS}/mlir/Dialect/LLVMIR/XTDKDLDialect.h")
  message(STATUS "XTDKDL dialect found (XTDK tree)")
  target_compile_definitions(TritonIntrinsicAttrTable PRIVATE TRITON_HAVE_XTDKDL)
else()
  message(STATUS "XTDKDL dialect absent (public LLVM)")
endif()
