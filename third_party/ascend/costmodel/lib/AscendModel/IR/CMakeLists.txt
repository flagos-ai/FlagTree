# The AscendModelIR library contains the dialect and ops implementation.
# TableGen targets are defined in include/AscendModel/CMakeLists.txt.

add_mlir_library(AscendModelIR
  AscendModelDialect.cpp
  AscendModelOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/AscendModel/IR

  DEPENDS
  AscendModelOpsIncGen
  AscendModelInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRInferTypeOpInterface
)
