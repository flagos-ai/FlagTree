add_triton_library(EvasLinalgIR
  LinalgOpsExt.cpp

  DEPENDS
  EvasLinalgOpsExtIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
)

target_compile_options(EvasLinalgIR PRIVATE -Wno-deprecated-declarations)

