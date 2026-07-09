add_mlir_dialect_library(GCUIR_gcu400
  Dialect.cpp
  Types.cpp
  Ops.cpp

  DEPENDS
  GCUTableGen_gcu400

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRTransformDialect
)

# Python extensions link these objects; require PIC or the gold linker may fail.
set_property(TARGET GCUIR_gcu400 PROPERTY POSITION_INDEPENDENT_CODE ON)

# Object files are often built on the obj.<name> companion target from add_mlir_dialect_library.
if(TARGET obj.GCUIR_gcu400)
  set_property(TARGET obj.GCUIR_gcu400 PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
