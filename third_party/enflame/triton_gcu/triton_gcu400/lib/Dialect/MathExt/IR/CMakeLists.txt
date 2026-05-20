add_mlir_dialect_library(MathExtIR_gcu400
  MathExtDialect.cpp
  MathExtOps.cpp
  MathExtTypes.cpp

  DEPENDS
  MathExtTableGen_gcu400

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)

# Python extensions link this static library; require PIC for correct linking.
set_property(TARGET MathExtIR_gcu400 PROPERTY POSITION_INDEPENDENT_CODE ON)

# Object files are often built on the obj.<name> companion target from add_mlir_dialect_library.
if(TARGET obj.MathExtIR_gcu400)
  set_property(TARGET obj.MathExtIR_gcu400 PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
