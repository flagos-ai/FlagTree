add_mlir_dialect_library(MemrefExtIR_gcu300
  MemrefExtDialect.cpp
  MemrefExtOps.cpp

  DEPENDS
  MemrefExtTableGen_gcu300

  LINK_LIBS PUBLIC
  MLIRIR
)

# Python extensions link this library; require PIC for correct linking.
set_property(TARGET MemrefExtIR_gcu300 PROPERTY POSITION_INDEPENDENT_CODE ON)
if(TARGET obj.MemrefExtIR_gcu300)
  set_property(TARGET obj.MemrefExtIR_gcu300 PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
