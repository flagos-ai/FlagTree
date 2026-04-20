add_mlir_dialect_library(MemrefExtIR${arch}
  MemrefExtDialect.cpp
  MemrefExtOps.cpp

  DEPENDS
  MemrefExtTableGen${arch}

  LINK_LIBS PUBLIC
  MLIRIR
)

# Python 扩展（.so）会把该静态库/对象文件注入链接，要求以 -fPIC 编译。
set_property(TARGET MemrefExtIR${arch} PROPERTY POSITION_INDEPENDENT_CODE ON)
if(TARGET obj.MemrefExtIR${arch})
  set_property(TARGET obj.MemrefExtIR${arch} PROPERTY POSITION_INDEPENDENT_CODE ON)
elseif(TARGET obj.MemrefExtIR)
  set_property(TARGET obj.MemrefExtIR PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
