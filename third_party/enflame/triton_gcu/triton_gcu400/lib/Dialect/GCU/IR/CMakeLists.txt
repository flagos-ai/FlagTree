add_mlir_dialect_library(GCUIR${arch}
  Dialect.cpp
  Types.cpp
  Ops.cpp

  DEPENDS
  GCUTableGen${arch}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)

# Python 扩展（.so）会把该 dialect 的实现对象文件注入/链接进去，要求以 -fPIC 编译。
# 否则 ld.gold 会报：requires dynamic R_X86_64_PC32 reloc ... recompile with -fPIC
set_property(TARGET GCUIR${arch} PROPERTY POSITION_INDEPENDENT_CODE ON)

# add_mlir_dialect_library 通常还会生成一个 object library（obj.<name>），
# 真正的 .o 是在那个 target 下编译的；只给静态库 target 设 PIC 不一定生效。
if(TARGET obj.GCUIR${arch})
  set_property(TARGET obj.GCUIR${arch} PROPERTY POSITION_INDEPENDENT_CODE ON)
elseif(TARGET obj.GCUIR)
  set_property(TARGET obj.GCUIR PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
