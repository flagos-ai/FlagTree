add_triton_library(TritonTools
  GenericSwizzling.cpp
  LayoutUtils.cpp
  LinearLayout.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  f2reduce
)

# MetaX-only source: the main tree has no matching source for the generic
# backend override mechanism.
target_sources(TritonTools PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../spec_cpp/lib/Tools/SwizzleDumpUtils.cpp
)
