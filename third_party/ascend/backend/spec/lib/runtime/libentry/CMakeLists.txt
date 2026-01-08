add_library(entry_ascend SHARED
  libentry.cpp
)
target_compile_options(entry_ascend PRIVATE -fexceptions -frtti)
