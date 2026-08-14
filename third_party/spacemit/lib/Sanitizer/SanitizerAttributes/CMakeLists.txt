add_library(SanitizerAttributes SHARED SanitizerAttributes.cpp)

# Allow undefined symbols in shared objects on Darwin (this is the default
# behaviour on Linux)
target_link_libraries(SanitizerAttributes "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

# This is set to -fvisibility=hidden in the top level CMake file
# which causes the llvmGetPassPluginInfo symbol to be hidden and
# an "entry point not found" error. Reset it just for this target
if(NOT MSVC)
  target_compile_options(SanitizerAttributes PRIVATE -fvisibility=default -fno-rtti)
endif()
