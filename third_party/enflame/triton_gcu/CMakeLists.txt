include(ExternalProject)

set(FETCHCONTENT_QUIET OFF)
if(CMAKE_TOOLCHAIN_FILE_FULL_PATH)
        message(STATUS "[triton-${arch}]: CMAKE_TOOLCHAIN_FILE_FULL_PATH: ${CMAKE_TOOLCHAIN_FILE_FULL_PATH}")
        set(TOOLCHAIN_PARM "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE_FULL_PATH}")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/gcu_compiler/include)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/gcu_compiler/lib)
include_directories(SYSTEM ${CMAKE_BINARY_DIR}/gcu_compiler/include)

# need set PIC for triton_gcu library
add_compile_options(-fPIC)

# Wheel ships all architectures; TARGET_PROFILE only affects deb/gcu_compiler layout.
set(archs gcu300 gcu400)
if(BUILD_GCU500)
    list(APPEND archs gcu500)
endif()
message(STATUS "Building GCU architectures for triton_gcu: ${archs}")
# Order matters for GLOBAL TRITON_ORIG_VERSION: each triton_${arch} run overwrites; use the last (e.g. gcu400).
foreach(arch ${archs})
        string(TOUPPER ${arch} ARCH)
        add_subdirectory(triton_${arch})
endforeach()
