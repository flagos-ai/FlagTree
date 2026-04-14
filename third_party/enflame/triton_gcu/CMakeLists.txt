

include(ExternalProject)

set(FETCHCONTENT_QUIET OFF)
if(CMAKE_TOOLCHAIN_FILE_FULL_PATH)
        message(STATUS "[triton-${arch}]: CMAKE_TOOLCHAIN_FILE_FULL_PATH: ${CMAKE_TOOLCHAIN_FILE_FULL_PATH}")
        set(TOOLCHAIN_PARM "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE_FULL_PATH}")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(USE_CAPS_LLVM_LEGACY)
    message(STATUS "USE_CAPS_LLVM_LEGACY is enabled, only build gcu300")
    set(archs gcu300)
else()
    message(STATUS "USE_CAPS_LLVM_LEGACY is disabled, build gcu300 and gcu400")
    set(archs gcu300 gcu400)
endif()
foreach(arch ${archs})
        # get upper case of arch
        string(TOUPPER ${arch} ARCH)
        add_subdirectory(triton_${arch})
endforeach()
