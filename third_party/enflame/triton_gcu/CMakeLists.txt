

include(ExternalProject)

set(FETCHCONTENT_QUIET OFF)
if(CMAKE_TOOLCHAIN_FILE_FULL_PATH)
        message(STATUS "[triton-${arch}]: CMAKE_TOOLCHAIN_FILE_FULL_PATH: ${CMAKE_TOOLCHAIN_FILE_FULL_PATH}")
        set(TOOLCHAIN_PARM "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE_FULL_PATH}")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

foreach(arch gcu300 gcu400)
        # get upper case of arch
        string(TOUPPER ${arch} ARCH)
        add_subdirectory(triton_${arch})
endforeach()
