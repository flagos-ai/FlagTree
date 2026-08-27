include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/gcu_compiler/include)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/gcu_compiler/lib)
include_directories(SYSTEM ${CMAKE_BINARY_DIR}/gcu_compiler/include)

# need set PIC for triton_gcu library
add_compile_options(-fPIC)

# Wheel ships all architectures; TARGET_PROFILE only affects deb/gcu_compiler layout.
if((NOT ENABLE_GCU300))
    set(archs gcu400)
else()
    set(archs gcu300 gcu400)
endif()
if(ENABLE_GCU500)
    list(APPEND archs gcu500)
endif()
message(STATUS "Building GCU architectures for triton_gcu: ${archs}")
foreach(arch ${archs})
        string(TOUPPER ${arch} ARCH)
        add_subdirectory(triton_${arch})
endforeach()

