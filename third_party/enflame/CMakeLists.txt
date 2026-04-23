#set(CMAKE_INCLUDE_DIRECTORIES_BEFORE ON)


set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

if (DEFINED ENV{LLVM_SYSPATH})
else()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add cmake module path so that include() can find the corresponding files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/2nd")

# Add build for triton_enflame plugin
add_triton_plugin(TritonEnflame
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_enflame.cc

    DEPENDS
    TritonTableGen
)

# Set include directories
target_include_directories(TritonEnflame PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_gcu/include
)

# Link necessary libraries
target_link_libraries(TritonEnflame PRIVATE
    # MLIR core libraries
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIROptLib

    # Python binding libraries
    Python3::Module
    pybind11::headers

    # If GCU-related libraries need to be linked, they can be added here
    # GCUIRgcu300
    # TritonGCUIR_gcu300
    # MLIRTritonGCUCommon_gcu300
    # MLIRTritonToGCU_gcu300
    # MLIRTritonGCUTransforms_gcu300
)

# Set compilation options
target_compile_definitions(TritonEnflame PRIVATE
    -DTRITON_ENFLAME_VERSION="1.0.0"
)


set(HACK_INCLUDE_DIRS true)
if(HACK_INCLUDE_DIRS)

    get_property(inc_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)

    list(REMOVE_ITEM inc_dirs "${CMAKE_SOURCE_DIR}/include")
    list(REMOVE_ITEM inc_dirs "${CMAKE_BINARY_DIR}/include")
    list(REMOVE_ITEM inc_dirs "${CMAKE_SOURCE_DIR}/third_party")
    list(REMOVE_ITEM inc_dirs "${CMAKE_BINARY_DIR}/third_party")
    list(REMOVE_ITEM inc_dirs "${MLIR_INCLUDE_DIRS}")
    list(REMOVE_ITEM inc_dirs "${LLVM_INCLUDE_DIRS}")

    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES "${include_dirs}")
endif()

# Add triton_gcu subdirectory build
add_subdirectory(triton_gcu)
