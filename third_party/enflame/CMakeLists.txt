#set(CMAKE_INCLUDE_DIRECTORIES_BEFORE ON)


set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

if (DEFINED ENV{LLVM_SYSPATH})
else()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 添加 cmake 模块路径，以便 include() 可以找到相应的文件
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/2nd")

# 添加 triton_enflame 插件的构建
add_triton_plugin(TritonEnflame
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_enflame.cc

    DEPENDS
    TritonTableGen
)

# 设置包含目录
target_include_directories(TritonEnflame PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_gcu/include
)

# 链接必要的库
target_link_libraries(TritonEnflame PRIVATE
    # MLIR 核心库
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIROptLib

    # Python 绑定库
    Python3::Module
    pybind11::headers

    # 如果需要链接 GCU 相关的库，可以在这里添加
    # GCUIRgcu300
    # TritonGCUIR_gcu300
    # MLIRTritonGCUCommon_gcu300
    # MLIRTritonToGCU_gcu300
    # MLIRTritonGCUTransforms_gcu300
)

# 设置编译选项
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

# 添加 triton_gcu 子目录构建
add_subdirectory(triton_gcu)
add_subdirectory(include)
