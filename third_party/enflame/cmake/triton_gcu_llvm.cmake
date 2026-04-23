# 使用预编译的 LLVM
# 检查环境变量中指定的 LLVM 路径
if(DEFINED ENV{KURAMA_LLVM_DIR})
    message(STATUS ": using user provide llvm path $ENV{KURAMA_LLVM_DIR}")
    set(KURAMA_LLVM_DIR "$ENV{KURAMA_LLVM_DIR}")
elseif(KURAMA_LLVM_DIR AND EXISTS ${KURAMA_LLVM_DIR}/lib/cmake)
    message(STATUS ": using previous exists llvm")
else()
    message(FATAL_ERROR "KURAMA_LLVM_DIR environment variable is not set or LLVM not found at specified path")
endif()
