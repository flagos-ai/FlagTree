# 使用预编译的 LLVM
# 检查环境变量中指定的 LLVM 路径
if(DEFINED ENV{KURAMA_LLVM_DIR_${ARCH}})
    message(STATUS ": using user provide llvm path $ENV{KURAMA_LLVM_DIR_${ARCH}}")
    set(KURAMA_LLVM_DIR_${ARCH} "$ENV{KURAMA_LLVM_DIR_${ARCH}}")
elseif(KURAMA_LLVM_DIR_${ARCH} AND EXISTS ${KURAMA_LLVM_DIR_${ARCH}}/lib/cmake)
    message(STATUS ": using previous exists llvm")
else()
    message(FATAL_ERROR "KURAMA_LLVM_DIR_${ARCH} environment variable is not set or LLVM not found at specified path")
endif()
