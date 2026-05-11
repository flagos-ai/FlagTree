# libz.so
# ln -s /usr/lib/x86_64-linux-gnu/libz.so.1 /home/zyuli/local/lib/libz.so
export LIBRARY_PATH=/home/zyuli/local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/home/zyuli/local/lib:$LD_LIBRARY_PATH 

# highest priority
# set clang include path
export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu:$C_INCLUDE_PATH

# cmake reads and stores into CMAKE_CXX_FLAGS
export CXXFLAGS="-Wno-gcc-install-dir-libstdcxx --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11"

# clang and lld
export PATH=/home/zyuli/local/clang-lld/bin:$PATH

# mlir
export PYTHONPATH=/home/zyuli/local/flagtree-mlir/python_packages/mlir_core:$PYTHONPATH

# llvm
export LLVM_INCLUDE_DIRS=/home/zyuli/llvm-project/build-mlir/include
export LLVM_LIBRARY_DIR=/home/zyuli/llvm-project/build-mlir/lib
export LLVM_SYSPATH=/home/zyuli/llvm-project/build-mlir

# flagtree
export MAX_JOBS=32
export TRITON_BUILD_WITH_CLANG_LLD=1
export TRITON_BUILD_DIR=/home/zyuli/build/flagtree_extern_call

