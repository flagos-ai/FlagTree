## spine-triton

spine-triton is forked from [microsoft/triton-shared](https://github.com/microsoft/triton-shared), which is a Shared Middle-Layer for Triton Compilation.

## QuickStart
1. env setup
~~~
apt update

apt install gcc g++ gdb libsleef-dev libnuma-dev libomp5 libgomp1 python3-dev

pip install PyYAML sympy torch opencv-python pybind11 --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
~~~

2. prebuild whl
~~~
pip install triton --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
~~~

3. mm demo
~~~
# python/examples/test_smt_mm.py
# export SPINE_TRITON_DUMP_PATH=./ir_dumps # for ir dump
python3 python/examples/test_smt_mm.py
~~~

## Build
1. llvm
~~~
# at spine-triton
# pull llvm-project
# llvm-project hash-tag at spine-triton/triton/cmake/llvm-hash.txt
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout f409804208e93ccf0fd10cc805c55f799126f38e

mkdir -p build-llvm-riscv64
pushd build-llvm-riscv64
cmake -G Ninja ../llvm-project/llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
   -DLLVM_TARGETS_TO_BUILD="RISCV" \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_BUILD_UTILS=ON \
   -DLLVM_ENABLE_LIBEDIT=OFF \
   -DLLVM_INSTALL_UTILS=ON \
   -DLLVM_INCLUDE_TESTS=OFF \
   -DLLVM_BUILD_TESTS=OFF \
   -DLLVM_ENABLE_LLD=OFF \
   -DCMAKE_INSTALL_PREFIX=installed \
   -DCMAKE_TOOLCHAIN_FILE=../cmake/linux_riscv64.toolchain.cmake

cmake --build . --target install --parallel 40
popd
~~~

2. download spine-mlir
~~~
wget https://github.com/spacemit-com/spine-mlir/releases/download/0.5.4/spine-mlir-riscv64-0.5.4.tar.gz
~~~

3. install triton
~~~
git submodule update --init --recursive
bash scripts/build.sh ${LLVM_INSTALL_DIR} {arch/x86_64/riscv64} {spine-mlir-install-dir}
~~~

## Code Style

Use the same pre-commit checks as CI before sending a PR.

~~~
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip pre-commit

# check only staged files
pre-commit run

# check all tracked files covered by the hooks
pre-commit run --all-files
~~~

## License

spine-triton is licensed under the [MIT license](/LICENSE).
