#!/usr/bin/env bash
# Build zlib for riscv64 and install it into the toolchain sysroot, so the
# cross-compiled triton can link -lz. Run this once per toolchain with
# RISCV_ROOT_PATH set (see README_build.md, "Cross-compile a riscv64 wheel").
set -e
wget https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
tar -zxvf zlib-1.3.1.tar.gz

# Build in a subshell to avoid polluting the outer environment
(
  cd zlib-1.3.1

  # Use SpacemiT cross-compiler
  export CC=${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-gcc
  export AR=${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-ar
  export RANLIB=${RISCV_ROOT_PATH}/bin/riscv64-unknown-linux-gnu-ranlib

  # Install into the toolchain sysroot
  ./configure --prefix=${RISCV_ROOT_PATH}/sysroot/usr
  make -j4
  make install
)

rm -rf zlib-1.3.1.tar.gz zlib-1.3.1
