#!/bin/bash

# 需要在triton的根目录执行这个脚本
if [ ! -d 'python/triton' ] || [ ! -d 'third_party/sunrise' ] ; then
    echo "This script must be executed in triton project root directory!"
    exit 1
fi

if [ $# -eq 1 ] && [ $1 = 'all' ] ; then
    rm -f python/triton/FileCheck
    rm -f third_party/sunrise/backend/lib/*.bc
fi

rm -rf python/triton.egg-info
rm -rf python/triton/_C
rm -rf build

echo "--- OK ---"
