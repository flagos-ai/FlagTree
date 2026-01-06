#!/bin/bash

# 参数检查
if [ $# -ne 2 ]; then
    echo "用法: $0 <requirements.txt> <existing_packages_dir>"
    exit 1
fi

REQUIREMENTS=$1
PACKAGES_DIR=$2

# 检查pip和requirements文件
if ! command -v pip &> /dev/null; then
    echo "错误: pip未安装"
    exit 1
fi

if [ ! -f "$REQUIREMENTS" ]; then
    echo "错误: 文件 $REQUIREMENTS 不存在"
    exit 1
fi

if [ ! -d "$PACKAGES_DIR" ]; then
    echo "错误: 目录 $PACKAGES_DIR 不存在"
    exit 1
fi

echo "正在检查并补充下载缺失的依赖包..."

# 使用exists-action=i选项，跳过已存在的包
pip download \
    -r "$REQUIREMENTS" \
    -d "$PACKAGES_DIR" \
    --exists-action=i \
    --no-deps  # 假设依赖项已经完整

if [ $? -ne 0 ]; then
    echo "错误: 下载依赖包失败"
    exit 1
fi

echo "完成! 缺失的依赖包已补充下载到 $PACKAGES_DIR"