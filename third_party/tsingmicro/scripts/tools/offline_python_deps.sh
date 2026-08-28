#!/bin/bash

# 定义颜色代码
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

setup_proxy() {
    PROXY=http://192.168.100.225:8889
    # Downloading python requirement is needed.
    export https_proxy=$PROXY http_proxy=$PROXY all_proxy=$PROXY
    export HTTPS_PROXY=$PROXY HTTP_PROXY=$PROXY ALL_PROXY=$PROXY
}

# 显示帮助信息
show_help() {
    echo "Python依赖包离线安装工具"
    echo "用法:"
    echo "  准备模式 (联网环境): $0 -p -r <requirements.txt> -o <output_dir>"
    echo "  安装模式 (离线环境): $0 -i -r <requirements.txt> -d <packages_dir>"
    echo "选项:"
    echo "  -p, --prepare       准备模式(下载依赖包)"
    echo "  -i, --install       安装模式(安装依赖包)"
    echo "  -r, --requirements  指定requirements文件"
    echo "  -o, --output        指定输出目录(准备模式)"
    echo "  -d, --directory     指定包目录(安装模式)"
    echo "  -v, --venv          指定虚拟环境目录(可选)"
    echo "  -h, --help          显示帮助信息"
}

# 检查命令是否存在
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}错误: $1 未安装${NC}"
        exit 1
    fi
}

# 准备模式
prepare_packages() {
    echo -e "${YELLOW}正在准备离线依赖包...${NC}"
    setup_proxy

    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 无法创建目录 $OUTPUT_DIR${NC}"
        exit 1
    fi

    # 下载依赖包
    echo -e "${GREEN}正在下载依赖包到 $OUTPUT_DIR ...${NC}"
    pip download -r "$REQUIREMENTS" -d "$OUTPUT_DIR"
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 下载依赖包失败${NC}"
        exit 1
    fi

    # 创建压缩包
    echo -e "${GREEN}正在创建压缩包...${NC}"
    tar -czf "${OUTPUT_DIR}.tar.gz" "$OUTPUT_DIR"

    echo -e "${GREEN}完成! 所有依赖包已下载到 $OUTPUT_DIR${NC}"
    echo -e "请将 ${YELLOW}${OUTPUT_DIR}.tar.gz${NC} 复制到离线环境"
}

# 安装模式
install_packages() {
    echo -e "${YELLOW}正在安装离线依赖包...${NC}"

    # 检查包目录
    if [ ! -d "$PACKAGES_DIR" ]; then
        echo -e "${RED}错误: 目录 $PACKAGES_DIR 不存在${NC}"
        exit 1
    fi

    # 如果指定了虚拟环境
    if [ -n "$VENV_DIR" ]; then
        echo -e "${GREEN}正在设置虚拟环境...${NC}"
        python -m venv "$VENV_DIR"
        if [ $? -ne 0 ]; then
            echo -e "${RED}错误: 创建虚拟环境失败${NC}"
            exit 1
        fi
        source "${VENV_DIR}/bin/activate"
    fi

    # 安装依赖包
    echo -e "${GREEN}正在安装依赖包...${NC}"
    pip install --no-index --find-links="$PACKAGES_DIR" -r "$REQUIREMENTS"
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 安装依赖包失败${NC}"
        exit 1
    fi

    echo -e "${GREEN}完成! 所有依赖包已成功安装${NC}"
    if [ -n "$VENV_DIR" ]; then
        echo -e "使用以下命令激活虚拟环境: ${YELLOW}source ${VENV_DIR}/bin/activate${NC}"
    fi
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--prepare)
            MODE="prepare"
            shift
            ;;
        -i|--install)
            MODE="install"
            shift
            ;;
        -r|--requirements)
            REQUIREMENTS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -d|--directory)
            PACKAGES_DIR="$2"
            shift 2
            ;;
        -v|--venv)
            VENV_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 检查pip
check_command pip

# 根据模式执行
case "$MODE" in
    "prepare")
        if [ -z "$REQUIREMENTS" ] || [ -z "$OUTPUT_DIR" ]; then
            echo -e "${RED}错误: 准备模式需要指定requirements文件和输出目录${NC}"
            show_help
            exit 1
        fi
        prepare_packages
        ;;
    "install")
        if [ -z "$REQUIREMENTS" ] || [ -z "$PACKAGES_DIR" ]; then
            echo -e "${RED}错误: 安装模式需要指定requirements文件和包目录${NC}"
            show_help
            exit 1
        fi
        install_packages
        ;;
    *)
        echo -e "${RED}错误: 必须指定模式(-p/--prepare 或 -i/--install)${NC}"
        show_help
        exit 1
        ;;
esac
