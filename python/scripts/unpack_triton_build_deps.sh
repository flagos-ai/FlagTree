#!/bin/bash

# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set -e

if [ "$OS" = "Windows_NT" ]; then
    RED=''; GREEN=''; YELLOW=''; NC=''
else
    RED='\033[1;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
fi
INFO="${GREEN}[INFO]${NC}"
NOTE="${YELLOW}[NOTE]${NC}"
ERROR="${RED}[ERROR]${NC}"

printfln() {
    printf "%b\n" "$@"
}

if [ $# -ge 1 ] && [ -f "$1" ]; then
    input_tar_gz="$1"
    printfln "${INFO} Use ${input_tar_gz} as input packed .tar.gz file"
else
    printfln "${ERROR} No input .tar.gz file specified"
    printfln "${INFO} Usage: sh $0 [input_tar_gz]"
    exit 1
fi

file_arch="${input_tar_gz##*-}"
file_arch="${file_arch%%.tar.gz}"
sys_arch="$(uname -m)"
case "${sys_arch}" in
    x86_64|amd64) sys_arch="x64" ;;
    aarch64|arm64) sys_arch="aarch64" ;;
esac
if [ "${file_arch}" != "${sys_arch}" ]; then
    printfln "${ERROR} Arch mismatch: input_file=${RED}${file_arch}${NC}, system=${RED}${sys_arch}${NC}"
    exit 1
fi

if [ -n "${TRITON_HOME}" ]; then
    output_dir="${TRITON_HOME}"
else
    output_dir="${HOME}/.triton"
fi

if [ -d "${output_dir}" ]; then
    last_output_dir="${output_dir}.$(date +%Y%m%d_%H%M%S)"
    if [ -d "${last_output_dir}" ]; then
        printfln "${ERROR} Backup directory ${last_output_dir} already exists"
        exit 1
    fi
fi

printfln "${NOTE} Will unpack following dirs to ${YELLOW}${output_dir}${NC}:"
tar tzf "${input_tar_gz}" |
    awk -F'/' '{
        sub(/^\.triton\/?/, ""); if ($0 == "") next
        if ($1 == "nvidia" && NF >= 2 && $2 != "") print "nvidia/"$2"/"
        else print $1"/"
    }' |
    uniq
printfln "${NOTE} Press any key to confirm and continue, or Ctrl+C to cancel ..."
read -r dummy

if [ -d "${output_dir}" ]; then
    mv "${output_dir}" "${last_output_dir}"
fi
mkdir -p "${output_dir}"
tar zxf "${input_tar_gz}" -C "${output_dir}" --strip-components=1
printfln "${INFO} Finished successfully."
