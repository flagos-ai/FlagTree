#!/bin/bash

set -e

# Print
if [ "$OS" = "Windows_NT" ]; then
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
else
    RED='\033[1;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m'
fi
INFO="${GREEN}[INFO]${NC}"
NOTE="${YELLOW}[NOTE]${NC}"
ERROR="${RED}[ERROR]${NC}"

printfln() {
    printf "%b
" "$@"
}

# Input
if [ $# -ge 1 ] && [ -f "$1" ]; then
    input_tar_gz="$1"
    printfln "${INFO} Use ${input_tar_gz} as input packed .tar.gz file"
else
    printfln "${ERROR} No input .tar.gz file specified"
    printfln "${INFO} Usage: sh $0 [input_tar_gz]"
    exit 1
fi

# Check system architecture
file_arch="${input_tar_gz##*-}"      # x64.tar.gz
file_arch="${file_arch%%.tar.gz}"    # x64
sys_arch="$(uname -m)"
case "${sys_arch}" in
    x86_64|amd64)  sys_arch="x64" ;;
    aarch64|arm64) sys_arch="aarch64" ;;
esac
if [ "${file_arch}" != "${sys_arch}" ]; then
    printfln "${ERROR} Arch mismatch: input_file=${RED}${file_arch}${NC}, system=${RED}${sys_arch}${NC}"
    exit 1
fi

# Output
if [ "${TRITON_HOME}" != "" ]; then
    output_dir="${TRITON_HOME}"
    printfln "${INFO} Use ${output_dir} as output directory because TRITON_HOME is set"
else
    output_dir="${HOME}/.triton"
    printfln "${INFO} Use ${output_dir} as default output directory"
fi

if [ -d "${output_dir}" ]; then
    last_output_dir=${output_dir}.$(date +%Y%m%d_%H%M%S)
    if [ -d "${last_output_dir}" ]; then
        printfln "${ERROR} Backup directory ${RED}${last_output_dir}${NC} already exists, retrying will resolve it"
        exit 1
    fi
    printfln "${NOTE} Output directory ${YELLOW}${output_dir}${NC} already exists, will mv to ${YELLOW}${last_output_dir}${NC}"
fi

# Check unpack dirs
printfln "${NOTE} Will unpack following dirs to ${YELLOW}${output_dir}${NC} (will be created):"
tar tzf "${input_tar_gz}" \
    | awk -F'/' '{
        sub(/^\.triton\/?/, ""); if ($0 == "") next
        if ($1 == "nvidia") {
            if (NF >= 2 && $2 != "") {
                print "nvidia/"$2"/"
            } else {
                print "nvidia/"
            }
        } else {
            print $1"/"
        }
    }' \
    | uniq
printfln "${NOTE} Press any key to confirm and continue, or Ctrl+C to cancel ..."
read dummy

# Create output dir
if [ -d "${output_dir}" ]; then
    set -x
    mv "${output_dir}" "${last_output_dir}"
    { set +x; } 2>/dev/null
fi
set -x
mkdir -p "${output_dir}"
{ set +x; } 2>/dev/null
printfln ""

# Unpack
printfln "${NOTE} Unpacking ${YELLOW}${input_tar_gz}${NC} into ${YELLOW}${output_dir}${NC}"
set -x
tar zxf "${input_tar_gz}" -C "${output_dir}" --strip-components=1
{ set +x; } 2>/dev/null

printfln "${INFO} Finished successfully."
