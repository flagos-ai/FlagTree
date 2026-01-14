#!/bin/bash

PROXY=http://192.168.100.225:8889
setup_proxy() {
    # Downloading python requirement is needed.
    export https_proxy=$PROXY http_proxy=$PROXY all_proxy=$PROXY
    export HTTPS_PROXY=$PROXY HTTP_PROXY=$PROXY ALL_PROXY=$PROXY
}

script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")
project_dir=$(realpath "$script_dir/../../..")

use_venv=ON
if [ $# -gt 0 ]; then
    if [[ "${1,,}" == "no_venv" ]]; then
        use_venv=OFF
    fi
fi

if [ "x$use_venv" == "xON" ]; then
    python3 -m venv $project_dir/.venv --prompt triton
    source $project_dir/.venv/bin/activate
fi

setup_proxy

apt install git
apt install lld
apt install ccache

# pip uninstall triton

triton_origin_req=$project_dir/python/requirements.txt
if [ ! -f $triton_origin_req ]; then
    triton_origin_req=$script_dir/requirements.txt
fi

if [ ! -f $triton_origin_req ]; then
    echo "error can't find:$triton_origin_req"
    exit
fi
pip3 install -r $triton_origin_req
pip3 install -r $script_dir/requirements_ts.txt
