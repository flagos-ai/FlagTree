#!/bin/bash

script_path=$(realpath "$0")
script_dir=$(dirname "$script_path")

if [ -z "${WORKSPACE+x}" ]; then
    WORKSPACE=$(realpath "$script_dir/..")
fi

bash $script_dir/base_run.sh $WORKSPACE $@