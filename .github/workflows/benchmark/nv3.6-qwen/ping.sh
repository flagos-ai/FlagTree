#!/bin/bash

source ${BENCH_SCRIPT_DIR}/disable_proxy.sh

curl http://127.0.0.1:8000/v1/models
