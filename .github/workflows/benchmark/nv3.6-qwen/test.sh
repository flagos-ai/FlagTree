#!/bin/bash

source ${BENCH_SCRIPT_DIR}/disable_proxy.sh

numactl --cpunodebind=0 --membind=0 \
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen36",
    "messages": [{"role": "user", "content": "请问 0.11 和 0.9 哪个大，为什么？"}]
  }'
