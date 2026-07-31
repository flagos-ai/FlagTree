#!/bin/bash

export no_proxy="127.0.0.1,localhost,::1"

numactl --cpunodebind=0 --membind=0 \
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen36",
    "messages": [{"role": "user", "content": "请问 0.11 和 0.9 哪个大，为什么？"}]
  }'
