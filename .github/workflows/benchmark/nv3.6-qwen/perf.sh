#!/bin/bash

start=$(date +%s)

export no_proxy="127.0.0.1,localhost"

nvidia-smi -i 4,5,6,7 -lgc 1830,1830
nvidia-smi --query-gpu=index,name,clocks.gr,clocks.mem,utilization.gpu --format=csv

#numactl --cpunodebind=0 --membind=0 \
#python3 all_perf.py --input-len=32768 --output-len=1024 --concurrency=64

numactl --cpunodebind=0 --membind=0 \
python3 all_perf.py --input-len=16384 --output-len=1024 --concurrency=64

numactl --cpunodebind=0 --membind=0 \
python3 all_perf.py --input-len=4096  --output-len=1024 --concurrency=64

nvidia-smi -i 4,5,6,7 -rgc
nvidia-smi --query-gpu=index,name,clocks.gr,clocks.mem,utilization.gpu --format=csv

end=$(date +%s)
duration=$((end - start))
minutes=$((duration / 60))
seconds=$((duration % 60))
echo "性能测试共耗时: ${minutes}分${seconds}秒"
