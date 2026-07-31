#!/bin/bash

source ${BENCH_SCRIPT_DIR}/disable_proxy.sh

start=$(date +%s)

nvidia-smi -i 4,5,6,7 -lgc 1830,1830
nvidia-smi --query-gpu=index,name,clocks.gr,clocks.mem,utilization.gpu --format=csv

#numactl --cpunodebind=0 --membind=0 \
#python3 all_perf.py --input-len=32768 --output-len=1024 --concurrency=64

numactl --cpunodebind=0 --membind=0 \
python3 all_perf.py --input-len=16384 --output-len=1024 --concurrency=64

numactl --cpunodebind=0 --membind=0 \
python3 ${BENCH_SCRIPT_DIR}/all_perf.py --input-len=4096  --output-len=1024 --concurrency=64

nvidia-smi -i 4,5,6,7 -rgc
nvidia-smi --query-gpu=index,name,clocks.gr,clocks.mem,utilization.gpu --format=csv

end=$(date +%s)
duration=$((end - start))
minutes=$((duration / 60))
seconds=$((duration % 60))
echo "[INFO] Benchmark elapsed time: ${minutes}m${seconds}s."
