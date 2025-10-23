#!/bin/bash
current_date=$(date +%Y%m%d)
pid_log="process_${current_date}.log"
max_parallel=12

fifo="/tmp/$$.fifo"
mkfifo $fifo
exec 9<>$fifo
rm -f $fifo

for ((i=0; i<$max_parallel; i++)); do
    echo >&9
done

> "$pid_log"  


if [ -d logs ]; then
  rm -rf logs
fi

mkdir logs

while IFS= read -r -d $'\0' file; do
    read -u 9  

    test_log="./logs/${file%.py}_${current_date}.log"

    {
        pytest -sv "$file" -n 16 > "$test_log" 2>&1
        echo >&9
    } &

    echo "[INFO] Activated $(basename "$file"), PID=$!, logging into $test_log."

done < <(find . -maxdepth 1 -type f -name "test_*.py" ! -name "test_common.py" -print0)

wait 
exec 9>&-

echo "[INFO] All test processes completed, pids logged into ${pid_log}"
