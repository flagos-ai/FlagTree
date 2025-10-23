#!/bin/bash

# 获取传入参数
param="$1"
input_ops="$2"
device_count="${3:-1}"           # 默认使用1个设备
threads_per_device="${4:-64}"    # 每个设备线程数，默认64

# 定义路径
DIR_TESTS="tests"
DIR_BENCHMARK="benchmark"
DAILY_LOG_DIR="/home/daily_log"
TIMESTAMP=$(date +"%Y%m%d")
LOG_ARCHIVE="test_flaggems_logs_${TIMESTAMP}.tar.gz"
SUMMARY_FILE="${WORKSPACE}/ascend/examples/summary.txt"  # 新增：统计信息文件

# 检查日志目录
mkdir -p "$DAILY_LOG_DIR" || { echo "无法创建日志目录 $DAILY_LOG_DIR"; exit 1; }

# 中央计数器文件定义
COUNTER_FILE=$(mktemp)
LOCK_FILE="/tmp/op_test_run.lock"
touch $LOCK_FILE

# ===== 修改：改进的统计结果收集机制 =====
# 使用文件存储统计结果
STATS_DIR=$(mktemp -d)
# 初始化设备统计文件
for ((device_id=0; device_id < device_count; device_id++)); do
    stats_file="${STATS_DIR}/device_${device_id}.stats"
    echo "success=0" > "$stats_file"
    echo "failure=0" >> "$stats_file"
    echo "skipped=0" >> "$stats_file"
    echo "error=0" >> "$stats_file"
done

# 原子更新统计
record_stats() {
    local device_id=$1
    local status=$2  # success/failure/skipped/error
    local stats_file="${STATS_DIR}/device_${device_id}.stats"

    (
        flock -x 20
        # 读取当前值
        current=$(grep "^${status}=" "$stats_file" | cut -d= -f2)
        # 更新值
        new_value=$((current + 1))
        # 替换文件中的值
        sed -i "s/^${status}=.*/${status}=${new_value}/" "$stats_file"
    ) 20>"${stats_file}.lock"
}

# 任务队列管理函数
init_task_queue() {
    local -n arr_ref=$1
    TASK_FILE=$(mktemp)
    printf "%s\n" "${arr_ref[@]}" > "$TASK_FILE"
    echo 0 > "$TASK_FILE.counter"
    echo "${#arr_ref[@]}" > "$COUNTER_FILE.total"
    echo 0 > "$COUNTER_FILE.completed"
}

get_next_task() {
    (
        # 文件锁保证原子操作
        flock -x 9
        counter=$(< $TASK_FILE.counter)
        total_tasks=$(wc -l < $TASK_FILE)

        if (( counter >= total_tasks )); then
            echo ""
            return
        fi

        task_name=$(sed -n "$((counter+1))p" $TASK_FILE)
        echo $((counter+1)) > "$TASK_FILE.counter"
        echo "$task_name"
    ) 9> "$TASK_FILE.lock"
}

# 原子更新完成计数器
update_progress() {
    (
        flock -x 11
        local current=$(< $COUNTER_FILE.completed)
        echo $((current + 1)) > $COUNTER_FILE.completed
        echo $((current + 1))
    ) 11> $LOCK_FILE
}

# 获取进度信息
get_progress() {
    (
        flock -s 11   # 共享锁（只读）
        completed=$(< $COUNTER_FILE.completed)
        total=$(< $COUNTER_FILE.total)
        echo "$completed $total"
    ) 11> $LOCK_FILE
}

cleanup_tasks() {
    rm -f "$TASK_FILE" "$TASK_FILE.counter" "$TASK_FILE.lock" $LOCK_FILE $COUNTER_FILE*
}

# 算子列表定义
OPS=("abs" "add" "addmm" "all" "amax" "argmax" "bitwise_and" "bitwise_not" "bitwise_or" "bmm" \
"cos" "CrossEntryLoss" "div" "dropout" "eq" "exp" "fill" "ge" "gelu" "group_norm" "gt" "isinf" \
"isnan" "rsub" "le" "linear" "log_softmax" "lt" "max" "mean" "min" "mm" "mul" "mv" \
"native_dropout" "ne" "neg" "pow" "prod" "reciprocal" "relu" "rsqrt" "sigmoid" "silu" \
"sin" "softmax" "sub" "sum" "tanh" "triu")

total_ops=${#OPS[@]}
echo "======================================"
echo "测试算子列表: ${OPS[@]}"
echo "算子总数: $total_ops"
echo "使用设备数量: $device_count"
echo "每设备线程数: $threads_per_device"
echo "======================================"

# 初始化性能计数器 - 修复开始时间显示问题
start_time=$(date +%s)  # 使用Unix时间戳

# 线程执行函数 - 正确性测试
run_tests_thread() {
    local device_id=$1
    local thread_id=$2
    local device_log_dir=$3
    local thread_log_dir="$device_log_dir/thread_${thread_id}"
    mkdir -p "$thread_log_dir"

    while true; do
        task_name=$(get_next_task)
        [[ -z "$task_name" ]] && break

        echo "[设备 $device_id-线程 $thread_id] 正在执行: pytest -m $task_name --ref cpu -sv"
        log_file="${thread_log_dir}/result_${task_name}.log"

        # 执行正确性测试并记录时间
        start_op=$(date +%s)
        pytest -m $task_name --dist=loadfile --ref cpu -sv &> "$log_file"
        exit_code=$?
        duration=$(( $(date +%s) - start_op ))

        # 根据退出码记录不同状态
        case $exit_code in
            0)
                status="success"
                ;;
            1)
                status="failure"
                ;;
            2)  # pytest跳过用例的退出码
                status="skipped"
                ;;
            *)
                status="error"
                ;;
        esac

        # 记录统计结果
        record_stats $device_id $status

        # 原子更新完成计数
        new_completed=$(update_progress)

        # 获取最新进度状态
        read completed total < <(get_progress)
        progress=$(( completed * 100 / total ))

        # 输出结果
        if [ $exit_code -ne 0 ]; then
            echo "[错误] [$device_id-$thread_id] $task_name 失败! (用时 ${duration}s, 进度: $completed/$total)"
        else
            echo "[成功] [$device_id-$thread_id] $task_name 完成! (用时 ${duration}s, 进度: $completed/$total)"
        fi
    done
}

# 线程执行函数 - 性能测试
run_benchmark_thread() {
    local device_id=$1
    local thread_id=$2
    local device_log_dir=$3
    local thread_log_dir="$device_log_dir/thread_${thread_id}"
    mkdir -p "$thread_log_dir"

    while true; do
        task_name=$(get_next_task)
        [[ -z "$task_name" ]] && break

        echo "[设备 $device_id-线程 $thread_id] 正在执行: pytest -m $task_name --level core --record log"
        log_file="${thread_log_dir}/benchmark_${task_name}.log"
        perf_file="${thread_log_dir}/perf_${task_name}.log"

        # 执行性能测试并记录时间
        start_op=$(date +%s)
        pytest -m $task_name --level core --record "$perf_file" &> "$log_file"
        exit_code=$?
        duration=$(( $(date +%s) - start_op ))

        # 根据退出码记录不同状态
        case $exit_code in
            0)
                status="success"
                ;;
            1)
                status="failure"
                ;;
            2)  # pytest跳过用例的退出码
                status="skipped"
                ;;
            *)
                status="error"
                ;;
        esac

        # 记录统计结果
        record_stats $device_id $status

        # 原子更新完成计数
        new_completed=$(update_progress)

        # 获取最新进度状态
        read completed total < <(get_progress)
        progress=$(( completed * 100 / total ))

        # 输出结果
        if [ $exit_code -ne 0 ]; then
            echo "[错误] [$device_id-$thread_id] $task_name 性能测试失败! (用时 ${duration}s, 进度: $completed/$total)"
        else
            echo "[成功] [$device_id-$thread_id] $task_name 性能测试完成! (用时 ${duration}s, 进度: $completed/$total)"
        fi
    done
}

# 设备主函数
run_device() {
    local device_id=$1
    local mode=$2
    local device_log_dir="device_${device_id}_logs"
    mkdir -p "$device_log_dir"

    # 创建设备内的线程池
    for ((thread_id=0; thread_id < threads_per_device; thread_id++)); do
        if [ "$mode" == "tests" ]; then
            run_tests_thread $device_id $thread_id "$device_log_dir" &
        elif [ "$mode" == "benchmark" ]; then
            run_benchmark_thread $device_id $thread_id "$device_log_dir" &
        fi
    done

    # 等待设备内所有线程完成
    wait
    echo "======== 设备 $device_id 上所有任务完成 ========"
}

# 根据参数执行测试
if [ "$param" == "tests" ]; then
    cd "$DIR_TESTS" || { echo "无法进入目录 $DIR_TESTS"; exit 1; }

    # 创建全局任务队列
    init_task_queue OPS

    # 启动设备主进程
    for ((device_id=0; device_id < device_count; device_id++)); do
        (
            export ASCEND_RT_VISIBLE_DEVICES=$device_id
            run_device $device_id "tests"
        ) &
    done

elif [ "$param" == "benchmark" ]; then
    cd "$DIR_BENCHMARK" || { echo "无法进入目录 $DIR_BENCHMARK"; exit 1; }

    # 性能测试使用单线程模式（保证准确性）
    if [ "$threads_per_device" -gt 1 ]; then
        echo "警告：性能测试模式下自动设置为单线程模式（每个设备1个线程）"
        threads_per_device=1
    fi

    # 创建全局任务队列
    init_task_queue OPS

    # 启动设备主进程
    for ((device_id=0; device_id < device_count; device_id++)); do
        (
            export ASCEND_RT_VISIBLE_DEVICES=$device_id
            run_device $device_id "benchmark"
        ) &
    done

else
    echo "参数错误! 用法:"
    echo "正确性测试: $0 tests \"算子列表\" [设备数量] [线程数]"
    echo "性能测试:   $0 benchmark \"算子列表\" [设备数量] [线程数]"
    cleanup_tasks
    exit 1
fi

# 等待所有设备完成
wait
cleanup_tasks

# ===== 修改：改进的统计信息汇总 =====
total_success=0
total_failure=0
total_skipped=0
total_error=0

# 按设备汇总结果
for ((device_id=0; device_id < device_count; device_id++)); do
    stats_file="${STATS_DIR}/device_${device_id}.stats"

    if [ -f "$stats_file" ]; then
        # 从文件加载统计
        d_success=$(grep '^success=' "$stats_file" | cut -d= -f2)
        d_failure=$(grep '^failure=' "$stats_file" | cut -d= -f2)
        d_skipped=$(grep '^skipped=' "$stats_file" | cut -d= -f2)
        d_error=$(grep '^error=' "$stats_file" | cut -d= -f2)

        total_success=$((total_success + d_success))
        total_failure=$((total_failure + d_failure))
        total_skipped=$((total_skipped + d_skipped))
        total_error=$((total_error + d_error))

        # 记录设备统计
        echo "设备 $device_id 完成情况: $d_success 成功, $d_failure 失败, $d_skipped 跳过, $d_error 错误"
    else
        echo "警告: 设备 $device_id 的统计文件未找到"
    fi
done

# 清理统计目录
rm -rf "$STATS_DIR"

# 计算总耗时
total_time=$(( $(date +%s) - start_time ))  # 使用绝对时间计算总耗时
hours=$(( total_time / 3600 ))
minutes=$(( (total_time % 3600) / 60 ))
seconds=$(( total_time % 60 ))
time_str=$(printf "%02dh %02dm %02ds" $hours $minutes $seconds)

# 计算平均耗时
if [[ $total_ops -gt 0 ]]; then
    completed_ops=$((total_success + total_failure + total_error))
    if [[ $completed_ops -gt 0 ]]; then
        avg_time=$((total_time / completed_ops))
        avg_min=$((avg_time / 60))
        avg_sec=$((avg_time % 60))
        avg_str=$(printf "%02dm %02ds" $avg_min $avg_sec)
    else
        avg_str="N/A"
    fi
else
    avg_str="N/A"
fi

# 生成统计信息摘要
{
    echo "===================== flaggems测试统计摘要 ====================="
    echo "执行类型:       ${param^}"
    echo "开始时间:       $(date -d @$start_time '+%Y-%m-%d %H:%M:%S')"
    echo "结束时间:       $(date '+%Y-%m-%d %H:%M:%S')"
    echo "测试日期:       $(date '+%Y-%m-%d')"
    echo "总耗时:         $time_str"
    echo "--------------------------------------------------------"
    echo "总算子数:       $total_ops"
    echo "成功用例数:     $total_success"
    echo "失败用例数:     $total_failure"
    echo "跳过用例数:     $total_skipped"
    echo "错误用例数:     $total_error"
    echo "完成用例数:     $((total_success + total_failure + total_error))"

    if [[ $total_ops -gt 0 ]]; then
        echo "完成率:         $(( (total_success + total_failure + total_error) * 100 / total_ops ))%"
    else
        echo "完成率:         N/A"
    fi

    if [[ $total_success -gt 0 ]] || [[ $total_failure -gt 0 ]] || [[ $total_error -gt 0 ]]; then
        success_rate=$(( total_success * 100 / (total_success + total_failure + total_error) ))
        echo "成功率:         ${success_rate}%"
    else
        echo "成功率:         N/A"
    fi

    echo "平均耗时/算子:   $avg_str"
    echo "--------------------------------------------------------"
    echo "设备数量:       $device_count"
    echo "每设备线程数:   $threads_per_device"
    echo "并行效率:       $(( (total_success + total_failure + total_error) * 100 / (device_count * threads_per_device * total_time) )) OPS/线程秒"
    echo "========================================================"
    echo ""
} | tee -a $SUMMARY_FILE  # 追加到统计文件并同时输出到控制台

# 归档所有日志文件
log_dirs=($(find . -maxdepth 1 -type d -name "device_*_logs" 2>/dev/null))
if [ ${#log_dirs[@]} -gt 0 ]; then
    echo "归档日志文件到 $LOG_ARCHIVE"
    tar -czf "$LOG_ARCHIVE" "${log_dirs[@]}"

    if mv "$LOG_ARCHIVE" "$DAILY_LOG_DIR"; then
        echo "日志已保存到: $DAILY_LOG_DIR/$LOG_ARCHIVE"
    else
        echo "警告：日志移动到 $DAILY_LOG_DIR 失败"
    fi

    # 清理临时日志
    rm -rf "${log_dirs[@]}"
else
    echo "警告：未找到任何日志目录，跳过归档"
fi

echo "所有算子测试执行完成!"
echo "详细统计信息已追加到: $SUMMARY_FILE"
exit 0