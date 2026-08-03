import os, sys
import json
import argparse
import queue
import time
import signal
import subprocess
import concurrent.futures
from loguru import logger
import multiprocessing
import datetime
from multiprocessing import Manager

run_log_dir = None
run_name = None


def print_fmt(message, level="INFO", flag="BENCH_CI"):
    global run_log_dir
    timestamp = datetime.datetime.now().strftime("%d %H:%M:%S")
    prefix = f"[{timestamp}][{level}][{flag}]"
    full_message = f"{prefix} {message}"
    print(full_message)
    log_file = os.path.join(run_log_dir, "ci_result_summary.log")
    with open(log_file, 'a', encoding='utf-8') as file:
        file.write(f"{full_message}\n")


def set_log_dir(dir):
    global run_log_dir
    run_log_dir = dir
    os.makedirs(run_log_dir, exist_ok=True)


def set_log_dir_by_run_name(l_run_name):
    global run_name
    run_name = l_run_name
    CASE_WORK_DIR = os.environ.get("TRITON_WORKSPACE")
    base_log_dir = os.path.join(CASE_WORK_DIR, "ci_log")
    run_log_dir = os.path.join(base_log_dir, run_name)
    set_log_dir(run_log_dir)


def set_log_dir_by_op(ops_name):
    global run_log_dir
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    l_run_name = f"bench_{ops_name}_{timestamp}"
    set_log_dir_by_run_name(l_run_name)


def read_json_ops_and_tasks(file_path, test_set_name=None):
    all_op_list = []
    test_op_list = None
    all_perf_task_dict = {}
    print_fmt(f"{file_path}", "INFO", "Bench CI")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            all_op_list = json_data.get('all_ops', [])
            if test_set_name is not None:
                test_op_list = json_data.get(test_set_name, [])
            all_perf_task_dict = json_data.get('all_perf_tasks', {})
    except FileNotFoundError:
        print_fmt(f"{file_path} not found!", "ERROR", "Bench CI")
    except json.JSONDecodeError:
        print_fmt(f"{file_path} data decode error!", "ERROR", "Bench CI")
    except Exception as e:
        print_fmt(f"{file_path} read json fail!", "ERROR", "Bench CI")
    return all_op_list, test_op_list, all_perf_task_dict


def check_card_status_i(card_id: str):
    try:
        output = subprocess.check_output(
            f"tsm_smi -i {card_id}",
            shell=True,
            text=True,
            stderr=subprocess.STDOUT
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"命令执行异常: {e}")
        return False
    if "No running processes found" in output:
        return True
    logger.warning(f"warning: 卡被占用: {card_id}")
    return False


def run_case(op_name, perf_tasks, card_id, warmup, iter_n, level, metrics, bench_dir, ops_log_dir, l_run_name):
    """
    Run all perf tasks for a single op on a single card.
    """
    set_log_dir_by_run_name(l_run_name)

    triton_cache_dir = f"/tmp/triton_log/{l_run_name}/triton_cache_" + card_id
    flaggems_cache_dir = f"/tmp/triton_log/{l_run_name}/flaggems_cache_" + card_id
    txda_skip_ops = os.getenv("TXDA_SKIP_OPS", "")

    env = os.environ.copy()
    env.update({
        'TXDA_VISIBLE_DEVICES': card_id,
        'TRITON_CACHE_DIR': triton_cache_dir,
        'FLAGGEMS_CACHE_DIR': flaggems_cache_dir,
        'TXDA_SKIP_OPS': txda_skip_ops,
    })

    os.makedirs(ops_log_dir, exist_ok=True)
    op_log_file = os.path.join(ops_log_dir, f"{op_name}_card{card_id}.log")

    print_fmt(f"[card {card_id}] {op_name} begin benchmark ({len(perf_tasks)} tasks)...", "INFO", "Bench CI")
    failed_count = 0
    succ_count = 0
    failed_task_list = []

    for task in perf_tasks:
        # Combine bench_dir + node ID into absolute path
        # e.g. /path/benchmark + test_reduction_perf.py::func -> /path/benchmark/test_reduction_perf.py::func
        node_path = os.path.join(bench_dir, task)
        cmd = [
            "python3", "-m", "pytest", node_path,
            "-v", "-s",
            f"--warmup={warmup}",
            f"--iter={iter_n}",
            f"--level={level}",
        ]
        if metrics:
            cmd.append(f"--metrics={metrics}")

        print_fmt(f"[card {card_id}] {task} start >>>>>>", "INFO", "Bench CI")
        # Write header
        with open(op_log_file, 'a') as log_f:
            log_f.write(f"=== card {card_id} op {op_name} ===\n")
            log_f.write(f"# cmd: {' '.join(cmd)}\n")

        # Launch pytest
        proc = subprocess.Popen(
            cmd,
            stdout=open(op_log_file, 'a'),
            stderr=subprocess.STDOUT,
            env=env,
        )

        # Monitor log file for timeout (same mechanism as test_flag_gems_ci.py)
        _interval = 60
        _threshold = 5
        _counter = 0
        _prev_size = os.path.getsize(op_log_file)
        while True:
            time.sleep(_interval)
            try:
                _cur_size = os.path.getsize(op_log_file)
            except FileNotFoundError:
                break
            if _cur_size == _prev_size:
                _counter += 1
                if _counter >= _threshold:
                    print_fmt(
                        f"[card {card_id}] {task} stuck "
                        f"(log unchanged {_threshold}x{_interval}s), terminating...",
                        "ERROR", "Bench CI"
                    )
                    with open(op_log_file, 'a') as _lf:
                        _lf.write(
                            f"\n[TIMEOUT] log unchanged for {_threshold}x{_interval}s, "
                            f"terminating...\n"
                        )
                    proc.send_signal(signal.SIGINT)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    break
            else:
                _counter = 0
                _prev_size = _cur_size
            if proc.poll() is not None:
                break

        result_rc = proc.wait()

        if result_rc == 0:
            print_fmt(f"[card {card_id}] {task} success!", "INFO", "Bench CI")
            succ_count += 1
        else:
            print_fmt(f"[card {card_id}] {task} failed (ret={result_rc})", "ERROR", "Bench CI")
            failed_count += 1
            failed_task_list.append((task, result_rc))

    if failed_count > 0:
        print_fmt(f"[card {card_id}] {op_name} completed, {failed_count}/{failed_count+succ_count} tasks failed.", "ERROR", "Bench CI")
        return failed_count, op_name, card_id, failed_task_list
    else:
        print_fmt(f"[card {card_id}] {op_name} completed, all {succ_count} tasks success.", "INFO", "Bench CI")
        return 0, op_name, card_id, []


def generate_bench_summary(log_dir):
    """Call gen_bench_summary.py to parse benchmark logs and write bench_summary.csv."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_script = os.path.join(script_dir, "gen_bench_summary.py")
    print_fmt(f"Generating benchmark summary from {log_dir}...", "INFO", "Bench CI")
    try:
        subprocess.run([sys.executable, summary_script, log_dir], check=True)
        print_fmt("Benchmark summary generated successfully", "INFO", "Bench CI")
    except subprocess.CalledProcessError as e:
        print_fmt(f"Failed to generate benchmark summary: {e}", "ERROR", "Bench CI")


def run_stage(args):
    global run_name
    test_set_name = args.test_set
    warmup = args.warmup
    iter_n = args.iter
    level = args.level
    metrics = args.metrics

    flaggems_root = args.flaggems_path or os.path.join(os.environ.get("TRITON_WORKSPACE"), "flaggems")
    json_file_dir = os.path.join(flaggems_root, "tests")
    json_file_path = os.path.join(json_file_dir, "flag_gems_ci_ops.json")
    all_op_list, test_op_list, all_perf_tasks = read_json_ops_and_tasks(json_file_path, test_set_name)

    if test_op_list is None:
        test_op_list = all_op_list
        print_fmt(f"{test_set_name} is None, use all_op_list", "[warn]", "Bench CI")

    # Compute card list
    NPU_IDS = [str(i) for i in range(args.device_count) if i not in args.skip_device]

    # Filter ops that have perf tasks
    pending_ops = []
    for op in test_op_list:
        if op in all_perf_tasks:
            pending_ops.append(op)
        else:
            print_fmt(f"{op} not in all_perf_tasks, discard!", "[warn]", "Bench CI")

    total_ops = len(pending_ops)
    total_tasks = sum(len(all_perf_tasks[op]) for op in pending_ops)
    print_fmt(f"test_set: {test_set_name}, ops: {total_ops}, total perf tasks: {total_tasks}", "INFO", "Bench CI")
    print_fmt(f"cards to use: {NPU_IDS}", "INFO", "Bench CI")
    print_fmt(f"warmup={warmup}, iter={iter_n}, level={level}, metrics={metrics}", "INFO", "Bench CI")

    # Setup log dirs
    bench_dir = os.path.join(flaggems_root, "benchmark")
    ops_log_dir = os.path.join(run_log_dir, "ops")
    os.makedirs(ops_log_dir, exist_ok=True)

    multiprocessing.set_start_method("spawn")
    with Manager() as manager:
        task_queue = manager.Queue()
        card_queue = manager.Queue()
        pass_queue = manager.Queue()
        fail_queue = manager.Queue()

        for op in pending_ops:
            task_queue.put(op)

        for card_id in NPU_IDS:
            if check_card_status_i(card_id):
                card_queue.put(card_id)
            else:
                print_fmt(f"Card {card_id} is not available, discard!", "[warn]", "Bench CI")

        process_count = min(card_queue.qsize(), task_queue.qsize())
        print_fmt(f"process count: {process_count}", "INFO", "Bench CI")

        def callback(future, card_id):
            try:
                result, op_name, cid, failed_task_list = future.result()
                if result == 0:
                    pass_queue.put(op_name)
                else:
                    fail_queue.put((op_name, cid, failed_task_list))
                if check_card_status_i(card_id):
                    card_queue.put(card_id)
                    print_fmt(f"Card {card_id} released and available!", "INFO", "Bench CI")
                else:
                    print_fmt(f"Card {card_id} is not available, not recycled!", "[warn]", "Bench CI")
            except Exception as e:
                print_fmt(f"Task Exception: {e}, Card {card_id} not recycled!", "ERROR", "Bench CI")

        with concurrent.futures.ProcessPoolExecutor(max_workers=process_count) as excutor:
            futures = {}
            for _ in range(process_count):
                op_name = task_queue.get()
                card_id = card_queue.get()
                future = excutor.submit(run_case, op_name, all_perf_tasks[op_name],
                                        card_id, warmup, iter_n, level, metrics,
                                        bench_dir, ops_log_dir, run_name)
                future.add_done_callback(lambda f, cid=card_id: callback(f, cid))
                futures[future] = (op_name, card_id)
                print_fmt(f"Started: {total_ops-task_queue.qsize()}/{total_ops}: {op_name} on card {card_id}", "INFO", "Bench CI")

            while not task_queue.empty():
                completed, _ = concurrent.futures.wait(
                    list(futures.keys()),
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in completed:
                    op_name, card_id = futures.pop(future)
                    print_fmt(f"Completed: {op_name} on card {card_id}", "INFO", "Bench CI")

                available = card_queue.qsize()
                to_submit = min(available, task_queue.qsize())
                for _ in range(to_submit):
                    try:
                        op_name = task_queue.get_nowait()
                        card_id = card_queue.get()
                        new_future = excutor.submit(run_case, op_name, all_perf_tasks[op_name],
                                                    card_id, warmup, iter_n, level, metrics,
                                                    bench_dir, ops_log_dir, run_name)
                        new_future.add_done_callback(lambda f, cid=card_id: callback(f, cid))
                        futures[new_future] = (op_name, card_id)
                        print_fmt(f"Started: {total_ops-task_queue.qsize()}/{total_ops}: {op_name} on card {card_id}", "INFO", "Bench CI")
                    except queue.Empty:
                        break

            completed, _ = concurrent.futures.wait(
                list(futures.keys()),
                return_when=concurrent.futures.ALL_COMPLETED
            )
            for future in completed:
                op_name, card_id = futures.pop(future)
                print_fmt(f"Completed: {op_name} on card {card_id}", "INFO", "Bench CI")

        print_fmt(f"Total ops: {total_ops}")
        succed_count = pass_queue.qsize()
        print_fmt(f"Passed ops: {succed_count}")
        for i in range(succed_count):
            print_fmt(f"\t{pass_queue.get()}")

        failed_count = fail_queue.qsize()
        print_fmt(f"Failed ops: {failed_count}")
        all_failed = 0
        for i in range(failed_count):
            op_name, cid, failed_task_list = fail_queue.get()
            print_fmt(f"[card {cid}] {op_name}:")
            for (tfunc, retcode) in failed_task_list:
                print_fmt(f"\t{tfunc}: {retcode}")
            all_failed += 1

        print_fmt("All benchmark tasks processed")

        # Generate summary CSV from collected logs
        generate_bench_summary(run_log_dir)

        if all_failed == 0:
            return 0
        else:
            return -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlagGems OP benchmark test for Triton CI.")
    parser.add_argument("--test_set", type=str, default="ci_ops",
                        help="op set name, defined in flag_gems_ci_ops.json")
    parser.add_argument("--device_count", type=int, default=1,
                        help="Maximum number of devices that can be used.")
    parser.add_argument("--skip_device", type=int, nargs='*', default=[],
                        help="Devices that need to be skipped.")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="Number of warmup runs before benchmark.")
    parser.add_argument("--iter", type=int, default=1000,
                        help="Number of reps for each benchmark run.")
    parser.add_argument("--level", type=str, default="core",
                        choices=["core", "comprehensive"],
                        help="Benchmark level: core or comprehensive.")
    parser.add_argument("--metrics", type=str, default="latency",
                        help="Benchmark metrics, e.g. latency, tflops.")
    parser.add_argument("--flaggems_path", type=str, default="",
                        help="Flaggems root path. Defaults to $TRITON_WORKSPACE/flaggems.")
    args = parser.parse_args()

    set_log_dir_by_op(args.test_set)

    print_fmt("------------------all env---------------------", "INFO", "Bench CI")
    for key, value in os.environ.items():
        print_fmt(f"{key}={value}")
    print_fmt("----------------------------------------------", "INFO", "Bench CI")

    start_time = time.time()
    exit_code = run_stage(args)
    end_time = time.time()
    print_fmt(f"time cost: {(end_time - start_time):.2f}s", "INFO", "Bench CI")

    if exit_code is not None:
        sys.exit(0 if exit_code == 0 else -1)
    else:
        sys.exit(-1)
