# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
vLLM 性能测试脚本

功能：
- 支持用户自定义 input_len、output_len、并发数
- 返回 vllm bench serve 的所有指标
"""

import subprocess
import sys
import re
import argparse
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# =============================================================================
# 服务配置（按需修改）
# =============================================================================
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
MODEL_NAME = "qwen36"
TOKENIZER_PATH = "/data/zhengyang/flagrelease/Qwen3.6-35B-A3B-nomtp"

# =============================================================================
# 默认测试参数
# =============================================================================
DEFAULT_INPUT_LEN = 4096
DEFAULT_OUTPUT_LEN = 1024
DEFAULT_CONCURRENCY = 64
WARMUP_ROUNDS = 4
TOTAL_ROUNDS = 10
OUTPUT_DIR = Path("./unit_test_output")

# 输出解析正则表达式
METRIC_PATTERNS = {
    "Successful requests": r"Successful requests:\s+(\d+)",
    #"Failed requests": r"Failed requests:\s+(\d+)",
    #"Benchmark duration (s)": r"Benchmark duration \(s\):\s+([\d.]+)",
    #"Total input tokens": r"Total input tokens:\s+(\d+)",
    #"Total generated tokens": r"Total generated tokens:\s+(\d+)",
    #"Request throughput (req/s)": r"Request throughput \(req/s\):\s+([\d.]+)",
    "Output token throughput (tok/s)": r"Output token throughput \(tok/s\):\s+([\d.]+)",
    "Total token throughput (tok/s)": r"Total token throughput \(tok/s\):\s+([\d.]+)",
    #"Mean TTFT (ms)": r"Mean TTFT \(ms\):\s+([\d.]+)",
    "Median TTFT (ms)": r"Median TTFT \(ms\):\s+([\d.]+)",
    #"P99 TTFT (ms)": r"P99 TTFT \(ms\):\s+([\d.]+)",
    #"Mean TPOT (ms)": r"Mean TPOT \(ms\):\s+([\d.]+)",
    "Median TPOT (ms)": r"Median TPOT \(ms\):\s+([\d.]+)",
    #"P99 TPOT (ms)": r"P99 TPOT \(ms\):\s+([\d.]+)",
    #"Mean ITL (ms)": r"Mean ITL \(ms\):\s+([\d.]+)",
    "Median ITL (ms)": r"Median ITL \(ms\):\s+([\d.]+)",
    #"P99 ITL (ms)": r"P99 ITL \(ms\):\s+([\d.]+)",
}


def parse_output(output):
    """解析 vllm bench serve 的输出文本，提取所有指标"""
    metrics = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, output)
        if match:
            val = match.group(1)
            metrics[key] = float(val) if "." in val else int(val)
        else:
            metrics[key] = None
    return metrics


def build_command(input_len, output_len, concurrency):
    """构建 vllm bench serve 命令"""
    return [
        "vllm",
        "bench",
        "serve",
        "--host",
        SERVER_HOST,
        "--port",
        str(SERVER_PORT),
        "--model",
        MODEL_NAME,
        "--tokenizer",
        TOKENIZER_PATH,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--endpoint",
        "/v1/completions",
        "--ignore-eos",
        "--trust-remote-code",
        "--num-prompts",
        str(concurrency),
        "--max-concurrency",
        str(concurrency),
        "--seed",
        "0",
    ]


def run_single_test(round_num, input_len, output_len, concurrency):
    """执行单轮测试并返回所有解析后的指标"""
    #print(f"\n--- 第 {round_num}/{TOTAL_ROUNDS} 轮 ---")
    cmd = build_command(input_len, output_len, concurrency)
    #print(f"  执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metrics = parse_output(result.stdout)
        #print(f"  第 {round_num} 轮完成")
        #for key, val in metrics.items():
        #    if val is not None:
        #        print(f"    {key}: {val}")
        return metrics
    except subprocess.CalledProcessError as e:
        print(f"  错误: 第 {round_num} 轮执行失败")
        print(f"  错误输出: {e.stderr[:200]}")
        return None
    except FileNotFoundError:
        print("  错误: vllm 命令未找到，请确认 vllm 已安装且在 PATH 中")
        return None
    except Exception as e:
        print(f"  错误: 第 {round_num} 轮发生未知错误: {e}")
        return None


def compute_average(all_round_metrics):
    """对后面轮的数值型指标取平均"""
    last = [m for m in all_round_metrics[WARMUP_ROUNDS:] if m is not None]
    if not last:
        return None

    avg = {}
    for key in METRIC_PATTERNS:
        values = [m[key] for m in last if m.get(key) is not None]
        if values:
            avg[key] = sum(values) / len(values)
        else:
            avg[key] = None
    return avg


def print_and_save_results(all_round_metrics, avg_metrics, input_len, output_len, concurrency):
    """打印每轮完整结果 + 后4轮平均值，并保存到文件"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"perf_results_in{input_len}_out{output_len}_c{concurrency}_{timestamp}.txt"
    filepath = OUTPUT_DIR / filename

    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    #out("=" * 70)
    #out("vLLM 性能测试结果")
    #out(f"  服务: {SERVER_HOST}:{SERVER_PORT}")
    #out(f"  模型: {MODEL_NAME}")
    #out(f"  输入长度: {input_len}, 输出长度: {output_len}, 并发数: {concurrency}")
    #out(f"  总轮数: {TOTAL_ROUNDS} (前{WARMUP_ROUNDS}轮为warmup，此后取平均)")
    #out(f"  执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    #out("=" * 70)

    for i, metrics in enumerate(all_round_metrics):
        #round_num = i + 1
        #label = "（warmup，不计入平均）" if round_num <= WARMUP_ROUNDS else ""
        #out(f"\n--- 第 {round_num} 轮 {label} ---")
        if metrics is None:
            out("  [FAILED]")
            continue
        #for key, val in metrics.items():
        #    if val is not None:
        #        out(f"  {key}: {val}")

    out("\n" + "=" * 70)
    out("结果平均值")
    out("=" * 70)
    if avg_metrics is None:
        out("  无有效数据，无法计算平均值")
    else:
        for key, val in avg_metrics.items():
            if val is not None:
                out(f"  {key}: {val:.2f}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n结果已保存至: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="vLLM 性能测试脚本")
    parser.add_argument("--input-len", type=int, default=None, help=f"输入长度 (默认: {DEFAULT_INPUT_LEN})")
    parser.add_argument("--output-len", type=int, default=None, help=f"输出长度 (默认: {DEFAULT_OUTPUT_LEN})")
    parser.add_argument("--concurrency", type=int, default=None, help=f"并发数 (默认: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不执行")
    args = parser.parse_args()

    input_len = args.input_len if args.input_len is not None else DEFAULT_INPUT_LEN
    output_len = args.output_len if args.output_len is not None else DEFAULT_OUTPUT_LEN
    concurrency = args.concurrency if args.concurrency is not None else DEFAULT_CONCURRENCY

    print("=" * 70)
    print("vLLM 性能测试")
    #print(f"  服务: {SERVER_HOST}:{SERVER_PORT}")
    #print(f"  模型: {MODEL_NAME}")
    print(f"  输入长度: {input_len}, 输出长度: {output_len}, 并发数: {concurrency}")
    print(f"  总轮数: {TOTAL_ROUNDS} (前{WARMUP_ROUNDS}轮warmup，此后取平均)")
    #print("=" * 70)

    if args.dry_run:
        print("[DRY RUN MODE]")
        cmd = build_command(input_len, output_len, concurrency)
        print(f"  命令: {' '.join(cmd)}")
        print(f"  将执行 {TOTAL_ROUNDS} 轮")
        print("\n[DRY RUN] 脚本验证完成，未实际执行测试。")
        return

    all_round_metrics = []
    for round_num in range(1, TOTAL_ROUNDS + 1):
        metrics = run_single_test(round_num, input_len, output_len, concurrency)
        all_round_metrics.append(metrics)

    avg_metrics = compute_average(all_round_metrics)
    print_and_save_results(all_round_metrics, avg_metrics, input_len, output_len, concurrency)


if __name__ == "__main__":
    main()
