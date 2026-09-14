"""Sequential, rotated-order vLLM measurements with immutable raw results."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from accuracy import require_same_tokens


def require_native_sequences(scenarios, reference, label):
    expected = {scenario["prompt_length"]: scenario["runs"][0] for scenario in reference}
    for scenario in scenarios:
        native = expected[scenario["prompt_length"]]
        for run in scenario["runs"]:
            if run["prompt_tokens"] != native["prompt_tokens"]:
                raise AssertionError("Validation prompts differ")
            require_same_tokens(run["token_ids"], native["token_ids"],
                                context=f"{label}: independent sequence, prompt {scenario['prompt_length']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--artifact", action="append", default=[], metavar="PROFILE=PATH",
                        help="Use an existing verified trial artifact instead of work-dir/PROFILE/artifact")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--residual-layout", choices=("auto", "sharded", "sharded-casts"), default="auto")
    parser.add_argument("--norm-layout", choices=("auto", "replicated", "replicated-residual"), default="auto")
    parser.add_argument("--residual-kernel", choices=("direct", "staged", "async"), default="direct")
    parser.add_argument("--glu-reduction-group", type=int, choices=(32, 64, 128), default=32)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        raise ValueError("Rounds and repeats must be positive")
    tutorial = Path(__file__).resolve().parent
    profiles = ("baseline", "scheduled", "serving")
    overrides = dict(item.split("=", 1) for item in args.artifact)
    if set(overrides) - set(profiles):
        raise ValueError("Unknown artifact profile")
    artifacts = {name: Path(overrides.get(name, args.work_dir / name / "artifact")) for name in profiles}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tutorial) + os.pathsep + env.get("PYTHONPATH", "")
    if args.compile:
        for profile in profiles:
            command = [sys.executable, str(tutorial / "optimize.py"), "--checkpoint", str(args.checkpoint),
                       "--work-dir", str(args.work_dir), "--profile", profile]
            if profile != "baseline":
                command.extend(("--residual-layout", args.residual_layout, "--norm-layout", args.norm_layout,
                                "--residual-kernel", args.residual_kernel,
                                "--glu-reduction-group", str(args.glu_reduction_group)))
            subprocess.run(command, env=env, check=True)
    for artifact in artifacts.values():
        subprocess.run([sys.executable, "-m", "triton.flagmega", "artifact", "verify", str(artifact)],
                       env=env, check=True)
    args.results.mkdir(parents=True, exist_ok=True)
    labels = ("native_vllm", *profiles)
    # Correctness gates are separate from, and precede, performance. Compare
    # independent requests as well as same-history single-step native tokens.
    validation_dir = args.results / "validation"
    validation_dir.mkdir(exist_ok=False)
    reference = None
    for label in labels:
        output = validation_dir / f"{label}.json"
        command = [sys.executable, str(tutorial / "benchmark_vllm.py"), "--checkpoint", str(args.checkpoint),
                   "--label", label, "--output", str(output), "--repeats", "1", "--warmups", "0",
                   "--prompt-lengths", "32", "128", "250", "1024", "2048"]
        if label in artifacts:
            command.extend(("--artifact", str(artifacts[label])))
        with output.with_suffix(".log").open("x") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        scenarios = json.loads(output.read_text())["scenarios"]
        if reference is None:
            reference = scenarios
        require_native_sequences(scenarios, reference, label)
        print(f"Exact greedy validation passed: {label}", flush=True)
    # Separate processes: the sequence acceptance above never executes a
    # reference forward inside a FlagMega request, even in diagnostic mode.
    for label in profiles:
        output = validation_dir / f"{label}.same_history.json"
        command = [sys.executable, str(tutorial / "benchmark_vllm.py"), "--checkpoint", str(args.checkpoint),
                   "--label", label, "--output", str(output), "--repeats", "1", "--warmups", "0",
                   "--prompt-lengths", "32", "128", "250", "1024", "2048",
                   "--artifact", str(artifacts[label]), "--validate-native"]
        with output.with_suffix(".log").open("x") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Same-history diagnostics passed: {label}", flush=True)
    for round_id in range(args.rounds):
        # Never run competing variants simultaneously on the same GPU.
        order = labels[round_id % len(labels):] + labels[:round_id % len(labels)]
        for label in order:
            output = args.results / f"round{round_id:02d}.{label}.json"
            command = [sys.executable, str(tutorial / "benchmark_vllm.py"), "--checkpoint", str(args.checkpoint),
                       "--label", label, "--output", str(output), "--repeats", str(args.repeats)]
            if label in artifacts:
                command.extend(("--artifact", str(artifacts[label])))
            with output.with_suffix(".log").open("x") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            # Reject immediately, before spending further rounds on a broken
            # candidate. render_results also rechecks all raw samples later.
            require_native_sequences(json.loads(output.read_text())["scenarios"], reference, label)
            print(f"Completed round {round_id}: {label}: {output}", flush=True)
    subprocess.run([sys.executable, str(tutorial / "render_results.py"), "--results", str(args.results)],
                   env=env, check=True)


if __name__ == "__main__":
    main()
