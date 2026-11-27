"""End-to-end: launcher device timing must not break TXDA graph capture.

The launcher's timing path records an end event and waits for the kernel, which
is illegal inside ``txStreamBeginCapture``/``txStreamEndCapture`` and used to
invalidate the capture (``Stream capture was invalidated by an earlier error``).

Timing is a runtime decision inside the launcher, so the same cached ``.so``
serves every case; each case still runs in its own subprocess to get a clean
environment and interpreter. Device execution is opt-in via
``TX81_RUN_DEVICE_TESTS=1``.

Matrix: eager / graph (warm) / graph (cold, first launch inside capture) x
{no env, TILELANG_TX_EVENT_TIMING (TX event path), TSM_PROFILER_EN
(libtsm_profiler path, exercised with a stub plugin), profiler+timing,
torch.profiler}.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import traceback

import pytest

RUN_DEVICE = os.getenv("TX81_RUN_DEVICE_TESTS", "1").lower() in ("1", "true", "yes")

TIMING_ENVS = ("TILELANG_TX_EVENT_TIMING", "TSM_PROFILER_EN")

# Placeholder replaced by a small locally-built libtsm_profiler.so.
STUB_PLACEHOLDER = "@PROFILER_STUB@"
STUB_KCORE_NS = 4242
STUB_AP_NS = 77

# id, mode, env overrides. mode: "eager" | "graph" | "graph_cold" | "graph_profile".
CASES = [
    ("eager_no_env", "eager", {}),
    ("eager_timing", "eager", {"TILELANG_TX_EVENT_TIMING": "1"}),
    ("eager_profiler_only", "eager", {"TSM_PROFILER_EN": "1"}),
    ("eager_profiler_stub", "eager", {"TSM_PROFILER_EN": "1", "LD_PRELOAD": STUB_PLACEHOLDER}),
    ("graph_no_env", "graph", {}),
    ("graph_timing", "graph", {"TILELANG_TX_EVENT_TIMING": "1"}),
    ("graph_profiler_only", "graph", {"TSM_PROFILER_EN": "1"}),
    ("graph_profiler_stub", "graph", {"TSM_PROFILER_EN": "1", "LD_PRELOAD": STUB_PLACEHOLDER}),
    ("graph_profiler_plus_timing", "graph", {"TSM_PROFILER_EN": "1", "TILELANG_TX_EVENT_TIMING": "1"}),
    ("graph_cold_timing", "graph_cold", {"TILELANG_TX_EVENT_TIMING": "1"}),
    ("graph_torch_profiler", "graph_profile", {"TSM_PROFILER_EN": "1"}),
]

RESULT_PREFIX = "RESULT_JSON:"

_STUB_PATH = None


def _profiler_stub() -> str:
    """Build a minimal libtsm_profiler.so exporting tsm_prof_begin/get_timing."""
    global _STUB_PATH
    if _STUB_PATH is not None:
        return _STUB_PATH

    cc = os.getenv("CC") or shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler available to build the profiler stub")

    include_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "include")
    stub_dir = tempfile.mkdtemp(prefix="tsm_prof_stub_")
    src = os.path.join(stub_dir, "stub.c")
    so = os.path.join(stub_dir, "libtsm_profiler_stub.so")
    with open(src, "w") as f:
        f.write(
            textwrap.dedent(f"""
            #include <string.h>
            #include "tsm_prof_query.h"

            uint64_t tsm_prof_begin(void) {{ return 0x1234u; }}

            int tsm_prof_get_timing(uint64_t token, tsm_prof_timing_t* out, uint32_t timeout_ms) {{
                if (out == 0) return -1;
                memset(out, 0, sizeof(*out));
                out->token = token;
                out->kcore_duration_nsec = {STUB_KCORE_NS};
                out->ap_duration_nsec = {STUB_AP_NS};
                out->ready = 1;
                return 0;
            }}
            """).lstrip())
    proc = subprocess.run(
        [cc, "-shared", "-fPIC", "-O2", "-I", include_dir, src, "-o", so],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"failed to build profiler stub: {proc.stderr}")
    _STUB_PATH = so
    return so


def _device_available() -> bool:
    if not RUN_DEVICE:
        return False
    try:
        import torch
        import torch_txda  # noqa: F401 - registers torch.txda

        return bool(torch.txda.is_available())
    except Exception:  # noqa: BLE001 - any import/runtime failure means "no device"
        return False


def _run_child(mode: str, env_overrides: dict[str, str]) -> dict:
    env = os.environ.copy()
    for name in TIMING_ENVS:
        env.pop(name, None)
    # Never let an inherited real plugin leak into the "no profiler" cases.
    env.pop("ROCP_TOOL_LIBRARIES", None)
    overrides = dict(env_overrides)
    if overrides.get("LD_PRELOAD") == STUB_PLACEHOLDER:
        stub = _profiler_stub()
        existing = env.get("LD_PRELOAD")
        overrides["LD_PRELOAD"] = f"{stub}:{existing}" if existing else stub
    env.update(overrides)
    env["FLAGTREE_BACKEND"] = env.get("FLAGTREE_BACKEND", "tsingmicro")

    argv = [sys.executable, os.path.abspath(__file__), "--run-case", mode]
    proc = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=900,
                          check=False,  # inspected below; keep the captured output for the failure message
                          )
    if proc.returncode != 0:
        pytest.fail(f"case exited {proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    pytest.fail(f"case produced no result line\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")


@pytest.mark.skipif(not _device_available(), reason="no Tx81 device available")
@pytest.mark.parametrize("case_id,mode,env_overrides", CASES, ids=[c[0] for c in CASES])
def test_device_timing_and_graph_capture(case_id, mode, env_overrides):
    result = _run_child(mode, env_overrides)

    assert result["ok"] is True, result
    assert result["max_diff"] == 0.0, result

    capturing = mode.startswith("graph")

    # Launcher never reports a bogus device time while capturing.
    if capturing:
        assert result["capture_ok"] is True, result
        assert result["capture_timing_kind"] == 0, result
        assert result["capture_kernel_time_ns"] == 0, result

    tx_event_requested = "TILELANG_TX_EVENT_TIMING" in env_overrides
    profiler_stub = env_overrides.get("LD_PRELOAD") == STUB_PLACEHOLDER

    if profiler_stub:
        # TSM_PROFILER_EN + a resolvable libtsm_profiler.so: the profiler path
        # reports its own numbers, both on eager warmup and before capture.
        assert result["eager_timing_kind"] == 2, result
        assert result["eager_kernel_time_ns"] == STUB_KCORE_NS, result
        assert result["eager_ap_duration_ns"] == STUB_AP_NS, result
    elif mode == "graph_profile":
        assert result["profiler_ok"] is True, result
        assert result["eager_timing_kind"] == 0, result
    elif mode == "graph_cold":
        # No eager launch: nothing to assert outside the capture window.
        pass
    elif tx_event_requested:
        # The TX event path is active on real (non-capture) launches; the eager
        # warmup always runs outside capture, even in the graph cases.
        assert result["eager_timing_kind"] == 1, result
        assert result["eager_kernel_time_ns"] > 0, result
    else:
        # TSM_PROFILER_EN without a resolvable plugin produces no timing.
        assert result["eager_timing_kind"] == 0, result


# ---------------------------------------------------------------------------
# Child process: one launcher timing decision per process.
# ---------------------------------------------------------------------------
def _child_main(mode: str) -> int:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x, y, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(y + offs, tl.load(x + offs, mask=mask) + 1.0, mask=mask)

    n = 1024
    grid = (n // 128, )
    x = torch.ones(n, device="txda", dtype=torch.float32)
    y = torch.empty_like(x)

    def launch():
        add_kernel[grid](x, y, n, BLOCK=128)

    def last_result():
        res = triton.runtime.driver.active.get_last_launch_res()
        return int(res.timing_kind), int(res.kernel_time_ns), int(res.ap_duration_ns)

    payload = {"ok": False, "mode": mode}
    try:
        if mode == "graph_cold":
            # First launch happens inside capture: the launcher is compiled and
            # loaded while the stream is already capturing.
            graph = torch.txda.TXDAGraph()
            y.zero_()
            with torch.txda.graph(graph):
                launch()
            capture_kind, capture_ns, capture_ap = last_result()
            payload["capture_ok"] = True
            payload["capture_timing_kind"] = capture_kind
            payload["capture_kernel_time_ns"] = capture_ns
            payload["capture_ap_duration_ns"] = capture_ap
            graph.replay()
            torch.txda.synchronize()
        else:
            # Warm up the JIT/launcher and record an eager launch result.
            launch()
            torch.txda.synchronize()
            eager_kind, eager_ns, eager_ap = last_result()
            payload["eager_timing_kind"] = eager_kind
            payload["eager_kernel_time_ns"] = eager_ns
            payload["eager_ap_duration_ns"] = eager_ap

            if mode in ("graph", "graph_profile"):
                graph = torch.txda.TXDAGraph()
                if mode == "graph_profile":
                    # Host profiler + graph must coexist: torch_txda patches
                    # torch.profiler.profile with txda_profile under
                    # TSM_PROFILER_EN. Capture and replay inside the profile.
                    activities = [
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.PrivateUse1,
                    ]
                    profiler_ctx = torch.profiler.profile(activities=activities)
                    profiler_ctx.__enter__()
                y.zero_()
                with torch.txda.graph(graph):
                    launch()
                capture_kind, capture_ns, capture_ap = last_result()
                payload["capture_ok"] = True
                payload["capture_timing_kind"] = capture_kind
                payload["capture_kernel_time_ns"] = capture_ns
                payload["capture_ap_duration_ns"] = capture_ap
                graph.replay()
                torch.txda.synchronize()
                if mode == "graph_profile":
                    profiler_ctx.__exit__(None, None, None)
                    payload["profiler_ok"] = len(profiler_ctx.key_averages()) > 0
            else:
                y.zero_()
                launch()
                torch.txda.synchronize()

        expected = torch.full((n, ), 2.0, device="txda", dtype=torch.float32)
        payload["max_diff"] = float(torch.max(torch.abs(y - expected)).item())
        payload["ok"] = True
    except Exception:  # noqa: BLE001 - report any device/runtime failure to the parent
        payload["error"] = traceback.format_exc()
        print(f"{RESULT_PREFIX}{json.dumps(payload)}", flush=True)
        return 1

    print(f"{RESULT_PREFIX}{json.dumps(payload)}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--run-case":
        sys.exit(_child_main(sys.argv[2]))
    print(textwrap.dedent(__doc__))
    sys.exit(0)
