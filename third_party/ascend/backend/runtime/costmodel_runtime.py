from __future__ import annotations

import hashlib
import json
import os
import warnings
from typing import Dict, List, Optional

from triton.runtime.cache import get_cache_manager

try:
    from triton.runtime.cache import triton_key as _triton_key
except ImportError:
    _triton_key = None


def _costmodel_cache_namespace() -> str:
    """Return a stable hex key accepted by triton.runtime.cache manager."""
    if _triton_key is None:
        raw = "costmodel_release322"
    else:
        try:
            raw = f"costmodel_{_triton_key()}"
        except Exception:
            raw = "costmodel_release322"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


_COSTMODEL_MEM_CACHE: Dict[str, float] = {}


def run_costmodel(ttir_or_path, extra_args=None, dump_ir_on_error=False):
    args = list(extra_args or [])
    if "-allow-unregistered-dialect" not in args:
        args.append("-allow-unregistered-dialect")

    from triton._C.libtriton import ascend as ascend_capi

    if os.path.exists(str(ttir_or_path)):
        with open(str(ttir_or_path), "r", encoding="utf-8") as f:
            mlir_text = f.read()
    else:
        mlir_text = ttir_or_path

    try:
        return ascend_capi.run_costmodel_inproc(mlir_text, args)
    except Exception as exc:
        if dump_ir_on_error and os.path.exists(str(ttir_or_path)):
            print(f"IR 文件: {ttir_or_path}")
        print(f"in-process costmodel failed: {exc}")
        return None


def make_costmodel_cache_key(ttir: str, extra_args: Optional[List[str]]) -> str:
    h = hashlib.sha256()
    h.update(ttir.encode("utf-8"))
    h.update(b"|")
    if extra_args:
        h.update(" ".join(extra_args).encode("utf-8"))
    h.update(b"|")
    h.update(b"inproc_costmodel_v2_loop_weighted")
    return h.hexdigest()


def load_costmodel_latency(cache_key: str) -> Optional[float]:
    cached = _COSTMODEL_MEM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cache_manager = get_cache_manager(_costmodel_cache_namespace())
    file_name = f"{cache_key}.json"
    payload_path = cache_manager.get_file(file_name)
    if payload_path is None:
        return None

    try:
        with open(payload_path, "r", encoding="utf-8") as f:
            parsed = json.load(f)
        latency = float(parsed["latency"])
        _COSTMODEL_MEM_CACHE[cache_key] = latency
        return latency
    except Exception:
        return None


def store_costmodel_latency(cache_key: str, latency: float) -> None:
    _COSTMODEL_MEM_CACHE[cache_key] = latency
    cache_manager = get_cache_manager(_costmodel_cache_namespace())
    file_name = f"{cache_key}.json"
    cache_manager.put(json.dumps({"latency": latency}), file_name, binary=False)


def parse_latency(stdout: str) -> float:
    import re

    match = re.search(r"Estimated Time:\s+([0-9.]+)\s*us", stdout)
    return float(match.group(1)) if match else float("inf")


def _warn_costmodel(msg: str) -> None:
    warnings.warn(f"costmodel_bench: {msg}", RuntimeWarning)


def _resolve_default_hardware_config() -> str:
    candidates = [
        os.path.join(os.path.dirname(__file__), "../../../costmodel/configs/ascend_910b.json"),
        os.path.join(os.path.dirname(__file__), "../../../../third_party/ascend/costmodel/configs/ascend_910b.json"),
        os.path.join(os.path.dirname(__file__), "../../../../../third_party/ascend/costmodel/configs/ascend_910b.json"),
    ]
    for candidate in candidates:
        path = os.path.abspath(candidate)
        if os.path.exists(path):
            return path
    return ""


def _build_costmodel_extra_args(arg_bindings: str, hardware_config: str = ""):
    base = "-ascend-perf-model"
    resolved_hardware_config = hardware_config or _resolve_default_hardware_config()
    # NOTE: Current inproc parser in installed runtime consumes only one payload
    # token after `-ascend-perf-model`. Frontend only forwards arg-bindings now.
    if arg_bindings:
        return [base, f"arg-bindings={arg_bindings}"]
    if resolved_hardware_config:
        return [base, f"hardware-config={resolved_hardware_config}"]
    return [base]


def _normalize_costmodel_item(item):
    if not isinstance(item, dict):
        return None, float("inf"), None

    config = item.get("config")
    if config is None:
        return None, float("inf"), None

    ttir = item.get("ttir")
    if not ttir:
        return config, float("inf"), None

    arg_bindings = item.get("arg_bindings", "")
    hardware_config = item.get("hardware_config", "")
    return config, None, (config, ttir, arg_bindings, hardware_config)


def _eval_one_costmodel_item(item):
    config, ttir, arg_bindings, hardware_config = item
    extra_args = _build_costmodel_extra_args(arg_bindings, hardware_config)
    cache_key = make_costmodel_cache_key(ttir, extra_args)
    cached = load_costmodel_latency(cache_key)
    if cached is not None:
        return config, cached

    output = run_costmodel(ttir_or_path=ttir, extra_args=extra_args)
    latency = float("inf") if output is None else parse_latency(output)
    store_costmodel_latency(cache_key, latency)
    return config, latency


def costmodel_bench(config_ttir_item):
    """Evaluate one candidate config with costmodel from a prebuilt TTIR payload.

    Args:
        config_ttir_item (dict): Per-config payload containing:
            - ``config``: config object returned as the result key.
            - ``ttir`` (str): TTIR text for costmodel evaluation.
            - ``arg_bindings`` (str, optional): Runtime bindings string passed
              to costmodel (for example ``"arg3=98432,pid_x=0"``).

    Returns:
        tuple: ``(config, latency_us)``. ``latency_us`` is ``float("inf")``
            when TTIR is missing/invalid or evaluation fails. Invalid items
            without a config return ``(None, float("inf"))``.
    """
    try:
        config, fallback_latency, pending_item = _normalize_costmodel_item(config_ttir_item)
        if config is None:
            return None, fallback_latency
        if pending_item is None:
            return config, fallback_latency
        return _eval_one_costmodel_item(pending_item)
    except Exception as exc:
        _warn_costmodel(f"unexpected failure: {exc}")
        if isinstance(config_ttir_item, dict) and config_ttir_item.get("config") is not None:
            return config_ttir_item["config"], float("inf")
        return None, float("inf")
