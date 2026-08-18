#!/usr/bin/env python3

import os
import socket
import sys

import triton
from triton._C.libtriton import ir, spacemit
from triton.backends import backends


def check(description, condition, detail=""):
    print(f"[{'OK ' if condition else 'FAIL'}] {description}" + (f"  ({detail})" if detail else ""))
    return condition


def main():
    ok = True
    ok &= check("import triton", True, f"version={triton.__version__}")

    names = sorted(backends)
    ok &= check("spacemit backend registered", "spacemit" in names, f"backends={names}")

    backend = backends["spacemit"]
    ok &= check(
        "compiler/driver",
        backend.compiler.__name__ == "CPUBackend" and backend.driver.__name__ == "CPUDriver",
        f"{backend.compiler.__name__}/{backend.driver.__name__}",
    )

    pads = [value for value in dir(ir.PADDING_OPTION) if value.startswith("PAD")]
    ok &= check("PADDING_OPTION adapt", {"PAD_NEG_INF", "PAD_INF"} <= set(pads), f"pads={pads}")

    submodules = [name for name in dir(spacemit) if not name.startswith("__")]
    ok &= check("spacemit C++ submod", {"load_dialects", "tle_ir", "xsmt_ir"} <= set(submodules), f"sub={submodules}")

    host = os.environ.get("SPINE_TRITON_RPC_HOST", "127.0.0.1")
    port = int(os.environ.get("SPINE_TRITON_RPC_PORT", "9999"))
    with socket.socket() as sock:
        sock.settimeout(5)
        reachable = sock.connect_ex((host, port)) == 0
    ok &= check("QEMU RPC reachable", reachable, f"{host}:{port}")

    print()
    print("=== minimal smoke: " + ("ALL PASS" if ok else "FAILED") + " ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
