#!/usr/bin/env bash

# Source this file so its environment and EXIT trap affect the current shell:
#   source third_party/spacemit/scripts/setup_rpc_session.sh

export no_proxy="${no_proxy:-127.0.0.1,localhost,::1}"
export SPINE_TRITON_CROSS_TOOLCHAIN="${SPINE_TRITON_CROSS_TOOLCHAIN:-$SPACEMIT_CACHE/toolchain}"

cleanup_spacemit_rpc() {
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "--- rpc-server process ---"
    if [ -f /tmp/rpc_server.pid ]; then
      ps -p "$(cat /tmp/rpc_server.pid)" 2>/dev/null || echo "process not running"
    fi
    echo "--- /tmp/rpc.log ---"
    cat /tmp/rpc.log 2>/dev/null || true
    echo "=== triton_dump dir ==="
    ls -la triton_dump/ 2>/dev/null || true
    for file in triton_dump/*; do
      [ -f "$file" ] || continue
      echo "--- $file ---"
      head -200 "$file"
    done
  fi

  if [ -f /tmp/rpc_server.pid ]; then
    kill "$(cat /tmp/rpc_server.pid)" 2>/dev/null || true
  fi
  pkill -f "spine-rpc-server" 2>/dev/null || true
}

trap cleanup_spacemit_rpc EXIT
