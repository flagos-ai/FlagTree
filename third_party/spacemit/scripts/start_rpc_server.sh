#!/usr/bin/env bash

set -euo pipefail

RPC_PORT="${RPC_PORT:-9999}"
SPACEMIT_CACHE="${SPACEMIT_CACHE:-${FLAGTREE_CACHE_DIR:-$HOME/.flagtree}/spacemit}"
QEMU_BIN="${QEMU_BIN:-$SPACEMIT_CACHE/jdsk-qemu/bin/qemu-riscv64}"
SYSROOT="${SYSROOT:-$SPACEMIT_CACHE/toolchain/sysroot}"
RPC_DIR="${RPC_DIR:-$SPACEMIT_CACHE/rpc_runtime_installed}"

setsid bash -c "
  exec '$QEMU_BIN' -L '$SYSROOT' \
    -cpu max,vlen=1024,elen=64,vext_spec=v1.0 \
    -E LD_LIBRARY_PATH='$RPC_DIR/lib:$SYSROOT/lib:$SYSROOT/usr/lib' \
    '$RPC_DIR/bin/spine-rpc-server' '$RPC_PORT'
" > /tmp/rpc.log 2>&1 &
echo $! > /tmp/rpc_server.pid

rpc_ready=0
for i in $(seq 1 60); do
  if python3 -c "import socket,sys; sys.exit(0 if socket.socket().connect_ex(('127.0.0.1',$RPC_PORT)) == 0 else 1)" \
      2>/dev/null; then
    echo "RPC server ready on 127.0.0.1:${RPC_PORT} (${i}s)"
    rpc_ready=1
    break
  fi
  if ! kill -0 "$(cat /tmp/rpc_server.pid)" 2>/dev/null; then
    echo "ERROR: rpc-server process exited early" >&2
    cat /tmp/rpc.log 2>/dev/null || true
    exit 1
  fi
  sleep 1
done

if [ "$rpc_ready" -ne 1 ]; then
  echo "ERROR: RPC server did not come up on port ${RPC_PORT}" >&2
  cat /tmp/rpc.log 2>/dev/null || true
  exit 1
fi
