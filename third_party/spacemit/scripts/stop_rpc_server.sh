#!/usr/bin/env bash

if [ -f /tmp/rpc_server.pid ]; then
  kill "$(cat /tmp/rpc_server.pid)" 2>/dev/null || true
fi
pkill -f "spine-rpc-server" 2>/dev/null || true
