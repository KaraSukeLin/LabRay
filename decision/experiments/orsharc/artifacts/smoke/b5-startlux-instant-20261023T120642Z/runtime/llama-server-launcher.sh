#!/bin/sh
set -eu
ROOT=/tmp/b5-startlux-smoke/llama-bundle/rootfs
APP="$ROOT/app"
LIBPATH="$APP:$ROOT/usr/local/cuda-12.8/targets/x86_64-linux/lib:$ROOT/usr/local/cuda-12.8/lib64:$ROOT/usr/lib/x86_64-linux-gnu:$ROOT/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$LIBPATH"
unset GGML_BACKEND_PATH || true
cd "$APP"
exec "$ROOT/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2" --library-path "$LIBPATH" "$APP/llama-server" "$@"
