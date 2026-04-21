#!/usr/bin/env bash
# Fetch comparison libraries for the comparative bench.
# Idempotent: skips clones that already exist.
#
# Licenses:
#   Berkeley SoftFloat 3e    BSD-3
#   ckormanyos/soft_double   Boost-1.0
#
# Both are compatible with our MIT root.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EXT="$HERE/external"
mkdir -p "$EXT"

clone() {
    local url=$1 dest=$2 ref=$3
    if [ -d "$dest/.git" ]; then
        echo "bench/external: $dest already present — skipping clone"
    else
        echo "bench/external: cloning $url -> $dest"
        git clone --depth 1 "$url" "$dest"
    fi
    ( cd "$dest" && git fetch --depth 1 origin "$ref" >/dev/null 2>&1 || true; git checkout "$ref" )
}

# Berkeley SoftFloat 3e (upstream "release" tag: 3e, master).
clone https://github.com/ucb-bar/berkeley-softfloat-3 "$EXT/softfloat" master

# ckormanyos/soft_double
clone https://github.com/ckormanyos/soft_double "$EXT/soft_double" main

echo
echo "Done. Now:"
echo "  cmake -S . -B build -DSOFT_FP64_BUILD_BENCH=ON"
echo "  cmake --build build --target bench_compare"
