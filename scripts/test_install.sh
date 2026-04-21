#!/usr/bin/env bash
# End-to-end smoke test for the install rules.
#
#   1. Configure + build + install soft_fp64 into a throwaway prefix.
#   2. Build a tiny downstream CMake project that consumes it via
#      find_package(soft_fp64 CONFIG REQUIRED).
#   3. Run the downstream binary and verify the output.
#   4. Sanity-check the generated pkg-config file.
#
# Run from the repo root:  bash scripts/test_install.sh
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "[test_install] root=$root"
echo "[test_install] tmp=$tmp"

# --- 1. Build + install into $tmp/prefix --------------------------------------
cmake -S "$root" -B "$tmp/build" \
    -DCMAKE_INSTALL_PREFIX="$tmp/prefix" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSOFT_FP64_BUILD_TESTS=OFF
cmake --build "$tmp/build" --parallel
cmake --install "$tmp/build"

# --- 2. Downstream consumer project -------------------------------------------
mkdir -p "$tmp/consumer"
cat > "$tmp/consumer/CMakeLists.txt" <<'CONSUMER_CMAKE'
cmake_minimum_required(VERSION 3.20)
project(consume_soft_fp64 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(soft_fp64 CONFIG REQUIRED)
add_executable(consume consume.cpp)
target_link_libraries(consume PRIVATE soft_fp64::soft_fp64)
CONSUMER_CMAKE

cat > "$tmp/consumer/consume.cpp" <<'CONSUMER_CPP'
#include "soft_fp64/soft_f64.h"
#include <cstdio>
int main() {
  double r = sf64_add(1.0, 2.0);
  std::printf("%.1f\n", r);
  return (r == 3.0) ? 0 : 1;
}
CONSUMER_CPP

cmake -S "$tmp/consumer" -B "$tmp/consumer/build" \
    -DCMAKE_PREFIX_PATH="$tmp/prefix" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$tmp/consumer/build"

out=$("$tmp/consumer/build/consume")
if [ "$out" != "3.0" ]; then
    echo "[test_install] FAIL: downstream consume output was '$out' (expected '3.0')"
    exit 1
fi

# --- 3. pkg-config smoke test -------------------------------------------------
if command -v pkg-config >/dev/null 2>&1; then
    export PKG_CONFIG_PATH="$tmp/prefix/lib/pkgconfig"
    echo "[test_install] pkg-config --cflags --libs soft_fp64:"
    pkg-config --cflags --libs soft_fp64
else
    echo "[test_install] pkg-config not found, skipping .pc check"
fi

echo "[test_install] OK"
