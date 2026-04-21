#!/usr/bin/env bash
# Local coverage helper — mirrors the CI `coverage` job so a contributor can
# reproduce the number before pushing. Requires `lcov` (brew install lcov on
# macOS, apt install lcov on Debian/Ubuntu) and clang/gcc with --coverage.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
build="$root/build-coverage"

rm -rf "$build"
cmake -S "$root" -B "$build" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_FLAGS="--coverage" \
    -DCMAKE_CXX_FLAGS="--coverage" \
    -DSOFT_FP64_WERROR=OFF
cmake --build "$build" --parallel

ctest --test-dir "$build" --output-on-failure

lcov --capture --directory "$build" \
     --output-file "$build/coverage.info" \
     --ignore-errors mismatch,gcov,source
lcov --remove "$build/coverage.info" \
     '/usr/*' "*/tests/*" "*/_deps/*" \
     --output-file "$build/coverage.cleaned" \
     --ignore-errors unused

lcov --summary "$build/coverage.cleaned"
echo "Coverage report: $build/coverage.cleaned"
