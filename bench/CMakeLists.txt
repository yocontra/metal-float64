# Microbenchmark harness.
#
# Self-contained: no Google Benchmark FetchContent, no external deps. The
# harness emits JSON to stdout so regression gates / comparisons can be
# scripted easily. Turned on at the root with -DSOFT_FP64_BUILD_BENCH=ON.

add_executable(bench_soft_fp64 bench_soft_fp64.cpp)
target_link_libraries(bench_soft_fp64 PRIVATE soft_fp64::soft_fp64)
target_compile_options(bench_soft_fp64 PRIVATE
    -O3
    -ffp-contract=off
    -fno-math-errno
    -Wall -Wextra -Wpedantic)
if(SOFT_FP64_WERROR)
    target_compile_options(bench_soft_fp64 PRIVATE -Werror)
endif()

# Comparative bench vs Berkeley SoftFloat 3e and ckormanyos/soft_double.
# The external libraries are vendored into `bench/external/` on first build
# (see `bench/fetch_external.cmake`). Runs only if the sources are present;
# otherwise builds only the soft-fp64 column.
include("${CMAKE_CURRENT_SOURCE_DIR}/fetch_external.cmake")

if(SOFT_FP64_BENCH_HAVE_SOFTFLOAT OR SOFT_FP64_BENCH_HAVE_SOFT_DOUBLE)
    add_executable(bench_compare bench_compare.cpp)
    target_link_libraries(bench_compare PRIVATE soft_fp64::soft_fp64)
    target_compile_options(bench_compare PRIVATE
        -O3
        -ffp-contract=off
        -fno-math-errno
        -Wall -Wextra -Wpedantic)
    if(SOFT_FP64_WERROR)
        target_compile_options(bench_compare PRIVATE -Werror)
    endif()

    if(SOFT_FP64_BENCH_HAVE_SOFTFLOAT)
        target_compile_definitions(bench_compare PRIVATE SOFT_FP64_HAVE_SOFTFLOAT=1)
        target_link_libraries(bench_compare PRIVATE softfloat3e_bench)
    endif()
    if(SOFT_FP64_BENCH_HAVE_SOFT_DOUBLE)
        target_compile_definitions(bench_compare PRIVATE SOFT_FP64_HAVE_SOFT_DOUBLE=1)
        target_include_directories(bench_compare PRIVATE
            "${SOFT_FP64_BENCH_SOFT_DOUBLE_INCLUDE}")
    endif()
else()
    message(STATUS "bench: comparative bench skipped "
                   "(bench/external/ missing — run bench/fetch_external.sh)")
endif()
