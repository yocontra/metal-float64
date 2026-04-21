// Self-contained microbenchmark harness for soft-fp64.
//
// Times each public sf64_* entry point over a fixed-size input array, in a
// loop whose iteration count auto-scales to at least --min-time-ms. Reports
// ns/op and Mops/sec. Emits JSON to stdout when --json is passed so the
// output can be diffed against bench/baseline.json for a regression gate.
//
// Design notes:
//   * No external deps (no Google Benchmark). Library consumers can vendor
//     soft-fp64 without pulling a test-harness tree.
//   * `-ffp-contract=off` is mandatory on this TU — otherwise the compiler
//     may fuse the host a*b+c inside the scaffolding, skewing fma vs
//     non-fma comparisons.
//   * Inputs are pre-materialised in arrays so the loop is a pure "call
//     sf64_*" and the memory fetch is the same across ops.
//   * All outputs feed a running accumulator to defeat dead-store
//     elimination; the accumulator is printed at end (behind a --dump flag).
//
// SPDX-License-Identifier: MIT

#include "soft_fp64/soft_f64.h"

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

// -------- pcg64 reproducible RNG (no <random> engine dependency on mt19937) --
// Using mt19937_64 for simplicity, seeded with a fixed constant. Bench output
// must be reproducible from a clean build.

constexpr uint64_t kSeed = 0xbadc0ffee0ddf00dULL;

constexpr size_t kBufSize = 1u << 14; // 16K doubles = 128 KB per buffer
constexpr int kDefaultMinTimeMs = 200;

// -------- clock ----------------------------------------------------------
using Clock = std::chrono::steady_clock;
using NsDur = std::chrono::duration<double, std::nano>;

double now_ns() {
    return std::chrono::duration_cast<NsDur>(Clock::now().time_since_epoch()).count();
}

// -------- input generators -----------------------------------------------
// Each bench gets a pre-filled input buffer. Distributions are tailored to
// the function under test (e.g. log wants positive inputs, pow wants a
// distribution that doesn't always overflow).

std::vector<double> gen_general(size_t n, uint64_t seed_mix = 0) {
    std::mt19937_64 rng(kSeed ^ seed_mix);
    std::vector<double> v(n);
    for (auto& x : v) {
        // Log-uniform in [1e-6, 1e6] with random sign.
        const double u = std::generate_canonical<double, 53>(rng); // [0,1)
        const double mag = std::exp(-13.81551 + u * 27.63102);     // log-uniform
        const uint64_t sign_bit = rng() & 1u;
        x = sign_bit ? -mag : mag;
    }
    return v;
}

std::vector<double> gen_positive(size_t n, uint64_t seed_mix = 0) {
    std::mt19937_64 rng(kSeed ^ seed_mix);
    std::vector<double> v(n);
    for (auto& x : v) {
        const double u = std::generate_canonical<double, 53>(rng);
        x = std::exp(-13.81551 + u * 27.63102); // (1e-6, 1e6)
    }
    return v;
}

std::vector<double> gen_unit(size_t n, uint64_t seed_mix = 0) {
    // [-1, 1] for asin/acos/atanh.
    std::mt19937_64 rng(kSeed ^ seed_mix);
    std::vector<double> v(n);
    for (auto& x : v) {
        const double u = std::generate_canonical<double, 53>(rng);
        x = -1.0 + 2.0 * u;
    }
    return v;
}

std::vector<double> gen_small(size_t n, uint64_t seed_mix = 0) {
    // [-10, 10] for trig / exp where large |x| is either pathological or
    // quickly saturating; realistic hot-path range.
    std::mt19937_64 rng(kSeed ^ seed_mix);
    std::vector<double> v(n);
    for (auto& x : v) {
        const double u = std::generate_canonical<double, 53>(rng);
        x = -10.0 + 20.0 * u;
    }
    return v;
}

std::vector<double> gen_exp_range(size_t n, uint64_t seed_mix = 0) {
    // [-40, 40] — exp/exp2/exp10 core range, avoids overflow but covers
    // reduction branches.
    std::mt19937_64 rng(kSeed ^ seed_mix);
    std::vector<double> v(n);
    for (auto& x : v) {
        const double u = std::generate_canonical<double, 53>(rng);
        x = -40.0 + 80.0 * u;
    }
    return v;
}

std::vector<double> gen_greater_than_one(size_t n, uint64_t seed_mix = 0) {
    // acosh requires x >= 1; we pick [1, 1e6] log-uniform.
    std::mt19937_64 rng(kSeed ^ seed_mix);
    std::vector<double> v(n);
    for (auto& x : v) {
        const double u = std::generate_canonical<double, 53>(rng);
        x = std::exp(u * 13.81551);
    }
    return v;
}

std::vector<int32_t> gen_int32(size_t n, uint64_t seed_mix = 0) {
    std::mt19937_64 rng(kSeed ^ seed_mix);
    std::vector<int32_t> v(n);
    for (auto& x : v)
        x = static_cast<int32_t>(rng());
    return v;
}

// -------- run loop --------------------------------------------------------
//
// Returns: ns/op (time per inner call) and Mops/sec. Auto-scales iteration
// count until at least min_time_ms elapses.
//
// The lambda `kernel(i)` does one call to the function under test at input
// index i mod buf.size().

struct Result {
    std::string name;
    double ns_per_op;
    double mops_per_sec;
    uint64_t iterations;
    double checksum;
};

template <typename K> Result run_bench(const std::string& name, K&& kernel, int min_time_ms) {
    const size_t warmup_iters = kBufSize;
    volatile double checksum = 0.0;

    // Warmup.
    for (size_t i = 0; i < warmup_iters; ++i)
        checksum += kernel(i);

    // Calibrate: double iter count until we've run >= min_time_ms.
    uint64_t iters = warmup_iters;
    double elapsed_ns = 0.0;
    while (true) {
        const double t0 = now_ns();
        double acc = 0.0;
        for (uint64_t i = 0; i < iters; ++i)
            acc += kernel(i);
        elapsed_ns = now_ns() - t0;
        checksum += acc;
        if (elapsed_ns >= static_cast<double>(min_time_ms) * 1.0e6)
            break;
        iters *= 2;
    }

    Result r;
    r.name = name;
    r.ns_per_op = elapsed_ns / static_cast<double>(iters);
    r.mops_per_sec = 1.0e3 / r.ns_per_op; // 1e9 ns/s / 1e6 ops → 1e3/ns
    r.iterations = iters;
    r.checksum = checksum;
    return r;
}

// -------- macro helpers --------------------------------------------------
//
// The "pass the buffer by reference" indirection is important: we want the
// compiler to treat the loop body as "load arr[i] → sf64_op → acc" with no
// inlining of test-harness arithmetic into the function under test.

#define BENCH_UNARY(NAME, FN, BUF)                                                                 \
    results.push_back(run_bench(                                                                   \
        NAME, [&](uint64_t i) -> double { return FN(BUF[i & (kBufSize - 1)]); }, min_time_ms))

#define BENCH_BINARY(NAME, FN, BUF_A, BUF_B)                                                       \
    results.push_back(run_bench(                                                                   \
        NAME,                                                                                      \
        [&](uint64_t i) -> double {                                                                \
            const size_t j = i & (kBufSize - 1);                                                   \
            return FN(BUF_A[j], BUF_B[j]);                                                         \
        },                                                                                         \
        min_time_ms))

#define BENCH_TERNARY(NAME, FN, BUF_A, BUF_B, BUF_C)                                               \
    results.push_back(run_bench(                                                                   \
        NAME,                                                                                      \
        [&](uint64_t i) -> double {                                                                \
            const size_t j = i & (kBufSize - 1);                                                   \
            return FN(BUF_A[j], BUF_B[j], BUF_C[j]);                                               \
        },                                                                                         \
        min_time_ms))

#define BENCH_INT_TO_F64(NAME, FN, BUF)                                                            \
    results.push_back(run_bench(                                                                   \
        NAME, [&](uint64_t i) -> double { return FN(BUF[i & (kBufSize - 1)]); }, min_time_ms))

#define BENCH_F64_TO_INT(NAME, FN, BUF)                                                            \
    results.push_back(run_bench(                                                                   \
        NAME,                                                                                      \
        [&](uint64_t i) -> double { return static_cast<double>(FN(BUF[i & (kBufSize - 1)])); },    \
        min_time_ms))

// -------- output ----------------------------------------------------------
void print_json(const std::vector<Result>& results) {
    std::printf("{\n  \"schema\": \"soft-fp64.bench.v1\",\n  \"results\": [\n");
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::printf("    {\"name\": \"%s\", \"ns_per_op\": %.4f, \"mops_per_sec\": %.4f, "
                    "\"iterations\": %" PRIu64 "}%s\n",
                    r.name.c_str(), r.ns_per_op, r.mops_per_sec, r.iterations,
                    i + 1 == results.size() ? "" : ",");
    }
    std::printf("  ]\n}\n");
}

void print_table(const std::vector<Result>& results) {
    std::printf("%-20s %14s %14s %14s\n", "op", "ns/op", "Mops/sec", "iterations");
    std::printf("%-20s %14s %14s %14s\n", "--------------------", "-------------", "-------------",
                "-------------");
    for (const auto& r : results) {
        std::printf("%-20s %14.4f %14.4f %14" PRIu64 "\n", r.name.c_str(), r.ns_per_op,
                    r.mops_per_sec, r.iterations);
    }
}

} // namespace

int main(int argc, char* argv[]) {
    int min_time_ms = kDefaultMinTimeMs;
    bool json = false;
    std::string filter;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--min-time-ms=", 0) == 0) {
            min_time_ms = std::atoi(a.c_str() + 14);
        } else if (a == "--json") {
            json = true;
        } else if (a.rfind("--filter=", 0) == 0) {
            filter = a.substr(9);
        } else if (a == "--help" || a == "-h") {
            std::printf("Usage: %s [--min-time-ms=N] [--json] [--filter=substr]\n", argv[0]);
            return 0;
        }
    }

    // Input buffers.
    const auto in_gen = gen_general(kBufSize);
    const auto in_gen2 = gen_general(kBufSize, 1);
    const auto in_gen3 = gen_general(kBufSize, 2);
    const auto in_pos = gen_positive(kBufSize);
    const auto in_pos2 = gen_positive(kBufSize, 1);
    const auto in_unit = gen_unit(kBufSize);
    const auto in_small = gen_small(kBufSize);
    const auto in_small2 = gen_small(kBufSize, 1);
    const auto in_exp = gen_exp_range(kBufSize);
    const auto in_gtone = gen_greater_than_one(kBufSize);
    const auto in_i32 = gen_int32(kBufSize);

    std::vector<Result> results;

    auto wanted = [&](const char* name) {
        return filter.empty() || std::string(name).find(filter) != std::string::npos;
    };

    // -------- arithmetic --------
    if (wanted("add"))
        BENCH_BINARY("add", sf64_add, in_gen, in_gen2);
    if (wanted("sub"))
        BENCH_BINARY("sub", sf64_sub, in_gen, in_gen2);
    if (wanted("mul"))
        BENCH_BINARY("mul", sf64_mul, in_gen, in_gen2);
    if (wanted("div"))
        BENCH_BINARY("div", sf64_div, in_gen, in_gen2);
    if (wanted("fma"))
        BENCH_TERNARY("fma", sf64_fma, in_gen, in_gen2, in_gen3);

    // -------- sqrt / rsqrt --------
    if (wanted("sqrt"))
        BENCH_UNARY("sqrt", sf64_sqrt, in_pos);
    if (wanted("rsqrt"))
        BENCH_UNARY("rsqrt", sf64_rsqrt, in_pos);

    // -------- convert --------
    if (wanted("from_i32"))
        BENCH_INT_TO_F64("from_i32", sf64_from_i32, in_i32);
    if (wanted("to_i32"))
        BENCH_F64_TO_INT("to_i32", sf64_to_i32, in_small);

    // -------- rounding --------
    if (wanted("floor"))
        BENCH_UNARY("floor", sf64_floor, in_gen);
    if (wanted("ceil"))
        BENCH_UNARY("ceil", sf64_ceil, in_gen);
    if (wanted("trunc"))
        BENCH_UNARY("trunc", sf64_trunc, in_gen);
    if (wanted("rint"))
        BENCH_UNARY("rint", sf64_rint, in_gen);

    // -------- compare --------
    if (wanted("fcmp_oeq"))
        results.push_back(run_bench(
            "fcmp_oeq",
            [&](uint64_t i) -> double {
                const size_t j = i & (kBufSize - 1);
                return static_cast<double>(sf64_fcmp(in_gen[j], in_gen2[j], 1));
            },
            min_time_ms));

    // -------- trig --------
    if (wanted("sin"))
        BENCH_UNARY("sin", sf64_sin, in_small);
    if (wanted("cos"))
        BENCH_UNARY("cos", sf64_cos, in_small);
    if (wanted("tan"))
        BENCH_UNARY("tan", sf64_tan, in_small);
    if (wanted("asin"))
        BENCH_UNARY("asin", sf64_asin, in_unit);
    if (wanted("acos"))
        BENCH_UNARY("acos", sf64_acos, in_unit);
    if (wanted("atan"))
        BENCH_UNARY("atan", sf64_atan, in_gen);
    if (wanted("atan2"))
        BENCH_BINARY("atan2", sf64_atan2, in_gen, in_gen2);

    // -------- hyperbolic --------
    if (wanted("sinh"))
        BENCH_UNARY("sinh", sf64_sinh, in_small);
    if (wanted("cosh"))
        BENCH_UNARY("cosh", sf64_cosh, in_small);
    if (wanted("tanh"))
        BENCH_UNARY("tanh", sf64_tanh, in_small);
    if (wanted("asinh"))
        BENCH_UNARY("asinh", sf64_asinh, in_gen);
    if (wanted("acosh"))
        BENCH_UNARY("acosh", sf64_acosh, in_gtone);
    if (wanted("atanh"))
        BENCH_UNARY("atanh", sf64_atanh, in_unit);

    // -------- exp / log --------
    if (wanted("exp"))
        BENCH_UNARY("exp", sf64_exp, in_exp);
    if (wanted("exp2"))
        BENCH_UNARY("exp2", sf64_exp2, in_exp);
    if (wanted("expm1"))
        BENCH_UNARY("expm1", sf64_expm1, in_small);
    if (wanted("log"))
        BENCH_UNARY("log", sf64_log, in_pos);
    if (wanted("log2"))
        BENCH_UNARY("log2", sf64_log2, in_pos);
    if (wanted("log1p"))
        BENCH_UNARY("log1p", sf64_log1p, in_gen);

    // -------- power / root --------
    if (wanted("pow"))
        BENCH_BINARY("pow", sf64_pow, in_pos, in_small2);
    if (wanted("cbrt"))
        BENCH_UNARY("cbrt", sf64_cbrt, in_gen);

    // -------- output --------
    if (json) {
        print_json(results);
    } else {
        print_table(results);
    }

    // Force the checksum to survive DSE.
    volatile double sink = 0.0;
    for (const auto& r : results)
        sink += r.checksum;
    (void)sink;

    return 0;
}
