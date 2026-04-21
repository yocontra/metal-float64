// Comparative microbench: soft-fp64 vs Berkeley SoftFloat 3e vs
// ckormanyos/soft_double.
//
// Same harness shape as bench_soft_fp64.cpp — pre-materialized input
// buffers, power-of-two indexing, auto-scaling iteration count. Emits
// one JSON row per (library, op) pair with a `.sf64` / `.softfloat` /
// `.soft_double` suffix so the comparison is grepable.
//
// Coverage (best effort):
//   core IEEE ops (add/sub/mul/div/fma/sqrt): all three libraries
//   convert / compare:                         sf64 + SoftFloat
//   transcendentals (sin, exp, log, pow, …):   sf64 + soft_double
//   (SoftFloat 3e ships no transcendentals — they're outside the project
//    scope; skipped rather than faked.)
//
// Build is conditional on the comparison libraries being vendored under
// bench/external/ — see fetch_external.sh and fetch_external.cmake.
// Flags `SOFT_FP64_HAVE_SOFTFLOAT` and `SOFT_FP64_HAVE_SOFT_DOUBLE` are
// set by CMake when the corresponding source tree is present.
//
// SPDX-License-Identifier: MIT

#include "soft_fp64/soft_f64.h"

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if SOFT_FP64_HAVE_SOFTFLOAT
extern "C" {
#include "softfloat.h"
}
#endif

#if SOFT_FP64_HAVE_SOFT_DOUBLE
#include "math/softfloat/soft_double.h"
#endif

namespace {

constexpr uint64_t kSeed = 0xbadc0ffee0ddf00dULL;
constexpr size_t kBufSize = 1u << 14;
constexpr int kDefaultMinTimeMs = 200;

using Clock = std::chrono::steady_clock;
using NsDur = std::chrono::duration<double, std::nano>;

double now_ns() {
    return std::chrono::duration_cast<NsDur>(Clock::now().time_since_epoch()).count();
}

// Distributions mirror bench_soft_fp64.cpp so per-op numbers line up.
std::vector<double> gen_general(size_t n, uint64_t seed_mix = 0) {
    std::mt19937_64 rng(kSeed ^ seed_mix);
    std::vector<double> v(n);
    for (auto& x : v) {
        const double u = std::generate_canonical<double, 53>(rng);
        const double mag = std::exp(-13.81551 + u * 27.63102);
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
        x = std::exp(-13.81551 + u * 27.63102);
    }
    return v;
}
std::vector<double> gen_small(size_t n, uint64_t seed_mix = 0) {
    std::mt19937_64 rng(kSeed ^ seed_mix);
    std::vector<double> v(n);
    for (auto& x : v) {
        const double u = std::generate_canonical<double, 53>(rng);
        x = -10.0 + 20.0 * u;
    }
    return v;
}
std::vector<double> gen_exp_range(size_t n, uint64_t seed_mix = 0) {
    std::mt19937_64 rng(kSeed ^ seed_mix);
    std::vector<double> v(n);
    for (auto& x : v) {
        const double u = std::generate_canonical<double, 53>(rng);
        x = -40.0 + 80.0 * u;
    }
    return v;
}

struct Result {
    std::string name;
    double ns_per_op = 0.0;
    double mops_per_sec = 0.0;
    uint64_t iterations = 0;
    double checksum = 0.0;
};

template <typename K> Result run_bench(const std::string& name, K&& kernel, int min_time_ms) {
    const size_t warmup_iters = kBufSize;
    volatile double checksum = 0.0;
    for (size_t i = 0; i < warmup_iters; ++i)
        checksum += kernel(i);

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
    r.mops_per_sec = 1.0e3 / r.ns_per_op;
    r.iterations = iters;
    r.checksum = checksum;
    return r;
}

void print_json(const std::vector<Result>& results) {
    std::printf("{\n  \"schema\": \"soft-fp64.bench-compare.v1\",\n  \"results\": [\n");
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
    std::printf("%-28s %14s %14s %14s\n", "op", "ns/op", "Mops/sec", "iterations");
    std::printf("%-28s %14s %14s %14s\n", "----------------------------", "-------------",
                "-------------", "-------------");
    for (const auto& r : results) {
        std::printf("%-28s %14.4f %14.4f %14" PRIu64 "\n", r.name.c_str(), r.ns_per_op,
                    r.mops_per_sec, r.iterations);
    }
}

// Bitcast helpers — keep the library-specific buffer-prep in one place.
#if SOFT_FP64_HAVE_SOFTFLOAT
float64_t to_sf(double x) {
    float64_t f;
    std::memcpy(&f.v, &x, sizeof(uint64_t));
    return f;
}
double from_sf(float64_t f) {
    double x;
    std::memcpy(&x, &f.v, sizeof(uint64_t));
    return x;
}
#endif

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

    const auto in_gen = gen_general(kBufSize);
    const auto in_gen2 = gen_general(kBufSize, 1);
    const auto in_gen3 = gen_general(kBufSize, 2);
    const auto in_pos = gen_positive(kBufSize);
    const auto in_pos2 = gen_positive(kBufSize, 1);
    const auto in_small = gen_small(kBufSize);
    const auto in_small2 = gen_small(kBufSize, 1);
    const auto in_exp = gen_exp_range(kBufSize);

    // Pre-cast inputs into each library's native doubleish type. Hot loops
    // then do a pure "load [j] → op → feed accumulator" so timings are not
    // diluted by the wrapping overhead.
#if SOFT_FP64_HAVE_SOFTFLOAT
    std::vector<float64_t> sf_in_gen(kBufSize);
    std::vector<float64_t> sf_in_gen2(kBufSize);
    std::vector<float64_t> sf_in_gen3(kBufSize);
    std::vector<float64_t> sf_in_pos(kBufSize);
    for (size_t i = 0; i < kBufSize; ++i) {
        sf_in_gen[i] = to_sf(in_gen[i]);
        sf_in_gen2[i] = to_sf(in_gen2[i]);
        sf_in_gen3[i] = to_sf(in_gen3[i]);
        sf_in_pos[i] = to_sf(in_pos[i]);
    }
#endif
#if SOFT_FP64_HAVE_SOFT_DOUBLE
    using math::softfloat::soft_double;
    std::vector<soft_double> sd_in_gen(kBufSize);
    std::vector<soft_double> sd_in_gen2(kBufSize);
    std::vector<soft_double> sd_in_gen3(kBufSize);
    std::vector<soft_double> sd_in_pos(kBufSize);
    std::vector<soft_double> sd_in_small(kBufSize);
    std::vector<soft_double> sd_in_small2(kBufSize);
    std::vector<soft_double> sd_in_exp(kBufSize);
    for (size_t i = 0; i < kBufSize; ++i) {
        sd_in_gen[i] = in_gen[i];
        sd_in_gen2[i] = in_gen2[i];
        sd_in_gen3[i] = in_gen3[i];
        sd_in_pos[i] = in_pos[i];
        sd_in_small[i] = in_small[i];
        sd_in_small2[i] = in_small2[i];
        sd_in_exp[i] = in_exp[i];
    }
#endif

    std::vector<Result> results;
    auto wanted = [&](const char* name) {
        return filter.empty() || std::string(name).find(filter) != std::string::npos;
    };

    // -------- core IEEE ops (add/sub/mul/div/fma/sqrt) --------
    auto core_op = [&](const char* op, auto sf64_fn, auto sf_fn, auto sd_fn) {
        if (wanted(op)) {
            {
                const std::string n = std::string(op) + ".sf64";
                results.push_back(run_bench(
                    n, [&](uint64_t i) -> double { return sf64_fn(i & (kBufSize - 1)); },
                    min_time_ms));
            }
#if SOFT_FP64_HAVE_SOFTFLOAT
            {
                const std::string n = std::string(op) + ".softfloat";
                results.push_back(run_bench(
                    n, [&](uint64_t i) -> double { return sf_fn(i & (kBufSize - 1)); },
                    min_time_ms));
            }
#else
            (void)sf_fn;
#endif
#if SOFT_FP64_HAVE_SOFT_DOUBLE
            {
                const std::string n = std::string(op) + ".soft_double";
                results.push_back(run_bench(
                    n, [&](uint64_t i) -> double { return sd_fn(i & (kBufSize - 1)); },
                    min_time_ms));
            }
#else
            (void)sd_fn;
#endif
        }
    };

#if SOFT_FP64_HAVE_SOFTFLOAT
#define SF_BIN(FN, A, B) [&](size_t j) { return from_sf(FN(A[j], B[j])); }
#define SF_UN(FN, A) [&](size_t j) { return from_sf(FN(A[j])); }
#define SF_TRI(FN, A, B, C) [&](size_t j) { return from_sf(FN(A[j], B[j], C[j])); }
#else
#define SF_BIN(FN, A, B) [](size_t) { return 0.0; }
#define SF_UN(FN, A) [](size_t) { return 0.0; }
#define SF_TRI(FN, A, B, C) [](size_t) { return 0.0; }
#endif

#if SOFT_FP64_HAVE_SOFT_DOUBLE
#define SD_BIN(OP, A, B) [&](size_t j) { return double(A[j] OP B[j]); }
#define SD_CALL1(FN, A) [&](size_t j) { return double(FN(A[j])); }
#define SD_CALL2(FN, A, B) [&](size_t j) { return double(FN(A[j], B[j])); }
#else
#define SD_BIN(OP, A, B) [](size_t) { return 0.0; }
#define SD_CALL1(FN, A) [](size_t) { return 0.0; }
#define SD_CALL2(FN, A, B) [](size_t) { return 0.0; }
#endif

    core_op(
        "add", [&](size_t j) { return sf64_add(in_gen[j], in_gen2[j]); },
        SF_BIN(f64_add, sf_in_gen, sf_in_gen2), SD_BIN(+, sd_in_gen, sd_in_gen2));
    core_op(
        "sub", [&](size_t j) { return sf64_sub(in_gen[j], in_gen2[j]); },
        SF_BIN(f64_sub, sf_in_gen, sf_in_gen2), SD_BIN(-, sd_in_gen, sd_in_gen2));
    core_op(
        "mul", [&](size_t j) { return sf64_mul(in_gen[j], in_gen2[j]); },
        SF_BIN(f64_mul, sf_in_gen, sf_in_gen2), SD_BIN(*, sd_in_gen, sd_in_gen2));
    core_op(
        "div", [&](size_t j) { return sf64_div(in_gen[j], in_gen2[j]); },
        SF_BIN(f64_div, sf_in_gen, sf_in_gen2), SD_BIN(/, sd_in_gen, sd_in_gen2));
    core_op(
        "fma", [&](size_t j) { return sf64_fma(in_gen[j], in_gen2[j], in_gen3[j]); },
        SF_TRI(f64_mulAdd, sf_in_gen, sf_in_gen2, sf_in_gen3),
    // soft_double has no FMA. Fall back to (a*b)+c so the column is
    // at least populated — this is what an application using soft_double
    // would write in the absence of fma.
#if SOFT_FP64_HAVE_SOFT_DOUBLE
        [&](size_t j) { return double(sd_in_gen[j] * sd_in_gen2[j] + sd_in_gen3[j]); }
#else
        [](size_t) { return 0.0; }
#endif
    );
    core_op(
        "sqrt", [&](size_t j) { return sf64_sqrt(in_pos[j]); },
#if SOFT_FP64_HAVE_SOFTFLOAT
        [&](size_t j) { return from_sf(f64_sqrt(sf_in_pos[j])); },
#else
        [](size_t) { return 0.0; },
#endif
#if SOFT_FP64_HAVE_SOFT_DOUBLE
        [&](size_t j) { return double(sqrt(sd_in_pos[j])); }
#else
        [](size_t) { return 0.0; }
#endif
    );

    // -------- transcendentals (sf64 + soft_double only) --------
    auto trans_op = [&](const char* op, auto sf64_fn, auto sd_fn) {
        if (wanted(op)) {
            const std::string n_sf64 = std::string(op) + ".sf64";
            results.push_back(run_bench(
                n_sf64, [&](uint64_t i) -> double { return sf64_fn(i & (kBufSize - 1)); },
                min_time_ms));
#if SOFT_FP64_HAVE_SOFT_DOUBLE
            const std::string n_sd = std::string(op) + ".soft_double";
            results.push_back(run_bench(
                n_sd, [&](uint64_t i) -> double { return sd_fn(i & (kBufSize - 1)); },
                min_time_ms));
#else
            (void)sd_fn;
#endif
        }
    };

    trans_op("sin", [&](size_t j) { return sf64_sin(in_small[j]); }, SD_CALL1(sin, sd_in_small));
    trans_op("cos", [&](size_t j) { return sf64_cos(in_small[j]); }, SD_CALL1(cos, sd_in_small));
    trans_op("tan", [&](size_t j) { return sf64_tan(in_small[j]); }, SD_CALL1(tan, sd_in_small));
    trans_op("exp", [&](size_t j) { return sf64_exp(in_exp[j]); }, SD_CALL1(exp, sd_in_exp));
    trans_op("log", [&](size_t j) { return sf64_log(in_pos[j]); }, SD_CALL1(log, sd_in_pos));
    trans_op(
        "pow", [&](size_t j) { return sf64_pow(in_pos[j], in_small2[j]); },
#if SOFT_FP64_HAVE_SOFT_DOUBLE
        [&](size_t j) { return double(pow(sd_in_pos[j], sd_in_small2[j])); }
#else
        [](size_t) { return 0.0; }
#endif
    );
    trans_op("sinh", [&](size_t j) { return sf64_sinh(in_small[j]); }, SD_CALL1(sinh, sd_in_small));
    trans_op("cosh", [&](size_t j) { return sf64_cosh(in_small[j]); }, SD_CALL1(cosh, sd_in_small));
    trans_op("tanh", [&](size_t j) { return sf64_tanh(in_small[j]); }, SD_CALL1(tanh, sd_in_small));

    if (json)
        print_json(results);
    else
        print_table(results);

    volatile double sink = 0.0;
    for (const auto& r : results)
        sink += r.checksum;
    (void)sink;

    return 0;
}
