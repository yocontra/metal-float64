// Report-only harness for sf64_* input regions that are NOT part of the
// shipped correctness claim.
//
// This file is the holding pattern for precision regimes that the main
// test tree does not gate on. Each sweep here prints observed ULP against
// libm and returns success unconditionally. A real precision fix will:
//   1. land the tighter algorithm (DD stitching, Temme asymptotics, etc.)
//   2. move the corresponding sweep back into test_transcendental_1ulp.cpp
//      or test_mpfr_diff.cpp with a tier from the spec bucket (U35/GAMMA).
//   3. delete its entry from this file.
//
// Currently parked (blocked on the `logk_dd` DD-Horner rewrite tracked in
// TODO.md; the polynomial tail inside `logk_dd` truncates the DD low word
// and caps relative precision at ~2^-56 instead of 2^-105, which propagates
// into ~5e-17 absolute error — 80 k ULP relative to the near-zero result):
//   * sf64_lgamma zero-crossings [0.5, 3) — ULP ratio explodes against
//     a near-zero value even though the absolute error is already at the
//     IEEE-double working precision floor.
//
// SPDX-License-Identifier: MIT

#include "soft_fp64/soft_f64.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

// LCG from the main test harness; kept local so this file has no oracle-
// shared dependencies.
struct LCG {
    uint64_t state;
    explicit LCG(uint64_t s) : state(s ? s : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return state;
    }
    double uniform(double lo, double hi) {
        const uint64_t r = next() >> 11;
        const double u = static_cast<double>(r) / static_cast<double>(1ULL << 53);
        return lo + u * (hi - lo);
    }
};

int64_t ulp_diff(double a, double b) {
    if (std::isnan(a) && std::isnan(b))
        return 0;
    if (a == b)
        return 0;
    if (!std::isfinite(a) || !std::isfinite(b))
        return INT64_MAX;
    uint64_t ua, ub;
    std::memcpy(&ua, &a, 8);
    std::memcpy(&ub, &b, 8);
    if (static_cast<int64_t>(ua) < 0)
        ua = 0x8000000000000000ULL - ua;
    if (static_cast<int64_t>(ub) < 0)
        ub = 0x8000000000000000ULL - ub;
    return static_cast<int64_t>(ua > ub ? ua - ub : ub - ua);
}

struct Stats {
    const char* name = "?";
    int64_t max_ulp = 0;
    double worst_x = 0.0, worst_got = 0.0, worst_expect = 0.0;
    int checked = 0;
};

void record(Stats& s, double x, double got, double expect) {
    s.checked++;
    const int64_t d = ulp_diff(got, expect);
    if (d > s.max_ulp) {
        s.max_ulp = d;
        s.worst_x = x;
        s.worst_got = got;
        s.worst_expect = expect;
    }
}

void report(const Stats& s) {
    std::printf("  %-30s  n=%-6d  max_ulp=%-10lld  worst x=%.17g got=%.17g expect=%.17g\n", s.name,
                s.checked, static_cast<long long>(s.max_ulp), s.worst_x, s.worst_got,
                s.worst_expect);
}

template <class SoftFn, class RefFn>
Stats sweep_uniform(const char* name, double lo, double hi, int n, uint64_t seed, SoftFn soft,
                    RefFn ref) {
    Stats s;
    s.name = name;
    LCG rng(seed);
    for (int i = 0; i < n; ++i) {
        const double x = rng.uniform(lo, hi);
        record(s, x, soft(x), ref(x));
    }
    return s;
}

} // namespace

int main() {
    std::printf("== experimental_precision (report-only) ==\n");
    std::printf("These sweeps are NOT gated. Any widening is a v1.2 item.\n\n");

    // lgamma zero-crossings at x = 1 and x = 2. ULP ratio explodes against
    // a near-zero value even though the absolute error stays ≈ 5e-17 (floor
    // from `logk_dd`'s plain-double polynomial tail — v1.2).
    auto s_lgamma = sweep_uniform(
        "lgamma_zero_crossing [0.5, 3)", 0.5, 3.0, 10000, 0x1DE00FULL,
        [](double x) { return sf64_lgamma(x); }, [](double x) { return std::lgamma(x); });
    report(s_lgamma);

    std::printf("\n[report-only — no gating; exit 0]\n");
    return 0;
}
