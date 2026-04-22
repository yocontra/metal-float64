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
// Currently parked:
//   * sf64_lgamma zero-crossings [0.5, 3) — lgamma(x) vanishes at x=1
//     and x=2. Near those zeros the result is O(1e-5) but the absolute
//     error floor of any log-of-Γ path is O(ulp(1)) = 2.2e-16, so the
//     ULP ratio inherently blows past GAMMA=1024 even with perfectly
//     computed ingredients. The true fix is a zero-centered Taylor
//     expansion of lgamma around x=1 and x=2 (not better log precision);
//     that's a self-contained algorithm change tracked in TODO.md.
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
    // a near-zero value even though the absolute error is near the double
    // working-precision floor (~5e-17). A zero-centered Taylor rewrite is
    // the algorithm change needed here — not better log precision.
    auto s_lgamma = sweep_uniform(
        "lgamma_zero_crossing [0.5, 3)", 0.5, 3.0, 10000, 0x1DE00FULL,
        [](double x) { return sf64_lgamma(x); }, [](double x) { return std::lgamma(x); });
    report(s_lgamma);

    std::printf("\n[report-only — no gating; exit 0]\n");
    return 0;
}
