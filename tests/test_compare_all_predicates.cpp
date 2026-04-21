// Exhaustive coverage of all 16 LLVM FCmpInst::Predicate values for
// sf64_fcmp + IEEE 754-2008 minimum/maximum semantics for
// fmin_precise / fmax_precise.
//
// Strategy:
//   1. Edge-corpus cross product (21 x 21 = 441 pairs, 16 preds each).
//   2. 10^4 random-bit-pattern pairs (covers NaN / inf / subnormal lanes).
//   3. Signed-zero tie-break verification for fmin_precise / fmax_precise.
//
// The reference `host_pred` uses native `double` comparisons, which is
// fine in the TEST (we are verifying our bit-level implementation matches
// the host FPU on an x86/ARM CPU). It is NOT fine inside the library.
//
// SPDX-License-Identifier: MIT

#include "host_oracle.h"
#include "soft_fp64/soft_f64.h"

namespace {

int host_pred(double a, double b, int pred) {
    const bool nan = std::isnan(a) || std::isnan(b);
    switch (pred) {
    case 0:
        return 0;
    case 1:
        return !nan && (a == b) ? 1 : 0;
    case 2:
        return !nan && (a > b) ? 1 : 0;
    case 3:
        return !nan && (a >= b) ? 1 : 0;
    case 4:
        return !nan && (a < b) ? 1 : 0;
    case 5:
        return !nan && (a <= b) ? 1 : 0;
    case 6:
        return !nan && (a != b) ? 1 : 0;
    case 7:
        return !nan ? 1 : 0;
    case 8:
        return nan ? 1 : 0;
    case 9:
        return (nan || (a == b)) ? 1 : 0;
    case 10:
        return (nan || (a > b)) ? 1 : 0;
    case 11:
        return (nan || (a >= b)) ? 1 : 0;
    case 12:
        return (nan || (a < b)) ? 1 : 0;
    case 13:
        return (nan || (a <= b)) ? 1 : 0;
    case 14:
        return (nan || (a != b)) ? 1 : 0;
    case 15:
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    using namespace host_oracle;

    // ---- edge-case cross product × all 16 predicates ----
    for (double a : edge_cases_f64()) {
        for (double b : edge_cases_f64()) {
            for (int p = 0; p < 16; ++p) {
                int got = sf64_fcmp(a, b, p);
                int want = host_pred(a, b, p);
                if (got != want) {
                    std::fprintf(stderr, "FAIL: fcmp(a=%a, b=%a, pred=%d) got=%d want=%d\n", a, b,
                                 p, got, want);
                    std::abort();
                }
            }
        }
    }

    // ---- 10^4 random pairs × all 16 predicates ----
    LCG rng;
    for (int i = 0; i < 10000; ++i) {
        double a = rng.next_double_any();
        double b = rng.next_double_any();
        for (int p = 0; p < 16; ++p) {
            int got = sf64_fcmp(a, b, p);
            int want = host_pred(a, b, p);
            if (got != want) {
                std::fprintf(stderr, "FAIL: fcmp(a=%a, b=%a, pred=%d) got=%d want=%d (iter=%d)\n",
                             a, b, p, got, want, i);
                std::abort();
            }
        }
    }

    // ---- fmin_precise / fmax_precise on edge corpus ----
    for (double a : edge_cases_f64()) {
        for (double b : edge_cases_f64()) {
            double gmin = sf64_fmin_precise(a, b);
            double gmax = sf64_fmax_precise(a, b);
            bool either_nan = std::isnan(a) || std::isnan(b);
            if (either_nan) {
                SF64_CHECK(std::isnan(gmin));
                SF64_CHECK(std::isnan(gmax));
            } else {
                // Signed-zero tie-break: fmin(-0,+0)=-0, fmax(-0,+0)=+0.
                // When a != b numerically, the smaller/larger wins. When
                // they are numerically equal (both zero, opposite signs),
                // sign decides.
                double want_min;
                double want_max;
                if (a < b) {
                    want_min = a;
                    want_max = b;
                } else if (a > b) {
                    want_min = b;
                    want_max = a;
                } else {
                    // a == b numerically. If both zero with opposite sign,
                    // apply tie-break. Otherwise both are bit-equal for
                    // purposes of the result; picking either is fine.
                    want_min = std::signbit(a) ? a : b;
                    want_max = !std::signbit(a) ? a : b;
                }
                SF64_CHECK_BITS(gmin, want_min);
                SF64_CHECK_BITS(gmax, want_max);
            }
        }
    }

    // ---- fmin_precise / fmax_precise: 10^4 random pairs ----
    // Assert no traps, NaN-propagation holds, and non-NaN result is one
    // of the two inputs.
    LCG rng2(0xA11CE1234567ABCDULL);
    for (int i = 0; i < 10000; ++i) {
        double a = rng2.next_double_any();
        double b = rng2.next_double_any();
        double gmin = sf64_fmin_precise(a, b);
        double gmax = sf64_fmax_precise(a, b);
        if (std::isnan(a) || std::isnan(b)) {
            SF64_CHECK(std::isnan(gmin));
            SF64_CHECK(std::isnan(gmax));
        } else {
            // Result must equal one of the two inputs (bit-wise or -0/+0
            // tie-break winner).
            SF64_CHECK(bits(gmin) == bits(a) || bits(gmin) == bits(b));
            SF64_CHECK(bits(gmax) == bits(a) || bits(gmax) == bits(b));
            // Ordering invariant: fmin <= fmax numerically.
            SF64_CHECK(!(gmin > gmax));
        }
    }

    // ---- targeted signed-zero tie-break cases ----
    {
        const double pos0 = +0.0;
        const double neg0 = -0.0;
        SF64_CHECK_BITS(sf64_fmin_precise(neg0, pos0), neg0);
        SF64_CHECK_BITS(sf64_fmin_precise(pos0, neg0), neg0);
        SF64_CHECK_BITS(sf64_fmax_precise(neg0, pos0), pos0);
        SF64_CHECK_BITS(sf64_fmax_precise(pos0, neg0), pos0);
    }

    std::printf("test_compare_all_predicates: OK\n");
    return 0;
}
