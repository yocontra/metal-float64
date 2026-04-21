// Bit-exact correctness tests for soft-fp64 arithmetic.
//
// Every edge-case pair from `host_oracle::edge_cases_f64()` is checked
// against the host FPU, plus 10^4 random pairs from a deterministic LCG.
// Both halves of the comparison use `SF64_CHECK_BITS`, which accepts any
// quiet NaN for an expected-NaN result but is otherwise bit-exact.
//
// SPDX-License-Identifier: MIT

#include "host_oracle.h"
#include "soft_fp64/soft_f64.h"

#include <cmath>
#include <cstdio>

int main() {
    using namespace host_oracle;

    // ---- neg: trivial sign-flip on the full corpus. -----------------------
    for (double a : edge_cases_f64()) {
        SF64_CHECK_BITS(sf64_neg(a), -a);
    }

    // ---- edge-case cross product ------------------------------------------
    //
    // Every (a, b) from the 21-entry corpus. Exercises:
    //   - signed zeros
    //   - +/- 1, +/- 2, 0.5, 1.5
    //   - subnormal min (positive and negative)
    //   - DBL_MIN / DBL_MAX  (overflow to inf, underflow to zero)
    //   - +/- inf
    //   - quiet NaN
    //   - epsilon
    //   - pi, e (generic normals)
    for (double a : edge_cases_f64()) {
        for (double b : edge_cases_f64()) {
            SF64_CHECK_BITS(sf64_add(a, b), a + b);
            SF64_CHECK_BITS(sf64_sub(a, b), a - b);
            SF64_CHECK_BITS(sf64_mul(a, b), a * b);
            SF64_CHECK_BITS(sf64_div(a, b), a / b);
            SF64_CHECK_BITS(sf64_rem(a, b), std::fmod(a, b));
        }
    }

    // ---- targeted cancellation / underflow cases --------------------------
    {
        // Exact cancellation at large magnitude.
        const double x = 1.0e16;
        SF64_CHECK_BITS(sf64_sub(x, x), x - x);

        // Near-cancellation preserves all bits.
        const double a = 1.0;
        const double b = 0.9999999999999999;
        SF64_CHECK_BITS(sf64_sub(a, b), a - b);

        // Overflow to inf.
        const double m = 1.7976931348623157e308; // near DBL_MAX
        SF64_CHECK_BITS(sf64_add(m, m), m + m);
        SF64_CHECK_BITS(sf64_mul(m, 2.0), m * 2.0);

        // Gradual underflow: normal * normal → subnormal.
        const double tiny = 2.2250738585072014e-308; // DBL_MIN
        const double small = 0.5;
        SF64_CHECK_BITS(sf64_mul(tiny, small), tiny * small);

        // Subnormal + subnormal stays subnormal.
        const double sub = from_bits(0x0000000000000001ULL); // denorm_min
        SF64_CHECK_BITS(sf64_add(sub, sub), sub + sub);

        // fmod cases.
        SF64_CHECK_BITS(sf64_rem(5.0, 3.0), std::fmod(5.0, 3.0));
        SF64_CHECK_BITS(sf64_rem(-5.0, 3.0), std::fmod(-5.0, 3.0));
        SF64_CHECK_BITS(sf64_rem(5.0, -3.0), std::fmod(5.0, -3.0));
        SF64_CHECK_BITS(sf64_rem(1.0, 0.1), std::fmod(1.0, 0.1));
    }

    // ---- random sweep -----------------------------------------------------
    {
        LCG rng;
        for (int i = 0; i < 10000; ++i) {
            const double a = rng.next_double_normal();
            const double b = rng.next_double_normal();
            SF64_CHECK_BITS(sf64_add(a, b), a + b);
            SF64_CHECK_BITS(sf64_sub(a, b), a - b);
            SF64_CHECK_BITS(sf64_mul(a, b), a * b);
            if (b != 0.0) {
                SF64_CHECK_BITS(sf64_div(a, b), a / b);
                SF64_CHECK_BITS(sf64_rem(a, b), std::fmod(a, b));
            }
        }
    }

    // ---- random sweep across ALL bit patterns (NaN, inf, subnormal) -------
    //
    // We can't test exact equality of div for truly arbitrary bit patterns
    // because some combinations produce NaN with implementation-defined
    // payload; `equal_exact_or_nan` collapses that for us.
    {
        LCG rng(0x0123456789ABCDEFULL);
        for (int i = 0; i < 4096; ++i) {
            const double a = rng.next_double_any();
            const double b = rng.next_double_any();
            SF64_CHECK_BITS(sf64_add(a, b), a + b);
            SF64_CHECK_BITS(sf64_sub(a, b), a - b);
            SF64_CHECK_BITS(sf64_mul(a, b), a * b);
            SF64_CHECK_BITS(sf64_div(a, b), a / b);
            SF64_CHECK_BITS(sf64_rem(a, b), std::fmod(a, b));
        }
    }

    std::printf("test_arithmetic_exact: OK\n");
    return 0;
}
