// Tests for soft-fp64 sqrt / rsqrt / fma.
//
// sqrt and fma must be bit-exact vs host std::sqrt and std::fma across
// the edge-case corpus and 10^4 random inputs. rsqrt has a 1-ULP tolerance.
//
// SPDX-License-Identifier: MIT

#include "host_oracle.h"
#include "soft_fp64/soft_f64.h"

#include <cmath>
#include <cstdio>

int main() {
    using namespace host_oracle;

    // ---- sqrt — edge cases must be bit-exact. ---------------------------
    for (double x : edge_cases_f64()) {
        double got = sf64_sqrt(x);
        double expect = std::sqrt(x);
        SF64_CHECK_BITS(got, expect);
    }

    // ---- sqrt — random non-negative doubles. ----------------------------
    {
        LCG rng;
        for (int i = 0; i < 10000; ++i) {
            double x = std::fabs(rng.next_double_normal());
            double got = sf64_sqrt(x);
            double expect = std::sqrt(x);
            SF64_CHECK_BITS(got, expect);
        }
    }

    // ---- sqrt — subnormal inputs. ---------------------------------------
    {
        // Walk several subnormal bit patterns explicitly.
        for (uint64_t i = 1; i <= 32; ++i) {
            double x = from_bits(i); // tiny subnormal
            double got = sf64_sqrt(x);
            double expect = std::sqrt(x);
            SF64_CHECK_BITS(got, expect);
        }
    }

    // ---- fma — edge cases. -----------------------------------------------
    for (double a : edge_cases_f64()) {
        for (double b : edge_cases_f64()) {
            for (double c : edge_cases_f64()) {
                double got = sf64_fma(a, b, c);
                double expect = std::fma(a, b, c);
                SF64_CHECK_BITS(got, expect);
            }
        }
    }

    // ---- fma — random triples. -------------------------------------------
    {
        LCG rng;
        for (int i = 0; i < 10000; ++i) {
            double a = rng.next_double_normal();
            double b = rng.next_double_normal();
            double c = rng.next_double_normal();
            double got = sf64_fma(a, b, c);
            double expect = std::fma(a, b, c);
            SF64_CHECK_BITS(got, expect);
        }
    }

    // ---- rsqrt — 1 ULP tolerance (not bit-exact). -----------------------
    {
        LCG rng;
        for (int i = 0; i < 1000; ++i) {
            double x = std::fabs(rng.next_double_normal());
            if (x < 1e-300 || x > 1e300)
                continue;
            double got = sf64_rsqrt(x);
            double expect = 1.0 / std::sqrt(x);
            double ulp = std::nextafter(expect, expect > 0 ? HUGE_VAL : -HUGE_VAL) - expect;
            SF64_CHECK(std::fabs(got - expect) <= std::fabs(ulp));
        }
    }

    std::printf("test_sqrt_fma_exact: OK\n");
    return 0;
}
