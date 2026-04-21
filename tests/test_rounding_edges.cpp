// Rounding / exponent-extraction test.
//
// Covers floor/ceil/trunc/round/rint, fract/modf, ldexp and frexp edges.
// fract/modf on integral or zero inputs short-circuit to +0 without
// calling sub; the exhaustive non-integral cases are covered by the
// transcendental and MPFR oracles.
//
// SPDX-License-Identifier: MIT

#include "host_oracle.h"
#include "soft_fp64/soft_f64.h"

#include <cfenv>
#include <climits>
#include <cmath>
#include <cstdio>
#include <limits>

int main() {
    using namespace host_oracle;

    // floor / ceil / trunc / round / rint — edge cases (NaN handled specially).
    for (double x : edge_cases_f64()) {
        if (std::isnan(x)) {
            SF64_CHECK(std::isnan(sf64_floor(x)));
            SF64_CHECK(std::isnan(sf64_ceil(x)));
            SF64_CHECK(std::isnan(sf64_trunc(x)));
            SF64_CHECK(std::isnan(sf64_round(x)));
            SF64_CHECK(std::isnan(sf64_rint(x)));
        } else {
            SF64_CHECK_BITS(sf64_floor(x), std::floor(x));
            SF64_CHECK_BITS(sf64_ceil(x), std::ceil(x));
            SF64_CHECK_BITS(sf64_trunc(x), std::trunc(x));
            SF64_CHECK_BITS(sf64_round(x), std::round(x));
            std::fesetround(FE_TONEAREST);
            SF64_CHECK_BITS(sf64_rint(x), std::nearbyint(x));
        }
    }

    // round halves-away-from-zero
    SF64_CHECK_BITS(sf64_round(0.5), 1.0);
    SF64_CHECK_BITS(sf64_round(-0.5), -1.0);
    SF64_CHECK_BITS(sf64_round(1.5), 2.0);
    SF64_CHECK_BITS(sf64_round(2.5), 3.0);
    SF64_CHECK_BITS(sf64_round(-2.5), -3.0);
    SF64_CHECK_BITS(sf64_round(0.4), 0.0);
    SF64_CHECK_BITS(sf64_round(-0.4), -0.0);

    // rint halves-to-even
    std::fesetround(FE_TONEAREST);
    SF64_CHECK_BITS(sf64_rint(0.5), 0.0);
    SF64_CHECK_BITS(sf64_rint(-0.5), -0.0);
    SF64_CHECK_BITS(sf64_rint(1.5), 2.0);
    SF64_CHECK_BITS(sf64_rint(2.5), 2.0);
    SF64_CHECK_BITS(sf64_rint(3.5), 4.0);
    SF64_CHECK_BITS(sf64_rint(-3.5), -4.0);

    // fract: integral / huge inputs where fract short-circuits to +0.
    SF64_CHECK_BITS(sf64_fract(-0.0), 0.0);
    SF64_CHECK_BITS(sf64_fract(0.0), 0.0);
    SF64_CHECK_BITS(sf64_fract(1e20), 0.0);
    SF64_CHECK_BITS(sf64_fract(-1e20), 0.0);
    SF64_CHECK_BITS(sf64_fract(1.0), 0.0);
    SF64_CHECK_BITS(sf64_fract(-1.0), 0.0);
    // NaN / inf behavior.
    SF64_CHECK(std::isnan(sf64_fract(std::nan(""))));
    SF64_CHECK(std::isnan(sf64_fract(std::numeric_limits<double>::infinity())));

    // ldexp
    SF64_CHECK_BITS(sf64_ldexp(1.0, -1074), std::numeric_limits<double>::denorm_min());
    SF64_CHECK(std::isinf(sf64_ldexp(1.0, 1024)));
    SF64_CHECK_BITS(sf64_ldexp(1.0, 0), 1.0);
    SF64_CHECK_BITS(sf64_ldexp(1.0, 1), 2.0);
    SF64_CHECK_BITS(sf64_ldexp(1.0, -1), 0.5);
    SF64_CHECK_BITS(sf64_ldexp(-1.0, 10), -1024.0);
    SF64_CHECK_BITS(sf64_ldexp(+0.0, 100), +0.0);
    SF64_CHECK_BITS(sf64_ldexp(-0.0, 100), -0.0);
    SF64_CHECK(std::isnan(sf64_ldexp(std::nan(""), 10)));
    SF64_CHECK(std::isinf(sf64_ldexp(std::numeric_limits<double>::infinity(), -10)));
    // Cross-check against std::ldexp on a spread of finite normals.
    {
        LCG rng;
        for (int i = 0; i < 500; ++i) {
            double x = rng.next_double_normal();
            int n = static_cast<int>(rng.next() % 400) - 200;
            double got = sf64_ldexp(x, n);
            double expect = std::ldexp(x, n);
            SF64_CHECK_BITS(got, expect);
        }
    }

    // ilogb
    SF64_CHECK(sf64_ilogb(0.0) == INT_MIN);
    SF64_CHECK(sf64_ilogb(-0.0) == INT_MIN);
    SF64_CHECK(sf64_ilogb(std::nan("")) == INT_MAX);
    SF64_CHECK(sf64_ilogb(std::numeric_limits<double>::infinity()) == INT_MAX);
    SF64_CHECK(sf64_ilogb(-std::numeric_limits<double>::infinity()) == INT_MAX);
    SF64_CHECK(sf64_ilogb(std::numeric_limits<double>::denorm_min()) == -1074);
    SF64_CHECK(sf64_ilogb(std::numeric_limits<double>::min()) == -1022);
    SF64_CHECK(sf64_ilogb(1.0) == 0);
    SF64_CHECK(sf64_ilogb(2.0) == 1);
    SF64_CHECK(sf64_ilogb(0.5) == -1);
    SF64_CHECK(sf64_ilogb(std::numeric_limits<double>::max()) == 1023);

    // logb
    SF64_CHECK(std::isinf(sf64_logb(0.0)) && sf64_logb(0.0) < 0.0);
    SF64_CHECK(std::isnan(sf64_logb(std::nan(""))));
    SF64_CHECK(std::isinf(sf64_logb(std::numeric_limits<double>::infinity())) &&
               sf64_logb(std::numeric_limits<double>::infinity()) > 0.0);
    SF64_CHECK_BITS(sf64_logb(1.0), 0.0);
    SF64_CHECK_BITS(sf64_logb(2.0), 1.0);
    SF64_CHECK_BITS(sf64_logb(0.5), -1.0);

    // frexp — zeros / inf / NaN
    int e = 999;
    double fr = sf64_frexp(+0.0, &e);
    SF64_CHECK_BITS(fr, +0.0);
    SF64_CHECK(e == 0);
    e = 999;
    fr = sf64_frexp(-0.0, &e);
    SF64_CHECK_BITS(fr, -0.0);
    SF64_CHECK(e == 0);
    e = 999;
    fr = sf64_frexp(std::numeric_limits<double>::infinity(), &e);
    SF64_CHECK(std::isinf(fr) && fr > 0);
    e = 999;
    fr = sf64_frexp(std::nan(""), &e);
    SF64_CHECK(std::isnan(fr));

    // frexp result is in [0.5, 1) for nonzero finite inputs.
    {
        LCG rng;
        for (int i = 0; i < 500; ++i) {
            double x = rng.next_double_normal();
            if (x == 0.0)
                continue;
            int ex = 0;
            double m = sf64_frexp(x, &ex);
            double am = std::fabs(m);
            SF64_CHECK(am >= 0.5 && am < 1.0);
        }
    }

    // Random round-trip: frexp then ldexp recovers the original, bit-exact.
    {
        LCG rng;
        for (int i = 0; i < 1000; ++i) {
            double x = rng.next_double_normal();
            int ex = 0;
            double m = sf64_frexp(x, &ex);
            double recon = sf64_ldexp(m, ex);
            SF64_CHECK_BITS(recon, x);
        }
    }

    // modf — integer part written, fractional part sign-preserved on edges.
    {
        double ipart = 0.0;
        double f;
        f = sf64_modf(+0.0, &ipart);
        SF64_CHECK_BITS(f, +0.0);
        SF64_CHECK_BITS(ipart, +0.0);
        f = sf64_modf(-0.0, &ipart);
        SF64_CHECK_BITS(f, -0.0);
        SF64_CHECK_BITS(ipart, -0.0);
        f = sf64_modf(std::numeric_limits<double>::infinity(), &ipart);
        SF64_CHECK_BITS(f, +0.0);
        SF64_CHECK(std::isinf(ipart) && ipart > 0);
        f = sf64_modf(-std::numeric_limits<double>::infinity(), &ipart);
        SF64_CHECK_BITS(f, -0.0);
        SF64_CHECK(std::isinf(ipart) && ipart < 0);
        f = sf64_modf(std::nan(""), &ipart);
        SF64_CHECK(std::isnan(f));
        SF64_CHECK(std::isnan(ipart));
        // Integer inputs: fractional part is signed zero, iptr = x.
        f = sf64_modf(3.0, &ipart);
        SF64_CHECK_BITS(f, +0.0);
        SF64_CHECK_BITS(ipart, 3.0);
        f = sf64_modf(-3.0, &ipart);
        SF64_CHECK_BITS(f, -0.0);
        SF64_CHECK_BITS(ipart, -3.0);
    }

    std::printf("test_rounding_edges: OK\n");
    return 0;
}
