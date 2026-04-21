// test_hypot_scaling: verify hypot uses the scaled formula and does NOT
// overflow for magnitudes |a| > 1.34e154 where naive sqrt(a*a + b*b) fails.
//
// SPDX-License-Identifier: MIT

#include "host_oracle.h"
#include "soft_fp64/soft_f64.h"

#include <cmath>
#include <limits>

int main() {
    using namespace host_oracle;

    // ---- edge-case pair sweep --------------------------------------------
    for (double a : edge_cases_f64()) {
        for (double b : edge_cases_f64()) {
            double got = sf64_hypot(a, b);
            double expect = std::hypot(a, b);
            // C99 F.10.4.3: hypot(±inf, NaN) == +inf. Check inf before NaN.
            if (std::isinf(expect) || std::isinf(a) || std::isinf(b)) {
                SF64_CHECK(std::isinf(got));
                SF64_CHECK(got > 0.0);
                continue;
            }
            if (std::isnan(got)) {
                SF64_CHECK(std::isnan(expect));
                continue;
            }
            if (std::isnan(expect)) {
                SF64_CHECK(std::isnan(got));
                continue;
            }
            // 2 ULP tolerance — scaled formula isn't bit-exact vs fdlibm.
            const double ulp = std::fabs(std::nextafter(expect, expect * 2.0) - expect);
            if (expect == 0.0) {
                SF64_CHECK(got == 0.0);
            } else {
                SF64_CHECK(std::fabs(got - expect) <= 2.0 * ulp);
            }
        }
    }

    // ---- overflow-sensitive case: scaled formula MUST succeed here -------
    const double large = 1.5e154;
    const double got = sf64_hypot(large, large);
    SF64_CHECK(std::isfinite(got));                 // MUST NOT overflow to +inf
    const double expect = std::hypot(large, large); // ~2.12e154
    SF64_CHECK(std::isfinite(expect));
    const double rel_err = std::fabs((got - expect) / expect);
    SF64_CHECK(rel_err < 1e-13);

    // Additional large-magnitude sanity: 1e200 -- far past the naive limit.
    const double huge = 1e200;
    const double got_huge = sf64_hypot(huge, huge);
    SF64_CHECK(std::isfinite(got_huge));
    const double expect_huge = std::hypot(huge, huge);
    SF64_CHECK(std::isfinite(expect_huge));
    SF64_CHECK(std::fabs((got_huge - expect_huge) / expect_huge) < 1e-13);

    // ---- C99 F.10.4.3: inf beats NaN --------------------------------------
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    SF64_CHECK(sf64_hypot(inf, nan) == inf);
    SF64_CHECK(sf64_hypot(nan, inf) == inf);
    SF64_CHECK(sf64_hypot(-inf, nan) == inf);
    SF64_CHECK(sf64_hypot(nan, -inf) == inf);

    // ---- standard identities ---------------------------------------------
    SF64_CHECK_BITS(sf64_hypot(+0.0, +0.0), +0.0);
    SF64_CHECK_BITS(sf64_hypot(-0.0, -0.0), +0.0);
    SF64_CHECK_BITS(sf64_hypot(+0.0, -0.0), +0.0);

    // ---- subnormal input: scaled formula handles it ----------------------
    const double tiny = std::numeric_limits<double>::denorm_min();
    const double got_tiny = sf64_hypot(tiny, tiny);
    const double expect_tiny = std::hypot(tiny, tiny);
    if (expect_tiny != 0.0) {
        const double ulp_t =
            std::fabs(std::nextafter(expect_tiny, expect_tiny * 2.0) - expect_tiny);
        SF64_CHECK(std::fabs(got_tiny - expect_tiny) <= 4.0 * ulp_t);
    }

    return 0;
}
