// f32 <-> f64 conversion, subnormal payload preservation.
//
// The critical invariant: sf64_from_f32 must preserve every
// f32 subnormal bit pattern through widening. Any compiler path that reads
// the operand as a `float` lvalue (instead of bit-casting) collapses
// subnormals on Apple6+ (MSL §6.20 flush-to-zero).
//
// SPDX-License-Identifier: MIT

#include "host_oracle.h"
#include "soft_fp64/soft_f64.h"

#include <cmath>
#include <cstdint>
#include <limits>

int main() {
    using namespace host_oracle;

    // Smallest subnormal f32 - must widen to exact f64 value without
    // payload loss.
    const float f_denorm = std::numeric_limits<float>::denorm_min();
    const double d_expect = static_cast<double>(f_denorm);
    const double d_got = sf64_from_f32(f_denorm);
    SF64_CHECK_BITS(d_got, d_expect);

    // Every f32 subnormal bit pattern (there are 2^23 - 1 positive ones;
    // stride through ~1% of them plus every power-of-two frac).
    for (uint32_t frac = 1; frac < (1u << 23); frac += 997) {
        const float f = f32_from_bits(frac); // positive subnormal
        const double got = sf64_from_f32(f);
        const double expect = static_cast<double>(f);
        SF64_CHECK_BITS(got, expect);

        const float fn = f32_from_bits(frac | 0x80000000u); // negative subnormal
        const double gotn = sf64_from_f32(fn);
        const double expectn = static_cast<double>(fn);
        SF64_CHECK_BITS(gotn, expectn);
    }

    // Every power-of-two fraction in the subnormal range — these are the
    // hardest to renormalize (leading-1 position varies by 1 bit).
    for (int k = 0; k < 23; ++k) {
        const uint32_t frac = 1u << k;
        const float f = f32_from_bits(frac);
        const double got = sf64_from_f32(f);
        const double expect = static_cast<double>(f);
        SF64_CHECK_BITS(got, expect);
    }

    // All f32 edge cases (normal, inf, NaN, signed zero, max).
    for (float f : edge_cases_f32()) {
        const double got = sf64_from_f32(f);
        const double expect = static_cast<double>(f);
        if (std::isnan(got)) {
            SF64_CHECK(std::isnan(expect));
        } else {
            SF64_CHECK_BITS(got, expect);
        }
    }

    // f64 -> f32 — all edge cases (including f64 subnormals that flush to 0,
    // huge values that overflow to inf, and NaN payloads).
    for (double d : edge_cases_f64()) {
        const float got = sf64_to_f32(d);
        const float expect = static_cast<float>(d);
        if (std::isnan(got)) {
            SF64_CHECK(std::isnan(expect));
        } else {
            SF64_CHECK(bits_f32(got) == bits_f32(expect));
        }
    }

    // f64 -> f32 rounding cases near the f32 subnormal boundary.
    // Sweep a handful of values around 2^-149 (denorm_min), 2^-126
    // (smallest normal), and the overflow boundary.
    const double tiny = static_cast<double>(std::numeric_limits<float>::denorm_min());
    const double small_normal = static_cast<double>(std::numeric_limits<float>::min());
    const double big = static_cast<double>(std::numeric_limits<float>::max());

    for (double d : {tiny, tiny * 0.5, tiny * 1.5, tiny * 2.0, small_normal, small_normal * 0.5,
                     small_normal * 0.75, big, big * 0.5}) {
        const float got = sf64_to_f32(d);
        const float expect = static_cast<float>(d);
        SF64_CHECK(bits_f32(got) == bits_f32(expect));

        const float got_neg = sf64_to_f32(-d);
        const float expect_neg = static_cast<float>(-d);
        SF64_CHECK(bits_f32(got_neg) == bits_f32(expect_neg));
    }

    // Random round-trip: normal f32 values widened to f64, then narrowed back
    // to f32 must return the exact same bit pattern.
    LCG rng;
    for (int i = 0; i < 20000; ++i) {
        const uint32_t b = static_cast<uint32_t>(rng.next());
        const float f = f32_from_bits(b);
        const double d = sf64_from_f32(f);
        const float back = sf64_to_f32(d);
        if (std::isnan(f)) {
            SF64_CHECK(std::isnan(back));
        } else {
            SF64_CHECK(bits_f32(back) == bits_f32(f));
        }
    }

    return 0;
}
