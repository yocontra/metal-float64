// Soft-fp64 classification / sign-magnitude / comparison extras:
//   isnan / isinf / isfinite / isnormal / signbit
//   fabs / copysign
//   fmin / fmax (NaN-flushing; NaN input -> other operand unless both NaN)
//   fdim / maxmag / minmag
//   nextafter
//   hypot   <-- CRITICAL: must use scaled formula, not sqrt(fma(a,a,b*b))
//
// Reference: C99 F.* / IEEE 754-2008 §5.3.1 for min/max semantics. Standard
// <math.h> reference impls for nextafter/hypot.
//
// hypot uses the scaled formula:
//     s = max(|a|, |b|)
//     t = min(|a|, |b|)
//     r = t / s
//     return s * sqrt(1 + r*r)
//   with these C99 F.10.4.3 payload rules:
//     hypot(±0, ±0) = +0
//     hypot(±inf, y) = +inf even if y is NaN
//     hypot(x, ±inf) = +inf even if x is NaN
//     otherwise if x or y is NaN, result is NaN
//   The naive `sqrt(fma(a,a,b*b))` formulation overflows when |a| > 1.34e154
//   and is WRONG.
//
// nextafter walks one representable value toward y:
//   - if x == y, return y (preserves sign of y per C99)
//   - if x == 0, return ±denorm_min toward y
//   - otherwise adjust the bit pattern by ±1, crossing exp boundaries naturally
//
// fmin/fmax per IEEE 754 §5.3.1 "minimum" / "maximum" are NaN-flushing:
//   NaN input -> return the other operand if it's non-NaN.
//   Only when BOTH are NaN do we return NaN.
//
// SPDX-License-Identifier: MIT

#include "internal.h"
#include "soft_fp64/soft_f64.h"

#include <cstdint>

using namespace soft_fp64::internal;

// ---- classification (pure bit ops) --------------------------------------

extern "C" int sf64_isnan(double x) {
    // SAFETY: bits_of is a bit_cast; purely read-only inspection.
    const uint64_t b = bits_of(x);
    return (extract_exp(b) == static_cast<uint32_t>(kExpMax)) && (extract_frac(b) != 0) ? 1 : 0;
}

extern "C" int sf64_isinf(double x) {
    // SAFETY: bit_cast; pure bit inspection.
    const uint64_t b = bits_of(x);
    return (extract_exp(b) == static_cast<uint32_t>(kExpMax)) && (extract_frac(b) == 0) ? 1 : 0;
}

extern "C" int sf64_isfinite(double x) {
    // SAFETY: bit_cast; pure bit inspection.
    return extract_exp(bits_of(x)) != static_cast<uint32_t>(kExpMax) ? 1 : 0;
}

extern "C" int sf64_isnormal(double x) {
    // SAFETY: bit_cast; pure bit inspection. Normal iff exponent in (0, 2047).
    const uint32_t e = extract_exp(bits_of(x));
    return (e != 0 && e != static_cast<uint32_t>(kExpMax)) ? 1 : 0;
}

extern "C" int sf64_signbit(double x) {
    // SAFETY: bit_cast; pure bit inspection.
    return static_cast<int>(bits_of(x) >> 63);
}

// ---- fabs / copysign (pure bit ops) -------------------------------------

extern "C" double sf64_fabs(double x) {
    // SAFETY: bit_cast then clear sign bit. No host FP arithmetic.
    return from_bits(bits_of(x) & ~kSignMask);
}

extern "C" double sf64_copysign(double x, double y) {
    // SAFETY: bit_cast; combine magnitude of x with sign of y via bitops only.
    return from_bits((bits_of(x) & ~kSignMask) | (bits_of(y) & kSignMask));
}

// ---- fmin / fmax (NaN-flushing, IEEE 754-2008 §5.3.1) -------------------

// File-static helper: signed bit-compare returning -1/0/+1 for a < b / a == b / a > b,
// with NaN handling delegated to the caller. Treats -0 == +0. Returns:
//   true  if a < b
//   false if a >= b (including a == b, or a > b)
// Both a and b must be non-NaN.
static inline bool bit_less_nonan(double a, double b) {
    // SAFETY: bit_cast to uint64_t for ordering. Two's-complement-like trick:
    //   positive doubles sort naturally by unsigned bits;
    //   negative doubles sort in REVERSE of unsigned bits.
    // Flip: for non-negative, set high bit; for negative, invert all bits.
    const uint64_t ab = bits_of(a);
    const uint64_t bb = bits_of(b);
    const uint64_t ak = (ab >> 63) != 0 ? ~ab : (ab | kSignMask);
    const uint64_t bk = (bb >> 63) != 0 ? ~bb : (bb | kSignMask);
    return ak < bk;
}

extern "C" double sf64_fmin(double a, double b) {
    // NaN flushing: if only one is NaN, return the other.
    if (sf64_isnan(a) != 0)
        return b;
    if (sf64_isnan(b) != 0)
        return a;
    // Signed-zero: fmin(-0, +0) = -0, fmin(+0, -0) = -0.
    const uint64_t ab = bits_of(a);
    const uint64_t bb = bits_of(b);
    if (is_zero_bits(ab) && is_zero_bits(bb)) {
        // Pick the one with sign bit set (if any).
        return from_bits((ab | bb) & kSignMask ? (ab & kSignMask ? ab : bb) : ab);
    }
    return bit_less_nonan(a, b) ? a : b;
}

extern "C" double sf64_fmax(double a, double b) {
    if (sf64_isnan(a) != 0)
        return b;
    if (sf64_isnan(b) != 0)
        return a;
    // Signed-zero: fmax(-0, +0) = +0.
    const uint64_t ab = bits_of(a);
    const uint64_t bb = bits_of(b);
    if (is_zero_bits(ab) && is_zero_bits(bb)) {
        // Prefer the positive zero.
        return from_bits((ab & kSignMask) == 0 ? ab : bb);
    }
    return bit_less_nonan(a, b) ? b : a;
}

// ---- fdim / maxmag / minmag ---------------------------------------------

extern "C" double sf64_fdim(double a, double b) {
    // C99: fdim(a,b) = (a > b) ? a - b : +0, NaN if either is NaN.
    if (sf64_isnan(a) != 0 || sf64_isnan(b) != 0) {
        return canonical_nan();
    }
    // a > b iff NOT (a < b OR a == b). Use bit compare for ordering.
    if (bit_less_nonan(b, a)) {
        // a > b: defer subtraction to the arithmetic TU.
        return sf64_sub(a, b);
    }
    return +0.0;
}

extern "C" double sf64_maxmag(double a, double b) {
    // NaN flushing consistent with fmin/fmax.
    if (sf64_isnan(a) != 0)
        return b;
    if (sf64_isnan(b) != 0)
        return a;
    const double aa = sf64_fabs(a);
    const double ab = sf64_fabs(b);
    if (bit_less_nonan(ab, aa))
        return a; // |a| > |b|
    if (bit_less_nonan(aa, ab))
        return b; // |b| > |a|
    // Tie-break via fmax (IEEE min/max semantics; signed-zero aware).
    return sf64_fmax(a, b);
}

extern "C" double sf64_minmag(double a, double b) {
    if (sf64_isnan(a) != 0)
        return b;
    if (sf64_isnan(b) != 0)
        return a;
    const double aa = sf64_fabs(a);
    const double ab = sf64_fabs(b);
    if (bit_less_nonan(aa, ab))
        return a; // |a| < |b|
    if (bit_less_nonan(ab, aa))
        return b; // |b| < |a|
    return sf64_fmin(a, b);
}

// ---- nextafter ----------------------------------------------------------

extern "C" double sf64_nextafter(double x, double y) {
    // NaN: propagate canonical NaN (any NaN input yields NaN).
    if (sf64_isnan(x) != 0 || sf64_isnan(y) != 0) {
        return canonical_nan();
    }

    // SAFETY: bit_cast both operands for bit-level neighbor arithmetic.
    const uint64_t xb = bits_of(x);
    const uint64_t yb = bits_of(y);

    // x == y (treating -0 == +0). C99 says return y (preserving sign of y).
    const bool x_is_zero = is_zero_bits(xb);
    const bool y_is_zero = is_zero_bits(yb);
    if (x_is_zero && y_is_zero)
        return y;

    // Equal non-zero magnitudes with equal signs: bit patterns identical.
    if (xb == yb)
        return y;

    // x == 0 (but y != 0): result is ±denorm_min with sign matching direction toward y.
    // Direction toward y is sign(y) since |y| > 0.
    if (x_is_zero) {
        const uint32_t y_sign = extract_sign(yb);
        return from_bits(pack(y_sign, 0, 1)); // ±denorm_min
    }

    // Determine direction: moving "up" in magnitude if |neighbor| > |x|.
    // x < y (numerically):
    //   - if x > 0: |result| > |x| (increment magnitude)
    //   - if x < 0: |result| < |x| (decrement magnitude, toward 0 then toward +)
    // x > y: reverse.
    const bool x_positive = (extract_sign(xb) == 0);
    // y_greater_than_x holds if y > x numerically. Use bit_less_nonan with
    // signed-zero special-case already handled above (both zero returned early).
    const bool y_greater_than_x = bit_less_nonan(x, y);

    // "move_up" = increase magnitude:
    //   x_pos && y>x   -> up
    //   x_pos && y<x   -> down
    //   x_neg && y<x   -> up
    //   x_neg && y>x   -> down
    const bool move_up = (x_positive == y_greater_than_x);

    // For positive x: up means bits+1, down means bits-1.
    // For negative x: up means bits+1 (further from 0 in negative direction
    //                 = larger magnitude = larger unsigned bits);
    //                 down means bits-1.
    // Combined: direction in raw unsigned bits depends on sign:
    //   x_positive: bits += move_up ? +1 : -1
    //   x_negative: bits += move_up ? +1 : -1
    // Wait -- negative doubles have increasing magnitude as raw bits increase
    // (since the sign bit is set and the magnitude bits are in the low 63 bits,
    //  adding 1 moves to larger magnitude). So the rule is the same:
    uint64_t rb = xb;
    if (move_up) {
        rb = rb + 1;
    } else {
        rb = rb - 1;
    }
    // Note: decrementing from the smallest positive subnormal (bits=1) yields
    // bits=0 which is +0.0 -- correct (nextafter(denorm_min, 0) == +0).
    // Decrementing from bits=0 (which we handled above as x==0 branch) would
    // underflow; that path is unreachable here.
    return from_bits(rb);
}

// ---- hypot (SCALED FORMULA) ---------------------------------------------

extern "C" double sf64_hypot(double a, double b) {
    // C99 F.10.4.3: inf beats NaN — if either input is infinite, return +inf
    // EVEN IF the other is NaN.
    if (sf64_isinf(a) != 0 || sf64_isinf(b) != 0) {
        return from_bits(kPositiveInf);
    }
    // Otherwise NaN propagates.
    if (sf64_isnan(a) != 0 || sf64_isnan(b) != 0) {
        return canonical_nan();
    }

    const double ax = sf64_fabs(a);
    const double ay = sf64_fabs(b);

    // Both zero -> +0.
    // SAFETY: bit_cast for zero check; no host FP.
    if (is_zero_bits(bits_of(ax)) && is_zero_bits(bits_of(ay))) {
        return +0.0;
    }

    // s = max(ax, ay), t = min(ax, ay). Both non-negative and non-NaN here,
    // so bit_less_nonan is equivalent to numeric <.
    const bool ax_lt_ay = bit_less_nonan(ax, ay);
    const double s = ax_lt_ay ? ay : ax;
    const double t = ax_lt_ay ? ax : ay;

    // Defensive: s == 0 implies t == 0 (since t <= s), handled above, but
    // double-check to avoid div-by-zero calls into sf64_div.
    if (is_zero_bits(bits_of(s)))
        return +0.0;

    // r = t / s ∈ [0, 1]. r*r ∈ [0, 1]. 1 + r*r ∈ [1, 2]. No overflow
    // regardless of the magnitude of a, b.
    const double r = sf64_div(t, s);
    const double r2 = sf64_mul(r, r);
    const double one_plus_r2 = sf64_add(1.0, r2);
    const double sq = sf64_sqrt(one_plus_r2);
    return sf64_mul(s, sq);
}
