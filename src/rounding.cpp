// Soft-fp64 rounding + exponent extraction:
//   floor / ceil / trunc / round / rint / fract / modf
//   ldexp / frexp / ilogb / logb
//
// Reference: Mesa `src/compiler/nir/nir_lower_double_ops.c` (floor/ceil/
// trunc/fract — all bit-level, no arithmetic). glibc `sysdeps/ieee754/dbl-64/`
// for subnormal edge cases in ldexp/frexp/ilogb.
//
// Invariant: no host-FPU `+`, `-`, `*`, `/` on a `double` lvalue. Everything
// is bit ops on `uint64_t` fields. `fract` and `modf` have a runtime
// dependency on `sf64_sub` for the subtraction of the integer part; every
// other symbol is pure integer manipulation.
//
// SPDX-License-Identifier: MIT

#include "internal.h"
#include "soft_fp64/soft_f64.h"

#include <climits>
#include <cstdint>

using namespace soft_fp64::internal;

// Forward-declared locally (rather than in internal.h) so fract / modf can
// call the IEEE subtraction without pulling the public header's inline
// decoration into this TU.
extern "C" double sf64_sub(double a, double b);

namespace {

// ---- helpers ------------------------------------------------------------

// Return true if the biased-exp field indicates inf or NaN.
SF64_ALWAYS_INLINE bool is_exp_max(uint64_t b) noexcept {
    return extract_exp(b) == kExpMax;
}

// trunc(x) expressed as bit manipulation only. Clears the fractional bits
// of the mantissa below the implicit point.
SF64_ALWAYS_INLINE double trunc_bits(double x) noexcept {
    // SAFETY: __builtin_bit_cast: identical-size POD round-trip, no UB.
    const uint64_t b = bits_of(x);
    const uint32_t e = extract_exp(b);

    // NaN or +/-inf: return x unchanged (NaN is passed through; its quiet
    // bit will already be set for all canonical NaNs produced elsewhere in
    // the library, and the IEEE-754 rule for trunc(NaN) is "NaN with the
    // same payload").
    if (e == kExpMax)
        return x;

    // Unbiased exponent. Using signed arithmetic here, not fp.
    const int unbiased = static_cast<int>(e) - kExpBias;

    if (unbiased < 0) {
        // |x| < 1 → truncates to signed zero, preserving sign of x.
        return make_signed_zero(extract_sign(b));
    }
    if (unbiased >= kFracBits) {
        // Already an integer (exp >= 52): nothing to clear.
        return x;
    }

    // Mask of the fractional bits below the integer point. For unbiased=e,
    // the integer point is at bit (52 - e) of the mantissa.
    const uint64_t frac_mask = kFracMask >> unbiased;
    // SAFETY: __builtin_bit_cast round-trip via from_bits; bit layout
    // guaranteed by IEEE-754 binary64.
    return from_bits(b & ~frac_mask);
}

// Returns true iff any fractional bits below the integer point are set.
// For exp values that make x already integral (or inf/NaN), returns false.
SF64_ALWAYS_INLINE bool has_fractional_part(uint64_t b) noexcept {
    const uint32_t e = extract_exp(b);
    if (e == kExpMax)
        return false; // inf/NaN — no "fraction" to speak of.
    const int unbiased = static_cast<int>(e) - kExpBias;
    if (unbiased >= kFracBits)
        return false; // already integer.
    if (unbiased < 0) {
        // |x| < 1, and x != 0 ⇒ fractional. x == 0 ⇒ no fraction.
        return (b & ~kSignMask) != 0;
    }
    const uint64_t frac_mask = kFracMask >> unbiased;
    return (b & frac_mask) != 0;
}

// Produce the next integer towards +/-inf from x, given that trunc(x) != x.
// Direction sign: 0 = towards +inf, 1 = towards -inf.
SF64_ALWAYS_INLINE double next_integer_away(double x_trunc, uint32_t direction_sign) noexcept {
    // Strategy: trunc(x) is an integer in fp64 form. "Next integer in the
    // direction of ±inf" for a normal value means add 1.0 in magnitude. We
    // can't call soft-fp64 add here without a runtime dep; instead we
    // construct `trunc(x) ± 1.0` via a bit-level increment of the mantissa
    // at the LSB of the integer part, carrying into the exponent on wrap.

    // SAFETY: __builtin_bit_cast round-trip via from_bits; POD-sized cast.
    uint64_t b = bits_of(x_trunc);
    const uint32_t sign = extract_sign(b);

    // If x_trunc is zero: the result is ±1.0 depending on `direction_sign`.
    // +0 → ceil(0.5)=1 needs away=towards-+inf (sign=0) → +1.0.
    // -0 → floor(-0.5)=-1 needs away=towards--inf (sign=1) → -1.0.
    if ((b & ~kSignMask) == 0) {
        // Build ±1.0 with sign = direction_sign.
        return from_bits(pack(direction_sign, kExpBias, 0));
    }

    const bool moving_away = (sign == direction_sign);
    // The LSB of the integer part sits at bit (52 - unbiased) of the
    // mantissa; for unbiased>=52 we're already at a huge integer and the
    // caller guarantees we won't hit this case (has_fractional_part was
    // false). For 0 <= unbiased < 52, we add/sub 1 at position
    // (52 - unbiased).
    const uint32_t e = extract_exp(b);
    const int unbiased = static_cast<int>(e) - kExpBias;
    // Caller guarantees unbiased is in [0, 51] here, since (a) x_trunc was
    // produced by trunc_bits from a value with fractional bits and (b)
    // |x| >= 1.
    const int ulp_shift = kFracBits - unbiased;
    const uint64_t one_ulp = uint64_t{1} << ulp_shift;

    if (moving_away) {
        // Adding 1 in magnitude: add one_ulp to the mantissa-plus-exp field.
        // Carry from mantissa into exp field happens naturally in the
        // uint64_t add; if it overflows the mantissa (all frac + implicit
        // bit set) then exp increments, which is the correct next-integer
        // representation (e.g. 2^53-1 + 1 = 2^53).
        b = b + one_ulp;
    } else {
        // Subtracting 1 in magnitude: subtract one_ulp from the combined
        // exp+mantissa field. Borrow from exp handles rollover down to a
        // smaller exponent correctly (e.g. 2.0 - 1.0 = 1.0, where the
        // exponent decrements).
        b = b - one_ulp;
    }
    // SAFETY: from_bits = bit_cast to double; always valid for any 64-bit
    // pattern (IEEE-754 doesn't have trap representations).
    return from_bits(b);
}

} // namespace

// -----------------------------------------------------------------------------
// trunc
// -----------------------------------------------------------------------------
extern "C" double sf64_trunc(double x) {
    return trunc_bits(x);
}

// -----------------------------------------------------------------------------
// floor = towards -inf
// -----------------------------------------------------------------------------
extern "C" double sf64_floor(double x) {
    // SAFETY: bit_cast for field inspection — no UB on any input bit pattern.
    const uint64_t b = bits_of(x);
    if (is_exp_max(b))
        return x; // NaN/inf unchanged.
    const double t = trunc_bits(x);
    if (!has_fractional_part(b))
        return t; // already integer.
    const uint32_t sign = extract_sign(b);
    if (sign == 0)
        return t; // +x: floor(+) = trunc(+).
    // Negative non-integer: move trunc one integer towards -inf.
    return next_integer_away(t, /*direction_sign=*/1u);
}

// -----------------------------------------------------------------------------
// ceil = towards +inf
// -----------------------------------------------------------------------------
extern "C" double sf64_ceil(double x) {
    const uint64_t b = bits_of(x);
    if (is_exp_max(b))
        return x;
    const double t = trunc_bits(x);
    if (!has_fractional_part(b))
        return t;
    const uint32_t sign = extract_sign(b);
    if (sign == 1)
        return t; // -x: ceil(-) = trunc(-).
    return next_integer_away(t, /*direction_sign=*/0u);
}

// -----------------------------------------------------------------------------
// round (halves-away-from-zero)
// -----------------------------------------------------------------------------
extern "C" double sf64_round(double x) {
    const uint64_t b = bits_of(x);
    if (is_exp_max(b))
        return x;

    const uint32_t e = extract_exp(b);
    const int unbiased = static_cast<int>(e) - kExpBias;
    const uint32_t sign = extract_sign(b);

    if (unbiased < 0) {
        // |x| < 1. Half-way is exactly 0.5 (unbiased = -1, frac = 0).
        // round(±0.5) = ±1.0 (away from zero). Anything with |x| < 0.5
        // rounds to ±0.0.
        if (unbiased == -1 && (b & kFracMask) == 0) {
            // |x| == 0.5 exactly → ±1.0.
            return from_bits(pack(sign, kExpBias, 0));
        }
        if (unbiased == -1) {
            // 0.5 < |x| < 1 → away-from-zero direction gives ±1.0.
            return from_bits(pack(sign, kExpBias, 0));
        }
        // |x| < 0.5 → ±0.0.
        return make_signed_zero(sign);
    }
    if (unbiased >= kFracBits)
        return x; // already integer (or huge).

    // Half-way bit is at position (51 - unbiased) within the mantissa.
    // (Bit 52-unbiased is the integer-LSB; the bit just below it is the
    // halfway.)
    const int half_bit_pos = kFracBits - unbiased - 1;
    const uint64_t half_mask = uint64_t{1} << half_bit_pos;
    const uint64_t below_half_mask = half_mask - 1;   // all bits below halfway
    const uint64_t frac_mask = kFracMask >> unbiased; // all bits below integer

    if ((b & frac_mask) == 0)
        return x; // exact integer already.

    const double t = trunc_bits(x);
    const bool half_set = (b & half_mask) != 0;
    const bool below_half_set = (b & below_half_mask) != 0;

    // Away-from-zero: round up when halfway bit set (regardless of below,
    // ties go away). Also round up when below-half set and halfway set
    // (that's > 0.5). When halfway is clear, fractional part is < 0.5 → down.
    if (half_set || below_half_set) {
        // half_set => >=0.5 of next ulp up; go away.
        if (half_set) {
            return next_integer_away(t, sign); // away from zero.
        }
        // half_set==0, below_half_set==1 means the fractional is < 0.5,
        // truncation direction.
        return t;
    }
    return t; // unreachable (we'd have returned for exact integer above).
}

// -----------------------------------------------------------------------------
// rint (halves-to-even)
// -----------------------------------------------------------------------------
extern "C" double sf64_rint(double x) {
    const uint64_t b = bits_of(x);
    if (is_exp_max(b))
        return x;

    const uint32_t e = extract_exp(b);
    const int unbiased = static_cast<int>(e) - kExpBias;
    const uint32_t sign = extract_sign(b);

    if (unbiased < 0) {
        // |x| < 1. 0.5 → 0.0 (even). 0.5 < |x| < 1 → ±1. |x| < 0.5 → ±0.
        if (unbiased == -1 && (b & kFracMask) == 0) {
            // ±0.5 → ±0.0 (0 is even).
            return make_signed_zero(sign);
        }
        if (unbiased == -1) {
            // 0.5 < |x| < 1 → nearest is ±1.
            return from_bits(pack(sign, kExpBias, 0));
        }
        return make_signed_zero(sign);
    }
    if (unbiased >= kFracBits)
        return x;

    const int half_bit_pos = kFracBits - unbiased - 1;
    const uint64_t half_mask = uint64_t{1} << half_bit_pos;
    const uint64_t below_half_mask = half_mask - 1;
    const uint64_t frac_mask = kFracMask >> unbiased;

    if ((b & frac_mask) == 0)
        return x; // exact integer.

    const double t = trunc_bits(x);
    const bool half_set = (b & half_mask) != 0;
    const bool below_half_set = (b & below_half_mask) != 0;

    if (!half_set) {
        // Frac < 0.5 → round towards zero (trunc).
        return t;
    }
    if (below_half_set) {
        // Frac > 0.5 → round away from zero.
        return next_integer_away(t, sign);
    }
    // Exact halfway: round to even. The integer-LSB of trunc(x) is at
    // (52 - unbiased) of the trunc bits. If that bit is 0, t is even → keep.
    // Otherwise t is odd → move one integer away from zero so the result is
    // even.
    //
    // NOTE: for unbiased == 0, the integer-LSB is bit 52 — which is the
    // implicit bit. For any nonzero x with unbiased==0, implicit bit is 1,
    // so trunc is ±1 which is odd; halfway case is ±1.5 → ±2. Correct.
    const int int_lsb_pos = kFracBits - unbiased;
    // SAFETY: bit_cast inspection of trunc-result bits.
    const uint64_t tb = bits_of(t);
    uint64_t int_lsb_val;
    if (int_lsb_pos == kFracBits) {
        // Integer-LSB is the implicit bit. Any nonzero normal has it set.
        // If trunc is zero, we wouldn't be here (exact-integer check above).
        int_lsb_val = 1;
    } else {
        int_lsb_val = (tb >> int_lsb_pos) & uint64_t{1};
    }
    if (int_lsb_val == 0)
        return t; // already even.
    return next_integer_away(t, sign);
}

// -----------------------------------------------------------------------------
// rint_r — mode-parametrized integer-rounding.
// -----------------------------------------------------------------------------
extern "C" double sf64_rint_r(sf64_rounding_mode mode, double x) {
    switch (mode) {
    case SF64_RTZ:
        return sf64_trunc(x);
    case SF64_RUP:
        return sf64_ceil(x);
    case SF64_RDN:
        return sf64_floor(x);
    case SF64_RNA:
        return sf64_round(x);
    case SF64_RNE:
    default:
        return sf64_rint(x);
    }
}

// -----------------------------------------------------------------------------
// fract = x - floor(x), with fract(-0) = +0 and fract(large) = +0
// -----------------------------------------------------------------------------
extern "C" double sf64_fract(double x) {
    const uint64_t b = bits_of(x);

    // NaN → NaN (quieted); inf → canonical NaN per GLSL fract(±inf)=NaN.
    if (is_nan_bits(b)) {
        // SAFETY: bit OR of quiet-bit on known NaN; result is still NaN.
        return from_bits(b | kQuietNaNBit);
    }
    if (is_inf_bits(b)) {
        return canonical_nan();
    }

    const double f = sf64_floor(x);
    // If x is already integral (incl. ±0, ±1, huge magnitudes), result is +0.
    // SAFETY: bit_cast for comparison of two fp64 bit patterns.
    if (bits_of(f) == b) {
        return from_bits(0ULL); // +0.0
    }
    // fract has no pure-bit formulation for non-integer inputs; the value
    // x - floor(x) genuinely requires an IEEE subtraction. The GLSL spec
    // pins fract in [0, 1).
    const double r = sf64_sub(x, f);
    // Guarantee non-negative zero exit (sub could produce -0 in theory; fract
    // is defined in [0, 1)).
    // SAFETY: bit_cast comparison to zero, preserves value for nonzero.
    if ((bits_of(r) & ~kSignMask) == 0) {
        return from_bits(0ULL);
    }
    return r;
}

// -----------------------------------------------------------------------------
// modf — fractional + integer parts, same sign as x
// -----------------------------------------------------------------------------
extern "C" double sf64_modf(double x, double* iptr) {
    const uint64_t b = bits_of(x);

    // NaN: store NaN in iptr, return NaN.
    if (is_nan_bits(b)) {
        // SAFETY: bit OR of quiet-bit on NaN; result is NaN.
        const double nan_q = from_bits(b | kQuietNaNBit);
        if (iptr)
            *iptr = nan_q;
        return nan_q;
    }

    // Inf: integer part is ±inf, fractional is sign-preserving 0.
    if (is_inf_bits(b)) {
        if (iptr)
            *iptr = x;
        // SAFETY: mask out everything but sign; produces ±0.
        return from_bits(b & kSignMask);
    }

    const double i = trunc_bits(x);
    if (iptr)
        *iptr = i;
    // Same subtraction-required rationale as fract above.
    const double frac = sf64_sub(x, i);
    // C spec: modf preserves the sign of x in the fractional part, including
    // for zero fractional. If sub produced +0 and x is negative, flip sign.
    // SAFETY: bit_cast to check for zero and re-stamp sign bit.
    const uint64_t fb = bits_of(frac);
    if ((fb & ~kSignMask) == 0) {
        return from_bits((fb & ~kSignMask) | (b & kSignMask));
    }
    return frac;
}

// -----------------------------------------------------------------------------
// ldexp — scale x by 2^n
// -----------------------------------------------------------------------------
extern "C" double sf64_ldexp(double x, int n) {
    const uint64_t b = bits_of(x);

    // NaN → NaN (quieted).
    if (is_nan_bits(b)) {
        // SAFETY: quiet-bit OR onto NaN.
        return from_bits(b | kQuietNaNBit);
    }
    // ±inf or ±0 are fixed points.
    if (is_inf_bits(b) || is_zero_bits(b))
        return x;

    const uint32_t sign = extract_sign(b);

    // Clamp n aggressively so internal `int` arithmetic can't overflow.
    // The widest useful range is roughly [-2097, 1023] + some slack; anything
    // outside underflows to zero or overflows to inf.
    if (n > 2100)
        n = 2100;
    if (n < -2100)
        n = -2100;

    // Normalize subnormal inputs into a (virtual) normal form so we can add
    // to the exponent uniformly.
    uint64_t frac = extract_frac(b);
    int exp_unbiased; // unbiased exponent before scaling
    if (extract_exp(b) == 0) {
        // Subnormal: renormalize. Bit position of highest set frac bit gives
        // us the shift amount.
        const int lz = clz64(frac);              // 0..63; frac != 0 here.
        const int shift = lz - (63 - kFracBits); // shift to put leading 1 at bit 52
        // SAFETY: frac != 0 since subnormal && zero excluded; shift > 0.
        frac = (frac << shift) & kFracMask;
        exp_unbiased =
            1 - kExpBias - shift; // smallest normal exp is 1-bias; subnormal lowers further
    } else {
        exp_unbiased = static_cast<int>(extract_exp(b)) - kExpBias;
    }

    const int new_exp_unbiased = exp_unbiased + n;

    // Overflow → ±inf.
    if (new_exp_unbiased > 1023) {
        return make_signed_inf(sign);
    }

    // Normal range: stamp new biased exponent.
    if (new_exp_unbiased >= -1022) {
        const uint32_t new_exp_biased = static_cast<uint32_t>(new_exp_unbiased + kExpBias);
        return from_bits(pack(sign, new_exp_biased, frac));
    }

    // Subnormal range: need to right-shift the full mantissa (implicit + frac)
    // by (−1022 − new_exp_unbiased) with round-to-nearest-even.
    const int rshift = -1022 - new_exp_unbiased; // >= 1
    if (rshift >= 64) {
        // Underflow all the way; result is ±0.
        return make_signed_zero(sign);
    }
    const uint64_t full_mant = kImplicitBit | frac; // 53-bit normal mantissa.

    // Compute guard, round, sticky bits for round-to-nearest-even.
    // After the shift, the retained mantissa is `full_mant >> rshift`.
    // Bits shifted out: top bit (guard), next (round), rest OR'd into sticky.
    uint64_t retained;
    uint64_t shifted_out;
    if (rshift == 0) {
        retained = full_mant;
        shifted_out = 0;
    } else {
        retained = full_mant >> rshift;
        // Mask of the bits we're dropping.
        const uint64_t drop_mask = (uint64_t{1} << rshift) - 1;
        shifted_out = full_mant & drop_mask;
    }

    // Round-to-nearest-even: the top shifted-out bit is "round"; any bit
    // below is "sticky".
    uint64_t round_bit = 0;
    uint64_t sticky_bit = 0;
    if (rshift > 0) {
        round_bit = (shifted_out >> (rshift - 1)) & uint64_t{1};
        const uint64_t sticky_mask =
            (rshift >= 2) ? ((uint64_t{1} << (rshift - 1)) - 1) : uint64_t{0};
        sticky_bit = (shifted_out & sticky_mask) ? uint64_t{1} : uint64_t{0};
    }

    // If round=1 && (sticky=1 || retained_lsb=1) → round up.
    if (round_bit != 0 && (sticky_bit != 0 || (retained & 1u) != 0)) {
        retained = retained + 1;
    }

    // retained may have overflowed into bit 52 (implicit bit) → promote to
    // smallest normal.
    if ((retained & kImplicitBit) != 0) {
        // That means the result became the smallest normal, 2^-1022.
        return from_bits(pack(sign, 1u, retained & kFracMask));
    }
    // Subnormal: biased exp is 0.
    return from_bits(pack(sign, 0u, retained & kFracMask));
}

// -----------------------------------------------------------------------------
// frexp — split into (fraction in [0.5, 1), *exp)
// -----------------------------------------------------------------------------
extern "C" double sf64_frexp(double x, int* exp) {
    const uint64_t b = bits_of(x);

    // NaN/inf: *exp unspecified → set to 0 for determinism; return x quieted
    // or unchanged.
    if (is_nan_bits(b)) {
        if (exp)
            *exp = 0;
        // SAFETY: quiet-bit OR on NaN.
        return from_bits(b | kQuietNaNBit);
    }
    if (is_inf_bits(b)) {
        if (exp)
            *exp = 0;
        return x;
    }
    // ±0 → *exp = 0, return x unchanged.
    if (is_zero_bits(b)) {
        if (exp)
            *exp = 0;
        return x;
    }

    const uint32_t sign = extract_sign(b);
    uint64_t frac = extract_frac(b);
    int exp_unbiased;

    if (extract_exp(b) == 0) {
        // Subnormal: renormalize.
        const int lz = clz64(frac);
        const int shift = lz - (63 - kFracBits);
        // SAFETY: frac != 0 → shift > 0; left-shift within 64-bit width.
        frac = (frac << shift) & kFracMask;
        exp_unbiased = 1 - kExpBias - shift;
    } else {
        exp_unbiased = static_cast<int>(extract_exp(b)) - kExpBias;
    }

    // Produce a value in [0.5, 1): biased exp = 1022.
    if (exp)
        *exp = exp_unbiased + 1;
    return from_bits(pack(sign, static_cast<uint32_t>(kExpBias - 1), frac));
}

// -----------------------------------------------------------------------------
// ilogb — integer binary logarithm
// -----------------------------------------------------------------------------
extern "C" int sf64_ilogb(double x) {
    const uint64_t b = bits_of(x);

    if (is_zero_bits(b))
        return INT_MIN; // FP_ILOGB0
    if (is_nan_bits(b))
        return INT_MAX; // FP_ILOGBNAN
    if (is_inf_bits(b))
        return INT_MAX;

    const uint32_t e = extract_exp(b);
    if (e == 0) {
        // Subnormal: unbiased = 1 - bias - (leading_zeros_in_frac - (63 - 52))
        // = -1022 - (lz - 11). For denorm_min, frac=1, lz=63, so unbiased =
        // -1022 - 52 = -1074. Matches spec.
        const uint64_t frac = extract_frac(b);
        const int lz = clz64(frac);
        const int shift = lz - (63 - kFracBits);
        return 1 - kExpBias - shift;
    }
    return static_cast<int>(e) - kExpBias;
}

// -----------------------------------------------------------------------------
// logb — floating-point binary logarithm
// -----------------------------------------------------------------------------
extern "C" double sf64_logb(double x) {
    const uint64_t b = bits_of(x);

    if (is_nan_bits(b)) {
        // SAFETY: NaN → quieted NaN.
        return from_bits(b | kQuietNaNBit);
    }
    if (is_zero_bits(b)) {
        // -inf, sign preserved? Per C: logb(±0) = -inf, no sign on result.
        return from_bits(kNegativeInf);
    }
    if (is_inf_bits(b)) {
        // +inf, regardless of sign of x.
        return from_bits(kPositiveInf);
    }

    const int ilog = sf64_ilogb(x);
    // Convert int to fp64 via the ABI conversion routine. This keeps logb
    // honest about not using host FPU arithmetic.
    // Forward-declared locally to avoid adding a soft_f64.h dependency here.
    extern double sf64_from_i32(int32_t);
    return sf64_from_i32(ilog);
}
