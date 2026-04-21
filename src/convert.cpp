// Soft-fp64 conversion: f32/iN/uN <-> f64, full width matrix.
//
// Reference: Mesa `src/compiler/glsl/float64.glsl` — __fp32_to_fp64,
// __fp64_to_fp32, __int_to_fp64, __uint_to_fp64, __fp64_to_int, __fp64_to_uint.
// Width variants (i8/i16/i64/u8/u16/u64) extend trivially from the i32/u32
// cases by sign-extending or zero-extending to 64-bit first.
//
// CRITICAL: Apple6+ fp32 is FTZ (MSL §6.20). `soft_f64_from_f32` must read
// the operand via __builtin_bit_cast(uint32_t, x), NOT through an
// intermediate `float` variable — the latter loses subnormal payload when
// the compiler rematerializes the float value. See tests/test_convert_subnormal.cpp.
//
// f64 -> iN/uN must saturate at the integer type's bounds, not wrap. For NaN
// inputs, return 0 (matches LLVM fptosi.sat/fptoui.sat semantics).
//
// SPDX-License-Identifier: MIT

#include "internal.h"
#include "soft_fp64/soft_f64.h"

#include <climits>
#include <cstdint>

namespace {

using soft_fp64::internal::bits_of;
using soft_fp64::internal::clz32;
using soft_fp64::internal::clz64;
using soft_fp64::internal::extract_exp;
using soft_fp64::internal::extract_frac;
using soft_fp64::internal::extract_sign;
using soft_fp64::internal::from_bits;
using soft_fp64::internal::kExpBias;
using soft_fp64::internal::kExpMax;
using soft_fp64::internal::kFracBits;
using soft_fp64::internal::kFracMask;
using soft_fp64::internal::kImplicitBit;
using soft_fp64::internal::kQuietNaNBit;
using soft_fp64::internal::kSignMask;
using soft_fp64::internal::pack;

// ---- u64 -> f64 (core integer widening path) ----------------------------
//
// Takes a non-zero 64-bit magnitude, returns the correctly-rounded fp64 bit
// pattern (sign = 0). The caller applies the sign separately.
SF64_ALWAYS_INLINE uint64_t u64_magnitude_to_fp64_bits(uint64_t mag) noexcept {
    // Pre: mag != 0.
    //
    // Find the position of the leading 1. After a left-shift by `lz`, the
    // leading 1 sits at bit 63. We want it at bit 52 (the implicit bit),
    // so we shift right by (63 - 52) = 11. The bits discarded by the
    // right-shift form the round/sticky payload for round-to-nearest-even.
    const int lz = __builtin_clzll(mag); // SAFETY: mag is non-zero (caller checks).
    // Exponent of the leading 1 in the magnitude (position within a 64-bit word).
    const int msb_pos = 63 - lz; // in [0, 63]
    const uint32_t exp = static_cast<uint32_t>(msb_pos + kExpBias);

    uint64_t frac; // 52 bits, implicit bit excluded
    if (msb_pos <= kFracBits) {
        // Fits exactly — no rounding needed. Shift left so the leading 1
        // lands at bit 52; strip the implicit bit afterwards.
        const int shift = kFracBits - msb_pos;
        frac = (mag << shift) & kFracMask;
        return pack(0u, exp, frac);
    }

    // msb_pos in [53, 63] — we need to round.
    const int shift = msb_pos - kFracBits;  // in [1, 11]
    const uint64_t mantissa = mag >> shift; // 53 bits including implicit
    const uint64_t dropped_mask = (uint64_t{1} << shift) - 1u;
    const uint64_t dropped = mag & dropped_mask;
    const uint64_t halfway = uint64_t{1} << (shift - 1);

    // Round-to-nearest-even: round up if dropped > halfway, or (dropped == halfway
    // && mantissa is odd).
    uint64_t rounded = mantissa;
    if (dropped > halfway || (dropped == halfway && (mantissa & 1u))) {
        rounded += 1u;
    }

    // Rounding can overflow the mantissa (e.g., 0x1FFFFFFFFFFFFFFF rounds up
    // to 0x2000000000000000). If the 54th bit is set, exp += 1, frac = 0.
    if (rounded & (uint64_t{1} << (kFracBits + 1))) {
        // Overflow to next binade — leading implicit bit shifts to bit 53.
        const uint32_t new_exp = exp + 1u;
        return pack(0u, new_exp, 0u);
    }

    const uint64_t frac_out = rounded & kFracMask;
    return pack(0u, exp, frac_out);
}

// ---- signed 64-bit integer -> f64 ---------------------------------------
SF64_ALWAYS_INLINE double i64_to_fp64(int64_t x) noexcept {
    if (x == 0)
        return from_bits(0u);

    const uint64_t raw = static_cast<uint64_t>(x);
    const uint32_t sign = static_cast<uint32_t>(raw >> 63);

    // Compute magnitude. INT64_MIN is its own magnitude (0x8000...0000) when
    // interpreted as unsigned — the two's-complement negation is the same
    // bit pattern, so this falls out naturally.
    uint64_t mag;
    if (sign) {
        // Negation: -x = ~x + 1 (wraps for INT64_MIN, giving 0x80000...0 which
        // is the correct magnitude — the fp64 rounding path handles it).
        mag = static_cast<uint64_t>(0) - raw;
    } else {
        mag = raw;
    }

    uint64_t bits = u64_magnitude_to_fp64_bits(mag);
    // OR in the sign bit.
    bits |= (static_cast<uint64_t>(sign) << 63);
    return from_bits(bits);
}

// ---- unsigned 64-bit integer -> f64 -------------------------------------
SF64_ALWAYS_INLINE double u64_to_fp64(uint64_t x) noexcept {
    if (x == 0)
        return from_bits(0u);
    return from_bits(u64_magnitude_to_fp64_bits(x));
}

// ---- f64 -> u64 magnitude, saturating & truncating ----------------------
//
// Returns the magnitude of `x` truncated toward zero, saturated to
// [0, 2^64 - 1]. Sign is handled by the caller. NaN returns 0. +inf / overflow
// returns UINT64_MAX; negative overflow returns UINT64_MAX (caller maps to
// 0 for unsigned dest or INT64_MIN for signed dest).
//
// `too_large` is set when the truncated magnitude does not fit in 64 bits.
SF64_ALWAYS_INLINE uint64_t fp64_to_u64_magnitude(double x, bool* too_large,
                                                  bool* is_nan) noexcept {
    const uint64_t b = bits_of(x);
    const uint32_t exp = extract_exp(b);
    const uint64_t frac = extract_frac(b);

    *too_large = false;
    *is_nan = false;

    if (exp == kExpMax) {
        if (frac != 0) {
            *is_nan = true;
            return 0;
        }
        // Infinity: caller saturates.
        *too_large = true;
        return ~uint64_t{0};
    }

    // Unbiased exponent (value of MSB position in mantissa form).
    const int e = static_cast<int>(exp) - kExpBias;
    if (exp == 0 || e < 0) {
        // Zero, subnormal, or |x| < 1 — truncates to 0.
        return 0;
    }

    // Mantissa with implicit bit. 53 bits wide with MSB at bit 52.
    const uint64_t mant = frac | kImplicitBit;

    // Shift so that the integer part of the mantissa aligns with bit 0.
    // If e > 52, left-shift by (e - 52) — but only up to 11 shifts fit
    // without overflowing a 64-bit value (since the leading bit already
    // sits at bit 52, shifting by 11 lands it at bit 63). Any e > 63
    // means the integer part has 65+ bits → overflow.
    if (e > 63) {
        *too_large = true;
        return ~uint64_t{0};
    }
    if (e >= kFracBits) {
        const int shift = e - kFracBits; // in [0, 11]
        return mant << shift;
    }
    // e in [0, 51] — shift right, discarding fraction bits.
    const int shift = kFracBits - e; // in [1, 52]
    return mant >> shift;
}

// ---- f64 -> signed integer, saturating to [type_min, type_max] ----------
//
// `imin`/`imax` are the destination type bounds as signed 64-bit values.
SF64_ALWAYS_INLINE int64_t fp64_to_signed(double x, int64_t imin, int64_t imax) noexcept {
    const uint64_t b = bits_of(x);
    const uint32_t sign = extract_sign(b);

    bool too_large = false;
    bool is_nan = false;
    const uint64_t mag = fp64_to_u64_magnitude(x, &too_large, &is_nan);

    if (is_nan)
        return 0;

    if (too_large) {
        return sign ? imin : imax;
    }

    // mag fits in 64 bits. Compare against the signed destination bounds.
    if (sign) {
        // Negative: value is -mag. Need -mag >= imin  <=>  mag <= -imin.
        // For imin = INT64_MIN, -imin overflows — the magnitude (uint64)
        // cap is 2^63 which is representable as uint64. Special-case that.
        uint64_t neg_cap;
        if (imin == INT64_MIN) {
            neg_cap = (uint64_t{1} << 63);
        } else {
            neg_cap = static_cast<uint64_t>(-imin);
        }
        if (mag > neg_cap)
            return imin;
        // Now -mag is representable as int64_t (or equals INT64_MIN for mag == 2^63).
        if (mag == (uint64_t{1} << 63)) {
            // -(2^63) = INT64_MIN; wider types also accept this.
            return INT64_MIN;
        }
        return -static_cast<int64_t>(mag);
    }

    // Positive: value is +mag. Saturate against imax (>= 0).
    const uint64_t pos_cap = static_cast<uint64_t>(imax);
    if (mag > pos_cap)
        return imax;
    return static_cast<int64_t>(mag);
}

// ---- f64 -> unsigned integer, saturating to [0, type_max] ---------------
SF64_ALWAYS_INLINE uint64_t fp64_to_unsigned(double x, uint64_t umax) noexcept {
    const uint64_t b = bits_of(x);
    const uint32_t sign = extract_sign(b);

    bool too_large = false;
    bool is_nan = false;
    const uint64_t mag = fp64_to_u64_magnitude(x, &too_large, &is_nan);

    if (is_nan)
        return 0;

    if (sign) {
        // Any negative (including -inf) saturates to 0 (the minimum of an
        // unsigned destination).
        //
        // Exception: negative values with |x| < 1 also saturate to 0 — same
        // result, falls through below since mag == 0.
        return 0;
    }

    if (too_large)
        return umax;
    if (mag > umax)
        return umax;
    return mag;
}

} // namespace

// -------------------------------------------------------------------------
// f32 <-> f64
// -------------------------------------------------------------------------

extern "C" double sf64_from_f32(float x) {
    // SAFETY: read the fp32 operand's raw bit pattern without ever
    // materializing a `float` lvalue — on Apple6+ (MSL §6.20) fp32 is
    // flush-to-zero, so a rematerialized float would collapse subnormals
    // before we get to inspect them. __builtin_bit_cast is a pure type pun
    // (no arithmetic), so the integer payload survives.
    const uint32_t b = __builtin_bit_cast(uint32_t, x);

    const uint32_t sign = (b >> 31) & 1u;
    const uint32_t exp32 = (b >> 23) & 0xFFu;
    const uint32_t frac32 = b & 0x7FFFFFu;

    if (exp32 == 0) {
        if (frac32 == 0) {
            // +/-0
            return from_bits(static_cast<uint64_t>(sign) << 63);
        }
        // Subnormal: renormalize. `lz` is the count of leading zeros in the
        // 32-bit word holding the 23-bit fraction (9 of those zeros are
        // always present — they're the sign + exp field). The MSB of the
        // fraction sits at position `p = 31 - lz`, so the unbiased exponent
        // is `p - 149` and the f64 biased exponent is `p - 149 + 1023`
        // = 874 + p = 905 - lz.
        //
        // To land the implicit bit in the fp64 MSB of the mantissa, we
        // left-shift the 32-bit fraction by `(52 - p) = lz + 21` bits when
        // interpreted as a uint64 with the leading 1 going to bit 52. Since
        // we drop the implicit bit on pack, mask against kFracMask.
        const int lz = clz32(frac32); // SAFETY: frac32 != 0 here.
        const int p = 31 - lz;        // position of leading 1 (0..22)
        const uint32_t f64_exp = static_cast<uint32_t>(905 - lz);
        // Left-shift to put the leading 1 at bit 52.
        const int left = 52 - p; // in [30, 52]
        const uint64_t f64_frac = (static_cast<uint64_t>(frac32) << left) & kFracMask;
        return from_bits(pack(sign, f64_exp, f64_frac));
    }

    if (exp32 == 0xFFu) {
        if (frac32 == 0) {
            // +/-inf
            return from_bits(pack(sign, kExpMax, 0));
        }
        // NaN — preserve payload and force quiet bit.
        const uint64_t payload = static_cast<uint64_t>(frac32) << 29;
        return from_bits(pack(sign, kExpMax, payload | kQuietNaNBit));
    }

    // Normal: rebias exponent (f32 bias 127 -> f64 bias 1023).
    const uint32_t f64_exp = exp32 + (kExpBias - 127);
    const uint64_t f64_frac = static_cast<uint64_t>(frac32) << 29;
    return from_bits(pack(sign, f64_exp, f64_frac));
}

extern "C" float sf64_to_f32(double x) {
    const uint64_t b = bits_of(x);
    const uint32_t sign = extract_sign(b);
    const uint32_t exp64 = extract_exp(b);
    const uint64_t frac64 = extract_frac(b);

    if (exp64 == kExpMax) {
        if (frac64 == 0) {
            // inf
            const uint32_t out = (sign << 31) | (0xFFu << 23);
            // SAFETY: bit-pattern construction; bit_cast to float is pure type pun.
            return __builtin_bit_cast(float, out);
        }
        // NaN — propagate high payload bits, force quiet bit (bit 22 of f32).
        const uint32_t payload = static_cast<uint32_t>(frac64 >> 29) & 0x7FFFFFu;
        const uint32_t out = (sign << 31) | (0xFFu << 23) | payload | 0x400000u;
        // SAFETY: NaN bit pattern; bit_cast is a type pun, not arithmetic.
        return __builtin_bit_cast(float, out);
    }

    if (exp64 == 0) {
        // Zero or f64 subnormal — all f64 subnormals are well below f32's
        // smallest subnormal, so they flush to +/-0 in f32.
        const uint32_t out = sign << 31;
        // SAFETY: signed zero bit pattern.
        return __builtin_bit_cast(float, out);
    }

    // Normal f64. Rebase exponent: f64_exp - 1023 + 127 = f64_exp - 896.
    const int unbiased = static_cast<int>(exp64) - kExpBias;
    const int new_exp = unbiased + 127;

    if (new_exp >= 0xFF) {
        // Overflow -> +/- inf.
        const uint32_t out = (sign << 31) | (0xFFu << 23);
        // SAFETY: infinity bit pattern.
        return __builtin_bit_cast(float, out);
    }

    // Mantissa with implicit bit (53 bits, MSB at 52).
    const uint64_t mant = frac64 | kImplicitBit;

    if (new_exp > 0) {
        // Normal f32 result. Keep top 24 bits (implicit + 23), round the
        // remaining 29 via round-to-nearest-even.
        const uint64_t guard_bit = uint64_t{1} << 28;
        const uint64_t round_mask = (uint64_t{1} << 29) - 1u;
        const uint64_t top24 = mant >> 29;          // 24 bits: implicit + 23 frac
        const uint64_t dropped = mant & round_mask; // 29 bits

        uint64_t rounded = top24;
        if (dropped > guard_bit || (dropped == guard_bit && (top24 & 1u))) {
            rounded += 1u;
        }

        uint32_t exp_out = static_cast<uint32_t>(new_exp);
        uint32_t frac_out;
        if (rounded & (uint64_t{1} << 24)) {
            // Mantissa overflowed past 24 bits -> exponent bump.
            exp_out += 1u;
            if (exp_out >= 0xFFu) {
                // Round-to-inf.
                const uint32_t inf_bits = (sign << 31) | (0xFFu << 23);
                // SAFETY: infinity bit pattern.
                return __builtin_bit_cast(float, inf_bits);
            }
            frac_out = 0; // rounded == 0x1000000; implicit bit shifts up.
        } else {
            frac_out = static_cast<uint32_t>(rounded) & 0x7FFFFFu;
        }
        const uint32_t out = (sign << 31) | (exp_out << 23) | frac_out;
        // SAFETY: assembled f32 bit pattern from components.
        return __builtin_bit_cast(float, out);
    }

    // new_exp <= 0 — subnormal or underflow. The true exponent is
    // (new_exp - 127) relative to f32; we need to produce a f32 subnormal
    // whose implicit bit is encoded as a shifted fraction with exp_field=0.
    //
    // Total right-shift from the 53-bit mant to the 23-bit subnormal frac:
    //   normal case: 29 (drop 29 low bits)
    //   each step new_exp < 1 costs another bit of shift
    //   shift = 29 + (1 - new_exp) = 30 - new_exp
    const int shift = 30 - new_exp;

    if (shift > 53) {
        // Entire 53-bit mantissa sits strictly below the guard position
        // (at bit shift-1 ≥ 53). Value magnitude is < denorm_min/2, so
        // round-to-nearest-even sends it to ±0. Also avoids the UB that
        // `uint64_t{1} << shift` triggers when shift ≥ 64.
        const uint32_t out = sign << 31;
        // SAFETY: signed-zero bit pattern.
        return __builtin_bit_cast(float, out);
    }

    const uint64_t guard_bit = uint64_t{1} << (shift - 1);
    const uint64_t round_mask = (uint64_t{1} << shift) - 1u;
    const uint64_t top = mant >> shift;
    const uint64_t dropped = mant & round_mask;

    uint64_t rounded = top;
    if (dropped > guard_bit || (dropped == guard_bit && (top & 1u))) {
        rounded += 1u;
    }

    if (rounded & (uint64_t{1} << 23)) {
        // Rounded up past subnormal range -> smallest normal.
        const uint32_t out = (sign << 31) | (1u << 23);
        // SAFETY: f32 smallest-normal bit pattern.
        return __builtin_bit_cast(float, out);
    }

    const uint32_t frac_out = static_cast<uint32_t>(rounded) & 0x7FFFFFu;
    const uint32_t out = (sign << 31) | frac_out;
    // SAFETY: f32 subnormal bit pattern.
    return __builtin_bit_cast(float, out);
}

// -------------------------------------------------------------------------
// intN/uintN -> f64 (widen then one common path)
// -------------------------------------------------------------------------

extern "C" double sf64_from_i8(int8_t x) {
    return i64_to_fp64(static_cast<int64_t>(x));
}
extern "C" double sf64_from_i16(int16_t x) {
    return i64_to_fp64(static_cast<int64_t>(x));
}
extern "C" double sf64_from_i32(int32_t x) {
    return i64_to_fp64(static_cast<int64_t>(x));
}
extern "C" double sf64_from_i64(int64_t x) {
    return i64_to_fp64(x);
}

extern "C" double sf64_from_u8(uint8_t x) {
    return u64_to_fp64(static_cast<uint64_t>(x));
}
extern "C" double sf64_from_u16(uint16_t x) {
    return u64_to_fp64(static_cast<uint64_t>(x));
}
extern "C" double sf64_from_u32(uint32_t x) {
    return u64_to_fp64(static_cast<uint64_t>(x));
}
extern "C" double sf64_from_u64(uint64_t x) {
    return u64_to_fp64(x);
}

// -------------------------------------------------------------------------
// f64 -> intN/uintN (saturating, truncating, NaN -> 0)
// -------------------------------------------------------------------------

extern "C" int8_t sf64_to_i8(double x) {
    return static_cast<int8_t>(fp64_to_signed(x, INT8_MIN, INT8_MAX));
}
extern "C" int16_t sf64_to_i16(double x) {
    return static_cast<int16_t>(fp64_to_signed(x, INT16_MIN, INT16_MAX));
}
extern "C" int32_t sf64_to_i32(double x) {
    return static_cast<int32_t>(fp64_to_signed(x, INT32_MIN, INT32_MAX));
}
extern "C" int64_t sf64_to_i64(double x) {
    return fp64_to_signed(x, INT64_MIN, INT64_MAX);
}

extern "C" uint8_t sf64_to_u8(double x) {
    return static_cast<uint8_t>(fp64_to_unsigned(x, UINT8_MAX));
}
extern "C" uint16_t sf64_to_u16(double x) {
    return static_cast<uint16_t>(fp64_to_unsigned(x, UINT16_MAX));
}
extern "C" uint32_t sf64_to_u32(double x) {
    return static_cast<uint32_t>(fp64_to_unsigned(x, UINT32_MAX));
}
extern "C" uint64_t sf64_to_u64(double x) {
    return fp64_to_unsigned(x, UINT64_MAX);
}
