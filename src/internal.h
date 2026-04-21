#pragma once

// Internal bit helpers shared across every soft-fp64 translation unit.
// Header-only, all inline, no state. Consumers:
//
//   arithmetic.cpp   add/sub/mul/div/rem/neg
//   compare.cpp      fcmp (16 predicates), fmin_precise, fmax_precise
//   convert.cpp      iN/uN/f32 <-> f64
//   sqrt_fma.cpp     sqrt, rsqrt, fma
//   rounding.cpp     floor/ceil/trunc/round/rint/fract/modf/ldexp/frexp/ilogb/logb
//   classify.cpp     isnan/isinf/fabs/copysign/hypot/nextafter/etc.
//
// Algorithms below mirror Mesa's `src/compiler/glsl/float64.glsl` extract/pack
// routines, simplified to use native 64-bit integer arithmetic (MSL supports
// `ulong`, so Mesa's GLSL-era u32-pair tricks are unnecessary).
//
// SPDX-License-Identifier: MIT

#include "soft_fp64/defines.h"

#include <cstdint>

namespace soft_fp64::internal {

// ---- IEEE-754 binary64 constants ----------------------------------------

inline constexpr uint64_t kSignMask = 0x8000000000000000ULL;
inline constexpr uint64_t kExpMask = 0x7FF0000000000000ULL;
inline constexpr uint64_t kFracMask = 0x000FFFFFFFFFFFFFULL;
inline constexpr uint64_t kImplicitBit = 0x0010000000000000ULL; // bit 52
inline constexpr uint64_t kQuietNaNBit = 0x0008000000000000ULL; // bit 51
inline constexpr uint64_t kPositiveInf = 0x7FF0000000000000ULL;
inline constexpr uint64_t kNegativeInf = 0xFFF0000000000000ULL;
inline constexpr uint64_t kCanonicalNaN = 0x7FF8000000000000ULL; // quiet, sign=0

inline constexpr int kExpBias = 1023;
inline constexpr int kExpMax = 0x7FF; // all ones
inline constexpr int kFracBits = 52;
inline constexpr int kMantissaW = kFracBits + 1; // 53 with implicit

// ---- bit-casts ----------------------------------------------------------

SF64_ALWAYS_INLINE uint64_t bits_of(double x) noexcept {
    return __builtin_bit_cast(uint64_t, x);
}

SF64_ALWAYS_INLINE double from_bits(uint64_t b) noexcept {
    return __builtin_bit_cast(double, b);
}

SF64_ALWAYS_INLINE uint32_t bits_of(float x) noexcept {
    return __builtin_bit_cast(uint32_t, x);
}

SF64_ALWAYS_INLINE float f32_from_bits(uint32_t b) noexcept {
    return __builtin_bit_cast(float, b);
}

// ---- field extraction ---------------------------------------------------

// Biased exponent field (0..2047). 0 = zero/subnormal, 2047 = inf/NaN.
SF64_ALWAYS_INLINE uint32_t extract_exp(uint64_t b) noexcept {
    return static_cast<uint32_t>((b >> 52) & 0x7FF);
}

SF64_ALWAYS_INLINE uint64_t extract_frac(uint64_t b) noexcept {
    return b & kFracMask;
}

SF64_ALWAYS_INLINE uint32_t extract_sign(uint64_t b) noexcept {
    return static_cast<uint32_t>(b >> 63);
}

// Pack sign/exp/frac into an IEEE-754 binary64. Assumes:
//   sign in {0, 1}
//   exp in [0, 2047] (biased)
//   frac in [0, (1<<52)-1]  (implicit bit NOT present)
SF64_ALWAYS_INLINE uint64_t pack(uint32_t sign, uint32_t exp, uint64_t frac) noexcept {
    return (static_cast<uint64_t>(sign & 1u) << 63) | (static_cast<uint64_t>(exp & 0x7FFu) << 52) |
           (frac & kFracMask);
}

// ---- classification (duplicated here so every TU is standalone) ---------

SF64_ALWAYS_INLINE bool is_nan_bits(uint64_t b) noexcept {
    return extract_exp(b) == kExpMax && extract_frac(b) != 0;
}

SF64_ALWAYS_INLINE bool is_inf_bits(uint64_t b) noexcept {
    return extract_exp(b) == kExpMax && extract_frac(b) == 0;
}

SF64_ALWAYS_INLINE bool is_zero_bits(uint64_t b) noexcept {
    return (b & ~kSignMask) == 0;
}

SF64_ALWAYS_INLINE bool is_subnormal_bits(uint64_t b) noexcept {
    return extract_exp(b) == 0 && extract_frac(b) != 0;
}

SF64_ALWAYS_INLINE bool is_finite_bits(uint64_t b) noexcept {
    return extract_exp(b) != kExpMax;
}

// ---- shift helpers (Mesa float64.glsl `shift64RightJamming`) ------------

// Right-shift with "jamming": OR all shifted-out bits into the LSB. Used in
// add/sub to track the sticky bit for round-to-nearest-even.
SF64_ALWAYS_INLINE uint64_t shift_right_jamming(uint64_t x, int count) noexcept {
    if (count == 0)
        return x;
    if (count >= 64)
        return (x != 0) ? 1u : 0u;
    const uint64_t mask = (uint64_t{1} << count) - 1;
    return (x >> count) | ((x & mask) != 0 ? 1u : 0u);
}

// Left-shift by `count`, saturating at 63 bits. `count` must be in [0, 63].
SF64_ALWAYS_INLINE uint64_t shift_left(uint64_t x, int count) noexcept {
    return x << count;
}

// Count leading zeros on a 64-bit value. Returns 64 for x=0.
SF64_ALWAYS_INLINE int clz64(uint64_t x) noexcept {
    return x == 0 ? 64 : __builtin_clzll(x);
}

// Count leading zeros on a 32-bit value. Returns 32 for x=0.
SF64_ALWAYS_INLINE int clz32(uint32_t x) noexcept {
    return x == 0 ? 32 : __builtin_clz(x);
}

// ---- canonical result builders ------------------------------------------

SF64_ALWAYS_INLINE double make_signed_zero(uint32_t sign) noexcept {
    return from_bits(static_cast<uint64_t>(sign & 1u) << 63);
}

SF64_ALWAYS_INLINE double make_signed_inf(uint32_t sign) noexcept {
    return from_bits(pack(sign, kExpMax, 0));
}

SF64_ALWAYS_INLINE double propagate_nan(uint64_t a_bits, uint64_t b_bits) noexcept {
    // Per IEEE 754-2008 §6.2.3: prefer the NaN of the signalling operand,
    // quiet the result. For a pair of quiet NaNs, pick `a`.
    const bool a_is_nan = is_nan_bits(a_bits);
    const bool b_is_nan = is_nan_bits(b_bits);
    if (a_is_nan && b_is_nan) {
        // Both NaN; prefer a, quieted.
        return from_bits(a_bits | kQuietNaNBit);
    }
    if (a_is_nan)
        return from_bits(a_bits | kQuietNaNBit);
    if (b_is_nan)
        return from_bits(b_bits | kQuietNaNBit);
    return from_bits(kCanonicalNaN);
}

SF64_ALWAYS_INLINE double canonical_nan() noexcept {
    return from_bits(kCanonicalNaN);
}

} // namespace soft_fp64::internal
