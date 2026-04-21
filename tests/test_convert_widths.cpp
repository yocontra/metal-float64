// Full width matrix of iN/uN <-> f64 conversions, plus saturation/NaN
// handling on the f64 -> integer direction.
//
// SPDX-License-Identifier: MIT

#include "host_oracle.h"
#include "soft_fp64/soft_f64.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

template <typename T> double call_from(T x) {
    if constexpr (std::is_same_v<T, int8_t>)
        return sf64_from_i8(x);
    else if constexpr (std::is_same_v<T, int16_t>)
        return sf64_from_i16(x);
    else if constexpr (std::is_same_v<T, int32_t>)
        return sf64_from_i32(x);
    else if constexpr (std::is_same_v<T, int64_t>)
        return sf64_from_i64(x);
    else if constexpr (std::is_same_v<T, uint8_t>)
        return sf64_from_u8(x);
    else if constexpr (std::is_same_v<T, uint16_t>)
        return sf64_from_u16(x);
    else if constexpr (std::is_same_v<T, uint32_t>)
        return sf64_from_u32(x);
    else if constexpr (std::is_same_v<T, uint64_t>)
        return sf64_from_u64(x);
}

template <typename T> void check_from_exact(T x) {
    const double got = call_from(x);
    const double expect = static_cast<double>(x);
    SF64_CHECK_BITS(got, expect);
}

} // namespace

int main() {
    using namespace host_oracle;

    // ---- boundary integer values for each width --------------------------
    check_from_exact<int8_t>(INT8_MIN);
    check_from_exact<int8_t>(INT8_MAX);
    check_from_exact<int8_t>(0);
    check_from_exact<int8_t>(-1);
    check_from_exact<int8_t>(1);

    check_from_exact<int16_t>(INT16_MIN);
    check_from_exact<int16_t>(INT16_MAX);
    check_from_exact<int16_t>(0);
    check_from_exact<int16_t>(-1);

    check_from_exact<int32_t>(INT32_MIN);
    check_from_exact<int32_t>(INT32_MAX);
    check_from_exact<int32_t>(0);
    check_from_exact<int32_t>(-1);
    check_from_exact<int32_t>(1 << 30);

    check_from_exact<int64_t>(INT64_MIN);
    check_from_exact<int64_t>(INT64_MAX);
    check_from_exact<int64_t>(0);
    check_from_exact<int64_t>(-1);
    check_from_exact<int64_t>(int64_t{1} << 62);
    check_from_exact<int64_t>(-(int64_t{1} << 62));
    // Values requiring round-to-nearest-even (> 2^53): ensure we match the
    // host FPU's rounding.
    check_from_exact<int64_t>((int64_t{1} << 53) + 1); // exact
    check_from_exact<int64_t>((int64_t{1} << 53) + 3); // needs rounding
    check_from_exact<int64_t>((int64_t{1} << 60) + 1);
    check_from_exact<int64_t>(INT64_MAX - 1);

    check_from_exact<uint8_t>(0);
    check_from_exact<uint8_t>(UINT8_MAX);
    check_from_exact<uint16_t>(0);
    check_from_exact<uint16_t>(UINT16_MAX);
    check_from_exact<uint32_t>(0);
    check_from_exact<uint32_t>(UINT32_MAX);
    check_from_exact<uint64_t>(0);
    check_from_exact<uint64_t>(UINT64_MAX);
    check_from_exact<uint64_t>(uint64_t{1} << 53);
    check_from_exact<uint64_t>((uint64_t{1} << 53) + 1); // rounds
    check_from_exact<uint64_t>(UINT64_MAX - 1);

    // ---- to_iN / to_uN saturation and NaN --------------------------------
    // Overflow to +inf direction saturates to type_max.
    SF64_CHECK(sf64_to_i8(1e100) == INT8_MAX);
    SF64_CHECK(sf64_to_i8(-1e100) == INT8_MIN);
    SF64_CHECK(sf64_to_i16(1e100) == INT16_MAX);
    SF64_CHECK(sf64_to_i16(-1e100) == INT16_MIN);
    SF64_CHECK(sf64_to_i32(1e100) == INT32_MAX);
    SF64_CHECK(sf64_to_i32(-1e100) == INT32_MIN);
    SF64_CHECK(sf64_to_i64(1e100) == INT64_MAX);
    SF64_CHECK(sf64_to_i64(-1e100) == INT64_MIN);

    // +/-inf saturates the same way.
    const double pinf = std::numeric_limits<double>::infinity();
    const double ninf = -std::numeric_limits<double>::infinity();
    SF64_CHECK(sf64_to_i32(pinf) == INT32_MAX);
    SF64_CHECK(sf64_to_i32(ninf) == INT32_MIN);
    SF64_CHECK(sf64_to_i64(pinf) == INT64_MAX);
    SF64_CHECK(sf64_to_i64(ninf) == INT64_MIN);
    SF64_CHECK(sf64_to_u32(pinf) == UINT32_MAX);
    SF64_CHECK(sf64_to_u32(ninf) == 0u);
    SF64_CHECK(sf64_to_u64(pinf) == UINT64_MAX);
    SF64_CHECK(sf64_to_u64(ninf) == 0u);

    // NaN -> 0 for every destination.
    const double qnan = std::nan("");
    SF64_CHECK(sf64_to_i8(qnan) == 0);
    SF64_CHECK(sf64_to_i16(qnan) == 0);
    SF64_CHECK(sf64_to_i32(qnan) == 0);
    SF64_CHECK(sf64_to_i64(qnan) == 0);
    SF64_CHECK(sf64_to_u8(qnan) == 0u);
    SF64_CHECK(sf64_to_u16(qnan) == 0u);
    SF64_CHECK(sf64_to_u32(qnan) == 0u);
    SF64_CHECK(sf64_to_u64(qnan) == 0u);

    // Unsigned destination: negative inputs saturate to 0.
    SF64_CHECK(sf64_to_u8(-1.0) == 0u);
    SF64_CHECK(sf64_to_u16(-1.0) == 0u);
    SF64_CHECK(sf64_to_u32(-1.0) == 0u);
    SF64_CHECK(sf64_to_u64(-1.0) == 0u);
    SF64_CHECK(sf64_to_u32(1e100) == UINT32_MAX);
    SF64_CHECK(sf64_to_u64(1e100) == UINT64_MAX);

    // Truncation toward zero.
    SF64_CHECK(sf64_to_i32(3.9) == 3);
    SF64_CHECK(sf64_to_i32(-3.9) == -3);
    SF64_CHECK(sf64_to_i32(0.9) == 0);
    SF64_CHECK(sf64_to_i32(-0.9) == 0);
    SF64_CHECK(sf64_to_i32(-0.0) == 0);
    SF64_CHECK(sf64_to_i64(123456789.9) == int64_t{123456789});
    SF64_CHECK(sf64_to_i64(-123456789.9) == int64_t{-123456789});

    // Boundary rounding: exact type_max should not saturate away.
    SF64_CHECK(sf64_to_i32(static_cast<double>(INT32_MAX)) == INT32_MAX);
    SF64_CHECK(sf64_to_i32(static_cast<double>(INT32_MIN)) == INT32_MIN);
    // INT32_MAX + 1 as a double overflows.
    SF64_CHECK(sf64_to_i32(2147483648.0) == INT32_MAX);
    SF64_CHECK(sf64_to_i32(-2147483649.0) == INT32_MIN);

    // INT64_MAX: (double)INT64_MAX rounds up to 2^63, which overflows signed
    // int64 — saturation should return INT64_MAX, which matches both LLVM
    // fptosi.sat and the operand's exact pre-rounded value.
    SF64_CHECK(sf64_to_i64(static_cast<double>(INT64_MIN)) == INT64_MIN);
    SF64_CHECK(sf64_to_i64(static_cast<double>(INT64_MAX)) == INT64_MAX);
    SF64_CHECK(sf64_to_u64(static_cast<double>(UINT64_MAX)) == UINT64_MAX);

    // A value slightly above INT64_MAX (exactly 2^63) saturates.
    SF64_CHECK(sf64_to_i64(9223372036854775808.0) == INT64_MAX);
    // A value slightly above UINT64_MAX (exactly 2^64) saturates.
    SF64_CHECK(sf64_to_u64(18446744073709551616.0) == UINT64_MAX);

    // ---- random round-trip for normal integer values --------------------
    // For |x| < 2^53, i64 -> f64 is exact; round-trip must be identity via
    // host's static_cast<double>(x) oracle.
    LCG rng;
    for (int i = 0; i < 10000; ++i) {
        const int64_t x = static_cast<int64_t>(rng.next());
        const double d = sf64_from_i64(x);
        SF64_CHECK_BITS(d, static_cast<double>(x));
    }
    for (int i = 0; i < 10000; ++i) {
        const uint64_t x = rng.next();
        const double d = sf64_from_u64(x);
        SF64_CHECK_BITS(d, static_cast<double>(x));
    }
    // Small widths — widen first, exact.
    for (int i = 0; i < 5000; ++i) {
        const int32_t x = static_cast<int32_t>(rng.next());
        const double d32 = sf64_from_i32(x);
        SF64_CHECK_BITS(d32, static_cast<double>(x));

        const uint32_t ux = static_cast<uint32_t>(rng.next());
        const double du = sf64_from_u32(ux);
        SF64_CHECK_BITS(du, static_cast<double>(ux));

        const int16_t s16 = static_cast<int16_t>(rng.next());
        SF64_CHECK_BITS(sf64_from_i16(s16), static_cast<double>(s16));

        const int8_t s8 = static_cast<int8_t>(rng.next());
        SF64_CHECK_BITS(sf64_from_i8(s8), static_cast<double>(s8));
    }

    return 0;
}
