#pragma once

// Shared test helpers: bit-cast, edge-case corpus, assertion macros.
// Included by every tests/test_*.cpp TU.
//
// SPDX-License-Identifier: MIT

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace host_oracle {

// ---- bit-cast -----------------------------------------------------------

inline uint64_t bits(double x) {
    uint64_t b;
    std::memcpy(&b, &x, sizeof(b));
    return b;
}

inline double from_bits(uint64_t b) {
    double d;
    std::memcpy(&d, &b, sizeof(d));
    return d;
}

inline uint32_t bits_f32(float x) {
    uint32_t b;
    std::memcpy(&b, &x, sizeof(b));
    return b;
}

inline float f32_from_bits(uint32_t b) {
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
}

// ---- canonical edge-case corpus ----------------------------------------

inline const std::array<double, 21>& edge_cases_f64() {
    static const std::array<double, 21> kValues = {
        +0.0,
        -0.0,
        +1.0,
        -1.0,
        +2.0,
        -2.0,
        0.5,
        1.5,
        std::numeric_limits<double>::min(), // smallest normal
        -std::numeric_limits<double>::min(),
        std::numeric_limits<double>::denorm_min(), // smallest subnormal
        -std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::max(),
        -std::numeric_limits<double>::max(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN(),
        -std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::epsilon(),
        3.14159265358979323846,
        2.71828182845904523536,
    };
    return kValues;
}

inline const std::array<float, 17>& edge_cases_f32() {
    static const std::array<float, 17> kValues = {
        +0.0f,
        -0.0f,
        +1.0f,
        -1.0f,
        0.5f,
        1.5f,
        std::numeric_limits<float>::min(),
        -std::numeric_limits<float>::min(),
        std::numeric_limits<float>::denorm_min(),
        -std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        3.14159265f,
        2.71828183f,
    };
    return kValues;
}

// ---- assertion helpers --------------------------------------------------

// Compare doubles bit-exact. NaN payloads can legitimately differ (sign
// bit, payload bits); treat any quiet NaN result as matching any quiet NaN
// expected.
inline bool equal_exact_or_nan(double got, double expect) {
    if (std::isnan(got) && std::isnan(expect))
        return true;
    return bits(got) == bits(expect);
}

#define SF64_CHECK(cond)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond);                  \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

#define SF64_CHECK_BITS(got, expect)                                                               \
    do {                                                                                           \
        const double _got = (got);                                                                 \
        const double _expect = (expect);                                                           \
        if (!::host_oracle::equal_exact_or_nan(_got, _expect)) {                                   \
            std::fprintf(stderr, "FAIL: %s:%d: got=%a (0x%016llx) expect=%a (0x%016llx)\n",        \
                         __FILE__, __LINE__, _got, (unsigned long long)::host_oracle::bits(_got),  \
                         _expect, (unsigned long long)::host_oracle::bits(_expect));               \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

// Deterministic PRNG — LCG so every test run covers the same corpus.
class LCG {
  public:
    explicit LCG(uint64_t seed = 0xDEADBEEFCAFEBABEULL) : state_(seed) {}
    uint64_t next() {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return state_;
    }
    double next_double_any() {
        // Uniformly distributed over all 2^64 bit patterns — stresses NaN /
        // inf / subnormal paths.
        return from_bits(next());
    }
    double next_double_normal() {
        // Limited to finite normal values in [-1e100, 1e100].
        while (true) {
            double d = from_bits(next());
            if (std::isfinite(d) && std::fabs(d) < 1e100)
                return d;
        }
    }

  private:
    uint64_t state_;
};

} // namespace host_oracle
