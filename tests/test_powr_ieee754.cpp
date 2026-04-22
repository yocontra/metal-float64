// Bit-exact IEEE 754-2019 §9.2.1 conformance for sf64_powr.
//
// Origin: closed after an independent review of origin/main found that
// sf64_powr was just `if (x<0) return qNaN; return sf64_pow(x, y);` —
// which inherits pow's degenerate-case returns (1 for (±0,±0), (+∞,0),
// (1,±∞), (NaN,0)). IEEE 754-2019 §9.2.1 requires qNaN for every one
// of those and +∞ with DIVBYZERO for the pole at (±0, y<0). The
// existing MPFR sweep (`pow x∈[1e-6,1e6], y∈[-50,50]`) never reaches
// the boundaries so the bug was invisible to CI.
//
// This test is bit-exact — every case has a fixed expected bit pattern
// drawn from §9.2.1, not an MPFR-derived ULP bound. No tier, no ULP
// tolerance.
//
// SPDX-License-Identifier: MIT

#include "soft_fp64/soft_f64.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

constexpr uint64_t kPosInf = 0x7FF0000000000000ULL;
constexpr uint64_t kNegInf = 0xFFF0000000000000ULL;
constexpr uint64_t kPosZero = 0x0000000000000000ULL;
constexpr uint64_t kNegZero = 0x8000000000000000ULL;
constexpr uint64_t kOne = 0x3FF0000000000000ULL;    // +1.0
constexpr uint64_t kTwo = 0x4000000000000000ULL;    // +2.0
constexpr uint64_t kHalf = 0x3FE0000000000000ULL;   // +0.5
constexpr uint64_t kNegOne = 0xBFF0000000000000ULL; // -1.0
constexpr uint64_t kCanonicalNaN = 0x7FF8000000000000ULL;

double from_bits(uint64_t b) noexcept {
    double d;
    std::memcpy(&d, &b, sizeof(d));
    return d;
}

uint64_t bits_of(double d) noexcept {
    uint64_t b;
    std::memcpy(&b, &d, sizeof(b));
    return b;
}

// Result class accepted by each case. "qNaN" means any bit pattern
// whose exponent is all-ones and whose mantissa MSB is set (the quiet
// bit). Exact NaN payload is not constrained — the library does not
// yet preserve sNaN payloads (tracked for 1.2+).
enum class Expect { QNaN, PosInf, PosZero, NegZero, ExactBits };

struct Case {
    uint64_t x_bits;
    uint64_t y_bits;
    Expect kind;
    uint64_t exact_bits; // used only when kind == ExactBits
    const char* why;
};

bool is_qnan(uint64_t b) {
    const uint64_t exp = (b >> 52) & 0x7FF;
    const uint64_t frac = b & 0x000FFFFFFFFFFFFFULL;
    const uint64_t quiet_bit = 0x0008000000000000ULL;
    return exp == 0x7FF && (frac & quiet_bit) != 0;
}

bool check(const Case& c, double got) {
    const uint64_t gb = bits_of(got);
    switch (c.kind) {
    case Expect::QNaN:
        return is_qnan(gb);
    case Expect::PosInf:
        return gb == kPosInf;
    case Expect::PosZero:
        return gb == kPosZero;
    case Expect::NegZero:
        return gb == kNegZero;
    case Expect::ExactBits:
        return gb == c.exact_bits;
    }
    return false;
}

const char* expect_name(const Case& c) {
    switch (c.kind) {
    case Expect::QNaN:
        return "qNaN";
    case Expect::PosInf:
        return "+inf";
    case Expect::PosZero:
        return "+0";
    case Expect::NegZero:
        return "-0";
    case Expect::ExactBits:
        return "<exact>";
    }
    return "?";
}

// IEEE 754-2019 §9.2.1 Table 9.1 for powr(x, y). Every row has a
// written "why" referencing the clause; no magic-constant targets.
const Case kCases[] = {
    // --- NaN propagation --------------------------------------------
    {kCanonicalNaN, kPosZero, Expect::QNaN, 0,
     "powr(NaN, 0) = qNaN — reviewer cited this exact case"},
    {kCanonicalNaN, kOne, Expect::QNaN, 0, "powr(NaN, 1) = qNaN"},
    {kCanonicalNaN, kPosInf, Expect::QNaN, 0, "powr(NaN, +inf) = qNaN"},
    {kOne, kCanonicalNaN, Expect::QNaN, 0, "powr(1, NaN) = qNaN"},
    {kTwo, kCanonicalNaN, Expect::QNaN, 0, "powr(2, NaN) = qNaN"},
    {kPosInf, kCanonicalNaN, Expect::QNaN, 0, "powr(+inf, NaN) = qNaN"},

    // --- x < 0 ------------------------------------------------------
    {kNegOne, kTwo, Expect::QNaN, 0, "powr(-1, 2) = qNaN — §9.2.1 'x < 0 → NaN'"},
    {kNegOne, kOne, Expect::QNaN, 0, "powr(-1, 1) = qNaN"},
    {bits_of(-0.5), kTwo, Expect::QNaN, 0, "powr(-0.5, 2) = qNaN"},
    {kNegInf, kOne, Expect::QNaN, 0, "powr(-inf, 1) = qNaN — -inf is x<0"},

    // --- (0 or ∞)^0 → qNaN  (the reviewer's primary bug set) --------
    {kPosZero, kPosZero, Expect::QNaN, 0, "powr(+0, +0) = qNaN"},
    {kPosZero, kNegZero, Expect::QNaN, 0, "powr(+0, -0) = qNaN"},
    {kNegZero, kPosZero, Expect::QNaN, 0, "powr(-0, +0) = qNaN — -0 also triggers 0^0"},
    {kNegZero, kNegZero, Expect::QNaN, 0, "powr(-0, -0) = qNaN"},
    {kPosInf, kPosZero, Expect::QNaN, 0, "powr(+inf, +0) = qNaN"},
    {kPosInf, kNegZero, Expect::QNaN, 0, "powr(+inf, -0) = qNaN"},

    // --- 1^±∞ → qNaN ------------------------------------------------
    {kOne, kPosInf, Expect::QNaN, 0, "powr(1, +inf) = qNaN"},
    {kOne, kNegInf, Expect::QNaN, 0, "powr(1, -inf) = qNaN"},

    // --- 0^(y<0) → +inf + DIVBYZERO ---------------------------------
    {kPosZero, kNegOne, Expect::PosInf, 0, "powr(+0, -1) = +inf — pole, raises DIVBYZERO"},
    {kPosZero, bits_of(-0.5), Expect::PosInf, 0, "powr(+0, -0.5) = +inf"},
    {kPosZero, kNegInf, Expect::PosInf, 0, "powr(+0, -inf) = +inf"},
    {kNegZero, kNegOne, Expect::PosInf, 0, "powr(-0, -1) = +inf — -0 treated as zero"},
    {kNegZero, kNegInf, Expect::PosInf, 0, "powr(-0, -inf) = +inf"},

    // --- 0^(y>0) → +0 -----------------------------------------------
    {kPosZero, kOne, Expect::PosZero, 0, "powr(+0, 1) = +0"},
    {kPosZero, kTwo, Expect::PosZero, 0, "powr(+0, 2) = +0"},
    {kPosZero, kHalf, Expect::PosZero, 0, "powr(+0, 0.5) = +0"},
    {kPosZero, kPosInf, Expect::PosZero, 0, "powr(+0, +inf) = +0"},
    {kNegZero, kOne, Expect::PosZero, 0, "powr(-0, 1) = +0 — result is always nonneg"},

    // --- +inf^(y>0) → +inf ; +inf^(y<0) → +0 ------------------------
    {kPosInf, kOne, Expect::PosInf, 0, "powr(+inf, 1) = +inf"},
    {kPosInf, kHalf, Expect::PosInf, 0, "powr(+inf, 0.5) = +inf"},
    {kPosInf, kNegOne, Expect::PosZero, 0, "powr(+inf, -1) = +0"},

    // --- x>1, ±inf ; 0<x<1, ±inf (finite-exponent corners) ----------
    {kTwo, kPosInf, Expect::PosInf, 0, "powr(2, +inf) = +inf"},
    {kTwo, kNegInf, Expect::PosZero, 0, "powr(2, -inf) = +0"},
    {kHalf, kPosInf, Expect::PosZero, 0, "powr(0.5, +inf) = +0"},
    {kHalf, kNegInf, Expect::PosInf, 0, "powr(0.5, -inf) = +inf"},

    // --- Finite trivial: powr(1, finite-y) = 1 ----------------------
    {kOne, kTwo, Expect::ExactBits, kOne, "powr(1, 2) = 1"},
    {kOne, kNegOne, Expect::ExactBits, kOne, "powr(1, -1) = 1"},
    {kOne, kPosZero, Expect::ExactBits, kOne,
     "powr(1, +0) = 1 (only 0^0 / inf^0 are NaN, not 1^0)"},
    {kOne, kNegZero, Expect::ExactBits, kOne, "powr(1, -0) = 1"},
};

} // namespace

int main() {
    int failures = 0;
    for (const Case& c : kCases) {
        const double x = from_bits(c.x_bits);
        const double y = from_bits(c.y_bits);
        const double got = sf64_powr(x, y);
        if (!check(c, got)) {
            ++failures;
            std::printf("FAIL: powr(%a, %a) got 0x%016llx  expected %s  (%s)\n", x, y,
                        static_cast<unsigned long long>(bits_of(got)), expect_name(c), c.why);
        }
    }
    if (failures == 0) {
        std::printf("test_powr_ieee754: %zu cases, all passed\n",
                    sizeof(kCases) / sizeof(kCases[0]));
        return 0;
    }
    std::printf("test_powr_ieee754: %d / %zu cases failed\n", failures,
                sizeof(kCases) / sizeof(kCases[0]));
    return 1;
}
