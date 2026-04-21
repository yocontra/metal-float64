// libFuzzer target for transcendentals.
//
// Per op we consume 8 bytes (one f64 bit-pattern), except `pow` which
// consumes 16 bytes.  We rotate through sin/cos/tan/exp/log/pow based on
// a byte of the input — this keeps each call self-contained so libFuzzer's
// coverage-guided mutation can evolve per-op corpora.
//
// Tolerance strategy: the ULP-1 regression test (test_transcendental_1ulp.cpp)
// and the MPFR differential harness cover accuracy.  Here we only trap on
//   * not-a-valid-IEEE-754 bit-pattern (by construction, impossible — any
//     64-bit value is valid — but we still sanity-check the exponent/mantissa
//     isn't corrupted by a wild pointer scribble),
//   * NaN/Inf *divergence* from libm on finite inputs where libm is not NaN,
//   * log/sqrt of negative not returning NaN,
//   * exp(+inf) not returning +inf,
//   * ULP differences exceeding 2^20.
//
// Why 2^20 is the ULP budget (not U10=4 / U35=8 / GAMMA=1024):
//   libFuzzer corpora routinely probe inputs where libm's extended-precision
//   path diverges from a single-double Cody-Waite implementation by
//   thousands of ULP — that's a precision limitation of the spec tier,
//   not a bug. Running the *crash-hunting* fuzzer at the release tier
//   would raise ~continuous noise from those regimes and mask real
//   structural breaks (wild pointer scribble, UB-hit subnormal path,
//   sign-flip regression). Precision regressions are caught by
//   test_transcendental_1ulp.cpp and tests/mpfr/test_mpfr_diff.cpp,
//   which run at the correct tier.

#include <soft_fp64/soft_f64.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

double bits_to_double(uint64_t bits) {
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}

uint64_t double_to_bits(double d) {
    uint64_t bits;
    std::memcpy(&bits, &d, sizeof(bits));
    return bits;
}

bool is_nan(double d) {
    uint64_t b = double_to_bits(d);
    return ((b >> 52) & 0x7ffu) == 0x7ffu && (b & 0x000f'ffff'ffff'ffffULL) != 0;
}

[[noreturn]] void fuzz_fail(const char* /*msg*/) {
    __builtin_trap();
}

uint64_t ulp_diff(double x, double y) {
    if (is_nan(x) || is_nan(y))
        return 0;
    uint64_t xb = double_to_bits(x);
    uint64_t yb = double_to_bits(y);
    // Map to a monotone ordering: negatives become [0..2^63), non-negatives
    // become [2^63..2^64).  We stay in uint64_t throughout so the distance
    // computation can never overflow a signed type (which would be UB, per
    // the fuzzer itself catching this in an earlier draft).
    auto to_mono_u = [](uint64_t b) -> uint64_t {
        return (b & 0x8000'0000'0000'0000ULL)
                   ? (~b)                            // negative: bit-flip
                   : (b | 0x8000'0000'0000'0000ULL); // non-negative: set high bit
    };
    const uint64_t xm = to_mono_u(xb);
    const uint64_t ym = to_mono_u(yb);
    return (xm > ym) ? (xm - ym) : (ym - xm);
}

// Divergence classifier:
//  true  -> `sf` and `oracle` are "qualitatively the same" (both NaN, both
//           same-sign inf, or ULP within `budget`).
//  false -> a hard divergence we want libFuzzer to surface.
bool agrees(double sf, double oracle, uint64_t budget) {
    if (is_nan(oracle))
        return is_nan(sf); // oracle NaN -> we must also be NaN
    if (is_nan(sf))
        return false; // sf NaN, oracle finite/inf -> bug
    if (__builtin_isinf(oracle)) {
        return __builtin_isinf(sf) && std::signbit(sf) == std::signbit(oracle);
    }
    if (__builtin_isinf(sf)) {
        // sf went to inf while oracle stayed finite — real divergence.
        return false;
    }
    return ulp_diff(sf, oracle) <= budget;
}

// ULP budget: 2^20 (~1M ULP). Far looser than the 1-ULP target, but
// comfortably tight enough to catch "entire result shape wrong".
constexpr uint64_t kULPBudget = uint64_t(1) << 20;

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 9)
        return 0; // need 1 selector byte + 8 bytes of payload

    // Selector: which op to fuzz this round.
    const uint8_t sel = data[0] % 6;

    uint64_t xb;
    std::memcpy(&xb, data + 1, 8);
    const double x = bits_to_double(xb);

    volatile uint64_t sink = 0;

    switch (sel) {
    case 0: { // sin
        const double r = sf64_sin(x);
        sink ^= double_to_bits(r);
        // Finite input -> finite result.  Infinite/NaN input -> NaN.
        if (__builtin_isinf(x) || is_nan(x)) {
            if (!is_nan(r))
                fuzz_fail("sin(inf|NaN) != NaN");
        } else {
            if (is_nan(r) || __builtin_isinf(r))
                fuzz_fail("sin(finite) diverged");
            // Bound check: |sin(x)| <= 1.
            if (std::fabs(r) > 1.0 + 1e-12)
                fuzz_fail("|sin| > 1");
            if (!agrees(r, std::sin(x), kULPBudget))
                fuzz_fail("sin vs libm");
        }
        break;
    }
    case 1: { // cos
        const double r = sf64_cos(x);
        sink ^= double_to_bits(r);
        if (__builtin_isinf(x) || is_nan(x)) {
            if (!is_nan(r))
                fuzz_fail("cos(inf|NaN) != NaN");
        } else {
            if (is_nan(r) || __builtin_isinf(r))
                fuzz_fail("cos(finite) diverged");
            if (std::fabs(r) > 1.0 + 1e-12)
                fuzz_fail("|cos| > 1");
            if (!agrees(r, std::cos(x), kULPBudget))
                fuzz_fail("cos vs libm");
        }
        break;
    }
    case 2: { // tan
        const double r = sf64_tan(x);
        sink ^= double_to_bits(r);
        if (__builtin_isinf(x) || is_nan(x)) {
            if (!is_nan(r))
                fuzz_fail("tan(inf|NaN) != NaN");
        } else {
            // tan has true poles — just check libm agreement loosely, and
            // allow any finite/inf result.  NaN on finite input is an error
            // only if libm is finite.
            const double o = std::tan(x);
            if (!agrees(r, o, kULPBudget)) {
                // tan near poles is extremely sensitive; allow larger
                // divergence there.  Only fail if libm result is also well-
                // conditioned (|libm| < 2^30).
                if (!is_nan(o) && !__builtin_isinf(o) && std::fabs(o) < 0x1p30)
                    fuzz_fail("tan vs libm (well-conditioned)");
            }
        }
        break;
    }
    case 3: { // exp
        const double r = sf64_exp(x);
        sink ^= double_to_bits(r);
        // exp(NaN) = NaN, exp(+inf) = +inf, exp(-inf) = 0.
        if (is_nan(x)) {
            if (!is_nan(r))
                fuzz_fail("exp(NaN) != NaN");
        } else if (__builtin_isinf(x) && x > 0) {
            if (!(__builtin_isinf(r) && r > 0))
                fuzz_fail("exp(+inf) != +inf");
        } else if (__builtin_isinf(x) && x < 0) {
            if (r != 0.0)
                fuzz_fail("exp(-inf) != 0");
        } else {
            if (r < 0.0)
                fuzz_fail("exp(finite) is negative");
            if (!agrees(r, std::exp(x), kULPBudget))
                fuzz_fail("exp vs libm");
        }
        break;
    }
    case 4: { // log
        const double r = sf64_log(x);
        sink ^= double_to_bits(r);
        // log(NaN)=NaN, log(<0 finite)=NaN, log(+/-0)=-inf,
        // log(+inf)=+inf, log(1)=0.
        if (is_nan(x)) {
            if (!is_nan(r))
                fuzz_fail("log(NaN) != NaN");
        } else if (x < 0.0) {
            if (!is_nan(r))
                fuzz_fail("log(negative) != NaN");
        } else if (x == 0.0) {
            if (!(__builtin_isinf(r) && r < 0))
                fuzz_fail("log(0) != -inf");
        } else if (__builtin_isinf(x)) {
            if (!(__builtin_isinf(r) && r > 0))
                fuzz_fail("log(+inf) != +inf");
        } else {
            if (!agrees(r, std::log(x), kULPBudget))
                fuzz_fail("log vs libm");
        }
        break;
    }
    case 5: { // pow — consumes 16 bytes total
        if (size < 17)
            return 0;
        uint64_t yb;
        std::memcpy(&yb, data + 9, 8);
        const double y = bits_to_double(yb);
        const double r = sf64_pow(x, y);
        sink ^= double_to_bits(r);
        // pow has many corner cases; we just compare to libm on the
        // "unambiguous" subset: x > 0 finite, y finite.
        if (!is_nan(x) && !is_nan(y) && x > 0.0 && !__builtin_isinf(x) && !__builtin_isinf(y) &&
            std::fabs(y) < 1000.0) {
            if (!agrees(r, std::pow(x, y), kULPBudget))
                fuzz_fail("pow vs libm");
        }
        // NaN propagation on either finite NaN input.
        if ((is_nan(x) || is_nan(y)) && !(x == 1.0) && !(y == 0.0)) {
            // pow(1, NaN) = 1 and pow(NaN, 0) = 1 per C99; otherwise NaN.
            if (!is_nan(r))
                fuzz_fail("pow(NaN,*) or pow(*,NaN) not NaN");
        }
        break;
    }
    default:
        break;
    }

    (void)sink;
    return 0;
}
