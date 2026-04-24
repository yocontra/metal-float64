// libFuzzer target for the `sf64_*_r(mode, ...)` surface.
//
// Each input byte sequence is interpreted as `(op_index, mode, a, b)`:
//   byte  0      - op selector (modulo the number of ops)
//   byte  1      - rounding mode (modulo 5)
//   bytes 2..9   - double `a` (raw IEEE-754 bits)
//   bytes 10..17 - double `b` (raw IEEE-754 bits)
//
// We run the selected `sf64_*_r(mode, a, b)` and compare bit-for-bit
// against MPFR at 3300-bit precision (arithmetic headroom for the full
// f64 exponent span). Any mismatch is a hard bug and trips
// `__builtin_trap()`, which libFuzzer reports as a crash finding.
//
// This target is not a precision harness — the crash hunt uses the
// same corpus as `fuzz_arithmetic.cpp` but multiplied by five rounding
// modes. The BIT_EXACT budget here is tight (0 ULP); any widening is
// forbidden by the integrity rules in CLAUDE.md, so a finding must be
// fixed in `src/` rather than papered over with a tolerance.
//
// Requires clang + libFuzzer. Built only when SOFT_FP64_BUILD_FUZZ=ON.
//
// SPDX-License-Identifier: MIT

#include "soft_fp64/rounding_mode.h"
#include "soft_fp64/soft_f64.h"

#include <mpfr.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

inline double bits_to_double(uint64_t b) {
    double d;
    std::memcpy(&d, &b, sizeof(d));
    return d;
}
inline uint64_t double_to_bits(double d) {
    uint64_t b;
    std::memcpy(&b, &d, sizeof(b));
    return b;
}

// Arithmetic precision high enough to preserve the exact value of any
// finite `a op b` or `a*b + c` over the full f64 exponent span. See the
// ARITH_PREC note in `tests/mpfr/test_mpfr_diff.cpp` for the
// derivation; we use the same value so the fuzzer's oracle matches the
// deterministic sweep oracle.
constexpr mpfr_prec_t ARITH_PREC = 3300;

mpfr_rnd_t sf_to_mpfr_direct(sf64_rounding_mode m) {
    switch (m) {
    case SF64_RNE:
        return MPFR_RNDN;
    case SF64_RTZ:
        return MPFR_RNDZ;
    case SF64_RUP:
        return MPFR_RNDU;
    case SF64_RDN:
        return MPFR_RNDD;
    case SF64_RNA:
        return MPFR_RNDN; // unused; RNA is handled via tie-detection
    }
    return MPFR_RNDN;
}

// Finalize a 3300-bit MPFR scratch to an f64 under `m`. For RNA, start
// from RNDN and swap to the magnitude-larger neighbor on confirmed
// halfway ties. See `finalize_d` in tests/mpfr/test_mpfr_diff.cpp for
// the full rationale (mirror of the same idiom).
double finalize_d(mpfr_t rm, sf64_rounding_mode m) {
    if (m == SF64_RNA) {
        const double rne = mpfr_get_d(rm, MPFR_RNDN);
        if (std::isnan(rne) || std::isinf(rne))
            return rne;
        const double down = mpfr_get_d(rm, MPFR_RNDD);
        const double up = mpfr_get_d(rm, MPFR_RNDU);
        if (down == up)
            return rne;
        mpfr_t dm, um, ld, lu;
        mpfr_init2(dm, ARITH_PREC);
        mpfr_init2(um, ARITH_PREC);
        mpfr_init2(ld, ARITH_PREC);
        mpfr_init2(lu, ARITH_PREC);
        mpfr_set_d(dm, down, MPFR_RNDN);
        mpfr_set_d(um, up, MPFR_RNDN);
        mpfr_sub(ld, rm, dm, MPFR_RNDN);
        mpfr_abs(ld, ld, MPFR_RNDN);
        mpfr_sub(lu, um, rm, MPFR_RNDN);
        mpfr_abs(lu, lu, MPFR_RNDN);
        const int cmp = mpfr_cmp(ld, lu);
        mpfr_clear(dm);
        mpfr_clear(um);
        mpfr_clear(ld);
        mpfr_clear(lu);
        if (cmp != 0)
            return rne;
        return (std::fabs(up) > std::fabs(down)) ? up : down;
    }
    return mpfr_get_d(rm, sf_to_mpfr_direct(m));
}

// IEEE 754-2008 §6.3 signed-zero rule for exact zero results from
// add/sub/fma (mirrored from `tests/mpfr/test_mpfr_diff.cpp`).
// The zero-plus-zero table depends on both operand signs and the
// mode; cancellation of nonzero operands follows the mode-dependent
// default (+0 except RDN → -0). Underflow to zero is NOT handled
// here — MPFR's `get_d` already preserves the underflow sign.
inline double zero_plus_zero_sign(double a, double b, sf64_rounding_mode m) {
    const int sa = std::signbit(a) ? 1 : 0;
    const int sb = std::signbit(b) ? 1 : 0;
    const int s = (m == SF64_RDN) ? (sa | sb) : (sa & sb);
    return s ? -0.0 : 0.0;
}
inline double apply_signed_zero_add(double r, mpfr_srcptr rm, double a, double b,
                                    sf64_rounding_mode m) {
    if (mpfr_zero_p(rm) == 0)
        return r;
    if (a == 0.0 && b == 0.0)
        return zero_plus_zero_sign(a, b, m);
    return (m == SF64_RDN) ? -0.0 : 0.0;
}
inline double apply_signed_zero_sub(double r, mpfr_srcptr rm, double a, double b,
                                    sf64_rounding_mode m) {
    if (mpfr_zero_p(rm) == 0)
        return r;
    if (a == 0.0 && b == 0.0)
        return zero_plus_zero_sign(a, -b, m);
    return (m == SF64_RDN) ? -0.0 : 0.0;
}
inline double apply_signed_zero_fma(double r, mpfr_srcptr rm, double a, double b, double c,
                                    sf64_rounding_mode m) {
    if (mpfr_zero_p(rm) == 0)
        return r;
    const bool product_is_zero = (a == 0.0) || (b == 0.0);
    const bool c_is_zero = (c == 0.0);
    if (product_is_zero && c_is_zero) {
        const int product_sign = (std::signbit(a) ? 1 : 0) ^ (std::signbit(b) ? 1 : 0);
        const double product_zero = product_sign ? -0.0 : 0.0;
        return zero_plus_zero_sign(product_zero, c, m);
    }
    return (m == SF64_RDN) ? -0.0 : 0.0;
}

// Oracle for each binary op. Uses `mpfr_*(rm, am, bm, MPFR_RNDN)` at
// ARITH_PREC; the final rounding mode is applied by `finalize_d`.
double oracle_add(double a, double b, sf64_rounding_mode m) {
    mpfr_t am, bm, rm;
    mpfr_init2(am, ARITH_PREC);
    mpfr_init2(bm, ARITH_PREC);
    mpfr_init2(rm, ARITH_PREC);
    mpfr_set_d(am, a, MPFR_RNDN);
    mpfr_set_d(bm, b, MPFR_RNDN);
    mpfr_add(rm, am, bm, MPFR_RNDN);
    double r = finalize_d(rm, m);
    r = apply_signed_zero_add(r, rm, a, b, m);
    mpfr_clear(am);
    mpfr_clear(bm);
    mpfr_clear(rm);
    return r;
}
double oracle_sub(double a, double b, sf64_rounding_mode m) {
    mpfr_t am, bm, rm;
    mpfr_init2(am, ARITH_PREC);
    mpfr_init2(bm, ARITH_PREC);
    mpfr_init2(rm, ARITH_PREC);
    mpfr_set_d(am, a, MPFR_RNDN);
    mpfr_set_d(bm, b, MPFR_RNDN);
    mpfr_sub(rm, am, bm, MPFR_RNDN);
    double r = finalize_d(rm, m);
    r = apply_signed_zero_sub(r, rm, a, b, m);
    mpfr_clear(am);
    mpfr_clear(bm);
    mpfr_clear(rm);
    return r;
}
double oracle_mul(double a, double b, sf64_rounding_mode m) {
    mpfr_t am, bm, rm;
    mpfr_init2(am, ARITH_PREC);
    mpfr_init2(bm, ARITH_PREC);
    mpfr_init2(rm, ARITH_PREC);
    mpfr_set_d(am, a, MPFR_RNDN);
    mpfr_set_d(bm, b, MPFR_RNDN);
    mpfr_mul(rm, am, bm, MPFR_RNDN);
    const double r = finalize_d(rm, m);
    mpfr_clear(am);
    mpfr_clear(bm);
    mpfr_clear(rm);
    return r;
}
double oracle_div(double a, double b, sf64_rounding_mode m) {
    mpfr_t am, bm, rm;
    mpfr_init2(am, ARITH_PREC);
    mpfr_init2(bm, ARITH_PREC);
    mpfr_init2(rm, ARITH_PREC);
    mpfr_set_d(am, a, MPFR_RNDN);
    mpfr_set_d(bm, b, MPFR_RNDN);
    mpfr_div(rm, am, bm, MPFR_RNDN);
    const double r = finalize_d(rm, m);
    mpfr_clear(am);
    mpfr_clear(bm);
    mpfr_clear(rm);
    return r;
}
double oracle_sqrt(double a, sf64_rounding_mode m) {
    mpfr_t am, rm;
    mpfr_init2(am, ARITH_PREC);
    mpfr_init2(rm, ARITH_PREC);
    mpfr_set_d(am, a, MPFR_RNDN);
    mpfr_sqrt(rm, am, MPFR_RNDN);
    const double r = finalize_d(rm, m);
    mpfr_clear(am);
    mpfr_clear(rm);
    return r;
}

// NaN-tolerant bit-equality. Any quiet NaN counts equal to any quiet
// NaN (matches the test_arithmetic_exact.cpp convention); non-NaN
// results must match bitwise.
bool same_value(double got, double expect) {
    if (std::isnan(got) && std::isnan(expect))
        return true;
    return double_to_bits(got) == double_to_bits(expect);
}

[[noreturn]] void fuzz_fail(const char* op, sf64_rounding_mode m, double a, double b, double got,
                            double expect) {
    uint64_t ab, bb, gb, eb;
    std::memcpy(&ab, &a, 8);
    std::memcpy(&bb, &b, 8);
    std::memcpy(&gb, &got, 8);
    std::memcpy(&eb, &expect, 8);
    std::fprintf(stderr,
                 "\n[fuzz_rounding_modes] %s mode=%d a=0x%016llx b=0x%016llx "
                 "got=0x%016llx expect=0x%016llx\n",
                 op, static_cast<int>(m), (unsigned long long)ab, (unsigned long long)bb,
                 (unsigned long long)gb, (unsigned long long)eb);
    __builtin_trap();
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Minimum consumption: 2 selector bytes + 2 * 8-byte doubles = 18.
    if (size < 18)
        return 0;

    const uint8_t op_index = data[0] % 5u;
    const uint8_t mode_index = data[1] % 5u;
    const sf64_rounding_mode mode = static_cast<sf64_rounding_mode>(mode_index);

    uint64_t ab, bb;
    std::memcpy(&ab, data + 2, 8);
    std::memcpy(&bb, data + 10, 8);
    const double a = bits_to_double(ab);
    const double b = bits_to_double(bb);

    double got = 0.0;
    double expect = 0.0;
    const char* op_name = "?";
    switch (op_index) {
    case 0:
        op_name = "add_r";
        got = sf64_add_r(mode, a, b);
        expect = oracle_add(a, b, mode);
        break;
    case 1:
        op_name = "sub_r";
        got = sf64_sub_r(mode, a, b);
        expect = oracle_sub(a, b, mode);
        break;
    case 2:
        op_name = "mul_r";
        got = sf64_mul_r(mode, a, b);
        expect = oracle_mul(a, b, mode);
        break;
    case 3:
        op_name = "div_r";
        got = sf64_div_r(mode, a, b);
        expect = oracle_div(a, b, mode);
        break;
    case 4:
        op_name = "sqrt_r";
        got = sf64_sqrt_r(mode, a);
        expect = oracle_sqrt(a, mode);
        break;
    }
    if (!same_value(got, expect)) {
        fuzz_fail(op_name, mode, a, b, got, expect);
    }

    // Optional: exercise fma_r with the same two operands and a third
    // pulled from the remaining bytes if present. Fma has the widest
    // rounding surface (three operands + fused product).
    if (size >= 26) {
        uint64_t cb;
        std::memcpy(&cb, data + 18, 8);
        const double c = bits_to_double(cb);
        const double fma_got = sf64_fma_r(mode, a, b, c);
        mpfr_t am, bm, cm, rm;
        mpfr_init2(am, ARITH_PREC);
        mpfr_init2(bm, ARITH_PREC);
        mpfr_init2(cm, ARITH_PREC);
        mpfr_init2(rm, ARITH_PREC);
        mpfr_set_d(am, a, MPFR_RNDN);
        mpfr_set_d(bm, b, MPFR_RNDN);
        mpfr_set_d(cm, c, MPFR_RNDN);
        mpfr_fma(rm, am, bm, cm, MPFR_RNDN);
        double fma_expect = finalize_d(rm, mode);
        fma_expect = apply_signed_zero_fma(fma_expect, rm, a, b, c, mode);
        mpfr_clear(am);
        mpfr_clear(bm);
        mpfr_clear(cm);
        mpfr_clear(rm);
        if (!same_value(fma_got, fma_expect)) {
            fuzz_fail("fma_r", mode, a, b, fma_got, fma_expect);
        }
    }

    // MPFR caches grow if we never free; trigger a per-run cleanup so
    // the fuzzer's long runs stay bounded. `mpfr_free_cache` is safe
    // to call frequently (no-op when caches are empty).
    mpfr_free_cache();

    return 0;
}
