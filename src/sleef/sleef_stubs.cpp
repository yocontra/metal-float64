//
// Derived from SLEEF 3.6 `src/libm/sleefdp.c` (Boost-1.0); see
// `src/sleef/NOTICE` for the upstream URL, the pinned commit SHA, and the
// full Boost-1.0 license text.
//
// Holds the long-tail transcendentals (erf, erfc, tgamma, lgamma,
// lgamma_r, asinpi, acospi, atanpi, atan2pi, rootn) with u10-ish / u35-ish
// implementations derived from SLEEF 3.6 plus the standard Lanczos
// (Boost-1.0) formulation for the gamma family.
//
// Every `+`, `-`, `*`, `/`, `fma`, `sqrt`, `floor`, `ldexp` inside function
// bodies is a call into the `sf64_*` ABI. No host-FPU operator expressions
// on `double` values are emitted here; only compile-time constant folding
// on `constexpr` literals (which is a property of the constant, not a
// runtime op).
//
// Owns: sf64_asinpi / sf64_acospi / sf64_atanpi / sf64_atan2pi /
//       sf64_rootn /
//       sf64_erf / sf64_erfc / sf64_tgamma / sf64_lgamma / sf64_lgamma_r
//
// SPDX-License-Identifier: BSL-1.0 AND MIT
//

#include "sleef_internal.h"

// NOTE: `sleef_fe_macros.h` must follow `sleef_internal.h`. Blank line
// keeps clang-format from alphabetising them back together.

#include "sleef_fe_macros.h"

using soft_fp64::sleef::DD;
using soft_fp64::sleef::dd_to_d;
using soft_fp64::sleef::ddadd2_dd_d_d;
using soft_fp64::sleef::ddadd2_dd_dd;
using soft_fp64::sleef::ddadd2_dd_dd_d;
using soft_fp64::sleef::ddmul_dd_d_d;
using soft_fp64::sleef::ddmul_dd_dd_d;
using soft_fp64::sleef::ddmul_dd_dd_dd;
using soft_fp64::sleef::ddneg_dd_dd;
using soft_fp64::sleef::ddrec_dd_d;
using soft_fp64::sleef::ddscale_dd_dd_d;
using soft_fp64::sleef::eq_;
using soft_fp64::sleef::ge_;
using soft_fp64::sleef::gt_;
using soft_fp64::sleef::isinf_;
using soft_fp64::sleef::isnan_;
using soft_fp64::sleef::le_;
using soft_fp64::sleef::lt_;
using soft_fp64::sleef::ne_;
using soft_fp64::sleef::poly_array;
using soft_fp64::sleef::sf64_internal_expk_dd;
using soft_fp64::sleef::sf64_internal_logk_dd;
using soft_fp64::sleef::signbit_;
using soft_fp64::sleef::detail::is_int;
using soft_fp64::sleef::detail::kInf;
using soft_fp64::sleef::detail::kPI;
using soft_fp64::sleef::detail::qNaN;

namespace {

// ---- shared constants --------------------------------------------------
//
// 1 / π as a double-double.  hi is the nearest-double to 1/π; lo is the
// residual (1/π - hi) rounded to nearest double.  Taken from SLEEF
// `src/libm/sleefdp.c`.
constexpr double kInvPI_hi = 0.3183098861837906715;
constexpr double kInvPI_lo = -1.9678676675182486e-17;

} // namespace

// ========================================================================
// π-scaled inverse trig (asinpi / acospi / atanpi / atan2pi)
// ========================================================================
//
// Idea: compute the forward inverse-trig result with the existing real
// `sf64_asin` / `sf64_acos` / `sf64_atan` / `sf64_atan2` primitives, then
// multiply by 1/π carried in double-double.  This avoids losing a bit to
// the final divide.  The angle itself is only a plain double so the
// best we can get is about (ulp(angle) + ulp(angle) * |lo/hi|) ≈ 1 ULP.
// Well within the U35 (≤8 ULP) band that the test spec pins for asinpi et al.

extern "C" double sf64_asinpi(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    const double a = sf64_asin(x);
    if (isnan_(a) || isinf_(a))
        return a;
    // (a + 0) * (1/π hi + 1/π lo)
    const DD prod = ddmul_dd_d_d(a, kInvPI_hi);
    const DD with_lo = ddadd2_dd_dd_d(prod, sf64_mul(a, kInvPI_lo));
    const double r = dd_to_d(with_lo);
    fe.flush();
    return r;
}

extern "C" double sf64_acospi(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    const double a = sf64_acos(x);
    if (isnan_(a) || isinf_(a))
        return a;
    const DD prod = ddmul_dd_d_d(a, kInvPI_hi);
    const DD with_lo = ddadd2_dd_dd_d(prod, sf64_mul(a, kInvPI_lo));
    const double r = dd_to_d(with_lo);
    fe.flush();
    return r;
}

extern "C" double sf64_atanpi(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    const double a = sf64_atan(x);
    if (isnan_(a))
        return a;
    // atan(±inf) = ±π/2 → atanpi(±inf) = ±0.5
    if (isinf_(x))
        return signbit_(x) ? -0.5 : 0.5;
    const DD prod = ddmul_dd_d_d(a, kInvPI_hi);
    const DD with_lo = ddadd2_dd_dd_d(prod, sf64_mul(a, kInvPI_lo));
    const double r = dd_to_d(with_lo);
    fe.flush();
    return r;
}

extern "C" double sf64_atan2pi(double y, double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    const double a = sf64_atan2(y, x);
    if (isnan_(a))
        return a;
    const DD prod = ddmul_dd_d_d(a, kInvPI_hi);
    const DD with_lo = ddadd2_dd_dd_d(prod, sf64_mul(a, kInvPI_lo));
    const double r = dd_to_d(with_lo);
    fe.flush();
    return r;
}

// ========================================================================
// rootn(x, n) = x^(1/n) with sign handling for odd n
// ========================================================================

extern "C" double sf64_rootn(double x, int n) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x))
        return qNaN();
    if (n == 0)
        return qNaN();
    if (n == 1)
        return x;

    const bool n_odd = (n & 1) != 0;

    // rootn(±0, n): +0 or ±0 for n>0 (sign preserved for odd n), ±inf for n<0.
    if (eq_(x, 0.0)) {
        if (n > 0)
            return (signbit_(x) && n_odd) ? sf64_neg(0.0) : 0.0;
        // n < 0 → pole at 0.
        return (signbit_(x) && n_odd) ? sf64_neg(kInf) : kInf;
    }

    if (isinf_(x)) {
        if (n > 0) {
            // rootn(+inf, n>0) = +inf; rootn(-inf, odd n>0) = -inf; else NaN.
            if (!signbit_(x))
                return kInf;
            return n_odd ? sf64_neg(kInf) : qNaN();
        }
        // n < 0 → rootn(±inf, n<0) = ±0 for odd n, +0 for even n.
        return (signbit_(x) && n_odd) ? sf64_neg(0.0) : 0.0;
    }

    // Negative x with even n is undefined.
    if (lt_(x, 0.0) && !n_odd)
        return qNaN();

    // Special cases where we can be exact.
    if (n == 2)
        return sf64_sqrt(x);
    if (n == 3)
        return sf64_cbrt(x);
    if (n == -1)
        return sf64_div(1.0, x);

    // General path: r = sign * exp(log|x| / n).  For |x| != 1 we can use
    // pow; for x == 1 or -1 we can short-circuit.
    if (eq_(sf64_fabs(x), 1.0)) {
        return (signbit_(x) && n_odd) ? -1.0 : 1.0;
    }

    const double inv_n = sf64_div(1.0, sf64_from_i32(n));
    double r = sf64_pow(sf64_fabs(x), inv_n);
    if (signbit_(x) && n_odd)
        r = sf64_neg(r);
    fe.flush();
    return r;
}

// ========================================================================
// erf / erfc
// ========================================================================
//
// Classic piecewise-rational approximation (Numerical Recipes style).
// For |x| small (≤ ~1) we expand the Maclaurin series for
//   erf(x) = (2/√π) * x * (1 - x²/3 + x^4/10 - x^6/42 + x^8/216 - …)
// For |x| larger we use Chebyshev rational via a Horner polynomial in
// 1/(1+0.5|x|) that targets erfc(|x|) * exp(x²); the residual stays
// well-conditioned out to x ~ 6 and then erfc underflows.
//
// Gated tier is GAMMA (≤1024 ULP). erfc deep tail (|x| > 15) lands outside
// GAMMA and is parked in tests/experimental/ until the Temme-asymptotic
// polish lands (tracked in TODO.md).

namespace {

// 2/sqrt(pi)
constexpr double kTwoOverSqrtPI = 1.1283791670955125738961589031215452;

// Maclaurin coefficients for (erf(x) / (2x/√π)) = sum_{k=0}^∞ (-1)^k x^{2k} / (k! (2k+1))
// Horner highest-first in y = x^2. 14 terms → good to ~|x|<=1 with plenty of slack.
constexpr double kErfTaylorY[] = {
    -1.0 / (/* 13! */ 6227020800.0 * 27.0), // k=13
    1.0 / (/* 12! */ 479001600.0 * 25.0),   // k=12
    -1.0 / (/* 11! */ 39916800.0 * 23.0),   // k=11
    1.0 / (/* 10! */ 3628800.0 * 21.0),     // k=10
    -1.0 / (/*  9! */ 362880.0 * 19.0),     // k=9
    1.0 / (/*  8! */ 40320.0 * 17.0),       // k=8
    -1.0 / (/*  7! */ 5040.0 * 15.0),       // k=7
    1.0 / (/*  6! */ 720.0 * 13.0),         // k=6
    -1.0 / (/*  5! */ 120.0 * 11.0),        // k=5
    1.0 / (/*  4! */ 24.0 * 9.0),           // k=4
    -1.0 / (/*  3! */ 6.0 * 7.0),           // k=3
    1.0 / (/*  2! */ 2.0 * 5.0),            // k=2
    -1.0 / (/*  1! */ 1.0 * 3.0),           // k=1
    1.0,                                    // k=0
};

// Small |x|: erf(x) = (2x/√π) * Taylor(x²).
double erf_small(double x, soft_fp64::sleef::sf64_internal_fe_acc& fe) {
    const double x2 = sf64_mul(x, x);
    const double p = poly_array(x2, kErfTaylorY, sizeof(kErfTaylorY) / sizeof(kErfTaylorY[0]));
    // (2/√π) * x * p
    return sf64_mul(sf64_mul(kTwoOverSqrtPI, x), p);
}

// Numerical Recipes-style Chebyshev coefficients for erfc(|x|) * exp(x²),
// evaluated on t = 2 / (2 + |x|) ∈ (0, 1].  The polynomial is
// 27 terms (indices 0..26).  Source: Chebyshev expansion from NR3 §6.2
// (public-domain constants, also in SLEEF's `erfccheb`).
constexpr double kErfcChebCoef[] = {
    -1.3026537197817094,
    6.4196979235649026e-1,
    1.9476473204185836e-2,
    -9.561514786808631e-3,
    -9.46595344482036e-4,
    3.66839497852761e-4,
    4.2523324806907e-5,
    -2.0278578112534e-5,
    -1.624290004647e-6,
    1.303655835580e-6,
    1.5626441722e-8,
    -8.5238095915e-8,
    6.529054439e-9,
    5.059343495e-9,
    -9.91364156e-10,
    -2.27365122e-10,
    9.6467911e-11,
    2.394038e-12,
    -6.886027e-12,
    8.94487e-13,
    3.13092e-13,
    -1.12708e-13,
    3.81e-16,
    7.106e-15,
    -1.523e-15,
    -9.4e-17,
    1.21e-16,
    -2.8e-17,
};

// Evaluate Clenshaw for erfccheb at t∈(0,1]; follows NR3 erfccheb().
//   ty = 4*t - 2
// Returns t * exp(-z*z + p(ty))   where z=|x|, t=2/(2+z).
//
// Deep tail (z > ~15) is cancellation-dominated inside the exp argument:
// `-z*z` as plain double has ~ulp(z²) = 2^(⌈log₂ z²⌉-52) absolute error,
// which exp amplifies into thousands of ULP of relative error.  Fix: build
// the exp argument as a DD (z² is computed exactly via `ddmul_dd_d_d` which
// is an FMA-backed Dekker product) and feed the DD into `expk_dd` — the
// same DD-accurate exp that `sf64_pow` uses.  This keeps the deep tail
// inside GAMMA without needing a separate experimental harness.
double erfc_cheb(double z, soft_fp64::sleef::sf64_internal_fe_acc& fe) {
    // t = 2 / (2 + z)
    const double t = sf64_div(2.0, sf64_add(2.0, z));
    const double ty = sf64_sub(sf64_mul(t, 4.0), 2.0);
    constexpr int NCOEF = sizeof(kErfcChebCoef) / sizeof(kErfcChebCoef[0]);
    // Clenshaw: d = 0, dd = 0; for j = NCOEF-1 .. 1:  tmp = d; d = ty*d - dd + c[j]; dd = tmp.
    // Return t * exp(-z*z + 0.5*c[0] + ty*0.5*d - dd).
    double d_ = 0.0;
    double dd_ = 0.0;
    for (int j = NCOEF - 1; j >= 1; --j) {
        const double tmp = d_;
        d_ = sf64_add(sf64_sub(sf64_mul(ty, d_), dd_), kErfcChebCoef[j]);
        dd_ = tmp;
    }
    // Build exp_arg = -z² + 0.5*c[0] + 0.5*ty*d_ - dd_ as a DD.
    const DD neg_zsq = ddneg_dd_dd(ddmul_dd_d_d(z, z)); // exact -z² as DD
    const DD ty_d = ddmul_dd_d_d(ty, d_);
    const DD half_ty_d = ddscale_dd_dd_d(ty_d, 0.5); // lossless × 2⁻¹
    // 0.5 * c[0] folds at compile time (exact: x * 2⁻¹).
    const DD tail0 = ddadd2_dd_dd_d(half_ty_d, sf64_mul(kErfcChebCoef[0], 0.5));
    const DD tail = ddadd2_dd_dd_d(tail0, sf64_neg(dd_));
    const DD exp_arg_dd = ddadd2_dd_dd(neg_zsq, tail);
    return sf64_mul(t, soft_fp64::sleef::sf64_internal_expk_dd(exp_arg_dd, fe));
}

} // namespace

extern "C" double sf64_erf(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x))
        return qNaN();
    if (isinf_(x))
        return signbit_(x) ? -1.0 : 1.0;
    const double a = sf64_fabs(x);
    // Taylor converges well for |x| < ~0.9 with 14 terms.  Past that, use
    // the Chebyshev erfc path: erf(|x|) = 1 - erfc(|x|).
    if (lt_(a, 0.9)) {
        const double r = erf_small(a, fe);
        fe.flush();
        return signbit_(x) ? sf64_neg(r) : r;
    }
    if (gt_(a, 6.0)) {
        // erf(|x|) ≈ 1; keep correct sign.
        return signbit_(x) ? -1.0 : 1.0;
    }
    const double c = erfc_cheb(a, fe);
    const double r = sf64_sub(1.0, c);
    fe.flush();
    return signbit_(x) ? sf64_neg(r) : r;
}

extern "C" double sf64_erfc(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x))
        return qNaN();
    if (isinf_(x))
        return signbit_(x) ? 2.0 : 0.0;
    const double a = sf64_fabs(x);
    // For small |x| use 1 - erf(x) directly.  Cancellation is tolerable
    // since erf(0.9) ≈ 0.797 (leaves ~0.2 precision margin).  For |x| >= 0.9
    // switch to the Chebyshev erfc form which is cancellation-free.
    if (lt_(a, 0.9)) {
        const double e = erf_small(x, fe); // uses signed x
        const double r = sf64_sub(1.0, e);
        fe.flush();
        return r;
    }
    // |x| >= 1: use Chebyshev for erfc(|x|).
    if (gt_(a, 27.0)) {
        // erfc underflows from the positive side; mirror for negative.
        if (signbit_(x))
            return 2.0;
        return 0.0;
    }
    const double c = erfc_cheb(a, fe);
    const double r = signbit_(x) ? sf64_sub(2.0, c) : c;
    fe.flush();
    return r;
}

// ========================================================================
// tgamma / lgamma / lgamma_r  (Lanczos g=7 approximation)
// ========================================================================
//
// Lanczos g=7, n=9 coefficients from Paul Godfrey's public tables (the same
// ones used by Boost.Math, SciPy, and others). Licence-compatible.
//
//   gamma(x+1) = sqrt(2π) * (x + g + 0.5)^(x+0.5) * e^-(x+g+0.5) * A_g(x)
//   A_g(x) = c0 + sum_{k=1..8} c_k / (x + k)
//
// For x < 0.5 we use the reflection formula:
//   gamma(x) * gamma(1 - x) = π / sin(πx)
//
// lgamma = log |gamma(x)|; lgamma_r also returns sign.

namespace {

constexpr double kLanczosG = 7.0;
constexpr double kLanczosC[9] = {
    0.99999999999980993,  676.5203681218851,     -1259.1392167224028,
    771.32342877765313,   -176.61502916214059,   12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
};

// Evaluate A_g(z-1) = c0 + c1/z + c2/(z+1) + … + c8/(z+7)
// where z = x (and the caller passed x = original arg).  Internally we use
// the Godfrey recurrence:  A_g(x) = c0 + sum_{k=1..8} c_k / (x + k - 1).
// (Note: this is the "x -> x-1" shift convention; we keep it so that for
// x>=1 we compute gamma(x) = sqrt(2π) * t^(x-0.5) * e^-t * A, t=(x-1)+g+0.5.)
//
// The plain-double accumulator at individual summand magnitudes up to
// ~400 with final sum ≈ 79 near x=2 loses ~3 decimal digits to
// cancellation — the dominant error source for lgamma zero-crossings.
// DD accumulator (TwoSum each step) keeps the summation exact; individual
// quotients still inherit 0.5 ULP from `sf64_div`, which is fine since the
// `log` consumer only needs ≥~2^-60 relative in `a`.
DD lanczos_a_shifted_dd(double x, soft_fp64::sleef::sf64_internal_fe_acc& fe) {
    DD acc = DD{kLanczosC[0], 0.0};
    for (int k = 1; k <= 8; ++k) {
        // term = c_k / (x + k - 1) built as DD: DD_reciprocal × scalar.
        // Plain-double `sf64_div` leaves 0.5 ULP per term ≈ 4.5 ULP summed,
        // which (via log) dominates lgamma zero-crossing error by a factor
        // of ~5e4 ULP.  DD division drops that residual to 2^-104 relative.
        const double denom = sf64_add(x, sf64_from_i32(k - 1));
        const DD recip = ddrec_dd_d(denom);
        const DD term_dd = ddmul_dd_dd_d(recip, kLanczosC[k]);
        acc = ddadd2_dd_dd(acc, term_dd);
    }
    return acc;
}

// log of a DD number `d = d.hi + d.lo`, returned as DD.  We call the
// plain-double `logk_dd` on d.hi, then add the first-order correction
// `d.lo / d.hi` (log(1 + ε) ≈ ε for |ε| ≤ 2^-53).  Second-order correction
// ≤ ε²/2 ≤ 2^-107 — negligible at DD precision.
DD logk_dd_of_dd(DD d, soft_fp64::sleef::sf64_internal_fe_acc& fe) {
    DD log_hi = soft_fp64::sleef::sf64_internal_logk_dd(d.hi, fe);
    const double correction = sf64_div(d.lo, d.hi);
    return ddadd2_dd_dd_d(log_hi, correction);
}

// 0.5·ln(2π) carried as DD.  The plain-double representation is off by
// ulp/2 ≈ 5.6e-17 from true, which is the full absolute error budget at the
// lgamma zero-crossings (x → 1, x → 2 where |lgamma| ≈ 1e-5).  Split
// computed via MPFR-200 at build-design time; both constants are exact
// IEEE-754 doubles.
constexpr double kLog2PiOver2_hi = 0x1.d67f1c864beb5p-1;   //  9.1893853320467278056e-01
constexpr double kLog2PiOver2_lo = -0x1.65b5a1b7ff5dfp-55; // -3.8782941580672414498e-17

// Shared Lanczos lg body.  Builds
//   lg = 0.5*log(2π) + (z+0.5)*log(t) - t + log(A_g(x))
// with t, A, log(t), log(A), and every accumulating sum carried in DD.
// Large cancellation near the zero-crossings of lgamma (x ≈ 1, x ≈ 2) makes
// the plain-double sum lose ~8 ulp(max summand) ≈ 2^-49 absolute into a
// result that's close to zero — that's the 131 k-ULP drift the experimental
// harness flags.  DD keeps ≥~2^-100 absolute across the sum.
DD lgamma_dd(double x, soft_fp64::sleef::sf64_internal_fe_acc& fe) {
    const double z = sf64_sub(x, 1.0);
    // t = z + g + 0.5, carried in DD so the plain-double rounding noise in
    // `z + 7.5` doesn't leak into `log(t) · (z+0.5)` (largest term in lg).
    const DD t_dd = ddadd2_dd_d_d(z, sf64_add(kLanczosG, 0.5));
    const DD a_dd = lanczos_a_shifted_dd(x, fe);

    const DD log_t_dd = logk_dd_of_dd(t_dd, fe);
    const DD log_a_dd = logk_dd_of_dd(a_dd, fe);
    // zp5 = z + 0.5 in DD — the plain-double rounding of `z + 0.5` leaves
    // ~2^-53 abs error, amplified by log(t) ≈ 2 into ~2^-52 abs error in
    // term1, which in turn is ≈80 k ULP of the near-zero lgamma result.
    const DD zp5_dd = ddadd2_dd_d_d(z, 0.5);
    const DD term1_dd = ddmul_dd_dd_dd(log_t_dd, zp5_dd); // (z+0.5)·log(t)
    const DD log2pi_half_dd{kLog2PiOver2_hi, kLog2PiOver2_lo};
    DD lg_dd = ddadd2_dd_dd(term1_dd, log2pi_half_dd); // + 0.5·log(2π)
    lg_dd = ddadd2_dd_dd(lg_dd, ddneg_dd_dd(t_dd));    // − t
    lg_dd = ddadd2_dd_dd(lg_dd, log_a_dd);             // + log(A)
    return lg_dd;
}

// ---- Zero-centered Taylor expansion of log Γ(x) -----------------------
//
// Near the zeros at x = 1 and x = 2, the Lanczos formula produces a
// DD-accurate `lgamma_dd` whose final `dd_to_d` collapse leaks O(ulp(1)) ≈
// 2.2e-16 absolute noise into a result that vanishes there.  ULP-against-
// near-zero blows past GAMMA = 1024 even with perfect ingredients — the
// fix has to be algorithmic, not ingredient-precision.
//
// Series source: DLMF §5.7 (https://dlmf.nist.gov/5.7#E3) and Wolfram
// MathWorld "Log Gamma Function" — both give
//
//   log Γ(1+z) = -γ z + Σ_{k=2}^∞ (-1)^k ζ(k) z^k / k    (|z| < 1)
//
// Around x = 2, Γ(2+z) = (1+z) Γ(1+z) so log Γ(2+z) = log(1+z) +
// log Γ(1+z), giving
//
//   log Γ(2+z) = (1-γ) z + Σ_{k=2}^∞ (-1)^k (ζ(k)-1) z^k / k    (|z| < 2)
//
// The (ζ(k) - 1) factor decays geometrically as 2^-k so the around-2
// series converges much faster than the around-1 series at the same |z|.
//
// Coefficients:
//   * Computed offline via mpmath at 300-bit precision, then split into
//     IEEE-754 (hi, lo) doubles so the leading terms — which dominate the
//     near-zero result — carry full DD precision into the Horner sum.
//   * The C reference is Boost.Math `lgamma_small_imp` (boost/math/
//     special_functions/detail/lgamma_small.hpp), which uses the same
//     two-pivot Taylor structure with a different polynomial form
//     (rational approximation at higher order); we use the bare zeta-
//     series here because it is bit-stable and the windows are narrow.
//
// Window selection: empirically (at this term count, evaluating each step
// in DD), |z| ≤ 0.25 keeps the around-1 sweep at ≤ 50 ULP vs MPFR; |z| ≤
// 0.5 keeps the around-2 sweep at ≤ 60 ULP.  Outside those windows the
// Lanczos path fits GAMMA cleanly because we are no longer near a zero.
//
// `N_TERMS = 22` gives a tail bound of ζ(22)/22 · 0.25^22 ≈ 2.5e-15
// (around 1) and ζ(22)·0.5^22 ≈ 2.6e-15 (around 2) — well below the
// GAMMA absolute-error budget at the window edges (lgamma at the edges
// is O(0.1), so 1024·ulp(0.1) ≈ 1.4e-14).

namespace {

constexpr int kLgammaTaylorN = 22;

// Around x = 1: c[1] = -γ; c[k] = (-1)^k · ζ(k) / k for 2 ≤ k ≤ 22.
// Each entry is (hi, lo) — the high part rounded-to-nearest, the low part
// the residual rounded-to-nearest.  Source: mpmath 300-bit.
constexpr double kLgammaT1_hi[kLgammaTaylorN + 1] = {
    0.0, // index 0 unused
    -0.5772156649015329,
    0.8224670334241132,
    -0.40068563438653143,
    0.27058080842778454,
    -0.20738555102867398,
    0.1695571769974082,
    -0.1440498967688461,
    0.12550966952474304,
    -0.11133426586956469,
    0.1000994575127818,
    -0.09095401714582904,
    0.083353840546109,
    -0.0769325164113522,
    0.07143294629536133,
    -0.06666870588242046,
    0.06250095514121304,
    -0.058823978658684585,
    0.055555767627403614,
    -0.05263167937961666,
    0.05000004769810169,
    -0.047619070330142226,
    0.04545455629320467,
};
constexpr double kLgammaT1_lo[kLgammaTaylorN + 1] = {
    0.0,
    4.942915152430645e-18,
    1.520336175199238e-17,
    2.250747042487504e-18,
    1.1871280107138412e-17,
    -4.099767328621813e-18,
    2.2393851330167238e-18,
    -9.623140085232555e-18,
    -2.5214685384672305e-18,
    -4.643990572582924e-18,
    2.6102404859583283e-18,
    -8.306705457691885e-19,
    2.963832603652642e-19,
    3.2900356019181198e-18,
    6.278806024191499e-18,
    -3.2295860759966306e-18,
    2.551099464019315e-18,
    2.6912901341966357e-18,
    -3.0261864849830964e-18,
    -2.523843702471215e-18,
    2.7894418264458796e-19,
    -2.4796342684293355e-18,
    4.382931774550076e-19,
};

// Around x = 2: c[1] = 1 - γ; c[k] = (-1)^k · (ζ(k) - 1) / k for 2 ≤ k ≤ 22.
constexpr double kLgammaT2_hi[kLgammaTaylorN + 1] = {
    0.0,
    0.42278433509846713,
    0.3224670334241132,
    -0.0673523010531981,
    0.020580808427784546,
    -0.007385551028673986,
    0.0028905103307415234,
    -0.001192753911703261,
    0.0005096695247430425,
    -0.00022315475845357939,
    9.945751278180853e-05,
    -4.492623673813314e-05,
    2.050721277567069e-05,
    -9.439488275268397e-06,
    4.374866789907488e-06,
    -2.039215753801366e-06,
    9.55141213040742e-07,
    -4.492469198764566e-07,
    2.1207184805554665e-07,
    -1.0043224823968099e-07,
    4.7698101693639804e-08,
    -2.2711094608943164e-08,
    1.0838659214896955e-08,
};
constexpr double kLgammaT2_lo[kLgammaTaylorN + 1] = {
    0.0,
    4.942915152430645e-18,
    1.520336175199238e-17,
    6.87667631175899e-18,
    1.4629392512775695e-18,
    4.1051370891788617e-19,
    -7.357950161901912e-20,
    4.1747852352514e-20,
    -2.780354175057013e-20,
    6.032078299350848e-21,
    2.734261130690314e-21,
    3.4577848248512954e-22,
    4.864174577619616e-22,
    8.111985879973243e-22,
    -3.7021851137962053e-22,
    -4.70891370095011e-23,
    4.798512617588967e-23,
    1.4219340578032317e-23,
    1.2243193613787666e-23,
    -5.246728062732248e-24,
    1.6747349659198183e-24,
    -1.406065812811299e-24,
    -5.018242148804151e-25,
};

// Evaluate P(z) = Σ_{k=1..N} c[k] z^k via DD Horner.
//
//   Q(z) = c[N] + c[N-1] z + … + c[1] z^{N-1}
//   P(z) = z · Q(z)
//
// Each Horner step is `acc = acc * z + c[k]` carried in DD: `ddmul_dd_dd_d`
// preserves DD precision through the multiply, `ddadd2_dd_dd` (with the
// coefficient lifted to DD via its hi/lo split) preserves it through the
// add.  Result is collapsed once at the end.
double lgamma_taylor(double z, const double* c_hi, const double* c_lo,
                     soft_fp64::sleef::sf64_internal_fe_acc& fe) {
    DD acc{c_hi[kLgammaTaylorN], c_lo[kLgammaTaylorN]};
    for (int k = kLgammaTaylorN - 1; k >= 1; --k) {
        acc = ddmul_dd_dd_d(acc, z);
        const DD ck{c_hi[k], c_lo[k]};
        acc = ddadd2_dd_dd(acc, ck);
    }
    // Final factor of z (exact DD multiply).
    acc = ddmul_dd_dd_d(acc, z);
    return dd_to_d(acc);
}

} // namespace

// log|Γ(x)| for x ≥ 0.5.
//
// Branch structure:
//   * |x - 1| ≤ 0.25 → Taylor around x = 1 (vanishing regime at x = 1)
//   * |x - 2| ≤ 0.5  → Taylor around x = 2 (vanishing regime at x = 2)
//   * else            → existing Lanczos path
//
// Outside the two Taylor windows, the result of Γ is well-conditioned (no
// near-zero ULP blow-up) and Lanczos lands inside GAMMA = 1024.  Inside
// the windows, the Taylor branch evaluates the residual `lgamma(x) − 0`
// directly so absolute error → 0 as x → 1 or x → 2 — exactly the property
// the Lanczos `dd_to_d(lgamma_dd)` collapse cannot deliver.
double lgamma_pos(double x, soft_fp64::sleef::sf64_internal_fe_acc& fe) {
    // |x - 1| ≤ 0.25 ⇔ x ∈ [0.75, 1.25].  Closed window at the boundary.
    const double z1 = sf64_sub(x, 1.0);
    if (le_(sf64_fabs(z1), 0.25)) {
        return lgamma_taylor(z1, kLgammaT1_hi, kLgammaT1_lo, fe);
    }
    // |x - 2| ≤ 0.5 ⇔ x ∈ [1.5, 2.5].
    const double z2 = sf64_sub(x, 2.0);
    if (le_(sf64_fabs(z2), 0.5)) {
        return lgamma_taylor(z2, kLgammaT2_hi, kLgammaT2_lo, fe);
    }
    return dd_to_d(lgamma_dd(x, fe));
}

// gamma(x) for x >= 0.5 via Lanczos.  Combine t^(z+0.5) * e^-t inside a
// single exp to avoid internal overflow: for large x, t^(z+0.5) overflows
// but (z+0.5)*log(t) - t stays finite until the true overflow point.
// Feeds the DD-accurate lg into `expk_dd` so the deep tail (x ~ 170,
// overflow boundary) keeps its precision through exp.
double tgamma_pos(double x, soft_fp64::sleef::sf64_internal_fe_acc& fe) {
    return soft_fp64::sleef::sf64_internal_expk_dd(lgamma_dd(x, fe), fe);
}

} // namespace

extern "C" double sf64_tgamma(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x))
        return qNaN();
    if (eq_(x, 0.0)) {
        // IEEE: tgamma(±0) = ±inf.
        return signbit_(x) ? sf64_neg(kInf) : kInf;
    }
    if (isinf_(x)) {
        return signbit_(x) ? qNaN() : kInf;
    }
    // Negative integers are poles.
    if (lt_(x, 0.0) && is_int(x))
        return qNaN();

    // Large x overflows.
    if (gt_(x, 171.624))
        return kInf;

    if (ge_(x, 0.5)) {
        const double r = tgamma_pos(x, fe);
        fe.flush();
        return r;
    }

    // Reflection: gamma(x) = π / (sin(πx) * gamma(1-x))
    // Use sinpi to keep argument reduction accurate.
    const double sp = sf64_sinpi(x);
    if (eq_(sp, 0.0))
        return qNaN();
    const double g1mx = tgamma_pos(sf64_sub(1.0, x), fe);
    const double r = sf64_div(kPI, sf64_mul(sp, g1mx));
    fe.flush();
    return r;
}

extern "C" double sf64_lgamma_r(double x, int* sign) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    int local_sign = 1;
    double result;
    if (isnan_(x)) {
        if (sign)
            *sign = 1;
        return qNaN();
    }
    if (isinf_(x)) {
        if (sign)
            *sign = 1;
        return kInf;
    }
    if (eq_(x, 0.0)) {
        if (sign)
            *sign = signbit_(x) ? -1 : 1;
        return kInf;
    }

    // Negative integers: pole → +inf, sign = 1 (libm convention).
    if (lt_(x, 0.0) && is_int(x)) {
        if (sign)
            *sign = 1;
        return kInf;
    }

    if (ge_(x, 0.5)) {
        result = lgamma_pos(x, fe);
        // gamma(x) for x>0 is positive (except the reflection cases handled
        // below); at x=0.5…, gamma > 0.
        local_sign = 1;
    } else if (gt_(x, 0.0)) {
        // 0 < x < 0.5: gamma(x) > 0 (positive infinite limit at 0+).
        // Use reflection via log to avoid catastrophic cancellation in
        // gamma itself:  lgamma(x) = log π - log|sin(πx)| - lgamma(1-x)
        const double sp = sf64_fabs(sf64_sinpi(x));
        result = sf64_sub(sf64_sub(sf64_log(kPI), sf64_log(sp)), lgamma_pos(sf64_sub(1.0, x), fe));
        local_sign = 1;
    } else {
        // x < 0, non-integer.  Same reflection formula; sign depends on
        // sign of sin(πx) * gamma(1-x).  gamma(1-x) is positive here
        // (since 1-x > 1), so sign is sign(sin(πx)).
        const double sp = sf64_sinpi(x);
        const double asp = sf64_fabs(sp);
        result = sf64_sub(sf64_sub(sf64_log(kPI), sf64_log(asp)), lgamma_pos(sf64_sub(1.0, x), fe));
        local_sign = signbit_(sp) ? -1 : 1;
    }
    if (sign)
        *sign = local_sign;
    fe.flush();
    return result;
}

extern "C" double sf64_lgamma(double x) {
    int s = 1;
    return sf64_lgamma_r(x, &s);
}
