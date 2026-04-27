//
// Derived from SLEEF 3.6 `src/libm/sleefdp.c` (Boost-1.0); see
// `src/sleef/NOTICE` for the upstream URL, the pinned commit SHA, and the
// full Boost-1.0 license text.
//
// Inverse trig + hyperbolics + pow family + cbrt + fmod/remainder.
// Every arithmetic op in this file is a call into the `sf64_*` ABI.
// No host FPU arithmetic.
//
// Owns: sf64_fmod / sf64_remainder /
//       sf64_asin / sf64_acos / sf64_atan / sf64_atan2 /
//       sf64_pow / sf64_powr / sf64_pown / sf64_cbrt /
//       sf64_sinh / sf64_cosh / sf64_tanh /
//       sf64_asinh / sf64_acosh / sf64_atanh
//
// π-scaled inverse trig (asinpi/acospi/atanpi/atan2pi) and rootn live in
// sleef_stubs.cpp.
//
// Tightened to SLEEF `xatan/xasin/xacos/xpow/xcbrt/xsinh/xcosh/xtanh/
// xasinh/xacosh/xatanh` pattern — minimax poly + DD composition for pow /
// cbrt / inverse hyperbolics. TU-local DD helpers live in an anonymous
// namespace so no new cross-TU symbols appear.
//
// SPDX-License-Identifier: BSL-1.0 AND MIT
//

#include "../internal_fenv.h"
#include "sleef_internal.h"

// NOTE: `sleef_fe_macros.h` must follow `sleef_internal.h` — the macros
// rewrite call sites at preprocessing time and would otherwise clobber
// the function definitions in the header above. The blank-line
// separator stops clang-format from restoring alphabetical order.

#include "sleef_fe_macros.h"

using soft_fp64::sleef::DD;
using soft_fp64::sleef::ddadd2_dd_d_d;
using soft_fp64::sleef::ddadd2_dd_dd;
using soft_fp64::sleef::ddadd2_dd_dd_d;
using soft_fp64::sleef::dddiv_dd_dd_dd;
using soft_fp64::sleef::ddmul_dd_d_d;
using soft_fp64::sleef::ddmul_dd_dd_d;
using soft_fp64::sleef::ddmul_dd_dd_dd;
using soft_fp64::sleef::ddneg_dd_dd;
using soft_fp64::sleef::ddnormalize_dd_dd;
using soft_fp64::sleef::ddscale_dd_dd_d;
using soft_fp64::sleef::ddsqu_dd_dd;
using soft_fp64::sleef::eq_;
using soft_fp64::sleef::ge_;
using soft_fp64::sleef::gt_;
using soft_fp64::sleef::is_snan_;
using soft_fp64::sleef::isinf_;
using soft_fp64::sleef::isnan_;
using soft_fp64::sleef::le_;
using soft_fp64::sleef::lt_;
using soft_fp64::sleef::ne_;
using soft_fp64::sleef::poly_array;
using soft_fp64::sleef::sf64_internal_exp_core;
using soft_fp64::sleef::sf64_internal_expk_dd;
using soft_fp64::sleef::sf64_internal_log_core;
using soft_fp64::sleef::sf64_internal_logk_dd;
using soft_fp64::sleef::signbit_;
using soft_fp64::sleef::detail::is_int;
using soft_fp64::sleef::detail::is_odd_int;
using soft_fp64::sleef::detail::kInf;
using soft_fp64::sleef::detail::kL2L;
using soft_fp64::sleef::detail::kL2U;
using soft_fp64::sleef::detail::kPI;
using soft_fp64::sleef::detail::kPI_2;
using soft_fp64::sleef::detail::kR_LN2;
using soft_fp64::sleef::detail::qNaN;

// ========================================================================
// fmod / remainder — exact, no transcendentals
// ========================================================================
// Implemented using sf64_rem (the arithmetic layer supplies an IEEE remainder).
// sf64_rem is IEEE-754 "remainder" (round to nearest even). fmod is
// round-to-zero. We implement fmod via the classic repeated subtraction /
// scaling identity without host FPU ops.

extern "C" double sf64_fmod(double x, double y) {
    // Stack-local fenv accumulator used by the sf64_* / DD macro helpers
    // threaded through the body. We never flush it at return — IEEE §5.3.1
    // says fmod is exact, so no inner arithmetic raise is observable on
    // TLS. Preserves the pre-1.1 mask semantics without needing
    // sf64_fe_save/restore around the loop.
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x) || isnan_(y)) {
        // IEEE 754 §7.2: sNaN input raises INVALID before payload propagation.
        if (is_snan_(x) || is_snan_(y)) {
            SF64_FE_RAISE(SF64_FE_INVALID);
        }
        return qNaN();
    }
    if (isinf_(x) || eq_(y, 0.0)) {
        SF64_FE_RAISE(SF64_FE_INVALID);
        return qNaN();
    }
    if (isinf_(y))
        return x;
    if (lt_(sf64_fabs(x), sf64_fabs(y)))
        return x;

    const double absx = sf64_fabs(x);
    const double absy = sf64_fabs(y);
    double r = absx;
    int ex, ey;
    (void)sf64_frexp(r, &ex);
    (void)sf64_frexp(absy, &ey);
    // Classic binary long-division remainder: at each power-of-two level
    // subtract absy<<k when it still fits. Loop down to diff==0 so we
    // correctly handle quotients whose bit sits at the lowest scale.
    int diff = ex - ey;
    while (diff >= 0) {
        const double s = sf64_ldexp(absy, diff);
        if (ge_(r, s))
            r = sf64_sub(r, s);
        diff -= 1;
    }
    // Intentionally do NOT flush `fe` — the IEEE exactness contract
    // requires any inner INEXACT/UNDERFLOW to be swallowed.
    return signbit_(x) ? sf64_neg(r) : r;
}

// IEEE-754 `remainder`: quotient rounded to nearest even. Implemented via
// fmod + tie-break so the sign and parity rule match glibc / TestFloat.
extern "C" double sf64_remainder(double x, double y) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x) || isnan_(y)) {
        // IEEE 754 §7.2: sNaN input raises INVALID before payload propagation.
        if (is_snan_(x) || is_snan_(y)) {
            SF64_FE_RAISE(SF64_FE_INVALID);
        }
        return qNaN();
    }
    if (isinf_(x) || eq_(y, 0.0)) {
        SF64_FE_RAISE(SF64_FE_INVALID);
        return qNaN();
    }
    if (isinf_(y))
        return x;

    // IEEE §5.3.1: `remainder` is exact and raises only INVALID (handled
    // above). The stack-local accumulator is deliberately not flushed —
    // every spurious inner INEXACT is discarded, matching the pre-1.1
    // save/restore behavior without the TLS roundtrip.
    const double absy = sf64_fabs(y);
    double r = sf64_fmod(x, y); // |r| <= |y|, sign of x; 0 if exact
    const double absr = sf64_fabs(r);
    // Compare 2|r| vs |y| via d = |y| - |r|. By Sterbenz, d is exact whenever
    // 2|r| >= |y| (|y|/2 <= |r| <= 2|y|), which is the only range where we
    // need an exact compare. Avoids the subnormal underflow in |y|*0.5 and
    // the potential overflow in 2*|r|.
    const double d = sf64_sub(absy, absr);
    if (gt_(absr, d)) {
        r = signbit_(r) ? sf64_add(r, absy) : sf64_sub(r, absy);
    } else if (eq_(absr, d)) {
        // Exact tie (2|r| == |y|, r != 0). Choose even quotient.
        const double q = sf64_div(sf64_sub(x, r), y);
        if (is_odd_int(q)) {
            r = signbit_(r) ? sf64_add(r, absy) : sf64_sub(r, absy);
        }
    }
    // Intentionally do NOT flush `fe`.
    return r;
}

// ========================================================================
// TU-local DD helpers and minimax coefficients
// ========================================================================
namespace {

constexpr double kPI_HI = 3.141592653589793116;
constexpr double kPI_LO = 1.2246467991473532072e-16;
constexpr double kPI_2_HI = 1.570796326794896558;
constexpr double kPI_2_LO = 6.1232339957367660360e-17;

// SLEEF 3.6 xatan minimax coefficients (from atan2k_u1, src/libm/sleefdp.c).
// atan(s) = s + s * t * P(t), with t = s² and P(0) = -1/3.
constexpr double kAtanMinimax[] = {
    -1.88796008463073496563746e-05, 0.000209850076645816976906797, -0.00110611831486672482563471,
    0.00370026744188713119232403,   -0.00889896195887655491740809, 0.016599329773529201970117,
    -0.0254517624932312641616861,   0.0337852580001353069993897,   -0.0407629191276836500001934,
    0.0466667150077840625632675,    -0.0523674852303482457616113,  0.0587666392926673580854313,
    -0.0666573579361080525984562,   0.0769219538311769618355029,   -0.090908995008245008229153,
    0.111111105648261418443745,     -0.14285714266771329383765,    0.199999999996591265594148,
    -0.333333333333311110369124,
};

// SLEEF 3.6 xasin minimax (from src/libm/sleefdp.c::asin).
constexpr double kAsinMinimax[] = {
    0.3161587650653934628e-1, -0.1581918243329996643e-1, 0.1929045477267910674e-1,
    0.6606077476277170610e-2, 0.1215360525577377331e-1,  0.1388715184501609218e-1,
    0.1735956991223614604e-1, 0.2237176181932048341e-1,  0.3038195928038132237e-1,
    0.4464285681377102438e-1, 0.7500000000378581611e-1,  0.1666666666666497543e+0,
};

inline double atan2k_core_poly(double s, soft_fp64::sleef::sf64_internal_fe_acc& fe) {
    return poly_array(s, kAtanMinimax, sizeof(kAtanMinimax) / sizeof(kAtanMinimax[0]));
}

// atan2k_u1(y, x): DD-carrying SLEEF-style atan2(|y|, |x|). Returns result
// in DD. Swaps (y,x) when |y|>|x| so the polynomial argument stays ≤ 1,
// eliminating the 0.414 reduction boundary that hurt the naive xatan.
DD atan2k_u1_dd(DD y, DD x, soft_fp64::sleef::sf64_internal_fe_acc& fe) {
    double q = 0.0;
    if (lt_(x.hi, 0.0)) {
        x.hi = sf64_neg(x.hi);
        x.lo = sf64_neg(x.lo);
        q = -2.0;
    }
    if (lt_(x.hi, y.hi)) {
        const DD t = x;
        x = y;
        y.hi = sf64_neg(t.hi);
        y.lo = sf64_neg(t.lo);
        q = sf64_add(q, 1.0);
    }

    DD s = dddiv_dd_dd_dd(y, x);
    DD t = ddsqu_dd_dd(s);
    t = ddnormalize_dd_dd(t);

    const double u = atan2k_core_poly(t.hi, fe);

    DD st = ddmul_dd_dd_dd(s, t);
    DD stu = ddmul_dd_dd_d(st, u);
    DD r = ddadd2_dd_dd(s, stu);

    if (ne_(q, 0.0)) {
        DD qpi2 = ddmul_dd_d_d(q, kPI_2_HI);
        qpi2.lo = sf64_fma(q, kPI_2_LO, qpi2.lo);
        r = ddadd2_dd_dd(qpi2, r);
    }
    return r;
}

inline DD dd_of(double x) {
    return DD{x, 0.0};
}

} // namespace

namespace soft_fp64::sleef {

// log|d| as DD. Port of SLEEF 3.6 sleefdp.c::logk.  Hidden-visibility so the
// symbol is usable from other SLEEF TUs (erfc/tgamma arg-reduction) without
// joining the public ABI.
[[gnu::visibility("hidden")]] DD sf64_internal_logk_dd(double d, sf64_internal_fe_acc& fe) {
    int e;
    double m = sf64_frexp(d, &e);
    if (lt_(m, 0.70710678118654752440)) {
        m = sf64_mul(m, 2.0);
        e -= 1;
    }

    DD mdd = dd_of(m);
    DD num = ddadd2_dd_dd_d(mdd, -1.0);
    DD den = ddadd2_dd_dd_d(mdd, 1.0);
    DD x = dddiv_dd_dd_dd(num, den);
    DD x2 = ddsqu_dd_dd(x);

    // SLEEF 3.6 logk coefficients.
    constexpr double kLogkCoef[] = {
        0.153487338491425068243146, 0.152519917006351951593857, 0.181863266251982985677316,
        0.222221366518767365905163, 0.285714294746548025383248, 0.399999999950799600689777,
        0.6666666666667778740063,
    };
    // DD Horner on the full x² DD pair. Evaluating the tail polynomial in
    // double-double carries ~100+ bits through the accumulator, compared to
    // ~53 bits when the tail was run against `x2.hi` as a plain double.
    DD t = DD{kLogkCoef[0], 0.0};
    constexpr int kLogkN = sizeof(kLogkCoef) / sizeof(kLogkCoef[0]);
    for (int i = 1; i < kLogkN; ++i) {
        t = ddmul_dd_dd_dd(t, x2);
        t = ddadd2_dd_dd_d(t, kLogkCoef[i]);
    }

    DD s = ddmul_dd_d_d(kL2U, sf64_from_i64(static_cast<int64_t>(e)));
    s.lo = sf64_fma(kL2L, sf64_from_i64(static_cast<int64_t>(e)), s.lo);
    s = ddadd2_dd_dd(s, ddscale_dd_dd_d(x, 2.0));
    DD xx2 = ddmul_dd_dd_dd(x, x2);
    s = ddadd2_dd_dd(s, ddmul_dd_dd_dd(xx2, t));
    return s;
}

// exp(DD) → double. Port of SLEEF 3.6 sleefdp.c::expk.  Hidden-visibility,
// reused by erfc/tgamma for DD-accurate exp argument reduction.
//
// SLEEF reconstruction after range reduction (with 8-term poly ending at 1/4!):
//   t = s*u + 1/6                // DD (scalar·DD + scalar)
//   t = s*t + 1/2                // DD
//   t = s + s²·t                 // DD
//   t = 1 + t                    // DD
//   result = ldexp(t.hi + t.lo, q)
[[gnu::visibility("hidden")]] double sf64_internal_expk_dd(DD d, sf64_internal_fe_acc& fe) {
    const double d_collapsed = sf64_add(d.hi, d.lo);

    if (gt_(d_collapsed, 709.78271289338399673222))
        return kInf;
    if (lt_(d_collapsed, -1000.0))
        return 0.0;

    const double qf = sf64_rint(sf64_mul(d_collapsed, kR_LN2));
    const int q = sf64_to_i32(qf);

    DD s = ddadd2_dd_dd_d(d, sf64_mul(qf, sf64_neg(kL2U)));
    s = ddadd2_dd_dd_d(s, sf64_mul(qf, sf64_neg(kL2L)));
    s = ddnormalize_dd_dd(s);

    // SLEEF 3.6 expk 8-term tail poly (ends at coefficient for s⁴ = 1/4!).
    // The remaining three leading terms (1, s, s²/2, s³/6) are reconstructed
    // in DD below for precision.
    constexpr double kExpkCoef8[] = {
        2.51069683420950419527139e-08, // s^10-ish
        2.76286166770270649116855e-07, 2.75572496725023574143864e-06, 2.48014973989819794114153e-05,
        0.000198412698809069797676111, 0.0013888888939977128960529,   0.00833333333332371417601081,
        0.0416666666665409524128449, // 1/4!
    };
    const double u = poly_array(s.hi, kExpkCoef8, sizeof(kExpkCoef8) / sizeof(kExpkCoef8[0]));

    // SLEEF reconstruction:
    //   t = s·u + 1/6   (DD: scalar u times DD s, add scalar 1/6)
    DD t = ddadd2_dd_dd_d(ddmul_dd_dd_d(s, u), 0.1666666666666666574e+0);
    //   t = s·t + 1/2
    t = ddadd2_dd_dd_d(ddmul_dd_dd_dd(s, t), 0.5);
    //   t = s + s²·t
    DD s2 = ddsqu_dd_dd(s);
    t = ddadd2_dd_dd(s, ddmul_dd_dd_dd(s2, t));
    //   t = 1 + t
    t = ddadd2_dd_d_dd(1.0, t);

    const double yy = sf64_add(t.hi, t.lo);
    return sf64_ldexp(yy, q);
}

// exp(DD) → DD. Port of SLEEF 3.6 sleefdp.c::expk2 (the DD-result form
// used by xtgamma_u1 reconstruction and xerfc_u15 deep-tail). Same
// reduction as expk_dd above (q = round((d.hi+d.lo)*log2(e)); s = d - q·ln2)
// but the final ldexp is applied componentwise to keep the result in DD.
//
// Hidden visibility — used by sleef_stubs.cpp's gamma/erfc ports, never
// part of the public ABI. The deep-underflow guard (`d.x < -1000 ⇒ 0`)
// matches upstream so the caller's expk2(d)·u multiplication doesn't
// produce a denormal trail.
[[gnu::visibility("hidden")]] DD sf64_internal_expk2_dd(DD d, sf64_internal_fe_acc& fe) {
    const double d_collapsed = sf64_add(d.hi, d.lo);
    const double qf = sf64_rint(sf64_mul(d_collapsed, kR_LN2));
    const int q = sf64_to_i32(qf);

    DD s = ddadd2_dd_dd_d(d, sf64_mul(qf, sf64_neg(kL2U)));
    s = ddadd2_dd_dd_d(s, sf64_mul(qf, sf64_neg(kL2L)));

    // SLEEF 3.6 expk2 10-term tail polynomial — Horner in s.hi (= s.x).
    // (The polynomial degree differs from expk_dd: expk2 carries the full
    // 10 terms whereas expk_dd lifts the leading three terms into the DD
    // reconstruction. expk2 keeps everything inside the DD reconstruction.)
    double u = +0.1602472219709932072e-9;
    u = sf64_fma(u, s.hi, +0.2092255183563157007e-8);
    u = sf64_fma(u, s.hi, +0.2505230023782644465e-7);
    u = sf64_fma(u, s.hi, +0.2755724800902135303e-6);
    u = sf64_fma(u, s.hi, +0.2755731892386044373e-5);
    u = sf64_fma(u, s.hi, +0.2480158735605815065e-4);
    u = sf64_fma(u, s.hi, +0.1984126984148071858e-3);
    u = sf64_fma(u, s.hi, +0.1388888888886763255e-2);
    u = sf64_fma(u, s.hi, +0.8333333333333347095e-2);
    u = sf64_fma(u, s.hi, +0.4166666666666669905e-1);

    DD t = ddadd2_dd_dd_d(ddmul_dd_dd_d(s, u), +0.1666666666666666574e+0);
    t = ddadd2_dd_dd_d(ddmul_dd_dd_dd(s, t), 0.5);
    t = ddadd2_dd_dd(s, ddmul_dd_dd_dd(ddsqu_dd_dd(s), t));

    t = ddadd2_dd_d_dd(1.0, t);

    // Componentwise ldexp by 2^q. sf64_ldexp scales by 2^q exactly when
    // |d| stays in normal range; the subsequent ldexp on t.lo can produce
    // a denormal which is what we want — it gets absorbed in the consumer's
    // collapse.
    t.hi = sf64_ldexp(t.hi, q);
    t.lo = sf64_ldexp(t.lo, q);

    return lt_(d.hi, -1000.0) ? DD{0.0, 0.0} : t;
}

// log(DD) → DD. Port of SLEEF 3.6 sleefdp.c::logk2 (lgamma_u1 calls it on
// the absolute value of the Lanczos quotient).
//
// Reduction: e = ilogb(d.hi * 4/3); m = d / 2^e; x = (m-1)/(m+1).
//   log(d) = e·log(2) + 2x·(1 + x²·P(x²))
// Polynomial is degree-7 in x²; the lead term is reconstructed in DD as
// 2x + 2x·x²·P, then `e·ln2` is added with a DD ln2 split.
//
// Uses sf64_frexp/ldexp_2k to keep the scaling FPU-free.
[[gnu::visibility("hidden")]] DD sf64_internal_logk2_dd(DD d, sf64_internal_fe_acc& fe) {
    // SLEEF: e = ilogbk(d.x * (1.0/0.75)). We approximate with frexp on the
    // shifted value: frexp returns (m, e) with |m| ∈ [0.5, 1). For the SLEEF
    // form we want e such that d.hi/2^e ∈ [0.75, 1.5) approximately. The
    // (1.0/0.75) shift turns this into [1, 2), and ilogb extracts the unbiased
    // exponent. frexp gives us m·2^e in [0.5, 1) range, so we adjust by 1.
    int e_frexp;
    (void)sf64_frexp(sf64_mul(d.hi, 1.0 / 0.75), &e_frexp);
    const int e = e_frexp - 1; // ilogb = frexp_e - 1

    // m = d / 2^e (componentwise); ldexp by -e is exact.
    DD m;
    m.hi = sf64_ldexp(d.hi, -e);
    m.lo = sf64_ldexp(d.lo, -e);

    DD x = dddiv_dd_dd_dd(ddadd2_dd_dd_d(m, -1.0), ddadd2_dd_dd_d(m, 1.0));
    DD x2 = ddsqu_dd_dd(x);

    // SLEEF 3.6 logk2 7-term polynomial in x² (Horner in x2.hi).
    constexpr double kLogk2Coef[] = {
        0.13860436390467167910856,  0.131699838841615374240845, 0.153914168346271945653214,
        0.181816523941564611721589, 0.22222224632662035403996,  0.285714285511134091777308,
        0.400000000000914013309483,
    };
    constexpr int kLogk2N = sizeof(kLogk2Coef) / sizeof(kLogk2Coef[0]);
    double t = kLogk2Coef[0];
    for (int i = 1; i < kLogk2N; ++i) {
        t = sf64_fma(t, x2.hi, kLogk2Coef[i]);
    }
    t = sf64_fma(t, x2.hi, 0.666666666666664853302393);

    // s = ln2 · e (DD), + 2x, + x³·t (where x³ = x·x²)
    DD s = ddmul_dd_dd_d(DD{0.693147180559945286226764, 2.319046813846299558417771e-17},
                         sf64_from_i32(e));
    s = ddadd_dd_dd_dd(s, ddscale_dd_dd_d(x, 2.0));
    s = ddadd_dd_dd_dd(s, ddmul_dd_dd_d(ddmul_dd_dd_dd(x2, x), t));

    return s;
}

} // namespace soft_fp64::sleef

// ========================================================================
// asin / acos / atan / atan2  (port of SLEEF 3.6 xasin/xacos/xatan/xatan2)
// ========================================================================

extern "C" double sf64_asin(double d) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(d))
        return qNaN();
    const double a = sf64_fabs(d);
    if (gt_(a, 1.0))
        return qNaN();

    const bool big = gt_(a, 0.5);
    double x2, x;
    if (big) {
        x2 = sf64_mul(sf64_sub(1.0, a), 0.5);
        x = sf64_sqrt(x2);
    } else {
        x = a;
        x2 = sf64_mul(x, x);
    }

    const double p = poly_array(x2, kAsinMinimax, sizeof(kAsinMinimax) / sizeof(kAsinMinimax[0]));

    const double inner = sf64_mul(x2, p);
    const double xip = sf64_mul(x, inner);
    DD u_dd = ddadd2_dd_d_d(x, xip);

    if (!big) {
        const double r = sf64_add(u_dd.hi, u_dd.lo);
        fe.flush();
        return signbit_(d) ? sf64_neg(r) : r;
    }

    DD two_u = ddscale_dd_dd_d(u_dd, 2.0);
    DD pi_2{kPI_2_HI, kPI_2_LO};
    DD r = ddadd2_dd_dd(pi_2, ddneg_dd_dd(two_u));
    const double rr = sf64_add(r.hi, r.lo);
    fe.flush();
    return signbit_(d) ? sf64_neg(rr) : rr;
}

extern "C" double sf64_acos(double d) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(d))
        return qNaN();
    const double a = sf64_fabs(d);
    if (gt_(a, 1.0))
        return qNaN();

    const bool big = gt_(a, 0.5);
    double x2, x;
    if (big) {
        x2 = sf64_mul(sf64_sub(1.0, a), 0.5);
        x = sf64_sqrt(x2);
    } else {
        x = a;
        x2 = sf64_mul(x, x);
    }

    const double p = poly_array(x2, kAsinMinimax, sizeof(kAsinMinimax) / sizeof(kAsinMinimax[0]));
    const double inner = sf64_mul(x2, p);
    const double xip = sf64_mul(x, inner);
    DD u_dd = ddadd2_dd_d_d(x, xip);

    if (!big) {
        DD pi_2{kPI_2_HI, kPI_2_LO};
        DD r = signbit_(d) ? ddadd2_dd_dd(pi_2, u_dd) : ddadd2_dd_dd(pi_2, ddneg_dd_dd(u_dd));
        const double rr = sf64_add(r.hi, r.lo);
        fe.flush();
        return rr;
    }

    DD two_u = ddscale_dd_dd_d(u_dd, 2.0);
    if (!signbit_(d)) {
        const double rr = sf64_add(two_u.hi, two_u.lo);
        fe.flush();
        return rr;
    }
    DD pi{kPI_HI, kPI_LO};
    DD r = ddadd2_dd_dd(pi, ddneg_dd_dd(two_u));
    const double rr = sf64_add(r.hi, r.lo);
    fe.flush();
    return rr;
}

extern "C" double sf64_atan(double d) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(d))
        return qNaN();
    if (isinf_(d))
        return signbit_(d) ? sf64_neg(kPI_2) : kPI_2;
    if (eq_(d, 0.0))
        return d;

    DD y = dd_of(sf64_fabs(d));
    DD x = dd_of(1.0);
    DD r = atan2k_u1_dd(y, x, fe);
    const double rr = sf64_add(r.hi, r.lo);
    fe.flush();
    return signbit_(d) ? sf64_neg(rr) : rr;
}

extern "C" double sf64_atan2(double y, double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x) || isnan_(y))
        return qNaN();

    if (eq_(x, 0.0) && eq_(y, 0.0)) {
        if (signbit_(x))
            return signbit_(y) ? sf64_neg(kPI) : kPI;
        return signbit_(y) ? sf64_neg(0.0) : 0.0;
    }
    if (eq_(y, 0.0)) {
        return signbit_(x) ? (signbit_(y) ? sf64_neg(kPI) : kPI)
                           : (signbit_(y) ? sf64_neg(0.0) : 0.0);
    }
    if (eq_(x, 0.0))
        return signbit_(y) ? sf64_neg(kPI_2) : kPI_2;

    if (isinf_(x) && isinf_(y)) {
        const double v = signbit_(x) ? sf64_mul(kPI, 0.75) : sf64_mul(kPI, 0.25);
        return signbit_(y) ? sf64_neg(v) : v;
    }
    if (isinf_(x)) {
        return signbit_(x) ? (signbit_(y) ? sf64_neg(kPI) : kPI)
                           : (signbit_(y) ? sf64_neg(0.0) : 0.0);
    }
    if (isinf_(y))
        return signbit_(y) ? sf64_neg(kPI_2) : kPI_2;

    DD yd = dd_of(sf64_fabs(y));
    DD xd = dd_of(sf64_fabs(x));
    DD r = atan2k_u1_dd(yd, xd, fe);
    double rr = sf64_add(r.hi, r.lo);

    if (signbit_(x))
        rr = sf64_sub(kPI, rr);
    fe.flush();
    return signbit_(y) ? sf64_neg(rr) : rr;
}

// ========================================================================
// pow / cbrt  — port of SLEEF 3.6 xpow / xcbrt_u1
// ========================================================================

extern "C" double sf64_pow(double x, double y) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (eq_(y, 0.0))
        return 1.0;
    if (eq_(x, 1.0))
        return 1.0;
    if (isnan_(x) || isnan_(y))
        return qNaN();

    const bool y_is_int = is_int(y);
    const bool y_is_odd = y_is_int && is_odd_int(y);

    if (eq_(x, 0.0)) {
        if (lt_(y, 0.0)) {
            return (signbit_(x) && y_is_odd) ? sf64_neg(kInf) : kInf;
        }
        return (signbit_(x) && y_is_odd) ? sf64_neg(0.0) : 0.0;
    }

    if (lt_(x, 0.0) && !y_is_int)
        return qNaN();

    if (isinf_(x)) {
        if (gt_(x, 0.0))
            return gt_(y, 0.0) ? kInf : 0.0;
        const double r = gt_(y, 0.0) ? kInf : 0.0;
        return y_is_odd ? sf64_neg(r) : r;
    }
    if (isinf_(y)) {
        if (eq_(sf64_fabs(x), 1.0))
            return 1.0;
        return (lt_(sf64_fabs(x), 1.0)) == (lt_(y, 0.0)) ? kInf : 0.0;
    }

    // DD composition: x^y = exp(y * log|x|). log in DD, y*log in DD.
    DD l = soft_fp64::sleef::sf64_internal_logk_dd(sf64_fabs(x), fe);
    DD yl = ddmul_dd_dd_d(l, y);
    double r = soft_fp64::sleef::sf64_internal_expk_dd(yl, fe);
    if (lt_(x, 0.0) && y_is_odd)
        r = sf64_neg(r);
    fe.flush();
    return r;
}

extern "C" double sf64_powr(double x, double y) {
    // IEEE 754-2019 §9.2.1 strict domain semantics. Every degenerate
    // case returns qNaN + INVALID except the pole at x=0, y<0 (which
    // returns +inf + DIVBYZERO). -0 is treated as a zero — `lt_` uses
    // ordered-less-than, so -0.0 < 0.0 is false.
    if (isnan_(x) || isnan_(y))
        return qNaN();
    if (lt_(x, 0.0)) {
        SF64_FE_RAISE(SF64_FE_INVALID);
        return qNaN();
    }
    const bool x_zero = eq_(x, 0.0);
    const bool x_one = eq_(x, 1.0);
    const bool x_inf = isinf_(x);
    const bool y_zero = eq_(y, 0.0);
    const bool y_inf = isinf_(y);
    // 0^0, (+∞)^0 → qNaN
    if ((x_zero || x_inf) && y_zero) {
        SF64_FE_RAISE(SF64_FE_INVALID);
        return qNaN();
    }
    // 1^(±∞) → qNaN
    if (x_one && y_inf) {
        SF64_FE_RAISE(SF64_FE_INVALID);
        return qNaN();
    }
    // 0^(y<0) → +∞ (pole)
    if (x_zero && lt_(y, 0.0)) {
        SF64_FE_RAISE(SF64_FE_DIVBYZERO);
        return soft_fp64::sleef::from_bits(0x7FF0000000000000ULL); // +inf
    }
    // 0^(y>0) → +0. Delegating to sf64_pow here would return -0 for
    // odd-integer y (standard pow semantics), but powr's result is
    // always nonneg on the positive-base half. At this point y is
    // neither NaN nor zero nor negative, so it's strictly positive.
    if (x_zero)
        return 0.0;
    // Non-exceptional: the positive-base pow limits at ±∞ match the
    // powr spec (x>1,+∞→+∞; x>1,-∞→+0; 0<x<1,+∞→+0; 0<x<1,-∞→+∞;
    // +∞,y>0→+∞; +∞,y<0→+0), so delegate.
    return sf64_pow(x, y);
}

extern "C" double sf64_pown(double x, int n) {
    return sf64_pow(x, sf64_from_i32(n));
}

extern "C" double sf64_cbrt(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x))
        return qNaN();
    if (eq_(x, 0.0) || isinf_(x))
        return x;

    const double a = sf64_fabs(x);

    // Decompose a = mant * 2^e with mant ∈ [0.5, 1). Split e = 3*q + rem
    // (rem ∈ {0,1,2}) so cbrt(a) = cbrt(mant * 2^rem) * 2^q. This lifts
    // subnormal inputs into the normal range before log/exp, avoiding the
    // underflow in sf64_internal_log_core that turned cbrt(denorm_min) into NaN.
    int e;
    const double mant = sf64_frexp(a, &e);
    int rem = e % 3;
    if (rem < 0)
        rem += 3;
    const int q = (e - rem) / 3;
    const double z = sf64_ldexp(mant, rem); // z ∈ [0.5, 4)

    // Seed via exp(log(z)/3) — ~15 bits accurate.
    double r = sf64_internal_exp_core(sf64_div(sf64_internal_log_core(z, fe), 3.0), fe);

    // Plain-double Newton step.
    {
        const double r2 = sf64_mul(r, r);
        const double q2 = sf64_div(z, r2);
        const double diff = sf64_sub(r, q2);
        r = sf64_sub(r, sf64_div(diff, 3.0));
    }

    // DD Newton step for last-ULP accuracy:  r ← r * (2 + z/r³) / 3.
    {
        DD rdd = dd_of(r);
        DD r3 = ddmul_dd_dd_d(ddsqu_dd_dd(rdd), r);
        DD adr3 = dddiv_dd_dd_dd(dd_of(z), r3);
        DD tp = ddadd2_dd_dd_d(adr3, 2.0);
        DD rtp = ddmul_dd_dd_dd(rdd, tp);
        constexpr double kOneThirdHi = 0.33333333333333331;
        constexpr double kOneThirdLo = 1.8503717077085941e-17;
        DD third{kOneThirdHi, kOneThirdLo};
        DD res = ddmul_dd_dd_dd(rtp, third);
        r = sf64_add(res.hi, res.lo);
    }

    // Recombine: cbrt(a) = r * 2^q.
    const double out = sf64_ldexp(r, q);
    fe.flush();
    return signbit_(x) ? sf64_neg(out) : out;
}

// ========================================================================
// Hyperbolic — Taylor for small |x|, exp-based for mid/large
// ========================================================================

extern "C" double sf64_sinh(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x))
        return qNaN();
    if (isinf_(x))
        return x;
    const double a = sf64_fabs(x);

    if (lt_(a, 1.0)) {
        const double x2 = sf64_mul(x, x);
        constexpr double kC[] = {
            1.0 / 355687428096000.0, 1.0 / 1307674368000.0, 1.0 / 6227020800.0, 1.0 / 39916800.0,
            1.0 / 362880.0,          1.0 / 5040.0,          1.0 / 120.0,        1.0 / 6.0,
        };
        const double p = poly_array(x2, kC, sizeof(kC) / sizeof(kC[0]));
        const double r = sf64_fma(sf64_mul(x, x2), p, x);
        fe.flush();
        return r;
    }

    if (lt_(a, 18.0)) {
        const double e = sf64_internal_exp_core(x, fe);
        const double en = sf64_internal_exp_core(sf64_neg(x), fe);
        const double r = sf64_mul(sf64_sub(e, en), 0.5);
        fe.flush();
        return r;
    }

    // sinh overflow boundary is log(2·DBL_MAX) ≈ 710.4758600739439, not
    // log(DBL_MAX) ≈ 709.7827 (the exp overflow threshold). In the window
    // (709.78, 710.4758] exp(a) overflows but sinh(a) ≈ exp(a)/2 still fits.
    // Compute exp(a - ln2) via expk_dd on the DD pair (a - kL2U, -kL2L) so
    // the ln2 subtraction keeps its low bits: a plain double `a - ln2` loses
    // ≈ ulp(a) ≈ 1e-13, and exp's derivative near 710 is 1.6e308, which
    // amplifies that loss into ≈ 450 ULP of error at the result.
    if (gt_(a, 710.4758600739439))
        return signbit_(x) ? sf64_neg(kInf) : kInf;
    if (gt_(a, 709.78)) {
        DD r = ddadd2_dd_d_d(a, sf64_neg(kL2U));
        r = ddadd2_dd_dd_d(r, sf64_neg(kL2L));
        const double big = soft_fp64::sleef::sf64_internal_expk_dd(r, fe);
        fe.flush();
        return signbit_(x) ? sf64_neg(big) : big;
    }
    // Large |x|: e^-|x| is ≪ 2^-52 of e^|x|, so sinh ≈ ±e^|x|/2. Always
    // evaluate exp on the positive magnitude and restore sign at the end —
    // evaluating on a negative x gives e^-|x| (tiny), which is the wrong
    // branch of sinh.
    const double half = sf64_mul(sf64_internal_exp_core(a, fe), 0.5);
    fe.flush();
    return signbit_(x) ? sf64_neg(half) : half;
}

extern "C" double sf64_cosh(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x))
        return qNaN();
    if (isinf_(x))
        return kInf;
    const double a = sf64_fabs(x);

    if (lt_(a, 0.25)) {
        const double x2 = sf64_mul(x, x);
        constexpr double kC[] = {
            1.0 / 87178291200.0, 1.0 / 479001600.0, 1.0 / 3628800.0, 1.0 / 40320.0,
            1.0 / 720.0,         1.0 / 24.0,        1.0 / 2.0,
        };
        const double p = poly_array(x2, kC, sizeof(kC) / sizeof(kC[0]));
        const double r = sf64_fma(x2, p, 1.0);
        fe.flush();
        return r;
    }

    if (gt_(a, 709.78))
        return kInf;
    const double e1 = sf64_internal_exp_core(x, fe);
    const double e2 = sf64_internal_exp_core(sf64_neg(x), fe);
    const double r = sf64_mul(sf64_add(e1, e2), 0.5);
    fe.flush();
    return r;
}

extern "C" double sf64_tanh(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x))
        return qNaN();
    if (isinf_(x))
        return signbit_(x) ? -1.0 : 1.0;

    const double a = sf64_fabs(x);

    if (lt_(a, 0.5)) {
        const double x2 = sf64_mul(x, x);
        constexpr double kSH[] = {
            1.0 / 6227020800.0, 1.0 / 39916800.0, 1.0 / 362880.0,
            1.0 / 5040.0,       1.0 / 120.0,      1.0 / 6.0,
        };
        constexpr double kCH[] = {
            1.0 / 87178291200.0, 1.0 / 479001600.0, 1.0 / 3628800.0, 1.0 / 40320.0,
            1.0 / 720.0,         1.0 / 24.0,        1.0 / 2.0,
        };
        const double ps = poly_array(x2, kSH, sizeof(kSH) / sizeof(kSH[0]));
        const double pc = poly_array(x2, kCH, sizeof(kCH) / sizeof(kCH[0]));
        const double sh = sf64_fma(sf64_mul(x, x2), ps, x);
        const double ch = sf64_fma(x2, pc, 1.0);
        const double r = sf64_div(sh, ch);
        fe.flush();
        return r;
    }
    if (gt_(a, 20.0))
        return signbit_(x) ? -1.0 : 1.0;

    const double e = sf64_internal_exp_core(sf64_mul(x, 2.0), fe);
    const double r = sf64_div(sf64_sub(e, 1.0), sf64_add(e, 1.0));
    fe.flush();
    return r;
}

// ========================================================================
// Inverse hyperbolic — DD-carried argument construction
// ========================================================================

extern "C" double sf64_asinh(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x))
        return qNaN();
    if (isinf_(x))
        return x;
    const double a = sf64_fabs(x);

    if (le_(a, 1.0)) {
        DD x2 = ddmul_dd_d_d(x, x);
        DD x2p1 = ddadd2_dd_dd_d(x2, 1.0);
        const double sroot = sf64_sqrt(sf64_add(x2p1.hi, x2p1.lo));
        const double denom = sf64_add(sroot, 1.0);
        DD t_dd = ddmul_dd_dd_d(x2, sf64_div(1.0, denom));
        DD arg = ddadd2_dd_dd_d(t_dd, x);
        const double r = sf64_log1p(sf64_add(arg.hi, arg.lo));
        fe.flush();
        return r;
    }

    DD x2 = ddmul_dd_d_d(a, a);
    DD x2p1 = ddadd2_dd_dd_d(x2, 1.0);
    const double sroot = sf64_sqrt(sf64_add(x2p1.hi, x2p1.lo));
    DD arg = ddadd2_dd_d_d(a, sroot);
    double r = sf64_internal_log_core(sf64_add(arg.hi, arg.lo), fe);
    fe.flush();
    return signbit_(x) ? sf64_neg(r) : r;
}

extern "C" double sf64_acosh(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x))
        return qNaN();
    if (lt_(x, 1.0))
        return qNaN();
    if (isinf_(x))
        return kInf;
    if (eq_(x, 1.0))
        return 0.0;

    // Near x==1, use the factored form to avoid (x²-1) cancellation.
    if (lt_(x, 2.0)) {
        const double xm1 = sf64_sub(x, 1.0);
        const double xp1 = sf64_add(x, 1.0);
        const double s = sf64_sqrt(sf64_mul(xm1, xp1));
        const double r = sf64_log1p(sf64_add(xm1, s));
        fe.flush();
        return r;
    }

    DD x2 = ddmul_dd_d_d(x, x);
    DD x2m1 = ddadd2_dd_dd_d(x2, -1.0);
    const double sroot = sf64_sqrt(sf64_add(x2m1.hi, x2m1.lo));
    DD arg = ddadd2_dd_d_d(x, sroot);
    const double r = sf64_internal_log_core(sf64_add(arg.hi, arg.lo), fe);
    fe.flush();
    return r;
}

extern "C" double sf64_atanh(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x))
        return qNaN();
    const double a = sf64_fabs(x);
    if (gt_(a, 1.0))
        return qNaN();
    if (eq_(a, 1.0))
        return signbit_(x) ? sf64_neg(kInf) : kInf;

    if (lt_(a, 0.5)) {
        const double t = sf64_div(sf64_mul(x, 2.0), sf64_sub(1.0, x));
        const double r = sf64_mul(sf64_log1p(t), 0.5);
        fe.flush();
        return r;
    }

    DD numer = ddadd2_dd_d_d(1.0, x);
    DD denom = ddadd2_dd_d_d(1.0, sf64_neg(x));
    DD q = dddiv_dd_dd_dd(numer, denom);
    double lq = sf64_internal_log_core(sf64_add(q.hi, q.lo), fe);
    const double r = sf64_mul(lq, 0.5);
    fe.flush();
    return r;
}
