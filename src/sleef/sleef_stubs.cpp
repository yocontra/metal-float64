//
// Derived from SLEEF 3.6 `src/libm/sleefdp.c` (Boost-1.0); see
// `src/sleef/NOTICE` for the upstream URL, the pinned commit SHA, and the
// full Boost-1.0 license text.
//
// Holds the long-tail transcendentals (erf, erfc, tgamma, lgamma,
// lgamma_r, asinpi, acospi, atanpi, atan2pi, rootn) ported from SLEEF
// 3.6 `xerf_u1`, `xerfc_u15`, `xtgamma_u1`, `xlgamma_u1`. All
// arithmetic flows through the `sf64_*` ABI.
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
using soft_fp64::sleef::ddabs_dd_dd;
using soft_fp64::sleef::ddadd2_dd_d_d;
using soft_fp64::sleef::ddadd2_dd_d_dd;
using soft_fp64::sleef::ddadd2_dd_dd;
using soft_fp64::sleef::ddadd2_dd_dd_d;
using soft_fp64::sleef::ddadd_dd_dd_dd;
using soft_fp64::sleef::dddiv_dd_dd_dd;
using soft_fp64::sleef::ddmul_d_dd_dd;
using soft_fp64::sleef::ddmul_dd_d_d;
using soft_fp64::sleef::ddmul_dd_dd_d;
using soft_fp64::sleef::ddmul_dd_dd_dd;
using soft_fp64::sleef::ddneg_dd_dd;
using soft_fp64::sleef::ddnormalize_dd_dd;
using soft_fp64::sleef::ddrec_dd_d;
using soft_fp64::sleef::ddrec_dd_dd;
using soft_fp64::sleef::ddscale_dd_dd_d;
using soft_fp64::sleef::ddsqu_dd_dd;
using soft_fp64::sleef::ddsub_dd_dd_dd;
using soft_fp64::sleef::eq_;
using soft_fp64::sleef::ge_;
using soft_fp64::sleef::gt_;
using soft_fp64::sleef::isinf_;
using soft_fp64::sleef::isnan_;
using soft_fp64::sleef::le_;
using soft_fp64::sleef::lt_;
using soft_fp64::sleef::ne_;
using soft_fp64::sleef::poly_array;
using soft_fp64::sleef::sf64_internal_expk2_dd;
using soft_fp64::sleef::sf64_internal_expk_dd;
using soft_fp64::sleef::sf64_internal_logk2_dd;
using soft_fp64::sleef::sf64_internal_logk_dd;
using soft_fp64::sleef::sf64_internal_sinpik_dd;
using soft_fp64::sleef::signbit_;
using soft_fp64::sleef::detail::is_int;
using soft_fp64::sleef::detail::kInf;
using soft_fp64::sleef::detail::kPI;
using soft_fp64::sleef::detail::qNaN;

namespace {

// ---- shared constants --------------------------------------------------
//
// 1 / pi as a double-double.  hi is the nearest-double to 1/pi; lo is the
// residual rounded to nearest double.  Taken from SLEEF
// `src/libm/sleefdp.c`.
constexpr double kInvPI_hi = 0.3183098861837906715;
constexpr double kInvPI_lo = -1.9678676675182486e-17;

} // namespace

// ========================================================================
// pi-scaled inverse trig (asinpi / acospi / atanpi / atan2pi)
// ========================================================================
//
// Idea: compute the forward inverse-trig result with the existing real
// `sf64_asin` / `sf64_acos` / `sf64_atan` / `sf64_atan2` primitives, then
// multiply by 1/pi carried in double-double.  This avoids losing a bit to
// the final divide.  The angle itself is only a plain double so the
// best we can get is about (ulp(angle) + ulp(angle) * |lo/hi|) ~ 1 ULP.
// Well within the U35 (<= 8 ULP) band that the test spec pins for asinpi et al.

extern "C" double sf64_asinpi(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    const double a = sf64_asin(x);
    if (isnan_(a) || isinf_(a))
        return a;
    // (a + 0) * (1/pi hi + 1/pi lo)
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
    // atan(+/-inf) = +/- pi/2 -> atanpi(+/-inf) = +/-0.5
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

    // rootn(+/-0, n): +0 or +/-0 for n>0 (sign preserved for odd n), +/-inf for n<0.
    if (eq_(x, 0.0)) {
        if (n > 0)
            return (signbit_(x) && n_odd) ? sf64_neg(0.0) : 0.0;
        // n < 0 -> pole at 0.
        return (signbit_(x) && n_odd) ? sf64_neg(kInf) : kInf;
    }

    if (isinf_(x)) {
        if (n > 0) {
            // rootn(+inf, n>0) = +inf; rootn(-inf, odd n>0) = -inf; else NaN.
            if (!signbit_(x))
                return kInf;
            return n_odd ? sf64_neg(kInf) : qNaN();
        }
        // n < 0 -> rootn(+/-inf, n<0) = +/-0 for odd n, +0 for even n.
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
// erf / erfc - port of SLEEF 3.6 xerf_u1 / xerfc_u15
// ========================================================================
//
// Faithful translation of `xerf_u1` and `xerfc_u15` from SLEEF 3.6
// `src/libm/sleefdp.c`.  Coefficients are taken verbatim from upstream
// (do not edit).  Each polynomial evaluation uses Horner-form
// `poly_array` (sleef_common.h) instead of upstream's Estrin POLY21 -
// the value is identical at infinite precision, the rounding error
// pattern differs by <= a few ULP.  This is the documented Horner-vs-
// Estrin tradeoff that takes our port from u1 (<=1 ULP) to U35
// (<=8 ULP) target.
//
// The DD reconstruction in xerf_u1 (4x ddsqu = 16th power) and the
// DD-arg expk2 in xerfc_u15 are preserved exactly.

namespace {

// --- xerf_u1 polynomials (SLEEF 3.6 sleefdp.c lines 2575-2596 / 2610-2631) ---

// Inner branch: x < 2.5.  21-term polynomial in x (Horner highest-first).
constexpr double kErfU1Mid[] = {
    -0.2083271002525222097e-14, +0.7151909970790897009e-13, -0.1162238220110999364e-11,
    +0.1186474230821585259e-10, -0.8499973178354613440e-10, +0.4507647462598841629e-9,
    -0.1808044474288848915e-8,  +0.5435081826716212389e-8,  -0.1143939895758628484e-7,
    +0.1215442362680889243e-7,  +0.1669878756181250355e-7,  -0.9808074602255194288e-7,
    +0.1389000557865837204e-6,  +0.2945514529987331866e-6,  -0.1842918273003998283e-5,
    +0.3417987836115362136e-5,  +0.3860236356493129101e-5,  -0.3309403072749947546e-4,
    +0.1060862922597579532e-3,  +0.2323253155213076174e-3,  +0.1490149719145544729e-3,
};

// Outer branch: 2.5 <= x <= 6.0. 21-term polynomial in x.
constexpr double kErfU1Tail[] = {
    -0.4024015130752621932e-18, +0.3847193332817048172e-16, -0.1749316241455644088e-14,
    +0.5029618322872872715e-13, -0.1025221466851463164e-11, +0.1573695559331945583e-10,
    -0.1884658558040203709e-9,  +0.1798167853032159309e-8,  -0.1380745342355033142e-7,
    +0.8525705726469103499e-7,  -0.4160448058101303405e-6,  +0.1517272660008588485e-5,
    -0.3341634127317201697e-5,  -0.2515023395879724513e-5,  +0.6539731269664907554e-4,
    -0.3551065097428388658e-3,  +0.1210736097958368864e-2,  -0.2605566912579998680e-2,
    +0.1252823202436093193e-2,  +0.1820191395263313222e-1,  -0.1021557155453465954e+0,
};

// SLEEF poly4dd: evaluates `(x*x)*(x*c3+c2) + (x*c1+c0)` where c0..c2 are
// DD constants and c3 is a plain double. Mirrors `poly4dd` upstream.
SF64_ALWAYS_INLINE DD poly4dd_(double x, double c3, DD c2, DD c1, DD c0,
                               soft_fp64::sleef::sf64_internal_fe_acc& fe) noexcept {
    const double x2 = sf64_mul(x, x);
    // p1 = c3*x + c2   (DD: scalar c3 times x is a plain double, then add c2 DD)
    const DD p1 = ddadd2_dd_d_dd(sf64_mul(c3, x), c2);
    // q1 = c1*x + c0   (DD * scalar + DD)
    const DD q1 = ddadd2_dd_dd(ddmul_dd_dd_d(c1, x), c0);
    // r = p1 * x^2 + q1 (DD*scalar + DD)
    return ddadd2_dd_dd(ddmul_dd_dd_d(p1, x2), q1);
}

} // namespace

extern "C" double sf64_erf(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x))
        return qNaN();
    // x = 0 short-circuit (preserves signed zero).
    if (eq_(x, 0.0))
        return x;
    if (isinf_(x))
        return signbit_(x) ? -1.0 : 1.0;

    const double xa = sf64_fabs(x);

    // Tiny argument: erf(x) ~ 2x/sqrt(pi).  SLEEF uses the same `1e-8` cutoff.
    // -1.12837... is -(2/sqrt(pi)); we apply the sign at the end via signbit_.
    if (lt_(xa, 1e-8)) {
        const double r = sf64_mul(1.12837916709551262756245475959, xa);
        fe.flush();
        return signbit_(x) ? sf64_neg(r) : r;
    }

    DD t2;
    if (lt_(xa, 2.5)) {
        // Inner branch: SLEEF's Abramowitz-and-Stegun rational form.
        //   t  = poly21(x; coeffs)
        //   t2 = poly4dd(x, t, [DD constants])
        //   t2 = 1 + t2 * x
        //   t2 = 1 / (t2)^16  (4x ddsqu)
        const double t = poly_array(xa, kErfU1Mid, 21);
        DD r2 = poly4dd_(xa, t, DD{0.0092877958392275604405, 7.9287559463961107493e-19},
                         DD{0.042275531758784692937, 1.3785226620501016138e-19},
                         DD{0.07052369794346953491, 9.5846628070792092842e-19}, fe);
        // r2 = 1 + r2 * x
        r2 = ddadd2_dd_d_dd(1.0, ddmul_dd_dd_d(r2, xa));
        // r2 = r2^16 (= ((((r2)^2)^2)^2)^2)
        r2 = ddsqu_dd_dd(r2);
        r2 = ddsqu_dd_dd(r2);
        r2 = ddsqu_dd_dd(r2);
        r2 = ddsqu_dd_dd(r2);
        // t2 = 1 / r2
        t2 = ddrec_dd_dd(r2);
    } else if (gt_(xa, 6.0)) {
        // erf(x) -> 1 for |x| > 6.  SLEEF computes `t2 = (0,0)` then
        // `t2 - 1 = (-1, 0)` and returns `mulsign(-(-1) - 0, x) = +/-1`.
        return signbit_(x) ? -1.0 : 1.0;
    } else {
        // Outer branch (2.5 <= x <= 6.0):
        //   t  = poly21(x; tail-coeffs)
        //   t2 = poly4dd(x, t, [DD constants])  (= argument to expk)
        //   t2 = (expk(t2), 0)
        const double t = poly_array(xa, kErfU1Tail, 21);
        DD r2 = poly4dd_(xa, t, DD{-0.63691044383641748361, -2.4249477526539431839e-17},
                         DD{-1.1282926061803961737, -6.2970338860410996505e-17},
                         DD{-1.2261313785184804967e-05, -5.5329707514490107044e-22}, fe);
        // SLEEF: `t2 = dd(expk(t2), 0)`. Our `sf64_internal_expk_dd` accepts
        // a DD argument (matches SLEEF expk).
        const double e = soft_fp64::sleef::sf64_internal_expk_dd(r2, fe);
        t2 = DD{e, 0.0};
    }

    // SLEEF: `t2 = ddadd2_d2_d2_d(t2, -1)`; final = mulsign(-t2.x - t2.y, x)
    t2 = ddadd2_dd_dd_d(t2, -1.0);
    double r = sf64_neg(sf64_add(t2.hi, t2.lo));
    if (signbit_(x))
        r = sf64_neg(r);
    fe.flush();
    return r;
}

extern "C" double sf64_erfc(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x))
        return qNaN();
    if (isinf_(x))
        return signbit_(x) ? 2.0 : 0.0;

    const double s = x;
    const double a = sf64_fabs(x);

    // SLEEF branches: o0 = a<1, o1 = a<2.2, o2 = a<4.2, o3 = a<27.3.
    const bool o0 = lt_(a, 1.0);
    const bool o1 = lt_(a, 2.2);
    const bool o2 = lt_(a, 4.2);
    const bool o3 = lt_(a, 27.3);

    // u = (a^2, a, 1/a) DD depending on branch.
    DD u;
    if (o0) {
        u = ddmul_dd_d_d(a, a);
    } else if (o1) {
        u = DD{a, 0.0};
    } else {
        u = dddiv_dd_dd_dd(DD{1.0, 0.0}, DD{a, 0.0});
    }

    // Branch-selected 18-term scalar polynomial in u.hi (= u.x). Coefficient
    // tables transcribed verbatim from upstream (sleefdp.c lines 2652-2669).
    constexpr int kErfcN = 18;
    static constexpr double kErfcCoefO0[kErfcN] = {
        +0.6801072401395386139e-20, -0.2161766247570055669e-18, +0.4695919173301595670e-17,
        -0.9049140419888007122e-16, +0.1634018903557410728e-14, -0.2783485786333451745e-13,
        +0.4463221276786415752e-12, -0.6711366622850136563e-11, +0.9422759050232662223e-10,
        -0.1229055530100229098e-08, +0.1480719281585086512e-07, -0.1636584469123399803e-06,
        +0.1646211436588923575e-05, -0.1492565035840623511e-04, +0.1205533298178967851e-03,
        -0.8548327023450850081e-03, +0.5223977625442187932e-02, -0.2686617064513125222e-01,
    };
    static constexpr double kErfcCoefO1[kErfcN] = {
        +0.3438010341362585303e-12, -0.1237021188160598264e-10, +0.2117985839877627852e-09,
        -0.2290560929177369506e-08, +0.1748931621698149538e-07, -0.9956602606623249195e-07,
        +0.4330010240640327080e-06, -0.1435050600991763331e-05, +0.3460139479650695662e-05,
        -0.4988908180632898173e-05, -0.1308775976326352012e-05, +0.2825086540850310103e-04,
        -0.6393913713069986071e-04, -0.2566436514695078926e-04, +0.5895792375659440364e-03,
        -0.1695715579163588598e-02, +0.2089116434918055149e-03, +0.1912855949584917753e-01,
    };
    static constexpr double kErfcCoefO2[kErfcN] = {
        -0.5757819536420710449e+2, +0.4669289654498104483e+3, -0.1796329879461355858e+4,
        +0.4355892193699575728e+4, -0.7456258884965764992e+4, +0.9553977358167021521e+4,
        -0.9470019905444229153e+4, +0.7387344321849855078e+4, -0.4557713054166382790e+4,
        +0.2207866967354055305e+4, -0.8217975658621754746e+3, +0.2268659483507917400e+3,
        -0.4633361260318560682e+2, +0.9557380123733945965e+1, -0.2958429331939661289e+1,
        +0.1670329508092765480e+0, +0.6096615680115419211e+0, +0.1059212443193543585e-2,
    };
    static constexpr double kErfcCoefO3[kErfcN] = {
        +0.2334249729638701319e+5, -0.4695661044933107769e+5, +0.3173403108748643353e+5,
        +0.3242982786959573787e+4, -0.2014717999760347811e+5, +0.1554006970967118286e+5,
        -0.6150874190563554293e+4, +0.1240047765634815732e+4, -0.8210325475752699731e+2,
        +0.3242443880839930870e+2, -0.2923418863833160586e+2, +0.3457461732814383071e+0,
        +0.5489730155952392998e+1, +0.1559934132251294134e-2, -0.1541741566831520638e+1,
        +0.2823152230558364186e-5, +0.6249999184195342838e+0, +0.1741749416408701288e-8,
    };
    const double* coefs = o0 ? kErfcCoefO0 : (o1 ? kErfcCoefO1 : (o2 ? kErfcCoefO2 : kErfcCoefO3));
    const double t = poly_array(u.hi, coefs, kErfcN);

    // d = u*t + DD-constant; then d*u + DD-constant; then d*u + DD-constant.
    // Branch-selected DD constants from sleefdp.c lines 2672-2685.
    DD d = ddmul_dd_dd_d(u, t);
    d = ddadd2_dd_dd(d, o0   ? DD{0.11283791670955126141, -4.0175691625932118483e-18}
                        : o1 ? DD{-0.10277263343147646779, -6.2338714083404900225e-18}
                        : o2 ? DD{-0.50005180473999022439, 2.6362140569041995803e-17}
                             : DD{-0.5000000000258444377, -4.0074044712386992281e-17});
    d = ddmul_dd_dd_dd(d, u);
    d = ddadd2_dd_dd(d, o0   ? DD{-0.37612638903183753802, 1.3391897206042552387e-17}
                        : o1 ? DD{-0.63661976742916359662, 7.6321019159085724662e-18}
                        : o2 ? DD{1.601106273924963368e-06, 1.1974001857764476775e-23}
                             : DD{2.3761973137523364792e-13, -1.1670076950531026582e-29});
    d = ddmul_dd_dd_dd(d, u);
    d = ddadd2_dd_dd(d, o0   ? DD{1.1283791670955125586, 1.5335459613165822674e-17}
                        : o1 ? DD{-1.1283791674717296161, 8.0896847755965377194e-17}
                        : o2 ? DD{-0.57236496645145429341, 3.0704553245872027258e-17}
                             : DD{-0.57236494292470108114, -2.3984352208056898003e-17});

    // Reconstruction (sleefdp.c lines 2687-2690):
    //   x = (o1 ? d : dd(-a, 0)) * a
    //   x = o1 ? x : (x + d)
    //   x = o0 ? (1 - x) : expk2(x)
    //   x = o1 ? x : x * u
    DD xx = ddmul_dd_dd_d(o1 ? d : DD{sf64_neg(a), 0.0}, a);
    if (!o1)
        xx = ddadd2_dd_dd(xx, d);
    xx = o0 ? ddsub_dd_dd_dd(DD{1.0, 0.0}, xx) : sf64_internal_expk2_dd(xx, fe);
    if (!o1)
        xx = ddmul_dd_dd_dd(xx, u);

    double r = o3 ? sf64_add(xx.hi, xx.lo) : 0.0;
    if (signbit_(s))
        r = sf64_sub(2.0, r);
    fe.flush();
    return r;
}

// ========================================================================
// tgamma / lgamma / lgamma_r - port of SLEEF 3.6 xtgamma_u1 / xlgamma_u1
// ========================================================================
//
// Faithful port of upstream `gammak`, `xtgamma_u1`, and `xlgamma_u1`.
// The SLEEF approach unifies the central and tail regions through a
// shared `gammak(a)` that returns a pair of DD numbers `(clc, div)` with
//   ln Gamma(|a|) ~ clc + ln(div)
//   Gamma(|a|)   ~ exp(clc) * div
// Three regimes inside gammak:
//   * tiny: |a| < 1e-306 - uses log(2^120) lift to keep magnitude
//   * reflection: a < 0.5 - uses sinpik(a - 2^28 * floor(a/2^28))
//   * normal:    a >= 0.5 - central poly + (a > 2.3 -> "shift up by 5" path)
//
// The polynomial is degree-22 in t with branch-selected coefficients for
// (o2, o0, else). Coefficients are transcribed verbatim from sleefdp.c
// lines 2490-2520.

namespace {

struct GammakRet {
    DD clc;
    DD div;
};

GammakRet gammak(double a, soft_fp64::sleef::sf64_internal_fe_acc& fe) {
    DD clc{0.0, 0.0};
    DD clln{1.0, 0.0};
    DD clld{1.0, 0.0};
    DD x;

    const bool otiny = lt_(sf64_fabs(a), 1e-306);
    const bool oref = lt_(a, 0.5);

    if (otiny) {
        x = DD{0.0, 0.0};
    } else if (oref) {
        x = ddadd2_dd_d_d(1.0, sf64_neg(a));
    } else {
        x = DD{a, 0.0};
    }

    const bool o0 = ge_(x.hi, 0.5) && le_(x.hi, 1.1);
    const bool o2 = gt_(x.hi, 2.3);

    // For 2.3 < x.hi <= 7.0: ratchet x up by 5 in DD, accumulate the
    // numerator product y = (x+1)(x+2)(x+3)(x+4) so that the polynomial
    // can be evaluated in the [shift-up] regime.
    DD y = ddnormalize_dd_dd(ddmul_dd_dd_dd(ddadd2_dd_dd_d(x, 1.0), x));
    y = ddnormalize_dd_dd(ddmul_dd_dd_dd(ddadd2_dd_dd_d(x, 2.0), y));
    y = ddnormalize_dd_dd(ddmul_dd_dd_dd(ddadd2_dd_dd_d(x, 3.0), y));
    y = ddnormalize_dd_dd(ddmul_dd_dd_dd(ddadd2_dd_dd_d(x, 4.0), y));

    const bool shift_up = o2 && le_(x.hi, 7.0);
    if (shift_up) {
        clln = y;
        x = ddadd2_dd_dd_d(x, 5.0);
    }

    // t: argument of the polynomial.
    //   o2 ? 1/x.hi : ddnormalize(x + (o0 ? -1 : -2)).hi
    double t;
    if (o2) {
        t = sf64_div(1.0, x.hi);
    } else {
        t = ddnormalize_dd_dd(ddadd2_dd_dd_d(x, o0 ? -1.0 : -2.0)).hi;
    }

    // 22-step Horner with branch-selected coefficients (sleefdp.c lines
    // 2490-2512). Coefficients verbatim from upstream - do not retune.
    double u;
    u = o2 ? -156.801412704022726379848862
           : (o0 ? +0.2947916772827614196e+2 : +0.7074816000864609279e-7);
    u = sf64_fma(u, t,
                 o2 ? +1.120804464289911606838558160000
                    : (o0 ? +0.1281459691827820109e+3 : +0.4009244333008730443e-6));
    u = sf64_fma(u, t,
                 o2 ? +13.39798545514258921833306020000
                    : (o0 ? +0.2617544025784515043e+3 : +0.1040114641628246946e-5));
    u = sf64_fma(u, t,
                 o2 ? -0.116546276599463200848033357000
                    : (o0 ? +0.3287022855685790432e+3 : +0.1508349150733329167e-5));
    u = sf64_fma(u, t,
                 o2 ? -1.391801093265337481495562410000
                    : (o0 ? +0.2818145867730348186e+3 : +0.1288143074933901020e-5));
    u = sf64_fma(u, t,
                 o2 ? +0.015056113040026424412918973400
                    : (o0 ? +0.1728670414673559605e+3 : +0.4744167749884993937e-6));
    u = sf64_fma(u, t,
                 o2 ? +0.179540117061234856098844714000
                    : (o0 ? +0.7748735764030416817e+2 : -0.6554816306542489902e-7));
    u = sf64_fma(u, t,
                 o2 ? -0.002481743600264997730942489280
                    : (o0 ? +0.2512856643080930752e+2 : -0.3189252471452599844e-6));
    u = sf64_fma(u, t,
                 o2 ? -0.029527880945699120504851034100
                    : (o0 ? +0.5766792106140076868e+1 : +0.1358883821470355377e-6));
    u = sf64_fma(u, t,
                 o2 ? +0.000540164767892604515196325186
                    : (o0 ? +0.7270275473996180571e+0 : -0.4343931277157336040e-6));
    u = sf64_fma(u, t,
                 o2 ? +0.006403362833808069794787256200
                    : (o0 ? +0.8396709124579147809e-1 : +0.9724785897406779555e-6));
    u = sf64_fma(u, t,
                 o2 ? -0.000162516262783915816896611252
                    : (o0 ? -0.8211558669746804595e-1 : -0.2036886057225966011e-5));
    u = sf64_fma(u, t,
                 o2 ? -0.001914438498565477526465972390
                    : (o0 ? +0.6828831828341884458e-1 : +0.4373363141819725815e-5));
    u = sf64_fma(u, t,
                 o2 ? +7.20489541602001055898311517e-05
                    : (o0 ? -0.7712481339961671511e-1 : -0.9439951268304008677e-5));
    u = sf64_fma(u, t,
                 o2 ? +0.000839498720672087279971000786
                    : (o0 ? +0.8337492023017314957e-1 : +0.2050727030376389804e-4));
    u = sf64_fma(u, t,
                 o2 ? -5.17179090826059219329394422e-05
                    : (o0 ? -0.9094964931456242518e-1 : -0.4492620183431184018e-4));
    u = sf64_fma(u, t,
                 o2 ? -0.000592166437353693882857342347
                    : (o0 ? +0.1000996313575929358e+0 : +0.9945751236071875931e-4));
    u = sf64_fma(u, t,
                 o2 ? +6.97281375836585777403743539e-05
                    : (o0 ? -0.1113342861544207724e+0 : -0.2231547599034983196e-3));
    u = sf64_fma(u, t,
                 o2 ? +0.000784039221720066627493314301
                    : (o0 ? +0.1255096673213020875e+0 : +0.5096695247101967622e-3));
    u = sf64_fma(u, t,
                 o2 ? -0.000229472093621399176949318732
                    : (o0 ? -0.1440498967843054368e+0 : -0.1192753911667886971e-2));
    u = sf64_fma(u, t,
                 o2 ? -0.002681327160493827160473958490
                    : (o0 ? +0.1695571770041949811e+0 : +0.2890510330742210310e-2));
    u = sf64_fma(u, t,
                 o2 ? +0.003472222222222222222175164840
                    : (o0 ? -0.2073855510284092762e+0 : -0.7385551028674461858e-2));
    u = sf64_fma(u, t,
                 o2 ? +0.083333333333333333335592087900
                    : (o0 ? +0.2705808084277815939e+0 : +0.2058080842778455335e-1));

    // y_path = (x - 0.5)*log(x) - x + 0.5*log(2*pi) - Stirling form, used
    // when o2 (a > 2.3).
    DD y_path = ddmul_dd_dd_dd(ddadd2_dd_dd_d(x, -0.5), sf64_internal_logk2_dd(x, fe));
    y_path = ddadd2_dd_dd(y_path, ddneg_dd_dd(x));
    y_path = ddadd2_dd_dd(y_path, DD{0.91893853320467278056, -3.8782941580672414498e-17});

    // z_path = poly3-in-t with DD final terms, scalar tail. SLEEF code
    // (sleefdp.c lines 2518-2521).
    DD z = ddadd2_dd_dd_d(ddmul_dd_d_d(u, t),
                          o0 ? -0.4006856343865314862e+0 : -0.6735230105319810201e-1);
    z = ddadd2_dd_dd_d(ddmul_dd_dd_d(z, t),
                       o0 ? +0.8224670334241132030e+0 : +0.3224670334241132030e+0);
    z = ddadd2_dd_dd_d(ddmul_dd_dd_d(z, t),
                       o0 ? -0.5772156649015328655e+0 : +0.4227843350984671345e+0);
    z = ddmul_dd_dd_d(z, t);

    clc = o2 ? y_path : z;

    // clld = (u*t + 1) for ALL o2 (not just shift_up). Upstream sleefdp.c
    // line 2525: `clld = o2 ? ... : clld;`. The shift_up path writes the
    // factorial product into clln above; clld here is the polynomial
    // correction multiplier that pairs with the Stirling form in clc.
    if (o2) {
        clld = ddadd2_dd_dd_d(ddmul_dd_d_d(u, t), 1.0);
    }

    y = clln; // save for later

    if (otiny) {
        // log(2^120) lift to keep tiny argument out of the singularity.
        clc = DD{83.1776616671934334590333, 3.67103459631568507221878e-15};
    } else if (oref) {
        // log(pi) - clc.
        clc = ddadd2_dd_dd(DD{1.1447298858494001639, 1.026595116270782638e-17}, ddneg_dd_dd(clc));
    }
    // clln update.
    if (otiny)
        clln = DD{1.0, 0.0};
    else if (!oref)
        clln = clld;

    // Reflection: x = clld * sinpik(a - 2^28 * floor(a/2^28)) - argument
    // reduction through 2^28 keeps sinpik's polynomial argument bounded.
    if (oref) {
        constexpr double kTwo28 = 268435456.0; // 2^28
        const double a_over_2_28 = sf64_mul(a, 1.0 / kTwo28);
        // SLEEF: `(int32_t)(a * 1/2^28)` - truncation toward zero.
        const int32_t k = sf64_to_i32(sf64_trunc(a_over_2_28));
        const double red = sf64_sub(a, sf64_mul(sf64_from_i32(k), kTwo28));
        x = ddmul_dd_dd_dd(clld, sf64_internal_sinpik_dd(red, fe));
    }

    // clld update: tiny -> a * 2^120; oref -> x; else -> y (= original clln).
    if (otiny) {
        // 2^120 = (2^60)^2 - both factors are exactly representable doubles.
        constexpr double k2_60 = 1152921504606846976.0;
        clld = DD{sf64_mul(a, sf64_mul(k2_60, k2_60)), 0.0};
    } else if (oref) {
        clld = x;
    } else {
        clld = y;
    }

    return GammakRet{clc, dddiv_dd_dd_dd(clln, clld)};
}

} // namespace

extern "C" double sf64_tgamma(double x) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;
    if (isnan_(x))
        return qNaN();

    GammakRet d = gammak(x, fe);
    DD ye = ddmul_dd_dd_dd(sf64_internal_expk2_dd(d.clc, fe), d.div);
    double r = sf64_add(ye.hi, ye.lo);

    // SLEEF post-conditions (sleefdp.c lines 2546-2548):
    //   r = (x == -inf || (x < 0 && isint(x)) || (xisnumber(x) && x < 0 && isnan(r))) ? NaN : r;
    //   r = ((x == +inf || xisnumber(x)) && x >= -DBL_MIN && (x == 0 || x > 200 || isnan(r))) ?
    //         mulsign(+inf, x) : r;
    const bool xnum = !isnan_(x) && !isinf_(x);
    if ((isinf_(x) && signbit_(x)) || (lt_(x, 0.0) && is_int(x)) ||
        (xnum && lt_(x, 0.0) && isnan_(r))) {
        fe.flush();
        return qNaN();
    }
    constexpr double kDblMin = 2.2250738585072014e-308;
    if ((eq_(x, kInf) || xnum) && ge_(x, sf64_neg(kDblMin)) &&
        (eq_(x, 0.0) || gt_(x, 200.0) || isnan_(r))) {
        fe.flush();
        return signbit_(x) ? sf64_neg(kInf) : kInf;
    }
    fe.flush();
    return r;
}

extern "C" double sf64_lgamma_r(double x, int* sign) {
    soft_fp64::sleef::sf64_internal_fe_acc fe;

    if (sign)
        *sign = 1;

    if (isnan_(x))
        return qNaN();
    if (isinf_(x)) {
        // SLEEF returns +inf for both +/-inf - sign=+1 by libm convention.
        return kInf;
    }
    if (lt_(x, 0.0) && is_int(x)) {
        // Pole: lgamma -> +inf.
        return kInf;
    }
    if (eq_(x, 0.0)) {
        // lgamma(+/-0) = +inf; sign reflects +/-0.
        if (sign)
            *sign = signbit_(x) ? -1 : 1;
        return kInf;
    }

    GammakRet d = gammak(x, fe);
    // SLEEF: y = d.a + log|d.b|;  r = y.x + y.y.
    DD ye = ddadd2_dd_dd(d.clc, sf64_internal_logk2_dd(ddabs_dd_dd(d.div), fe));
    double r = sf64_add(ye.hi, ye.lo);
    if (isinf_(x) || isnan_(r))
        r = kInf;

    // Sign: positive when Gamma(x) > 0. For x > 0, sign = +1. For x < 0
    // (non-integer): sign = sign of d.div.hi (which carries the sign of
    // Gamma(x) up to the strictly-positive expk(clc) magnitude lift).
    int local_sign = signbit_(d.div.hi) ? -1 : 1;
    if (sign)
        *sign = local_sign;

    fe.flush();
    return r;
}

extern "C" double sf64_lgamma(double x) {
    int s = 1;
    return sf64_lgamma_r(x, &s);
}
