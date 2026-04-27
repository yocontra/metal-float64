// Soft-fp64 compare: fcmp (all 16 LLVM predicates), fmin_precise, fmax_precise.
//
// Reference: Mesa `src/compiler/glsl/float64.glsl` — __feq64, __fne64,
// __flt64, __fge64, __fle64, __fgt64. Mesa only ships ordered predicates;
// unordered variants are `isnan(a) || isnan(b) || ordered_pred(a,b)`.
//
// `pred` is the LLVM FCmpInst::Predicate enum (0..15). Design goal: when
// AdaptiveCpp's MSL emitter consumes this library (SF64_ABI redefined to
// HIPSYCL_SSCP_BUILTIN), `pred` is a compile-time constant at every
// callsite and the switch const-folds away. We therefore structure the
// implementation as an `always_inline` static helper `fcmp_impl` — the
// extern-"C" entry point is a thin strong-symbol wrapper around it, so
// the stand-alone host build (which links these entry points cross-TU,
// with no LTO) still gets a symbol it can call.
//
// fmin_precise / fmax_precise follow IEEE 754-2008 minimum/maximum: if
// either input is NaN, the result is a canonical quiet NaN (in contrast
// with fmin/fmax in classify.cpp which flush NaN).
//
// Implementation note: this TU deliberately never performs host-`double`
// arithmetic. Every decision goes through integer bit ops on `uint64_t`
// (ordering is resolved via a two's-complement transform of the IEEE
// sign-magnitude bit pattern) because the library will eventually run on
// a device with no fp64 FPU, where host `<` on `double` would either be
// emulated (calls back into us — infinite recursion risk) or generate an
// unsupported instruction.
//
// SPDX-License-Identifier: MIT

#include "internal.h"
#include "soft_fp64/soft_f64.h"

namespace {

using soft_fp64::internal::bits_of;
using soft_fp64::internal::from_bits;
using soft_fp64::internal::is_nan_bits;
using soft_fp64::internal::is_zero_bits;
using soft_fp64::internal::kSignMask;
using soft_fp64::internal::propagate_nan;

// Ordered less-than on raw IEEE-754 binary64 bit patterns. Precondition:
// neither `a` nor `b` is NaN — callers must gate on that separately.
//
// Trick: converting sign-magnitude to two's-complement makes a signed
// integer compare reproduce the IEEE numerical order. Flip all non-sign
// bits when the sign bit is set.
SF64_ALWAYS_INLINE bool ordered_lt(uint64_t a, uint64_t b) noexcept {
    // Handle signed zeros explicitly: IEEE says -0 == +0, so neither is <
    // the other. Without this, the two's-complement transform would report
    // -0 < +0 which would break FCMP_OLT on that pair.
    if (is_zero_bits(a) && is_zero_bits(b)) {
        return false;
    }
    // SAFETY: reinterpret uint64_t as int64_t — both are 64-bit,
    // implementation-defined but well-specified on two's-complement
    // platforms (clang guarantees). The shifted value is masked with
    // 0x7FFF...FFFF so the sign bit itself is preserved.
    const int64_t sa = static_cast<int64_t>(a) ^ ((static_cast<int64_t>(a) >> 63) &
                                                  static_cast<int64_t>(0x7FFFFFFFFFFFFFFFLL));
    const int64_t sb = static_cast<int64_t>(b) ^ ((static_cast<int64_t>(b) >> 63) &
                                                  static_cast<int64_t>(0x7FFFFFFFFFFFFFFFLL));
    return sa < sb;
}

// Ordered equality on raw IEEE-754 binary64 bit patterns. Precondition:
// neither `a` nor `b` is NaN.
//
// Two cases qualify as equal:
//   1. Identical bit patterns.
//   2. Both are zero (of either sign): +0 == -0 per IEEE.
SF64_ALWAYS_INLINE bool ordered_eq(uint64_t a, uint64_t b) noexcept {
    if (a == b)
        return true;
    return is_zero_bits(a) && is_zero_bits(b);
}

// Core predicate switch. `always_inline` so that when the caller passes a
// constant `pred`, clang collapses this to a single branch.
SF64_ALWAYS_INLINE int fcmp_impl(double a_, double b_, int pred) noexcept {
    // SAFETY: bit_cast double -> uint64_t — both 8 bytes, no trap
    // representations; defined by __builtin_bit_cast semantics.
    const uint64_t a = bits_of(a_);
    const uint64_t b = bits_of(b_);
    const bool nan = is_nan_bits(a) || is_nan_bits(b);
    switch (pred) {
    case 0:
        return 0; // FCMP_FALSE
    case 1:
        return !nan && ordered_eq(a, b) ? 1 : 0; // FCMP_OEQ
    case 2:
        return !nan && ordered_lt(b, a) ? 1 : 0; // FCMP_OGT
    case 3:
        return !nan && !ordered_lt(a, b) ? 1 : 0; // FCMP_OGE
    case 4:
        return !nan && ordered_lt(a, b) ? 1 : 0; // FCMP_OLT
    case 5:
        return !nan && !ordered_lt(b, a) ? 1 : 0; // FCMP_OLE
    case 6:
        return !nan && !ordered_eq(a, b) ? 1 : 0; // FCMP_ONE
    case 7:
        return !nan ? 1 : 0; // FCMP_ORD
    case 8:
        return nan ? 1 : 0; // FCMP_UNO
    case 9:
        return (nan || ordered_eq(a, b)) ? 1 : 0; // FCMP_UEQ
    case 10:
        return (nan || ordered_lt(b, a)) ? 1 : 0; // FCMP_UGT
    case 11:
        return (nan || !ordered_lt(a, b)) ? 1 : 0; // FCMP_UGE
    case 12:
        return (nan || ordered_lt(a, b)) ? 1 : 0; // FCMP_ULT
    case 13:
        return (nan || !ordered_lt(b, a)) ? 1 : 0; // FCMP_ULE
    case 14:
        return (nan || !ordered_eq(a, b)) ? 1 : 0; // FCMP_UNE
    case 15:
        return 1; // FCMP_TRUE
    default:
        return 0;
    }
}

SF64_ALWAYS_INLINE double fmin_precise_impl(double a_, double b_) noexcept {
    // SAFETY: bit_cast double -> uint64_t — see fcmp_impl rationale.
    const uint64_t a = bits_of(a_);
    const uint64_t b = bits_of(b_);
    if (is_nan_bits(a) || is_nan_bits(b)) {
        return propagate_nan(a, b);
    }
    // Signed-zero tie-break: when both are zero (possibly opposite signs),
    // prefer -0. If either operand has the sign bit set, result is -0.
    if (is_zero_bits(a) && is_zero_bits(b)) {
        const bool any_negative = ((a | b) & kSignMask) != 0;
        return from_bits(any_negative ? kSignMask : 0ULL);
    }
    return ordered_lt(a, b) ? a_ : b_;
}

SF64_ALWAYS_INLINE double fmax_precise_impl(double a_, double b_) noexcept {
    // SAFETY: bit_cast double -> uint64_t — see fcmp_impl rationale.
    const uint64_t a = bits_of(a_);
    const uint64_t b = bits_of(b_);
    if (is_nan_bits(a) || is_nan_bits(b)) {
        return propagate_nan(a, b);
    }
    // Signed-zero tie-break: prefer +0. If either operand lacks the sign
    // bit, result is +0.
    if (is_zero_bits(a) && is_zero_bits(b)) {
        const bool any_positive = ((a & kSignMask) == 0) || ((b & kSignMask) == 0);
        return from_bits(any_positive ? 0ULL : kSignMask);
    }
    return ordered_lt(a, b) ? b_ : a_;
}

} // namespace

// ---- ABI entry points ---------------------------------------------------
//
// Strong external `extern "C"` definitions. Each delegates to the
// `always_inline` helper above, which means the switch in fcmp_impl is
// fully materialized inside this wrapper (out-of-line, standard codegen).
// When AdaptiveCpp redefines SF64_ABI to HIPSYCL_SSCP_BUILTIN at library
// consumption time, the emitter inlines these wrappers at each callsite
// and the switch const-folds on the constant `pred`.

// SF64_NO_OPT: keeps clang -O3 from pattern-matching the inlined fcmp_impl
// body back into `fcmp <pred> double` instructions. AdaptiveCpp's MSL
// emitter would route those through __acpp_sscp_fcmp_f64 → sf64_fcmp,
// producing infinite mutual recursion on AGX. See defines.h.
SF64_NO_OPT extern "C" int sf64_fcmp(double a, double b, int pred) {
    return fcmp_impl(a, b, pred);
}

extern "C" double sf64_fmin_precise(double a, double b) {
    return fmin_precise_impl(a, b);
}

extern "C" double sf64_fmax_precise(double a, double b) {
    return fmax_precise_impl(a, b);
}
