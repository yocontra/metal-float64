// Public sf64_fe_* surface. Storage lives here; call sites in arithmetic /
// sqrt_fma / convert / sleef raise into it through the SF64_FE_RAISE macro
// in src/internal_fenv.h.
//
// SOFT_FP64_FENV_MODE:
//   0 — disabled: every public entry is a no-op (or returns 0).
//   1 — tls: per-thread accumulator, thread_local storage duration.
//   2 — explicit: reserved for a future caller-state ABI; today same as 0.
//
// SPDX-License-Identifier: MIT

#include "internal_fenv.h"
#include "soft_fp64/soft_f64.h"

namespace soft_fp64::internal {
#if SOFT_FP64_FENV_MODE == 1
thread_local unsigned sf64_internal_fe_flags = 0u;
#endif
} // namespace soft_fp64::internal

extern "C" unsigned sf64_fe_getall(void) {
#if SOFT_FP64_FENV_MODE == 1
    return soft_fp64::internal::sf64_internal_fe_flags;
#else
    return 0u;
#endif
}

extern "C" int sf64_fe_test(unsigned mask) {
#if SOFT_FP64_FENV_MODE == 1
    return (soft_fp64::internal::sf64_internal_fe_flags & mask) != 0 ? 1 : 0;
#else
    (void)mask;
    return 0;
#endif
}

extern "C" void sf64_fe_raise(unsigned mask) {
#if SOFT_FP64_FENV_MODE == 1
    soft_fp64::internal::sf64_internal_fe_flags |= mask;
#else
    (void)mask;
#endif
}

extern "C" void sf64_fe_clear(unsigned mask) {
#if SOFT_FP64_FENV_MODE == 1
    soft_fp64::internal::sf64_internal_fe_flags &= ~mask;
#else
    (void)mask;
#endif
}

extern "C" void sf64_fe_save(sf64_fe_state_t* out) {
    if (out == nullptr)
        return;
#if SOFT_FP64_FENV_MODE == 1
    out->flags = soft_fp64::internal::sf64_internal_fe_flags;
#else
    out->flags = 0u;
#endif
}

extern "C" void sf64_fe_restore(const sf64_fe_state_t* in) {
    if (in == nullptr)
        return;
#if SOFT_FP64_FENV_MODE == 1
    soft_fp64::internal::sf64_internal_fe_flags = in->flags;
#else
    (void)in;
#endif
}
