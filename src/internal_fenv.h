#pragma once

// Internal IEEE-754 exception-flag plumbing.
//
// Three build modes, selected by SOFT_FP64_FENV_MODE:
//   0 (disabled) : SF64_FE_RAISE expands to (void)0 — zero-cost
//   1 (tls)      : per-thread accumulator; SF64_FE_RAISE |='s into it
//   2 (explicit) : reserved for 1.2+ caller-state ABI — currently same as disabled
//
// The `SOFT_FP64_FENV_MODE` macro is injected by CMakeLists.txt. The public
// `sf64_fe_{getall,test,raise,clear,save,restore}` surface in soft_f64.h is
// defined in src/fenv.cpp and reads/writes the same storage.
//
// SPDX-License-Identifier: MIT

#include "soft_fp64/defines.h"
#include "soft_fp64/soft_f64.h"

#include <cstdint>

#ifndef SOFT_FP64_FENV_MODE
#define SOFT_FP64_FENV_MODE 1
#endif

namespace soft_fp64::internal {

#if SOFT_FP64_FENV_MODE == 1
// Thread-local flag accumulator. Defined in src/fenv.cpp. Hidden visibility
// keeps the TLS wrapper symbol off the shipped ABI surface (cf. CLAUDE.md
// §Hard constraints: cross-TU internals carry `sf64_internal_` + the
// explicit hidden-visibility attribute, matching src/sleef/sleef_internal.h).
[[gnu::visibility("hidden")]] extern thread_local unsigned sf64_internal_fe_flags;

[[gnu::visibility("hidden")]] SF64_ALWAYS_INLINE void
sf64_internal_fe_raise_bits(unsigned bits) noexcept {
    sf64_internal_fe_flags |= bits;
}
#define SF64_FE_RAISE(bits) (::soft_fp64::internal::sf64_internal_fe_raise_bits((bits)))
#else
[[gnu::visibility("hidden")]] SF64_ALWAYS_INLINE void
sf64_internal_fe_raise_bits(unsigned /*bits*/) noexcept {
}
#define SF64_FE_RAISE(bits) ((void)0)
#endif

} // namespace soft_fp64::internal
