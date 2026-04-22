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
// Two raise paths exist for hot-path cost reasons:
//
//   * SF64_FE_RAISE(bits) — direct TLS store. Use in cold paths
//     (classify, rem, rounding ops). Each expansion in tls mode compiles
//     to one thread_local access (`bl __tlv_get_addr` on Apple Silicon,
//     ~7ns), so repeated use in a single public op is expensive.
//
//   * sf64_internal_fe_acc — stack-local accumulator with .raise(bits)
//     and .flush(). Public entries (sf64_add, sf64_to_i32, etc.)
//     declare one, thread it by reference through every helper that
//     might raise, and flush once at return — one TLS store per
//     public call instead of one per raise site. Use for the hot
//     arithmetic / convert / sqrt / fma paths where an op may hit
//     several SF64_FE_RAISE points (rounding inexact + underflow +
//     overflow checks). IEEE 754 §7 flags are sticky accumulators;
//     the spec is silent on mid-op observability, so deferring the
//     store to end-of-op is conformant.
//
// In disabled mode, sf64_internal_fe_acc is an empty class whose
// methods are no-ops — the parameter threading DCEs completely, so
// there is no disabled-mode overhead vs the old free-standing
// helper shape.

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
[[gnu::visibility("hidden"),
  gnu::tls_model("initial-exec")]] extern thread_local unsigned sf64_internal_fe_flags;

[[gnu::visibility("hidden")]] SF64_ALWAYS_INLINE void
sf64_internal_fe_raise_bits(unsigned bits) noexcept {
    sf64_internal_fe_flags |= bits;
}
#define SF64_FE_RAISE(bits) (::soft_fp64::internal::sf64_internal_fe_raise_bits((bits)))

// Stack-local flag accumulator. One instance per public op; threaded by
// reference through every helper that might raise; flushed to the
// thread-local store once at return. Keeps `bl __tlv_get_addr` off the
// inner loop.
class sf64_internal_fe_acc {
  public:
    SF64_ALWAYS_INLINE void raise(unsigned bits) noexcept { bits_ |= bits; }
    SF64_ALWAYS_INLINE void flush() noexcept {
        if (bits_ != 0u) {
            sf64_internal_fe_flags |= bits_;
        }
    }

  private:
    unsigned bits_ = 0;
};
#else
[[gnu::visibility("hidden")]] SF64_ALWAYS_INLINE void
sf64_internal_fe_raise_bits(unsigned /*bits*/) noexcept {
}
#define SF64_FE_RAISE(bits) ((void)0)

// Empty accumulator in disabled mode: methods are no-ops, the class
// has no state, and passing it by reference through the helper tree
// DCEs completely under -O2+.
class sf64_internal_fe_acc {
  public:
    SF64_ALWAYS_INLINE void raise(unsigned /*bits*/) noexcept {}
    SF64_ALWAYS_INLINE void flush() noexcept {}
};
#endif

} // namespace soft_fp64::internal
