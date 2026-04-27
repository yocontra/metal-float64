#pragma once

// Internal IEEE-754 exception-flag plumbing.
//
// Three build modes, selected by SOFT_FP64_FENV_MODE:
//   0 (disabled) : SF64_FE_RAISE expands to (void)0 — zero-cost; every
//                  sf64_fe_* and sf64_*_ex flag-related entry compiles to a
//                  no-op. The TLS storage symbol is omitted.
//   1 (tls)      : per-thread accumulator; the default `sf64_*` entry points
//                  flush into a thread_local sticky bag at end-of-op.
//                  Public TLS surface (`sf64_fe_getall` / clear / raise /
//                  test / save / restore) reads/writes that bag.
//                  The parallel `sf64_*_ex` ABI is also emitted: when the
//                  caller passes a non-null sf64_fe_state_t*, the
//                  accumulator flushes into the caller's struct instead of
//                  TLS — null pointer falls back to the TLS bag so a single
//                  `_ex` call can be used as a "skip-flags" override even
//                  in TLS mode.
//   2 (explicit) : caller-state ABI is the only one. Thread-local storage
//                  is omitted entirely (so consumers like Metal / WebGPU
//                  GPU kernels, where `thread_local` is not available, can
//                  link). `sf64_*_ex` accept `sf64_fe_state_t*`; null
//                  pointer means "drop the flags". The TLS surface
//                  (`sf64_fe_getall` etc.) compiles to no-op stubs so
//                  adapters that reference both surfaces still link, but
//                  observably it always reports 0.
//
// The `SOFT_FP64_FENV_MODE` macro is injected by CMakeLists.txt. The
// public `sf64_fe_*` / `sf64_fe_*_ex` surface in soft_f64.h is defined in
// src/fenv.cpp and reads/writes the same storage paths described above.
//
// Two raise paths exist for hot-path cost reasons:
//
//   * SF64_FE_RAISE(bits) — direct TLS store. Use in cold paths
//     (classify, rem, rounding ops). Each expansion in tls mode compiles
//     to one thread_local access (`bl __tlv_get_addr` on Apple Silicon,
//     ~7ns), so repeated use in a single public op is expensive. Under
//     mode 0 / 2 it compiles to nothing.
//
//   * sf64_internal_fe_acc — accumulator threaded by reference through
//     every helper that might raise. The public entry constructs one and
//     calls `flush()` once at return. The accumulator carries an optional
//     `sf64_fe_state_t*` so the same body services both ABIs:
//       - default-constructed: flush to TLS (mode 1) or no-op (mode 0/2);
//       - constructed from a state pointer: flush directly into the
//         caller's struct (modes 1 and 2). A null pointer is honoured as
//         "drop flags" in mode 2; in mode 1 it falls back to TLS so a
//         consumer that wants the existing TLS bag for a one-off can pass
//         null without bypassing the accumulator.
//     IEEE 754 §7 flags are sticky accumulators; the spec is silent on
//     mid-op observability so deferring the store to end-of-op is
//     conformant.
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
// Thread-local flag accumulator. Hidden visibility keeps the TLS wrapper
// symbol off the shipped ABI surface (cf. CLAUDE.md §Hard constraints:
// cross-TU internals carry `sf64_internal_` + the explicit hidden-visibility
// attribute, matching src/sleef/sleef_internal.h).
//
// Defined as `inline thread_local` in this header rather than `extern` +
// out-of-line so cross-TU access sites see the constant initializer
// directly. Without that, clang on Linux ELF emits R_X86_64_PC32
// references to the `_ZTH...` TLS init wrapper at every access site, and
// the wrapper is never defined in fenv.cpp.o (the constant init `= 0u`
// suppresses wrapper emission). Result was a PIE link failure on Ubuntu
// clang. The inline-thread_local form avoids the extern/wrapper dance
// entirely; comdat dedupes the variable to a single per-thread slot in
// the final image.
[[gnu::visibility("hidden"),
  gnu::tls_model("initial-exec")]] inline thread_local unsigned sf64_internal_fe_flags = 0u;

[[gnu::visibility("hidden")]] SF64_ALWAYS_INLINE void
sf64_internal_fe_raise_bits(unsigned bits) noexcept {
    sf64_internal_fe_flags |= bits;
}
#define SF64_FE_RAISE(bits) (::soft_fp64::internal::sf64_internal_fe_raise_bits((bits)))

// Accumulator. Two construction shapes:
//   * default: flush to the thread_local sticky bag at end-of-op. Used by
//     the existing `sf64_*` and `sf64_*_r` entries — preserves pre-1.2
//     default-ABI behaviour bit-exactly.
//   * tls_redirect_t tag (i.e. with a `sf64_fe_state_t*`): flush directly
//     into the caller's struct, never touch TLS. Used by the `sf64_*_ex`
//     parallel ABI. A null pointer is interpreted as "drop flags" — the
//     GPU/SIMT consumers this surface exists for typically don't care
//     about flag observability and want zero overhead, while a TLS-mode
//     caller that wants the default bag can simply call the unsuffixed
//     `sf64_*` entry.
class sf64_internal_fe_acc {
  public:
    SF64_ALWAYS_INLINE sf64_internal_fe_acc() noexcept = default;
    SF64_ALWAYS_INLINE explicit sf64_internal_fe_acc(sf64_fe_state_t* state) noexcept
        : state_(state), explicit_state_(true) {}
    SF64_ALWAYS_INLINE void raise(unsigned bits) noexcept { bits_ |= bits; }
    SF64_ALWAYS_INLINE void flush() noexcept {
        if (bits_ == 0u) {
            return;
        }
        if (explicit_state_) {
            if (state_ != nullptr) {
                state_->flags |= bits_;
            }
        } else {
            sf64_internal_fe_flags |= bits_;
        }
    }

  private:
    sf64_fe_state_t* state_ = nullptr;
    unsigned bits_ = 0;
    bool explicit_state_ = false;
};
#elif SOFT_FP64_FENV_MODE == 2
// Explicit-state mode: no thread-local storage exists. The accumulator
// flushes only when constructed from a non-null `sf64_fe_state_t*`.
// Default construction is still allowed (the default `sf64_*` /
// `sf64_*_r` entries continue to declare a stack-local accumulator) but
// the resulting state is dropped at end-of-op — there is nowhere to
// store it without TLS. Consumers that need flag observability must use
// the `sf64_*_ex` surface.
//
// SF64_FE_RAISE expands to nothing in this mode for the same reason —
// the cold-path raise sites have no accumulator in scope and would have
// nowhere to go.
#define SF64_FE_RAISE(bits) ((void)0)

class sf64_internal_fe_acc {
  public:
    SF64_ALWAYS_INLINE sf64_internal_fe_acc() noexcept = default;
    SF64_ALWAYS_INLINE explicit sf64_internal_fe_acc(sf64_fe_state_t* state) noexcept
        : state_(state) {}
    SF64_ALWAYS_INLINE void raise(unsigned bits) noexcept {
        if (state_ != nullptr) {
            bits_ |= bits;
        }
    }
    SF64_ALWAYS_INLINE void flush() noexcept {
        if (state_ != nullptr && bits_ != 0u) {
            state_->flags |= bits_;
        }
    }

  private:
    sf64_fe_state_t* state_ = nullptr;
    unsigned bits_ = 0;
};
#else
[[gnu::visibility("hidden")]] SF64_ALWAYS_INLINE void
sf64_internal_fe_raise_bits(unsigned /*bits*/) noexcept {
}
#define SF64_FE_RAISE(bits) ((void)0)

// Empty accumulator in disabled mode: methods are no-ops, the class
// has no state, and passing it by reference through the helper tree
// DCEs completely under -O2+. The state-pointer constructor exists so
// the `_ex` surface can compile in this mode (it just drops the flag
// regardless of the pointer); CMake leaves the `_ex` symbols out of
// the archive entirely under `disabled` so this branch is normally
// unreachable, but the constructor stays available for completeness.
class sf64_internal_fe_acc {
  public:
    SF64_ALWAYS_INLINE sf64_internal_fe_acc() noexcept = default;
    SF64_ALWAYS_INLINE explicit sf64_internal_fe_acc(sf64_fe_state_t* /*state*/) noexcept {}
    SF64_ALWAYS_INLINE void raise(unsigned /*bits*/) noexcept {}
    SF64_ALWAYS_INLINE void flush() noexcept {}
};
#endif

} // namespace soft_fp64::internal
