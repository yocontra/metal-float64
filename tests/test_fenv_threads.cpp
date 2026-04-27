// Multi-threaded stress test for the TLS fenv accumulator.
//
// The 1.1 fenv accumulator stores per-thread flag bits in thread_local
// storage. The single-thread isolation check in test_fenv.cpp confirms
// that two sequentially-run threads do not see each other's bits, but
// it does not exercise *concurrent* execution where a host TLS bug
// (e.g. accidental file-scope `static` accumulator) would manifest as
// inter-thread bit leakage.
//
// This test puts two threads into a tight loop simultaneously, each
// raising a single, non-overlapping flag bit:
//   - Thread A: sf64_div(0, 0)         → INVALID only
//   - Thread B: sf64_div(1.0, 0.0)     → DIVBYZERO only
//
// Both ops are documented in test_fenv.cpp as raising exactly that one
// bit and nothing else, so any bit observed beyond the expected one is
// a TLS contention bug. The two threads synchronise through a
// std::atomic<int> rendezvous (no sleep()), then run kIters tight-loop
// iterations of clear → op → assert-flags-exact.
//
// Iteration count (10000 ops/thread) is large enough to expose a race
// on commodity hardware while keeping wall-clock under a second. If the
// suite ever grows tighter, this is the first knob to revisit.
//
// Under SOFT_FP64_FENV=disabled the accumulator is hard-wired to 0 and
// every raise/clear is a no-op — there is no per-thread state to leak
// or be raced on, so the test reduces to a no-op stub at compile time.
//
// SPDX-License-Identifier: MIT

#include "soft_fp64/soft_f64.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

#ifndef SF64_TEST_FENV_MODE
#define SF64_TEST_FENV_MODE 1
#endif

#if SF64_TEST_FENV_MODE == 1

namespace {

constexpr int kIters = 10000;

// Per-thread mismatch counters. We could `std::abort()` from the worker,
// but it's friendlier to surface "thread A saw 13 mismatches, first
// at iter=42 with flags=0x03" so the failure mode is legible.
struct ThreadResult {
    std::atomic<int> mismatch_count{0};
    std::atomic<int> first_bad_iter{-1};
    std::atomic<unsigned> first_bad_flags{0u};
};

// Bit-OR of all five fenv flags — used to assert no flag *at all* is
// set after the per-iteration clear, and that no extraneous bit beyond
// the expected one is present after the op.
constexpr unsigned kAllFlags =
    SF64_FE_INVALID | SF64_FE_DIVBYZERO | SF64_FE_OVERFLOW | SF64_FE_UNDERFLOW | SF64_FE_INEXACT;

// C++17 forbids `double` as a non-type template parameter, so the
// per-thread inputs are passed as plain function arguments. The
// expected-flag literal is templated only to keep it visible inside
// the assertion (and so the loop body shows up at -O2 with the
// expected bits inlined). There is no host fp64 arithmetic in the
// harness — A and B are forwarded straight into sf64_div.
template <unsigned ExpectedFlag>
void worker_loop(double a, double b, std::atomic<int>* rendezvous, ThreadResult* out) {
    // Bump the rendezvous and spin until both threads have arrived.
    // No sleep — busy-wait on the atomic so both threads enter the hot
    // loop within nanoseconds of each other.
    rendezvous->fetch_add(1, std::memory_order_acq_rel);
    while (rendezvous->load(std::memory_order_acquire) < 2) {
        // spin
    }

    int mismatches = 0;
    int first_bad = -1;
    unsigned first_bad_flags = 0u;

    for (int i = 0; i < kIters; ++i) {
        sf64_fe_clear(kAllFlags);
        // Defensive: confirm the clear worked on this thread's
        // accumulator before we observe what the op raised. If this
        // ever sees non-zero bits, another thread has scribbled into
        // *our* TLS slot.
        const unsigned post_clear = sf64_fe_getall();
        if (post_clear != 0u) {
            ++mismatches;
            if (first_bad < 0) {
                first_bad = i;
                first_bad_flags = post_clear;
            }
            continue;
        }

        (void)sf64_div(a, b);

        const unsigned got = sf64_fe_getall();
        if (got != ExpectedFlag) {
            ++mismatches;
            if (first_bad < 0) {
                first_bad = i;
                first_bad_flags = got;
            }
        }
    }

    out->mismatch_count.store(mismatches, std::memory_order_release);
    out->first_bad_iter.store(first_bad, std::memory_order_release);
    out->first_bad_flags.store(first_bad_flags, std::memory_order_release);
}

void run_test() {
    std::atomic<int> rendezvous{0};
    ThreadResult result_a;
    ThreadResult result_b;

    // Thread A: 0/0 → INVALID only (cf. test_fenv.cpp "div 0/0 → INVALID").
    std::thread thread_a(worker_loop<SF64_FE_INVALID>, 0.0, 0.0, &rendezvous, &result_a);
    // Thread B: 1/0 → DIVBYZERO only (cf. test_fenv.cpp "1/0 → DIVBYZERO").
    std::thread thread_b(worker_loop<SF64_FE_DIVBYZERO>, 1.0, 0.0, &rendezvous, &result_b);

    thread_a.join();
    thread_b.join();

    bool failed = false;
    const int a_count = result_a.mismatch_count.load();
    const int b_count = result_b.mismatch_count.load();

    if (a_count != 0) {
        std::fprintf(stderr,
                     "FAIL: thread A (0/0 → INVALID) saw %d mismatches; "
                     "first at iter=%d flags=0x%02x (expected 0x%02x)\n",
                     a_count, result_a.first_bad_iter.load(), result_a.first_bad_flags.load(),
                     SF64_FE_INVALID);
        failed = true;
    }
    if (b_count != 0) {
        std::fprintf(stderr,
                     "FAIL: thread B (1/0 → DIVBYZERO) saw %d mismatches; "
                     "first at iter=%d flags=0x%02x (expected 0x%02x)\n",
                     b_count, result_b.first_bad_iter.load(), result_b.first_bad_flags.load(),
                     SF64_FE_DIVBYZERO);
        failed = true;
    }

    if (failed) {
        std::fflush(stderr);
        std::abort();
    }

    std::fprintf(stdout,
                 "  thread_a (INVALID): %d iters, 0 mismatches\n"
                 "  thread_b (DIVBYZERO): %d iters, 0 mismatches\n",
                 kIters, kIters);
}

} // namespace

int main() {
    std::fputs("test_fenv_threads:\n", stdout);
    run_test();
    std::fputs("test_fenv_threads: per-thread fenv isolation under concurrency: ok\n", stdout);
    return 0;
}

#else // SF64_TEST_FENV_MODE != 1

int main() {
    // Under `disabled` mode there is no accumulator to race on — every
    // raise/clear is a no-op and getall() is hard-wired to 0. The test
    // is genuinely-not-applicable, so skip cleanly. This is NOT a
    // tolerance-hack skip: there is no flag state for the test to
    // observe, so any assertion about it would be vacuous.
    std::fputs("test_fenv_threads: SOFT_FP64_FENV=disabled — skipped (no accumulator)\n", stdout);
    return 0;
}

#endif
