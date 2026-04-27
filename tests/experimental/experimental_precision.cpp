// Report-only harness for sf64_* input regions that are NOT part of the
// shipped correctness claim.
//
// This file is the holding pattern for precision regimes that the main
// test tree does not gate on. Each sweep here prints observed ULP against
// libm and returns success unconditionally. A real precision fix will:
//   1. land the tighter algorithm (DD stitching, Temme asymptotics, etc.)
//   2. move the corresponding sweep back into test_transcendental_1ulp.cpp
//      or test_mpfr_diff.cpp with a tier from the spec bucket (U35/GAMMA).
//   3. delete its entry from this file.
//
// Currently parked: (none)
//
// Graduated to the gated tree:
//   * sf64_lgamma zero-crossings [0.5, 3) — moved to
//     tests/mpfr/test_mpfr_diff.cpp (`lgamma_zeros` sweep, GAMMA tier)
//     after the zero-centered Taylor branches landed in
//     src/sleef/sleef_stubs.cpp (`lgamma_pos`).  The Taylor pivots at
//     x = 1 (window |x-1| ≤ 0.25) and x = 2 (window |x-2| ≤ 0.5) keep
//     absolute error → 0 as x → zero, so the ULP ratio against MPFR
//     stays bounded at the zeros and the full (0.5, 3) range fits
//     GAMMA = 1024 with ≥ 16× headroom.
//
// SPDX-License-Identifier: MIT

#include <cstdio>

int main() {
    std::printf("== experimental_precision (report-only) ==\n");
    std::printf("These sweeps are NOT gated. Any widening is a v1.2 item.\n\n");

    // No parked sweeps. The lgamma zero-crossings on (0.5, 3) graduated
    // to tests/mpfr/test_mpfr_diff.cpp at GAMMA tier — see the header
    // comment above for the graduation note.
    //
    // To park a new regime here:
    //   1. Restore the LCG / ulp_diff / Stats / record / report / sweep
    //      helpers (last revision before this commit kept them).
    //   2. Add a sweep block in this main().
    //   3. Document the regime in the file-header comment.

    std::printf("[report-only — no gating; exit 0]\n");
    return 0;
}
