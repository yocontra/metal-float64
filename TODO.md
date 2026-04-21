# TODO

Pre-1.0 work. Single source of truth for what's open.

## Precision + test hygiene

- **Corpus-size claim parity.** Any README / file-header / test-comment
  corpus size must match the loop body — no `"2¹⁸"` prose when the
  loop runs 10 000, etc. Audit `tests/test_sqrt_fma_exact.cpp`,
  `tests/test_compare_all_predicates.cpp`,
  `tests/test_convert_widths.cpp` and resync prose to match loop
  bounds.
- **`sincos` correctness test.** The current block at
  `tests/test_transcendental_1ulp.cpp:500-564` compares
  `sf64_sincos(x)` against `sf64_sin(x)` / `sf64_cos(x)` — a tautology,
  not a correctness check. Add an independent MPFR-oracle sweep in
  `tests/mpfr/test_mpfr_diff.cpp` at U10 for both outputs.
- **pow per-region ULP claim parity.** `include/soft_fp64/soft_f64.h`
  Doxygen cites ≤3 / ≤2 / ≤6 ULP for the three pow windows, but all
  three CI gates in `tests/mpfr/test_mpfr_diff.cpp:533-541` are
  `U35 = 8`. Either tighten per-window assertions (preferred if the
  measured worst-case actually fits) or soften the Doxygen prose to
  "≤8 ULP across all three windows".
- **Fuzz ULP budgets.** `fuzz/fuzz_sqrt_fma.cpp` carries a 2²⁰-ULP
  budget with no written rationale in its file header
  (`fuzz/fuzz_transcendental.cpp` already has one). Either tighten
  both to the release tier or document the rationale at the top of
  each target.
- **Payne-Hanek stress breadth (libm-oracle sweep).** Extend the
  `k ∈ {2⁴⁰, 2⁴⁵, 2⁵⁰}` corpus in
  `tests/test_transcendental_1ulp.cpp:720-724` to also include
  `2⁵⁰⁰` and `2⁹⁰⁰`. The MPFR-oracle sweep in
  `tests/test_coverage_mpfr.cpp` already covers those.

## Numerical work

- **`logk_dd` DD-Horner rewrite.** `sf64_pow` drifts above U35 in the
  "near-unit base × huge exponent" corner (`x ∈ [0.5, 2], |y| ≳ 200`)
  because `logk_dd` in `src/sleef/sleef_inv_hyp_pow.cpp` evaluates its
  tail polynomial on `x².hi` as a plain double, capping the log DD at
  ~2⁻⁵⁶ relative. Fix: evaluate the minimax polynomial in full DD
  arithmetic (DD Horner) and promote coefficient storage to DD pairs
  for the high-degree terms. Expected to move the worst-case pow from
  ~40 ULP to ≤4 ULP across the full double range. Also closes the
  `lgamma` `(0.5, 3)` zero-crossing report-only sweep.

## Feature work — not yet implemented

- **Non-RNE rounding modes.** `sf64_*_r(mode, …)` variants taking an
  explicit mode (`SF64_RNE`, `SF64_RTZ`, `SF64_RUP`, `SF64_RDN`,
  `SF64_RNA` — IEEE-754 §4.3). Enables hardware-emulation frontends
  (RISC-V `frm` CSR, ARM FPCR, x86 MXCSR), interval arithmetic, and
  freestanding runtimes. No ABI break — default stays RNE. Internal
  round-pack primitives in `src/internal.h` already abstract the
  rounding step; parametrize on mode. TestFloat emits vectors for all
  five modes; MPFR oracle: swap `MPFR_RNDN`.
- **IEEE exception flags + thread-local fenv.** Strict §7 conformance.
  Flag bits: `SF64_FE_{INVALID, DIVBYZERO, OVERFLOW, UNDERFLOW,
  INEXACT}` matching `<fenv.h>`. Entry points: `sf64_fe_{clear, test,
  raise, getall}`; optional `sf64_fe_state_t` opaque `_save` /
  `_restore` for freestanding / GPU contexts. Thread-local default
  (`thread_local` / `__thread`); build option
  `SOFT_FP64_FENV=tls|explicit|disabled`. Measured cost expected
  ≤10% on the hot arithmetic path, zero when disabled. TestFloat
  already emits expected-flag bits.
- **sNaN payload preservation.** Currently quiet-on-entry (sNaN →
  qNaN with canonical payload). Consumers needing §6.2 payload
  preservation require a `SOFT_FP64_SNAN_PROPAGATE` build option.
  Depends on the exception-flag work above (preservation raises
  `SF64_FE_INVALID`). TestFloat has dedicated sNaN vectors; wire them.
- **`soft-fp128` sibling.** Same design playbook (Mesa arithmetic
  port + SLEEF transcendentals + TestFloat + MPFR oracle) extended to
  113-bit significand. Storage wrapper + full conversion matrix
  (`f64 ↔ f128`, `i128 ↔ f128`), u10 transcendentals vs MPFR
  300-bit. Likely ships as a separate package once fp64 stabilizes.

## Infra debt

- **`scripts/pre_push.sh` local guard.** Full unfiltered ctest,
  `clang-format-19 --dry-run --Werror` over `include/ src/ tests/`
  (excluding vendored trees), and
  `python3 bench/compare.py current.json bench/baseline.json
  --threshold=0.10` clean.

## Not on the roadmap

- **fp16 / bfloat16.** Different design space — table-driven
  implementations win at that precision. No fp64 synergy.
- **Decimal FP (IEEE-754 §3.5).** Different rounding philosophy; see
  `libdfp` or Intel's DFP library.
- **Complex-number math** (`csin`, `clog`, etc). Pure wrapper work on
  top of real scalars; belongs in a consumer.
- **Guest-FPU emulation glue.** `fenv` compat, `softfp` ABI lowering,
  calling-convention shims belong in the frontend (compiler runtime,
  emulator), not here. `sf64_*` is the contract; how you call it is
  your problem.
- **Runtime CPU dispatch for fast paths.** The point is
  architecture-independent bit-exactness. If you have fast native
  fp64, you don't need this library.
