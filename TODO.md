# TODO

Single source of truth for open work. Items closed at 1.0 / 1.1 are
recorded in `CHANGELOG.md`, not here.

## Pre-1.1 — in flight

Uncommitted in the working tree on top of commit `c3c1b90`. These
block the `v1.1.0` tag.

### Track D perf recovery — D3/D4 landed, E/F/G outstanding

D3 (`SF64_ALWAYS_INLINE` constant-propagation rescue on arithmetic
helpers) and D4 (stack-local `sf64_internal_fe_acc` flag accumulator
threaded through `src/arithmetic.cpp`, `src/convert.cpp`,
`src/sqrt_fma.cpp`; `initial-exec` TLS model) are in the working tree.
ctest is green in both `SOFT_FP64_FENV=tls` (19/19) and
`SOFT_FP64_FENV=disabled` (18/18) Release builds. Full details of the
chosen design live in the plan file
`/Users/contra/.claude/plans/this-is-an-unfinished-quirky-crab.md`
(Tracks D/E/F/G).

Remaining bench deltas on macOS M2 Max, Release, `--min-time-ms=500`
(all three block `v1.1.0`):

| op     | 1.0    | disabled        | tls (vs disabled) |
|--------|--------|-----------------|-------------------|
| pow    | 1324.2 | 1899.6 (+43%) ✗ | 1953.2 (+2.8%)    |
| floor  |    6.5 |    7.3 (+12%) ✗ |    6.4 (−12.9%)   |
| to_i32 |    4.9 |    2.7 (−45%)   |    7.1 (+166%) ✗  |

All other hot ops (add/sub/mul/div/sqrt/fma and every transcendental
except pow) are within +10% of 1.0 in both modes.

**Track E — `sf64_pow` cross-TU inlining fix.** Root cause: DD
helpers in `src/sleef/sleef_common.h` (ddadd2_*, ddmul_*, ddsqu_*,
ddrec_*, dddiv_*, ddnormalize_*, dd_to_d, ddscale_*) call `sf64_add`
/ `sf64_sub` / `sf64_mul` / `sf64_div` / `sf64_fma` as `extern "C"`
ABI entries. `SF64_ALWAYS_INLINE` does not cross TU boundaries
without LTO, so each DD primitive emits real call instructions,
Track B2's `mode` parameter passes via register as a runtime value,
and `sf64_internal_should_round_up`'s 5-way switch
(`src/internal.h:192-205`) never folds. `sf64_pow` makes ~160+
arithmetic ABI calls per invocation; ~3.6 ns per call of extra
dispatch + switch + frame cost = +576 ns on the 1324 ns baseline,
matching the +43% observation. Disasm confirmed on the in-tree
artifact `build-disabled/CMakeFiles/soft_fp64.dir/src/sleef/sleef_inv_hyp_pow.cpp.o`.

Fix is the D3-Step-3 template promotion: add a new header
`src/internal_arith.h` exposing `sf64_internal_{add,sub,mul,div,fma,sqrt}_rne`
as `SF64_ALWAYS_INLINE` bodies (with no `mode` parameter — RNE is
hard-specialized). `sf64_add` etc. public entries become thin
wrappers. All 284 direct `sf64_*` arithmetic call sites across
`src/sleef/*.cpp` + `src/sleef/sleef_common.h` swap to the `_rne`
variants, threading an `sf64_internal_fe_acc& fe` through every
helper and every SLEEF public entry. Scope: ~800 lines move from
anon namespace to header; threading dozens of functions;
284 sed-style call site swaps. **Alternative rejected:** LTO on the
library target (breaks static archive ABI + `install-smoke` `nm -g`
gate).

- Critical files: `src/internal_arith.h` (new); `src/arithmetic.cpp`,
  `src/sqrt_fma.cpp` (move bodies out + keep `_r_impl` for directed
  rounding); `src/sleef/sleef_common.h` (DD helpers take fe, call
  `_rne`); `src/sleef/sleef_trig.cpp`, `sleef_exp_log.cpp`,
  `sleef_inv_hyp_pow.cpp`, `sleef_stubs.cpp` (public entries declare
  / thread / flush fe; call sites swap to `_rne`); `CHANGELOG.md`
  (Performance table update).
- Verification: full unfiltered ctest green in both modes; disasm of
  `sf64_pow` shows no `bl _sf64_add`/`bl _sf64_mul` calls and no
  5-way mode switch; `compare.py current-disabled.json
  bench/baseline.json --threshold=0.10` — pow within +10%;
  tls-vs-disabled delta on pow stays at ~+3%.
- Cross-verification: cold-briefed `staff-code-reviewer` with diff +
  pre/post disasm snippet + pre/post `compare.py` tables.

**Track F — `sf64_floor` disabled-mode +12%.** Source is
byte-identical to 1.0; every helper floor calls was already
`SF64_ALWAYS_INLINE` pre-D3; no call path through arithmetic.cpp /
sqrt_fma.cpp / convert.cpp. The 0.78 ns delta on a 6.5 ns baseline is
consistent with measurement noise.
- Action: re-measure with `--min-time-ms=2000` ×3 samples. If all
  three land ≤+10% vs 1.0, note in `CHANGELOG.md` that 500 ms runs
  are jitter-dominated on this op and declare resolved. If the +12%
  persists, carry forward to v1.1.1 as instruction-cache alignment
  investigation and ship 1.1 with the regression documented.
- Critical files: none in `src/`. `CHANGELOG.md` note + TODO.md
  carry-over if unresolved.

**Track G — `sf64_to_i32` tls-vs-disabled +166% (structural Apple
Silicon TLS floor).** In tls mode the single `fe.flush()` at public
entry compiles to `bl __tlv_get_addr; ldr; orr; str`, adding ~4.45 ns
(one Apple Silicon TLS roundtrip). On a 2.68 ns `to_i32` baseline
that is +166%; in absolute terms it is ~5 ns and is the structural
minimum for any IEEE-§7 sticky-flag implementation that writes to
TLS. Investigation confirmed:
  (a) the `fe_acc` stack accumulator is NOT the overhead — the TLS
      store itself is;
  (b) per-site `SF64_FE_RAISE` (no accumulator) would be strictly
      worse (4–6 TLS stores vs 1);
  (c) every `to_iN`/`to_uN` entry is multi-raise-site, so no
      single-raise-site bypass candidate exists;
  (d) `initial-exec` TLS model is already applied — Darwin's
      `__tlv_get_addr` ABI limits its effect.

No `src/` change. Budget revision + documentation:
- Update `bench/compare.py` with a `--cheap-op-absolute-budget=5.0`
  flag that exempts any op with baseline < 15 ns from the
  percentage gate provided the absolute delta is ≤ the flag value;
  emit a "carveout" row visible to reviewers.
- Update `.github/workflows/ci.yml` `bench-regression` job to pass
  the new flag.
- Update `CHANGELOG.md` 1.1.0 entry with a "Performance — fenv-tls
  mode" subsection stating the absolute TLS cost (~5 ns per public
  op on Apple Silicon) and the cheap-op carveout.
- Add "Known perf properties" subsection below cross-referencing
  the `explicit`-mode follow-up that eliminates the TLS floor.
- Critical files: `bench/compare.py`, `.github/workflows/ci.yml`,
  `CHANGELOG.md`.

**Track E/F/G ordering + risk.**
1. Track E first — largest regression, clear mechanical fix, high
   scope-creep risk (keep the TU-split diff surgical; ctest green at
   every intermediate commit).
2. Track G next — no `src/` change, just doc + bench-gate wiring.
   Unblocks the release-gate bench comparison.
3. Track F last — cheap remeasure; resolve or carry to v1.1.1.

### Existing pre-1.1 polish (carry-over)

- **Fenv raise-site coverage across arithmetic / convert / sqrt-fma.**
  `src/arithmetic.cpp`, `src/convert.cpp`, `src/sqrt_fma.cpp` add
  INEXACT / UNDERFLOW / OVERFLOW / INVALID raise sites at the
  IEEE-754 §7 locations; `src/internal_fenv.h` now carries a
  two-raise-path scheme (a hot inline accumulator macro plus the
  out-of-line `SF64_FE_RAISE`) so the INEXACT fast path does not
  touch TLS storage on every op. Before commit: run the TestFloat
  `fl2` column gate over the full 7.16M-vector corpus under
  `SOFT_FP64_FENV=tls` and confirm every expected flag bit is
  raised — the CI cell exists but needs a green pass after the
  new raise sites land.
- **ACPP Metal adapter staging picks up `SOFT_FP64_FENV_MODE`.**
  `adapters/acpp_metal/CMakeLists.txt` and
  `cmake/rewrite_sleef_include.cmake` need to propagate the
  top-level `SOFT_FP64_FENV` build option into the staged source
  tree. Without it the adapter's Metal bitcode always compiles with
  `SOFT_FP64_FENV_MODE=0` (disabled) regardless of the core
  configuration, so fenv flag raising silently no-ops on Metal.

### Track A — 1.0 precision closures (not yet started)

Parked non-blocking at the 1.0 tag; target is v1.1.

- **A1 — `logk_dd` DD-Horner rewrite.**
  `src/sleef/sleef_inv_hyp_pow.cpp:207-235`. Current: 7-term tail
  polynomial evaluated against `x².hi` as a plain double (line 227),
  capping the log DD at ~2⁻⁵⁶ relative. Change: replace
  `poly_array(x2.hi, kLogkCoef, ...)` with a DD Horner chain
  (`ddmul_dd_dd_d` + `ddadd2_dd_dd_d` per step) against the full
  `x2` DD pair; promote the top 2–3 coefficients to DD-pair
  storage. Signature / call site in `src/sleef/sleef_stubs.cpp`
  unchanged. Expected: `sf64_pow` worst-case ~40 ULP → ≤4 ULP across
  double range; `tests/mpfr/test_mpfr_diff.cpp` pow gates demote
  U35 → U10 if measured worst-case fits; `sf64_lgamma` `(0.5, 3)`
  zero-crossings graduate from
  `tests/experimental/experimental_precision.cpp` to the shipped
  suite at GAMMA tier.
- **A2 — `sf64_sinh` overflow-boundary fix.**
  `src/sleef/sleef_inv_hyp_pow.cpp:550-557`. Current flushes to ±∞ at
  `|x| > 709.78` (the `exp` overflow threshold). Correct threshold
  is `log(2·DBL_MAX) ≈ 710.4758`. Insert an intermediate branch for
  `|x| ∈ (709.78, 710.4758]` that evaluates
  `sf64_internal_exp_core(|x| − LN2) * 0.5`. Extend
  `tests/mpfr/test_mpfr_diff.cpp`'s sinh sweep with spot-check rows
  at `{709.79, 710.0, 710.4, 710.48, ±}` gated at U35 = 8.
- **A3 — Payne–Hanek deep-reduction breadth.**
  `tests/test_transcendental_1ulp.cpp:720-724` — append
  `std::ldexp(1.0, 500)` and `std::ldexp(1.0, 900)` to the `ks[]`
  array to match `tests/test_coverage_mpfr.cpp:248-254`.

### Track B — non-RNE rounding mode tests (partial)

Public `sf64_*_r` surface landed with B1/B2/B3; B4/B5 test and bench
work is partial.

- **B4.1 — MPFR harness mode parametrization.** Thread `mode` through
  `record()` / `sweep*_uniform()` in `tests/mpfr/test_mpfr_diff.cpp`.
  Map `sf64_rounding_mode` → `mpfr_rnd_t`
  (`MPFR_RNDN/Z/U/D/A`). Run every existing bit-exact + U10/U35
  sweep in all five modes.
- **B4.2 — TestFloat mode loop.**
  `tests/testfloat/run_testfloat.cpp` currently hardcodes RNE; add a
  mode loop + regenerate `tests/testfloat/vectors/` per mode.
- **B4.3 — per-mode bit-exact rows.**
  `tests/test_arithmetic_exact.cpp` + `tests/test_sqrt_fma_exact.cpp`
  + `tests/test_convert_widths.cpp` grow explicit per-mode rows
  guarded by host FPU fenv.
- **B4.4 — new fuzz target.** `fuzz/fuzz_rounding_modes.cpp`
  exercises the `_r(mode, ...)` surface.
- **B5 — per-mode bench deltas.** Track E refactor may re-measure
  `_r` path cost; document in `CHANGELOG.md`.

### Track C — fenv test wiring (partial)

- **TestFloat `fl2` column.** `tests/testfloat/run_testfloat.cpp:21`
  currently carries the "flags ignored" carve-out; remove it, parse
  the `fl2` token, compare against `sf64_fe_getall()` after each
  row, fail on mismatch. 7.16M-vector corpus must pass under
  `SOFT_FP64_FENV=tls`.
- **Thread-safety test.** Two pthreads calling arithmetic with
  differing inputs must observe independent flag accumulators —
  extend `tests/test_fenv.cpp` or add a dedicated
  `tests/test_fenv_threads.cpp`.

## Post-1.1

### Numerical

- **`sf64_lgamma` zero-crossings on `(0.5, 3)`.** `lgamma(x)` vanishes
  at `x = 1` and `x = 2`; near those zeros the result is O(1e-5) but
  the absolute error floor of any log-of-Γ path is O(ulp(1)) ≈ 2.2e-16,
  so the ULP ratio blows past GAMMA=1024 even with a perfectly
  computed log ingredient. v1.1's `logk_dd` DD-Horner rewrite
  confirmed the issue is algorithmic, not ingredient-precision. The
  proper fix is a **zero-centered Taylor expansion** around `x=1` and
  `x=2` — a branch inside `sf64_lgamma` that detects the vanishing
  regime and returns `(x-1)·P₁(x)` or `(x-2)·P₂(x)` with
  coefficients computed from the known series for `ψ(x)`. Currently
  parked report-only in `tests/experimental/experimental_precision.cpp`;
  gated promotion to GAMMA tier requires the rewrite to land first.

### Feature surface (not yet implemented)

- **`SOFT_FP64_FENV=explicit` caller-provided state ABI.** v1.1 reserves
  the `explicit` mode in CMake but compiles it identically to `disabled`
  (zero-cost no-op raise sites, `sf64_fe_*` surface present but
  stateless). The target shape is `sf64_fe_*` variants that take an
  `sf64_fe_state_t*` directly, enabling GPU/freestanding kernels that
  can't rely on `thread_local`. Requires a parallel ABI to avoid
  breaking 1.x consumers. TestFloat `run_testfloat.cpp` skips the
  7.16M-vector flag gate under explicit mode today; wire it up once
  the surface lands.
- **sNaN payload preservation.** v1.0–v1.1 quiet sNaN on entry (sNaN →
  qNaN with canonical payload). v1.1 raises `SF64_FE_INVALID` on that
  entry when fenv is enabled. Consumers needing §6.2 full payload
  preservation require a `SOFT_FP64_SNAN_PROPAGATE` build option that
  preserves the signalling payload bits through the quiet-bit force.
  TestFloat has dedicated sNaN vectors; `tests/testfloat/run_testfloat.cpp`
  currently skips them (documented carve-out). Target: v1.2.
- **`soft-fp128` sibling.** Same design playbook (Mesa arithmetic
  port + SLEEF transcendentals + TestFloat + MPFR oracle) extended to
  113-bit significand. Storage wrapper + full conversion matrix
  (`f64 ↔ f128`, `i128 ↔ f128`), u10 transcendentals vs MPFR
  300-bit. Likely ships as a separate package once fp64 stabilizes.

### ACPP Metal adapter follow-ups

- **`__acpp_sscp_lgamma_r_f64` forwarding.** The adapter currently
  leaves the trap-stub in place (see
  `adapters/acpp_metal/README.md` → "Skipped"). Blocks on adding a
  `sf64_lgamma_r` core entry point — computing the Γ sign from
  `sf64_tgamma(x)` round-trips through an overflow-prone path for
  `|x| > 170`. Will land with whichever core release exposes
  `sf64_lgamma_r`.
- **`_r`-variant forwarders.** Once the non-RNE `sf64_*_r` surface
  from v1.1 is stable on the Metal target, the adapter gains
  one-line `__acpp_sscp_soft_f64_*_r` forwarders. No core change
  required — pure forwarding, zero ULP to add.
- **`sf64_fe_*` surface on Metal.** The adapter may optionally
  re-export the fenv surface for kernels that care about
  accumulated-flag reporting. `SOFT_FP64_FENV=disabled` stays the
  default on GPU targets (no `thread_local` support on Metal SSCP);
  an `explicit`-state ABI (v1.2+) lines this up.

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
