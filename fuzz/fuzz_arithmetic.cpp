// libFuzzer target for sf64 arithmetic (add/sub/mul/div).
//
// libFuzzer feeds us arbitrary byte sequences; we consume 16 bytes as two
// `uint64_t` bit-patterns (reinterpreted as doubles) and exercise the
// basic arithmetic ops.  We're NOT trying to ULP-grade the results here —
// that's what `tests/test_arithmetic_exact.cpp` and the MPFR harness
// do.  The fuzzer hunts for:
//
//   * Traps, aborts, assertion failures inside sf64_* implementations.
//   * Sanitizer findings (UBSan shift overflows, ASan OOB on internal
//     tables, etc.).
//   * Bit-patterns that are not valid IEEE-754 encodings — note that any
//     64-bit bit-pattern is a valid IEEE-754 double (including all NaN
//     payloads), so "validity" reduces to "no floating-point exception
//     escaped our sanitizer build".
//   * Commutativity breaks for add/mul on the non-NaN subset.
//   * Identity breaks (add-0, mul-1) on the non-NaN subset.

#include <soft_fp64/soft_f64.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

double bits_to_double(uint64_t bits) {
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}

uint64_t double_to_bits(double d) {
    uint64_t bits;
    std::memcpy(&bits, &d, sizeof(bits));
    return bits;
}

bool is_nan(double d) {
    uint64_t b = double_to_bits(d);
    // IEEE-754: exponent all ones AND mantissa non-zero => NaN.
    return ((b >> 52) & 0x7ffu) == 0x7ffu && (b & 0x000f'ffff'ffff'ffffULL) != 0;
}

// libFuzzer treats __builtin_trap() as a hard crash, which is what we want.
[[noreturn]] void fuzz_fail(const char* /*msg*/) {
    __builtin_trap();
}

// Two bit-patterns are "arithmetically equal" if they represent the same
// value — we cannot just compare bits because +0 and -0 are equal and the
// sign of NaN and payload bits can differ.  We compare on bit equality for
// non-NaN, non-zero values and treat NaN==NaN as true.
bool same_value(double x, double y) {
    if (is_nan(x) && is_nan(y))
        return true;
    if (x == 0.0 && y == 0.0)
        return true; // +0 == -0
    return double_to_bits(x) == double_to_bits(y);
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 16)
        return 0;

    uint64_t ab, bb;
    std::memcpy(&ab, data, 8);
    std::memcpy(&bb, data + 8, 8);

    const double a = bits_to_double(ab);
    const double b = bits_to_double(bb);

    // --- Exercise every op; any trap/abort inside counts as a finding.
    const double r_add = sf64_add(a, b);
    const double r_sub = sf64_sub(a, b);
    const double r_mul = sf64_mul(a, b);
    const double r_div = sf64_div(a, b);

    // Sink results into sanitizer-visible memory so optimizer can't drop calls.
    volatile uint64_t sink = 0;
    sink ^= double_to_bits(r_add);
    sink ^= double_to_bits(r_sub);
    sink ^= double_to_bits(r_mul);
    sink ^= double_to_bits(r_div);
    (void)sink;

    // --- Commutativity: add and mul on non-NaN inputs must commute.
    if (!is_nan(a) && !is_nan(b)) {
        const double r_add_rev = sf64_add(b, a);
        const double r_mul_rev = sf64_mul(b, a);
        if (!same_value(r_add, r_add_rev))
            fuzz_fail("add not commutative");
        if (!same_value(r_mul, r_mul_rev))
            fuzz_fail("mul not commutative");
    }

    // --- Identity: a + 0 == a, a * 1 == a for finite, non-NaN a.
    // We skip NaN (propagates), +/-inf (inf + 0 is inf but sign rules
    // differ under rounding modes we don't toggle here — still safe, but
    // keeping this tight makes divergences louder).
    if (!is_nan(a)) {
        const double zero = 0.0;
        const double one = 1.0;
        const double add0 = sf64_add(a, zero);
        const double mul1 = sf64_mul(a, one);
        // Exception: a == -0 and we add +0 yields +0 in round-to-nearest,
        // which is the IEEE-754 rule; `same_value` treats +0==-0 so this
        // comparison remains correct.
        if (!same_value(a, add0))
            fuzz_fail("a + 0 != a");
        if (!same_value(a, mul1))
            fuzz_fail("a * 1 != a");
    }

    // --- Sub-from-self: a - a == 0 for finite non-NaN a.
    if (!is_nan(a) && !__builtin_isinf(a)) {
        const double self_sub = sf64_sub(a, a);
        if (is_nan(self_sub))
            fuzz_fail("a - a is NaN for finite a");
        if (self_sub != 0.0)
            fuzz_fail("a - a != 0 for finite a");
    }

    return 0;
}
