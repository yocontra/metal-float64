// libFuzzer target for sf64_fcmp(a, b, pred).
//
// Consumes 16 bytes (two f64 bit-patterns) and one predicate byte. The
// predicate is taken mod 16 to cover the full LLVM FCmpInst encoding.
//
// Oracle strategy:
//   * Ordered predicates (OEQ/OGT/OGE/OLT/OLE/ONE): compare against the
//     host `a OP b` for non-NaN pairs. If either is NaN, ordered predicates
//     must return 0 (and sf64_fcmp MUST agree).
//   * Unordered predicates (UEQ/UGT/UGE/ULT/ULE/UNE): compare against
//     `isnan(a) || isnan(b) || (a OP b)`.
//   * FCMP_ORD: `!isnan(a) && !isnan(b)`.
//   * FCMP_UNO: ` isnan(a) ||  isnan(b)`.
//   * FCMP_FALSE: must always return 0.
//   * FCMP_TRUE : must always return 1.

#include <soft_fp64/soft_f64.h>

#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

double bits_to_double(uint64_t bits) {
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}

[[noreturn]] void fuzz_fail(const char*) {
    __builtin_trap();
}

int expected(int pred, double a, double b) {
    const bool na = std::isnan(a);
    const bool nb = std::isnan(b);
    const bool uno = na || nb;
    const bool ord = !uno;
    switch (pred) {
    case 0:
        return 0;
    case 1:
        return (ord && a == b) ? 1 : 0; // OEQ
    case 2:
        return (ord && a > b) ? 1 : 0; // OGT
    case 3:
        return (ord && a >= b) ? 1 : 0; // OGE
    case 4:
        return (ord && a < b) ? 1 : 0; // OLT
    case 5:
        return (ord && a <= b) ? 1 : 0; // OLE
    case 6:
        return (ord && a != b) ? 1 : 0; // ONE
    case 7:
        return ord ? 1 : 0; // ORD
    case 8:
        return uno ? 1 : 0; // UNO
    case 9:
        return (uno || a == b) ? 1 : 0; // UEQ
    case 10:
        return (uno || a > b) ? 1 : 0; // UGT
    case 11:
        return (uno || a >= b) ? 1 : 0; // UGE
    case 12:
        return (uno || a < b) ? 1 : 0; // ULT
    case 13:
        return (uno || a <= b) ? 1 : 0; // ULE
    case 14:
        return (uno || a != b) ? 1 : 0; // UNE
    case 15:
        return 1;
    default:
        return 0;
    }
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 17)
        return 0;
    uint64_t ab, bb;
    std::memcpy(&ab, data, 8);
    std::memcpy(&bb, data + 8, 8);
    const int pred = static_cast<int>(data[16] & 0xf);

    const double a = bits_to_double(ab);
    const double b = bits_to_double(bb);

    const int got = sf64_fcmp(a, b, pred);
    const int want = expected(pred, a, b);

    if ((got & 1) != (want & 1))
        fuzz_fail("sf64_fcmp disagrees with oracle");

    // Invariants across predicate pairs that must hold for every (a,b):
    //   OEQ ^ UEQ = ORD xor nothing — not an xor but: OEQ => UEQ always,
    //   because UEQ = UNO | OEQ.
    if (sf64_fcmp(a, b, 1) == 1 && sf64_fcmp(a, b, 9) != 1)
        fuzz_fail("OEQ => UEQ broken");
    //   ORD and UNO are complementary.
    if (sf64_fcmp(a, b, 7) == sf64_fcmp(a, b, 8))
        fuzz_fail("ORD and UNO not complementary");
    //   FALSE always 0, TRUE always 1.
    if (sf64_fcmp(a, b, 0) != 0)
        fuzz_fail("FCMP_FALSE not 0");
    if (sf64_fcmp(a, b, 15) != 1)
        fuzz_fail("FCMP_TRUE not 1");

    return 0;
}
