// libFuzzer target for narrow-int <-> f64 conversions.
//
// Covers all 8 narrow-int entry points:
//   sf64_from_i8 / sf64_from_i16 / sf64_from_u8 / sf64_from_u16
//   sf64_to_i8   / sf64_to_i16   / sf64_to_u8   / sf64_to_u16
//
// Consumes 16 bytes: byte[0] selects the op (mod 8), bytes [8..16) carry
// an 8-byte payload reinterpreted as the appropriate input type.
//
// Oracle:
//   * Widening (int -> f64): exact for all 8/16-bit integers (< 2^53),
//     so static_cast<double>(narrow) is the oracle.
//   * Narrowing (f64 -> int): C99 truncation semantics. For in-range
//     finite inputs, static_cast<NarrowT>(f64) is the oracle and a bit-
//     exact match is required. For NaN / +/-inf / out-of-range we only
//     require that the function returns *some* deterministic value
//     (we re-invoke and check consistency), matching sf64's documented
//     "deterministic wrap/saturate" guarantee.

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

// Is x finite AND its truncated value within [lo, hi]? Used to decide
// whether we can assert bit-exact match against static_cast.
bool in_range(double x, double lo, double hi) {
    if (!std::isfinite(x))
        return false;
    const double t = std::trunc(x);
    return t >= lo && t <= hi;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 16)
        return 0;

    const uint8_t sel = data[0] & 0x7; // 0..7

    // 8-byte payload for the integer / f64 input.
    uint64_t payload;
    std::memcpy(&payload, data + 8, 8);

    volatile uint64_t sink = 0;

    switch (sel) {
    case 0: { // sf64_from_i8
        const int8_t v = static_cast<int8_t>(payload & 0xff);
        const double got = sf64_from_i8(v);
        const double oracle = static_cast<double>(v);
        if (got != oracle)
            fuzz_fail("sf64_from_i8 != static_cast");
        sink ^= static_cast<uint64_t>(got);
        break;
    }
    case 1: { // sf64_from_i16
        const int16_t v = static_cast<int16_t>(payload & 0xffff);
        const double got = sf64_from_i16(v);
        const double oracle = static_cast<double>(v);
        if (got != oracle)
            fuzz_fail("sf64_from_i16 != static_cast");
        sink ^= static_cast<uint64_t>(got);
        break;
    }
    case 2: { // sf64_from_u8
        const uint8_t v = static_cast<uint8_t>(payload & 0xff);
        const double got = sf64_from_u8(v);
        const double oracle = static_cast<double>(v);
        if (got != oracle)
            fuzz_fail("sf64_from_u8 != static_cast");
        sink ^= static_cast<uint64_t>(got);
        break;
    }
    case 3: { // sf64_from_u16
        const uint16_t v = static_cast<uint16_t>(payload & 0xffff);
        const double got = sf64_from_u16(v);
        const double oracle = static_cast<double>(v);
        if (got != oracle)
            fuzz_fail("sf64_from_u16 != static_cast");
        sink ^= static_cast<uint64_t>(got);
        break;
    }
    case 4: { // sf64_to_i8
        const double x = bits_to_double(payload);
        const int8_t got = sf64_to_i8(x);
        if (in_range(x, -128.0, 127.0)) {
            const int8_t oracle = static_cast<int8_t>(x);
            if (got != oracle)
                fuzz_fail("sf64_to_i8 in-range mismatch");
        } else {
            // Determinism: repeat invocation must yield identical bits.
            if (sf64_to_i8(x) != got)
                fuzz_fail("sf64_to_i8 nondeterministic");
        }
        sink ^= static_cast<uint64_t>(static_cast<uint8_t>(got));
        break;
    }
    case 5: { // sf64_to_i16
        const double x = bits_to_double(payload);
        const int16_t got = sf64_to_i16(x);
        if (in_range(x, -32768.0, 32767.0)) {
            const int16_t oracle = static_cast<int16_t>(x);
            if (got != oracle)
                fuzz_fail("sf64_to_i16 in-range mismatch");
        } else {
            if (sf64_to_i16(x) != got)
                fuzz_fail("sf64_to_i16 nondeterministic");
        }
        sink ^= static_cast<uint64_t>(static_cast<uint16_t>(got));
        break;
    }
    case 6: { // sf64_to_u8
        const double x = bits_to_double(payload);
        const uint8_t got = sf64_to_u8(x);
        if (in_range(x, 0.0, 255.0)) {
            const uint8_t oracle = static_cast<uint8_t>(x);
            if (got != oracle)
                fuzz_fail("sf64_to_u8 in-range mismatch");
        } else {
            if (sf64_to_u8(x) != got)
                fuzz_fail("sf64_to_u8 nondeterministic");
        }
        sink ^= static_cast<uint64_t>(got);
        break;
    }
    case 7: { // sf64_to_u16
        const double x = bits_to_double(payload);
        const uint16_t got = sf64_to_u16(x);
        if (in_range(x, 0.0, 65535.0)) {
            const uint16_t oracle = static_cast<uint16_t>(x);
            if (got != oracle)
                fuzz_fail("sf64_to_u16 in-range mismatch");
        } else {
            if (sf64_to_u16(x) != got)
                fuzz_fail("sf64_to_u16 nondeterministic");
        }
        sink ^= static_cast<uint64_t>(got);
        break;
    }
    }

    (void)sink;
    return 0;
}
