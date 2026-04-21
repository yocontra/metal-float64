// Smoke test: confirms the library links and the umbrella header compiles.
// Takes the address of one symbol from each ABI group to force linkage.
// Per-function correctness lives in the sibling test_*.cpp files.
//
// SPDX-License-Identifier: MIT

#include "soft_fp64/soft_fp64.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace mf = soft_fp64;

int main() {
    mf::float64_t zero;
    assert(zero.bits() == 0);

    mf::float64_t one_point_five(0x3FF8000000000000ULL);
    double as_host;
    static_assert(sizeof(double) == sizeof(uint64_t));
    const uint64_t bits = one_point_five.bits();
    std::memcpy(&as_host, &bits, sizeof(double));
    assert(as_host == 1.5);

    mf::vec<int, 3> v{{1, 2, 3}};
    assert(v[0] == 1 && v[1] == 2 && v[2] == 3);
    assert(v.size() == 3);

    // Reference every ABI symbol group to prove they link.
    auto* p_add = &sf64_add;
    auto* p_fcmp = &sf64_fcmp;
    auto* p_from32 = &sf64_from_f32;
    auto* p_sqrt = &sf64_sqrt;
    auto* p_floor = &sf64_floor;
    auto* p_isnan = &sf64_isnan;
    assert(p_add != nullptr);
    assert(p_fcmp != nullptr);
    assert(p_from32 != nullptr);
    assert(p_sqrt != nullptr);
    assert(p_floor != nullptr);
    assert(p_isnan != nullptr);

    return 0;
}
