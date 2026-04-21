#pragma once

// Soft-float IEEE-754 double-precision storage wrapper. Bit-identical to
// `double` on little-endian targets; the wrapper exists so frontends that
// can't use `double` directly (e.g. Metal Shading Language pre-3.0) can
// still pass the value through.
//
// SPDX-License-Identifier: MIT

#include "defines.h"

#include <cstdint>

namespace soft_fp64 {

// IEEE-754 binary64 storage wrapper. Bit-identical to `double` on little-endian
// targets; the wrapper exists so frontends that can't use `double` directly
// (e.g. Metal Shading Language pre-3.0) can still pass this through.
class float64_t {
  public:
    float64_t() = default;
    explicit constexpr float64_t(uint64_t bits) noexcept : data_(bits) {}

    constexpr uint64_t bits() const noexcept { return data_; }

  private:
    uint64_t data_{0};
};

static_assert(sizeof(float64_t) == 8, "float64_t must be exactly 64 bits");

} // namespace soft_fp64
