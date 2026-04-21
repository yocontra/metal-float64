#pragma once

// vec<T, N> — small fixed-width vector wrapper.
//
// SPDX-License-Identifier: MIT

#include "defines.h"

#include <cstddef>

namespace soft_fp64 {

template <typename T, std::size_t N> struct vec {
    static_assert(N >= 1 && N <= 4, "vec<T,N> supported for N in 1..4");

    T data[N];

    constexpr T& operator[](std::size_t i) noexcept { return data[i]; }
    constexpr const T& operator[](std::size_t i) const noexcept { return data[i]; }

    static constexpr std::size_t size() noexcept { return N; }
};

template <typename T> using vec1 = vec<T, 1>;
template <typename T> using vec2 = vec<T, 2>;
template <typename T> using vec3 = vec<T, 3>;
template <typename T> using vec4 = vec<T, 4>;

} // namespace soft_fp64
