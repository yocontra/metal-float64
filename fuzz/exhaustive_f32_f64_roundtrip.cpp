// Exhaustive f32 <-> f64 round-trip test.
//
// We iterate every 32-bit representation (0 .. 0xFFFF_FFFF) and assert
//
//     sf64_to_f32(sf64_from_f32(b)) == b
//
// holds bit-exactly for every non-NaN f32 bit-pattern.
//
// For NaN encodings, only the "is-NaN-ness" must round-trip.  The payload
// and quiet bit are NOT guaranteed to survive — soft_fp64 (like most
// conforming implementations) may quiet signalling NaNs on entry and
// canonicalize the payload.  This matches IEEE-754 and does not consitute
// a bug.  We document it in the test output so anyone re-running this
// test understands the NaN-class exception.
//
// Expected runtime on an M-series mac at -O2: ~5–15 minutes.
// The ctest registration timeout is 900s; we chunk the iteration into
// parallelizable sections in case someone wants to shard later, but keep
// the default single-threaded so it works unmodified on CI.

#include <soft_fp64/soft_f64.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr uint64_t kTotal = 1ULL << 32; // 4'294'967'296

float bits_to_f32(uint32_t b) {
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
}

uint32_t f32_to_bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
}

bool f32_is_nan(uint32_t b) {
    // IEEE-754 binary32: sign(1) | exp(8) | mantissa(23).
    // NaN: exp == 0xFF && mantissa != 0.
    return ((b >> 23) & 0xFFu) == 0xFFu && (b & 0x007F'FFFFu) != 0;
}

struct Stats {
    std::atomic<uint64_t> checked{0};
    std::atomic<uint64_t> nan_canonicalized{0}; // NaN round-tripped to NaN but different bits
    std::atomic<uint64_t> nan_preserved{0};     // NaN round-tripped bit-exact
    std::atomic<uint64_t> mismatches{0};        // non-NaN mismatches (BUG)
    // First offending bit-patterns (up to 8) for crash report.
    std::atomic<uint32_t> first_bad[8]{};
    std::atomic<uint32_t> first_bad_got[8]{};
    std::atomic<uint32_t> first_bad_count{0};
};

void check_range(uint64_t lo, uint64_t hi, Stats* s) {
    for (uint64_t i = lo; i < hi; ++i) {
        const uint32_t b = static_cast<uint32_t>(i);
        const float f_in = bits_to_f32(b);

        // sf64_from_f32 widens to double, then sf64_to_f32 narrows back.
        const double d = sf64_from_f32(f_in);
        const float f2 = sf64_to_f32(d);
        const uint32_t b2 = f32_to_bits(f2);

        s->checked.fetch_add(1, std::memory_order_relaxed);

        if (f32_is_nan(b)) {
            // NaN class: must remain NaN, but payload may canonicalize.
            if (!f32_is_nan(b2)) {
                // NaN went to a finite value — that IS a bug.
                uint64_t cnt = s->mismatches.fetch_add(1, std::memory_order_relaxed);
                uint32_t idx = s->first_bad_count.fetch_add(1, std::memory_order_relaxed);
                if (idx < 8) {
                    s->first_bad[idx].store(b, std::memory_order_relaxed);
                    s->first_bad_got[idx].store(b2, std::memory_order_relaxed);
                }
                (void)cnt;
            } else if (b2 != b) {
                s->nan_canonicalized.fetch_add(1, std::memory_order_relaxed);
            } else {
                s->nan_preserved.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            // Non-NaN: must be bit-exact.
            if (b2 != b) {
                s->mismatches.fetch_add(1, std::memory_order_relaxed);
                uint32_t idx = s->first_bad_count.fetch_add(1, std::memory_order_relaxed);
                if (idx < 8) {
                    s->first_bad[idx].store(b, std::memory_order_relaxed);
                    s->first_bad_got[idx].store(b2, std::memory_order_relaxed);
                }
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    const auto t0 = std::chrono::steady_clock::now();

    // Parallelize across hardware threads — 4B iterations is embarassingly
    // parallel.  Hardware threads default if available; fallback to 1.
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 1;
    // Cap at 16 so we don't spawn too many on big-iron CI boxes.
    if (nthreads > 16)
        nthreads = 16;

    std::fprintf(stderr,
                 "[exhaustive_f32_f64_roundtrip] checking all %llu f32 bit-patterns "
                 "using %u threads...\n",
                 static_cast<unsigned long long>(kTotal), nthreads);

    Stats stats;
    std::vector<std::thread> workers;
    workers.reserve(nthreads);

    const uint64_t chunk = kTotal / nthreads;
    for (unsigned t = 0; t < nthreads; ++t) {
        const uint64_t lo = chunk * t;
        const uint64_t hi = (t + 1 == nthreads) ? kTotal : (chunk * (t + 1));
        workers.emplace_back(check_range, lo, hi, &stats);
    }
    for (auto& w : workers)
        w.join();

    const auto t1 = std::chrono::steady_clock::now();
    const auto secs = std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count();

    const uint64_t checked = stats.checked.load();
    const uint64_t mism = stats.mismatches.load();
    const uint64_t nan_c = stats.nan_canonicalized.load();
    const uint64_t nan_p = stats.nan_preserved.load();

    std::fprintf(stderr,
                 "[exhaustive_f32_f64_roundtrip] checked=%llu mismatches=%llu "
                 "nan_preserved=%llu nan_canonicalized=%llu time=%llds\n",
                 static_cast<unsigned long long>(checked), static_cast<unsigned long long>(mism),
                 static_cast<unsigned long long>(nan_p), static_cast<unsigned long long>(nan_c),
                 static_cast<long long>(secs));

    if (checked != kTotal) {
        std::fprintf(stderr, "FAIL: checked count (%llu) != expected (%llu)\n",
                     static_cast<unsigned long long>(checked),
                     static_cast<unsigned long long>(kTotal));
        return 2;
    }

    if (mism != 0) {
        std::fprintf(stderr, "FAIL: %llu non-NaN round-trip mismatches\n",
                     static_cast<unsigned long long>(mism));
        const uint32_t shown = stats.first_bad_count.load();
        const uint32_t n = (shown < 8) ? shown : 8;
        for (uint32_t i = 0; i < n; ++i) {
            std::fprintf(stderr, "  example #%u: in=0x%08x got=0x%08x\n", i,
                         stats.first_bad[i].load(), stats.first_bad_got[i].load());
        }
        return 1;
    }

    std::fprintf(stderr, "PASS: all 2^32 f32 bit-patterns round-trip.\n"
                         "      NaN payloads are canonicalized on quiet; "
                         "this is IEEE-754-conformant and expected.\n");
    return 0;
}
