/**
 * madhava_sec_capi.cpp — C ABI for the Madhava-Sec C++ engine.
 * Exposes build + score_all so any language (Python via ctypes,
 * Rust, Go, C) can call the native core with the mathematical
 * Cauchy-Schwarz guarantee intact.
 *
 *   C:  MadhavaSec* eng = madhava_sec_new(384, 64, 128);
 *       madhava_sec_build(eng, data, n);            // data: n*384 float32
 *       float* s = madhava_sec_score(eng, q);        // q: 384 float32
 *       free(s); madhava_sec_free(eng);
 *
 * BSL 1.1 | pay@winnex.ai
 */

#include "madhava_core.h"

extern "C" {

typedef void* MadhavaSec;

MadhavaSec madhava_sec_new(int dim, int stage1, int stage2) {
    auto* e = new madhava_sec::MadhavaSecEngine(dim, stage1, stage2);
    return (void*)e;
}

void madhava_sec_build(MadhavaSec h, const float* data, int n) {
    auto* e = (madhava_sec::MadhavaSecEngine*)h;
    e->build(data, n, false);
}

/* score_all: returns n floats, caller must free() */
float* madhava_sec_score(MadhavaSec h, const float* q) {
    auto* e = (madhava_sec::MadhavaSecEngine*)h;
    std::vector<float> s = e->score_all(q);
    float* out = (float*)malloc(s.size() * sizeof(float));
    std::memcpy(out, s.data(), s.size() * sizeof(float));
    return out;
}

/* max_score: single float, the classification score (max over centroids) */
float madhava_sec_max_score(MadhavaSec h, const float* q) {
    auto* e = (madhava_sec::MadhavaSecEngine*)h;
    std::vector<float> s = e->score_all(q);
    float m = -1e30f;
    for (float v : s) if (v > m) m = v;
    return m;
}

/* verify_bounds: violations + checked via out-params */
void madhava_sec_verify(MadhavaSec h, const float* q, long* violations, long* checked) {
    auto* e = (madhava_sec::MadhavaSecEngine*)h;
    auto [vrate, maxv, n] = e->verify_bounds(q, 1000);
    *violations = (long)(vrate * n);
    *checked = n;
}

/* verify_batch: verify a whole batch of queries in ONE C++ call.
   Samples up to max_checks_per_query vectors per query (verify_bounds'
   internal cap). Sums violations/checks across all queries — the honest
   full-scale bound audit for a benchmark. No per-query ctypes hop. */
void madhava_sec_verify_batch(MadhavaSec h, const float* queries, int n_queries,
                              long* violations, long* checked) {
    auto* e = (madhava_sec::MadhavaSecEngine*)h;
    long total_v = 0, total_c = 0;
    for (int i = 0; i < n_queries; i++) {
        auto [vrate, maxv, n] = e->verify_bounds(queries + (size_t)i * e->D, 1000);
        total_v += (long)(vrate * n);
        total_c += n;
    }
    *violations = total_v;
    *checked = total_c;
}

void madhava_sec_free(MadhavaSec h) {
    delete (madhava_sec::MadhavaSecEngine*)h;
}

} // extern "C"
