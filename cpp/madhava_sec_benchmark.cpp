/**
 * Madhava-Sec Benchmark — C++ Core
 * ==================================
 * Valida as 3 Claims do framework em C++ com SIMD:
 *   1. Cauchy-Schwarz UB Pruning (0 violacoes)
 *   2. int8 Quantization Error (cosine > 0.9999)
 *   3. GS Diversity via ortogonalizacao
 *
 * Compilar: make
 * Executar: ./madhava_sec_benchmark
 *
 * BSL 1.1 | pay@winnex.ai
 */

#include "madhava_core.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

using namespace madhava_sec;

// ============================================================
// HELPERS
// ============================================================

float rand_float(std::mt19937& rng, std::uniform_real_distribution<float>& dist) {
    return dist(rng);
}

void normalize(float* v, int d) {
    float n = 0;
    for (int i = 0; i < d; i++) n += v[i] * v[i];
    n = std::sqrt(n);
    if (n > 1e-10f)
        for (int i = 0; i < d; i++) v[i] /= n;
}

// ============================================================
// CLAIM 1: BOUND VIOLATIONS
// ============================================================

void test_bound_violations(int dim, int n_vecs, int n_queries) {
    std::cout << "\n=== CLAIM 1: Cauchy-Schwarz Bound Violations ===" << std::endl;
    std::cout << "  D=" << dim << " N=" << n_vecs << " Q=" << n_queries << std::endl;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Gerar dados sinteticos na esfera unitaria
    std::vector<float> data((size_t)n_vecs * dim);
    for (int i = 0; i < n_vecs; i++) {
        for (int j = 0; j < dim; j++)
            data[(size_t)i * dim + j] = rand_float(rng, dist);
        normalize(&data[(size_t)i * dim], dim);
    }

    // Build engine
    int s1 = std::min(64, dim);
    int s2 = std::min(128, dim);
    MadhavaSecEngine engine(dim, s1, s2);
    engine.build(data.data(), n_vecs, true);

    // Memoria
    std::cout << "  Memoria: " << engine.memory_gb() << " GB" << std::endl;

    // Queries e verificacao
    std::vector<float> queries((size_t)n_queries * dim);
    for (int qi = 0; qi < n_queries; qi++) {
        for (int j = 0; j < dim; j++)
            queries[(size_t)qi * dim + j] = rand_float(rng, dist);
        normalize(&queries[(size_t)qi * dim], dim);
    }

    long total_viol = 0, total_checked = 0;
    double max_viol_ratio = 0;

    for (int qi = 0; qi < n_queries; qi++) {
        auto [vrate, maxv, checked] = engine.verify_bounds(
            &queries[(size_t)qi * dim], 200);
        total_checked += checked;
        total_viol += (long)(vrate * checked);
        if (maxv > max_viol_ratio) max_viol_ratio = maxv;
    }

    double viol_pct = total_checked > 0 ?
        100.0 * total_viol / total_checked : 0;
    std::cout << "  Violacoes: " << total_viol << "/" << total_checked
              << " (" << viol_pct << "%)" << std::endl;
    std::cout << "  Max ratio: " << max_viol_ratio << std::endl;

    if (viol_pct == 0)
        std::cout << "  ✅ 0% violacao — bound garantido" << std::endl;
    else
        std::cout << "  ❌ " << viol_pct << "% violacao!" << std::endl;
}

// ============================================================
// CLAIM 2: INT8 QUANTIZATION ERROR
// ============================================================

void test_int8_quantization(int dim, int n_vecs) {
    std::cout << "\n=== CLAIM 2: int8 Quantization Error ===" << std::endl;
    std::cout << "  D=" << dim << " N=" << n_vecs << std::endl;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Gerar dados e calcular escalas
    std::vector<float> data((size_t)n_vecs * dim);
    std::vector<float> max_abs(dim, 0);

    for (int i = 0; i < n_vecs; i++) {
        for (int j = 0; j < dim; j++) {
            float v = rand_float(rng, dist);
            data[(size_t)i * dim + j] = v;
            if (std::fabs(v) > max_abs[j]) max_abs[j] = std::fabs(v);
        }
        normalize(&data[(size_t)i * dim], dim);
    }

    // Recalcular max_abs apos normalizacao
    std::fill(max_abs.begin(), max_abs.end(), 0);
    for (int i = 0; i < n_vecs; i++) {
        for (int j = 0; j < dim; j++) {
            float v = data[(size_t)i * dim + j];
            if (std::fabs(v) > max_abs[j]) max_abs[j] = std::fabs(v);
        }
    }

    // Escalas
    std::vector<float> scale(dim);
    for (int j = 0; j < dim; j++)
        scale[j] = std::max(max_abs[j] / 127.0f * 1.05f, 1e-10f);

    // Quantizar e medir erro
    double total_mse = 0, total_cos = 0;
    int8_t* qvec = new int8_t[dim];
    float* rvec = new float[dim];

    for (int i = 0; i < n_vecs; i++) {
        const float* v = &data[(size_t)i * dim];

        // Quantizar
        for (int j = 0; j < dim; j++) {
            int qi = (int)(v[j] / scale[j] + (v[j] >= 0 ? 0.5f : -0.5f));
            if (qi > 127) qi = 127;
            if (qi < -128) qi = -128;
            qvec[j] = (int8_t)qi;
        }

        // Reconstruir
        for (int j = 0; j < dim; j++)
            rvec[j] = (float)qvec[j] * scale[j];

        // MSE
        double mse = 0;
        for (int j = 0; j < dim; j++) {
            double diff = v[j] - rvec[j];
            mse += diff * diff;
        }
        total_mse += mse / dim;

        // Cosseno
        float dot = dot_f32(v, rvec, dim);
        float nv = std::sqrt(dot_f32(v, v, dim));
        float nr = std::sqrt(dot_f32(rvec, rvec, dim));
        float cos = dot / (nv * nr + 1e-10f);
        total_cos += cos;
    }

    delete[] qvec;
    delete[] rvec;

    double mean_mse = total_mse / n_vecs;
    double mean_cos = total_cos / n_vecs;

    std::cout << "  MSE medio: " << mean_mse << std::endl;
    std::cout << "  Cosine medio: " << mean_cos << std::endl;
    std::cout << "  Compressao: 4x (float32 -> int8)" << std::endl;

    if (mean_cos > 0.9999)
        std::cout << "  ✅ Cosine > 0.9999" << std::endl;
    else
        std::cout << "  ⚠️ Cosine = " << mean_cos << std::endl;
}

// ============================================================
// CLAIM 3: GS RANDOM PROJECTION ORTHOGONALITY
// ============================================================

void test_gs_orthogonality(int dim, int n_proj) {
    std::cout << "\n=== CLAIM 3: MGS Orthogonality ===" << std::endl;
    std::cout << "  D=" << dim << " N_proj=" << n_proj << std::endl;

    MadhavaSecEngine engine(dim, n_proj, std::min(n_proj * 2, dim));

    float* P = engine.mk_proj_mgs(n_proj, 42);

    // Medir ||P * P^T - I||
    double max_err = 0;
    for (int i = 0; i < n_proj; i++) {
        for (int j = 0; j < n_proj; j++) {
            float dot = dot_f32(&P[i * dim], &P[j * dim], dim);
            double err = std::fabs(dot - (i == j ? 1.0f : 0.0f));
            if (err > max_err) max_err = err;
        }
    }

    delete[] P;

    std::cout << "  Max orthogonality error: " << max_err << std::endl;

    if (max_err < 1e-5)
        std::cout << "  ✅ ||P*P^T - I|| < 1e-5" << std::endl;
    else
        std::cout << "  ⚠️ Erro: " << max_err << std::endl;
}

// ============================================================
// MAIN
// ============================================================

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "============================================" << std::endl;
    std::cout << "  MADHAVA-SEC C++ CORE BENCHMARK" << std::endl;
    std::cout << "============================================" << std::endl;

    #if defined(__AVX2__) && defined(__FMA__)
    std::cout << "  SIMD: AVX2+FMA" << std::endl;
    #else
    std::cout << "  SIMD: NONE" << std::endl;
    #endif
    std::cout << "  Threads: " << omp_get_max_threads() << std::endl;

    // Claim 1: Bound violations em 85D (tool vectors)
    test_bound_violations(85, 5000, 20);

    // Claim 1: Bound violations em 384D (embeddings)
    test_bound_violations(384, 2000, 10);

    // Claim 2: int8 quantization em 384D
    test_int8_quantization(384, 1000);

    // Claim 3: MGS orthogonality
    test_gs_orthogonality(384, 64);
    test_gs_orthogonality(384, 128);

    std::cout << "\n============================================" << std::endl;
    std::cout << "  FIM. BSL 1.1 | pay@winnex.ai" << std::endl;
    std::cout << "============================================" << std::endl;
    return 0;
}
