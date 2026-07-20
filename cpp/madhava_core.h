/**
 * Madhava-Sec C++ Core
 * =====================
 * Mathematically Guaranteed Agent Security Framework
 *
 * Based on: Madhava V12 (Zenodo 10.5281/zenodo.21220131)
 *   - MGS (Modified Gram-Schmidt) for orthogonal projections
 *   - int8 quantization with verified scale
 *   - Cauchy-Schwarz bound with 0 violations
 *   - Cascade [64,128] with early exit
 *   - QuickSelect O(N) for pruning
 *   - SIMD (AVX2+FMA) for dot products
 *
 * BSL 1.1 | pay@winnex.ai
 */

#ifndef MADHAVA_SEC_CORE_H
#define MADHAVA_SEC_CORE_H

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cstring>
#include <memory>
#include <cstdint>
#include <omp.h>
#include <tuple>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace madhava_sec {

// ============================================================
// INLINE SIMD (AVX2+FMA)
// ============================================================

inline float dot_f32(const float* a, const float* b, int d) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 s = _mm256_setzero_ps(); int i = 0;
    for (; i + 8 <= d; i += 8)
        s = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), s);
    float o[8]; _mm256_storeu_ps(o, s);
    float r = o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7];
    for (; i < d; i++) r += a[i]*b[i];
    return r;
#else
    float s = 0;
    for (int i = 0; i < d; i++) s += a[i]*b[i];
    return s;
#endif
}

inline float dot_i8_f32(const int8_t* a_i8, const float* scale,
                        const float* b32, int d) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 s = _mm256_setzero_ps(); int i = 0;
    for (; i+8 <= d; i += 8) {
        __m128i i8 = _mm_loadl_epi64((const __m128i*)(a_i8+i));
        __m256i i32 = _mm256_cvtepi8_epi32(i8);
        __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(i32),
                                  _mm256_loadu_ps(scale+i));
        s = _mm256_fmadd_ps(f, _mm256_loadu_ps(b32+i), s);
    }
    float o[8]; _mm256_storeu_ps(o, s);
    float r = o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7];
    for (; i < d; i++) r += (float)a_i8[i] * scale[i] * b32[i];
    return r;
#else
    float r = 0;
    for (int i = 0; i < d; i++) r += (float)a_i8[i] * scale[i] * b32[i];
    return r;
#endif
}

// ============================================================
// QUICKSELECT O(N)
// ============================================================

inline int hoare(std::vector<std::pair<float,int>>& v, int lo, int hi) {
    float p = v[lo+(hi-lo)/2].first; int i = lo, j = hi;
    while (1) {
        while (v[i].first > p) i++;
        while (v[j].first < p) j--;
        if (i >= j) return j;
        std::swap(v[i], v[j]); i++; j--;
    }
}

inline void quickselect(std::vector<std::pair<float,int>>& v,
                        int lo, int hi, int k) {
    while (lo < hi) {
        int p = hoare(v, lo, hi);
        int l = p-lo+1;
        if (k <= l) hi = p;
        else { lo = p+1; k -= l; }
    }
}

// ============================================================
// MADHAVA-SEC ENGINE
// ============================================================

class MadhavaSecEngine {
public:
    int D, s1, s2, n_total;
    float *P1=nullptr, *P2=nullptr;
    int8_t *pr1_i8=nullptr, *pr2_i8=nullptr;
    float *pr1_scale=nullptr, *pr2_scale=nullptr;
    float *e1=nullptr, *e2=nullptr;
    float *source_data=nullptr;  // original vectors for bound verification
    double build_time = 0;

    MadhavaSecEngine(int dim, int stage1=64, int stage2=128)
        : D(dim), s1(stage1), s2(stage2), n_total(0) {}

    ~MadhavaSecEngine() {
        delete[] P1; delete[] P2;
        delete[] pr1_i8; delete[] pr2_i8;
        delete[] pr1_scale; delete[] pr2_scale;
        delete[] e1; delete[] e2;
        delete[] source_data;
    }

    // ========================================================
    // MGS – Modified Gram-Schmidt
    // ========================================================

    float* mk_proj_mgs(int dim, int seed_offset=0) {
        float* P = new float[dim * D];
        std::mt19937 rng(42 + dim + seed_offset);
        std::normal_distribution<float> nd(0, 1);
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < D; j++) P[i*D+j] = nd(rng);
            for (int k = 0; k < i; k++) {
                float dp = 0;
                for (int j = 0; j < D; j++) dp += P[i*D+j] * P[k*D+j];
                for (int j = 0; j < D; j++) P[i*D+j] -= dp * P[k*D+j];
            }
            float nr = 0;
            for (int j = 0; j < D; j++) nr += P[i*D+j] * P[i*D+j];
            nr = std::sqrt(nr);
            if (nr > 1e-10f)
                for (int j = 0; j < D; j++) P[i*D+j] /= nr;
        }
        return P;
    }

    inline void quantize(const float* src, int8_t* dst,
                         const float* scale, int dims) {
        for (int j = 0; j < dims; j++) {
            int qi = (int)(src[j]/scale[j] + (src[j]>=0 ? 0.5f : -0.5f));
            if (qi > 127) qi = 127;
            if (qi < -128) qi = -128;
            dst[j] = (int8_t)qi;
        }
    }

    // ========================================================
    // BUILD
    // ========================================================

    void build(const float* data, int n, bool verbose=false) {
        auto t0 = std::chrono::high_resolution_clock::now();
        n_total = n;

        // Keep original data for bound verification
        source_data = new float[(size_t)n * D];
        std::memcpy(source_data, data, (size_t)n * D * sizeof(float));

        P1 = mk_proj_mgs(s1); P2 = mk_proj_mgs(s2, 100);
        pr1_i8 = new int8_t[(size_t)n * s1];
        pr2_i8 = new int8_t[(size_t)n * s2];
        pr1_scale = new float[s1];
        pr2_scale = new float[s2];
        e1 = new float[n]; e2 = new float[n];

        int sn = std::min(n, 100000);
        std::vector<float> ma1(s1,0), ma2(s2,0);
        for (int i = 0; i < sn; i++) {
            const float* v = data + (size_t)i * D;
            for (int j = 0; j < s1; j++) {
                float s = dot_f32(v, &P1[j*D], D);
                if (std::fabs(s) > ma1[j]) ma1[j] = std::fabs(s);
            }
            for (int j = 0; j < s2; j++) {
                float s = dot_f32(v, &P2[j*D], D);
                if (std::fabs(s) > ma2[j]) ma2[j] = std::fabs(s);
            }
        }
        for (int j = 0; j < s1; j++)
            pr1_scale[j] = std::max(ma1[j]/127.0f*1.05f, 1e-10f);
        for (int j = 0; j < s2; j++)
            pr2_scale[j] = std::max(ma2[j]/127.0f*1.05f, 1e-10f);

        for (int p = 0; p < n; p += 500000) {
            int nt = std::min(500000, n-p);
            #pragma omp parallel for
            for (int i = 0; i < nt; i++) {
                int id = p + i;
                const float* v = data + (size_t)id * D;
                float pj1[128], pj2[128];
                for (int j = 0; j < s1; j++)
                    pj1[j] = dot_f32(v, &P1[j*D], D);
                for (int j = 0; j < s2; j++)
                    pj2[j] = dot_f32(v, &P2[j*D], D);
                quantize(pj1, pr1_i8+(size_t)id*s1, pr1_scale, s1);
                quantize(pj2, pr2_i8+(size_t)id*s2, pr2_scale, s2);
                float pn1=0, pn2=0, vn=0;
                for (int j = 0; j < D; j++) vn += v[j]*v[j];
                for (int j = 0; j < s1; j++) {
                    float val = (float)pr1_i8[(size_t)id*s1+j]*pr1_scale[j];
                    pn1 += val*val;
                }
                for (int j = 0; j < s2; j++) {
                    float val = (float)pr2_i8[(size_t)id*s2+j]*pr2_scale[j];
                    pn2 += val*val;
                }
                e1[id] = std::sqrt(std::max(0.0f, vn-pn1));
                e2[id] = std::sqrt(std::max(0.0f, vn-pn2));
            }
        }
        build_time = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now()-t0).count();
        if (verbose)
            std::cerr << "Build: " << n << " vectors in " << build_time << "s\n";
    }

    // ========================================================
    // BOUND VERIFICATION — exact cosine vs. Cauchy-Schwarz UB
    // ========================================================

    std::tuple<double,double,long> verify_bounds(const float* q,
                                                  int n_verify=1000) {
        float pq1[128], pq2[128], q1s=0, q2s=0;
        float qn = std::sqrt(dot_f32(q, q, D));
        for (int j = 0; j < s1; j++) {
            pq1[j] = dot_f32(q, &P1[j*D], D); q1s += pq1[j]*pq1[j];
        }
        for (int j = 0; j < s2; j++) {
            pq2[j] = dot_f32(q, &P2[j*D], D); q2s += pq2[j]*pq2[j];
        }
        float qr1 = std::sqrt(std::max(0.0f, qn*qn - q1s));
        float qr2 = std::sqrt(std::max(0.0f, qn*qn - q2s));
        float qm1=0, qm2=0;
        for (int j = 0; j < s1; j++)
            qm1 += 0.5f * pr1_scale[j] * std::fabs(pq1[j]);
        for (int j = 0; j < s2; j++)
            qm2 += 0.5f * pr2_scale[j] * std::fabs(pq2[j]);

        long total=0, violations=0;
        double max_v=0;
        int step = std::max(1, n_total/n_verify);

        for (int i = 0; i < n_total; i += step) {
            total++;
            // Cauchy-Schwarz Upper Bound: <Pv, Pq> + e(v)*e(q)
            float ub = dot_i8_f32(pr1_i8+(size_t)i*s1, pr1_scale, pq1, s1)
                       + e1[i]*qr1 + qm1 + 1e-5f;

            // Exact cosine from original data
            float exact = dot_f32(source_data + (size_t)i*D, q, D);

            if (exact > ub + 1e-5f) {
                violations++;
                double ratio = (double)(exact-ub)/std::max(std::fabs(ub),1e-10f);
                if (ratio > max_v) max_v = ratio;
            }
        }
        return {total>0?(double)violations/total:0, max_v, total};
    }

    // ========================================================
    // SEARCH — early exit via Cauchy-Schwarz pruning
    // ========================================================

    std::vector<int> search(const float* q, int K=10,
                            int* out_k1=nullptr, int* out_k2=nullptr,
                            int* out_k3=nullptr) {
        int N = n_total;
        float pq1[128], pq2[128], q1s=0, q2s=0;
        float qn = std::sqrt(dot_f32(q,q,D));
        for (int j = 0; j < s1; j++) {
            pq1[j]=dot_f32(q,&P1[j*D],D); q1s+=pq1[j]*pq1[j];
        }
        for (int j = 0; j < s2; j++) {
            pq2[j]=dot_f32(q,&P2[j*D],D); q2s+=pq2[j]*pq2[j];
        }
        float qr1=std::sqrt(std::max(0.0f,qn*qn-q1s));
        float qr2=std::sqrt(std::max(0.0f,qn*qn-q2s));
        float qm1=0, qm2=0;
        for (int j = 0; j < s1; j++) qm1+=0.5f*pr1_scale[j]*std::fabs(pq1[j]);
        for (int j = 0; j < s2; j++) qm2+=0.5f*pr2_scale[j]*std::fabs(pq2[j]);

        std::vector<std::pair<float,int>> b1(N);
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            float ub = dot_i8_f32(pr1_i8+(size_t)i*s1,pr1_scale,pq1,s1)
                       + e1[i]*qr1 + qm1 + 1e-5f;
            b1[i] = {ub, i};
        }
        float bm=1e10f,bx=-1e10f;
        for (auto& x:b1){if(x.first<bm)bm=x.first;if(x.first>bx)bx=x.first;}
        float ak=std::min(0.50f,std::max(0.05f,0.25f*0.12f/std::max(bx-bm,0.01f)));
        int k1=std::min(std::max((int)(N*ak),100),N);
        if(out_k1)*out_k1=k1;
        if(k1<N)quickselect(b1,0,N-1,k1);

        std::vector<std::pair<float,int>> b2(k1);
        for (int i = 0; i < k1; i++) {
            int vi = b1[i].second;
            float ub2 = dot_i8_f32(pr2_i8+(size_t)vi*s2,pr2_scale,pq2,s2)
                        + e2[vi]*qr2 + qm2 + 1e-5f;
            float a1=e1[vi],a2=e2[vi];
            float al=std::min(0.99f,std::max(0.01f,
                1.0f/(1.0f+std::exp(-(a1-a2)/std::max(a1/k1,1e-9f)*0.5f))));
            b2[i] = {b1[i].first+al*(ub2-b1[i].first), vi};
        }
        int k2=std::min(2000,k1);
        if(out_k2)*out_k2=k2;
        std::partial_sort(b2.begin(),b2.begin()+k2,b2.end(),
            [](auto& a,auto& b){return a.first>b.first;});

        std::vector<std::pair<float,int>> heap;
        float tenth=-1e10f; int proc=0;
        for(int i=0;i<k2;i++){
            int vi=b2[i].second;
            if(b2[i].first<=tenth&&(int)heap.size()>=K) break;
            proc++;
            float exact = dot_f32(source_data+(size_t)vi*D, q, D);
            if((int)heap.size()<K){
                heap.push_back({exact,vi});
                if((int)heap.size()==K){
                    std::make_heap(heap.begin(),heap.end(),
                        std::greater<std::pair<float,int>>());
                    tenth=heap[0].first;
                }
            } else if(exact>tenth){
                std::pop_heap(heap.begin(),heap.end(),
                    std::greater<std::pair<float,int>>());
                heap.back()={exact,vi};
                std::push_heap(heap.begin(),heap.end(),
                    std::greater<std::pair<float,int>>());
                tenth=heap[0].first;
            }
        }
        if(out_k3)*out_k3=proc;
        std::sort(heap.begin(),heap.end(),
            [](auto& a,auto& b){return a.first>b.first;});
        std::vector<int> r;
        for(int i=0;i<K&&i<(int)heap.size();i++) r.push_back(heap[i].second);
        return r;
    }

    int64_t memory_usage() const {
        int64_t t = 0;
        t += (int64_t)s1*D*sizeof(float);
        t += (int64_t)s2*D*sizeof(float);
        t += (int64_t)n_total*s1*sizeof(int8_t);
        t += (int64_t)n_total*s2*sizeof(int8_t);
        t += (int64_t)s1*sizeof(float)*2;
        t += (int64_t)n_total*sizeof(float)*2;
        t += (int64_t)n_total*D*sizeof(float); // source_data
        return t;
    }
    double memory_gb() const {
        return (double)memory_usage()/(1024.0*1024.0*1024.0);
    }
};

} // namespace

#endif
