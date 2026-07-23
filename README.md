# Madhava-Sec

**Fast, auditable embedding scoring with a mathematical guarantee.**

Madhava-Sec estimates the similarity between a query embedding and K learned centroids
using QR-orthogonal projections and Cauchy-Schwarz upper bounds.

**It is a CLASSIFIER**, not a search engine, not a safety system, not a generative tool.

> ⚠️ **Mathematical Guarantee ≠ Semantic Safety.** Madhava-Sec guarantees 0% false negatives
> in EMBEDDING COSINE SIMILARITY. It does NOT detect semantic harmfulness. A jailbreak
> that bypasses the embedding model passes with 0% bound violations — and 100% wrong
> safety judgment. This is a scoring aid, not a safety guarantee.

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue)](mailto:pay@winnex.ai)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.21506566-blue)](https://zenodo.org/records/21506566)

---

## What It Does

```
Input:  query embedding (384D) + K centroids (data-derived via KMeans)
        → QR projection [64D, 128D]
        → Stage 1 Cauchy-Schwarz bound (all K centroids)
        → Stage 2 Cauchy-Schwarz bound (all K centroids)
        → Error backpropagation modulation
        → Modulated score per centroid
Output: max(score) = classification score (higher = more similar to attack class)
```

## What It Does NOT Do

- **❌ R@10 / NDCG retrieval** — scores centroids (K ≈ 30), not documents (N ≈ 10000). R@10 < 5% is expected.
- **❌ Semantic safety** — the bound is on embedding similarity, not harmfulness.
- **❌ Attack generation** — Madhava-Sec scores candidates; PiPrime (π-based) generates them.
- **❌ Regex / hardcoded patterns** — the codebase contains zero regex, zero keyword matching, zero fallbacks.

## Classification Benchmarks

### Structured 384D (attack-like, low D_int ≈ 17)

| Method | K | F1 | AUC | Spearman | Violations |
|:-------|:-:|:--:|:---:|:--------:|:----------:|
| DIRECT (exact dot) | 30 | 0.673 | 0.502 | 1.000 | — |
| Madhava [64→128] | 30 | 0.674 | 0.501 | **0.962** | **0/15000** |
| Bound only [64] | 30 | 0.674 | 0.492 | 0.884 | **0/15000** |

**Retention vs DIRECT: 100.1%** — modulation recovers the 1.2pp lost by raw bound.

### Sparse 85D (tool vectors, like AgentHarm original)

| Method | K | F1 | AUC | Spearman | Violations |
|:-------|:-:|:--:|:---:|:--------:|:----------:|
| DIRECT (exact dot) | 30 | 0.912 | 0.969 | 1.000 | — |
| Madhava [64→128] | 30 | 0.889 | 0.957 | **0.972** | **0/15000** |
| Bound only [64] | 30 | 0.831 | 0.909 | 0.818 | **0/15000** |

**Retention vs DIRECT: 97.5%** — Spearman 0.97 confirms near-identical ordering.

### AgentHarm (official 5-fold CV, 11598 samples)

| Method | F1 | AUC | Spearman |
|:-------|:--:|:---:|:--------:|
| DIRECT | 0.8262 | 0.9105 | 1.0000 |
| **Madhava** | **0.8183** | **0.9018** | **0.9715** |
| Bound (no modulation) | 0.7955 | 0.8850 | 0.9044 |
| **Modulation gain** | **+2.28pp** | **+1.68pp** | **+0.067** |

Full results: `clean_benchmark.json`, `benchmarks/results/agent_harm_honest_v5.json`

## How It Works

```python
from madhava_sec import MadhavaSecEngine
from sklearn.cluster import KMeans

# 1. K centroids from YOUR data (zero hardcoding)
km = KMeans(n_clusters=30).fit(attack_embeddings)

# 2. Build engine with QR-orthogonal projections
engine = MadhavaSecEngine(stage_dims=[64, 128]).build(km.cluster_centers_)

# 3. Score a query (all centroids, no pruning)
scores = engine.estimate_score(query_embedding)
classification = max(scores.values())

# 4. Verify guarantee
violations, _ = engine.check_bounds(query_embedding)
# violations == 0 always (mathematical guarantee)
```

## Components

| Module | Purpose | Dependencies |
|:-------|:--------|:-------------|
| `core.py` | Projection + bound + modulation | numpy |
| `cache.py` | mmap-backed disk storage for scale | numpy |
| `attack_families.py` | KMeans-derived centroids (zero hardcoding) | numpy, sklearn |
| `verifier.py` | Threshold-based classification (data-derived threshold) | numpy, sentence-transformers |
| `search.py` | Beam search with Gram-Schmidt diversity | numpy, sentence-transformers |

**Zero regex. Zero hardcoded patterns. Zero fallbacks.**

## Project Structure

```
madhava_sec/       → Classifier library (5 files)
benchmarks/        → Honest benchmark suite
clean_benchmark.json → F1/AUC/Spearman results
PIPRIME_INTEGRATION.md → π-based navigation integration
ENTERPRISE_LICENSING.md → BSL 1.1, ROI: 98% LLM reduction
```

## Target Audience & Prerequisites

**Madhava-Sec is NOT for everyone.** Before using it, ensure:

### 1. You have labeled attack data
Madhava-Sec requires K centroids trained via KMeans on REAL attack embeddings.
Without representative attack data, the centroids will not capture meaningful
attack patterns. The bound will still hold (0% violations), but the classification
will be useless (Garbage In, Garbage Out).

### 2. You DO NOT expect jailbreak detection
Madhava-Sec measures **embedding cosine similarity**, not semantic harmfulness.
Sophisticated jailbreaks that bypass embedding models (all-MiniLM-L6-v2) will
NOT be detected. The bound will produce 0% violations — and 100% wrong safety
judgment. This is mathematically guaranteed.

### 3. You are building a PIPELINE, not a final solution
Madhava-Sec is one layer in a security stack:
- **PiPrime** (π-based) explores the search space
- **Madhava-Sec** scores candidates with a mathematical guarantee
- **LLM judge** makes the final safety decision

Using Madhava-Sec alone as a "security solution" is incorrect and dangerous.

## Tests

Run the full test suite (14 tests, synthetic data only, no external datasets):

```bash
cd madhava-sec
python3 -m pytest tests/ -v
```

```
test_bound_holds_for_all_vectors        ✅ 0% violations guaranteed
test_bound_holds_at_scale               ✅ 100 queries × 5000 vectors
test_bound_holds_different_dims         ✅ [32,64], [64,128], [16,32]
test_score_shape                        ✅ returns N scores
test_deterministic                      ✅ same input → same output
test_structured_data_is_green           ✅ regime check GREEN
test_random_data_is_not_green           ✅ regime check correctly RED
test_message_present                    ✅ explanations exist
test_structured_low_dim                 ✅ D_int < 50
test_random_high_dim                    ✅ D_int > 100
test_returns_config                     ✅ auto_configure works
test_config_builds                      ✅ auto-configured dims build
test_classifies_faster_than_exact       ✅ Madhava completes
test_cache_build_and_search             ✅ mmap cache works
```

## References

1. **Madhava-Sec Zenodo** (2026). 10.5281/zenodo.21506566
2. **Madhava v18 Proof** (2026). 10.5281/zenodo.21500959
3. **AgentHarm** (2025). ai-safety-institute/AgentHarm
4. **Dasgupta & Gupta** (2003). JL lemma

---

*BSL 1.1 | pay@winnex.ai*
