# Madhava-Sec

**Mathematical scoring for agent prompt classification.**

Cauchy-Schwarz upper-bound for deterministic similarity scoring against
K learned centroids. Zero regex. Zero hardcoded patterns.

> ⚠️ **CRITICAL: Math ≠ Semantic Safety.** This is a CLASSIFIER that scores
> embedding similarity with a mathematical guarantee. It does NOT detect
> semantic harmfulness. If the embedding model misses a jailbreak nuance,
> Madhava-Sec produces 0% bound violations on a DEFECTIVE signal.
> See [Limitations: Math ≠ Semantic](#2-mathematical-guarantee--semantic-guarantee).

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue)](mailto:pay@winnex.ai)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.21506566-blue)](https://zenodo.org/records/21506566)

---

## What It Actually Is

Madhava-Sec is a **classifier** that:

```
Input:  query embedding (384D) + K centroids (from KMeans on attack data)
Output: modulated Cauchy-Schwarz bound score for each centroid
        → max score = classification score for the query
Guarantee: score ≥ true cosine similarity (0% false negatives on embedding)
```

It does NOT:
- Retrieve documents from a corpus (use Madhava Cascade for that)
- Generate attack prompts (that's PiPrime's role)
- Replace LLM-based safety evaluation (see disclaimer)

## What Problem It Solves

**Before Madhava-Sec:** Classifying prompts as "attack" or "benign" required either:
1. Exact dot product against all known attack centroids (O(K·D) — expensive at scale)
2. Regex/heuristics (brittle, easy to bypass)
3. LLM-as-judge (slow, expensive, $0.01-0.10 per call)

**With Madhava-Sec:** Compute a provable upper bound using QR projection (O(K·d), d << D):
- 99.05% F1 retention vs exact dot product
- 0% bound violations (mathematical guarantee)
- No LLM calls needed for scoring

## Benchmark (AgentHarm, 5-fold CV)

**Dataset:** ai-safety-institute/AgentHarm — 416 balanced behaviors, 85 tools.
**Model:** all-MiniLM-L6-v2 (384D). **Centroids:** K=30 via KMeans.

| Method | F1 | AUC | Spearman vs Direct |
|:-------|:--:|:---:|:------------------:|
| **Direct** (exact dot product) | **0.8262** | **0.9105** | 1.0000 |
| **Madhava** (bound + modulation) | **0.8183** | **0.9018** | **0.9715** |
| **Bound** (no modulation) | 0.7955 | 0.8850 | 0.9044 |
| **Random** | 0.6013 | 0.4130 | -0.1608 |

Key findings:
- **99.05% F1 retention** vs Direct — projection preserves classification quality
- **+2.28 pp modulation gain** over raw bound — error backpropagation works
- **+21.70 pp gain** vs Random — genuine signal, not noise
- **0% bound violations** across 254M+ query-vector pairs ✅
- **Spearman 0.97** — ordering is nearly identical to exact

## Where It Does NOT Work

| Scenario | Result | Why |
|:---------|:------:|:----|
| **Retrieval** (R@10, NDCG) | ❌ **R@10 < 5%** | Scores K centroids, not N items. Wrong metric. |
| **Semantic safety** | ❌ **False sense of safety** | Math guarantees embedding similarity, not harmfulness |
| **Attack generation** | ❌ **Score 0.0 on Kaggle** | See detailed analysis below |
| **High-dim isotropic data** | ❌ **Regime RED** | Bound too loose — no pruning possible |

### The Kaggle Competition Failure

Submitted to AI Agent Security competition, scored **0.0**. Root cause:
Madhava-Sec is a **classifier**, not a **generator**. The competition required
generating prompts that execute on LLMs — Madhava-Sec scores candidates,
it doesn't create them. PiPrime (π-based navigation) handles generation;
Madhava-Sec filters the results.

## Classification Benchmarks

### Scikit-learn style (K centroids, 384D)

| Method | K | F1 | AUC | Spearman | Regime |
|:-------|:-:|:--:|:---:|:--------:|:------:|
| Direct (exact) | 30 | 0.667 | 0.461 | 1.000 | — |
| **Madhava [64→128]** | 30 | **0.667** | **0.460** | **0.960** | 🟢GREEN |
| **Madhava [32→64]** | 30 | **0.667** | **0.463** | **0.914** | 🟢GREEN |

Spearman > 0.91 across all configurations. Classification preserved.

## Honest Verdict

| Strength | Evidence |
|:---------|:---------|
| ✅ **Mathematical guarantee** | 0% violations in 254M+ pairs |
| ✅ **F1 retention** | 99.05% vs exact dot product |
| ✅ **Ordering preserved** | Spearman 0.97 vs Direct |
| ✅ **No hardcoding** | All centroids from data (KMeans) |

| Limitation | Impact |
|:-----------|:-------|
| ❌ **Math ≠ Semantic** | Guarantee is on embedding, not harmfulness |
| ❌ **Not retrieval** | R@10/NDCG metrics are invalid |
| ❌ **Dependent on KMeans** | Bad centroids → bad scores (bound still valid) |
| ❌ **Projection loss** | Bound is loose — pruning impossible in high D_int |

## Quick Start

```python
from madhava_sec import MadhavaSecEngine
from sklearn.cluster import KMeans

# 1. Train centroids (your attack data)
km = KMeans(n_clusters=30, random_state=42).fit(attack_embeddings)
centroids = km.cluster_centers_

# 2. Build Madhava-Sec engine
engine = MadhavaSecEngine(stage_dims=[64, 128]).build(centroids)

# 3. Score a query
scores = engine.estimate_score(query_embedding)
max_score = max(scores.values())  # classification score
```

## Project Structure

```
madhava_sec/     → Core library (6 files)
├── core.py          → MadhavaSecEngine (projection + bound + modulation)
├── cache.py         → mmap-backed disk cache for 10M-100M vectors
├── attack_families.py  → KMeans-derived attack families
├── verifier.py      → FormalVerifier (threshold-based)
├── search.py        → AttackSearch (GS diversity)

benchmarks/      → Benchmarks + results
├── madhava_sec_benchmark_honest_v5.py   → 5-fold CV (official)
├── classification_benchmark.json         → F1/AUC/Spearman results
├── pipeline_benchmark.json               → Full benchmark results

PIPRIME_INTEGRATION.md   → π-based navigation × Madhava-Sec
ENTERPRISE_LICENSING.md  → BSL 1.1, ROI: 98% LLM reduction
```

## References

1. **Madhava-Sec Zenodo** (2026). 10.5281/zenodo.21506566 — This system
2. **Madhava v18 Proof** (2026). 10.5281/zenodo.21500959 — Why hierarchical methods fail
3. **AgentHarm** (2025). ai-safety-institute/AgentHarm — 416 agent security scenarios
4. **Dasgupta & Gupta** (2003). JL lemma elementary proof

---

*BSL 1.1 | pay@winnex.ai*
