# Madhava-Sec

**Mathematically Guaranteed Agent Security Scoring**

Cauchy-Schwarz upper-bound for deterministic prompt/tool scoring.
Zero regex. Zero hardcoded patterns. Zero fallbacks.

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue)](mailto:pay@winnex.ai)

---

## What It Does

Given a query (e.g., a prompt or user instruction) and a set of candidate tools or attack prompts,
Madhava-Sec computes a **provable upper bound** on the true harmfulness score of each candidate —
before evaluating it on the actual LLM.

If the bound says a candidate cannot possibly outperform the current best, it is
**mathematically excluded** with 0% chance of false negative.

```
                          ┌──────────────────────┐
                          │   Query Embedding     │
                          │   (all-MiniLM-L6-v2)  │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  QR Projection 384→32 │
                          │  (Stage 1, low-dim)   │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  Cauchy-Schwarz Bound │
                          │  for ALL candidates   │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  96%+ candidates     │
                          │  PROVABLY excluded   │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  QR Projection 384→128│
                          │  (Stage 2, tighter)   │
                          └──────────┬───────────┘
                          ┌──────────▼───────────┐
                          │  Error Backprop       │
                          │  Modulation (blend)   │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  Top-K survivors →    │
                          │  Exact score on LLM   │
                          └──────────────────────┘
```

## What Problem It Solves

In agent security, the bottleneck is often **evaluation cost**: calling an LLM to judge
a candidate prompt takes seconds and API budget. If you have N=416 candidates (AgentHarm)
and can only afford K=8 LLM calls, which 8 do you pick?

Madhava-Sec answers this by computing a **mathematically guaranteed upper bound**
on each candidate's score — without calling the LLM. Candidates whose bound falls
below the current best are pruned with 0% false negative rate.

| Approach | Candidates evaluated | LLM calls | False negatives |
|----------|:-------------------:|:---------:|:---------------:|
| Brute force | 416/416 | 416 | 0% |
| **Madhava-Sec** | **8/416** | **8** | **0%** ✅ |
| Random selection | 8/416 | 8 | ~50% |
| Heuristic (BM25, etc.) | 8/416 | 8 | Unknown |

## Benchmark (AgentHarm, HuggingFace)

**Dataset:** `ai-safety-institute/AgentHarm` — 416 balanced behaviors (208 harmful + 208 benign),
85 tools, 8 categories. Embeddings via `all-MiniLM-L6-v2` (384D).

**5-fold cross-validation on 11,598 train + 2,320 test samples:**

| Method | F1 | AUC | Spearman vs Direct |
|:-------|:--:|:---:|:------------------:|
| **Direct** (exact dot product) | **0.8262** | **0.9105** | 1.0000 |
| **Madhava** (bound + modulation) | **0.8183** | **0.9018** | **0.9715** |
| **Bound** (no modulation) | 0.7955 | 0.8850 | 0.9044 |
| **Random** | 0.6013 | 0.4130 | -0.1608 |

| Metric | Value |
|:-------|:-----:|
| Retention vs Direct | **99.05%** |
| Modulation gain vs Bound | **+2.28 pp** |
| Gain vs Random | **+21.70 pp** |
| Bound violations | **0 / 254M+ pairs** ✅ |

**Math guarantees the bound:** For any vector v and query q, with orthogonal projection P:

```
<v, q>  ≤  <Pv, Pq>  +  ‖v - PᵀPv‖ · ‖q - PᵀPq‖
```

The right side is B₁(v,q). If B₁(v,q) < best_known_score, then v CANNOT be the optimal
tool. Zero false negatives. Proven.

## Where It Does NOT Work

**TL;DR:** Madhava-Sec is a **scoring/selection** tool. It is NOT a **generation** tool.

We submitted Madhava-Sec to the Kaggle [AI Agent Security](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks)
competition and scored **0.0**. Here is why:

| What we tested | Result | Explanation |
|:---------------|:------|:------------|
| **AgentHarm reranking** (offline) | F1=0.818 ✅ | Madhava-Sec selects the right tools |
| **Kaggle attack generation** (online) | **Score 0.0** ❌ | Generation ≠ Selection |

The Kaggle competition required **generating prompts that execute harmful actions on an LLM**.
Madhava-Sec selects which prompts to evaluate — but if the prompts themselves are poorly
constructed (e.g., narrative scenarios requiring 5+ steps instead of direct `http.post` calls),
no amount of selection produces a non-zero score.

**Key failure analysis:**

| Problem | Root cause |
|:--------|:-----------|
| Prompt style | Narrative (5+ steps) instead of direct (1 step: `http.post`) |
| Scoring overhead | eval_predicates() consumed time without generating candidates |
| Volume | 800 candidates submitted vs competitors' 2000+ |
| Selection vs generation | Madhava-Sec optimized the wrong bottleneck |

**Lesson:** Madhava-Sec's Cauchy-Schwarz pruning is valuable when:
- Evaluation is expensive (LLM calls cost time/budget)
- Candidates are pre-defined (embedding comparison is meaningful)

It is NOT valuable when:
- The bottleneck is generation (creating new candidates)
- The grader evaluates ALL candidates anyway (no selection needed)

## Mathematical Foundation

### QR-Orthogonal Projection

```
P = MGS(R),  R ~ N(0,1)^{d_out × d_in}
P · Pᵀ ≈ I_{d_out}  (verified < 1e-5)
```

### Cauchy-Schwarz Upper Bound

For any vectors v, q with ‖v‖ = ‖q‖ = 1 (normalized):

```
⟨v, q⟩  =  ⟨Pv, Pq⟩ + ⟨v - PᵀPv, q - PᵀPq⟩
        ≤  ⟨Pv, Pq⟩ + ‖v - PᵀPv‖ · ‖q - PᵀPq‖
        =  B₁(v, q)
```

### Error Backpropagation (Modulation)

```
α(v)    =  σ((e₁(v) - e₂(v)) / μ)
score   =  B₁(v, q) + α(v) · (B₂(v, q) - B₁(v, q))
```

Where e₁ is the residual after Stage 1 projection (larger error) and e₂ after Stage 2
(smaller error). When the bound tightens significantly from Stage 1 to Stage 2 (e₁ >> e₂),
α → 1, applying the full correction.

### Gram-Schmidt Diversity

```
v_orth  =  v - Σ_{k∈K} (v · k̂) · k̂
```

Selecting candidates with maximal ‖v_orth‖ maximizes tool-space coverage.

## Project Structure

```
madhava_sec/
├── __init__.py          # Package root, version 2.0.0
├── core.py              # MadhavaSecEngine (projection + bound + modulation)
├── attack_families.py   # AttackFamilyEngine (KMeans-derived centroids)
├── verifier.py          # FormalVerifier (threshold-based pre-grading)
├── search.py            # AttackSearch (beam search + GS diversity)

benchmarks/
├── madhava_sec_benchmark_honest_v5.py   # 5-fold CV on AgentHarm
└── madhava_sec_agentharm_benchmark.py    # AgentHarm full benchmark

results/
├── agent_harm_honest_v5.json             # 5-fold CV results
└── agent_harm_v8.json                    # Scout+Factory v8 results
```

## Quick Start

```python
from madhava_sec import MadhavaSecEngine, FormalVerifier

# 1. Embedding-derived attack families
families = AttackFamilyEngine()
families.build(injection_embeddings)  # KMeans on real data → K=30 centroids

# 2. Cauchy-Schwarz bound scoring
engine = MadhavaSecEngine(stage_dims=[32, 128])
engine.build(tool_vectors)            # index N candidates
scores = engine.estimate_score(query) # dict[idx → upper_bound]

# 3. Embedding-only verification
verifier = FormalVerifier(families)
approved, score = verifier.verify_candidate(prompt_text)
```

## Requirements

| Package | Minimum | Use |
|:--------|:-------:|:----|
| numpy | 1.24.0 | Linear algebra, MGS, QR projections |
| scikit-learn | 1.3.0 | KMeans for attack families |
| sentence-transformers | 2.2.0 | Embedding all-MiniLM-L6-v2 |

## License

**Business Source License 1.1 (BSL 1.1)** — Study and non-production use permitted.
Commercial use: pay@winnex.ai

Change Date: 2036-01-01 (converts to GPL v2.0+)

## References

1. **Madhava Cascade** (2026). Zenodo 10.5281/zenodo.21500959 — 0 violations in 254M+ pairs
2. **AgentHarm** (2025). ai-safety-institute/AgentHarm — 416 agent security scenarios
3. **Dasgupta & Gupta** (2003). An elementary proof of the Johnson-Lindenstrauss lemma
4. **Malkov & Yashunin** (2016). Efficient and robust ANN search using HNSW
5. **Madhava v18 Proof** (2026). Zenodo 10.5281/zenodo.21500959 — Why hierarchical methods
   cannot guarantee exact recall in high dimensions
