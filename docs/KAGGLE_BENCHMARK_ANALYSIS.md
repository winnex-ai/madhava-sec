# Madhava-Sec — Deep Analysis, Kaggle Benchmark, and the C++ Engine

> Technical document accompanying the benchmark with a specialized Kaggle
> dataset and the native C++ engine. Complements the README with the
> engineering rationale.

## 1. The real bottleneck: dependence on attack-centroid quality

Madhava-Sec is a **similarity scorer with a mathematical bound**, not an
end-to-end trained semantic classifier. The architecture is:

```
training attacks → embedding → KMeans → centroids C
query q → embedding → score = max_c ⟨q, c⟩   (with Cauchy-Schwarz bound)
```

The mathematical guarantee (0 bound violations) is **unconditional** — it
holds for any `q`, any `C`. But the **practical value** of the score depends
entirely on the quality of `C`:

> **If the training set is poor or outdated, the bound only proves proximity
> to a bad set. The mathematical guarantee does not save bad training data.**

Honest consequences:

1. **GIGO (garbage in, garbage out).** The score is only as good as the
   attack examples used to train the centroids. A training set with stale
   patterns does not cover new jailbreaks.
2. **The bound guarantees ordering, not semantics.** If the embedding does
   not capture the malicious pattern (e.g., a jailbreak wrapped in
   `base64`), the score can be low **with zero bound violations** — the
   bound is correct, the meaning is wrong.
3. **Structural mitigation:** large, diverse, **specialized** training sets
   (like the Kaggle dataset used here) reduce the GIGO risk; the
   `SafetyEnsemble` (multi-embedder) reduces single-embedder blind spots.

This is why the README frames Madhava-Sec as **one layer** of a security
pipeline, not a standalone safety system.

## 2. Modest F1/AUC: honest positioning

Across several datasets, F1 lands between ~0.67 and ~0.89. This is not an
accident — it is the expected regime for the technique:

| Property | Madhava-Sec | Elite classifier (LLM judge / tuned) |
|:---------|:-----------:|:-------------------------------------:|
| Cost per query | ~µs (C++ SIMD) | $0.01–0.10 |
| Latency | ~1–5 ms | ~2 s |
| Mathematical guarantee | **Yes (0 violations)** | No |
| Semantic accuracy | Medium (cosine similarity) | High |
| Explainability | Mathematical bound per candidate | Heuristic |

Madhava-Sec does not compete with an elite classifier on semantic accuracy.
It competes on the **cost × guarantee** axis. The correct usage is
**provable pruning**: filter out provably-safe candidates and escalate only
the survivors to an LLM judge.

## 3. The C++ engine is the core

`madhava_core.h` implements the entire math in native C++:

| Component | Detail |
|:----------|:-------|
| Projection | MGS (Modified Gram-Schmidt), orthogonality ‖P·Pᵀ−I‖ < 1e-5 |
| Quantization | int8 with per-column verified scale (4× compression) |
| Bound | Cauchy-Schwarz: `⟨q,c⟩ ≤ ⟨Pq,Pc⟩ + ‖q_⊥‖·‖c_⊥‖` |
| Pruning | QuickSelect O(N) + early exit |
| SIMD | AVX2+FMA on dot products |
| Parallelism | OpenMP (build and score) |
| C ABI | `libmadhava_sec.so` — consumable from Python (ctypes), Rust, Go, C |

### Native scoring

`score_all()` walks **all** centroids with the modulated bound
(`B1 + α·(B2−B1)`), the same scheme as the Python implementation, but with
int8-quantized projections and SIMD. The benchmark compares:

- **MADHAVA** (Python, float32 QR): the honest v5
- **MADHAVA_NATIVE** (C++ int8 + SIMD): the real engine

### Native engine results (C++)

```
SIMD: AVX2+FMA | Threads: 28
Bound violations: 0/4000 (D=85) and 0/2000 (D=384)
int8 cosine: 0.999991 (MSE ~0)
MGS orthogonality: ‖P·Pᵀ−I‖ < 1e-5
```

## 4. Kaggle benchmark (this work)

Dataset: **`krishnayadav456wrsty/prompt-injection-and-jailbreak-detection-dataset`**
— specialized in prompt injection + jailbreak, 20k texts (917 injection,
19,083 benign, ~4.5% attack — a realistic imbalance), with categories
(Jailbreak, Role-Playing, Instruction Override, Multilingual, Obfuscation…)
and hard negatives.

Protocol (identical to honest v5):
- 5-fold stratified CV
- per-method independent threshold, optimized **on train** (F1), evaluated **on test**
- same KMeans centroids for all methods (only the score differs)

Reported metrics: **F1, AUC, precision, recall, specificity, MCC**, Spearman
vs exact dot product, bound violations, latency.

## 5. Verdict

- The mathematical bound is real and portable to C++ (0 violations).
- F1/AUC vary with the dataset — **that is the signal, not noise**.
- Use Madhava-Sec for **cheap provable pruning**; use an LLM judge for
  final high-stakes decisions.
- The training data behind the centroids is the most important lever.
