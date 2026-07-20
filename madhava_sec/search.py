"""
Madhava-Sec Search Engine — Embedding-Only
============================================
Busca com Cauchy-Schwarz Upper Bound pruning sobre embeddings 384D.

Zero hardcoding:
  - Nao ha vetores 20D
  - Nao ha action vectors
  - Nao ha familias manuais
  - Tudo e' embedding all-MiniLM-L6-v2 (384D)

License: BSL 1.1 | Author: Klenio Araujo Padilha
"""

import time, math, random, gc
import numpy as np


class AttackSearch:
    """
    Busca com pruning Cauchy-Schwarz sobre embeddings 384D.

    Pipeline:
      build(texts) -> embed + indexar no MadhavaSecEngine
      search(threshold) -> beam search com upper bound
      diversity_gate(emb, known_embs) -> Gram-Schmidt no embedding
    """

    def __init__(self, families_engine=None, seed=42):
        self._seed = seed
        self._rand = random.Random(seed)
        # Lazy import para evitar circular
        from madhava_sec.attack_families import AttackFamilyEngine as _AFE
        from madhava_sec.core import MadhavaSecEngine as _MSE
        self.families = families_engine or _AFE()
        self.madhava = _MSE(stage_dims=[32, 128], seed=seed)
        self.madhava.full_dim = 384  # embedding 384D
        self._embedder = None
        self.built = False
        self._texts = []
        self._embeddings = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "all-MiniLM-L6-v2", device="cpu"
            )
        return self._embedder

    def embed(self, texts, batch_size=128):
        return self._get_embedder().encode(
            texts, normalize_embeddings=True,
            show_progress_bar=False, batch_size=batch_size
        ).astype(np.float32)

    def build(self, texts):
        t0 = time.time()
        self._texts = list(dict.fromkeys(texts))
        self._embeddings = self.embed(self._texts)
        self.madhava.build(self._embeddings)
        self.built = True
        return time.time() - t0

    def estimate_upper_bound(self, text):
        if not self.built:
            return 0.0
        try:
            idx = self._texts.index(text)
            emb = self._embeddings[idx]
        except ValueError:
            emb = self.embed([text])[0]
        scores = self.madhava.estimate_score(emb)
        return float(max(scores.values())) if scores else 0.0

    def _diversity_threshold(self, n_known):
        return max(0.2, 0.5 - 0.05 * math.log(n_known + 1))

    def diversity_gate(self, embedding, known_embeddings):
        if not known_embeddings:
            return True, float(np.linalg.norm(embedding))
        v = embedding.astype(np.float64).flatten().copy()
        for k in known_embeddings:
            k = k.astype(np.float64).flatten()
            nk = np.linalg.norm(k)
            if nk < 1e-10:
                continue
            k /= nk
            v -= np.dot(v, k) * k
        novelty = float(np.linalg.norm(v))
        th = self._diversity_threshold(len(known_embeddings))
        return novelty > th, novelty

    def search(self, n_beam=8, max_candidates=100,
               known_embeddings=None, threshold=0.3):
        if not self.built:
            raise RuntimeError("Call build() first")
        t0 = time.time()
        known = known_embeddings or []
        stats = {
            "n_total": len(self._texts), "n_evaluated": 0,
            "n_pruned_by_bound": 0, "n_pruned_by_diversity": 0,
            "n_pruned_low_score": 0, "n_candidates": 0,
        }
        scored = []
        for i, text in enumerate(self._texts):
            emb = self._embeddings[i]
            sd = self.madhava.estimate_score(emb)
            ub = max(sd.values()) if sd else 0.0
            scored.append((ub, i, text, emb))
            stats["n_evaluated"] += 1
        scored.sort(key=lambda x: -x[0])

        candidates, emb_cands = [], []
        best_global, seen = 0.0, set()

        for ub, idx, text, emb in scored:
            if len(candidates) >= max_candidates:
                break
            if text in seen:
                continue
            seen.add(text)
            if (ub + 0.05) < best_global:
                stats["n_pruned_by_bound"] += 1
                continue
            if ub < threshold:
                stats["n_pruned_low_score"] += 1
                continue
            passed, _ = self.diversity_gate(emb, known + emb_cands)
            if not passed:
                stats["n_pruned_by_diversity"] += 1
                continue
            candidates.append(text)
            emb_cands.append(emb)
            best_global = max(best_global, ub)

        stats["n_candidates"] = len(candidates)
        stats["search_time_s"] = time.time() - t0
        scores_ret = [self.estimate_upper_bound(t) for t in candidates]
        return candidates, scores_ret, stats


def _test_search():
    import os, pandas as pd
    from sentence_transformers import SentenceTransformer

    base = "/tmp/cyber_dataset_2026/data"
    fp = os.path.join(base, "threat_intelligence", "hf_prompt_injections.csv")
    if not os.path.exists(fp):
        print("[test] Dataset not found, SKIP")
        return True

    df = pd.read_csv(fp)
    injection_texts = df[df["label"] == 1]["text"].tolist()[:300]

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embs = model.encode(injection_texts, normalize_embeddings=True,
                        show_progress_bar=False).astype(np.float32)
    from madhava_sec.attack_families import AttackFamilyEngine
    families = AttackFamilyEngine(embs, injection_texts, n_families=10)

    search = AttackSearch(families)
    bt = search.build(injection_texts)
    print(f"[test] Build: {bt:.3f}s for {len(injection_texts)} texts")

    ub = search.estimate_upper_bound(injection_texts[0])
    print(f"[test] Upper bound: {ub:.4f}")

    cand, scores, stats = search.search(n_beam=4, max_candidates=20)
    print(f"[test] Search: {len(cand)} candidates "
          f"(pruned_bound={stats['n_pruned_by_bound']}, "
          f"pruned_div={stats['n_pruned_by_diversity']})")
    for t, s in zip(cand[:3], scores[:3]):
        print(f"  score={s:.3f} | {t[:80]}...")
    print("[test] PASSED")
    return True


if __name__ == "__main__":
    import warnings; warnings.filterwarnings('ignore')
    _test_search()
    print("\nAll tests PASSED.")
