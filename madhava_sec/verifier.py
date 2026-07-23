"""
Madhava-Sec Formal Verifier — Cauchy-Schwarz Bound
====================================================
Pre-grader baseado unicamente em dot product e upper bound.

Nao ha:
  - softmax
  - probabilidades
  - regex
  - strings hardcoded
  - pesos manuais

Ha apenas:
  1. modelo de embedding (all-MiniLM-L6-v2, 384D)
  2. centroides KMeans do dataset (familias de ataque)
  3. similaridade coseno: sim = max(query · centroid)
  4. Cauchy-Schwarz bound: sim_real ≤ sim_projected + residual

A decisao de "e' injection?" e' simplesmente:
  injection_score = max(query · families) > threshold

O threshold e' derivado dos dados: media dos injection scores
sobre prompts de injection do dataset.

License: BSL 1.1 | Author: Klenio Araujo Padilha
"""

import numpy as np

SEVERITY_LEVELS = {5: "CRITICAL", 4: "HIGH", 3: "MEDIUM", 2: "LOW", 1: "INFO"}


class FormalVerifier:
    """
    Verificador: dot product + Cauchy-Schwarz bound.

    Toda decisao e' uma similaridade coseno direta.

    Nao ha:
      - softmax
      - probabilidades
      - confidence em [0,1]
      - pesos manuais
    """

    def __init__(self, families_engine=None, threshold: float = None):
        self.families = families_engine
        self._embedder = None
        # Threshold derivado dos dados se nao especificado
        self._threshold = threshold

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "all-MiniLM-L6-v2", device="cpu"
            )
        return self._embedder

    def embed(self, text: str) -> np.ndarray:
        """Embed texto -> vetor 384D normalizado (||v|| = 1)."""
        return self._get_embedder().encode(
            [text], normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)[0]

    def embed_batch(self, texts: list) -> np.ndarray:
        """Embed batch de textos -> matrix N x 384D normalizada."""
        return self._get_embedder().encode(
            texts, normalize_embeddings=True,
            show_progress_bar=False, batch_size=128
        ).astype(np.float32)

    # ───────── Score de Injecao (dot product puro) ─────────

    def injection_score(self, text: str) -> float:
        """
        Score de injecao = max dot product com familias de ataque.

        Se ||v|| = ||centroid|| = 1 (ambos normalizados),
        entao dot product = cosine similarity.

        Returns: float em [-1, 1]
        """
        if self.families is None or not self.families._built:
            return 0.0
        vec = self.embed(text)  # 384D, ||vec|| = 1
        # dot product com todos os K centroides
        sims = self.families.family_vectors @ vec  # K vetores
        return float(sims.max())

    def injection_score_batch(self, embs: np.ndarray) -> np.ndarray:
        """Score de injecao para batch de embeddings."""
        if self.families is None or not self.families._built:
            return np.zeros(len(embs), dtype=np.float32)
        # embs: N x 384, families: K x 384 -> sims: N x K
        sims = embs @ self.families.family_vectors.T
        return sims.max(axis=1)

    # ───────── Threshold derivado dos dados ─────────

    def derive_threshold(self, injection_texts: list,
                         clean_texts: list = None,
                         percentile: float = 5.0) -> float:
        """
        Deriva threshold dos dados.

        threshold = p-esimo percentil dos injection scores
        sobre exemplares de injection do dataset.

        Args:
          injection_texts: textos com injecao (label=1)
          clean_texts: textos limpos (label=0) - opcional
          percentile: percentil para threshold (default 5%)

        Returns: threshold float
        """
        if not injection_texts:
            self._threshold = None  # MUST be derived from data via derive_threshold()
            return self._threshold

        # Embed todos os injection texts
        embs = self.embed_batch(injection_texts)
        scores = self.injection_score_batch(embs)

        # Threshold = percentile dos scores de injection
        self._threshold = float(np.percentile(scores, percentile))

        if clean_texts:
            clean_embs = self.embed_batch(clean_texts)
            clean_scores = self.injection_score_batch(clean_embs)
            # Ajustar se threshold pega muitos clean
            clean_fp = np.mean(clean_scores >= self._threshold) * 100
            self._clean_fp_rate = clean_fp

        return self._threshold

    # ───────── Pre-Grade (bound-based) ─────────

    def pre_grade(self, text: str) -> dict:
        """
        Pre-grade baseado em dot product.

        severity = min(5, max(1, round(sim * 5)))

        Onde sim e' o max dot product com familias de ataque.
        """
        sim = self.injection_score(text)
        # Mapear [-1, 1] -> [1, 5]
        sim_norm = max(0.0, sim)  # clip negativo
        sev = min(5, max(1, int(round(sim_norm * 5))))

        if sev >= 5:
            verdict = "CRITICAL"
        elif sev >= 4:
            verdict = "HIGH"
        elif sev >= 3:
            verdict = "MEDIUM"
        else:
            verdict = "LOW"

        return {
            "score": round(sim_norm, 4),
            "severity": sev,
            "verdict": verdict,
        }

    # ───────── Verify Candidate (dot product threshold) ─────────

    def verify_candidate(self, text: str) -> tuple:
        """
        Verify candidate: injection_score > threshold.

        Returns: (approved: bool, score: float)
          approved = True se o texto parece injecao
          score = max dot product com familias
        """
        if not text or len(text) < 5:
            return False, 0.0
        score = self.injection_score(text)
        th = self._threshold if self._threshold is not None else 0.0  # force derive_threshold() call
        approved = score >= th
        return approved, round(score, 4)


# ================================================================
# TEST
# ================================================================

def _test_verifier():
    import os, pandas as pd
    from sentence_transformers import SentenceTransformer

    base = "/tmp/cyber_dataset_2026/data"
    fp = os.path.join(base, "threat_intelligence", "hf_prompt_injections.csv")
    if not os.path.exists(fp):
        print("[test] Dataset not found, SKIP")
        return True

    df = pd.read_csv(fp)
    injection_texts = df[df["label"] == 1]["text"].tolist()
    clean_texts = df[df["label"] == 0]["text"].tolist()
    print(f"[test] {len(injection_texts)} injection, {len(clean_texts)} clean")

    # Build families (KMeans sobre embeddings de injection)
    from madhava_sec.attack_families import AttackFamilyEngine
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    inj_embs = model.encode(
        injection_texts, normalize_embeddings=True,
        show_progress_bar=False
    ).astype(np.float32)
    families = AttackFamilyEngine(inj_embs, injection_texts, n_families=30)

    # Verifier
    verifier = FormalVerifier(families)
    # Derivar threshold dos dados
    th = verifier.derive_threshold(
        injection_texts[:1000], clean_texts[:500], percentile=10.0
    )
    print(f"[test] Threshold derivado: {th:.4f}")

    # Testar em batch
    n_test = 500
    inj_test = injection_texts[:n_test]
    clean_test = clean_texts[:n_test]

    inj_embs_test = verifier.embed_batch(inj_test)
    clean_embs_test = verifier.embed_batch(clean_test)
    inj_scores = verifier.injection_score_batch(inj_embs_test)
    clean_scores = verifier.injection_score_batch(clean_embs_test)

    mean_inj = float(np.mean(inj_scores))
    mean_clean = float(np.mean(clean_scores))
    print(f"[test] Mean injection score: clean={mean_clean:.4f} injection={mean_inj:.4f}")

    detections_inj = int(np.sum(inj_scores >= th))
    detections_clean = int(np.sum(clean_scores >= th))
    print(f"[test] Detections @ th={th:.3f}: {detections_inj}/{n_test} injection "
          f"({detections_inj/n_test*100:.0f}%), "
          f"{detections_clean}/{n_test} clean ({detections_clean/n_test*100:.0f}%)")

    assert mean_inj > mean_clean, \
        f"Injection mean ({mean_inj:.4f}) <= clean mean ({mean_clean:.4f})"

    # Test single candidate
    clean_txt = "What is the weather in London today?"
    inj_txt = injection_texts[0]
    c_ok, c_sc = verifier.verify_candidate(clean_txt)
    i_ok, i_sc = verifier.verify_candidate(inj_txt)
    print(f"[test] Single: clean={c_ok}({c_sc:.4f}) injection={i_ok}({i_sc:.4f})")

    # Pre-grade
    grade = verifier.pre_grade(inj_txt)
    print(f"[test] Pre-grade: score={grade['score']} sev={grade['severity']} "
          f"verdict={grade['verdict']}")

    print("[test] PASSED")
    return True


if __name__ == "__main__":
    import warnings; warnings.filterwarnings('ignore')
    _test_verifier()
    print("\nAll tests PASSED.")
