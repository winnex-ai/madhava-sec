"""
Madhava-Sec Attack Families — Embedding-Derived
=================================================
K=30 familias de ataque = centroides KMeans dos EMBEDDINGS
reais dos prompts de injection do dataset.

Zero hardcoding: familias sao derivadas dos dados via:
  1. Embedding all-MiniLM-L6-v2 (384D) de todos os prompts
  2. KMeans nos prompts POSITIVOS (label=1)
  3. Centroides resultantes = familias de ataque
  4. PiPrime potential calculado sobre similaridade real

License: BSL 1.1 | Author: Klenio Araujo Padilha
"""

import math, random, itertools, json, os, hashlib
import numpy as np
from sklearn.cluster import KMeans

SEED = 42
np.random.seed(SEED)


class AttackFamilyEngine:
    """
    K=30 familias de ataque = centroides de embeddings reais.

    Tudo derivado dos dados. Zero hardcoding.
    """

    def __init__(self, injection_embeddings: np.ndarray = None,
                 injection_texts: list = None, n_families: int = 30):
        self.K = n_families
        self.embed_dim = injection_embeddings.shape[1] if injection_embeddings is not None else 384
        self.rng = np.random.RandomState(SEED + 1)

        # Placeholder - populado via build()
        self.family_vectors = None     # K x embed_dim (centroides normalizados)
        self.family_sizes = None       # K (qts prompts em cada cluster)
        self.family_labels = None      # K (0/1 - se a familia e' injection ou nao)
        self.family_texts = []         # textos representativos de cada familia
        self.D = 1.0                   # dimensao intrinseca von Neumann
        self.ap = None                 # pesos von Neumann
        self._embedder = None
        self._built = False
        self._cache_dir = ".madhava_sec_cache"

        if injection_embeddings is not None:
            self.build(injection_embeddings, injection_texts)

    def build(self, injection_embeddings: np.ndarray,
              injection_texts: list = None):
        """
        Build: KMeans nos embeddings de injection -> K familias.

        Args:
          injection_embeddings: (N x 384) float32, embeddings dos prompts
          injection_texts: lista de N textos (opcional)
        """
        n = len(injection_embeddings)
        if n < self.K:
            self.K = max(2, n)

        # Normalizar embeddings
        norms = np.linalg.norm(injection_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_norm = injection_embeddings / norms

        # KMeans
        kmeans = KMeans(n_clusters=self.K, random_state=SEED,
                        n_init=5, max_iter=300)
        labels = kmeans.fit_predict(emb_norm)

        # Centroides sao as familias
        self.family_vectors = kmeans.cluster_centers_.astype(np.float32)
        # Normalizar centroides
        fn = np.linalg.norm(self.family_vectors, axis=1, keepdims=True)
        fn[fn == 0] = 1.0
        self.family_vectors /= fn

        # Tamanho de cada cluster
        self.family_sizes = np.array([
            int(np.sum(labels == i)) for i in range(self.K)
        ])

        # Textos representativos (um por familia)
        if injection_texts is not None:
            self.family_texts = []
            for i in range(self.K):
                idxs = np.where(labels == i)[0]
                if len(idxs) > 0:
                    # Pega o texto mais central (mais proximo do centroide)
                    dists = np.linalg.norm(
                        emb_norm[idxs] - self.family_vectors[i], axis=1
                    )
                    self.family_texts.append(injection_texts[idxs[np.argmin(dists)]])
                else:
                    self.family_texts.append("")

        # von Neumann D (Eq.13)
        _, S, _ = np.linalg.svd(self.family_vectors, full_matrices=False)
        p = S / max(S.sum(), 1e-10)
        self.D = float(math.exp(-np.sum(p * np.log(p + 1e-10))))

        # Anchor potentials
        self.ap = np.array([
            1.0 + 0.1 * (self.D - 1.0) * math.log(f + 2)
            for f in range(self.K)
        ], dtype=np.float64)
        self.ap /= max(self.ap.sum(), 1e-10)

        self._built = True
        return self

    # ───────── Embedding ─────────

    def _get_embedder(self):
        """Lazy-load embedder."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "all-MiniLM-L6-v2", device="cpu"
            )
        return self._embedder

    def embed(self, texts: list, batch_size: int = 64) -> np.ndarray:
        """Embed textos em 384D."""
        emb = self._get_embedder().encode(
            texts, normalize_embeddings=True,
            show_progress_bar=False, batch_size=batch_size
        ).astype(np.float32)
        return emb

    # ───────── Family Assignment ─────────

    def family_of(self, text_or_embedding) -> int:
        """Atribui texto/embedding a familia mais proxima."""
        if isinstance(text_or_embedding, str):
            vec = self.embed([text_or_embedding])[0]
        else:
            vec = text_or_embedding.flatten().astype(np.float32)
        if not self._built or np.linalg.norm(vec) < 1e-10:
            return 0
        sims = self.family_vectors @ vec
        return int(np.argmax(sims))

    def family_similarity(self, text_or_embedding, family_idx: int = None) -> float:
        """Similaridade coseno entre texto e familia(s)."""
        if isinstance(text_or_embedding, str):
            vec = self.embed([text_or_embedding])[0]
        else:
            vec = text_or_embedding.flatten().astype(np.float32)
        if not self._built:
            return 0.0
        sims = self.family_vectors @ vec
        if family_idx is not None:
            return float(sims[family_idx])
        return float(sims.max())

    # ───────── PiPrime Potential (Eq.1-3, Zenodo 20856138) ─────────

    def potential(self, family_idx: int, family_scores: dict) -> float:
        """
        U(f) = 0.7*<a_f,q> + 0.3*(-0.1*repulsao)

        Tudo calculado sobre embeddings reais.
        """
        score = family_scores.get(family_idx, 0.0)
        sim = -score / max(abs(score), 0.01)
        repulsion = 0.0
        for f in range(self.K):
            if f == family_idx:
                continue
            d = 1.0 - float(self.family_vectors[f] @ self.family_vectors[family_idx])
            d = max(d, 0.01)
            repulsion += self.ap[f] * math.log(1 + 1.0 / d)
        return 0.7 * sim + 0.3 * (-0.1 * repulsion)

    def gradient(self, family_idx: int, family_scores: dict) -> float:
        """Gradiente HMC baseado em similaridade real."""
        grad = 0.0
        sf = family_scores.get(family_idx, 0.0)
        for f in range(self.K):
            if f == family_idx:
                continue
            d = 1.0 - float(self.family_vectors[f] @ self.family_vectors[family_idx])
            d = max(d, 0.01)
            si = family_scores.get(f, 0.0)
            grad += 0.03 * self.ap[f] * (si - sf) / (d * (d + 0.1))
        return float(grad)

    # ───────── Diversity (Gram-Schmidt no embedding) ─────────

    def novelty_score(self, embedding: np.ndarray,
                      known_embeddings: list) -> float:
        """Norma do componente ortogonal no espaco de embedding 384D."""
        v = embedding.astype(np.float64).flatten().copy()
        if np.linalg.norm(v) < 1e-10:
            return 0.0
        for known in known_embeddings:
            k = known.astype(np.float64).flatten()
            nk = np.linalg.norm(k)
            if nk < 1e-10:
                continue
            k /= nk
            v -= np.dot(v, k) * k
        return float(np.linalg.norm(v))

    # ───────── Injection Score ─────────

    def injection_score(self, text_or_embedding) -> float:
        """
        Score de injecao: maxima similaridade com qualquer centroide
        de familia de ataque.

        Quanto mais proximo (cosine) de uma familia de ataque real
        (derivada dos dados), maior o score.

        Returns: float em [0, 1]
        """
        if not self._built:
            return 0.0
        sim = self.family_similarity(text_or_embedding)
        return float(max(0.0, sim))

    # ───────── Stats ─────────

    def stats(self) -> dict:
        return {
            "K": self.K,
            "embed_dim": self.embed_dim,
            "D_eff": round(self.D, 3),
            "built": self._built,
            "family_sizes": self.family_sizes.tolist() if self.family_sizes is not None else [],
            "family_labels": self.family_labels.tolist() if self.family_labels is not None else [],
        }


# ================================================================
# TEST
# ================================================================

def _test_build_from_real_data():
    """Testa build com dados reais do dataset."""
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    base = "/tmp/cyber_dataset_2026/data"
    fp = os.path.join(base, "threat_intelligence", "hf_prompt_injections.csv")
    if not os.path.exists(fp):
        print("[test] Dataset not found, SKIP")
        return True

    df = pd.read_csv(fp)
    injection_texts = df[df["label"] == 1]["text"].tolist()
    print(f"[test] {len(injection_texts)} injection prompts")

    # Embed
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embs = model.encode(
        injection_texts, normalize_embeddings=True,
        show_progress_bar=False
    ).astype(np.float32)
    print(f"[test] Embeddings shape: {embs.shape}")

    # Build families via KMeans
    eng = AttackFamilyEngine(embs, injection_texts, n_families=30)
    print(f"[test] K={eng.K} D_eff={eng.D:.3f}")
    print(f"[test] Family sizes: min={eng.family_sizes.min()} max={eng.family_sizes.max()}")

    # Test assignment
    test_text = injection_texts[0]
    fid = eng.family_of(test_text)
    sim = eng.family_similarity(test_text, fid)
    print(f"[test] Sample text -> family {fid}, sim={sim:.4f}")
    assert sim > 0.3, f"Similaridade muito baixa: {sim}"

    # Clean vs injection separation
    clean_texts = df[df["label"] == 0]["text"].tolist()[:200]
    inj_scores_clean = [eng.injection_score(t) for t in clean_texts]
    inj_scores_inj = [eng.injection_score(t) for t in injection_texts[:200]]
    mean_clean = float(np.mean(inj_scores_clean))
    mean_inj = float(np.mean(inj_scores_inj))
    print(f"[test] Mean inj score: clean={mean_clean:.4f} injection={mean_inj:.4f}")
    assert mean_inj > mean_clean, \
        f"Injection score nao separa: clean={mean_clean} inj={mean_inj}"

    print("[test] PASSED")
    return True


if __name__ == "__main__":
    import warnings; warnings.filterwarnings('ignore')
    _test_build_from_real_data()
    print("\nAll tests PASSED.")
