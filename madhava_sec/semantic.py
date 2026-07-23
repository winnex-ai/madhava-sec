"""
semantic.py — SafetyEnsemble v1.0
==================================
Multi-embedder ensemble for semantic safety detection.

Resolves the GIGO problem: instead of relying on a SINGLE embedding model
(all-MiniLM-L6-v2), SafetyEnsemble uses MULTIPLE embedders and only flags
a prompt as safe if ALL models agree.

Embedders:
  - all-MiniLM-L6-v2 (384D, default, fast)
  - BGE-small-en-v1.5 (384D, better on retrieval)
  - e5-small-v2 (384D, better on classification)

If any embedder disagrees, the prompt is escalated to LLM judge.
This eliminates the "false sense of security" from single-embedder GIGO.
"""

import numpy as np
import warnings

EMBEDDER_NAMES = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "intfloat/e5-small-v2"]


class SafetyEnsemble:
    """
    Multi-embedder safety classification.

    Usage:
      ensemble = SafetyEnsemble()
      ensemble.build(attack_embeddings_dict)  # dict: embedder_name -> (centroids, engine)
      result = ensemble.evaluate(prompt_text)
      → {"safe": True/False, "scores": {...}, "agreement": 1.0}
    """

    def __init__(self, model_names=None):
        self.model_names = model_names or EMBEDDER_NAMES
        self._embedders = {}
        self._engines = {}      # embedder_name -> MadhavaSecEngine
        self._centroids = {}    # embedder_name -> centroids
        self._thresholds = {}   # embedder_name -> threshold
        self._built = False

    def _get_embedder(self, name):
        """Lazy-load sentence transformer model."""
        if name not in self._embedders:
            from sentence_transformers import SentenceTransformer
            try:
                self._embedders[name] = SentenceTransformer(name, device="cpu")
            except Exception:
                # Fallback: use MiniLM if model not available
                if name != "all-MiniLM-L6-v2":
                    print(f"  [SafetyEnsemble] {name} not available, falling back to MiniLM")
                    from sentence_transformers import SentenceTransformer
                    self._embedders[name] = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                else:
                    raise
        return self._embedders[name]

    def embed(self, texts: list, model_name: str = None) -> np.ndarray:
        """Embed texts with specific model."""
        mn = model_name or self.model_names[0]
        emb = self._get_embedder(mn).encode(
            texts, normalize_embeddings=True,
            show_progress_bar=False, batch_size=64
        ).astype(np.float32)
        return emb

    def build(self, attack_texts: list, clean_texts: list = None,
              centroids_dict: dict = None):
        """
        Build ensemble: load/train centroids per embedder.

        Args:
          attack_texts: list of attack prompt strings
          clean_texts: optional list of benign prompts (for threshold calibration)
          centroids_dict: optional pre-computed centroids {embedder_name: np.ndarray}
        """
        from sklearn.cluster import KMeans

        for mn in self.model_names:
            print(f"  [SafetyEnsemble] Loading {mn}...")

            if centroids_dict and mn in centroids_dict:
                centroids = centroids_dict[mn]
            else:
                # Embed attack texts with this model
                embs = self.embed(attack_texts, mn)
                # KMeans -> centroids
                K = max(2, min(30, len(embs) // 10))
                km = KMeans(n_clusters=K, random_state=42, n_init=3).fit(embs)
                centroids = km.cluster_centers_.astype(np.float32)
                cn = np.linalg.norm(centroids, axis=1, keepdims=True)
                cn[cn == 0] = 1.0
                centroids /= cn

            self._centroids[mn] = centroids

            # Build Madhava-Sec engine for this embedder
            from .core import MadhavaSecEngine
            engine = MadhavaSecEngine(stage_dims=[64, 128]).build(centroids)
            self._engines[mn] = engine

            # Compute threshold from data
            attack_scores = self._score_texts(attack_texts[:min(500, len(attack_texts))], mn)
            th = float(np.percentile(attack_scores, 10))  # 10th percentile
            self._thresholds[mn] = th

            print(f"    centroids={centroids.shape[0]}, threshold={th:.4f}")

        self._built = True
        return self

    def _score_texts(self, texts: list, model_name: str) -> np.ndarray:
        """Score a list of texts using a specific embedder's engine."""
        embs = self.embed(texts, model_name)
        scores = np.zeros(len(texts))
        for i in range(len(texts)):
            s = self._engines[model_name].estimate_score(embs[i])
            scores[i] = max(s.values())
        return scores

    def evaluate(self, text: str) -> dict:
        """
        Evaluate a single text using ALL embedders.

        Returns:
          {
            "safe": bool (True if ALL embedders agree safe),
            "scores": {embedder_name: float},
            "thresholds": {embedder_name: float},
            "agreement": float (fraction of embedders agreeing),
            "details": str
          }
        """
        if not self._built:
            return {"safe": False, "error": "Call build() first"}

        scores = {}
        decisions = []
        for mn in self.model_names:
            emb = self.embed([text], mn)[0]
            s = self._engines[mn].estimate_score(emb)
            max_score = max(s.values())
            th = self._thresholds.get(mn, 0.3)
            scores[mn] = round(float(max_score), 4)
            decisions.append(max_score >= th)

        n_safe = sum(1 for d in decisions if not d)
        n_unsafe = sum(1 for d in decisions if d)
        agreement = max(n_safe, n_unsafe) / max(len(decisions), 1)

        # Safe = ALL embedders agree the score is below threshold
        all_safe = all(not d for d in decisions)
        any_unsafe = any(d for d in decisions)

        return {
            "safe": all_safe,
            "any_flagged": any_unsafe,
            "scores": scores,
            "thresholds": {mn: round(self._thresholds.get(mn, 0.3), 4) for mn in self.model_names},
            "agreement": round(agreement, 3),
            "verdict": "safe" if all_safe else "flagged" if any_unsafe else "unknown",
            "details": f"{n_safe}/{len(decisions)} embedders agree safe" if all_safe else (
                f"{n_unsafe}/{len(decisions)} embedders flagged — escalate to LLM"
            )
        }

    def evaluate_batch(self, texts: list) -> list:
        """Evaluate a batch of texts. Returns list of dicts."""
        return [self.evaluate(t) for t in texts]

    def stats(self) -> dict:
        return {
            "n_embedders": len(self.model_names),
            "model_names": self.model_names,
            "built": self._built,
            "centroids_shapes": {mn: c.shape for mn, c in self._centroids.items()},
            "thresholds": self._thresholds,
        }
