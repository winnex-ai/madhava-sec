"""
semantic.py — SafetyEnsemble v2.0
==================================
Multi-embedder ensemble with weighted consensus and batch embedding cache.

Fixes v1.x limitations:
  - Performance: batch embedding cache (embed once per model, reuse)
  - Consensus: weighted by calibration F1, not simple binary agreement
  - Confidence: calibrated score, not just safe/unsafe boolean

Embedders:
  - all-MiniLM-L6-v2 (384D, default, fast)
  - BAAI/bge-small-en-v1.5 (384D, better on retrieval)
  - intfloat/e5-small-v2 (384D, better on classification)
"""

import time, numpy as np

EMBEDDER_NAMES = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "intfloat/e5-small-v2"]


class SafetyEnsemble:
    """
    Multi-embedder safety classification with weighted consensus.

    Key improvements over v1:
      embed(texts):           embeds ALL texts at once, caches results
      evaluate(text):         weighted consensus from calibration F1
      calibrate(attacks, bening): computes per-model F1 for weighting

    Usage:
      ensemble = SafetyEnsemble()
      ensemble.build(attack_texts)
      ensemble.calibrate(attack_texts, clean_texts)  # sets weights
      result = ensemble.evaluate(prompt_text)
      → {"safe": 0.92, "verdict": "safe", "confidences": {...}}
    """

    def __init__(self, model_names=None, cache_dir=None):
        self.model_names = model_names or EMBEDDER_NAMES
        self._embedders = {}
        self._engines = {}       # embedder_name -> MadhavaSecEngine
        self._centroids = {}     # embedder_name -> centroids
        self._thresholds = {}    # embedder_name -> threshold
        self._weights = {}       # embedder_name -> calibration weight
        self._cache = {}         # model_name -> {texts_hash: embeddings}
        self._built = False
        self._calibrated = False

    # ─────────────── Batch Embedding Cache ───────────────

    def _get_embedder(self, name):
        if name not in self._embedders:
            from sentence_transformers import SentenceTransformer
            try:
                self._embedders[name] = SentenceTransformer(name, device="cpu")
            except Exception:
                if name != "all-MiniLM-L6-v2":
                    print(f"  [SafetyEnsemble] {name} not available, falling back to MiniLM")
                    self._embedders[name] = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                else:
                    raise
        return self._embedders[name]

    def embed(self, texts, model_name=None):
        """Embed ALL texts at once. Caches per model."""
        mn = model_name or self.model_names[0]
        if mn not in self._cache:
            self._cache[mn] = {}
        key = str(hash(tuple(texts[:10])))  # simple content hash key
        if key not in self._cache[mn]:
            emb = self._get_embedder(mn).encode(
                texts, normalize_embeddings=True,
                show_progress_bar=False, batch_size=128
            ).astype(np.float32)
            self._cache[mn][key] = emb
        return self._cache[mn][key]

    # ─────────────── Build ───────────────

    def build(self, attack_texts, clean_texts=None, centroids_dict=None,
              embed_all=False, cluster_method="auto"):
        """
        Build ensemble: load/train centroids per embedder.

        Args:
          cluster_method:
            "kmeans" — hard spherical clusters (fast, K fixed)
            "hdbscan" — density-based, detects noise points, variable K
            "auto" — try HDBSCAN first, fallback to KMeans if no clusters found
        """
        for mn in self.model_names:
            print(f"  [SafetyEnsemble] Loading {mn} (method={cluster_method})...")
            t0 = time.time()

            if centroids_dict and mn in centroids_dict:
                centroids = centroids_dict[mn]
            else:
                if embed_all:
                    attack_embs = self.embed(attack_texts, mn)
                else:
                    attack_embs = self.embed(attack_texts[:min(2000, len(attack_texts))], mn)

                centroids = self._compute_centroids(attack_embs, method=cluster_method)

            self._centroids[mn] = centroids

            from .core import MadhavaSecEngine
            engine = MadhavaSecEngine(stage_dims=[64, 128]).build(centroids)
            self._engines[mn] = engine

            attack_scores = self._score_texts(attack_texts[:min(500, len(attack_texts))], mn)
            # Youden's J threshold (maximizes TPR - FPR)
            from sklearn.metrics import roc_curve
            if clean_texts:
                clean_scores = self._score_texts(clean_texts[:min(500, len(clean_texts))], mn)
                all_s = np.concatenate([attack_scores, clean_scores])
                all_l = np.array([1]*len(attack_scores) + [0]*len(clean_scores))
                fpr, tpr, thr = roc_curve(all_l, all_s)
                youden = tpr - fpr
                th = float(thr[np.argmax(youden)])
            else:
                th = float(np.percentile(attack_scores, 10))

            self._thresholds[mn] = th
            print(f"    centroids={centroids.shape[0]}, threshold={th:.4f} ({time.time()-t0:.1f}s)")

        self._built = True
        return self

    def _compute_centroids(self, embeddings, method="auto"):
        """
        Compute centroids from attack embeddings.

        Supports:
          - kmeans: fixed K, spherical clusters
          - hdbscan: density-based, detects noise, irregular shapes
          - auto: HDBSCAN with KMeans fallback
        """
        embs = embeddings.astype(np.float32)

        if method in ("hdbscan", "auto"):
            try:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(min_cluster_size=max(3, len(embs)//100),
                                             min_samples=1, metric="euclidean",
                                             gen_min_span_tree=False, core_dist_n_jobs=1)
                labels = clusterer.fit_predict(embs)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                if n_clusters >= 2:
                    # Centroids = mean of each cluster (excluding noise label -1)
                    centroids = []
                    for cid in range(n_clusters):
                        mask = labels == cid
                        if mask.sum() > 0:
                            centroids.append(embs[mask].mean(axis=0))
                    centroids = np.array(centroids, dtype=np.float32)
                    # Normalize
                    cn = np.linalg.norm(centroids, axis=1, keepdims=True)
                    cn[cn == 0] = 1.0
                    centroids /= cn
                    print(f"      HDBSCAN: {n_clusters} clusters + noise ({int((labels==-1).sum())} pts)")
                    return centroids
                elif method == "auto":
                    print(f"      HDBSCAN: only {n_clusters} cluster(s), falling back to KMeans")
                else:
                    print(f"      HDBSCAN: only {n_clusters} cluster(s), consider increasing data")

            except ImportError:
                if method == "hdbscan":
                    print(f"      hdbscan not installed. Install: pip install hdbscan")
                # fall through to KMeans

        # KMeans fallback
        from sklearn.cluster import KMeans
        K = max(2, min(30, len(embs) // 10))
        km = KMeans(n_clusters=K, random_state=42, n_init=3).fit(embs)
        centroids = km.cluster_centers_.astype(np.float32)
        cn = np.linalg.norm(centroids, axis=1, keepdims=True)
        cn[cn == 0] = 1.0; centroids /= cn
        return centroids

    def _score_texts(self, texts, model_name):
        """Batch score texts using cached embeddings."""
        embs = self.embed(texts, model_name)
        scores = np.zeros(len(texts))
        for i in range(len(texts)):
            s = self._engines[model_name].estimate_score(embs[i])
            scores[i] = max(s.values())
        return scores

    # ─────────────── Weighted Calibration ───────────────

    def calibrate(self, attack_texts, clean_texts):
        """
        Compute per-model F1 weights from calibration data.

        Models that perform better on the calibration set get higher weight
        in the consensus decision.

        Weight formula:
          w_i = F1_i / sum(F1_all)

        This ensures a model that systematically fails on certain attack
        types has proportionally less influence on the final verdict.
        """
        from sklearn.metrics import f1_score

        for mn in self.model_names:
            # Score attack texts
            attack_scores = self._score_texts(attack_texts[:min(500, len(attack_texts))], mn)
            # Score clean texts
            clean_scores = self._score_texts(clean_texts[:min(500, len(clean_texts))], mn)

            all_scores = np.concatenate([attack_scores, clean_scores])
            all_labels = np.array([1]*len(attack_scores) + [0]*len(clean_scores))

            # Best threshold for this model
            best_f1 = 0.0
            for th in np.linspace(all_scores.min(), all_scores.max(), 200):
                pred = (all_scores >= th).astype(np.int32)
                f1 = f1_score(all_labels, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1

            self._weights[mn] = best_f1
            print(f"  [Calibration] {mn}: F1={best_f1:.4f}")

        # Normalize weights to sum to 1
        total = sum(self._weights.values())
        if total > 0:
            for mn in self._weights:
                self._weights[mn] /= total
        else:
            equal = 1.0 / max(len(self._weights), 1)
            for mn in self._weights:
                self._weights[mn] = equal

        self._calibrated = True
        return self

    # ─────────────── Evaluation ───────────────

    def evaluate(self, text):
        """
        Evaluate with weighted consensus.

        Returns:
          {
            "safe": float (0-1, calibrated confidence that it's safe),
            "verdict": "safe" / "flagged" / "uncertain",
            "confidences": {embedder_name: {"score": float, "flagged": bool}},
            "weighted_score": float,
            "agreement": float,
          }
        """
        if not self._built:
            return {"safe": 0.0, "verdict": "uncertain", "error": "Call build() first"}

        # Use default weights if not calibrated
        if not self._calibrated:
            equal = 1.0 / max(len(self.model_names), 1)
            for mn in self.model_names:
                self._weights[mn] = equal

        confidences = {}
        weighted_flagged = 0.0
        weighted_total = 0.0

        for mn in self.model_names:
            emb = self.embed([text], mn)[0]
            s = self._engines[mn].estimate_score(emb)
            max_score = max(s.values())
            th = self._thresholds.get(mn, 0.3)
            flagged = max_score >= th
            w = self._weights.get(mn, 1.0 / max(len(self.model_names), 1))
            confidences[mn] = {"score": round(float(max_score), 4), "flagged": bool(flagged)}
            if flagged:
                weighted_flagged += w
            weighted_total += w

        safe_confidence = 1.0 - weighted_flagged  # 0 = unsafe, 1 = safe

        if safe_confidence >= 0.8:
            verdict = "safe"
        elif safe_confidence >= 0.5:
            verdict = "uncertain"
        else:
            verdict = "flagged"

        return {
            "safe": round(safe_confidence, 3),
            "verdict": verdict,
            "confidences": confidences,
            "weighted_score": round(weighted_flagged, 3),
            "agreement": round(1.0 - abs(weighted_flagged - 0.5) * 2, 3),
        }

    def evaluate_batch(self, texts):
        return [self.evaluate(t) for t in texts]

    def stats(self):
        return {
            "n_embedders": len(self.model_names),
            "model_names": self.model_names,
            "built": self._built,
            "calibrated": self._calibrated,
            "weights": {mn: round(w, 4) for mn, w in self._weights.items()},
            "thresholds": self._thresholds,
            "centroids_shapes": {mn: c.shape for mn, c in self._centroids.items()},
        }
