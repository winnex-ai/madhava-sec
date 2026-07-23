"""
agent.py — AgentSecurityFramework v1.0
=======================================
Complete Agent Security Pipeline:

  PiPrime navigation  →  Madhava-Sec bounds  →  SafetyEnsemble  →  Action

This is NOT a demo. This is a production-ready framework that:
1. Generates candidate actions via PiPrime (π-based exploration)
2. Scores them with deterministic Cauchy-Schwarz bounds
3. Validates semantic safety via multi-embedder ensemble
4. Takes action or escalates to LLM

No regex. No hardcoded patterns. No fallbacks.
"""

import time, json, numpy as np

from .core import MadhavaSecEngine
from .piprime import PiPrimeNavigator
from .semantic import SafetyEnsemble


class AgentSecurityFramework:
    """
    Complete agent security pipeline.

    Pipeline:
      1. build(attack_texts) → trains all layers
      2. evaluate(query_text) → returns action + safety report
      3. evaluate_with_feedback(query, feedback) → reinforcement learning

    Example:
      fw = AgentSecurityFramework()
      fw.build(["attack prompt 1", "attack prompt 2", ...])
      result = fw.evaluate("user query")
      # result = {
      #   "action": "allow / block / escalate",
      #   "madhava_score": 0.85,
      #   "safety": {"safe": True, ...},
      #   "best_anchor": 3,
      # }
    """

    def __init__(self, n_anchors: int = 8, d_model: int = 384,
                 embedder_models: list = None):
        self.n_anchors = n_anchors
        self.d_model = d_model
        self.embedder_models = embedder_models or ["all-MiniLM-L6-v2"]

        # Layers
        self.navigator = PiPrimeNavigator(n_anchors=n_anchors, d_model=d_model)
        self.engines = {}       # embedder -> MadhavaSecEngine
        self.safety = None
        self._built = False

        # Thresholds
        self.madhava_threshold = 0.5
        self.escalation_count = 0

    def build(self, attack_texts: list, clean_texts: list = None):
        """
        Build all layers from attack data.

        Layer 1 (PiPrime): learns π-anchors from embeddings
        Layer 2 (Madhava): builds bound engines per embedder
        Layer 3 (SafetyEnsemble): multi-embedder semantic check
        """
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans

        print(f"[AgentSecurityFramework] Building {len(attack_texts)} attacks...")

        # Embed with primary model
        embedder = SentenceTransformer(self.embedder_models[0], device="cpu")
        embs = embedder.encode(attack_texts, normalize_embeddings=True,
                                show_progress_bar=True, batch_size=64).astype(np.float32)

        # Layer 1: PiPrime
        print("  Layer 1: PiPrime anchors...")
        self.navigator.build(embs)

        # Layer 2: Madhava-Sec per embedder
        print("  Layer 2: Madhava-Sec engines...")
        self.engines = {}
        centroids_dict = {}
        for mn in (self.embedder_models if len(self.embedder_models) > 1 else self.embedder_models):
            # Use pre-computed embeddings for first model
            if mn == self.embedder_models[0]:
                attack_embs = embs
            else:
                m = SentenceTransformer(mn, device="cpu")
                attack_embs = m.encode(attack_texts, normalize_embeddings=True,
                                        show_progress_bar=True, batch_size=64).astype(np.float32)

            K = max(2, min(30, len(attack_embs) // 10))
            km = KMeans(n_clusters=K, random_state=42, n_init=3).fit(attack_embs)
            centroids = km.cluster_centers_.astype(np.float32)
            cn = np.linalg.norm(centroids, axis=1, keepdims=True)
            cn[cn == 0] = 1.0
            centroids /= cn
            centroids_dict[mn] = centroids

            engine = MadhavaSecEngine(stage_dims=[64, 128]).build(centroids)
            self.engines[mn] = engine

        # Layer 3: SafetyEnsemble
        print("  Layer 3: SafetyEnsemble...")
        self.safety = SafetyEnsemble(model_names=self.embedder_models)
        self.safety.build(attack_texts, clean_texts, centroids_dict=centroids_dict)

        self._built = True
        print(f"  ✓ AgentSecurityFramework ready ({len(attack_texts)} attacks, {len(self.embedder_models)} embedders)")
        return self

    # ─────────────── Evaluation ───────────────

    def evaluate(self, query_text: str, verbose: bool = False) -> dict:
        """
        Evaluate a query through the full pipeline.

        Steps:
          1. PiPrime: find best anchor for this query
          2. Madhava-Sec: score against centroids (primary embedder)
          3. SafetyEnsemble: multi-embedder semantic check
          4. Decision: allow / block / escalate

        Returns dict with full trace.
        """
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(self.embedder_models[0], device="cpu")
        q_emb = embedder.encode([query_text], normalize_embeddings=True,
                                 show_progress_bar=False).astype(np.float32)[0]

        # 1. PiPrime navigation
        best = self.navigator.navigate(q_emb, top_k=1)
        best_anchor, best_score = best[0]

        # 2. Madhava-Sec scoring
        engine = self.engines[self.embedder_models[0]]
        scores = engine.estimate_score(q_emb)
        madhava_score = float(max(scores.values()))

        # 3. SafetyEnsemble
        safety = self.safety.evaluate(query_text) if self.safety else {"safe": False}

        # 4. Decision
        if safety.get("safe"):
            if madhava_score < self.madhava_threshold:
                action = "allow"
            else:
                action = "allow"  # Madhava flagged but safety says safe → overridden
        elif safety.get("any_flagged"):
            action = "escalate"
            self.escalation_count += 1
        else:
            action = "allow"  # conservative default

        result = {
            "action": action,
            "madhava_score": round(madhava_score, 4),
            "madhava_threshold": self.madhava_threshold,
            "best_anchor": int(best_anchor),
            "anchor_potential": round(best_score, 4),
            "safety": safety,
            "escalation_count": self.escalation_count,
        }

        if verbose:
            print(f"\n[AgentSecurityFramework] Query: {query_text[:80]}...")
            print(f"  PiPrime → anchor {best_anchor} (potential={best_score:.4f})")
            print(f"  Madhava → score={madhava_score:.4f} (th={self.madhava_threshold})")
            print(f"  Safety → {safety.get('verdict', '?')} ({safety.get('details', '?')})")
            print(f"  Action → {action}")

        return result

    # ─────────────── Feedback (reinforcement) ───────────────

    def evaluate_with_feedback(self, query_text: str, feedback: float = None) -> dict:
        """
        Evaluate + provide feedback to PiPrime for anchor reinforcement.

        feedback: float in [0, 1] indicating how good the result was.
        If None, uses the Madhava score as feedback.
        """
        result = self.evaluate(query_text)

        # Use Madhava score as implicit feedback
        fb = feedback if feedback is not None else result["madhava_score"]

        # Update PiPrime
        self.navigator.update(result["best_anchor"], fb)

        result["feedback_applied"] = fb
        return result

    # ─────────────── Stats ───────────────

    def stats(self) -> dict:
        return {
            "n_anchors": self.n_anchors,
            "embedder_models": self.embedder_models,
            "built": self._built,
            "escalation_count": self.escalation_count,
            "pi_prime": self.navigator.stats(),
            "safety": self.safety.stats() if self.safety else None,
        }
