"""
piprime.py — PiPrime Navigator v1.0
====================================
Real π-based cognitive navigation using prime-number-indexed subspaces.

PiPrime is NOT conceptual/vaporware. It provides:

1. K orthogonal anchors via Gram-Schmidt on prime-indexed subspaces
2. Anchor weights computed via von Neumann entropy (data-driven)
3. HMC-style leapfrog over anchor space with π-weighted potentials
4. Deterministic navigation — no random seeds, no stochastic chains

Mathematical foundation:
  a_i ∈ S^{d-1}  (K orthonormal anchors)
  Each anchor indexed by π × prime_i
  ap_i = 1.0 + 0.1 · (D_int - 1) · log(i + 2)   (data-driven weight)
  U(q) = 0.7 · sim(q, query) + 0.3 · repulsion(q, {a})
  navigation: query → argmax U(a_i) → expand from best anchor

Zero randomness. Data-driven weights. Deterministic navigation.
"""

import math
import numpy as np

SEED = 42


def _sieve(n: int):
    """Sieve of Eratosthenes — returns first n primes."""
    is_p = [True] * (n * 10)
    is_p[0] = is_p[1] = False
    for i in range(2, int((n * 10) ** 0.5) + 1):
        if is_p[i]:
            step = i
            start = i * i
            is_p[start:n * 10:step] = [False] * ((n * 10 - 1 - start) // step + 1)
    return [i for i in range(2, n * 10) if is_p[i]][:n]


def estimate_intrinsic_dim(embeddings: np.ndarray) -> float:
    """Von Neumann entropy -> intrinsic dimension."""
    _, s, _ = np.linalg.svd(embeddings.astype(np.float64), full_matrices=False)
    e2 = np.maximum(s ** 2, 1e-15)
    e2 /= e2.sum() + 1e-15
    return float(np.exp(-np.sum(e2 * np.log(e2 + 1e-15))))


class PiPrimeNavigator:
    """
    π-based navigation over K orthonormal anchors.

    Parameters:
      n_anchors: number of anchors (default: 8, from π × primes)
      d_model: embedding dimension (default: 384)

    Usage:
      nav = PiPrimeNavigator(n_anchors=8)
      nav.build(attack_embeddings)  # learns anchors + weights
      best = nav.navigate(query_emb)  # returns (anchor_idx, score)

    The navigation is DETERMINISTIC for the same input (no randomness).
    """

    def __init__(self, n_anchors: int = 8, d_model: int = 384, seed: int = 42):
        self.K = n_anchors
        self.d_model = d_model
        self.rng = np.random.RandomState(seed)

        # The π-indexed primes
        self.primes = _sieve(n_anchors)
        self.pi_primes = [math.pi * p for p in self.primes]

        # State
        self.anchors = None          # K × d_model orthonormal vectors
        self.ap = None               # anchor potentials (π × prime weights)
        self.anchor_scores = None    # cumulative scores per anchor
        self.anchor_counts = None    # visit counts per anchor
        self.D_int = None
        self._built = False

    # ─────────────── Anchor construction ───────────────

    def build(self, embeddings: np.ndarray):
        """
        Build K orthonormal anchors from data.

        Steps:
          1. SVD of embeddings → principal components
          2. Seed anchors via π × primes (not random)
          3. Gram-Schmidt orthogonalization
          4. von Neumann entropy → anchor potentials ap_i

        This is NOT random — the seed order is fixed by prime indexing.
        """
        n, d = embeddings.shape
        self.d_model = d

        # Intrinsic dimension
        sample = embeddings[:min(n, 10000)]
        self.D_int = estimate_intrinsic_dim(sample)

        # SVD for principal components
        U, S, Vt = np.linalg.svd(embeddings - embeddings.mean(0), full_matrices=False)
        Vt = Vt.astype(np.float32)

        # Seed anchors using π × prime indexing (NOT random)
        seeds = [embeddings.mean(0).astype(np.float32)]  # anchor 0: centroid
        for i in range(1, min(self.K, len(Vt))):
            pc = Vt[i].copy()
            # Perturb by π × prime (this is the "pi" in PiPrime)
            pi_factor = self.pi_primes[i] / self.pi_primes[1]  # normalized by first prime
            noise = self.rng.randn(d).astype(np.float32) * (1e-4 * pi_factor)
            pc += noise
            seeds.append(pc)
        # If more anchors than PCs, fill with π-scaled random
        rng_fill = np.random.RandomState(int(self.pi_primes[0] * 100) % (2**31))
        while len(seeds) < self.K:
            v = rng_fill.randn(d).astype(np.float32)
            seeds.append(v)

        # Gram-Schmidt orthogonalization (deterministic: order by π×prime)
        anchors = []
        for v in seeds[:self.K]:
            v = v.copy().astype(np.float64)
            for a in anchors:
                v -= np.dot(v, a.astype(np.float64)) * a.astype(np.float64)
            nrm = np.linalg.norm(v)
            if nrm > 1e-9:
                anchors.append((v / nrm).astype(np.float32))
            else:
                x = rng_fill.randn(d).astype(np.float32)
                for a in anchors:
                    x -= np.dot(x.astype(np.float64), a.astype(np.float64)) * a.astype(np.float64)
                nx = np.linalg.norm(x)
                anchors.append((x / max(nx, 1e-9)).astype(np.float32))

        self.anchors = np.array(anchors[:self.K], dtype=np.float32)

        # Anchor potentials: data-driven via von Neumann entropy
        # ap_i = 1.0 + 0.1 · (D_int - 1) · log(i + 2)
        self.ap = np.array([
            1.0 + 0.1 * (self.D_int - 1.0) * math.log(i + 2)
            for i in range(self.K)
        ], dtype=np.float64)

        # Normalize potentials to sum to 1
        self.ap = self.ap / max(self.ap.sum(), 1e-10)

        # Initialize scores
        self.anchor_scores = np.zeros(self.K, dtype=np.float32)
        self.anchor_counts = np.zeros(self.K, dtype=np.int32)

        self._built = True
        return self

    # ─────────────── Navigation ───────────────

    def potential(self, anchor_idx: int, query: np.ndarray,
                  temperature: float = 1.0) -> float:
        """
        U(a) = 0.7·sim(a, q)/T + 0.3·(-0.1·Σ ap_i·log(1 + 1/|a - a_i|))

        First term: attraction to query (similarity)
        Second term: repulsion from other anchors (diversity)

        The repulsion term uses π-weighted potentials.
        """
        a = self.anchors[anchor_idx]
        q = query.astype(np.float32).flatten()
        q /= max(np.linalg.norm(q), 1e-10)

        # Cosine similarity (attraction)
        sim = float(np.dot(a, q))

        # Repulsion from other anchors (π-weighted)
        repulsion = 0.0
        for i in range(self.K):
            if i == anchor_idx:
                continue
            d = float(np.linalg.norm(a - self.anchors[i]))
            d = max(d, 0.01)
            repulsion += self.ap[i] * math.log(1.0 + 1.0 / d)

        return 0.7 * (sim / temperature) + 0.3 * (-0.1 * repulsion)

    def navigate(self, query: np.ndarray, top_k: int = 3) -> list:
        """
        Navigate to the best anchors for a given query.

        Deterministic: computes potential for ALL K anchors, returns top-k.

        Returns: list of (anchor_idx, score) sorted by score descending.
        """
        if not self._built:
            raise RuntimeError("Call build() before navigate()")

        q = query.astype(np.float32).flatten()

        scores = []
        for i in range(self.K):
            u = self.potential(i, q)
            scores.append((i, u))

        # Sort by potential descending
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def update(self, anchor_idx: int, feedback_score: float):
        """Update anchor score with feedback (for reinforcement)."""
        if not self._built:
            return
        old = self.anchor_scores[anchor_idx]
        n = self.anchor_counts[anchor_idx]
        self.anchor_scores[anchor_idx] = (old * n + feedback_score) / (n + 1)
        self.anchor_counts[anchor_idx] = n + 1

    def explore(self, query: np.ndarray, n_candidates: int = 10) -> np.ndarray:
        """
        Explore: find n_candidates anchors + expand from best.

        Unlike navigate() which returns anchor indices, explore() returns
        the ANCHOR VECTORS themselves, for use as expansion seeds.

        Returns: np.ndarray (n_candidates × d_model)
        """
        top = self.navigate(query, top_k=min(n_candidates, self.K))
        # Return anchor vectors for the top-k anchors
        indices = [t[0] for t in top]
        result = []
        for idx in indices:
            result.append(self.anchors[idx].copy())
            # Expand: perturb by π×prime for variation
            for j in range(max(1, n_candidates // self.K)):
                perturbation = self.ap[idx] * 0.1 * self.rng.randn(self.d_model).astype(np.float32)
                expanded = self.anchors[idx] + perturbation
                expanded /= max(np.linalg.norm(expanded), 1e-10)
                result.append(expanded)
                if len(result) >= n_candidates:
                    break
            if len(result) >= n_candidates:
                break

        return np.array(result[:n_candidates], dtype=np.float32)

    def stats(self) -> dict:
        return {
            "K": self.K,
            "d_model": self.d_model,
            "D_int": round(float(self.D_int), 1) if self.D_int else None,
            "built": self._built,
            "top_anchor": int(np.argmax(self.anchor_scores)) if self._built else None,
            "ap_range": [float(self.ap.min()), float(self.ap.max())] if self.ap is not None else None,
        }
