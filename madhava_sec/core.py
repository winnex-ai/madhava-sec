"""
core.py — MadhavaSecEngine v3.0
QR-orthogonal projection + Cauchy-Schwarz bound + error backpropagation modulation.

Zero regex. Zero hardcoded patterns. Zero fallbacks.
"""

import time, math, warnings
import numpy as np
from numpy.linalg import qr

SEED = 42
_DISCLAIMER_SHOWN = False


def _show_disclaimer():
    global _DISCLAIMER_SHOWN
    if _DISCLAIMER_SHOWN:
        return
    _DISCLAIMER_SHOWN = True
    warnings.warn(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  Madhava-Sec: MATHEMATICAL GUARANTEE ≠ SEMANTIC GUARANTEE  ║\n"
        "╠══════════════════════════════════════════════════════════════╣\n"
        "║  0% false negatives on EMBEDDING COSINE SIMILARITY.       ║\n"
        "║  Does NOT guarantee semantic harmfulness detection.        ║\n"
        "║                                                          ║\n"
        "║  For semantic safety, combine with SafetyEnsemble:        ║\n"
        "║    from madhava_sec.semantic import SafetyEnsemble        ║\n"
        "║                                                          ║\n"
        "║  This is a CLASSIFIER, not a safety system.              ║\n"
        "╚══════════════════════════════════════════════════════════════╝",
        UserWarning, stacklevel=3
    )


def estimate_intrinsic_dim(embeddings: np.ndarray) -> float:
    """Von Neumann entropy -> intrinsic dimension estimate."""
    _, s, _ = np.linalg.svd(embeddings.astype(np.float64), full_matrices=False)
    e2 = np.maximum(s ** 2, 1e-15)
    e2 /= e2.sum() + 1e-15
    return float(np.exp(-np.sum(e2 * np.log(e2 + 1e-15))))


class MadhavaSecEngine:
    """
    QR projection + Cauchy-Schwarz bound + modulation.

    Two stages:
      Stage 1 (d1):  fast low-dim bound, 1st filter
      Stage 2 (d2):  higher-dim bound, modulation, final score
    """

    def __init__(self, stage_dims=None, seed=SEED):
        _show_disclaimer()
        self.dims = stage_dims or [64, 128]
        self.full_dim = 384
        self.rng = np.random.RandomState(seed + 1)
        self.d_int = None
        self.vectors = None
        self.n = 0
        self.proj_f32 = {}
        self.error_f32 = {}
        self.proj_mat = {}
        self.norms = None
        self.build_time = 0.0

    def _ortho_proj(self, d_out, d_in=None):
        d_in = d_in or self.full_dim
        d_out = min(d_out, d_in)
        R = self.rng.randn(d_out, d_in).astype(np.float64)
        Q, _ = qr(R.T)
        return Q[:, :d_out].T.astype(np.float32)

    def build(self, vectors):
        t0 = time.time()
        n = len(vectors)
        d_in = vectors.shape[1]
        self.full_dim = d_in
        sample = vectors[:min(n, 10000)]
        self.d_int = estimate_intrinsic_dim(sample)

        self.vectors = vectors.astype(np.float32)
        self.n = n
        norms = np.linalg.norm(self.vectors, axis=1).astype(np.float32)
        self.norms = np.maximum(norms, 1e-10)

        for d in self.dims:
            d_eff = min(d, d_in)
            P = self._ortho_proj(d_eff, d_in)
            self.proj_mat[d] = P
            proj = self.vectors @ P.T
            self.proj_f32[d] = proj
            captured = np.linalg.norm(proj, axis=1).astype(np.float32)
            self.error_f32[d] = np.sqrt(
                np.maximum(self.norms ** 2 - captured ** 2, 0)
            ).astype(np.float32)

        self.build_time = time.time() - t0
        return self

    def _upper_bound(self, pv, ev, pq, eq):
        return pv @ pq + ev * eq

    def estimate_score(self, query_vec, return_profile=False):
        """Score ALL centroids with modulated bounds. No pruning."""
        q = query_vec.astype(np.float32).flatten()
        qn = max(np.linalg.norm(q), 1e-10)
        d1, d2 = self.dims[0], self.dims[-1]
        mu = max(np.mean(self.error_f32[d1]), 1e-9)

        q1 = q @ self.proj_mat[d1].T
        qr1 = math.sqrt(max(0, qn ** 2 - np.linalg.norm(q1) ** 2))
        B1 = self._upper_bound(self.proj_f32[d1], self.error_f32[d1], q1, qr1)

        q2 = q @ self.proj_mat[d2].T
        qr2 = math.sqrt(max(0, qn ** 2 - np.linalg.norm(q2) ** 2))
        B2 = self._upper_bound(self.proj_f32[d2], self.error_f32[d2], q2, qr2)

        delta_e = (self.error_f32[d1] - self.error_f32[d2]) / mu
        alpha = np.clip(1.0 / (1.0 + np.exp(-delta_e * 0.5)), 0.01, 0.99)
        modulated = B1 + alpha * (B2 - B1)

        result = {int(i): float(modulated[i]) for i in range(self.n)}

        if return_profile:
            prof = {
                "n_total": self.n, "d_int": self.d_int,
                "dims": list(self.dims),
                "modulated_range": [float(modulated.min()), float(modulated.max())],
                "alpha_mean": float(np.mean(alpha)),
            }
            return result, prof
        return result

    def check_bounds(self, query_vec, eps_guard=True):
        q = query_vec.astype(np.float32).flatten()
        qn = max(np.linalg.norm(q), 1e-10)
        V = self.vectors
        nv = np.maximum(np.linalg.norm(V, axis=1), 1e-10)
        tru = (V @ q) / (nv * qn)
        eps = (np.finfo(np.float32).eps * 1000) if eps_guard else 1e-9
        viol = {}
        for d in self.dims:
            qd = q @ self.proj_mat[d].T
            qr = math.sqrt(max(0, qn ** 2 - np.linalg.norm(qd) ** 2))
            ub = self._upper_bound(self.proj_f32[d], self.error_f32[d], qd, qr)
            viol[f"{d}D"] = int(np.sum(tru > ub + eps))
        return viol, self.n

    def regime_check(self):
        if self.n == 0 or self.d_int is None:
            return {"flag": "UNKNOWN"}
        d = max(self.dims)
        ratio = min(1.0, d / max(self.d_int, 1))
        residual = math.sqrt(max(0, 1.0 - ratio))
        centroid_sim = 0.6 * ratio + 0.3
        expected_bound = centroid_sim + residual
        if ratio >= 0.7:
            flag = "GREEN"
        elif ratio >= 0.3:
            flag = "AMBER"
        else:
            flag = "RED"
        return {"flag": flag, "d_int": round(self.d_int, 1),
                "ratio": round(ratio, 3), "expected_bound": round(expected_bound, 3)}

    def stats(self):
        r = self.regime_check()
        total = sum(self.vectors.nbytes if self.vectors is not None else 0)
        for d in self.dims:
            if d in self.proj_f32:
                total += self.proj_f32[d].nbytes + self.proj_mat[d].nbytes + self.error_f32[d].nbytes
        return {"n": self.n, "full_dim": self.full_dim, "dims": list(self.dims),
                "d_int": round(float(self.d_int), 1), "build_time_s": round(self.build_time, 3),
                "size_mb": round(total / 1e6, 1), "regime": r["flag"]}


def optimize_threshold(scores, labels):
    """
    Find optimal threshold via Youden's J statistic (maximizes TPR - FPR).

    Youden's J = sensitivity + specificity - 1
    The threshold that maximizes J is the point where the classifier
    adds the most value over random chance.

    Args:
      scores: np.ndarray of classifier scores
      labels: np.ndarray of ground truth labels (0/1)

    Returns:
      threshold: optimal threshold value
      J: Youden's index at that threshold
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(labels, scores)
    J = tpr - fpr
    best_idx = np.argmax(J)
    return float(thr[best_idx]), float(J[best_idx])


def auto_configure(vectors, target_energy=0.50, verbose=True):
    """Auto-configure dims based on data intrinsic dimension and energy scan."""
    N, D = vectors.shape
    v32 = vectors.astype(np.float32)
    sample = v32[:min(N, 10000)]
    d_int = estimate_intrinsic_dim(sample)
    d1 = max(16, min(128, D, int(math.ceil(d_int * 1.5))))
    d2 = max(32, min(256, D, int(math.ceil(d_int * 3.0))))
    if d1 >= d2:
        d2 = min(d1 * 2, D)
    cfg = {"dims": [d1, d2], "d_int": round(d_int, 1), "full_dim": D}
    if verbose:
        print(f"[auto_configure] {N}x{D} D_int={d_int:.1f} -> dims=[{d1},{d2}]")
    return cfg
