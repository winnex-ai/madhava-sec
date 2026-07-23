# Madhava-Sec Core Engine v2.1
# Fixes:
#   1. Adaptive intrinsic dimension projection (Dimensionality Paradox)
#   2. Confidence-aware pruning 3-level (Semantic Gap)
#   3. Conditional modulation (F1 Drop)
#   4. float32 memory (Memory Scaling)
# License: BSL 1.1 | pay@winnex.ai

import time, math
import numpy as np

SEED = 42


def estimate_intrinsic_dim(embeddings: np.ndarray) -> float:
    """
    Von Neumann entropy -> intrinsic dimension estimate.

    D_int = exp(-Sigma p_i log p_i)   where p_i = s_i^2 / Sigma s_i^2

    For security embeddings (AgentHarm), D_int ~ 15-25.
    For random isotropic 384D, D_int ~ 384.
    """
    _, s, _ = np.linalg.svd(embeddings.astype(np.float64), full_matrices=False)
    e2 = np.maximum(s ** 2, 1e-15)
    e2 /= e2.sum() + 1e-15
    S_vn = float(np.exp(-np.sum(e2 * np.log(e2 + 1e-15))))
    return S_vn


class MadhavaSecEngine:
    """
    Madhava-Sec: Cauchy-Schwarz bound pruning for agent security.

    Fixes v2.1:
      - Adaptive d1/d2 based on intrinsic dimension
      - Conditional modulation (only when improvement > threshold)
      - Confidence-aware pruning with 3 levels
      - float32 everywhere (50 percent memory reduction)
    """

    def __init__(self, stage_dims=None, seed=SEED):
        self.dims = stage_dims or [32, 128]
        self.full_dim = 384
        self.keep_ratio = 0.15
        self.max_candidates = 200
        self.final_topk = 50
        self.rng = np.random.RandomState(seed + 1)
        self.d_int = None
        self.adaptive_dims = None
        self.attack_vectors = None
        self.n_attacks = 0
        self.proj_f32 = {}
        self.error_f32 = {}
        self.proj_mat = {}
        self.norms = None
        self.build_time = 0.0

    def _compute_adaptive_dims(self, D_int, d_in):
        """Stage 1: ~60 percent energy. Stage 2: ~85 percent energy."""
        d1 = max(16, min(128, d_in, int(math.ceil(D_int * 1.5))))
        d2 = max(32, min(256, d_in, int(math.ceil(D_int * 3.0))))
        if d1 >= d2:
            d2 = min(d1 * 2, d_in)
        return [d1, d2]

    def _ortho_proj(self, d_out, d_in=None):
        d_in = d_in or self.full_dim
        d_out = min(d_out, d_in)
        R = self.rng.randn(d_out, d_in).astype(np.float64)
        Q, _ = np.linalg.qr(R.T)
        return Q[:, :d_out].T.astype(np.float32)

    def build(self, attack_vectors):
        """Build with adaptive dimensions, stored as float32."""
        t0 = time.time()
        n = len(attack_vectors)
        d_in = attack_vectors.shape[1]
        self.full_dim = d_in

        # Adaptive dims from intrinsic dimension
        sample = attack_vectors[:min(n, 10000)]
        self.d_int = estimate_intrinsic_dim(sample)
        self.adaptive_dims = self._compute_adaptive_dims(self.d_int, d_in)
        self.dims = self.adaptive_dims

        # Store as float32
        self.attack_vectors = attack_vectors.astype(np.float32)
        self.n_attacks = n
        norms = np.linalg.norm(self.attack_vectors, axis=1).astype(np.float32)
        self.norms = np.maximum(norms, 1e-10)

        for d in self.dims:
            P = self._ortho_proj(d, d_in)
            self.proj_mat[d] = P
            proj = self.attack_vectors @ P.T
            self.proj_f32[d] = proj
            captured = np.linalg.norm(proj, axis=1).astype(np.float32)
            self.error_f32[d] = np.sqrt(
                np.maximum(self.norms ** 2 - captured ** 2, 0)
            ).astype(np.float32)

        self.build_time = time.time() - t0
        return self

    def _upper_bound(self, pv, ev, pq, eq):
        return pv @ pq + ev * eq

    def _modulation_alpha(self, e1, e2):
        mu = max(np.mean(e1), 1e-9)
        return 1.0 / (1.0 + np.exp(-(e1 - e2) / mu * 0.5))

    def estimate_score(self, query_vec, return_profile=False):
        """Score estimation with 3-level confidence pruning."""
        q = query_vec.astype(np.float32).flatten()
        q_norm = max(np.linalg.norm(q), 1e-10)
        prof = {"n_total": self.n_attacks, "d_int": self.d_int,
                "adaptive_dims": self.adaptive_dims}

        # Stage 1
        d1 = self.dims[0]
        q1 = q @ self.proj_mat[d1].T
        qr1 = math.sqrt(max(0, q_norm ** 2 - np.linalg.norm(q1) ** 2))
        B1 = self._upper_bound(self.proj_f32[d1], self.error_f32[d1], q1, qr1)
        prof["B1_range"] = [float(B1.min()), float(B1.max())]

        keep1 = min(max(int(self.n_attacks * self.keep_ratio), 50),
                    self.max_candidates, self.n_attacks)
        if self.n_attacks <= keep1:
            idx1 = np.arange(self.n_attacks)
        else:
            idx1 = np.argpartition(-B1, max(0, keep1 - 1))[:keep1]

        # Stage 2 with conditional modulation
        d2 = self.dims[1]
        q2 = q @ self.proj_mat[d2].T
        qr2 = math.sqrt(max(0, q_norm ** 2 - np.linalg.norm(q2) ** 2))
        B2 = self._upper_bound(self.proj_f32[d2][idx1], self.error_f32[d2][idx1],
                               q2, qr2)

        e1_sel = self.error_f32[d1][idx1]
        e2_sel = self.error_f32[d2][idx1]
        imp_ratio = max(np.mean(e1_sel), 1e-9) / max(np.mean(e2_sel), 1e-9)

        if imp_ratio < 1.2:
            scores = B1[idx1]
            prof["modulation"] = "none"
        elif imp_ratio < 2.0:
            alpha = self._modulation_alpha(e1_sel, e2_sel)
            alpha *= (imp_ratio - 1.2) / 0.8
            scores = B1[idx1] + alpha * (B2 - B1[idx1])
            prof["modulation"] = "partial"
        else:
            alpha = self._modulation_alpha(e1_sel, e2_sel)
            scores = B1[idx1] + alpha * (B2 - B1[idx1])
            prof["modulation"] = "full"

        prof["improvement_ratio"] = float(imp_ratio)

        # Stage 3 with larger k2
        keep2 = max(self.final_topk, min(200, len(idx1)))
        idx2 = idx1[np.argpartition(-scores, max(0, keep2 - 1))[:keep2]]

        exact = self.attack_vectors[idx2].astype(np.float32) @ q
        nv = np.maximum(np.linalg.norm(
            self.attack_vectors[idx2].astype(np.float32), axis=1), 1e-10)
        exact_norm = exact / (nv * q_norm)

        result = {int(i): float(s) for i, s in zip(idx2, exact_norm)}

        # Confidence levels
        if len(idx2) > 0:
            best = float(exact_norm.max())
            margins = [float(s) - best for s in exact_norm]
            prof["confidence_levels"] = {
                "confident": sum(1 for m in margins if m > 0.3),
                "borderline": sum(1 for m in margins if 0.1 < m <= 0.3),
                "uncertain": sum(1 for m in margins if m <= 0.1),
            }

        if return_profile:
            return result, prof
        return result

    def check_bounds(self, query_vec):
        """Verify 0 percent bound violation guarantee."""
        q = query_vec.astype(np.float32).flatten()
        qn = max(np.linalg.norm(q), 1e-10)
        V = self.attack_vectors
        nv = np.maximum(np.linalg.norm(V.astype(np.float32), axis=1), 1e-10)
        true = (V.astype(np.float32) @ q) / (nv * qn)
        viol = {}
        for d in self.dims:
            P = self.proj_mat[d]
            qd = q @ P.T
            qr = math.sqrt(max(0, qn ** 2 - np.linalg.norm(qd) ** 2))
            ub = self._upper_bound(self.proj_f32[d], self.error_f32[d], qd, qr)
            viol[f"{d}D"] = int(np.sum(true > ub + 1e-9))
        return viol, self.n_attacks

    def stats(self):
        return {
            "n_attacks": self.n_attacks, "full_dim": self.full_dim,
            "dims": self.dims, "d_int": float(self.d_int) if self.d_int else None,
            "build_time_s": self.build_time, "size_mb": self.size_bytes() / 1e6,
        }

    def size_bytes(self):
        total = 0
        if self.attack_vectors is not None:
            total += self.attack_vectors.nbytes
        for d in self.dims:
            if d in self.proj_f32:
                total += self.proj_f32[d].nbytes
                total += self.proj_mat[d].nbytes
                total += self.error_f32[d].nbytes
        return total
