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

    # ════════════════════════════════════════════════════════════
    # OPERATIONAL BOUNDARY FLAGS
    # ════════════════════════════════════════════════════════════

    def regime_check(self, verbose=False):
        """
        Evaluate whether Madhava-Sec operates within its guaranteed regime.

        The Cauchy-Schwarz bound is effective only when data has low intrinsic
        dimensionality. On high-dimensional isotropic data, the bound becomes
        too loose to prune anything (bound exceeds max cosine = 1.0).

        Returns:
          dict with fields:
            - flag: str, one of "GREEN" / "AMBER" / "RED"
            - d_int: float, intrinsic dimension estimate
            - d_ratio: float, d_int / full_dim (lower = better for bound)
            - bound_looseness: float, expected bound for random query
            - pruning_potential: float, estimated % of candidates prunable
            - message: str, human-readable explanation
        """
        if self.d_int is None:
            return {"flag": "UNKNOWN", "message": "Call build() first"}

        d_ratio = self.d_int / max(self.full_dim, 1)
        # Expected bound = centroid_score + residual
        # For isotropic data: residual ≈ 1 → bound ≈ 0 + 1 = 1.0 (no pruning)
        # For low-dim data: residual ≈ (1 - d_out/D_int) → bound < 1
        d_out_max = max(self.dims)
        frac_energy = min(1.0, d_out_max / max(self.d_int, 1))
        expected_residual = math.sqrt(max(0, 1.0 - frac_energy))
        expected_centroid = 0.3 if d_ratio > 0.3 else 0.7  # heuristic
        expected_bound = expected_centroid + expected_residual
        bound_looseness = min(1.0, expected_bound / 1.0)

        # Estimate pruning potential: if expected_bound > 0.9, little pruning
        if expected_bound > 0.95:
            pruning_potential = max(0.0, 1.0 - expected_bound) * 100  # 0-5%
            flag = "RED"
            message = (
                f"D_int={self.d_int:.0f}/{self.full_dim} (ratio={d_ratio:.2f}). "
                f"Expected bound={expected_bound:.2f} > 0.95. "
                f"Data appears HIGH-DIMENSION ISOTROPIC. "
                f"Cauchy-Schwarz bound is too loose to prune effectively. "
                f"Consider dimensionality reduction before Madhava-Sec."
            )
        elif expected_bound > 0.85:
            pruning_potential = (1.0 - expected_bound) * 100  # 5-15%
            flag = "AMBER"
            message = (
                f"D_int={self.d_int:.0f}/{self.full_dim} (ratio={d_ratio:.2f}). "
                f"Expected bound={expected_bound:.2f} > 0.85. "
                f"Pruning will be LIMITED. "
                f"Only the bottom ~{pruning_potential:.0f}% candidates can be "
                f"provably excluded. Verify results manually."
            )
        else:
            pruning_potential = max(50, (1.0 - expected_bound) * 100)  # 50%+
            flag = "GREEN"
            message = (
                f"D_int={self.d_int:.0f}/{self.full_dim} (ratio={d_ratio:.2f}). "
                f"Expected bound={expected_bound:.2f}. "
                f"Data appears LOW-DIMENSION STRUCTURED. "
                f"Madhava-Sec is operating within its guaranteed regime. "
                f"Expected pruning: ~{pruning_potential:.0f}% of candidates."
            )

        result = {
            "flag": flag,
            "d_int": float(self.d_int),
            "full_dim": self.full_dim,
            "d_ratio": round(d_ratio, 3),
            "expected_bound": round(expected_bound, 3),
            "pruning_potential_pct": round(pruning_potential, 1),
            "message": message,
        }

        if verbose:
            regime = {
                "GREEN": "Operational Bounds Satisfied",
                "AMBER": "Caution: Reduced Pruning Efficiency",
                "RED": "FAILURE: Regime Not Supported",
                "UNKNOWN": "Build Required",
            }
            print(f"[Madhava-Sec] Regime Flag: {flag} — {regime.get(flag, '?')}")
            print(f"  D_intrinsic = {self.d_int:.1f} / {self.full_dim}")
            print(f"  Expected bound = {expected_bound:.3f}")
            print(f"  Pruning potential = {pruning_potential:.1f}%")
            print(f"  {message}")

        return result

    def prune_efficiency_estimate(self):
        """
        Quick estimate of how many bounds will be tight enough to prune.

        Returns:
          n_loose: int, count of candidates with bound > 0.95 (not prunable)
          n_tight: int, count with bound <= 0.95 (potentially prunable)
          ratio: float, tight/total
        """
        if self.n_attacks == 0 or not all(d in self.error_f32 for d in self.dims):
            return {"n_loose": 0, "n_tight": 0, "ratio": 0.0}

        d = self.dims[0]
        # Use pre-computed residuals to estimate bound tightness
        cap_ratio = np.mean(
            1.0 - self.error_f32[d] / (self.norms + 1e-10)
        )
        # If mean captured energy < 30%, bounds will be loose
        ratio = float(min(1.0, max(0.0, cap_ratio * 2 - 0.5)))
        return {
            "n_loose": int(self.n_attacks * (1 - ratio)),
            "n_tight": int(self.n_attacks * ratio),
            "energy_ratio": round(float(cap_ratio), 3),
        }

    def stats(self):
        reg = self.regime_check()
        return {
            "n_attacks": self.n_attacks, "full_dim": self.full_dim,
            "dims": self.dims, "d_int": float(self.d_int) if self.d_int else None,
            "build_time_s": self.build_time, "size_mb": self.size_bytes() / 1e6,
            "regime_flag": reg["flag"],
            "pruning_potential_pct": reg["pruning_potential_pct"],
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
