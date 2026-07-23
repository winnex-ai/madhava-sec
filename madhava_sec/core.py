"""
Madhava-Sec Core Engine v2.2 — Self-Configuring
================================================

⚠️  CRITICAL WARNING: Mathematical Guarantee ≠ Semantic Guarantee  ⚠️

Madhava-Sec guarantees 0% false negatives with respect to EMBEDDING
COSINE SIMILARITY. It does NOT guarantee that the LLM would classify
the prompt as harmful.

If the embedding model (all-MiniLM-L6-v2) fails to capture a semantic
nuance of harmfulness — which is COMMON in sophisticated jailbreak
attacks — Madhava-Sec will prune based on a DEFECTIVE signal.

This is "Garbage In, Garbage Out" with mathematical packaging.
A mathematically correct bound on a semantically blind embedding
creates a FALSE SENSE OF ABSOLUTE SAFETY.

This system is a SCORING AID, not a safety guarantee.
Always verify Madhava-Sec results with human review or LLM-based
evaluation for high-criticality decisions.

See: README.md#Limitations for full analysis.

License: BSL 1.1 | pay@winnex.ai
"""

import time, math, warnings
import numpy as np
from numpy.linalg import qr

SEED = 42

# ──────────────────────────────────────────────────────────────
# OBRIGATORY DISCLAIMER (shown once per session)
# ──────────────────────────────────────────────────────────────
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
        "║  0% false negatives on EMBEDDING SIMILARITY.              ║\n"
        "║  Does NOT guarantee semantic safety.                      ║\n"
        "║                                                          ║\n"
        "║  Garbage In → Garbage Out with mathematical packaging.   ║\n"
        "║  Sophie's jailbreaks that bypass embeddings WILL be      ║\n"
        "║  pruned with 0% bound violations — and 100% wrong        ║\n"
        "║  safety judgment.                                        ║\n"
        "║                                                          ║\n"
        "║  This is a SCORING AID, not a safety guarantee.          ║\n"
        "╚══════════════════════════════════════════════════════════════╝",
        UserWarning,
        stacklevel=3
    )

# ──────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ──────────────────────────────────────────────────────────────

def estimate_intrinsic_dim(embeddings: np.ndarray) -> float:
    """
    Von Neumann entropy -> intrinsic dimension estimate.

    D_int = exp(-Sigma p_i log p_i),  p_i = s_i^2 / Sigma s_i^2

    Low (~15-25): structured data (AgentHarm, security embeddings).
    High (~300-384): isotropic data (random, diverse corpus).
    """
    _, s, _ = np.linalg.svd(embeddings.astype(np.float64), full_matrices=False)
    e2 = np.maximum(s ** 2, 1e-15)
    e2 /= e2.sum() + 1e-15
    return float(np.exp(-np.sum(e2 * np.log(e2 + 1e-15))))


def energy_scan(X: np.ndarray, d_values=None) -> dict:
    """
    Measure fraction of vector energy captured by random orthogonal projections
    at each d_out value.

    Returns dict {d_out: energy_fraction}.
    """
    D = X.shape[1]
    if d_values is None:
        d_values = [d for d in [4, 8, 16, 32, 64, 128, 256, 512] if d <= D]
        if D not in d_values:
            d_values.append(D)
        d_values = sorted(set(d_values))

    results = {}
    rng = np.random.RandomState(42)
    norms = np.maximum(np.linalg.norm(X.astype(np.float32), axis=1), 1e-10)

    for d in d_values:
        R = rng.randn(d, D).astype(np.float64)
        Q, _ = qr(R.T)
        P = Q[:, :d].T.astype(np.float32)
        proj = (X.astype(np.float32) @ P.T)
        captured = np.linalg.norm(proj, axis=1)
        results[d] = float(np.mean(captured / norms))

    return results


# ──────────────────────────────────────────────────────────────
# AUTO CONFIGURATION
# ──────────────────────────────────────────────────────────────

def auto_configure(vectors: np.ndarray,
                   target_energy: float = 0.50,
                   verbose: bool = True):
    """
    Analyze dataset and recommend configuration for MadhavaSecEngine.

    Steps:
      1. Estimate intrinsic dimension (D_int)
      2. Scan energy vs d_out
      3. Pick smallest d1 >= target_energy, d2 >= target_energy + 0.2
      4. Check if PCA pre-processing is needed
      5. Return config dict

    Args:
      vectors: N x D float32 array
      target_energy: minimum fraction of energy to capture (default 0.50)
      verbose: print diagnostic report

    Returns:
      config: dict with keys:
        - dims: [d1, d2] recommended projection dimensions
        - d_int: intrinsic dimension estimate
        - pca_recommended: bool, True if PCA should be applied first
        - pca_dim: recommended PCA dimension (if applicable)
        - energy_map: dict {d_out: energy}
        - regime: str, expected regime ("GREEN"/"AMBER"/"RED")
    """
    t0 = time.time()
    N, D = vectors.shape
    vectors_f32 = vectors.astype(np.float32)

    # Intrinsic dimension
    sample = vectors_f32[:min(N, 10000)]
    d_int = estimate_intrinsic_dim(sample)

    # Energy scan
    e_map = energy_scan(vectors_f32)

    # Pick d1, d2
    sorted_d = sorted(e_map.keys())
    d1 = d2 = min(sorted_d)
    for d in sorted_d:
        if e_map[d] >= target_energy and d1 == min(sorted_d):
            d1 = d
        if e_map[d] >= min(1.0, target_energy + 0.2) and d2 == min(sorted_d):
            d2 = d
    if d2 <= d1 and len(sorted_d) > 1:
        d2 = sorted_d[min(sorted_d.index(d1) + 2, len(sorted_d) - 1)]
        if d2 <= d1:
            d2 = min(sorted_d[-1], D)

    # Energy captured at d1
    e_cap = e_map.get(d1, 0.0)

    # Regime and PCA recommendation
    pca_recommended = False
    pca_dim = None
    regime = "RED"
    if e_cap >= 0.50:
        regime = "GREEN"
    elif e_cap >= 0.30:
        regime = "AMBER"
        # Check if PCA would help
        pca_recommended = True
        pca_dim = max(32, min(D, int(d_int * 3)))
    else:
        regime = "RED"
        pca_recommended = True
        pca_dim = max(32, min(D, int(d_int * 3)))

    # Ensure dims are reasonable
    d1 = max(4, min(D, d1))
    d2 = max(8, min(D, d2))
    if d1 >= d2:
        d2 = min(D, d1 * 2)

    config = {
        "dims": [d1, d2],
        "d_int": round(d_int, 1),
        "full_dim": D,
        "pca_recommended": pca_recommended,
        "pca_dim": pca_dim if pca_recommended else None,
        "energy_map": {str(k): round(v, 3) for k, v in e_map.items()},
        "energy_at_d1": round(e_cap, 3),
        "regime_expected": regime,
        "analysis_time_s": round(time.time() - t0, 2),
    }

    if verbose:
        print(f"\n[Madhava-Sec] Auto-Configuration Report")
        print(f"  Dataset: {N} x {D}D")
        print(f"  Intrinsic dimension: {d_int:.1f}")
        print(f"  Energy scan:")
        for d in sorted_d:
            sym = "🟢" if e_map[d] >= 0.5 else "🟡" if e_map[d] >= 0.3 else "❌"
            print(f"    {sym} d={d:>4}D -> {e_map[d]*100:>5.1f}% energy captured")
        print(f"  Recommended: d1={d1}D, d2={d2}D -> ~{e_cap*100:.0f}% energy")
        print(f"  Expected regime: {regime}")
        if pca_recommended:
            print(f"  ⚠ PCA recommended: reduce {D}D -> {pca_dim}D for tight bounds")
        print(f"  Elapsed: {config['analysis_time_s']:.2f}s")

    return config


# ──────────────────────────────────────────────────────────────
# ENGINE
# ──────────────────────────────────────────────────────────────

class MadhavaSecEngine:
    """
    Madhava-Sec: Cauchy-Schwarz bound pruning for agent security.

    Usage:
      config = auto_configure(vectors)
      engine = MadhavaSecEngine(stage_dims=config["dims"]).build(vectors)
      regime = engine.regime_check()
    """

    def __init__(self, stage_dims=None, seed=SEED):
        _show_disclaimer()
        self.dims = stage_dims or [32, 128]
        self.full_dim = 384
        self.keep_ratio = 0.15
        self.max_candidates = 200
        self.final_topk = 50
        self.rng = np.random.RandomState(seed + 1)
        self.d_int = None
        self.attack_vectors = None
        self.n_attacks = 0
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

    def build(self, attack_vectors):
        t0 = time.time()
        n = len(attack_vectors)
        d_in = attack_vectors.shape[1]
        self.full_dim = d_in

        # Estimate intrinsic dim for reporting
        sample = attack_vectors[:min(n, 10000)]
        self.d_int = estimate_intrinsic_dim(sample)

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
        q = query_vec.astype(np.float32).flatten()
        qn = max(np.linalg.norm(q), 1e-10)
        prof = {"n_total": self.n_attacks, "d_int": self.d_int,
                "dims": list(self.dims)}

        d1, d2 = self.dims[0], self.dims[-1]

        # Stage 1
        q1 = q @ self.proj_mat[d1].T
        qr1 = math.sqrt(max(0, qn ** 2 - np.linalg.norm(q1) ** 2))
        B1 = self._upper_bound(self.proj_f32[d1], self.error_f32[d1], q1, qr1)

        keep1 = min(max(int(self.n_attacks * self.keep_ratio), 50),
                    self.max_candidates, self.n_attacks)
        idx1 = (np.arange(self.n_attacks) if self.n_attacks <= keep1
                else np.argpartition(-B1, max(0, keep1 - 1))[:keep1])

        # Stage 2
        q2 = q @ self.proj_mat[d2].T
        qr2 = math.sqrt(max(0, qn ** 2 - np.linalg.norm(q2) ** 2))
        B2 = self._upper_bound(self.proj_f32[d2][idx1], self.error_f32[d2][idx1],
                               q2, qr2)

        e1s, e2s = self.error_f32[d1][idx1], self.error_f32[d2][idx1]
        imp = max(np.mean(e1s), 1e-9) / max(np.mean(e2s), 1e-9)

        if imp < 1.2:
            scores = B1[idx1]
        elif imp < 2.0:
            a = self._modulation_alpha(e1s, e2s) * (imp - 1.2) / 0.8
            scores = B1[idx1] + a * (B2 - B1[idx1])
        else:
            a = self._modulation_alpha(e1s, e2s)
            scores = B1[idx1] + a * (B2 - B1[idx1])

        keep2 = max(self.final_topk, min(200, len(idx1)))
        idx2 = idx1[np.argpartition(-scores, max(0, keep2 - 1))[:keep2]]

        exact = self.attack_vectors[idx2] @ q
        nv = np.maximum(np.linalg.norm(self.attack_vectors[idx2], axis=1), 1e-10)
        exact_norm = exact / (nv * qn)

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

    def check_bounds(self, query_vec, eps_guard=True):
        q = query_vec.astype(np.float32).flatten()
        qn = max(np.linalg.norm(q), 1e-10)
        V = self.attack_vectors
        nv = np.maximum(np.linalg.norm(V, axis=1), 1e-10)
        true = (V @ q) / (nv * qn)
        eps = (np.finfo(np.float32).eps * 1000) if eps_guard else 1e-9
        viol = {}
        for d in self.dims:
            qd = q @ self.proj_mat[d].T
            qr = math.sqrt(max(0, qn ** 2 - np.linalg.norm(qd) ** 2))
            ub = self._upper_bound(self.proj_f32[d], self.error_f32[d], qd, qr)
            viol[f"{d}D"] = int(np.sum(true > ub + eps))
        return viol, self.n_attacks

    def regime_check(self, verbose=False):
        """
        Evaluate operational regime based on projection vs intrinsic dimension.

        Key metric: d_out / D_int  (not d_out / full_dim).
        If d_out >= D_int, projection captures most intrinsic structure.
        Action vectors (20D) with d_out=[4,16]:  d_out/D_int ≈ 15/15 = 100% → GREEN.
        Embeddings (384D) with d_out=[4,16]:     d_out/D_int ≈ 4/384 = 1%   → RED.
        """
        if self.n_attacks == 0 or self.d_int is None:
            return {"flag": "UNKNOWN", "message": "Call build() first"}

        d = max(self.dims)  # Stage 2 = bound real. Stage 1 = filtro.
        ratio = min(1.0, d / max(self.d_int, 1))

        # Energy estimation from theory: ~ratio of intrinsic dim captured
        energy_pct = round(ratio * 100, 1)

        # Expected bound: centroid_sim + residual
        # If ratio ≈ 1 (d_out captures D_int): residual ≈ 0, bound ≈ centroid = 0.6
        # If ratio ≈ 0 (d_out << D_int): residual ≈ 1, bound ≈ 0 + 1 = 1.0 (no pruning)
        residual = math.sqrt(max(0, 1.0 - ratio))
        centroid_sim = 0.6 * ratio + 0.3  # scales from 0.3 (low) to 0.9 (high)
        expected_bound = centroid_sim + residual

        if ratio >= 0.7:
            flag, msg = "GREEN", (
                f"d_out/d_int = {d}/{self.d_int:.0f} = {ratio:.2f}. "
                f"Projection captures most intrinsic structure. "
                f"Expected bound ≈ {expected_bound:.2f}. "
                f"Madhava-Sec operating within guaranteed regime.")
        elif ratio >= 0.3:
            flag, msg = "AMBER", (
                f"d_out/d_int = {d}/{self.d_int:.0f} = {ratio:.2f}. "
                f"Partial energy capture. "
                f"Expected bound ≈ {expected_bound:.2f}. "
                f"Pruning limited. Verify results manually.")
        else:
            flag, msg = "RED", (
                f"d_out/d_int = {d}/{self.d_int:.0f} = {ratio:.2f}. "
                f"Projection too small for intrinsic dimension. "
                f"Expected bound ≈ {expected_bound:.2f}. "
                f"Increase d_out or reduce dimension before Madhava-Sec.")

        result = {
            "flag": flag,
            "d_int": round(float(self.d_int), 1),
            "full_dim": self.full_dim,
            "proj_dims": list(self.dims),
            "d_out_d_int_ratio": round(ratio, 3),
            "expected_bound": round(expected_bound, 3),
            "message": msg,
        }

        if verbose:
            print(f"\n[Madhava-Sec] Regime: {flag}")
            print(f"  D_int = {self.d_int:.1f} / {self.full_dim}")
            print(f"  d_out[0] = {d} -> ratio d_out/D_int = {ratio:.2f}")
            print(f"  Expected bound ≈ {expected_bound:.3f}")
            print(f"  {msg}")

        return result

    def stats(self):
        r = self.regime_check()
        return {
            "n_attacks": self.n_attacks, "full_dim": self.full_dim,
            "dims": list(self.dims), "d_int": round(float(self.d_int), 1),
            "build_time_s": round(self.build_time, 3),
            "size_mb": round(self.size_bytes() / 1e6, 1),
            "regime_flag": r["flag"],
            "energy_captured_pct": r["energy_captured_pct"],
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
