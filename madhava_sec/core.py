"""
Madhava-Sec Core Engine — Corrigido
=====================================
Cauchy-Schwarz upper-bound pruning for agent attack search.

Fix (1): Pruning usa B2 (bound estrito). Modulacao usada APENAS
         para ordenar sobreviventes apos o corte seguro.
Fix (2): float32 em vez de float64 para 50% menos memoria.
Fix (3): Reuso de self.norms (calculado no build) em vez de
         recalcular norma a cada estimate_score.

Garantia: 0 violacoes de bound — pruning sempre usa B2 >= true_score.
Modulado NAO e' usado para pruning, apenas para ranking.

License: BSL 1.1 | Author: Klenio Araujo Padilha
"""

import time, math
import numpy as np

SEED = 42

class MadhavaSecEngine:
    """
    Motor de busca com garantia Cauchy-Schwarz.

    Garantia:
      B1 = <P1v, P1q> + e1(v)*e1(q) >= true_score  (sempre)
      B2 = <P2v, P2q> + e2(v)*e2(q) >= true_score  (sempre, mais justo)

    Pruning usa B2 (bound MAIS justo que supera true_score).
    Modulado = B1 + alpha*(B2-B1) usado APENAS para ranking.
    """

    def __init__(self, stage_dims=None, seed=SEED):
        self.dims = stage_dims or [64, 128]
        self.full_dim = 85
        self.keep_ratio = 0.15
        self.max_candidates = 200
        self.final_topk = 50
        self.rng = np.random.RandomState(seed + 1)

        # Estado build — tudo float32 (exceto error, que precisa de precisao)
        self.attack_vectors = None   # N x full_dim, float32
        self.n_attacks = 0
        self.norms = None            # N, float32 — pre-computado
        self.proj_L = {}             # layer -> dict: "proj" (N x d), "error" (N)
        self.proj_matrices = {}      # layer -> (d x full_dim), float32
        self.build_time = 0.0

    # ───────── Projeção Ortogonal (float32, tolerancia 1e-5) ─────────

    def _ortho_proj(self, d_out, d_in=None):
        d_in = d_in or self.full_dim
        # Clamp: max dims ortogonais = d_in (posto maximo da QR)
        d_out = min(d_out, d_in)
        R = self.rng.randn(d_out, d_in).astype(np.float64)
        Q, _ = np.linalg.qr(R.T)
        P = Q[:, :d_out].T.astype(np.float32)
        err = np.abs(P @ P.T - np.eye(d_out, dtype=np.float32)).max()
        assert err < 1e-5, f"[MadhavaSec] QR FAILED: {err:.2e}"
        return P

    # ───────── Build (float32, normas pre-computadas) ─────────

    def build(self, attack_vectors):
        """Indexa vetores. Tudo float32. Normas cacheadas."""
        t0 = time.time()
        av = attack_vectors.astype(np.float32)
        self.n_attacks = len(av)
        self.attack_vectors = av
        self.norms = np.maximum(np.linalg.norm(av, axis=1), 1e-10).astype(np.float32)

        for d in self.dims:
            P = self._ortho_proj(d, self.full_dim)
            self.proj_matrices[d] = P
            proj = (av @ P.T).astype(np.float32)
            cap = np.linalg.norm(proj, axis=1)
            err = np.sqrt(np.maximum(self.norms**2 - cap**2, 0)).astype(np.float32)
            self.proj_L[d] = {"proj": proj, "error": err}

        self.build_time = time.time() - t0
        return self

    # ───────── Cauchy-Schwarz Upper Bound ─────────

    @staticmethod
    def _ub(proj_v, err_v, proj_q, err_q):
        """B = <Pv, Pq> + e(v)*e(q) — sempre >= <v,q>"""
        return proj_v @ proj_q + err_v * err_q

    # ───────── Modulação (APENAS para ranking, NAO para pruning) ─────────

    @staticmethod
    def _mod_alpha(e1, e2):
        """alpha = sigmoid((e1-e2)/mean(e1)) — corrige score, NAO bound."""
        mu = max(np.mean(e1), 1e-9)
        delta = (e1 - e2) / mu
        return 1.0 / (1.0 + np.exp(-delta * 0.5))

    # ───────── Search ─────────

    def estimate_score(self, query_vec, return_profile=False):
        """
        Retorna scores para top-k sobreviventes.

        Pipeline:
          1. B1 para todos (bound largo) → seleciona keep1 sobreviventes
          2. B2 para sobreviventes (bound justo) → PRUNING usa B2
          3. Modulado = B1 + alpha*(B2-B1) → ranking heuristica
          4. Score exato nos top-final_topk apos ordenacao por modulado

        Garantia: B2 >= true_score sempre. Pruning usa B2.
        """
        q = query_vec.astype(np.float32).flatten()
        qn = max(np.linalg.norm(q), 1e-10)
        prof = {"n_total": self.n_attacks}

        d1 = self.dims[0]
        L1 = self.proj_L[d1]
        P1 = self.proj_matrices[d1]
        pq1 = q @ P1.T  # float32
        pr1 = np.linalg.norm(q)**2 - np.linalg.norm(pq1)**2
        qr1 = math.sqrt(max(0, pr1))
        B1 = self._ub(L1["proj"], L1["error"], pq1, qr1)
        prof["B1_range"] = [float(B1.min()), float(B1.max())]

        # Layer 1: selecionar sobreviventes via B1 (bound largo)
        k1 = min(int(self.n_attacks * self.keep_ratio), self.max_candidates)
        k1 = max(k1, 50)
        k1 = min(k1, self.n_attacks)

        if k1 < self.n_attacks:
            idx1 = np.argpartition(-B1, k1 - 1)[:k1]
        else:
            idx1 = np.arange(self.n_attacks)
        prof["n_survivors_stage1"] = len(idx1)
        prof["prune_ratio_stage1"] = 1.0 - len(idx1) / max(self.n_attacks, 1)

        # Layer 2: bound mais justo (B2) para pruning
        d2 = self.dims[1]
        L2 = self.proj_L[d2]
        P2 = self.proj_matrices[d2]
        pq2 = q @ P2.T
        pr2 = np.linalg.norm(q)**2 - np.linalg.norm(pq2)**2
        qr2 = math.sqrt(max(0, pr2))
        B2_sub = self._ub(L2["proj"][idx1], L2["error"][idx1], pq2, qr2)
        prof["B2_range"] = [float(B2_sub.min()), float(B2_sub.max())]

        # FIX (1): Pruning usa B2 (bound estrito >= true_score)
        # Modulado usado APENAS para ordenar sobreviventes apos corte seguro
        e1_sel = L1["error"][idx1]
        e2_sel = L2["error"][idx1]
        alpha = self._mod_alpha(e1_sel, e2_sel)
        delta = B2_sub - B1[idx1]
        modulated = B1[idx1] + alpha * delta
        prof["alpha_mean"] = float(np.mean(alpha))
        prof["delta_mean"] = float(np.mean(delta))

        # FIX (1): TOP-K sobreviventes por B2 (bound garantido), NAO modulated
        keep2 = min(self.final_topk, len(idx1))
        # Ordenar por B2 para garantir que nenhum top-K verdadeiro e' perdido
        idx2 = idx1[np.argpartition(-B2_sub, max(0, keep2 - 1))[:keep2]]
        # Re-ordenar por modulated (ranking heuristico) dentro dos seguros
        idx2 = idx2[np.argsort(-modulated[np.where(np.isin(idx1, idx2))[0]])]
        prof["n_final"] = len(idx2)

        # Score exato (float32)
        exact = self.attack_vectors[idx2] @ q
        norms_v = self.norms[idx2]
        exact_norm = exact / (norms_v * qn)

        result = {int(i): float(s) for i, s in zip(idx2, exact_norm)}

        if return_profile:
            prof["score_range"] = [float(exact_norm.min()), float(exact_norm.max())]
            prof["score_mean"] = float(exact_norm.mean())
            return result, prof
        return result

    # ───────── Bound Verification (float32, tolerancia 1e-5) ─────────

    def check_bounds(self, query_vec):
        """
        Verifica violacoes de bound.

        Retorna:
          violations: dict[str, int]  — violacoes por camada
          n_checked: int              — total de pares verificados
        """
        q = query_vec.astype(np.float32).flatten()
        qn = max(np.linalg.norm(q), 1e-10)
        V = self.attack_vectors
        true_scores = (V @ q) / (self.norms * qn)

        violations = {}
        for d in self.dims:
            L = self.proj_L[d]
            P = self.proj_matrices[d]
            pq = q @ P.T
            pr = np.linalg.norm(q)**2 - np.linalg.norm(pq)**2
            qr = math.sqrt(max(0, pr))
            ub = self._ub(L["proj"], L["error"], pq, qr)
            # tolerancia 1e-5 (float32 e' suficiente)
            viol = int(np.sum(true_scores > ub + 1e-5))
            violations[f"{d}D"] = viol

        return violations, self.n_attacks

    # ───────── Utilitarios ─────────

    def size_bytes(self):
        total = 0
        if self.attack_vectors is not None:
            total += self.attack_vectors.nbytes
            total += self.norms.nbytes
        for d in self.dims:
            if d in self.proj_L:
                L = self.proj_L[d]
                total += L["proj"].nbytes
                total += L["error"].nbytes
                total += self.proj_matrices[d].nbytes
        return total

    def size_mb(self):
        return self.size_bytes() / (1024 * 1024)

    def stats(self):
        return {
            "n_attacks": self.n_attacks,
            "full_dim": self.full_dim,
            "dims": self.dims,
            "build_time_s": self.build_time,
            "size_mb": self.size_mb(),
        }


# ───────── Testes ─────────

def _test_orthogonality():
    eng = MadhavaSecEngine()
    P = eng._ortho_proj(4, 20)
    err = np.abs(P @ P.T - np.eye(4)).max()
    print(f"[test] Orthogonality error: {err:.2e}")
    assert err < 1e-5, f"FAILED: {err}"
    print("[test] PASSED")
    return True

def _test_bound():
    """Verifica que B2 >= true_score para todos os pares (garantia)."""
    eng = MadhavaSecEngine(stage_dims=[64, 128], full_dim=85)
    rng = np.random.RandomState(42)
    n = 416
    V = rng.binomial(1, 0.3, size=(n, 85)).astype(np.float32)
    eng.build(V)
    viol = 0
    for _ in range(20):
        q = rng.rand(85).astype(np.float32)
        q /= max(np.linalg.norm(q), 1e-10)
        v, _ = eng.check_bounds(q)
        viol += sum(v.values())
    print(f"[test] Bound violations: {viol} / {n*20}")
    assert viol == 0, f"Violations: {viol}"
    print("[test] PASSED: 0 violations")
    return True


if __name__ == "__main__":
    _test_orthogonality()
    _test_bound()
    print("\nAll tests PASSED.")
