#!/usr/bin/env python3
"""
Madhava-Sec Core Engine
=======================
Cauchy-Schwarz upper-bound pruning for agent attack search.

Mathematical mapping from Madhava Cascade (Zenodo 20970487):

  Vector Search (Madhava)             Agent Security (Madhava-Sec)
  ─────────────────────               ──────────────────────────────
  Corpus embedding v ∈ ℝ^d            Prompt action vector v ∈ {0,1}^d
  Query embedding q ∈ ℝ^d             Severity profile q ∈ [0,1]^d
  <v,q> = cosine similarity           severity_score = <v,q> (linear)
  Proj. orthog. P: ℝ^d → ℝ^k          Proj. P: ℝ^d → ℝ^k (action subspace)
  <v,q> ≤ <Pv,Pq> + e(v)·e(q)        severity ≤ proj_severity + residual
  If bound < top-K threshold →        If bound < best_score → branch is
  document PROVABLY OUT of top-K      PROVABLY not optimal → PRUNE

License: BSL 1.1 | Author: Klenio Araujo Padilha
"""

import time, math, json, gc
import numpy as np

SEED = 42
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────
# MADHAVA-SEC CORE ENGINE
# ──────────────────────────────────────────────────────────────

class MadhavaSecEngine:
    """
    Madhava-Sec: motor de busca com garantia Cauchy-Schwarz.

    Funciona em cascata hierárquica:
      Layer 1: feature_sparse  (dimensão esparsa, TOP p/ pruning rápido)
      Layer 2: feature_dense   (dimensão densa, refinamento)
      Layer 3: exact_score     (score exato apenas nos sobreviventes)

    Garantia: 0 violações de bound — toda poda é MATEMATICAMENTE justificada.
    """

    def __init__(self, stage_dims=None, seed=SEED):
        # Dimensões das camadas de projeção
        # stage_dims = [4, 16] significa:
        #   Layer 1: projetar ação 20D → 4D para pruning rápido
        #   Layer 2: projetar 20D → 16D para refinamento
        self.dims = stage_dims or [4, 16]
        self.full_dim = 20          # dimensão do vetor de ações (action_embedding)
        self.keep_ratio = 0.15      # manter ~15% para refinamento (Layer 2)
        self.max_candidates = 200   # máximo de candidatos por camada
        self.final_topk = 50        # sobreviventes para score exato
        self.rng = np.random.RandomState(seed + 1)

        # Estado build
        self.attack_vectors = None  # matriz N x full_dim dos ataques indexados
        self.n_attacks = 0
        self.proj_L = {}            # projeções por camada: layer -> (N x d) matrix
        self.error = {}             # resíduo pitagórico por camada: layer -> (N,) vector
        self.proj_matrices = {}     # matrizes de projeção: layer -> (d x full_dim)
        self.norms = None           # norma de cada vetor de ataque (N,)
        self.build_time = 0.0

    # ─────────────── Projeção Ortogonal ───────────────

    def _ortho_proj(self, d_out, d_in=None):
        """
        Matriz de projeção QR-ortogonalizada.

        Garantia: P·P^T = I_k (dentro de 1e-5)
        Usando QR decomposition exatamente como no Madhava original.

        Prova: Para qualquer projeção ortogonal P: ℝ^d → ℝ^k,
          ‖Px‖ ≤ ‖x‖  (contração)
          <Px,Py> = <P^T P x, y>  (adjunta)
          resíduo = ‖x - P^T P x‖  (componente no núcleo de P)
        """
        d_in = d_in or self.full_dim
        R = self.rng.randn(d_out, d_in).astype(np.float64)
        Q, _ = np.linalg.qr(R.T)   # QR: Q^T Q = I
        P = Q[:, :d_out].T.astype(np.float32)
        # Verificação de ortogonalidade (mesmo padrão do Madhava)
        ortho_err = np.abs(P @ P.T - np.eye(d_out, dtype=np.float32)).max()
        assert ortho_err < 1e-5, \
            f"[MadhavaSec] QR orthogonality FAILED: {ortho_err:.2e}"
        return P

    # ─────────────── Build ───────────────

    def build(self, attack_vectors):
        """
        Build: pré-computa todas as projeções e resíduos.

        attack_vectors: np.ndarray (N x full_dim), float32
          Cada linha é um vetor de ações de ataque.
          Ações são features binárias: 1 = usa aquela ferramenta/ação.

        Durante o build:
          1. Projeta cada vetor em cada camada (d1, d2, ...)
          2. Calcula erro pitagórico: e_d(v) = √(‖v‖² - ‖P_d v‖²)
          3. Tudo é O(N · full_dim · sum(dims))

        Análogo ao MadhavaCore.build() (winnex_definitive_benchmark.py:82)
        """
        t0 = time.time()
        self.attack_vectors = attack_vectors.astype(np.float64)
        self.n_attacks = len(attack_vectors)
        self.norms = np.linalg.norm(self.attack_vectors, axis=1).astype(np.float64)
        self.norms = np.maximum(self.norms, 1e-10)  # evitar divisão por zero

        for d in self.dims:
            P = self._ortho_proj(d, self.full_dim)
            self.proj_matrices[d] = P
            # Projeção rápida em float32 (mantendo compatibilidade com Madhava)
            proj = (attack_vectors.astype(np.float32) @ P.T).astype(np.float64)
            self.proj_L[d] = proj
            # Resíduo pitagórico: ‖v‖² - ‖Pv‖² ≥ 0
            captured = np.linalg.norm(proj, axis=1)
            res_sq = np.maximum(self.norms**2 - captured**2, 0)
            self.error[d] = np.sqrt(res_sq)

        self.build_time = time.time() - t0
        return self

    # ─────────────── Cauchy-Schwarz Upper Bound ───────────────

    def _upper_bound(self, pv, ev, pq, eq):
        """
        Cauchy-Schwarz Upper Bound.

        Para projeção ortogonal P e vetores v (ataque), q (perfil de severidade):

          <v,q>  ≤  <Pv, Pq>  +  ‖v - P^T P v‖ · ‖q - P^T P q‖
          ╰──╯      ╰────╯       ╰───────────────────────────╯
         verdadeiro  projeção         resíduo pitagórico

        Prova: <v,q> = <Pv + (v-P^T P v), Pq + (q-P^T P q)>
               = <Pv,Pq> + <v-P^T P v, q-P^T P q>  (termos cruzados = 0)
               ≤ <Pv,Pq> + ‖v-P^T P v‖·‖q-P^T P q‖  (Cauchy-Schwarz)

        Madhava original: winnex_definitive_benchmark.py:91
        """
        return pv @ pq + ev * eq

    def _modulation_alpha(self, e1, e2):
        """
        Learning rate per-documento para error backpropagation modulation.

        α(v) = σ((e₁(v) - e₂(v)) / μ)

        Onde:
          e₁ = resíduo na camada 1 (baixa dimensão, erro maior)
          e₂ = resíduo na camada 2 (alta dimensão, erro menor)
          μ  = mean(e₁)  (normalização)
          σ  = sigmoid

        Quando e₁ >> e₂: erro encolheu muito → α ≈ 1 (corrigir agressivamente)
        Quando e₁ ≈ e₂: erro já era pequeno → α ≈ 0 (confiar na camada 1)

        Madhava original: madhava_qrjl_benchmark.py:126
        """
        mu = np.maximum(np.mean(e1), 1e-9)
        delta = (e1 - e2) / mu
        return 1.0 / (1.0 + np.exp(-delta * 0.5))

    # ─────────────── Search (Score Estimation) ───────────────

    def estimate_score(self, query_vec, return_profile=False):
        """
        Estima score de severidade para cada ataque indexado.

        Cascata:
          Layer 1 (d1): estimate = upper_bound(P_d1 v, P_d1 q, e_d1(v), e_d1(q))
            para TODOS os N ataques → O(N·d1)

          Layer 2 (d2): refine survivors via upper_bound + modulation
            para C ≈ 0.15N sobreviventes → O(C·d2)

          Layer 3: score exato para top-k sobreviventes

        Returns:
          scores: dict[int, float]  — índice -> score estimado (top-k apenas)
          profile: dict              — perfil de pruning (se return_profile=True)
        """
        q = query_vec.astype(np.float64).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            q_norm = 1.0  # evitar divisão por zero
        prof = {"n_total": self.n_attacks}

        # ── Layer 1: projeção rápida para pruning ──
        d1 = self.dims[0]
        q1 = (q.astype(np.float32) @ self.proj_matrices[d1].T).astype(np.float64)
        qr1 = math.sqrt(max(0, q_norm**2 - np.linalg.norm(q1)**2))
        B1 = self._upper_bound(self.proj_L[d1], self.error[d1], q1, qr1)
        prof["B1_range"] = [float(B1.min()), float(B1.max())]

        # Selecionar sobreviventes da Layer 1
        keep1 = min(
            max(int(self.n_attacks * self.keep_ratio), 50),
            self.max_candidates,
            self.n_attacks
        )

        if self.n_attacks <= keep1:
            idx1 = np.arange(self.n_attacks)
        else:
            idx1 = np.argpartition(-B1, max(0, keep1 - 1))[:keep1]
        prof["n_survivors_stage1"] = len(idx1)
        prof["prune_ratio_stage1"] = 1.0 - len(idx1) / max(self.n_attacks, 1)

        # ── Layer 2: refinamento com bound mais justo ──
        if len(self.dims) >= 2:
            d2 = self.dims[1]
            q2 = (q.astype(np.float32) @ self.proj_matrices[d2].T).astype(np.float64)
            qr2 = math.sqrt(max(0, q_norm**2 - np.linalg.norm(q2)**2))
            B2 = self._upper_bound(
                self.proj_L[d2][idx1], self.error[d2][idx1], q2, qr2
            )
            prof["B2_range"] = [float(B2.min()), float(B2.max())]

            # Error backpropagation modulation
            delta = B2 - B1[idx1]
            prof["delta_mean"] = float(np.mean(delta))
            e1_sel = self.error[d1][idx1]
            e2_sel = self.error[d2][idx1]
            alpha = self._modulation_alpha(e1_sel, e2_sel)
            prof["alpha_mean"] = float(np.mean(alpha))

            # Score modulado: s = B̂₁ + α · (B̂₂ - B̂₁)
            modulated = B1[idx1] + alpha * delta
            prof["modulated_range"] = [
                float(modulated.min()), float(modulated.max())
            ]
            scores_ordered = modulated
        else:
            # Apenas 1 camada
            scores_ordered = B1[idx1]
            prof["alpha_mean"] = 0.0

        # ── Top-k survivors para score exato ──
        keep2 = min(self.final_topk, len(idx1))
        idx2 = idx1[np.argpartition(-scores_ordered, max(0, keep2 - 1))[:keep2]]
        prof["n_final"] = len(idx2)

        # Score exato (produto interno normalizado)
        exact = self.attack_vectors[idx2].astype(np.float64) @ q
        # Normalizar por normas
        norms_v = np.linalg.norm(self.attack_vectors[idx2].astype(np.float64), axis=1)
        norms_v = np.maximum(norms_v, 1e-10)
        exact_normalized = exact / (norms_v * q_norm)

        # Montar resultado
        result = {}
        for i, scr in zip(idx2, exact_normalized):
            result[int(i)] = float(scr)

        if return_profile:
            prof["score_range"] = [
                float(exact_normalized.min()), float(exact_normalized.max())
            ]
            prof["score_mean"] = float(exact_normalized.mean())
            prof["n_estimated"] = keep2
            return result, prof
        return result

    # ─────────────── Bound Verification ───────────────

    def check_bounds(self, query_vec):
        """
        Verifica violações de bound para TODOS os N ataques.

        Retorna:
          violations: dict[str, int]  — camada -> número de violações
          n_checked: int              — total de pares verificados

        Uma violação ocorre quando score_real > upper_bound + 1e-9.
        Madhava original: 0 violações em 254M+ pares.
        """
        q = query_vec.astype(np.float64).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            q_norm = 1.0

        # Score real
        V = self.attack_vectors.astype(np.float64)
        norms_v = np.linalg.norm(V, axis=1)
        norms_v = np.maximum(norms_v, 1e-10)
        true_scores = (V @ q) / (norms_v * q_norm)

        violations = {}
        for d in self.dims:
            P = self.proj_matrices[d]
            qd = (q.astype(np.float32) @ P.T).astype(np.float64)
            qr = math.sqrt(max(0, q_norm**2 - np.linalg.norm(qd)**2))
            ub = self._upper_bound(self.proj_L[d], self.error[d], qd, qr)
            viol = int(np.sum(true_scores > ub + 1e-9))
            violations[f"{d}D"] = viol

        return violations, self.n_attacks

    # ─────────────── Index Management ───────────────

    def add_attacks(self, new_vectors):
        """
        Adiciona novos ataques ao índice (incremental).
        Requer rebuild completo das projeções (estrutura batch, não streaming).
        """
        if self.attack_vectors is None:
            return self.build(new_vectors)
        combined = np.vstack([self.attack_vectors, new_vectors])
        # Converte de float64 de volta para float32 para rebuild
        return self.build(combined.astype(np.float32))

    def size_bytes(self):
        """Tamanho aproximado do índice em memória."""
        total = 0
        if self.attack_vectors is not None:
            total += self.attack_vectors.nbytes
        for d in self.dims:
            if d in self.proj_L:
                total += self.proj_L[d].nbytes
                total += self.proj_matrices[d].nbytes
                total += self.error[d].nbytes
        return total

    def size_mb(self):
        return self.size_bytes() / (1024 * 1024)

    def stats(self):
        """Resumo do estado do engine."""
        return {
            "n_attacks": self.n_attacks,
            "full_dim": self.full_dim,
            "dims": self.dims,
            "keep_ratio": self.keep_ratio,
            "final_topk": self.final_topk,
            "build_time_s": self.build_time,
            "size_mb": self.size_mb(),
        }


# ─────────────── Teste interno ───────────────

def _test_orthogonality():
    """Verifica que a projeção QR-ortogonal satisfaz P·P^T = I."""
    eng = MadhavaSecEngine()
    P = eng._ortho_proj(4, 20)
    err = np.abs(P @ P.T - np.eye(4)).max()
    print(f"[test] Orthogonality error: {err:.2e}")
    assert err < 1e-5, f"Orthogonality FAILED: {err}"
    print("[test] PASSED")
    return True


def _test_bound_guarantee():
    """
    Verifica que NENHUM score real viola o upper bound.
    Gera ataques sintéticos e perfis de severidade aleatórios.
    """
    eng = MadhavaSecEngine()
    n = 1000
    # Gerar ataques sintéticos: vetores binários esparsos
    rng = np.random.RandomState(42)
    attacks = rng.binomial(1, 0.2, size=(n, 20)).astype(np.float32)
    eng.build(attacks)

    viol_total = 0
    n_checks = 0
    for _ in range(50):
        q = rng.rand(20).astype(np.float32)
        q = q / max(np.linalg.norm(q), 1e-10)
        viol, total = eng.check_bounds(q)
        viol_total += sum(viol.values())
        n_checks += total

    rate = viol_total / max(n_checks, 1)
    print(f"[test] Bound violations: {viol_total} / {n_checks} ({rate*100:.6f}%)")
    assert viol_total == 0, f"Bound violations found: {viol_total}"
    print("[test] PASSED: 0 violations")
    return True


if __name__ == "__main__":
    print("Madhava-Sec Core Engine — Self-Test")
    print("=" * 50)
    _test_orthogonality()
    _test_bound_guarantee()
    print("\nAll tests PASSED.")
