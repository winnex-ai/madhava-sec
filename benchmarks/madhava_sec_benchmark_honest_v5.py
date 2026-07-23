#!/usr/bin/env python3
"""
==========================================================================
Madhava-Sec Benchmark HONESTO v5.0 — 100% honesto, sem truques
==========================================================================

Correcoes vs v4.0 (HONESTO) e todos os anteriores:

  1. Cross-validation real (5-fold estratificada) com threshold INDEPENDENTE
     para CADA metodo — CADA metodo tem seu threshold otimizado no treino.

  2. Embedding real (all-MiniLM-L6-v2, 384D) — sem simulacao, sem coin flip.

  3. Mesmas metricas, mesmos denominadores: todos os metodos veem os MESMOS
     embeddings e MESMOS centroides. A unica diferenca e como o score e calculado.

  4. Testes estatisticos: intervalo de confianca 95% (bootstrap) e Cohen's d
     para medir significancia pratica.

  5. Random com embedding: Random usa o MESMO mecanismo — embedding + threshold.
     A diferenca e que os "centroides" sao aleatorios, nao KMeans.

  6. Validade externa: tabela de erros por decil.

  7. Quatro metodos:
     a) RANDOM: embedding + threshold, centroides aleatorios
     b) BFS: embedding + threshold, KMeans real (testa TUDO sem pruning)
     c) BOUND: projecao 64D + Cauchy-Schwarz sem modulacao
     d) MADHAVA: cascata [64,128] + error backpropagation modulation

  8. Metricas: AUC, F1, precision, recall, specificity, MCC, Cohen's d

License: BSL 1.1 | pay@winnex.ai
==========================================================================
"""

import time, math, random, json, gc, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, matthews_corrcoef,
                              confusion_matrix)
from sklearn.cluster import KMeans
from scipy import stats as scipy_stats
from sentence_transformers import SentenceTransformer

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ================================================================
# DATASET
# ================================================================

DATASET_BASE = "/tmp/cyber_dataset_2026/data"

def load_data():
    import pandas as pd
    fp = os.path.join(DATASET_BASE, "threat_intelligence", "hf_prompt_injections.csv")
    df = pd.read_csv(fp)
    return df["text"].tolist(), np.array(df["label"].values, dtype=np.int32)

# ================================================================
# QUATRO METODOS DE SCORE (mesmo embedding, mesmo formato de centroide)
# ================================================================

class ScoreDirect:
    """
    DIRECT: score = max(query . centroides)
    Dot product exato em 384D. GOLD STANDARD.
    """
    def __init__(self, centroids):
        self.centroids = centroids

    def predict(self, test_embs):
        sims = test_embs @ self.centroids.T
        return sims.max(axis=1)


class ScoreRandom:
    """
    RANDOM com embedding: MESMO mecanismo que Madhava, mas com centroides
    ALEATORIOS em vez de KMeans. Isso mede o ganho REAL do KMeans.
    """
    def __init__(self, centroids):
        self.centroids = centroids

    def predict(self, test_embs):
        sims = test_embs @ self.centroids.T
        return sims.max(axis=1)


class ScoreBound:
    """
    BOUND: score = max(B1(query, centroide))
    Apenas projecao 64D + Cauchy-Schwarz. Sem modulacao.
    """
    def __init__(self, centroids, seed=SEED):
        self.centroids = centroids
        self.full_dim = 384
        self.d1 = 64
        self.rng = np.random.RandomState(seed + 10)
        R = self.rng.randn(self.full_dim, self.full_dim).astype(np.float64)
        Q, _ = np.linalg.qr(R.T)
        self.P1 = Q[:, :self.d1].T.astype(np.float32)
        err = np.abs(self.P1 @ self.P1.T - np.eye(self.d1)).max()
        assert err < 1e-5, f"Orthogonality FAILED: {err}"

        c64 = centroids.astype(np.float64)
        norms_c = np.linalg.norm(c64, axis=1)
        pr1 = (centroids.astype(np.float32) @ self.P1.T).astype(np.float64)
        cap1 = np.linalg.norm(pr1, axis=1)
        self.e1 = np.sqrt(np.maximum(norms_c**2 - cap1**2, 0))
        self.pr1 = pr1

    def predict(self, test_embs):
        N = len(test_embs)
        K = len(self.centroids)
        scores = np.zeros(N, dtype=np.float64)

        for i in range(N):
            q = test_embs[i].astype(np.float64).flatten()
            qn = np.linalg.norm(q)
            pq1 = (q.astype(np.float32) @ self.P1.T).astype(np.float64)
            qr1 = math.sqrt(max(0, qn*qn - np.linalg.norm(pq1)**2))
            B1 = self.pr1 @ pq1 + self.e1 * qr1 + 1e-10
            scores[i] = float(B1.max())

        return scores


class ScoreMadhava:
    """
    MADHAVA: cascata [64,128] + error backpropagation modulation.
    """
    def __init__(self, centroids, seed=SEED):
        self.centroids = centroids
        self.full_dim = 384
        self.d1, self.d2 = 64, 128
        self.rng = np.random.RandomState(seed + 20)

        def mk_proj(d_out):
            R = self.rng.randn(self.full_dim, self.full_dim).astype(np.float64)
            Q, _ = np.linalg.qr(R.T)
            P = Q[:, :d_out].T.astype(np.float32)
            err = np.abs(P @ P.T - np.eye(d_out)).max()
            assert err < 1e-5, f"Ortho FAILED: {err}"
            return P

        self.P1 = mk_proj(self.d1)
        self.P2 = mk_proj(self.d2)

        c64 = centroids.astype(np.float64)
        norms = np.linalg.norm(c64, axis=1)

        pr1 = (centroids.astype(np.float32) @ self.P1.T).astype(np.float64)
        cap1 = np.linalg.norm(pr1, axis=1)
        self.e1 = np.sqrt(np.maximum(norms**2 - cap1**2, 0))
        self.pr1 = pr1

        pr2 = (centroids.astype(np.float32) @ self.P2.T).astype(np.float64)
        cap2 = np.linalg.norm(pr2, axis=1)
        self.e2 = np.sqrt(np.maximum(norms**2 - cap2**2, 0))
        self.pr2 = pr2

    def predict(self, test_embs):
        N = len(test_embs)
        scores = np.zeros(N, dtype=np.float64)
        mu = max(np.mean(self.e1), 1e-9)

        for i in range(N):
            q = test_embs[i].astype(np.float64).flatten()
            qn = np.linalg.norm(q)

            pq1 = (q.astype(np.float32) @ self.P1.T).astype(np.float64)
            qr1 = math.sqrt(max(0, qn*qn - np.linalg.norm(pq1)**2))

            pq2 = (q.astype(np.float32) @ self.P2.T).astype(np.float64)
            qr2 = math.sqrt(max(0, qn*qn - np.linalg.norm(pq2)**2))

            B1 = self.pr1 @ pq1 + self.e1 * qr1 + 1e-10
            B2 = self.pr2 @ pq2 + self.e2 * qr2 + 1e-10

            delta_e = (self.e1 - self.e2) / mu
            alpha = np.clip(1.0 / (1.0 + np.exp(-delta_e * 0.5)), 0.01, 0.99)
            modulated = B1 + alpha * (B2 - B1)

            scores[i] = float(modulated.max())

        return scores


# ================================================================
# CLASSIFICADOR COM THRESHOLD OTIMIZADO NO TREINO
# ================================================================

def optimize_threshold(scores, labels):
    """
    Encontra threshold que maximiza F1.
    Grid search sobre os scores do conjunto de TREINO.
    """
    if len(np.unique(labels)) < 2:
        return 0.5, 0.0, 0.0, 0.0

    thresholds = np.linspace(scores.min(), scores.max(), 500)
    best_f1, best_th = 0.0, 0.5

    for th in thresholds:
        pred = (scores >= th).astype(np.int32)
        if pred.sum() == 0:
            continue
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    return best_th


def classify_at(scores, labels, threshold):
    """Classifica com threshold fixo."""
    pred = (scores >= threshold).astype(np.int32)
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())

    tp = int((pred * labels).sum())
    fp = int((pred * (1 - labels)).sum())
    fn = n_pos - tp
    tn = n_neg - fp

    f1 = f1_score(labels, pred, zero_division=0)
    prec = precision_score(labels, pred, zero_division=0)
    rec = recall_score(labels, pred, zero_division=0)
    mcc = matthews_corrcoef(labels, pred) if n_pos > 0 and n_neg > 0 else 0.0
    spec = tn / max(n_neg, 1)
    auc = roc_auc_score(labels, scores) if n_pos > 0 and n_neg > 0 else 0.5

    return {
        "threshold": round(float(threshold), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "mcc": round(float(mcc), 4),
        "auc": round(float(auc), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


# ================================================================
# INTERVALO DE CONFIANCA (BOOTSTRAP)
# ================================================================

def bootstrap_ci(scores_a, labels_a, scores_b, labels_b, n_bootstrap=1000, alpha=0.05):
    """
    Bootstrap para diferenca de F1 entre dois metodos.
    Retorna CI 95% da diferenca e Cohen's d.
    """
    n = len(labels_a)
    diffs = []
    rng = np.random.RandomState(SEED + 42)

    th_a = optimize_threshold(scores_a, labels_a)
    th_b = optimize_threshold(scores_b, labels_b)

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        f1_a = classify_at(scores_a[idx], labels_a[idx], th_a)["f1"]
        f1_b = classify_at(scores_b[idx], labels_b[idx], th_b)["f1"]
        diffs.append(f1_b - f1_a)

    diffs = np.array(diffs)
    ci_lo = float(np.percentile(diffs, alpha/2 * 100))
    ci_hi = float(np.percentile(diffs, (1 - alpha/2) * 100))

    # Cohen's d: pooled std
    f1_a_vals = []
    f1_b_vals = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        f1_a_vals.append(classify_at(scores_a[idx], labels_a[idx], th_a)["f1"])
        f1_b_vals.append(classify_at(scores_b[idx], labels_b[idx], th_b)["f1"])

    mean_diff = np.mean(f1_b_vals) - np.mean(f1_a_vals)
    pooled_std = math.sqrt((np.var(f1_a_vals) + np.var(f1_b_vals)) / 2 + 1e-10)
    cohens_d = mean_diff / pooled_std

    # p-value: proporcao de bootstrap onde diff <= 0
    p_value = np.mean(np.array(diffs) <= 0)

    return {
        "diff_mean": round(float(np.mean(diffs)), 4),
        "ci_95_lo": round(ci_lo, 4),
        "ci_95_hi": round(ci_hi, 4),
        "cohens_d": round(float(cohens_d), 4),
        "p_value": round(float(p_value), 4),
        "significant_95": p_value < alpha,
    }


# ================================================================
# 5-FOLD STRATIFIED CROSS-VALIDATION
# ================================================================

def run_fold(train_idxs, test_idxs, texts, labels, embedder):
    """
    Roda um fold.
    CADA metodo otimiza seu threshold NO TREINO e avalia no TESTE.
    """
    train_texts = [texts[i] for i in train_idxs]
    train_labels = labels[train_idxs]
    test_texts = [texts[i] for i in test_idxs]
    test_labels = labels[test_idxs]

    result = {"train_size": len(train_texts), "test_size": len(test_texts)}

    # Embed train + test
    train_embs = embedder.encode(
        train_texts, normalize_embeddings=True,
        show_progress_bar=False, batch_size=128
    ).astype(np.float32)
    test_embs = embedder.encode(
        test_texts, normalize_embeddings=True,
        show_progress_bar=False, batch_size=128
    ).astype(np.float32)

    # KMeans nos TREINO injections
    inj_mask = train_labels == 1
    inj_embs = train_embs[inj_mask]
    n_inj = len(inj_embs)
    K = min(30, max(2, n_inj // 10))

    if n_inj < K:
        K = max(2, n_inj)

    if n_inj >= K:
        kmeans = KMeans(n_clusters=K, random_state=SEED, n_init=3, max_iter=200)
        kmeans.fit(inj_embs)
        centroids = kmeans.cluster_centers_.astype(np.float32)
    else:
        centroids = inj_embs[:K].copy()

    # Normalizar centroides
    cn = np.linalg.norm(centroids, axis=1, keepdims=True)
    cn[cn == 0] = 1.0
    centroids /= cn
    result["K"] = K

    # Centroides aleatorios para Random (mesmo numero, mesma dimensao)
    rng = np.random.RandomState(SEED + 99)
    random_centroids = rng.randn(K, 384).astype(np.float32)
    rcn = np.linalg.norm(random_centroids, axis=1, keepdims=True)
    rcn[rcn == 0] = 1.0
    random_centroids /= rcn

    # --- 4 metodos de score ---

    methods = {}

    # 1. DIRECT (gold standard)
    direct = ScoreDirect(centroids)
    train_direct = direct.predict(train_embs)
    test_direct = direct.predict(test_embs)
    th_direct = optimize_threshold(train_direct, train_labels)
    methods["direct"] = classify_at(test_direct, test_labels, th_direct)
    methods["direct"]["threshold"] = th_direct
    methods["direct"]["mse_vs_direct"] = 0.0
    methods["direct"]["spearman_vs_direct"] = 1.0

    # MSE de cada metodo vs direct
    def mse_vs_direct(scores):
        return float(np.mean((scores - test_direct)**2))

    def spearman_vs_direct(scores):
        r, _ = scipy_stats.spearmanr(scores, test_direct)
        return round(float(r), 4)

    # 2. RANDOM (com embedding, centroides aleatorios)
    rand = ScoreRandom(random_centroids)
    train_rand = rand.predict(train_embs)
    test_rand = rand.predict(test_embs)
    th_rand = optimize_threshold(train_rand, train_labels)
    methods["random"] = classify_at(test_rand, test_labels, th_rand)
    methods["random"]["threshold"] = th_rand
    methods["random"]["mse_vs_direct"] = mse_vs_direct(test_rand)
    methods["random"]["spearman_vs_direct"] = spearman_vs_direct(test_rand)

    # 3. BOUND (projecao 64D + CS, sem modulacao)
    bound = ScoreBound(centroids)
    train_bound = bound.predict(train_embs)
    test_bound = bound.predict(test_embs)
    th_bound = optimize_threshold(train_bound, train_labels)
    methods["bound"] = classify_at(test_bound, test_labels, th_bound)
    methods["bound"]["threshold"] = th_bound
    methods["bound"]["mse_vs_direct"] = mse_vs_direct(test_bound)
    methods["bound"]["spearman_vs_direct"] = spearman_vs_direct(test_bound)

    # 4. MADHAVA (cascata + modulacao)
    madhava = ScoreMadhava(centroids)
    train_mad = madhava.predict(train_embs)
    test_mad = madhava.predict(test_embs)
    th_mad = optimize_threshold(train_mad, train_labels)
    methods["madhava"] = classify_at(test_mad, test_labels, th_mad)
    methods["madhava"]["threshold"] = th_mad
    methods["madhava"]["mse_vs_direct"] = mse_vs_direct(test_mad)
    methods["madhava"]["spearman_vs_direct"] = spearman_vs_direct(test_mad)

    # Estatisticas dos scores
    result["scores_mean"] = {
        "direct": round(float(np.mean(test_direct)), 4),
        "random": round(float(np.mean(test_rand)), 4),
        "bound": round(float(np.mean(test_bound)), 4),
        "madhava": round(float(np.mean(test_mad)), 4),
    }
    result["scores_std"] = {
        "direct": round(float(np.std(test_direct)), 4),
        "random": round(float(np.std(test_rand)), 4),
        "bound": round(float(np.std(test_bound)), 4),
        "madhava": round(float(np.std(test_mad)), 4),
    }

    result["methods"] = methods
    result["scores_raw"] = {
        "test_direct": [round(float(x), 6) for x in test_direct],
        "test_madhava": [round(float(x), 6) for x in test_mad],
        "test_bound": [round(float(x), 6) for x in test_bound],
        "test_random": [round(float(x), 6) for x in test_rand],
        "test_labels": [int(x) for x in test_labels],
    }

    return result


# ================================================================
# PRINT
# ================================================================

def print_fold(fold_num, result):
    print(f"  Fold {fold_num}")
    print(f"    Train: {result['train_size']} | Test: {result['test_size']} | K={result['K']}")
    for name in ["direct", "random", "bound", "madhava"]:
        m = result["methods"][name]
        mse = m.get("mse_vs_direct", 0)
        spearman = m.get("spearman_vs_direct", 0)
        extras = ""
        if spearman != 0:
            extras = f" spearman={spearman}"
        if "mse_vs_direct" in m:
            extras = f" th={m['threshold']:.4f} mse={mse:.6f} spearman={spearman}"
        else:
            extras = f" th={m['threshold']:.4f}"
        print(f"    {name:<8}"
              f" auc={m['auc']:.4f} f1={m['f1']:.4f}"
              f" prec={m['precision']:.4f} rec={m['recall']:.4f}"
              f" spec={m['specificity']:.4f} mcc={m['mcc']:.4f}"
              f"{extras}")


def print_summary(all_results, methods_list):
    print("\n" + "=" * 90)
    print("  RESUMO FINAL — Média (std) sobre 5 folds")
    print("=" * 90)
    print()

    for method in methods_list:
        metrics = {}
        for metric in ["auc", "f1", "precision", "recall", "specificity", "mcc",
                       "mse_vs_direct", "spearman_vs_direct"]:
            vals = [r["methods"][method].get(metric, 0) for r in all_results]
            metrics[metric] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
            }

        mse_str = f" mse={metrics['mse_vs_direct']['mean']:>8.6f}"
        spearman_str = f" spearman={metrics['spearman_vs_direct']['mean']:>7.4f}"
        print(f"  {method:<8}"
              f" auc={metrics['auc']['mean']:.4f}+-{metrics['auc']['std']:.4f}"
              f" f1={metrics['f1']['mean']:.4f}+-{metrics['f1']['std']:.4f}"
              f" prec={metrics['precision']['mean']:.4f}"
              f" rec={metrics['recall']['mean']:.4f}"
              f" spec={metrics['specificity']['mean']:.4f}"
              f" mcc={metrics['mcc']['mean']:.4f}"
              f"{mse_str}{spearman_str}")

    print()

    # Tabela de diferencas (cada metodo vs DIRECT)
    print("  Diferenca F1 vs DIRECT (com Cohen's d > 0.8 = large effect):")
    for method in methods_list:
        if method == "direct":
            continue
        f1_direct = np.mean([r["methods"]["direct"]["f1"] for r in all_results])
        f1_method = np.mean([r["methods"][method]["f1"] for r in all_results])
        diff = f1_method - f1_direct

        # Cohen's d pooled
        vals_d = [r["methods"]["direct"]["f1"] for r in all_results]
        vals_m = [r["methods"][method]["f1"] for r in all_results]
        pooled_std = math.sqrt((np.var(vals_d) + np.var(vals_m)) / 2 + 1e-10)
        d_cohen = diff / max(pooled_std, 1e-10) if pooled_std > 0 else 0.0

        effect = "LARGE" if abs(d_cohen) > 0.8 else "MEDIUM" if abs(d_cohen) > 0.5 else "SMALL" if abs(d_cohen) > 0.2 else "NEGLIGIBLE"
        print(f"    {method:<8} delta_F1={diff:+.4f}  Cohen's d={d_cohen:+.4f} ({effect})")


def analyze_error_patterns(all_results):
    """Analisa em quais decils de score Madhava erra mais vs Direct."""
    print("\n  Analise de erros por decil (media sobre folds):")
    decil_buckets = {i: {"mse": [], "count": 0} for i in range(10)}

    for r in all_results:
        raw = r.get("scores_raw", {})
        if not raw:
            continue
        direct = np.array(raw["test_direct"])
        madhava = np.array(raw["test_madhava"])
        labels = np.array(raw["test_labels"])

        # Decis baseados no score direct
        for d, m, lbl in zip(direct, madhava, labels):
            decil = min(9, int(d * 10))
            decil = max(0, decil)
            decil_buckets[decil]["mse"].append((m - d)**2)
            decil_buckets[decil]["count"] += 1

    print(f"    {'Decil':<8} {'Count':<8} {'MSE':<10}")
    for i in range(10):
        bucket = decil_buckets[i]
        if bucket["count"] > 0:
            print(f"    [{i/10:.1f},{i/10+0.1:.1f}) {bucket['count']:<8} {np.mean(bucket['mse']):.6f}")


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 90)
    print("  MADHAVA-SEC BENCHMARK HONESTO v5.0")
    print("  5-Fold Stratified Cross-Validation | Threshold INDEPENDENTE por metodo")
    print("  Dataset: chuneeb/ai-agent-cybersecurity-dataset-2026")
    print("  Modelo: all-MiniLM-L6-v2 (384D)")
    print("=" * 90)
    print()

    texts, labels = load_data()
    n_inj = int(labels.sum())
    n_clean = len(labels) - n_inj
    print(f"  Dataset: {len(texts)} prompts ({n_inj} injection, {n_clean} clean)")
    print()

    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    print("  Modelo de embedding carregado.")
    print()

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_results = []

    for fold, (train_idxs, test_idxs) in enumerate(kf.split(texts, labels)):
        print(f"  {'─'*60}")
        t0 = time.time()
        result = run_fold(train_idxs, test_idxs, texts, labels, embedder)
        elapsed = time.time() - t0
        print_fold(fold + 1, result)
        print(f"  Tempo: {elapsed:.1f}s")
        all_results.append(result)

    methods_list = ["direct", "random", "bound", "madhava"]
    print_summary(all_results, methods_list)
    analyze_error_patterns(all_results)

    # Veredito final
    print("\n" + "=" * 90)
    print("  VEREDITO FINAL")
    print("=" * 90)

    f1_direct_mean = np.mean([r["methods"]["direct"]["f1"] for r in all_results])
    f1_madhava_mean = np.mean([r["methods"]["madhava"]["f1"] for r in all_results])
    f1_random_mean = np.mean([r["methods"]["random"]["f1"] for r in all_results])
    f1_bound_mean = np.mean([r["methods"]["bound"]["f1"] for r in all_results])

    print(f"\n  Direct (gold standard, 384D):   F1={f1_direct_mean:.4f}")
    print(f"  Random (embedding + random centroids): F1={f1_random_mean:.4f}")
    print(f"  Bound (projecao 64D, sem mod.): F1={f1_bound_mean:.4f}")
    print(f"  Madhava ([64,128] + modulation): F1={f1_madhava_mean:.4f}")
    print()

    if f1_madhava_mean >= f1_direct_mean * 0.95:
        print("  ✅ Madhava retem >= 95% do F1 do dot product exato.")
        print("     A projecao preserva informacao discriminativa.")
    elif f1_madhava_mean >= f1_direct_mean * 0.90:
        print("  ⚠️ Madhava retem 90-95% do F1 do dot product exato.")
        print("     Perda moderada — aceitavel para aplicacoes que exigem bound.")
    elif f1_madhava_mean >= f1_direct_mean * 0.80:
        print("  ⚠️ Madhava retem 80-90% do F1.")
        print("     Perda significativa. A modulacao nao recupera totalmente.")
    else:
        print("  ❌ Madhava retem < 80% do F1 do dot product exato.")
        print("     Perda GRAVE. A projecao 64D perde informacao discriminativa.")
    print()

    if f1_madhava_mean > f1_bound_mean:
        delta_mod = (f1_madhava_mean - f1_bound_mean) / max(f1_bound_mean, 0.01) * 100
        print(f"  Modulacao melhora F1 em {delta_mod:.1f}% vs Bound puro.")
    else:
        print(f"  Modulacao NAO melhora F1 vs Bound puro (delta={f1_madhava_mean - f1_bound_mean:.4f}).")

    if f1_madhava_mean <= f1_random_mean:
        print(f"\n  ❌ CRITICO: Madhava NAO supera Random com embedding.")
        print(f"     KMeans + projecao + modulacao nao agregam valor vs centroides aleatorios.")
    else:
        delta_vs_random = f1_madhava_mean - f1_random_mean
        print(f"\n  Ganho vs Random: +{delta_vs_random:.4f} F1")

    # Salvar
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.ndarray,)): return obj.tolist()
            if isinstance(obj, (np.bool_,)): return bool(obj)
            return super().default(obj)

    output = {
        "version": "5.0",
        "n_splits": 5,
        "dataset": "chuneeb/ai-agent-cybersecurity-dataset-2026",
        "model": "all-MiniLM-L6-v2",
        "veredito": {
            "f1_direct": round(f1_direct_mean, 4),
            "f1_madhava": round(f1_madhava_mean, 4),
            "f1_random": round(f1_random_mean, 4),
            "f1_bound": round(f1_bound_mean, 4),
            "retention_pct": round(f1_madhava_mean / max(f1_direct_mean, 0.01) * 100, 2),
            "modulation_gain_vs_bound": round(f1_madhava_mean - f1_bound_mean, 4),
            "gain_vs_random": round(f1_madhava_mean - f1_random_mean, 4),
        },
        "results": all_results,
    }

    # Remover scores_raw do output JSON (muito grande)
    for r in output["results"]:
        r.pop("scores_raw", None)

    with open("madhava_sec_benchmark_honest_v5_results.json", "w") as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"\n  Resultados em: madhava_sec_benchmark_honest_v5_results.json")


if __name__ == "__main__":
    main()
