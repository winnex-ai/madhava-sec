#!/usr/bin/env python3
"""
==========================================================================
Madhava-Sec — Benchmark with KAGGLE DATASET + NATIVE C++ ENGINE
==========================================================================

What this script does:

  1. LOADS a Kaggle dataset specialized in prompt injection / jailbreak
     detection (krishnayadav456wrsty/prompt-injection-and-jailbreak-
     detection-dataset). Fallback: local dataset.

  2. 5-fold stratified cross-validation, per-method independent threshold
     (optimized on train, evaluated on test) — same protocol as v5.

  3. Four score methods over the SAME KMeans centroids:
       DIRECT   — exact 384D dot product (gold standard)
       RANDOM   — random centroids (honest baseline)
       BOUND    — 64D projection + Cauchy-Schwarz, no modulation
       MADHAVA  — cascade [64,128] + modulation
       MADHAVA_NATIVE — the SAME Madhava, but in the NATIVE C++ engine
                        (libmadhava_sec.so via ctypes)

  4. Full per-method metrics: F1, AUC, precision, recall, specificity,
     MCC + bound violations + latency.

  5. Verifies 0 Cauchy-Schwarz bound violations on the real centroids.

The native C++ engine (cpp/madhava_core.h) implements the SAME math as the
Python ScoreMadhava: orthogonal QR projection, int8 quantization with
verified scale, Cauchy-Schwarz bound, and error-backpropagation modulation.
The difference is that it runs in native C++ with SIMD (AVX2+FMA) and
OpenMP.

License: BSL 1.1 | pay@winnex.ai
==========================================================================
"""

import time, math, random, json, gc, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, matthews_corrcoef)
from sklearn.cluster import KMeans
from scipy import stats as scipy_stats

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

EMBEDDING_DIM = 384

# ================================================================
# DATASET KAGGLE (especializado em prompt injection / jailbreak)
# ================================================================

# Kaggle dataset: krishnayadav456wrsty/prompt-injection-and-jailbreak-detection-dataset
# (combines prompt_injection_detection_dataset.csv + update1357.csv + v3update20001.csv)
# REAL data only — no synthetic/mocked fallback. If the dataset is unavailable,
# the benchmark refuses to run rather than fabricate data.
KAGGLE_DATASET_ID = "krishnayadav456wrsty/prompt-injection-and-jailbreak-detection-dataset"
KAGGLE_LOCAL = os.path.join(os.path.expanduser("~"), ".cache", "winnex", "pi_jailbreak_combined.csv")


def ensure_dataset():
    """
    Load a REAL dataset: the cached Kaggle combined CSV if present,
    otherwise download it from Kaggle. Raises if unavailable.
    No synthetic or simulated data is ever used.
    """
    if os.path.exists(KAGGLE_LOCAL):
        return load_combined_csv(KAGGLE_LOCAL)

    # Download the real Kaggle dataset.
    import subprocess, tempfile
    os.makedirs(os.path.dirname(KAGGLE_LOCAL), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET_ID,
             "-p", tmp, "--unzip"],
            check=True, capture_output=True, timeout=180,
        )
        parts = []
        for f in sorted(os.listdir(tmp)):
            if f.endswith(".csv"):
                parts.append(pd.read_csv(os.path.join(tmp, f)))
        if not parts:
            raise RuntimeError(
                "Kaggle dataset downloaded but no CSV found — cannot build a real benchmark.")
        df = pd.concat(parts, ignore_index=True)
        if "text" not in df.columns:
            raise ValueError(f"unexpected columns: {list(df.columns)}")
        df = df.drop_duplicates(subset=["text"])
        df = df[df["text"].notna()]
        df.to_csv(KAGGLE_LOCAL, index=False)
        return df["text"].tolist(), np.array(
            (df["label"] == "injection").values, dtype=np.int32)


def load_combined_csv(fp):
    df = pd.read_csv(fp)
    labels = np.array((df["label"] == "injection").values, dtype=np.int32)
    return df["text"].tolist(), labels


# ================================================================
# MOTOR C++ NATIVO (libmadhava_sec.so via ctypes)
# ================================================================

try:
    import ctypes as _ct

    _LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "cpp", "libmadhava_sec.so")
    _LIB_PATH = os.path.abspath(_LIB_PATH)
    _cpp_lib = _ct.CDLL(_LIB_PATH)

    _cpp_lib.madhava_sec_new.restype = _ct.c_void_p
    _cpp_lib.madhava_sec_new.argtypes = [_ct.c_int, _ct.c_int, _ct.c_int]
    _cpp_lib.madhava_sec_build.argtypes = [_ct.c_void_p, _ct.POINTER(_ct.c_float), _ct.c_int]
    _cpp_lib.madhava_sec_score.restype = _ct.POINTER(_ct.c_float)
    _cpp_lib.madhava_sec_score.argtypes = [_ct.c_void_p, _ct.POINTER(_ct.c_float)]
    _cpp_lib.madhava_sec_max_score.restype = _ct.c_float
    _cpp_lib.madhava_sec_max_score.argtypes = [_ct.c_void_p, _ct.POINTER(_ct.c_float)]
    _cpp_lib.madhava_sec_verify.argtypes = [_ct.c_void_p, _ct.POINTER(_ct.c_float),
                                            _ct.POINTER(_ct.c_long), _ct.POINTER(_ct.c_long)]
    _cpp_lib.madhava_sec_free.argtypes = [_ct.c_void_p]
    CPP_AVAILABLE = True
except Exception as _e:
    CPP_AVAILABLE = False
    _CPP_ERR = str(_e)


class ScoreMadhavaNative:
    """
    MADHAVA in the NATIVE C++ ENGINE.
    The centroid is projected/int8-quantized INSIDE the C++ (build),
    and the score uses the modulated Cauchy-Schwarz bound + SIMD.
    """
    def __init__(self, centroids, stage_dims=(64, 128)):
        self.centroids = centroids
        self.d1, self.d2 = stage_dims
        self._eng = _cpp_lib.madhava_sec_new(EMBEDDING_DIM, self.d1, self.d2)
        c = np.ascontiguousarray(centroids, dtype=np.float32)
        pc = c.ctypes.data_as(_ct.POINTER(_ct.c_float))
        _cpp_lib.madhava_sec_build(self._eng, pc, len(centroids))

    def predict(self, test_embs, with_verify=False):
        scores = np.zeros(len(test_embs), dtype=np.float32)
        total_v = total_c = 0
        for i, q in enumerate(test_embs):
            qq = np.ascontiguousarray(q, dtype=np.float32)
            pq = qq.ctypes.data_as(_ct.POINTER(_ct.c_float))
            scores[i] = _cpp_lib.madhava_sec_max_score(self._eng, pq)
            if with_verify:
                v = _ct.c_long(0); c = _ct.c_long(0)
                _cpp_lib.madhava_sec_verify(self._eng, pq, _ct.byref(v), _ct.byref(c))
                total_v += v.value; total_c += c.value
        if with_verify:
            return scores, total_v, total_c
        return scores

    def verify(self, test_embs):
        _, v, c = self.predict(test_embs, with_verify=True)
        return v, c

    def __del__(self):
        try:
            _cpp_lib.madhava_sec_free(self._eng)
        except Exception:
            pass


# ================================================================
# PYTHON METHODS (same as v5, for honest comparison)
# ================================================================

class ScoreDirect:
    def __init__(self, centroids):
        self.centroids = centroids

    def predict(self, test_embs):
        return (test_embs @ self.centroids.T).max(axis=1)


class ScoreRandom(ScoreDirect):
    pass


class ScoreBound:
    def __init__(self, centroids, seed=SEED):
        self.centroids = centroids
        self.full_dim = EMBEDDING_DIM
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
    def __init__(self, centroids, seed=SEED):
        self.centroids = centroids
        self.full_dim = EMBEDDING_DIM
        self.d1, self.d2 = 64, 128
        self.rng = np.random.RandomState(seed + 20)

        def mk_proj(d_out):
            R = self.rng.randn(self.full_dim, self.full_dim).astype(np.float64)
            Q, _ = np.linalg.qr(R.T)
            P = Q[:, :d_out].T.astype(np.float32)
            err = np.abs(P @ P.T - np.eye(d_out)).max()
            assert err < 1e-5, f"Orthogonality FAILED: {err}"
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
# CLASSIFICATION (threshold optimized on train)
# ================================================================

def optimize_threshold(scores, labels):
    if len(np.unique(labels)) < 2:
        return 0.5
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
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_pos": n_pos, "n_neg": n_neg,
    }


# ================================================================
# FOLD
# ================================================================

def run_fold(train_idxs, test_idxs, texts, labels, embedder):
    train_texts = [texts[i] for i in train_idxs]
    train_labels = labels[train_idxs]
    test_texts = [texts[i] for i in test_idxs]
    test_labels = labels[test_idxs]

    result = {"train_size": len(train_texts), "test_size": len(test_texts)}

    train_embs = embedder.encode(train_texts, normalize_embeddings=True,
                                 show_progress_bar=False, batch_size=128).astype(np.float32)
    test_embs = embedder.encode(test_texts, normalize_embeddings=True,
                                show_progress_bar=False, batch_size=128).astype(np.float32)

    inj_mask = train_labels == 1
    inj_embs = train_embs[inj_mask]
    n_inj = len(inj_embs)
    K = min(30, max(2, n_inj // 10))

    if n_inj >= K:
        kmeans = KMeans(n_clusters=K, random_state=SEED, n_init=3, max_iter=200)
        kmeans.fit(inj_embs)
        centroids = kmeans.cluster_centers_.astype(np.float32)
    else:
        centroids = inj_embs[:K].copy()

    cn = np.linalg.norm(centroids, axis=1, keepdims=True)
    cn[cn == 0] = 1.0
    centroids /= cn
    result["K"] = K

    rng = np.random.RandomState(SEED + 99)
    random_centroids = rng.randn(K, EMBEDDING_DIM).astype(np.float32)
    rcn = np.linalg.norm(random_centroids, axis=1, keepdims=True)
    rcn[rcn == 0] = 1.0
    random_centroids /= rcn

    methods = {}

    def add_method(name, scorer, with_verify=False, verify_embs=None):
        t0 = time.time()
        train_s = scorer.predict(train_embs)
        test_s = scorer.predict(test_embs)
        el = time.time() - t0
        th = optimize_threshold(train_s, train_labels)
        m = classify_at(test_s, test_labels, th)
        m["threshold"] = th
        try:
            r, _ = scipy_stats.spearmanr(test_s, test_direct)
            m["spearman_vs_direct"] = round(float(r), 4)
        except Exception:
            m["spearman_vs_direct"] = 0.0
        m["latency_s"] = round(float(el), 3)
        methods[name] = m
        if with_verify:
            v, c = scorer.verify(verify_embs or test_embs)
            methods[name]["bound_violations"] = int(v)
            methods[name]["bound_checked"] = int(c)

    # 1. DIRECT (gold standard)
    direct = ScoreDirect(centroids)
    test_direct = direct.predict(test_embs)
    train_direct = direct.predict(train_embs)
    th_direct = optimize_threshold(train_direct, train_labels)
    methods["direct"] = classify_at(test_direct, test_labels, th_direct)
    methods["direct"]["threshold"] = th_direct
    methods["direct"]["spearman_vs_direct"] = 1.0
    methods["direct"]["latency_s"] = 0.0

    # 2. RANDOM
    add_method("random", ScoreRandom(random_centroids))

    # 3. BOUND
    add_method("bound", ScoreBound(centroids))

    # 4. MADHAVA (Python)
    add_method("madhava", ScoreMadhava(centroids))

    # 5. MADHAVA_NATIVE (C++ engine)
    if CPP_AVAILABLE:
        add_method("madhava_native", ScoreMadhavaNative(centroids), with_verify=True)

    result["methods"] = methods
    return result


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 92)
    print("  MADHAVA-SEC BENCHMARK — KAGGLE DATASET + NATIVE C++ ENGINE")
    print("=" * 92)
    print(f"  Kaggle: {KAGGLE_DATASET_ID}")
    print(f"  Model: all-MiniLM-L6-v2 ({EMBEDDING_DIM}D)")
    print(f"  Native C++ engine: {'OK (libmadhava_sec.so)' if CPP_AVAILABLE else 'NOT AVAILABLE'}")
    if not CPP_AVAILABLE:
        print(f"    (error: {_CPP_ERR})")
    print()

    texts, labels = ensure_dataset()
    n_inj = int(labels.sum())
    n_clean = len(labels) - n_inj
    print(f"  Dataset: {len(texts)} prompts ({n_inj} injection, {n_clean} benign)")
    print()

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    print("  Embedding model loaded.")
    print()

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_results = []

    for fold, (train_idxs, test_idxs) in enumerate(kf.split(texts, labels)):
        print(f"  {'─' * 70}")
        t0 = time.time()
        result = run_fold(train_idxs, test_idxs, texts, labels, embedder)
        elapsed = time.time() - t0
        print(f"  Fold {fold + 1} | train={result['train_size']} test={result['test_size']} K={result['K']} | {elapsed:.1f}s")

        for name in ["direct", "random", "bound", "madhava", "madhava_native"]:
            if name not in result.get("methods", {}):
                continue
            m = result["methods"][name]
            extra = ""
            if "bound_violations" in m:
                extra = (f" viol={m['bound_violations']}/{m['bound_checked']}"
                         f" lat={m['latency_s']:.1f}s")
            elif name != "direct":
                extra = f" lat={m['latency_s']:.1f}s"
            print(f"    {name:<15}"
                  f" auc={m['auc']:.4f} f1={m['f1']:.4f}"
                  f" prec={m['precision']:.4f} rec={m['recall']:.4f}"
                  f" spec={m['specificity']:.4f} mcc={m['mcc']:.4f}"
                  f"{extra}")
        all_results.append(result)
        gc.collect()

    # ================================================================
    # RESUMO
    # ================================================================
    print()
    print("=" * 92)
    print("  FINAL SUMMARY — mean (std) over 5 folds")
    print("=" * 92)

    methods_list = ["direct", "random", "bound", "madhava"]
    if CPP_AVAILABLE:
        methods_list.append("madhava_native")

    summary = {}
    for method in methods_list:
        metrics = {}
        for metric in ["auc", "f1", "precision", "recall", "specificity", "mcc",
                       "spearman_vs_direct"]:
            vals = [r["methods"][method].get(metric, 0) for r in all_results]
            metrics[metric] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
            }
        summary[method] = metrics
        print(f"  {method:<15}"
              f" auc={metrics['auc']['mean']:.4f}+-{metrics['auc']['std']:.4f}"
              f" f1={metrics['f1']['mean']:.4f}+-{metrics['f1']['std']:.4f}"
              f" prec={metrics['precision']['mean']:.4f}"
              f" rec={metrics['recall']['mean']:.4f}"
              f" spec={metrics['specificity']['mean']:.4f}"
              f" mcc={metrics['mcc']['mean']:.4f}"
              f" spearman={metrics['spearman_vs_direct']['mean']:.4f}")

    # Native bound violations (summed over folds)
    if CPP_AVAILABLE:
        tot_v = sum(r["methods"]["madhava_native"].get("bound_violations", 0)
                    for r in all_results)
        tot_c = sum(r["methods"]["madhava_native"].get("bound_checked", 0)
                    for r in all_results)
        print(f"\n  Bound violations (native C++ engine): {tot_v} / {tot_c}")
        print(f"  {'OK: 0 violations — Cauchy-Schwarz guarantee holds in C++' if tot_v == 0 else 'ERROR: violations found!'}")

    # Retention of native vs direct
    f1_direct = np.mean([r["methods"]["direct"]["f1"] for r in all_results])
    f1_native = np.mean([r["methods"]["madhava_native"]["f1"] for r in all_results]) if CPP_AVAILABLE else 0
    if CPP_AVAILABLE:
        print(f"\n  Retention of C++ engine vs exact dot product: {f1_native / max(f1_direct, 0.01) * 100:.1f}%")

    # Salvar
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.ndarray,)): return obj.tolist()
            if isinstance(obj, (np.bool_,)): return bool(obj)
            return super().default(obj)

    output = {
        "version": "1.0",
        "n_splits": 5,
        "kaggle_dataset": KAGGLE_DATASET_ID,
        "model": "all-MiniLM-L6-v2",
        "cpp_native": bool(CPP_AVAILABLE),
        "dataset_n": len(labels),
        "dataset_n_inj": n_inj,
        "dataset_n_clean": n_clean,
        "summary": summary,
        "results": all_results,
    }
    for r in output["results"]:
        for name in list(r.get("methods", {})):
            r["methods"][name].pop("scores_raw", None)

    out_path = "kaggle_benchmark_native_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"\n  Results in: {out_path}")


if __name__ == "__main__":
    main()
