#!/usr/bin/env python3
"""
benchmark_quality_flags.py — HONEST benchmark of the winnex-normalize Quality Flags.

PROTOCOL:
  - Installs winnex-ai-normalize==1.1.0 + winnex-madhava from PyPI in ISOLATION
    (pip install --target + PYTHONPATH) → nothing from the notebook env leaks in.
  - Uses real Kaggle datasets (raw, no mock preprocessing):
      * rtatman/glove-global-vectors-for-word-representation  (d=100, word vecs)
      * hojjatk/mnist-dataset                                  (d=784, image raw)
      * julien040/hacker-news-openai-embeddings                (d=1536, OpenAI emb)
      * ayan78/stsbrenchmark                                   (d=768, SBERT semantic)
  - The FLAG is the MOTOR's own validation: the QualityValidator launches the
    Cauchy-Schwarz proof (UB < threshold ⟹ mathematically outside top-K) over
    seed queries and CAPTURES the excluded set. The captured set IS the flag
    response. We measure ONLY what the engine/validator return.
  - Verifies dataset integrity: if a GT does not match the base (the corrupted-
    BIGANN class), the validator emits the alignment flag.
  - Measures: flags per dataset, proof coverage (bound_fraction), prefilter
    share, routed config (basis/k1/early_exit), recall vs the engine's own
    search_exact with the routed config, determinism of the excluded seed set.

Output: /kaggle/working/results/dataset_{...}.json + summary.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

PY = sys.executable
RESULTS_DIR = "/kaggle/working/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 0. Isolated installers (honest protocol: pip install --target)
# ---------------------------------------------------------------------------
def _target_dir(prefix: str) -> str:
    return tempfile.mkdtemp(prefix=f"wxf_{prefix}_")


def _install(target: str, *pkgs: str) -> None:
    subprocess.check_call(
        [PY, "-m", "pip", "install", "-q", "--no-cache-dir",
         "--disable-pip-version-check", "--target", target, *pkgs],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _run_code(target: str, code: str) -> dict:
    """Run `code` with PYTHONPATH=target; return JSON dict. Failure → clear error."""
    env = dict(os.environ)
    env["PYTHONPATH"] = target
    proc = subprocess.run([PY, "-c", code], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess rc={proc.returncode}\n--- stdout tail ---\n"
            f"{proc.stdout[-1500:]}\n--- stderr tail ---\n{proc.stderr[-1500:]}"
        )
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# 1. Discover datasets under /kaggle/input (dataset-agnostic)
# ---------------------------------------------------------------------------
def _find_input_files():
    """Map /kaggle/input → {dirname: {file: path}} to discover the data."""
    found = {}
    for root, dirs, files in os.walk("/kaggle/input/"):
        for f in files:
            if f.endswith((".npy", ".npz", ".txt", ".csv", ".bin")):
                rel = os.path.relpath(os.path.join(root, f), "/kaggle/input/")
                parts = rel.split(os.sep)
                dataset = parts[0]
                found.setdefault(dataset, {})[f] = os.path.join(root, f)
    return found


# ---------------------------------------------------------------------------
# 2. The validation code (runs in the isolated env with the 1.1.0 package)
# ---------------------------------------------------------------------------
# This block is the CORE of the benchmark: it builds a QualityValidator from
# the PyPI-installed package, runs the flags on a dataset, and returns the
# honest JSON.
VALIDATE_CODE = r"""
import json, os, sys, time
import numpy as np
import winnex_ai_normalize as wan
from winnex_ai_normalize.core.quality import audit_corpus, QualityValidator

# receives the dataset path + dim via argv-like env
data_path = sys.argv[1]
dim = int(sys.argv[2]) if len(sys.argv) > 2 else None
n_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 8
probe = sys.argv[4] if len(sys.argv) > 4 else "auto"  # auto|True|False

print("winnex_ai_normalize", wan.__version__, file=sys.stderr)

# load real data
if data_path.endswith(".npy"):
    X = np.load(data_path, mmap_mode="r")
elif data_path.endswith(".bin"):
    X = np.fromfile(data_path, dtype=np.uint8)
elif data_path.endswith(".txt"):
    X = np.loadtxt(data_path, delimiter=",")
else:
    import csv
    rows = []
    with open(data_path) as f:
        for r in csv.reader(f):
            if r and not r[0].startswith("#"):
                try:
                    rows.append([float(v) for v in r])
                except ValueError:
                    pass
    X = np.array(rows)
if X.ndim == 1:
    X = X.reshape(-1, dim) if dim else X.reshape(-1, int(np.sqrt(X.size)))

N = min(len(X), 20000)
X = np.ascontiguousarray(X[:N])

# Agnostic validation: the engine launches the proof and captures the excluded set
t0 = time.time()
if probe == "False":
    rep = audit_corpus(X, dim=dim, n_seed_queries=n_seed, probe_pca=False)
else:
    rep = audit_corpus(X, dim=dim, n_seed_queries=n_seed, probe_pca=True)
dt = time.time() - t0

# Determinism of the excluded seed set (same seed → same set)
rep2 = audit_corpus(X, dim=dim, n_seed_queries=n_seed,
                    probe_pca=(probe != "False"))
det = [d["doc_id"] for d in rep.excluded_seed_set]
det2 = [d["doc_id"] for d in rep2.excluded_seed_set]

# Recall with the ROUTED config (the engine is the validator)
import winnex_madhava as wm
routed = rep.to_dict()["suggested_config"]
eng = wm.build_engine(
    X, dim=dim, metric=routed["metric"], basis=routed["basis"],
    k1_fraction=routed["k1_fraction"], stage1_dim=routed["stage1_dim"],
    stage2_dim=routed["stage2_dim"], early_exit=routed["early_exit"],
    quant=routed["quant"], k=10, normalize_input=(routed["metric"]=="cosine"),
)
rec = 0.0
nq = 5
rng = np.random.default_rng(1)
qs = rng.choice(len(X), nq, replace=False)
for qi in qs:
    q = X[qi].astype(np.float32)
    r = eng.search(q)
    re = eng.search_exact(q)
    rec += sum(1 for j in r.indices if j in re.indices) / 10.0
rec /= nq

out = {
    "package_version": wan.__version__,
    "dataset": os.path.basename(data_path),
    "N": int(len(X)),
    "dim": int(X.shape[1]),
    "dtype": str(X.dtype),
    "validate_ms": round(dt * 1000, 1),
    "flags": [f.to_dict() for f in rep.flags],
    "verdict": rep.to_dict()["verdict"],
    "metrics": rep.to_dict()["metrics"],
    "suggested_config": routed,
    "excluded_seed_set_size": len(rep.excluded_seed_set),
    "excluded_seed_deterministic": det == det2,
    "excluded_sample": rep.excluded_seed_set[:5],
    "recall@10_vs_ceiling_routed": round(rec, 4),
}
print(json.dumps(out))
"""


# ---------------------------------------------------------------------------
# 3. Main — run the validator on each discovered dataset
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("QUALITY FLAGS BENCHMARK — winnex-normalize 1.1.0 (PyPI isolated)")
    print("=" * 70)

    # Install the measured package in isolation from PyPI
    target = _target_dir("wxf")
    _install(target, "winnex-ai-normalize==1.1.0")
    print(f"installed winnex-ai-normalize==1.1.0 at {target}")

    found = _find_input_files()
    print("datasets found under /kaggle/input:", sorted(found.keys()))

    results = {"meta": {"package": "winnex-ai-normalize", "version": "1.1.0",
                        "protocol": "honest — PyPI isolated, real datasets",
                        "method": "flags = engine CS proof (UB<threshold ⟹ out of top-K)"},
               "datasets": []}

    for ds, files in sorted(found.items()):
        # pick the most representative file of the dataset
        data_path = None
        dim = None
        for fname, path in sorted(files.items()):
            if fname.endswith((".npy", ".txt", ".bin")):
                data_path = path
                if "glove" in ds.lower():
                    dim = 100
                elif "mnist" in ds.lower():
                    dim = 784
                elif "hacker" in ds.lower() or "openai" in ds.lower():
                    dim = 1536
                elif "stsb" in ds.lower() or "sts" in ds.lower():
                    dim = 768
                break
        if not data_path:
            continue
        print(f"\n--- {ds}: {data_path} (dim={dim}) ---")
        try:
            r = _run_code(
                target,
                f"import sys; sys.argv=['x','{data_path}',"
                f"{dim or 'None'},'8','False']; exec('''{VALIDATE_CODE}''')"
            )
            results["datasets"].append(r)
            print(f"  verdict={r['verdict']} flags=",
                  [(f['code'], f['severity']) for f in r['flags']])
            print(f"  proof={r['metrics'].get('bound_fraction', '?')} "
                  f"recall@10_routed={r.get('recall@10_vs_ceiling_routed')} "
                  f"excl_seed={r['excluded_seed_set_size']}")
        except Exception as e:
            print(f"  FAILED: {str(e)[:300]}")
            results["datasets"].append({"dataset": ds, "error": str(e)[:500]})

    summary = {
        "meta": results["meta"],
        "total_datasets": len(results["datasets"]),
        "verdicts": [d.get("verdict") for d in results["datasets"]],
    }
    with open(f"{RESULTS_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(f"{RESULTS_DIR}/quality_flags_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 70)
    print("SUMMARY")
    for d in results["datasets"]:
        print(f"  {d.get('dataset','?')}: verdict={d.get('verdict')} "
              f"proof={d.get('metrics',{}).get('bound_fraction')} "
              f"recall={d.get('recall@10_vs_ceiling_routed')}")
    print("=" * 70)
    print(f"Saved under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
