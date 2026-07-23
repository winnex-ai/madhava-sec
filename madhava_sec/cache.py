"""
Madhava-Sec Disk Cache — mmap-backed storage for large-scale projection data
=============================================================================
Stores projection matrices, projected vectors, and residuals on disk via numpy
memory-mapped arrays. Enables scaling to 10M-100M vectors without OOM.

Memory strategy:
  - Projection matrices (P): always in RAM (d_out x d_in, typically KB)
  - Raw embeddings: mmap on disk (N x d_in x 4B, GB)
  - Projected vectors: mmap on disk (N x d_out x 4B, GB)
  - Residuals: mmap on disk (N x 1 x 4B, MB)
  - Norms: mmap on disk (N x 1 x 4B, MB)

Access: all arrays support numpy indexing transparently via mmap.

License: BSL 1.1 | pay@winnex.ai
"""

import os, time, math, shutil, warnings, tempfile
import numpy as np
from numpy.linalg import qr

SEED = 42


def _estimate_intrinsic_dim(embeddings: np.ndarray) -> float:
    """Von Neumann entropy -> intrinsic dimension estimate."""
    _, s, _ = np.linalg.svd(embeddings.astype(np.float64), full_matrices=False)
    e2 = np.maximum(s ** 2, 1e-15)
    e2 /= e2.sum() + 1e-15
    return float(np.exp(-np.sum(e2 * np.log(e2 + 1e-15))))


class MadhavaSecCache:
    """
    mmap-backed cache for Madhava-Sec projection data.

    Usage:
      cache = MadhavaSecCache("/tmp/madhava_cache", dims=[32, 128])
      cache.build(vectors)  # stores all projections on disk
      B1 = cache.bound_stage1(query)  # reads from mmap transparently
      cache.clear()  # deletes disk files
    """

    def __init__(self, cache_dir: str = None, dims=None, seed=SEED):
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="madhava_sec_")
        self.dims = dims or [32, 128]
        self.rng = np.random.RandomState(seed + 1)

        # Metadata
        self.n = 0
        self.d_in = 0
        self.d_int = None
        self.build_time = 0.0

        # In-memory: projection matrices (small, always RAM)
        self.proj_mat = {}

        # mmap-backed: large arrays
        self._emb_fp = None     # embeddings: N x d_in float32
        self._norms_fp = None   # norms: N float32
        self._proj_fp = {}      # proj per dim: N x d_out float32
        self._error_fp = {}     # error per dim: N float32
        self._ready = False

    # ─────────────── Build ───────────────

    def build(self, vectors: np.ndarray):
        """
        Build projection cache on disk.

        Memory: projection matrices in RAM (~KB), everything else mmap'd.
        """
        t0 = time.time()
        n, d_in = vectors.shape
        self.n = n
        self.d_in = d_in

        # Estimate intrinsic dim
        sample = vectors[:min(n, 10000)]
        self.d_int = _estimate_intrinsic_dim(sample)

        # Create mmap files
        os.makedirs(self.cache_dir, exist_ok=True)
        self._create_mmaps(n, d_in)

        # Write embeddings
        self._emb_fp[:] = vectors.astype(np.float32)
        self._emb_fp.flush()

        # Compute norms
        norms = np.linalg.norm(self._emb_fp, axis=1)
        self._norms_fp[:] = np.maximum(norms, 1e-10)
        self._norms_fp.flush()

        # Build projections
        for d in self.dims:
            d_eff = min(d, d_in)
            P = self._ortho_proj(d_eff, d_in)
            self.proj_mat[d] = P

            # Project in batches to manage memory
            batch_size = max(1, min(50000, 2_000_000_000 // (d_in * 4)))
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = self._emb_fp[start:end]
                proj = batch @ P.T
                self._proj_fp[d][start:end] = proj
                captured = np.linalg.norm(proj, axis=1)
                err = np.sqrt(np.maximum(
                    self._norms_fp[start:end] ** 2 - captured ** 2, 0
                ))
                self._error_fp[d][start:end] = err

            self._proj_fp[d].flush()
            self._error_fp[d].flush()

        self.build_time = time.time() - t0
        self._ready = True
        return self

    def _create_mmaps(self, n, d_in):
        """Create memory-mapped numpy arrays on disk."""
        self._emb_fp = np.memmap(
            os.path.join(self.cache_dir, "embeddings.bin"),
            dtype=np.float32, mode='w+', shape=(n, d_in)
        )
        self._norms_fp = np.memmap(
            os.path.join(self.cache_dir, "norms.bin"),
            dtype=np.float32, mode='w+', shape=(n,)
        )
        for d in self.dims:
            d_eff = min(d, d_in)
            self._proj_fp[d] = np.memmap(
                os.path.join(self.cache_dir, f"proj_{d}.bin"),
                dtype=np.float32, mode='w+', shape=(n, d_eff)
            )
            self._error_fp[d] = np.memmap(
                os.path.join(self.cache_dir, f"error_{d}.bin"),
                dtype=np.float32, mode='w+', shape=(n,)
            )

    def _ortho_proj(self, d_out, d_in):
        """QR-orthogonal projection matrix (RAM)."""
        R = self.rng.randn(d_out, d_in).astype(np.float64)
        Q, _ = qr(R.T)
        return Q[:, :d_out].T.astype(np.float32)

    # ─────────────── Load existing cache ───────────────

    def load(self, cache_dir: str):
        """Load a previously built cache from disk."""
        self.cache_dir = cache_dir
        meta = np.load(os.path.join(cache_dir, "meta.npz"))
        self.n = int(meta["n"])
        self.d_in = int(meta["d_in"])
        self.d_int = float(meta["d_int"]) if "d_int" in meta else None
        self.dims = meta["dims"].tolist() if hasattr(meta["dims"], 'tolist') else list(meta["dims"])
        self.build_time = float(meta.get("build_time", 0))

        self._emb_fp = np.memmap(
            os.path.join(cache_dir, "embeddings.bin"),
            dtype=np.float32, mode='r', shape=(self.n, self.d_in)
        )
        self._norms_fp = np.memmap(
            os.path.join(cache_dir, "norms.bin"),
            dtype=np.float32, mode='r', shape=(self.n,)
        )
        for d in self.dims:
            d_eff = min(d, self.d_in)
            self._proj_fp[d] = np.memmap(
                os.path.join(cache_dir, f"proj_{d}.bin"),
                dtype=np.float32, mode='r', shape=(self.n, d_eff)
            )
            self._error_fp[d] = np.memmap(
                os.path.join(cache_dir, f"error_{d}.bin"),
                dtype=np.float32, mode='r', shape=(self.n,)
            )
            # Load projection matrix
            self.proj_mat[d] = np.load(os.path.join(cache_dir, f"P_{d}.npy"))

        self._ready = True
        return self

    def save_metadata(self):
        """Save metadata for future load()."""
        np.savez(
            os.path.join(self.cache_dir, "meta.npz"),
            n=self.n, d_in=self.d_in, d_int=self.d_int,
            dims=self.dims, build_time=self.build_time
        )
        for d in self.dims:
            np.save(os.path.join(self.cache_dir, f"P_{d}.npy"), self.proj_mat[d])

    # ─────────────── Query Operations ───────────────

    def bound_stage1(self, query: np.ndarray) -> np.ndarray:
        """
        Compute Stage 1 Cauchy-Schwarz bound for ALL vectors.
        Returns B1: (N,) float32. All computation is mmap'd.
        """
        q = query.astype(np.float32).flatten()
        d = self.dims[0]
        q_proj = q @ self.proj_mat[d].T
        q_norm = max(np.linalg.norm(q), 1e-10)
        q_residual = math.sqrt(max(0, q_norm ** 2 - np.linalg.norm(q_proj) ** 2))
        # mmap transparent: _proj_fp[d] @ q_proj + _error_fp[d] * q_residual
        B1 = self._proj_fp[d] @ q_proj + self._error_fp[d] * q_residual
        return B1 + 1e-5

    def bound_stage2(self, query: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """
        Compute Stage 2 bound for a subset of indices.
        Returns B2: (len(idx),) float32.
        """
        q = query.astype(np.float32).flatten()
        d = self.dims[-1]
        q_proj = q @ self.proj_mat[d].T
        q_norm = max(np.linalg.norm(q), 1e-10)
        q_residual = math.sqrt(max(0, q_norm ** 2 - np.linalg.norm(q_proj) ** 2))
        # mmap indexed: reads only the needed rows
        proj_sub = self._proj_fp[d][idx]
        err_sub = self._error_fp[d][idx]
        B2 = proj_sub @ q_proj + err_sub * q_residual
        return B2 + 1e-5

    def exact_scores(self, idx: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Exact cosine scores for a subset of indices (mmap'd)."""
        q = query.astype(np.float32).flatten()
        qn = max(np.linalg.norm(q), 1e-10)
        vecs = self._emb_fp[idx]
        scores = vecs @ q
        nv = np.maximum(np.linalg.norm(vecs, axis=1), 1e-10)
        return scores / (nv * qn)

    # ─────────────── Search ───────────────

    def search(self, query: np.ndarray, k: int = 10,
               keep_ratio: float = 0.15, max_candidates: int = 200,
               final_topk: int = 50):
        """
        Full search pipeline using mmap'd cache.

        Returns:
          indices: np.ndarray of top-k indices
          scores: np.ndarray of top-k scores
          profile: dict with search profile
        """
        q = query.astype(np.float32).flatten()
        t0 = time.time()
        prof = {}

        # Stage 1
        t1 = time.time()
        B1 = self.bound_stage1(q)
        prof["stage1_ms"] = (time.time() - t1) * 1000

        keep1 = min(max(int(self.n * keep_ratio), 50), max_candidates, self.n)
        if self.n <= keep1:
            idx1 = np.arange(self.n)
        else:
            idx1 = np.argpartition(-B1, max(0, keep1 - 1))[:keep1]
        prof["stage1_survivors"] = len(idx1)

        # Stage 2
        t2 = time.time()
        B2 = self.bound_stage2(q, idx1)
        e1s = self._error_fp[self.dims[0]][idx1]
        e2s = self._error_fp[self.dims[-1]][idx1]
        imp = max(np.mean(e1s), 1e-9) / max(np.mean(e2s), 1e-9)

        if imp < 1.2:
            scores = B1[idx1]
        elif imp < 2.0:
            mu = max(np.mean(e1s), 1e-9)
            alpha = (1.0 / (1.0 + np.exp(-(e1s - e2s) / mu * 0.5))) * (imp - 1.2) / 0.8
            scores = B1[idx1] + alpha * (B2 - B1[idx1])
        else:
            mu = max(np.mean(e1s), 1e-9)
            alpha = 1.0 / (1.0 + np.exp(-(e1s - e2s) / mu * 0.5))
            scores = B1[idx1] + alpha * (B2 - B1[idx1])

        prof["stage2_ms"] = (time.time() - t2) * 1000
        prof["improvement_ratio"] = float(imp)

        # Stage 3
        keep2 = max(final_topk, min(200, len(idx1)))
        idx2 = idx1[np.argpartition(-scores, max(0, keep2 - 1))[:keep2]]
        exact = self.exact_scores(idx2, q)
        topk = np.argsort(-exact)[:k]
        prof["stage3_ms"] = (time.time() - t2) * 1000
        prof["total_ms"] = (time.time() - t0) * 1000

        return idx2[topk], exact[topk], prof

    # ─────────────── Cache Info ───────────────

    def size_bytes(self) -> int:
        """Total disk usage of cache files."""
        total = 0
        for root, dirs, files in os.walk(self.cache_dir):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
        return total

    def size_mb(self) -> float:
        return self.size_bytes() / (1024 * 1024)

    def clear(self):
        """Delete all cache files from disk."""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def stats(self) -> dict:
        return {
            "n": self.n,
            "d_in": self.d_in,
            "dims": list(self.dims),
            "d_int": round(float(self.d_int), 1) if self.d_int else None,
            "build_time_s": round(self.build_time, 3),
            "cache_size_mb": round(self.size_mb(), 1),
            "cache_dir": self.cache_dir,
            "ready": self._ready,
        }
