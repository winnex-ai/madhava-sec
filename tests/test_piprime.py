"""Tests for PiPrimeNavigator — real π-based navigation."""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from madhava_sec.piprime import PiPrimeNavigator

rng = np.random.RandomState(42)

def _data(n=500, d=384):
    X = np.hstack([rng.randn(n,20)*3, rng.randn(n,d-20)*0.1]).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-10)
    return X

def test_build_creates_anchors():
    nav = PiPrimeNavigator(n_anchors=8).build(_data())
    assert nav.anchors.shape == (8, 384), f"Expected (8,384), got {nav.anchors.shape}"

def test_anchors_are_orthonormal():
    nav = PiPrimeNavigator(n_anchors=8).build(_data(500))
    G = nav.anchors @ nav.anchors.T
    err = np.abs(G - np.eye(8)).max()
    assert err < 1e-4, f"Orthonormality error: {err}"

def test_navigation_deterministic():
    nav = PiPrimeNavigator(n_anchors=8).build(_data())
    q = _data(1)[0]
    t1 = nav.navigate(q)
    t2 = nav.navigate(q)
    assert [x[0] for x in t1] == [x[0] for x in t2], "Navigation not deterministic"

def test_explore_produces_candidates():
    nav = PiPrimeNavigator(n_anchors=8).build(_data())
    q = _data(1)[0]
    cand = nav.explore(q, n_candidates=10)
    assert cand.shape[0] == 10, f"Expected 10 candidates, got {cand.shape[0]}"
    assert cand.shape[1] == 384

def test_d_int_estimated():
    nav = PiPrimeNavigator(n_anchors=8)
    X = _data()
    nav.build(X)
    assert nav.D_int is not None and nav.D_int > 0

def test_ap_weights_sum_to_one():
    nav = PiPrimeNavigator(n_anchors=8).build(_data())
    assert abs(nav.ap.sum() - 1.0) < 1e-6

def test_potential_returns_float():
    nav = PiPrimeNavigator(n_anchors=8).build(_data())
    q = _data(1)[0]
    u = nav.potential(0, q)
    assert isinstance(u, float)
    assert not np.isnan(u)

def test_update_works():
    nav = PiPrimeNavigator(n_anchors=8).build(_data())
    nav.update(0, 0.85)
    assert nav.anchor_scores[0] == 0.85
    assert nav.anchor_counts[0] == 1
    nav.update(0, 0.95)
    assert abs(nav.anchor_scores[0] - 0.9) < 1e-6

def test_stats():
    nav = PiPrimeNavigator(n_anchors=8).build(_data())
    s = nav.stats()
    assert s["K"] == 8
    assert s["built"] == True

def test_different_n_anchors():
    for K in [4, 16, 32]:
        nav = PiPrimeNavigator(n_anchors=K).build(_data())
        assert nav.anchors.shape[0] == K

def test_orthogonality_multiple_seeds():
    for seed in [42, 123, 999]:
        nav = PiPrimeNavigator(n_anchors=8, seed=seed).build(_data())
        G = nav.anchors @ nav.anchors.T
        err = np.abs(G - np.eye(8)).max()
        assert err < 1e-4, f"Seed {seed}: orth error {err}"

def test_explore_respects_n_candidates():
    nav = PiPrimeNavigator(n_anchors=8).build(_data())
    q = _data(1)[0]
    for n in [5, 20, 50]:
        cand = nav.explore(q, n_candidates=n)
        assert cand.shape[0] == n, f"Expected {n}, got {cand.shape[0]}"
