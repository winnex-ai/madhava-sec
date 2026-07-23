"""
Madhava-Sec: Mathematically Guaranteed Agent Security Scoring
=============================================================

⚠️  IMPORTANT WARNING  ⚠️
Mathematical Guarantee ≠ Semantic Guarantee.

0% false negatives on EMBEDDING COSINE SIMILARITY.
Does NOT guarantee semantic harmfulness detection.

See README.md Limitations section for details.

License: BSL 1.1 | pay@winnex.ai
"""
__version__ = "2.2.0"

# Show disclaimer on import
from .core import _show_disclaimer
_show_disclaimer()

from .core import MadhavaSecEngine, auto_configure, estimate_intrinsic_dim
from .attack_families import AttackFamilyEngine
from .verifier import FormalVerifier, SEVERITY_LEVELS
from .search import AttackSearch
