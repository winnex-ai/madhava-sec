"""
Madhava-Sec: Mathematically Guaranteed Agent Security Framework
===============================================================
Embedding-derived attack detection. Zero hardcoding. Zero fallbacks.

License: BSL 1.1 | Author: Klenio Araujo Padilha
"""
__version__ = "2.0.0"
from .core import MadhavaSecEngine
from .attack_families import AttackFamilyEngine
from .verifier import FormalVerifier, SEVERITY_LEVELS
from .search import AttackSearch
