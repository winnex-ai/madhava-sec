"""
Madhava-Sec: Mathematically Guaranteed Agent Security Framework
===============================================================

PiPrime navigation + Cauchy-Schwarz bounds + Semantic Safety ensemble.

Architecture:
  PiPrime (π-based navigation)  →  Madhava-Sec (bounds)  →  Semantic Safety (ensemble)
    Exploracao do espaco de busca     Classificacao com garantia     Decisao multi-embedder

License: BSL 1.1 | pay@winnex.ai
"""
__version__ = "3.0.0"

from .core import MadhavaSecEngine, auto_configure, estimate_intrinsic_dim
from .piprime import PiPrimeNavigator
from .semantic import SafetyEnsemble
from .agent import AgentSecurityFramework
