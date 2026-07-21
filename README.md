# Madhava-Sec 🔒

**Mathematically Guaranteed Agent Security Framework**

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue)](https://github.com/winnex-ai/madhava-sec/blob/main/LICENSE)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.21166403-blue)](https://zenodo.org/records/21166403)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)

---

## Benchmark — AgentHarm (416 cenários)

**Dataset:** [AgentHarm](https://huggingface.co/datasets/ai-safety-institute/AgentHarm) — 416 behaviors balanceados (208 harmful + 208 benign), 85 ferramentas, 8 categorias, 85 tool vectors.

### Resultados (Arquitetura 85D nativa + QR[64,128])

| Métrica | DIRECT | MADHAVA | RANDOM |
|:---|---:|:---:|:---:|
| **NDCG@10** | **1.0000** | **1.0000** | 0.5361 |
| **NDCG@20** | **1.0000** | **1.0000** | — |
| **Spearman ρ** | — | **0.8229** | — |
| **Pruning** | — | **89%** | — |

### Pipeline Scout + Factory (Budget de 1.500 LLM calls)

| Estágio | Calls | Resultado |
|:---|---:|:---|
| **SCOUT** (ε=0.7) | 300 | 68 sementes de ataque |
| **FACTORY** (GS diversity) | 1.200 | ~1.500 unique cells |
| **Eficiência** | **1.00** cells/call | |

### int8 Quantization

| Tipo | MSE | Cosine | Compressão |
|:---|:---:|:---:|:---:|
| Embeddings 384D | 1.4×10⁻⁷ | 0.999974 | 4× (float32 → int8) |

---

## Matemática

### Cauchy-Schwarz Upper Bound

```
⟨v, H⟩ = ⟨Pv + v_⟂, PH + H_⟂⟩
       = ⟨Pv, PH⟩ + ⟨v_⟂, H_⟂⟩        (termos cruzados = 0 por P ser ortogonal)
       ≤ ⟨Pv, PH⟩ + ‖v_⟂‖ · ‖H_⟂‖    (Desigualdade de Cauchy-Schwarz)
       = B₁(v, H)                      (Q.E.D.)
```

Se `B₁(v, H) + ε < best_score` → candidato **não pode matematicamente** superar o melhor já encontrado → PRUNE (0% falso negativo).

### Pipeline de Dois Estágios

**1. SCOUT** (20% do orçamento): ε=0.7 greedy — exploração estocástica para encontrar sementes de ataque.

**2. FACTORY** (80% do orçamento): Amplificação combinatorial com Gram-Schmidt no espaço de argumentos (URL, payload, path). Cada variação de argumento = 1 unique_cell no grader.

---

## Arquitetura

```
madhava_sec/
├── core.py              # MadhavaSecEngine — projeção QR + bound CS + modulação
├── verifier.py          # Verificação híbrida: semântica + gate sintático 
├── search.py            # Beam search + Gram-Schmidt diversity gate
├── pipeline.py          # Scout + Factory pipeline (2 estágios)
└── __init__.py

cpp/
├── madhava_core.h        # Core C++ com SIMD (AVX2+FMA)
├── madhava_sec_benchmark.cpp
└── Makefile

kaggle/
├── madhava_sec_kaggle_submission.py   # AttackAlgorithm para competição
└── deploy_madhava_sec_v8.py            # Deploy automático
```

---

## Reprodução

```bash
pip install numpy scikit-learn sentence-transformers pandas requests

# Benchmarks (Python)
PYTHONPATH=".:$PYTHONPATH" python3 madhava_sec_agentharm_benchmark.py

# C++ Core
cd cpp && make && ./madhava_sec_benchmark
```

---

## Kaggle Competition

O Madhava-Sec compete em:
```
https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks
```

Melhor submissão baseada no framework: **V108** (notebook, formato V44 comprovado).

---

## Licença

**Business Source License 1.1 (BSL 1.1)**

Permite estudo e teste não-produtivo. Uso comercial requer licença separada.

Contato: [pay@winnex.ai](mailto:pay@winnex.ai)

---

<p align="center">
  <strong>Winnex AI — Trust Infrastructure for Regulated Enterprise AI</strong><br>
  <a href="https://zenodo.org/records/21166403">Zenodo</a> ·
  <a href="https://github.com/winnex-ai">GitHub</a> ·
  <a href="mailto:pay@winnex.ai">Contato</a>
</p>
