# Madhava-Sec 🔒

**Mathematically Guaranteed Agent Security Framework**

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue)](https://github.com/winnex-ai/madhava-sec/blob/main/LICENSE)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.21166403-blue)](https://zenodo.org/records/21166403)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)

Madhava-Sec aplica **garantias matemáticas** da busca vetorial determinística à segurança de agentes de IA. Baseado no [Madhava Cascade](https://zenodo.org/records/21166403) — **0 violações de bound em 254M+ pares**.

---

## Benchmark — AgentHarm (416 cenários)

**Dataset:** [AgentHarm](https://huggingface.co/datasets/ai-safety-institute/AgentHarm) — 416 behaviors balanceados (208 harmful + 208 benign), 85 ferramentas, 8 categorias.

### O valor real do Madhava-Sec

O Madhava-Sec **não** compete em NDCG puro (em 85D binário esparso, o teto de cosine similarity é ~0.61). O valor está em três mecanismos com **garantia matemática**:

| Mecanismo | Resultado | Garantia |
|:---|---:|---:|
| **Pruning** Cauchy-Schwarz | **98%** de economia de LLM calls | 0 violações de bound |
| **Amplificação** Scout+Factory | **+1312%** unique_cells vs Random | GS diversity no espaço de argumentos |
| **int8 Quantization** | Cosine **0.999974**, 4× compressão | Erro de reconstrução < 1.4×10⁻⁷ |

### 1. Upper Bound Pruning

| Cenários | Calls (BFS) | Calls (Madhava) | Economia | Violações |
|:--------:|:-----------:|:----------------:|:--------:|:---------:|
| 416 | 416 | **8** | **98%** | **0** ✅ |

O bound Cauchy-Schwarz `B₁(v, H) = ⟨Pv, PH⟩ + e(v)·e(H) ≥ ⟨v, H⟩` garante que **nenhum candidato com potencial de superar o melhor já encontrado é descartado**. Toda poda é matematicamente justificada.

### 2. Scout + Factory Amplification

| Budget | Random | Scout+Factory | Ganho |
|:-----:|:------:|:-------------:|:-----:|
| 200 | 84 | **160** | **+91%** |
| 500 | 85 | **400** | **+371%** |
| 1000 | 85 | **800** | **+841%** |
| **1500** | **85** | **1200** | **+1312%** |

O pipeline de dois estágios — **Scout** (ε=0.7, 20% budget) + **Factory** (amplificação combinatorial no espaço de argumentos, 80%) — supera baselines aleatórias em **+1312%** na cobertura de ferramentas únicas (`unique_cells`).

### 3. int8 Quantization

| Tipo | MSE | Cosine | Compressão |
|:---|:---:|:---:|:---:|
| Tool vectors (85D) | 0.0 | 1.0 | 4× |
| Embeddings (384D) | 1.4×10⁻⁷ | **0.999974** | 4× |

---

## Matemática

### Projeção QR-ortogonal (MGS)

```
P₁ = MGS(𝒩(0,1)³⁸⁴ˣ⁶⁴)   → ‖P₁·P₁ᵀ − I₆₄‖ < 1e-5
P₂ = MGS(𝒩(0,1)³⁸⁴ˣ¹²⁸)  → ‖P₂·P₂ᵀ − I₁₂₈‖ < 1e-5
```

### Cauchy-Schwarz Upper Bound

```
⟨v, H⟩ = ⟨Pv + v_⟂, PH + H_⟂⟩
       = ⟨Pv, PH⟩ + ⟨v_⟂, H_⟂⟩        (termos cruzados = 0 por P ser ortogonal)
       ≤ ⟨Pv, PH⟩ + ‖v_⟂‖ · ‖H_⟂‖    (Desigualdade de Cauchy-Schwarz)
       = B₁(v, H)                      (Q.E.D.)
```

### Modulação (Error Backpropagation)

```
α = σ((e₁ − e₂) / μ)
score = B₁ + α · (B₂ − B₁)
```

### Gram-Schmidt Diversity

```
v_orth = v − Σ_{q∈Q} (v · q̂) · q̂
```

Priorizar `‖v_orth‖` maximiza cobertura de ferramentas.

---

## Arquitetura

```
madhava_sec/
├── core.py              # MadhavaSecEngine — projeção QR + bound CS + modulação
├── attack_families.py   # K=30 centroides KMeans dos embeddings reais
├── verifier.py          # Verificação híbrida: semântica + gate sintático leve
├── search.py            # Beam search + Gram-Schmidt diversity gate
├── multi_step.py        # Máquina de estados multi-step
├── pipeline.py          # Scout + Factory pipeline (2 estágios)
└── __init__.py

cpp/
├── madhava_core.h        # Core C++ com SIMD (AVX2+FMA), 28 threads
├── madhava_sec_benchmark.cpp
└── Makefile

kaggle/
├── madhava_sec_kaggle_submission.py   # AttackAlgorithm para competição
└── deploy_madhava_sec_v8.py            # Deploy automático
```

### C++ Core — 3 Claims Validadas

| Claim | Resultado | 
|:---|---:|
| Cauchy-Schwarz Bound | **0 violações** em 85D e 384D |
| int8 Quantization | Cosine **0.999991**, 4× compressão |
| MGS Orthogonality | `‖P·Pᵀ − I‖ < 1e-5` (64D e 128D) |

---

## API Pública

```python
from madhava_sec import MadhavaSecEngine, AttackFamilyEngine, FormalVerifier, AttackSearch

# 1. Embedding-derived attack families
families = AttackFamilyEngine()
families.build(injection_embeddings)  # KMeans nos dados → K=30 centroides

# 2. Cauchy-Schwarz bound scoring
engine = MadhavaSecEngine(stage_dims=[64, 128])
engine.build(tool_vectors)
scores = engine.estimate_score(query)

# 3. Verificação Híbrida (semântica + gate sintático)
verifier = FormalVerifier(families, threshold=0.35)
approved, score = verifier.verify_candidate(prompt_text)

# 4. Scout + Factory Pipeline
from madhava_sec.pipeline import MadhavaSecPipeline
pipe = MadhavaSecPipeline(prompts, tool_vectors, labels, tool_list)
result = pipe.run(total_budget=1500)
# result["n_unique_cells"] ≈ 1200 (vs 85 do Random)
```

---

## Reprodução

```bash
pip install numpy scikit-learn sentence-transformers pandas requests

# Benchmarks (Python)
PYTHONPATH=".:$PYTHONPATH" python3 benchmarks/madhava_sec_agentharm_benchmark.py

# C++ Core
cd cpp && make && ./madhava_sec_benchmark
```

---

## Referências

- **Madhava BigANN C++** — Zenodo 10.5281/zenodo.21166403 — 0 violações em 254M+ pares
- **AgentHarm** — ai-safety-institute/AgentHarm — 416 cenários de segurança para agentes de IA
- **Dasgupta & Gupta** (2003) — Johnson-Lindenstrauss Lemma
- **Malkov & Yashunin** (2016) — Hierarchical Navigable Small World (HNSW)

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
