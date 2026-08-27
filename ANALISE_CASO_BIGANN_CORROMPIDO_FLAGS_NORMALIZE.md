# O Caso do BIGANN Corrompido — e o Nascimento do Sistema de Flags

**Data:** 2026-08-27
**Status:** VERIFICADO — caso documentado, causa raiz provada, solução (sistema de flags) implementada no winnex-ai-normalize 1.1.0
**Método:** reconstrução a partir dos documentos de análise + código real + medições locais

---

## 1. O que aconteceu (a descoberta)

O dataset oficial do Kaggle `shurangwu/bigann-100m` — usado para validar o motor Madhava em escala (10M, 100M) — estava **silenciosamente corrompido**. O problema não era nos dados brutos, era no **alinhamento entre o dataset e o ground truth oficial**:

> O `base.u8bin` do dataset tem **ordem de vetores reordenada** em relação ao `base.u8bin` canônico do BIGANN. O ground-truth oficial (`unif_groundtruth_10k.bin`) aponta para os IDs da **ordem canônica** — portanto aponta para os **vetores errados** neste base.

**A assinatura da corrupção (verificada empiricamente):**
- O GT top-1 id **nunca** é o vizinho verdadeiro: **0/500 hits** no scan exato.
- O L2² dos IDs do GT ≈ **aleatório** (não são vizinhos de verdade).
- O `gt_validated=false, gt_recall_scan_exact=0.0` no runtime do notebook.
- Um GT de teste (`/tmp/gt_test.bin`) tinha IDs **impossíveis** (1.036.831.949 > 60.000 vetores) — sinal inconfundível de que os IDs apontam para outro universo.

---

## 2. O diagnóstico (a segunda descoberta — o offset ×2)

A investigação (`DEBUG_L2_GT_OFFSET.md`) revelou que havia **duas camadas** de desalinhamento:

**Camada 1 — offset de amostragem ×2:**
```
GT[i]  ↔  query 2i     (o GT amostra a cada 2 queries)
```
- Contra a query alinhada `2i`: 50/50 OK.
- Contra a query `i` (sem offset): apenas 13/50 OK.

**Camada 2 — distâncias denormal (~0):**
- As distâncias oficiais do GT são todas **2e-35 a 1e-39** — lixo numérico.
- O arquivo só carrega os **IDs**; as distâncias são inutilizáveis.
- **Consequência:** a avaliação deve usar apenas os IDs, nunca as distâncias.

**O efeito no recall medido:**
- Antes do diagnóstico: recall@10 = **0.006** (parecia bug fatal do motor).
- Após o offset ×2: recall@10 = **0.426** (o motor sempre esteve certo).
- Em 100M com GT denso: **NDCG@10 = R@10 = 1.0000** (o motor é exato).

**A lição técnica imediata:** o "recall 0.006" era um **artefato de avaliação**, não uma falha do motor. O motor Madhava estava correto o tempo todo — quem estava errado era o *dataset + GT* que alimentavam a avaliação.

---

## 3. A correção no benchmark (a resposta honesta)

A resposta da Winnex foi **recusar o GT corrompido** e definir um ceiling válido:

```
GT oficial corrompido (reordenado)  →  IGNORADO (provado inválido)
Ceiling válido: search_exact do próprio motor no MESMO subset
```

Isso garante comparação **"apples-to-apples"**: todos os métodos (madhava, HNSW, IVF, FlatIP) medem recall vs o **mesmo** exact-scan sobre o **mesmo** subset. Sem usar o GT quebrado, os resultados ficaram:

| Método | R@10 (vs ceiling exato) | Build |
|---|---|---|
| Exact-scan ceiling | 1.0000 | 3.0s |
| **Madhava bound** | **0.9960** | **2.0s** |
| **Madhava speed GPU** | **1.0000** | 1.5s |
| HNSW(ef=128) | 0.9760 | 159s |
| IVF-PQ | 0.4780 | 10s |

**Por que é honesto:** não reporta recall contra um GT corrompido; reporta contra um ceiling que o próprio motor define e valida. O README do madhava documenta tudo com a correção `⚠️ GT-validity correction (2026-08-08)` e marca os benchmarks antigos como `superseded` / "não devem ser citados".

---

## 4. A lição estrutural (o gargalo de recall mal atribuído)

O caso BIGANN expôs um problema **estrutural** que vai muito além de um dataset corrompido:

### O recall fim-a-fim é determinado ANTES do motor

```
recall_fim_a_fim ≈ qualidade dos dados de entrada  ×  prefilter heurístico  ×  bound  ×  postfilter exato
                        ▲                                ▲
                   (o BIGANN FALHOU aqui)           (depende de k1_fraction)
```

O motor de busca — com toda a sua prova Cauchy-Schwarz de 0 violações — **só pode ser tão bom quanto os dados que recebe**:

1. **Se o dataset é corrompido** (ordem reordenada, GT errado, NaN, dimensão errada): o motor devolve top-K "corretos" para dados que não significam o que se pensa. `bound_violations == 0` continua verdadeiro — mas o recall **semântico** é zero. **O motor não tem como saber.**

2. **Se o embedding é de baixa qualidade** (provider de terceiros, failover que troca de espaço vetorial no meio, modelo fraco): o recall é limitado pela qualidade do embedding, **antes mesmo de tocar o motor**. Um `recall@10=1.0` vs o ceiling pode ser 1.0 **enquanto o recall semântico real é baixo** — o benchmark é self-consistente e cego à relevância real.

3. **Se o prefilter é heurístico** (`k1_fraction=0.05`): em dimensão alta com bound frouxo, os 95% cortados **não têm prova** — são heurística pura. A "garantia por documento" só vale para o que *sobreviveu* ao corte.

### O erro de atribuição

> O recall fim-a-fim estava sendo tratado como **responsabilidade do motor de busca**, quando na verdade ele é **determinado muito antes** — na **qualidade dos dados que entram no sistema**.

O BIGANN foi o sintoma perfeito: o motor Madhava **nunca falhou** — foi o **dado de entrada** que estava corrompido, e **nada no pipeline validava isso**. Um dataset corrompido entrou silenciosamente, distorceu os benchmarks por semanas, e só foi descoberto por uma auditoria forense dedicada.

---

## 5. A solução — o sistema de flags no winnex-ai-normalize

Se o recall é decidido na entrada, a validação deve acontecer **na entrada**. O winnex-ai-normalize 1.1.0 tornou-se o **guardião da qualidade de dados**:

### 5.1 A validação é o próprio motor (sem reimplementar matemática)

A flag não é uma heurística externa — é a **validação nativa do motor Madhava**:

```
Cauchy-Schwarz (upper bound):
    UB(v,q) = ⟨Pv,Pq⟩ + e(v)·e(q)
    UB(v,q) < threshold(K)  ⟹  v é matematicamente IMPOSSÍVEL de estar no top-K

O QualityValidator LANÇA essa prova sobre queries-semente e CAPTURA o
conjunto excluído (audit_ids / audit_threshold / pruned_by_bound /
pruned_by_prefilter). O conjunto capturado É a resposta da flag.
```

### 5.2 As flags (agnósticas ao dataset)

| Flag | O que valida | Severidade |
|---|---|---|
| `dataset.foldable` | **cobertura da prova CS** (fração do corpus provadamente fora do top-K) | pass/warn |
| `dataset.nan` | NaN/inf (corrompe scores com 0 violações) | **fail** |
| `dataset.degenerate` | variância zero | **fail** |
| `embedding.resolution` | gap top-1 vs top-K (qualidade do embedding de terceiros) | warn |
| `embedding.provider_drift` | drift do espaço entre batches | warn/fail |
| `embedding.cross_provider` | troca de provider = espaços incomparáveis | warn |
| `embedding.dim_shift` | mudança de dimensão (a classe do offset ×2) | **fail** |
| `embedding.anisotropy` | colapso do embedding (artifact LLM) | warn |
| `integrity.corpus_alignment` | corpus vs referência desalinhado (a classe do BIGANN) | **fail** |

### 5.3 O roteador de configuração (protege o motor de dados não estruturados)

O validador **decide a config do motor com base na prova que o motor produziu** — protegendo o recall quando os dados são difíceis:

- **Prova ≥ 50%** → `pca_corpus`, `k1=0.05` (foldable, bound restaurado)
- **Prova 20-50%** → `random`, `k1=0.10` (moderado)
- **Prova < 20%** → probe com `pca_corpus` do próprio motor; se provar ≥50% → pca; senão → `random`, `k1=0.20` (o prefilter é o recall gate)
- **`early_exit=False` sempre** (o fix P0: em dim ≥ 384 o early-exit quebra recall)

### 5.4 Validação real (medida)

| Dataset | Prova CS do motor | Roteamento |
|---|---|---|
| **BIGANN base.u8bin** (60K×128, L2) | **99.3%** provado fora do top-10 | `pca_corpus`, k1=0.05 |
| **arXiv OpenAI** (d=1536, random) | **0.0%** (Fold Limit: bound frouxo) | `random`, k1=0.20 |
| **arXiv OpenAI** (d=1536, probe PCA) | **80.2%** | `pca_corpus`, k1=0.05 |
| **Qwen2.5-0.5B quantizado** (d=896) | 0.0% (14 textos, vizinhos no top-10) | `random`, k1=0.20 |

### 5.5 Como o sistema detecta a classe do BIGANN

A corrupção do BIGANN tinha a assinatura: **ordem dos vetores ≠ ordem que o GT espera**. O sistema de flags detecta essa classe de problema de forma **genérica** (sem lógica específica de BIGANN):

- **`reference=` (corpus vs referência):** se o chamador fornecer um ground-truth ou um lote de referência, o validador compara os centroids → `corpus_alignment` FAIL quando divergem.
- **`dim_shift`:** se um lote chega com dimensão diferente da anterior → FAIL (a classe do offset).
- **`excluded_seed_set` determinístico:** o conjunto capturado é reproduzível (mesma seed → mesmo conjunto) — um auditor pode verificar que a prova é real.

---

## 6. Conclusão

O caso do BIGANN corrompido ensinou algo que nenhum benchmark corrigido captura totalmente:

1. **O motor nunca foi o problema.** O Madhava provou (com 0 violações) estar correto em todos os subsets — o recall 0.006 era o *dataset de entrada* distorcendo a avaliação.

2. **O recall fim-a-fim é decidido na entrada, não na busca.** Dataset corrompido, embedding de terceiros fraco, failover que troca de espaço — tudo isso determina o recall **antes** do motor. Tratá-lo como responsabilidade do motor é o erro de atribuição que o BIGANN expôs.

3. **O sistema de flags é a resposta estrutural.** O winnex-ai-normalize agora valida a qualidade de dados/embeddings na entrada, usando a **própria prova Cauchy-Schwarz do motor** como a flag, e roteia a configuração para preservar recall — protegendo o motor de dados não estruturados.

**A lição final:** a honestidade matemática da Winnex não é só não inflar benchmarks — é **recusar-se a reportar contra dados que não confia**. O BIGANN corrompido foi ignorado (não "corrigido"); o ceiling válido foi definido e usado; e agora o sistema de flags impede que a próxima corrupção entre silenciosamente.

---

## 7. Fontes

- `ANALISE_HONESTIDADE_KAGGLE_FORENSE.md` — o Kaggle como laboratório forense (correção de GT falso)
- `DEBUG_L2_GT_OFFSET.md` — o diagnóstico do offset ×2 e das distâncias denormal
- `winnex-madhava/README.md` §"GT-validity correction (2026-08-08)" e §"Real benchmark" — a correção documentada
- `kaggle/bench_production_4frentes/benchmark_production_4frentes.py` — ceiling exato no mesmo subset
- `winnex-ai-normalize/winnex_ai_normalize/core/quality.py` — o QualityValidator (a solução)
- Medições locais: BIGANN 60K (proof 99%), arXiv d=1536 (random 0% → PCA 80%), Qwen2.5 quantizado
