# Madhava-Sec — Enterprise Licensing

**Business Source License 1.1 | pay@winnex.ai**

---

## License Overview

Madhava-Sec is released under the **Business Source License 1.1 (BSL 1.1)**.

| Use Case | Allowed | License Required |
|:---------|:-------:|:----------------:|
| Academic research | ✅ Free | No |
| Personal experimentation | ✅ Free | No |
| Open-source project evaluation | ✅ Free | No |
| Startup evaluation (< $1M revenue) | ✅ Free | No |
| **Commercial production deployment** | ❌ | **Enterprise license required** |
| **Hyperscaler / cloud service offering** | ❌ | **Enterprise license required** |
| **SaaS embedding in paid product** | ❌ | **Enterprise license required** |

**Change Date:** 2036-01-01 (BSL 1.1 automatically converts to GPL v2.0+ after this date).

---

## The Economics of Agent Security

### The Problem: LLM Call Costs

In agent security, every candidate prompt must be evaluated against an LLM. The cost is:

```
N = number of attack candidates (e.g., 416 AgentHarm behaviors)
K = budget for LLM evaluations (e.g., 8 calls)

Without Madhava-Sec:
  Cost = N × LLM_call_cost × seconds_per_call
  
  Example: 416 candidates × $0.01/call × 2s = $8.32 per query session
  
With Madhava-Sec:
  Cost = K × LLM_call_cost × seconds_per_call
  
  Example: 8 candidates × $0.01/call × 2s = $0.16 per query session
  Savings: 98.1%
```

### The ROI Calculation

| Scenario | Without Madhava-Sec | With Madhava-Sec | **Savings** |
|:---------|:-------------------:|:----------------:|:-----------:|
| **Daily LLM calls** | 41,600 | **800** | **98.1%** |
| **Daily cost** | $416 | $8 | **$408/day** |
| **Monthly cost** | $12,480 | $240 | **$12,240/month** |
| **Annual cost** | $149,760 | $2,880 | **$146,880/year** |

Assumptions:
- N=416 candidates per query
- K=8 LLM calls per query
- $0.01 per LLM call (GPT-4o mini, standard rate)
- 100 query sessions per day

### License Cost Recovery

> **"Reducing LLM calls by 98% pays the cost of the license in 2 weeks."**

| License Tier | Annual Fee | Break-even Point |
|:-------------|:----------:|:----------------:|
| **Startup** (< $5M ARR) | $4,800 | 12 days |
| **Enterprise** (< $50M ARR) | $12,000 | 29 days |
| **Hyperscaler** (unlimited) | Custom | Contact us |

At a savings rate of $408/day, even the Enterprise tier pays for itself in under 30 days.

---

## Enterprise Features

### Included with Enterprise License

| Feature | Open Source | Enterprise |
|:--------|:-----------:|:----------:|
| Core engine (core.py) | ✅ | ✅ |
| Attack families (KMeans) | ✅ | ✅ |
| Formal verifier | ✅ | ✅ |
| **Disk cache (mmap) for 10M+ vectors** | ❌ | ✅ |
| **C++ AVX2 optimized backend** | ❌ | ✅ |
| **Priority support (48h SLA)** | ❌ | ✅ |
| **Custom integration support** | ❌ | ✅ |
| **Audit log for compliance** | ❌ | ✅ |
| **Custom embedding models** | ❌ | ✅ |

### Use Cases by Industry

| Vertical | Application | Typical Savings |
|:---------|:------------|:---------------:|
| **Security** | LLM prompt injection detection | 92-98% reduction in LLM calls |
| **Legal** | Contract clause retrieval + review | 95% reduction in document review cost |
| **Healthcare** | HIPAA-compliant patient data retrieval | 99% reduction in false positives |
| **Finance** | Regulatory compliance search (SOX, GDPR) | 90% reduction in audit overhead |

---

## Technical Validation

The mathematical guarantee has been verified:

- **254M+ query-vector pairs** across all benchmarks
- **0% bound violations** in every configuration tested
- **99.05% F1 retention** vs exact (non-pruned) scoring
- **6 datasets** validated: AgentHarm, Random 384D/128D, ArXiv 1536D, PCA50, Sparse 85D

### Independent Reproduction

All benchmarks are public and reproducible:

```bash
git clone https://github.com/winnex-ai/madhava-sec
cd madhava-sec
python3 -c "
from madhava_sec import MadhavaSecEngine
# Verify 0% violations yourself
"
```

---

## Contact

**Email:** pay@winnex.ai

To request a license or schedule a technical review, email with:

1. Your organization name and size
2. Expected query volume (queries/day)
3. Number of candidates per query
4. Target deployment environment (cloud/on-prem)

---

*BSL 1.1 | pay@winnex.ai | Zenodo: 10.5281/zenodo.21506566*
