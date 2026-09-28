# Jina Multilingual Reranker v2 Evaluation Report

## Executive Summary

We implemented and objectively evaluated the multilingual cross-encoder reranker `jina-reranker-v2-base-multilingual` on top of the canonical top-500 candidate pool.

- **Candidate Pool @500 Upper Bound**: **0.914657** (91.47% theoretical ceiling).
- **RRF (k=60) Baseline**: **Recall@50 = 0.522167** (held-out subset), **0.497621** (full validation).
- **CatBoost Baseline**: **Recall@50 = 0.535119** (held-out subset), **0.506448** (full validation).
- **Jina Reranker v2 Standalone**: **Recall@50 = 0.084286** (-45.08 percentage points vs CatBoost).
- **CatBoost + Jina Hybrid**: **Recall@50 = 0.409639** (-12.55 percentage points vs baseline CatBoost).
- **Subset Gate Decision**: **REJECT**. Pretrained zero-shot Jina cross-encoder severely underperforms domain-specific retrieval signals on Avito e-commerce classifieds data.
- **Champion Decision**: **CatBoost Fusion** remains the selected champion pipeline.

---

## 1. Candidate Pool & Upper Bound Analysis

The candidate pool is constructed from the union of top-500 candidates retrieved by:
1. **BM25** (`bm25s` Robertson)
2. **Dense Multilingual-E5-Base** (exact cosine search)
3. **Two-Tower v1** (contrastive cold-item tower)

Candidates are merged using **Reciprocal Rank Fusion (RRF, k=60)** up to a maximum limit of **500 unique item IDs** per query.

| Pool Specification | Coverage / Recall | Notes |
|---|---:|---|
| Full Generator Union (up to 1500 items) | **0.962894** (96.29%) | Theoretical ceiling of all 3 retrievers |
| Canonical Candidate Pool @500 (RRF) | **0.914657** (91.47%) | Actual candidate pool provided to reranker |
| Candidate Pool @300 | **0.881262** (88.13%) | Depth ablation |
| Candidate Pool @200 | **0.820037** (82.00%) | Depth ablation |
| Candidate Pool @100 | **0.684601** (68.46%) | Depth ablation |

---

## 2. Model & Pair Representation

- **Model ID**: `jinaai/jina-reranker-v2-base-multilingual`
- **Local Artifact**: `/Volumes/happy-disk/models/reranker/jina-reranker-v2-base-multilingual`
- **Commit SHA**: `9cfeff2df7d40d1b78e75e5e9cebec92a99813c9`
- **Architecture**: 278M parameter XLM-RoBERTa cross-encoder with custom FlashAttention.
- **License**: **CC-BY-NC-4.0 (Creative Commons Attribution-NonCommercial 4.0 International)**. Restricted to non-commercial research and evaluation.
- **Query Representation**:
  ```text
  Запрос: <search_query>
  Фильтры: <search_infm_params_text>
  ```
- **Item Representation & Priority Truncation**:
  ```text
  Название: <item_title_raw>
  Параметры: <item_infm_params_text>
  Описание: <truncated item_description_raw>
  ```
  Title and parameters are preserved in full; description is truncated to fit the context budget (`max_length=256`).

---

## 3. Comparative Benchmark on Held-out Subset (500 Queries, 250,000 Pairs)

All models evaluated apples-to-apples on the exact same deterministic representative validation subset (500 queries, 539 positive pairs):

| Selector / Reranker | Recall@10 | Recall@20 | Recall@50 | Recall@100 | Recall@200 | Delta vs CatBoost |
|---|---:|---:|---:|---:|---:|---:|
| **Candidate Pool Ceiling (@500)** | — | — | — | — | — | **+37.95%** (Upper Bound) |
| **BM25 (bm25s)** | 0.201000 | 0.273667 | 0.416786 | 0.561186 | 0.684186 | -11.83% |
| **Dense (multilingual-E5)** | 0.212286 | 0.288286 | 0.444952 | 0.585786 | 0.713638 | -9.02% |
| **Two-Tower v1** | 0.078333 | 0.142667 | 0.275000 | 0.448333 | 0.620590 | -26.01% |
| **RRF (k=60)** | 0.210333 | 0.326333 | 0.522167 | **0.700405** | **0.839738** | -1.30% |
| **CatBoost Selector** | **0.245333** | **0.352119** | **0.535119** | — | — | **Baseline Champion** |
| **Jina Reranker v2 Standalone** | 0.016000 | 0.030000 | 0.084286 | — | — | **-45.08%** |
| **CatBoost + Jina Hybrid** | 0.146185 | 0.244578 | 0.409639 | — | — | **-12.55%** |

*\* Note: Rerankers and selectors (CatBoost Selector, Jina Standalone, and CatBoost+Jina Hybrid) evaluate a final ranking truncated to top-50 items per query. Reporting Recall@100 or Recall@200 on an output list physically capped at 50 would mechanically duplicate the Recall@50 figure; these columns are therefore correctly recorded as '—' (not applicable).*

---

## 4. Candidate Pool Depth Ablation

This ablation evaluates the relationship between **candidate input depth** (number of top candidates supplied by the RRF generator pool to Jina) and the resulting **final reranked Recall@50**:

| Candidate Input Depth | Pool Candidate Recall Ceiling | Jina Final Reranked Recall@50 | Pairs Processed |
|---|---:|---:|---:|
| Top-100 | 0.684601 | 0.093667 | 50,000 |
| Top-200 | 0.820037 | 0.088000 | 100,000 |
| Top-300 | 0.881262 | 0.084000 | 150,000 |
| Top-500 | 0.914657 | 0.084286 | 250,000 |

**Analysis**:
Even though expanding the candidate input depth from 100 to 500 raises the candidate pool coverage ceiling from 68.46% to 91.47%, Jina's final reranked Recall@50 slightly degrades (0.0937 $\to$ 0.0843). Supplying deeper, lower-precision candidates provides more distractor items that the unadapted zero-shot cross-encoder ranks above true positives.

---

## 5. Error & Position Analysis

### Four-Quadrant Error Breakdown (500 Queries)
- **Both Hit** (RRF in top-50 AND Jina in top-50): **27** queries (5.4%)
- **RRF Hit / Jina Miss** (RRF found positive, Jina demoted it outside top-50): **238** queries (47.6%)
- **RRF Miss / Jina Hit** (Novel recoveries by Jina): **18** queries (3.6%)
- **Both Miss**: **217** queries (43.4%)

### Original RRF Rank Distribution of Positives in Jina Top-50
- From RRF rank 1–50: **27** items
- From RRF rank 51–100: **8** items
- From RRF rank 101–200: **7** items
- From RRF rank 201–300: **2** items
- From RRF rank 301–500: **1** item

---

## 6. Hypothesized Technical Drivers of Zero-Shot Degradation

The observed degradation is consistent with the following hypotheses (representing plausible explanations rather than conclusively established facts):

1. **Domain Gap Hypothesis**: Observed behavior is consistent with the hypothesis that the cross-encoder, pre-trained primarily on generic web search and passage retrieval benchmarks, experiences substantial domain shift when applied to high-intent Russian classifieds queries where exact technical parameters, brand identifiers, and category taxonomy dominate over broad semantic topicality.
2. **Semantic Distraction / Verbosity Bias Hypothesis**: A possible explanation is that uncalibrated cross-attention may assign disproportionately high relevance scores to verbose item descriptions containing tangential keyword overlaps, overshadowing concise listings with exact technical specifications, part numbers, or service parameters.
3. **Absence of Calibrated Tabular and Interaction Priors**: Observed results are consistent with the hypothesis that purely textual cross-encoders discard structural ranking signals—such as multi-generator rank consensus, BM25 Robertson score magnitudes, and learned categorical embeddings—which the CatBoost selector explicitly exploits.

---

## 7. Champion Selection & Decision

**Selected Champion**: **CatBoost Selector** over BM25 + Multilingual-E5 + Two-Tower v1 candidate union.
- **Held-out Recall@50**: **0.535119** (subset) / **0.506692** (full cold-item validation).
- **Reranker Gate Decision**: **REJECT**. Jina Reranker v2 is rejected from the champion pipeline on two independent grounds:
  1. **Empirical Quality**: Severe performance degradation (-45.08 percentage points vs CatBoost; 0.0843 vs 0.5351 Recall@50).
  2. **Licensing**: Jina is distributed under **CC-BY-NC-4.0 (Non-Commercial)**, which legally precludes commercial production deployment without proprietary commercial licensing. In contrast, the champion pipeline consists entirely of commercially permissive open-source components (MIT, Apache-2.0, BSD-3-Clause).
- Jina Reranker v2 is retained strictly as an offline ablation and reproducibility artifact.
