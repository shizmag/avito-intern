# Model and dependency disclosure

All external models used in this repository are explicitly provisioned, fingerprinted with SHA-256 manifests, and executed offline without silent runtime network calls.

## Transformer Models

| Model Name | Hugging Face ID | Local Path | Revision | License | Purpose | Remote Code |
|---|---|---|---|---|---|---|
| Multilingual-E5-Base | `intfloat/multilingual-e5-base` | `artifacts/models/multilingual-e5-base` | `d128750597153bb5987e10b1c3493a34e5a4502a` | MIT | Dense candidate retrieval (512-dim embedding) | False |
| Paraphrase-MiniLM-L12 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `artifacts/models/paraphrase-multilingual-MiniLM-L12-v2` | `bf3bf13ab40c3157080a7ab344c831b9ad18b5eb` | Apache-2.0 | Lightweight dense baseline / smoke testing | False |
| Jina Reranker v2 | `jinaai/jina-reranker-v2-base-multilingual` | `/Volumes/happy-disk/models/reranker/jina-reranker-v2-base-multilingual` | `9cfeff2df7d40d1b78e75e5e9cebec92a99813c9` | CC-BY-NC-4.0 | Multilingual cross-encoder candidate reranker (offline ablation only) | True (`trust_remote_code=True` required for custom XLM-RoBERTa architecture) |

## Licenses and Terms of Use

1. **Jina Reranker v2 Multilingual** (`jinaai/jina-reranker-v2-base-multilingual`):
   - **License**: Creative Commons Attribution-NonCommercial 4.0 International (**CC-BY-NC-4.0**).
   - **Architecture**: 278M-parameter multilingual sequence classification model derived from XLM-RoBERTa with custom FlashAttention and causal masking implementations.
   - **Usage in Repo**: Strictly an offline research/evaluation ablation over the top-500 candidate pool.
   - **Commercial Status**: Non-commercial license legally precludes commercial deployment without separate commercial licensing from Jina AI; excluded from the release champion pipeline.

2. **Multilingual-E5-Base** (`intfloat/multilingual-e5-base`):
   - **License**: MIT License.
   - **Usage in Repo**: Dense text encoder for bi-encoder retrieval.

3. **CatBoost** (`catboost`):
   - **License**: Apache License 2.0.
   - **Usage in Repo**: Feature-based fusion selector over lexical, dense, and two-tower ranks.

4. **bm25s** (`bm25s`):
   - **License**: MIT License.
   - **Usage in Repo**: Fast Robertson BM25 lexical retrieval.

5. **Two-Tower Model (v1 / v2)** (`src/avito_candidate_generation/two_tower.py`):
   - **License**: Custom PyTorch code; PyTorch framework is licensed under BSD-3-Clause.
   - **Usage in Repo**: Contrastive dual-encoder candidate generation for cold items.
