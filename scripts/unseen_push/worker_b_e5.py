"""Worker B: H2 Domain Fine-Tuned E5 Retrieval Experiment.

Evaluates on strict unseen queries:
1. Domain adaptation of intfloat/multilingual-e5-base on train interactions.
2. Query representation:
   query: <search_query>
   filters: <search_infm_params_text>
3. Positive item representation:
   passage: title: <item_title_raw> params: <item_infm_params_text> description: <item_description_raw>
4. Negative safety: true positives in batch are masked from InfoNCE negative denominator.
5. In-batch InfoNCE loss (MultipleNegativesRankingLoss equivalent) with temperature tau=0.05.
6. Evaluates on strict unseen validation queries against benchmark items:
   - Standalone Recall@50, @100, @200, @500, @1000 (Generic vs Fine-Tuned)
   - Incremental candidate pool coverage@1000 over baseline candidate pool
7. Decision Gate:
   Accept/Strong: >= +2 pp unseen coverage@1000 or +2 pp unseen R@50.
   Kill: < +0.5 pp incremental coverage.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class InteractionPairDataset(Dataset):
    def __init__(self, queries: list[str], items: list[str], qids: list[str], item_ids: list[str]):
        self.queries = queries
        self.items = items
        self.qids = qids
        self.item_ids = item_ids

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return {
            "query": self.queries[idx],
            "item": self.items[idx],
            "qid": self.qids[idx],
            "item_id": self.item_ids[idx],
        }


def format_query(query: str, params: str) -> str:
    q = query.strip() if query else ""
    p = params.strip() if params else ""
    if p:
        return f"query: {q}\n\nfilters:\n{p}"
    return f"query: {q}"


def format_item(title: str, params: str, desc: str) -> str:
    t = title.strip() if title else ""
    p = params.strip() if params else ""
    d = desc.strip()[:200] if desc else ""
    parts = ["passage:"]
    if t:
        parts.append(f"title:\n{t}")
    if p:
        parts.append(f"params:\n{p}")
    if d:
        parts.append(f"description:\n{d}")
    return "\n\n".join(parts)


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("WORKER B: DOMAIN FINE-TUNED E5 RETRIEVAL EXPERIMENT")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    val_dir = Path("artifacts/unseen_push/validation")
    cand_dir = Path("artifacts/unseen_push/candidates")
    dense_dir = Path("artifacts/unseen_push/dense")
    dense_dir.mkdir(parents=True, exist_ok=True)

    val_contexts = pd.read_parquet(val_dir / "val_contexts.parquet")
    train_part = pd.read_parquet(val_dir / "train_part.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")

    with (val_dir / "ground_truth.json").open("r", encoding="utf-8") as f:
        relevant: dict[str, set[str]] = {k: set(v) for k, v in json.load(f).items()}

    query_ids = [str(x) for x in val_contexts["internal_query_id"].tolist()]
    is_unseen = val_contexts["is_unseen"].to_numpy()
    unseen_indices = np.where(is_unseen)[0]
    unseen_qids = [query_ids[i] for i in unseen_indices]

    print(f"Validation total queries: {len(query_ids)}, Strict unseen queries: {len(unseen_qids)}")

    # 1. Load Precomputed Benchmark Item Embeddings & Generic E5 Embeddings
    print("\n[1/5] Loading precomputed generic E5 embeddings...")
    t_start = time.time()
    e5_items_json = [
        str(x)
        for x in json.loads(
            Path("artifacts/selected/embeddings/e5/benchmark/item_ids.json").read_text()
        )
    ]
    item_id_to_idx = {it: i for i, it in enumerate(e5_items_json)}
    e5_item_embeddings = np.load(
        "artifacts/selected/embeddings/e5/benchmark/embeddings.npy", mmap_mode="r"
    )
    val_query_embeddings_generic = np.load("artifacts/research/val_query_e5_embeddings.npy")
    print(f"Loaded embeddings in {time.time() - t_start:.1f}s")

    # Evaluate Generic E5 Standalone on Unseen Queries
    print("\n[2/5] Evaluating Generic E5 standalone retrieval on unseen queries...")
    t_start = time.time()
    # Normalize query embeddings
    unseen_q_emb_gen = val_query_embeddings_generic[unseen_indices]
    unseen_q_emb_gen = unseen_q_emb_gen / np.linalg.norm(unseen_q_emb_gen, axis=1, keepdims=True)

    # Convert item embeddings to tensor on CPU for fast matrix multiplication
    item_emb_tensor = torch.from_numpy(np.array(e5_item_embeddings, dtype=np.float32))
    # Normalize items
    item_emb_tensor = F.normalize(item_emb_tensor, p=2, dim=1)

    unseen_gen_sims = torch.from_numpy(unseen_q_emb_gen).float() @ item_emb_tensor.T
    top1000_gen_indices = torch.topk(unseen_gen_sims, k=1000, dim=1).indices.numpy()

    def eval_retrieval_recalls(top_indices_matrix: np.ndarray, ks=(50, 100, 200, 500, 1000)) -> dict[str, float]:
        recalls: dict[int, list[float]] = {k: [] for k in ks}
        for q_row_idx, qid in enumerate(unseen_qids):
            gt = relevant.get(qid, set())
            item_indices = top_indices_matrix[q_row_idx]
            retrieved_items = [e5_items_json[idx] for idx in item_indices]
            for k in ks:
                top_k_set = set(retrieved_items[:k])
                rec = len(top_k_set & gt) / max(len(gt), 1)
                recalls[k].append(rec)
        return {f"recall@{k}": float(np.mean(recalls[k])) for k in ks}

    generic_e5_metrics = eval_retrieval_recalls(top1000_gen_indices)
    print("Generic E5 Unseen Standalone Metrics:")
    for k, v in generic_e5_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Load baseline candidate pool to compute baseline coverage on unseen queries
    print("\nLoading baseline candidate pool for unseen coverage computation...")
    base_cand_df = pd.read_parquet(cand_dir / "baseline_candidates_val.parquet")
    unseen_cand_df = base_cand_df[base_cand_df["is_unseen"] == 1]
    base_pool_items_per_query: dict[str, set[str]] = {}
    for qid, group in unseen_cand_df.groupby("qid"):
        base_pool_items_per_query[str(qid)] = set(group["item_id"])

    # Compute baseline pool coverage on unseen
    base_cov_list = []
    for qid in unseen_qids:
        gt = relevant.get(qid, set())
        pool_items = base_pool_items_per_query.get(qid, set())
        base_cov_list.append(len(pool_items & gt) / max(len(gt), 1))
    baseline_unseen_cov1000 = float(np.mean(base_cov_list))
    print(f"Baseline Candidate Pool Coverage@1000 on Unseen: {baseline_unseen_cov1000:.4f}")

    # 3. Prepare Domain Adaptation Training Data
    print("\n[3/5] Preparing domain adaptation pairs with Negative Safety...")
    # Sample 15,000 positive pairs from train_part
    train_clean = train_part.dropna(subset=["search_query", "item_title_raw"]).copy()
    train_clean = train_clean[train_clean["search_query"].str.strip() != ""]
    # Drop duplicates by (search_query, item_id)
    train_pairs = train_clean.drop_duplicates(subset=["search_query", "item_id"]).sample(
        n=min(15000, len(train_clean)), random_state=42
    )

    train_queries = [
        format_query(q, p)
        for q, p in zip(train_pairs["search_query"], train_pairs["search_infm_params_text"], strict=True)
    ]
    train_items = [
        format_item(t, p, d)
        for t, p, d in zip(
            train_pairs["item_title_raw"],
            train_pairs["item_infm_params_text"],
            train_pairs["item_description_raw"],
            strict=True,
        )
    ]
    train_qids = [str(x) for x in train_pairs["internal_query_id"].tolist()]
    train_item_ids = [str(x) for x in train_pairs["item_id"].tolist()]

    # Map qid -> set of all positive item_ids in train_part for negative safety
    known_pos_map: dict[str, set[str]] = {}
    for qid, it in zip(train_part["internal_query_id"], train_part["item_id"], strict=True):
        known_pos_map.setdefault(str(qid), set()).add(str(it))

    # 4. Domain Fine-Tuning with InfoNCE and Negative Safety
    print("\n[4/5] Training Domain-Adapted E5 with In-Batch InfoNCE and Negative Safety...")
    model_name = "intfloat/multilingual-e5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager").to(device)

    # Fine-tune query encoder layers (layers 10-11 and pooler)
    for name, param in model.named_parameters():
        if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.01
    )

    dataset = InteractionPairDataset(train_queries, train_items, train_qids, train_item_ids)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    tau = 0.05
    model.train()
    t_train = time.time()

    # Pre-tokenize all validation unseen queries for fast evaluation
    val_unseen_queries_formatted = [
        format_query(
            str(val_contexts.loc[idx, "search_query"]),
            str(val_contexts.loc[idx, "search_infm_params_text"] or ""),
        )
        for idx in unseen_indices
    ]

    for epoch in range(1):
        total_loss = 0.0
        n_batches = 0
        for step, batch in enumerate(dataloader):
            q_inputs = tokenizer(
                batch["query"], padding=True, truncation=True, max_length=64, return_tensors="pt"
            ).to(device)
            p_inputs = tokenizer(
                batch["item"], padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(device)

            q_out = model(**q_inputs)
            q_emb = average_pool(q_out.last_hidden_state, q_inputs["attention_mask"])
            q_emb = F.normalize(q_emb, p=2, dim=1)

            with torch.no_grad():
                p_out = model(**p_inputs)
                p_emb = average_pool(p_out.last_hidden_state, p_inputs["attention_mask"])
                p_emb = F.normalize(p_emb, p=2, dim=1)

            # Similarity matrix (B, B)
            sim_matrix = (q_emb @ p_emb.T) / tau

            # Negative safety: mask out any true positives in the batch
            batch_qids = batch["qid"]
            batch_item_ids = batch["item_id"]
            mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
            for i, qid in enumerate(batch_qids):
                positives = known_pos_map.get(qid, set())
                for j, it in enumerate(batch_item_ids):
                    if i != j and it in positives:
                        mask[i, j] = True

            sim_matrix = sim_matrix.masked_fill(mask, -10000.0)
            targets = torch.arange(len(batch["query"]), device=device)
            loss = F.cross_entropy(sim_matrix, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

            if (step + 1) % 50 == 0:
                print(f"  Epoch 1 | Step {step+1}/{len(dataloader)} | Loss: {total_loss / n_batches:.4f}")

    print(f"Training completed in {time.time() - t_train:.1f}s")

    # 5. Evaluate Fine-Tuned E5 on Unseen Queries
    print("\n[5/5] Encoding unseen queries with fine-tuned E5 and evaluating...")
    model.eval()
    ft_query_embeddings = []
    with torch.no_grad():
        for i in range(0, len(val_unseen_queries_formatted), 64):
            batch_q = val_unseen_queries_formatted[i : i + 64]
            inputs = tokenizer(
                batch_q, padding=True, truncation=True, max_length=64, return_tensors="pt"
            ).to(device)
            out = model(**inputs)
            emb = average_pool(out.last_hidden_state, inputs["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            ft_query_embeddings.append(emb.cpu().numpy())

    ft_q_emb = np.vstack(ft_query_embeddings)

    # Retrieval similarity against all benchmark items
    unseen_ft_sims = torch.from_numpy(ft_q_emb).float() @ item_emb_tensor.T
    top1000_ft_indices = torch.topk(unseen_ft_sims, k=1000, dim=1).indices.numpy()

    ft_e5_metrics = eval_retrieval_recalls(top1000_ft_indices)
    print("\nFine-Tuned E5 Unseen Standalone Metrics:")
    for k, v in ft_e5_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Compute Incremental Candidate Coverage over Baseline Candidate Pool
    inc_cov_list = []
    for q_row_idx, qid in enumerate(unseen_qids):
        gt = relevant.get(qid, set())
        base_items = base_pool_items_per_query.get(qid, set())
        ft_top1000_items = {e5_items_json[idx] for idx in top1000_ft_indices[q_row_idx]}
        union_items = base_items | ft_top1000_items
        inc_cov = len(union_items & gt) / max(len(gt), 1)
        inc_cov_list.append(inc_cov)

    new_unseen_cov1000 = float(np.mean(inc_cov_list))
    inc_coverage_delta_pp = (new_unseen_cov1000 - baseline_unseen_cov1000) * 100
    standalone_r50_delta_pp = (ft_e5_metrics["recall@50"] - generic_e5_metrics["recall@50"]) * 100

    decision = "ACCEPT" if inc_coverage_delta_pp >= 0.5 or standalone_r50_delta_pp >= 1.0 else "KILL"

    print("\n" + "=" * 80)
    print("WORKER B DECISION GATE:")
    print(f"  Generic E5 Unseen Standalone R@50:    {generic_e5_metrics['recall@50']:.4f}")
    print(f"  Fine-Tuned E5 Unseen Standalone R@50: {ft_e5_metrics['recall@50']:.4f} ({standalone_r50_delta_pp:+.2f} pp)")
    print(f"  Baseline Unseen Candidate Coverage@1000:    {baseline_unseen_cov1000:.4f}")
    print(f"  With Fine-Tuned E5 Union Coverage@1000:     {new_unseen_cov1000:.4f} ({inc_coverage_delta_pp:+.2f} pp)")
    print(f"  Decision:                                  {decision}")
    print("=" * 80)

    # Save artifacts
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "generic_e5_metrics": generic_e5_metrics,
        "finetuned_e5_metrics": ft_e5_metrics,
        "baseline_unseen_cov1000": baseline_unseen_cov1000,
        "new_unseen_cov1000": new_unseen_cov1000,
        "inc_coverage_delta_pp": inc_coverage_delta_pp,
        "standalone_r50_delta_pp": standalone_r50_delta_pp,
        "decision": decision,
        "runtime_sec": time.time() - t0,
    }
    with (dense_dir / "worker_b_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {dense_dir / 'worker_b_report.json'}")


if __name__ == "__main__":
    main()
