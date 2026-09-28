"""Train and save domain fine-tuned E5 query encoder and precompute embeddings.

Saves:
- artifacts/unseen_push/dense/finetuned_e5_query_encoder.pt
- artifacts/unseen_push/dense/val_query_ft_e5_embeddings.npy (all 5,347 validation queries)
- artifacts/unseen_push/dense/benchmark_query_ft_e5_embeddings.npy (all 2,452 benchmark queries)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
    print("TRAINING DOMAIN FINE-TUNED E5 AND SAVING QUERY EMBEDDINGS")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    val_dir = Path("artifacts/unseen_push/validation")
    dense_dir = Path("artifacts/unseen_push/dense")
    dense_dir.mkdir(parents=True, exist_ok=True)

    val_contexts = pd.read_parquet(val_dir / "val_contexts.parquet")
    train_part = pd.read_parquet(val_dir / "train_part.parquet")
    benchmark_queries = pd.read_parquet("data/benchmark_queries.parquet")

    # 1. Sample 15,000 positive training pairs
    train_clean = train_part.dropna(subset=["search_query", "item_title_raw"]).copy()
    train_clean = train_clean[train_clean["search_query"].str.strip() != ""]
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

    known_pos_map: dict[str, set[str]] = {}
    for qid, it in zip(train_part["internal_query_id"], train_part["item_id"], strict=True):
        known_pos_map.setdefault(str(qid), set()).add(str(it))

    # 2. Setup Model & Tokenizer
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
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    tau = 0.05

    model.train()
    print("Training 1 epoch with In-Batch InfoNCE and Negative Safety...")
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

        sim_matrix = (q_emb @ p_emb.T) / tau

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

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

    print("Training finished. Saving model weights...")
    torch.save(model.state_dict(), dense_dir / "finetuned_e5_query_encoder.pt")

    # 3. Precompute validation query embeddings
    print("Precomputing all 5,347 validation query embeddings...")
    model.eval()
    val_queries_fmt = [
        format_query(
            str(q),
            str(p) if pd.notna(p) else "",
        )
        for q, p in zip(val_contexts["search_query"], val_contexts["search_infm_params_text"], strict=True)
    ]

    val_embeddings = []
    with torch.no_grad():
        for i in range(0, len(val_queries_fmt), 64):
            batch = val_queries_fmt[i : i + 64]
            inp = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            out = model(**inp)
            emb = average_pool(out.last_hidden_state, inp["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            val_embeddings.append(emb.cpu().numpy())

    val_emb_arr = np.vstack(val_embeddings)
    np.save(dense_dir / "val_query_ft_e5_embeddings.npy", val_emb_arr)
    print(f"Saved validation embeddings shape: {val_emb_arr.shape}")

    # 4. Precompute benchmark query embeddings
    print("Precomputing all 2,452 benchmark query embeddings...")
    bm_queries_fmt = [
        format_query(
            str(q),
            str(p) if pd.notna(p) else "",
        )
        for q, p in zip(benchmark_queries["search_query"], benchmark_queries["search_infm_params_text"], strict=True)
    ]

    bm_embeddings = []
    with torch.no_grad():
        for i in range(0, len(bm_queries_fmt), 64):
            batch = bm_queries_fmt[i : i + 64]
            inp = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            out = model(**inp)
            emb = average_pool(out.last_hidden_state, inp["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            bm_embeddings.append(emb.cpu().numpy())

    bm_emb_arr = np.vstack(bm_embeddings)
    np.save(dense_dir / "benchmark_query_ft_e5_embeddings.npy", bm_emb_arr)
    print(f"Saved benchmark embeddings shape: {bm_emb_arr.shape}")

    print(f"All dense artifacts successfully saved in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
