import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import re
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.max_colwidth", 120)
    DATA_DIR = Path("data")
    TRAIN_PATH = DATA_DIR / "train.parquet"
    QUERIES_PATH = DATA_DIR / "benchmark_queries.parquet"
    ITEMS_PATH = DATA_DIR / "benchmark_items.parquet"
    return ITEMS_PATH, QUERIES_PATH, TRAIN_PATH, mo, np, pd, plt, re, sns


@app.cell
def _(ITEMS_PATH, QUERIES_PATH, TRAIN_PATH, np, pd, plt, re, sns):
    def build_eda(train_path, queries_path, items_path):
        train = pd.read_parquet(train_path)
        queries = pd.read_parquet(queries_path)
        items = pd.read_parquet(items_path)
        row_counts = {"train": len(train), "benchmark_queries": len(queries), "benchmark_items": len(items)}

        dataset_overview = pd.DataFrame(
            {
                "dataset": list(row_counts),
                "rows": list(row_counts.values()),
                "columns": [train.shape[1], queries.shape[1], items.shape[1]],
                "memory_mb": [
                    train.memory_usage(deep=True).sum() / 1024**2,
                    queries.memory_usage(deep=True).sum() / 1024**2,
                    items.memory_usage(deep=True).sum() / 1024**2,
                ],
            }
        )
        schema = pd.concat(
            [
                pd.DataFrame({"dataset": "train", "column": train.columns, "dtype": train.dtypes.astype(str).values}),
                pd.DataFrame({"dataset": "benchmark_queries", "column": queries.columns, "dtype": queries.dtypes.astype(str).values}),
                pd.DataFrame({"dataset": "benchmark_items", "column": items.columns, "dtype": items.dtypes.astype(str).values}),
            ],
            ignore_index=True,
        )
        null_profile = pd.concat(
            [
                train.isna().sum().rename("nulls").to_frame().assign(dataset="train"),
                queries.isna().sum().rename("nulls").to_frame().assign(dataset="benchmark_queries"),
                items.isna().sum().rename("nulls").to_frame().assign(dataset="benchmark_items"),
            ]
        ).reset_index(names="column")
        null_profile["null_pct"] = null_profile.apply(lambda r: 100 * r["nulls"] / row_counts[r["dataset"]], axis=1)

        id_quality = pd.DataFrame(
            {
                "check": [
                    "train duplicate rows",
                    "train duplicate query-item pairs",
                    "train unique query texts",
                    "train unique item ids",
                    "benchmark duplicate query ids",
                    "benchmark duplicate item ids",
                    "benchmark query texts",
                ],
                "value": [
                    int(train.duplicated().sum()),
                    int(train.duplicated(["search_query", "item_id"]).sum()),
                    int(train["search_query"].nunique()),
                    int(train["item_id"].nunique()),
                    int(queries["query_id"].duplicated().sum()),
                    int(items["item_id"].duplicated().sum()),
                    int(queries["search_query"].nunique()),
                ],
            }
        )

        benchmark_item_ids = set(items["item_id"].astype(str))
        train_query_set = set(train["search_query"].astype(str))
        exact_query_mask = queries["search_query"].astype(str).isin(train_query_set)
        context_cols = ["search_query", "search_location_id", "search_is_delivery_search", "search_infm_params_text", "search_category"]
        train_context_keys = set(train[context_cols].astype(str).agg("\x1f".join, axis=1))
        benchmark_context_keys = queries[context_cols].astype(str).agg("\x1f".join, axis=1)
        exact_context_mask = benchmark_context_keys.isin(train_context_keys)
        train_item_in_benchmark = train["item_id"].astype(str).isin(benchmark_item_ids)
        coverage_summary = pd.DataFrame(
            {
                "metric": [
                    "benchmark queries with exact train query text",
                    "benchmark queries with exact full search context",
                    "train rows whose item is in benchmark corpus",
                    "unique train items also in benchmark corpus",
                ],
                "value": [
                    int(exact_query_mask.sum()),
                    int(exact_context_mask.sum()),
                    int(train_item_in_benchmark.sum()),
                    len(set(train.loc[train_item_in_benchmark, "item_id"].astype(str))),
                ],
                "share": [
                    exact_query_mask.mean(),
                    exact_context_mask.mean(),
                    train_item_in_benchmark.mean(),
                    len(set(train.loc[train_item_in_benchmark, "item_id"].astype(str))) / train["item_id"].nunique(),
                ],
            }
        )

        train_with_benchmark_flag = train.assign(_in_benchmark=train["item_id"].astype(str).isin(benchmark_item_ids))
        query_item_agg = train_with_benchmark_flag.groupby("search_query", sort=False).agg(
            train_rows=("item_id", "size"),
            train_unique_items=("item_id", "nunique"),
            benchmark_rows=("_in_benchmark", "sum"),
        )
        benchmark_unique_by_query = (
            train_with_benchmark_flag.loc[train_with_benchmark_flag["_in_benchmark"]]
            .groupby("search_query")["item_id"]
            .nunique()
        )
        query_item_agg["benchmark_unique_items"] = benchmark_unique_by_query.reindex(query_item_agg.index, fill_value=0)
        benchmark_query_stats = queries[["query_id", "search_query"]].merge(
            query_item_agg, left_on="search_query", right_index=True, how="left"
        ).fillna(0)
        benchmark_query_stats["exact_query_in_train"] = benchmark_query_stats["search_query"].isin(train_query_set)
        exact_quantiles = benchmark_query_stats.loc[exact_query_mask, "benchmark_unique_items"].quantile([0, .25, .5, .75, .9, .95, .99, 1])
        relevant_count_summary = pd.DataFrame(
            {
                "stat": ["min", "p25", "median", "p75", "p90", "p95", "p99", "max", "zero relevant train items"],
                "value": [*exact_quantiles.tolist(), int((benchmark_query_stats.loc[exact_query_mask, "benchmark_unique_items"] == 0).sum())],
            }
        )

        train_query_frequency = train.groupby("search_query").size()
        train_items_per_query = train.groupby("search_query")["item_id"].nunique()
        query_volume_summary = pd.DataFrame(
            {
                "measure": ["rows/query", "unique items/query"],
                "min": [train_query_frequency.min(), train_items_per_query.min()],
                "p25": [train_query_frequency.quantile(.25), train_items_per_query.quantile(.25)],
                "median": [train_query_frequency.median(), train_items_per_query.median()],
                "p75": [train_query_frequency.quantile(.75), train_items_per_query.quantile(.75)],
                "p90": [train_query_frequency.quantile(.90), train_items_per_query.quantile(.90)],
                "p95": [train_query_frequency.quantile(.95), train_items_per_query.quantile(.95)],
                "p99": [train_query_frequency.quantile(.99), train_items_per_query.quantile(.99)],
                "max": [train_query_frequency.max(), train_items_per_query.max()],
            }
        )

        categorical_summary = pd.DataFrame(
            {
                "column": ["search_category", "item_category_id", "item_microcat_id", "search_location_id", "item_location_id"],
                "train_unique": [train[c].nunique() for c in ["search_category", "item_category_id", "item_microcat_id", "search_location_id", "item_location_id"]],
                "benchmark_unique": [
                    queries["search_category"].nunique(),
                    items["item_category_id"].nunique(),
                    items["item_microcat_id"].nunique(),
                    queries["search_location_id"].nunique(),
                    items["item_location_id"].nunique(),
                ],
            }
        )
        contextual_signals = pd.DataFrame(
            {
                "signal_on_selected_train_rows": [
                    "search_category == item_category_id",
                    "search_location_id == item_location_id",
                    "item_is_phone_hidden",
                    "item_is_message_forbidden",
                ],
                "share": [
                    (train["search_category"] == train["item_category_id"]).mean(),
                    (train["search_location_id"] == train["item_location_id"]).mean(),
                    train["item_is_phone_hidden"].mean(),
                    train["item_is_message_forbidden"].mean(),
                ],
            }
        )

        numeric_cols = ["item_price", "item_rating", "item_rating_reviews_count", "item_latitude", "item_longitude"]
        numeric_summary = pd.concat(
            [
                train[numeric_cols].apply(pd.to_numeric, errors="coerce").describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T.assign(dataset="train"),
                items[numeric_cols].apply(pd.to_numeric, errors="coerce").describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T.assign(dataset="benchmark_items"),
            ]
        ).reset_index(names="column")

        def tokens(value):
            return re.findall(r"[0-9a-zа-яё]+", str(value).lower().replace("ё", "е"))

        query_text_summary = pd.DataFrame(
            {
                "dataset": ["train", "benchmark_queries"],
                "unique_queries": [train["search_query"].nunique(), queries["search_query"].nunique()],
                "median_tokens": [train["search_query"].map(tokens).str.len().median(), queries["search_query"].map(tokens).str.len().median()],
                "p90_tokens": [train["search_query"].map(tokens).str.len().quantile(.9), queries["search_query"].map(tokens).str.len().quantile(.9)],
                "median_chars": [train["search_query"].str.len().median(), queries["search_query"].str.len().median()],
                "p90_chars": [train["search_query"].str.len().quantile(.9), queries["search_query"].str.len().quantile(.9)],
            }
        )
        item_text_summary = pd.DataFrame(
            {
                "field": ["item_title_raw", "item_infm_params_text", "item_description_raw"],
                "benchmark_nonempty_share": [
                    items["item_title_raw"].fillna("").str.strip().ne("").mean(),
                    items["item_infm_params_text"].fillna("").str.strip().ne("").mean(),
                    items["item_description_raw"].fillna("").str.strip().ne("").mean(),
                ],
                "benchmark_unique_share": [
                    items["item_title_raw"].nunique() / len(items),
                    items["item_infm_params_text"].nunique() / len(items),
                    items["item_description_raw"].nunique() / len(items),
                ],
                "median_chars": [
                    items["item_title_raw"].fillna("").str.len().median(),
                    items["item_infm_params_text"].fillna("").str.len().median(),
                    items["item_description_raw"].fillna("").str.len().median(),
                ],
            }
        )

        exact_pair_counts = (
            train.loc[
                train["item_id"].astype(str).isin(benchmark_item_ids)
                & train["search_query"].isin(set(queries.loc[exact_query_mask, "search_query"]))
            ]
            .groupby(["search_query", "item_id"], sort=False)
            .size()
            .rename("frequency")
            .reset_index()
            .sort_values(["search_query", "frequency", "item_id"], ascending=[True, False, True])
        )
        exact_train_items = exact_pair_counts.groupby("search_query")["item_id"].agg(list)
        direct_candidate_summary = pd.DataFrame(
            {
                "query_id": queries["query_id"],
                "search_query": queries["search_query"],
                "exact_train_items_in_benchmark": queries["search_query"].map(exact_train_items).map(lambda x: len(x) if isinstance(x, list) else 0),
            }
        )
        direct_candidate_summary["direct_replay_proxy_recall_at_50"] = np.minimum(
            50 / direct_candidate_summary["exact_train_items_in_benchmark"].replace(0, np.nan), 1
        ).fillna(0)
        direct_candidate_summary["can_fill_50_from_exact_history"] = direct_candidate_summary["exact_train_items_in_benchmark"] >= 50
        direct_replay_summary = direct_candidate_summary[["direct_replay_proxy_recall_at_50", "can_fill_50_from_exact_history"]].agg(["mean", "sum"])

        positive_pool = train[train["item_id"].astype(str).isin(benchmark_item_ids)]
        positive_sample = positive_pool.sample(n=min(50_000, len(positive_pool)), random_state=42)

        def token_set(value):
            return set(tokens(value))

        query_token_sets = positive_sample["search_query"].map(token_set)
        title_token_sets = positive_sample["item_title_raw"].map(token_set)
        params_token_sets = positive_sample["item_infm_params_text"].map(token_set)
        all_text = positive_sample["item_title_raw"].fillna("") + " " + positive_sample["item_infm_params_text"].fillna(" ") + " " + positive_sample["item_description_raw"].fillna(" ")
        all_token_sets = all_text.map(token_set)
        token_recall = pd.DataFrame(
            {
                "field": ["title", "title + params", "title + params + description"],
                "mean_query_token_recall": [
                    np.mean([len(q & t) / len(q) if q else 0 for q, t in zip(query_token_sets, title_token_sets, strict=True)]),
                    np.mean([len(q & (t | p)) / len(q) if q else 0 for q, t, p in zip(query_token_sets, title_token_sets, params_token_sets, strict=True)]),
                    np.mean([len(q & a) / len(q) if q else 0 for q, a in zip(query_token_sets, all_token_sets, strict=True)]),
                ],
                "all_query_tokens_present": [
                    np.mean([q <= t for q, t in zip(query_token_sets, title_token_sets, strict=True)]),
                    np.mean([q <= (t | p) for q, t, p in zip(query_token_sets, title_token_sets, params_token_sets, strict=True)]),
                    np.mean([q <= a for q, a in zip(query_token_sets, all_token_sets, strict=True)]),
                ],
            }
        )
        def normalize_text(value):
            return re.sub(r"\s+", " ", str(value).lower().replace("ё", "е")).strip()

        positive_signal_summary = pd.DataFrame(
            {
                "signal": [
                    "query phrase occurs in all item text",
                    "search category equals item category",
                    "search location equals item location",
                ],
                "share_on_selected_sample": [
                    np.mean([normalize_text(q) in normalize_text(text) for q, text in zip(positive_sample["search_query"], all_text, strict=True)]),
                    (positive_sample["search_category"] == positive_sample["item_category_id"]).mean(),
                    (positive_sample["search_location_id"] == positive_sample["item_location_id"]).mean(),
                ],
            }
        )

        item_popularity = train[train["item_id"].astype(str).isin(benchmark_item_ids)]["item_id"].value_counts()
        popularity_summary = pd.DataFrame(
            {
                "stat": ["benchmark items", "items ever selected in train", "selected item share", "top-50 selected rows share"],
                "value": [
                    len(items),
                    len(item_popularity),
                    len(item_popularity) / len(items),
                    item_popularity.head(50).sum() / item_popularity.sum(),
                ],
            }
        )
        benchmark_query_distribution = pd.DataFrame(
            {
                "feature": ["search_location_id", "search_infm_params_text", "search_category", "search_is_delivery_search"],
                "unique_values": [queries[c].nunique() for c in ["search_location_id", "search_infm_params_text", "search_category", "search_is_delivery_search"]],
                "most_frequent_value_share": [queries[c].value_counts(normalize=True).iloc[0] for c in ["search_location_id", "search_infm_params_text", "search_category", "search_is_delivery_search"]],
            }
        )
        feature_distributions = {
            "train_search_category": train["search_category"].value_counts().sort_index(),
            "benchmark_search_category": queries["search_category"].value_counts().sort_index(),
            "train_delivery": train["search_is_delivery_search"].value_counts().sort_index(),
            "benchmark_delivery": queries["search_is_delivery_search"].value_counts().sort_index(),
            "train_item_category": items["item_category_id"].value_counts().head(20),
            "benchmark_item_category": items["item_category_id"].value_counts().head(20),
        }
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.histplot(benchmark_query_stats.loc[exact_query_mask, "benchmark_unique_items"], bins=30, ax=axes[0])
        axes[0].set(title="Train-derived benchmark item counts", xlabel="unique train items in benchmark corpus")
        sns.histplot(benchmark_query_stats.loc[exact_query_mask, "train_rows"], bins=30, ax=axes[1])
        axes[1].set(title="Training observations per exact query", xlabel="train rows")
        plt.tight_layout()

        return {
            "dataset_overview": dataset_overview,
            "schema": schema,
            "null_profile": null_profile,
            "id_quality": id_quality,
            "coverage_summary": coverage_summary,
            "benchmark_query_stats": benchmark_query_stats,
            "relevant_count_summary": relevant_count_summary,
            "query_volume_summary": query_volume_summary,
            "categorical_summary": categorical_summary,
            "contextual_signals": contextual_signals,
            "numeric_summary": numeric_summary,
            "query_text_summary": query_text_summary,
            "item_text_summary": item_text_summary,
            "exact_pair_counts": exact_pair_counts,
            "direct_candidate_summary": direct_candidate_summary,
            "direct_replay_summary": direct_replay_summary,
            "token_recall": token_recall,
            "positive_signal_summary": positive_signal_summary,
            "popularity_summary": popularity_summary,
            "benchmark_query_distribution": benchmark_query_distribution,
            "feature_distributions": feature_distributions,
            "fig": fig,
            "query_values": queries["search_query"].tolist(),
        }

    eda = build_eda(TRAIN_PATH, QUERIES_PATH, ITEMS_PATH)
    return (eda,)


@app.cell
def _(eda):
    dataset_overview = eda["dataset_overview"]
    schema = eda["schema"]
    null_profile = eda["null_profile"]
    id_quality = eda["id_quality"]
    coverage_summary = eda["coverage_summary"]
    benchmark_query_stats = eda["benchmark_query_stats"]
    relevant_count_summary = eda["relevant_count_summary"]
    query_volume_summary = eda["query_volume_summary"]
    categorical_summary = eda["categorical_summary"]
    contextual_signals = eda["contextual_signals"]
    numeric_summary = eda["numeric_summary"]
    query_text_summary = eda["query_text_summary"]
    item_text_summary = eda["item_text_summary"]
    exact_pair_counts = eda["exact_pair_counts"]
    direct_candidate_summary = eda["direct_candidate_summary"]
    direct_replay_summary = eda["direct_replay_summary"]
    token_recall = eda["token_recall"]
    positive_signal_summary = eda["positive_signal_summary"]
    popularity_summary = eda["popularity_summary"]
    benchmark_query_distribution = eda["benchmark_query_distribution"]
    feature_distributions = eda["feature_distributions"]
    fig = eda["fig"]
    query_values = eda["query_values"]
    return (
        benchmark_query_distribution,
        benchmark_query_stats,
        categorical_summary,
        contextual_signals,
        coverage_summary,
        dataset_overview,
        direct_replay_summary,
        exact_pair_counts,
        feature_distributions,
        fig,
        id_quality,
        item_text_summary,
        null_profile,
        numeric_summary,
        popularity_summary,
        positive_signal_summary,
        query_text_summary,
        query_values,
        query_volume_summary,
        relevant_count_summary,
        schema,
        token_recall,
    )


@app.cell
def _(mo):
    mo.md("""
    # Avito candidate-generation EDA

    Purpose: understand positive-label structure, corpus coverage, query reuse, lexical signals, and fallback layers for a high-recall top-50 candidate generator.

    Run locally with `uv run marimo edit eda.py`.
    """)
    return


@app.cell
def _(dataset_overview, mo):
    mo.ui.table(dataset_overview)
    return


@app.cell
def _(mo, null_profile, schema):
    mo.vstack([mo.ui.table(schema), mo.ui.table(null_profile)])
    return


@app.cell
def _(coverage_summary, id_quality, mo):
    mo.vstack([mo.ui.table(id_quality), mo.ui.table(coverage_summary)])
    return


@app.cell
def _(direct_replay_summary, mo, query_volume_summary, relevant_count_summary):
    mo.vstack([mo.ui.table(query_volume_summary), mo.ui.table(relevant_count_summary), mo.ui.table(direct_replay_summary)])
    return


@app.cell
def _(
    benchmark_query_distribution,
    categorical_summary,
    contextual_signals,
    mo,
):
    mo.vstack([mo.ui.table(categorical_summary), mo.ui.table(contextual_signals), mo.ui.table(benchmark_query_distribution)])
    return


@app.cell
def _(item_text_summary, mo, numeric_summary, query_text_summary):
    mo.vstack([mo.ui.table(numeric_summary), mo.ui.table(query_text_summary), mo.ui.table(item_text_summary)])
    return


@app.cell
def _(fig, mo):
    mo.as_html(fig)
    return


@app.cell
def _(
    feature_distributions,
    mo,
    popularity_summary,
    positive_signal_summary,
    token_recall,
):
    tables = [mo.ui.table(value.reset_index() if hasattr(value, "reset_index") else value) for value in feature_distributions.values()]
    mo.vstack(tables + [mo.ui.table(popularity_summary), mo.ui.table(positive_signal_summary), mo.ui.table(token_recall)])
    return


@app.cell
def _(mo, query_values):
    query_picker = mo.ui.dropdown(options=query_values, value=query_values[0], label="Inspect benchmark query")
    return (query_picker,)


@app.cell
def _(benchmark_query_stats, exact_pair_counts, mo, query_picker):
    selected_query = query_picker.value
    selected_stats = benchmark_query_stats[benchmark_query_stats["search_query"] == selected_query]
    selected_history = exact_pair_counts[exact_pair_counts["search_query"] == selected_query].head(50)
    selected_query_report = {
        "query": selected_query,
        "benchmark_stats": selected_stats,
        "exact_history_top50": selected_history,
    }
    mo.vstack([mo.md(f"### {selected_query}"), mo.ui.table(selected_stats), mo.ui.table(selected_history)])
    return


@app.cell
def _(mo):
    mo.md("""
    ## What to build

    1. **Exact history first.** Normalize query text, replay historical item IDs, rank by query-item frequency, deduplicate, and cap at 50. Add exact full-context replay as a separate source.
    2. **Corpus guard.** Intersect every historical result with `benchmark_items`; train contains selected items outside the submission corpus. Preserve IDs as strings.
    3. **Unseen-query retrieval.** Use word and character n-gram TF-IDF/BM25 over title, parameters, and description. Positive diagnostics show title+params+description has strongest token coverage.
    4. **Soft context boosts.** Boost matching category, microcategory, location, coordinates, and filter text; avoid hard gates because Recall@50 is primary.
    5. **Backfill.** Fill remaining slots from same microcategory/category and location, then global popularity. Deduplicate at every stage.

    Exact-history values here are diagnostic proxies: local files contain no benchmark relevance labels.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Submission checklist

    - `answer.csv` has exactly `query_id,answer`.
    - One row per benchmark query; no duplicate query IDs.
    - At most 50 unique item IDs per answer, separated by one space.
    - IDs remain lowercase strings exactly as stored in `benchmark_items.parquet`.
    - Validate every output ID against the benchmark corpus before upload.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
