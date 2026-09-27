# Category retrieval ablation

{
  "schema_version": 1,
  "status": "PASS",
  "split": "cold-item-validation",
  "methods": {
    "global": {
      "method": "global",
      "split": "cold-item-validation",
      "n_queries": 43413,
      "n_items": 34370,
      "metrics": {
        "recall@10": 0.1826241596506264,
        "recall@20": 0.26274057998195927,
        "recall@50": 0.39995341014966473,
        "recall@100": 0.5248241741417748,
        "recall@200": 0.6540667841770046,
        "recall@500": 0.7891208824575559
      },
      "mean_candidate_corpus_size": 34370.0,
      "p50_candidate_corpus_size": 34370.0,
      "p95_candidate_corpus_size": 34370.0,
      "empty_category_rate": 0.0,
      "fallback_rate": 0.0,
      "runtime_sec": 76.3209099159576
    },
    "hard": {
      "method": "hard",
      "split": "cold-item-validation",
      "n_queries": 43413,
      "n_items": 34370,
      "metrics": {
        "recall@10": 0.18267022880042022,
        "recall@20": 0.26278664913175315,
        "recall@50": 0.39999947929945856,
        "recall@100": 0.5248702432915687,
        "recall@200": 0.6541128533267985,
        "recall@500": 0.7891669516073496
      },
      "mean_candidate_corpus_size": 34366.41678759819,
      "p50_candidate_corpus_size": 34368.0,
      "p95_candidate_corpus_size": 34368.0,
      "empty_category_rate": 0.0,
      "fallback_rate": 0.0,
      "runtime_sec": 83.30576391599607
    },
    "fallback": {
      "method": "fallback",
      "split": "cold-item-validation",
      "n_queries": 43413,
      "n_items": 34370,
      "metrics": {
        "recall@10": 0.18267022880042022,
        "recall@20": 0.26278664913175315,
        "recall@50": 0.39999947929945856,
        "recall@100": 0.5248702432915687,
        "recall@200": 0.6541128533267985,
        "recall@500": 0.7891669516073496
      },
      "mean_candidate_corpus_size": 34366.41678759819,
      "p50_candidate_corpus_size": 34368.0,
      "p95_candidate_corpus_size": 34368.0,
      "empty_category_rate": 0.0,
      "fallback_rate": 0.0,
      "runtime_sec": 87.21443237492349
    }
  },
  "category_fields": {
    "query": "search_category",
    "item": "search_category"
  },
  "location_policy": "not filtered",
  "config_hash": "512ad065faff9cfc85fb6d2e794c6bf66bbd75d02a730e999afd945ce60ffc68"
}



## Policy decision

Selected policy: **hard**. On clean cold-item validation, hard and fallback both improved Recall@50 from 0.399953 to 0.399999 (+0.000046), with zero empty-category queries and mean category survivor corpus 34,366 versus global 34,370. Fallback had no observed use and no quality advantage. Hard is selected; location remains unfiltered.
