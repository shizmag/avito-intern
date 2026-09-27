# Validation corpus audit

- Status: **PASS**
- Primary protocol: clean cold-item validation.
- Stress protocol: not run; no defensible known-negative distractor source is present.

## Provenance

- `validation_items` come from deterministic item-level SHA-256 split with seed 42 and 10% validation proportion. Their size is approximately 10% of canonical training items, not benchmark items.
- Validation queries are queries with at least one observed positive on a validation item.
- `train_items` are not appended as negatives: missing interaction is not a guaranteed negative.

## Counts

| table | rows | unique items/queries |
|---|---:|---:|
| `raw_train` | 497,673 | 344,825 |
| `train_items` | 275,899 | 275,899 |
| `validation_items` | 34,370 | 34,370 |
| `benchmark_items` | 189,212 | 189,212 |
| `validation_queries` | 43,413 | 43,413 |
| `ground_truth` | 46,360 | 34,370 |

## Leakage and corpus checks

- validation∩train items: **0**
- validation∩benchmark items: **1,795**
- validation items found in raw train: **34,370** (expected: split derives from observed interactions, not temporal unseen-item source)
- ground-truth items subset of validation corpus: **True**
- validation/benchmark corpus ratio: **0.1816** (benchmark is 5.51x larger)

## Decision

Primary clean Recall is valid for item-disjoint cold-item comparison because candidate items are restricted to validation split and ground-truth positives are known. It is not a benchmark-sized distractor test. Do not label every train item negative. Any future large-corpus run must be a separate stress Recall with explicit negative/availability semantics.

## Distribution summary

### validation_items

```json
{
  "search_category": {
    "missing": 0,
    "nunique": 2,
    "top5": {
      "114": 34368,
      "0": 2
    }
  },
  "search_location_id": {
    "missing": 0,
    "nunique": 1430,
    "top5": {
      "637640": 2126,
      "653240": 1541,
      "107620": 820,
      "633540": 749,
      "650400": 612
    }
  },
  "item_price": {
    "missing": 0,
    "quantiles": {
      "0.01": -1.0,
      "0.5": 1000.0,
      "0.99": 99651.89999999886
    }
  },
  "item_rating": {
    "missing": 2297,
    "quantiles": {
      "0.01": 0.0,
      "0.5": 5.0,
      "0.99": 5.0
    }
  },
  "item_title_raw_chars": {
    "quantiles": {
      "0.01": 7.0,
      "0.5": 33.0,
      "0.99": 60.30999999999767
    }
  },
  "item_description_raw_chars": {
    "quantiles": {
      "0.01": 24.0,
      "0.5": 869.0,
      "0.99": 5496.0
    }
  }
}
```

### benchmark_items

```json
{
  "item_price": {
    "missing": 0,
    "quantiles": {
      "0.01": -1.0,
      "0.5": 1000.0,
      "0.99": 150000.0
    }
  },
  "item_rating": {
    "missing": 17631,
    "quantiles": {
      "0.01": 0.0,
      "0.5": 5.0,
      "0.99": 5.0
    }
  },
  "item_title_raw_chars": {
    "quantiles": {
      "0.01": 7.0,
      "0.5": 35.0,
      "0.99": 53.0
    }
  },
  "item_description_raw_chars": {
    "quantiles": {
      "0.01": 27.0,
      "0.5": 1054.0,
      "0.99": 5541.0
    }
  }
}
```


## Quantitative distribution comparison

Categorical fields report total variation and Jensen-Shannon divergence; numeric/text fields report quantile deltas and KS statistic.

```json
{
  "item_microcat_id": {
    "validation_nunique": 201,
    "benchmark_nunique": 752,
    "validation_missing": 0,
    "benchmark_missing": 0,
    "total_variation": 0.20900458327664087,
    "js_divergence_bits": 0.053903418959817304,
    "top_validation": {
      "1289835": 1597,
      "86470": 1290,
      "86467": 938,
      "1289833": 877,
      "86469": 777,
      "2059415": 754,
      "44730": 621,
      "2097890": 563,
      "1178215": 555,
      "2301617": 547
    },
    "top_benchmark": {
      "86456": 6355,
      "2301563": 4272,
      "86467": 4064,
      "86470": 3858,
      "1289835": 3696,
      "4141": 3417,
      "2303435": 3267,
      "86458": 3112,
      "2097890": 2927,
      "86469": 2895
    }
  },
  "item_category_id": {
    "validation_nunique": 3,
    "benchmark_nunique": 47,
    "validation_missing": 0,
    "benchmark_missing": 0,
    "total_variation": 0.009827519134516165,
    "js_divergence_bits": 0.004849610676267208,
    "top_validation": {
      "114": 34367,
      "27": 2,
      "81": 1
    },
    "top_benchmark": {
      "114": 187336,
      "19": 357,
      "112": 141,
      "40": 122,
      "33": 103,
      "10": 98,
      "20": 84,
      "25": 82,
      "36": 74,
      "111": 74
    }
  },
  "item_location_id": {
    "validation_nunique": 1406,
    "benchmark_nunique": 2877,
    "validation_missing": 0,
    "benchmark_missing": 0,
    "total_variation": 0.12393819265225009,
    "js_divergence_bits": 0.03206770648642772,
    "top_validation": {
      "637640": 2598,
      "653240": 1846,
      "633540": 848,
      "650400": 645,
      "654070": 637,
      "652000": 589,
      "641780": 576,
      "640860": 547,
      "646600": 495,
      "661420": 491
    },
    "top_benchmark": {
      "637640": 19543,
      "653240": 12145,
      "633540": 4494,
      "650400": 3659,
      "641780": 3178,
      "654070": 3157,
      "640860": 3070,
      "652000": 3027,
      "661420": 2726,
      "625810": 2575
    }
  },
  "item_price": {
    "validation_count": 34370,
    "benchmark_count": 189212,
    "validation_missing": 0,
    "benchmark_missing": 0,
    "quantiles": {
      "q01_validation": -1.0,
      "q50_validation": 1000.0,
      "q99_validation": 99651.89999999886,
      "q01_benchmark": -1.0,
      "q50_benchmark": 1000.0,
      "q99_benchmark": 150000.0
    },
    "quantile_delta": {
      "q01": 0.0,
      "q50": 0.0,
      "q99": -50348.10000000114
    },
    "ks_statistic": 0.03653276531574268
  },
  "item_rating": {
    "validation_count": 32073,
    "benchmark_count": 171581,
    "validation_missing": 2297,
    "benchmark_missing": 17631,
    "quantiles": {
      "q01_validation": 0.0,
      "q50_validation": 5.0,
      "q99_validation": 5.0,
      "q01_benchmark": 0.0,
      "q50_benchmark": 5.0,
      "q99_benchmark": 5.0
    },
    "quantile_delta": {
      "q01": 0.0,
      "q50": 0.0,
      "q99": 0.0
    },
    "ks_statistic": 0.01414673214423745
  },
  "item_title_raw_chars": {
    "validation_count": 34370,
    "benchmark_count": 189212,
    "validation_missing": 0,
    "benchmark_missing": 0,
    "quantiles": {
      "q01_validation": 7.0,
      "q50_validation": 33.0,
      "q99_validation": 60.30999999999767,
      "q01_benchmark": 7.0,
      "q50_benchmark": 35.0,
      "q99_benchmark": 53.0
    },
    "quantile_delta": {
      "q01": 0.0,
      "q50": -2.0,
      "q99": 7.309999999997672
    },
    "ks_statistic": 0.03241792733566162
  },
  "item_description_raw_chars": {
    "validation_count": 34370,
    "benchmark_count": 189212,
    "validation_missing": 0,
    "benchmark_missing": 0,
    "quantiles": {
      "q01_validation": 24.0,
      "q50_validation": 869.0,
      "q99_validation": 5496.0,
      "q01_benchmark": 27.0,
      "q50_benchmark": 1054.0,
      "q99_benchmark": 5541.0
    },
    "quantile_delta": {
      "q01": -3.0,
      "q50": -185.0,
      "q99": -45.0
    },
    "ks_statistic": 0.06918448988297865
  }
}
```

## Coldness caveat

Validation items are cold relative to `train_items` (zero overlap), but not unseen in raw source: all validation items originate from observed raw interactions. Validation and benchmark item IDs also overlap (1,795 exact IDs). This is recorded as caveat, not hidden.
