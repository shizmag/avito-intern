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

