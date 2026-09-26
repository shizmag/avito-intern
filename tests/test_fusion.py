import pandas as pd
from avito_candidate_generation.fusion.features import build_pair_features
from avito_candidate_generation.fusion.rrf import reciprocal_rank_fusion
from avito_candidate_generation.oof import build_oof_table
from avito_candidate_generation.training.hard_negatives import mine_hard_negatives

def candidates():
    return pd.DataFrame({"internal_query_id":["q","q","q"],"item_id":["a","b","c"],"source":["bm25","bm25","dense"],"score":[3.,2.,1.],"rank":[1,2,1]})

def test_rrf_union_features_and_oof():
    result=reciprocal_rank_fusion(candidates(),limit=2)
    assert result.iloc[0].item_id == "a"
    features=build_pair_features(candidates())
    assert int(features.iloc[0].retriever_count) == 1
    oof=build_oof_table(features,pd.DataFrame({"internal_query_id":["q"],"item_id":["a"]}),fold=0)
    assert set(oof.label)=={0,1}

def test_hard_negatives_remove_all_known_positives():
    result=mine_hard_negatives(candidates(),pd.DataFrame({"internal_query_id":["q"],"item_id":["a"]}),max_per_query=5)
    assert "a" not in set(result.item_id)
