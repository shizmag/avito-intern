import pandas as pd
import pytest

from avito_candidate_generation.data import stable_internal_query_id
from avito_candidate_generation.schemas import (
    SchemaError,
    validate_string_ids,
    validate_unique,
)


def test_ids_and_duplicates() -> None:
    validate_string_ids(["0" * 16], "id")
    with pytest.raises(SchemaError):
        validate_string_ids([1], "id")
    with pytest.raises(SchemaError):
        validate_unique(pd.DataFrame({"a": ["x", "x"]}), ["a"])


def test_internal_query_id_is_deterministic_and_sensitive() -> None:
    value = {"q": "hello", "loc": 1}
    assert stable_internal_query_id(value, ["q", "loc"]) == stable_internal_query_id(
        value, ["q", "loc"]
    )
    assert stable_internal_query_id(value, ["q", "loc"]) != stable_internal_query_id(
        {"q": "bye", "loc": 1}, ["q", "loc"]
    )
