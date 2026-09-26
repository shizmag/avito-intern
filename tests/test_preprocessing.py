from avito_candidate_generation.preprocessing import (
    normalize_text,
    represent_text,
    tokenize,
)


def test_normalization_and_missing():
    assert normalize_text("  Ёж\tв NFKC  ")[0] == "еж в nfkc"
    result = represent_text(None)
    assert result.was_missing and result.normalized == "" and result.tokens == ()


def test_tokens_preserve_digits_and_mixed_tokens():
    assert tokenize("iphone 12 pro 2.5", min_length=1) == (
        "iphone",
        "12",
        "pro",
        "2",
        "5",
    )
