import pytest

from query_hints import normalize_proposed_queries


def test_normalize_proposed_queries_canonicalizes_and_deduplicates():
    assert normalize_proposed_queries(
        [
            "  Docker\tinstallation   Ubuntu  ",
            "Docker installation Ubuntu?",
            "ＡＩ model releases",
        ]
    ) == [
        "Docker installation Ubuntu",
        "AI model releases",
    ]


@pytest.mark.parametrize(
    "value",
    [
        "not-a-list",
        [],
        ["one", "two", "three", "four", "five", "six"],
        [123],
        ["   "],
        ["x" * 181],
        ["Docker\x00docs"],
        ["Docker \ud800 docs"],
        ["Docker\u200bdocs"],
        ["Docker\ufeffdocs"],
        ["Docker \u202edocs"],
        ["Docker \u2066docs\u2069"],
    ],
)
def test_normalize_proposed_queries_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        normalize_proposed_queries(value)


def test_normalize_proposed_queries_accepts_omission():
    assert normalize_proposed_queries(None) is None
