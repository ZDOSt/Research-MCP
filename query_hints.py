import re
import unicodedata
from typing import Optional


MAX_PROPOSED_SEARCH_QUERIES = 5
PROPOSED_SEARCH_QUERY_MAX_CHARS = 180
_BIDI_CONTROL_CHARACTERS = frozenset(
    {
        "\u061c",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
)


def proposed_query_dedupe_key(value: str) -> str:
    normalized = re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", value),
    ).strip()
    return normalized.rstrip(" .,!?:;").casefold()


def normalize_proposed_queries(value: object) -> Optional[list[str]]:
    """Validate and canonicalize optional calling-model search suggestions."""

    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("proposed_queries must be a list or null")
    if not 1 <= len(value) <= MAX_PROPOSED_SEARCH_QUERIES:
        raise ValueError(
            f"proposed_queries must contain 1-{MAX_PROPOSED_SEARCH_QUERIES} items"
        )

    output: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"proposed_queries[{index}] must be a string")
        normalized = re.sub(
            r"\s+",
            " ",
            unicodedata.normalize("NFKC", item),
        ).strip()
        if not normalized:
            raise ValueError(f"proposed_queries[{index}] must not be blank")
        if len(normalized) > PROPOSED_SEARCH_QUERY_MAX_CHARS:
            raise ValueError(
                f"proposed_queries[{index}] exceeds "
                f"{PROPOSED_SEARCH_QUERY_MAX_CHARS} characters"
            )
        if any(
            unicodedata.category(character) in {"Cc", "Cf", "Cs"}
            and not character.isspace()
            for character in normalized
        ) or any(character in _BIDI_CONTROL_CHARACTERS for character in normalized):
            raise ValueError(
                f"proposed_queries[{index}] contains a control, format, surrogate, or bidi character"
            )
        try:
            normalized.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError(
                f"proposed_queries[{index}] must be valid UTF-8 text"
            ) from exc
        key = proposed_query_dedupe_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output
