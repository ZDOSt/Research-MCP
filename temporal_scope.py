import re


_AS_OF_RE = re.compile(r"\bas\s+of\b", re.I)
_RELATIVE_STATE_RE = re.compile(
    r"\b(?:today(?:'s)?|yesterday|tomorrow|current(?:ly)?)\b",
    re.I,
)
_COMPARISON_CUE_RE = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?|difference|differ(?:ence|s|ed)?|"
    r"changed?\s+(?:from|since)|then\s+and\s+now)\b",
    re.I,
)
_BETWEEN_SCOPE_CONNECTOR_RE = re.compile(
    r"\b(?:and|with|against|versus|vs\.?|to)\b",
    re.I,
)


def is_mixed_as_of_relative_comparison(value: str) -> bool:
    """Return whether one request explicitly compares an as-of state to now."""
    query = str(value or "")
    as_of = _AS_OF_RE.search(query)
    relative_states = list(_RELATIVE_STATE_RE.finditer(query))
    if as_of is None or not relative_states:
        return False
    if _COMPARISON_CUE_RE.search(query):
        return True

    for relative in relative_states:
        if relative.end() <= as_of.start():
            between = query[relative.end() : as_of.start()]
        elif as_of.end() <= relative.start():
            between = query[as_of.end() : relative.start()]
        else:
            continue
        if len(between) <= 120 and _BETWEEN_SCOPE_CONNECTOR_RE.search(between):
            return True
    return False
