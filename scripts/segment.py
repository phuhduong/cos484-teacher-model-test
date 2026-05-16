"""Segment reasoning traces into numbered units and reconstruct annotated traces.

A "unit" is a (start, end) tuple of char offsets into the original trace.
"""

import re

_PARA_RE = re.compile(r"\n\n+")
_SENT_RE = re.compile(r"(?<=[A-Za-z])[.!?]\s+(?=[A-Z])")
_BULLET_RE = re.compile(r"\n[ \t]*(?:[-*]|\d+[.)])\s")


def segment(trace):
    """Split trace into reasoning units. Each unit is (start, end) offsets.

    Boundaries:
      - Paragraph breaks (\\n\\n+)
      - Sentence breaks ([.!?] followed by whitespace and an uppercase letter,
        with the period preceded by a letter — so enumerators like "1." don't trigger)
      - Line-leading bullets/enumerators (-, *, 1., 1))
    """
    n = len(trace)
    splits = []  # (unit_end, next_unit_start)
    for m in _PARA_RE.finditer(trace):
        splits.append((m.start(), m.end()))
    for m in _SENT_RE.finditer(trace):
        splits.append((m.start() + 1, m.end()))
    for m in _BULLET_RE.finditer(trace):
        splits.append((m.start(), m.start() + 1))
    splits.sort()

    dedup = []
    last_next = 0
    for ue, ns in splits:
        if ue < last_next:
            continue
        dedup.append((ue, ns))
        last_next = ns

    units = []
    cursor = 0
    while cursor < n and trace[cursor].isspace():
        cursor += 1
    for ue, ns in dedup:
        if ue > cursor:
            units.append((cursor, ue))
        cursor = ns
        while cursor < n and trace[cursor].isspace():
            cursor += 1
    end = n
    while end > cursor and trace[end - 1].isspace():
        end -= 1
    if end > cursor:
        units.append((cursor, end))
    return units


def reconstruct(trace, units, cache_ids):
    """Insert ' [CACHE]' after each selected unit's content. Round-trip safe."""
    valid = sorted({i for i in cache_ids if 0 <= i < len(units)})
    parts = []
    cursor = 0
    for i in valid:
        end = units[i][1]
        parts.append(trace[cursor:end])
        parts.append(" [CACHE]")
        cursor = end
    parts.append(trace[cursor:])
    return "".join(parts)


def format_units(trace, units):
    """Render units as a numbered list for the teacher prompt."""
    return "\n".join(f"[{i}] {trace[s:e]}" for i, (s, e) in enumerate(units))
