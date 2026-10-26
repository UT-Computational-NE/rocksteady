"""
Grouping segments by state or by subtype, for the tables of ``metrics``,
``optimal`` and ``report``: which rows or columns a ``by`` makes, and the
checks on the subtypes the segments carry.
"""

from typing import Dict, List, Sequence, Tuple

from .states import Segment, State

ALL = "all"  # the row that pools every segment


BY = ("state", "subtype")


def check_by(by: str) -> None:
    if by not in BY:
        raise ValueError(f"by must be one of {BY}, got {by!r}")


def subtype_column(segs: Sequence[Segment], index):
    """
    Each segment's subtype as a column (None, not NaN, for none: pandas 3
    would infer a string column, with NaN for None).
    """
    import pandas as pd

    return pd.Series([s.subtype for s in segs], index=index, dtype=object)


def subtype_states(segs: Sequence[Segment], reserved: Sequence[str] = ()) -> Dict[str, str]:
    """
    The state of each subtype the segments carry, ``{subtype: state}``. A
    subtype named like a state or one of ``reserved``, or carried by
    segments of two states, is refused (``ValueError``).
    """
    names = {*reserved, *(st.value for st in State)}
    out: Dict[str, str] = {}
    for i, s in enumerate(segs):
        if s.subtype is None:
            continue
        if s.subtype in names:
            raise ValueError(f"segment {i} has subtype {s.subtype!r}, the name of a state or of a row "
                             f"({', '.join(sorted(names))}); give it another name")
        st = out.setdefault(s.subtype, State(s.state).value)
        if st != State(s.state).value:
            raise ValueError(f"subtype {s.subtype!r} is carried by segments of two states ({st!r} and "
                             f"{State(s.state).value!r}); a subtype refines one state")
    return out


def groups(segs: Sequence[Segment], rows: Sequence[str], by: str = "state") -> List[Tuple[str, List[int]]]:
    """
    The rows of a table by state or by subtype, each with the positions in
    ``segs`` of the segments it pools. ``rows`` are ``"all"`` or states.
    By ``"state"``, each row pools its state's segments (``"all"``, every
    segment). By ``"subtype"``, each state's row is followed by a row for
    each subtype its segments carry, sorted by name, which pools that
    subtype's segments; a state whose segments carry none has only its own
    row. A subtype named like a state or ``"all"``, or carried by segments
    of two states, is refused (``subtype_states``).
    """
    check_by(by)
    subtypes = subtype_states(segs, (ALL,)) if by == "subtype" else {}
    out = []
    for row in rows:
        out.append((str(row), [i for i, s in enumerate(segs) if row == ALL or s.state == State(row)]))
        for sub in sorted(k for k, st in subtypes.items() if st == row):
            out.append((sub, [i for i, s in enumerate(segs) if s.subtype == sub]))
    return out
