"""Duplicate scoring and IMP conversion (Laws of Duplicate Bridge, Law 77)."""

from __future__ import annotations

from bisect import bisect_left

from .auction import Contract, Denomination
from .deal import Vulnerability


def contract_score(level: int, denomination: Denomination, doubled: int,
                   tricks: int, vulnerable: bool) -> int:
    """Duplicate score for the declaring side; negative when the contract fails.

    ``tricks`` is the number of tricks taken by declarer's side (0..13) and
    ``doubled`` is 0, 1 (X) or 2 (XX).
    """
    if not 1 <= level <= 7:
        raise ValueError(f"level must be 1..7, got {level}")
    if doubled not in (0, 1, 2):
        raise ValueError(f"doubled must be 0, 1 or 2, got {doubled}")
    if not 0 <= tricks <= 13:
        raise ValueError(f"tricks must be 0..13, got {tricks}")

    needed = level + 6
    if tricks < needed:
        return -_undertrick_penalty(needed - tricks, doubled, vulnerable)

    multiplier = (1, 2, 4)[doubled]
    if denomination is Denomination.NOTRUMP:
        trick_score = (40 + 30 * (level - 1)) * multiplier
        per_trick = 30
    elif denomination in (Denomination.HEARTS, Denomination.SPADES):
        trick_score = 30 * level * multiplier
        per_trick = 30
    else:
        trick_score = 20 * level * multiplier
        per_trick = 20

    score = trick_score
    if trick_score >= 100:
        score += 500 if vulnerable else 300
    else:
        score += 50
    if level == 6:
        score += 750 if vulnerable else 500
    elif level == 7:
        score += 1500 if vulnerable else 1000
    score += 50 * doubled  # "insult": 50 for making doubled, 100 redoubled

    over = tricks - needed
    if doubled == 0:
        score += over * per_trick
    else:
        score += over * (200 if vulnerable else 100) * doubled
    return score


def _undertrick_penalty(down: int, doubled: int, vulnerable: bool) -> int:
    if doubled == 0:
        return down * (100 if vulnerable else 50)
    if vulnerable:
        penalty = 200 + 300 * (down - 1)
    else:
        penalty = 100 + 200 * min(down - 1, 2) + 300 * max(down - 3, 0)
    return penalty * doubled


def score_ns(contract: Contract | None, tricks: int, vulnerability: Vulnerability) -> int:
    """Score of a board from North-South's point of view.

    ``contract`` is None for a passed-out board; ``tricks`` counts tricks
    taken by the declaring side.
    """
    if contract is None:
        return 0
    s = contract_score(contract.level, contract.denomination, contract.doubled, tricks,
                       vulnerability.is_vulnerable(contract.declarer))
    return s if contract.declarer % 2 == 0 else -s


# Upper bounds (inclusive) of the score difference for 0, 1, ..., 23 IMPs.
_IMP_BOUNDS = (10, 40, 80, 120, 160, 210, 260, 310, 360, 420, 490, 590, 740,
               890, 1090, 1290, 1490, 1740, 1990, 2240, 2490, 2990, 3490, 3990)


def imps(diff: int) -> int:
    """Convert a total-point difference into IMPs (signed, -24..24)."""
    i = bisect_left(_IMP_BOUNDS, abs(diff))
    return i if diff >= 0 else -i
