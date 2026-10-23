"""Hand constraints read from the auction with a few fixed rules.

A stand-in until the bidding model (M2) can supply its own inferences. The
rules cover a plain natural system (five-card majors, 15-17 notrump, weak
twos) and only the clearest calls: passes before the opening bid, the
opening bid itself and the first direct overcall. Everything else adds
nothing.
"""

from __future__ import annotations

from .auction import Auction, Bid, Denomination, Pass
from .deal import Seat
from .sampling import NO_CONSTRAINT, HandConstraint

ANY = (0, 13)
BALANCED = ((2, 5),) * 4


def _suit(denomination: Denomination, length: tuple[int, int], hcp: tuple[int, int]) -> HandConstraint:
    lengths = [ANY] * 4
    lengths[int(denomination)] = length
    return HandConstraint(hcp, tuple(lengths))


def _opening(bid: Bid) -> HandConstraint:
    d, level = bid.denomination, bid.level
    if d is Denomination.NOTRUMP:
        return {1: HandConstraint((15, 17), BALANCED), 2: HandConstraint((20, 21), BALANCED)}.get(level, NO_CONSTRAINT)
    if level == 1:
        major = d in (Denomination.HEARTS, Denomination.SPADES)
        return _suit(d, (5, 13) if major else (3, 13), (11, 21))
    if level == 2:
        if d is Denomination.CLUBS:
            return HandConstraint((22, 37))
        return _suit(d, (6, 6), (5, 11))
    if level in (3, 4):
        return _suit(d, (7, 13), (4, 10))
    return NO_CONSTRAINT


def _overcall(bid: Bid, opening: Bid) -> HandConstraint:
    d = bid.denomination
    if d is Denomination.NOTRUMP:
        return HandConstraint((15, 18), BALANCED) if bid.level == 1 else NO_CONSTRAINT
    cheapest = opening.level if d > opening.denomination else opening.level + 1
    if bid.level == cheapest and bid.level <= 3:
        return _suit(d, (5, 13), (8, 17))
    if bid.level == cheapest + 1:  # jump overcall: weak
        return _suit(d, (6, 13), (5, 10))
    return NO_CONSTRAINT


def constraints_from_auction(auction: Auction) -> dict[Seat, HandConstraint]:
    """Constraints on the original hands of the seats whose calls the rules read."""
    out: dict[Seat, HandConstraint] = {}

    def add(seat: Seat, c: HandConstraint) -> None:
        out[seat] = out.get(seat, NO_CONSTRAINT).intersect(c)

    calls = auction.calls
    for i, call in enumerate(calls):
        seat = auction.dealer.next(i)
        if call is Pass:
            add(seat, HandConstraint((0, 11)))
            continue
        if not isinstance(call, Bid):
            break
        add(seat, _opening(call))
        # The direct seat's overcall; a pass there can hide a strong hand, so it says nothing.
        if i + 1 < len(calls) and isinstance(calls[i + 1], Bid):
            add(seat.next(), _overcall(calls[i + 1], call))
        break
    return out
