"""Seats, vulnerability and full deals (four hands)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, IntEnum

from .cards import FULL_DECK_MASK, Card, Hand


class Seat(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    @property
    def symbol(self) -> str:
        return "NESW"[self]

    @classmethod
    def from_symbol(cls, s: str) -> Seat:
        return cls("NESW".index(s.upper()))

    def next(self, steps: int = 1) -> Seat:
        return Seat((self + steps) % 4)

    @property
    def partner(self) -> Seat:
        return self.next(2)

    def same_side(self, other: Seat) -> bool:
        return self % 2 == other % 2


class Vulnerability(Enum):
    NONE = "None"
    NS = "NS"
    EW = "EW"
    BOTH = "All"

    def is_vulnerable(self, seat: Seat) -> bool:
        if self is Vulnerability.BOTH:
            return True
        if self is Vulnerability.NONE:
            return False
        return (seat % 2 == 0) == (self is Vulnerability.NS)

    @classmethod
    def for_board(cls, board: int) -> Vulnerability:
        """Standard duplicate vulnerability rotation for board numbers 1..16 (repeating)."""
        table = [cls.NONE, cls.NS, cls.EW, cls.BOTH,
                 cls.NS, cls.EW, cls.BOTH, cls.NONE,
                 cls.EW, cls.BOTH, cls.NONE, cls.NS,
                 cls.BOTH, cls.NONE, cls.NS, cls.EW]
        return table[(board - 1) % 16]


def dealer_for_board(board: int) -> Seat:
    return Seat((board - 1) % 4)


@dataclass(frozen=True)
class Deal:
    hands: tuple[Hand, Hand, Hand, Hand]  # indexed by Seat
    dealer: Seat = Seat.NORTH
    vulnerability: Vulnerability = Vulnerability.NONE

    def __post_init__(self) -> None:
        if len(self.hands) != 4:
            raise ValueError("a deal has four hands")
        seen = 0
        for h in self.hands:
            if len(h) != 13:
                raise ValueError(f"each hand must have 13 cards, got {len(h)}")
            if seen & h.mask:
                raise ValueError("hands overlap")
            seen |= h.mask
        assert seen == FULL_DECK_MASK

    def __getitem__(self, seat: Seat) -> Hand:
        return self.hands[seat]

    def owner(self, card: Card) -> Seat:
        for seat in Seat:
            if card in self.hands[seat]:
                return seat
        raise AssertionError("unreachable: every card has an owner")

    @classmethod
    def random(
        cls,
        rng: random.Random | None = None,
        dealer: Seat = Seat.NORTH,
        vulnerability: Vulnerability = Vulnerability.NONE,
    ) -> Deal:
        rng = rng or random.Random()
        deck = list(range(52))
        rng.shuffle(deck)
        hands = tuple(Hand.from_cards(Card(i) for i in deck[k * 13:(k + 1) * 13]) for k in range(4))
        return cls(hands, dealer, vulnerability)  # type: ignore[arg-type]

    @classmethod
    def parse_pbn(cls, text: str, dealer: Seat = Seat.NORTH,
                  vulnerability: Vulnerability = Vulnerability.NONE) -> Deal:
        """Parse a PBN [Deal] value like 'N:AKQ.JT9.876.5432 ...' (hands clockwise from the named seat)."""
        first, rest = text.strip().split(":", 1)
        start = Seat.from_symbol(first)
        parts = rest.split()
        if len(parts) != 4:
            raise ValueError("PBN deal needs four hands")
        hands: list[Hand] = [Hand()] * 4
        for i, part in enumerate(parts):
            hands[start.next(i)] = Hand.parse_pbn(part)
        return cls(tuple(hands), dealer, vulnerability)  # type: ignore[arg-type]

    def to_pbn(self, start: Seat = Seat.NORTH) -> str:
        return start.symbol + ":" + " ".join(self.hands[start.next(i)].to_pbn() for i in range(4))
