"""Cards and hands.

A card is encoded as an integer 0..51: ``suit * 13 + rank_index`` with
suits ordered C, D, H, S and ranks 2..A. A hand is a 52-bit mask, which keeps
hands cheap to copy and maps directly onto the input encodings used by
neural bidding and play models.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Iterator


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    @property
    def symbol(self) -> str:
        return "CDHS"[self]

    @classmethod
    def from_symbol(cls, s: str) -> Suit:
        return cls("CDHS".index(s.upper()))


class Rank(IntEnum):
    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12

    @property
    def symbol(self) -> str:
        return "23456789TJQKA"[self]

    @classmethod
    def from_symbol(cls, s: str) -> Rank:
        s = s.upper()
        if s == "10":
            s = "T"
        return cls("23456789TJQKA".index(s))


HCP = {Rank.ACE: 4, Rank.KING: 3, Rank.QUEEN: 2, Rank.JACK: 1}


@dataclass(frozen=True, order=True)
class Card:
    index: int

    def __post_init__(self) -> None:
        if not 0 <= self.index < 52:
            raise ValueError(f"card index out of range: {self.index}")

    @classmethod
    def of(cls, suit: Suit, rank: Rank) -> Card:
        return cls(int(suit) * 13 + int(rank))

    @classmethod
    def parse(cls, text: str) -> Card:
        """Parse 'SA', 'HT', 'D10', 'C2'."""
        text = text.strip()
        if len(text) < 2:
            raise ValueError(f"bad card: {text!r}")
        return cls.of(Suit.from_symbol(text[0]), Rank.from_symbol(text[1:]))

    @property
    def suit(self) -> Suit:
        return Suit(self.index // 13)

    @property
    def rank(self) -> Rank:
        return Rank(self.index % 13)

    def __str__(self) -> str:
        return self.suit.symbol + self.rank.symbol

    def __repr__(self) -> str:
        return f"Card({self})"


FULL_DECK_MASK = (1 << 52) - 1


@dataclass(frozen=True)
class Hand:
    """An immutable set of cards stored as a 52-bit mask."""

    mask: int = 0

    def __post_init__(self) -> None:
        if self.mask & ~FULL_DECK_MASK:
            raise ValueError("hand mask has bits outside the deck")

    @classmethod
    def from_cards(cls, cards: Iterable[Card]) -> Hand:
        mask = 0
        for c in cards:
            bit = 1 << c.index
            if mask & bit:
                raise ValueError(f"duplicate card {c}")
            mask |= bit
        return cls(mask)

    @classmethod
    def parse_pbn(cls, text: str) -> Hand:
        """Parse one PBN hand, 'S.H.D.C' e.g. 'AKQ.JT9.876.5432'."""
        parts = text.split(".")
        if len(parts) != 4:
            raise ValueError(f"PBN hand needs 4 suits: {text!r}")
        cards = []
        for suit, holding in zip((Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS), parts):
            for ch in holding.strip():
                cards.append(Card.of(suit, Rank.from_symbol(ch)))
        return cls.from_cards(cards)

    def to_pbn(self) -> str:
        return ".".join(
            "".join(c.rank.symbol for c in sorted(self.cards_in(s), reverse=True))
            for s in (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS)
        )

    def __contains__(self, card: Card) -> bool:
        return bool(self.mask >> card.index & 1)

    def __iter__(self) -> Iterator[Card]:
        m = self.mask
        while m:
            low = m & -m
            yield Card(low.bit_length() - 1)
            m ^= low

    def __len__(self) -> int:
        return self.mask.bit_count()

    def add(self, card: Card) -> Hand:
        if card in self:
            raise ValueError(f"{card} already in hand")
        return Hand(self.mask | 1 << card.index)

    def remove(self, card: Card) -> Hand:
        if card not in self:
            raise ValueError(f"{card} not in hand")
        return Hand(self.mask & ~(1 << card.index))

    def cards_in(self, suit: Suit) -> list[Card]:
        return [c for c in self if c.suit == suit]

    def suit_length(self, suit: Suit) -> int:
        return (self.mask >> (int(suit) * 13) & 0x1FFF).bit_count()

    def shape(self) -> tuple[int, int, int, int]:
        """Suit lengths in S, H, D, C order."""
        return tuple(self.suit_length(s) for s in (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS))  # type: ignore[return-value]

    def hcp(self) -> int:
        return sum(HCP.get(c.rank, 0) for c in self)

    def __str__(self) -> str:
        return self.to_pbn()
